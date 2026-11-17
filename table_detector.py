"""One-shot calibration proposals from a YOLO table segmentation model."""
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.capi import onnxruntime_pybind11_state as ort_errors

SIZE = 640
MIN_SCORE = 0.45
MODEL_PATH = Path(__file__).with_name('table_seg_640.onnx')
_session = None
_busy = Lock()


def jpeg_dimensions(data):
    """Read baseline/progressive JPEG dimensions before allocating pixels."""
    if data[:2] != b'\xff\xd8':
        return None
    i = 2
    while i + 4 <= len(data) and data[i] == 0xff:
        i += 1
        if data[i] == 0xff:
            continue
        marker = data[i]
        i += 1
        if marker in (0xda, 0xd9):
            break
        length = int.from_bytes(data[i:i+2], 'big')
        if length < 2 or i + length > len(data):
            break
        if marker in (0xc0, 0xc1, 0xc2) and length >= 8:
            height = int.from_bytes(data[i+3:i+5], 'big')
            width = int.from_bytes(data[i+5:i+7], 'big')
            return (width, height) if width and height else None
        i += length
    return None


def prepare(image):
    h, w = image.shape[:2]
    scale = SIZE / max(h, w)
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    x, y = (SIZE - rw) // 2, (SIZE - rh) // 2
    canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    canvas[y:y + rh, x:x + rw] = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, (x, y, rw, rh)


def corners_from_mask(mask):
    h, w = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < h * w * 0.025:
        return None
    # A clipped tabletop cannot provide four actual corners.
    bx, by, bw, bh = cv2.boundingRect(contour)
    if bx < 2 or by < 2 or bx + bw > w - 2 or by + bh > h - 2:
        return None
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    best = None
    original = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(original, [contour], -1, 1, cv2.FILLED)
    for fraction in np.linspace(.008, .055, 16):
        approx = cv2.approxPolyDP(hull, float(fraction * perimeter), True).reshape(-1, 2)
        if len(approx) != 4:
            continue
        filled = np.zeros_like(original)
        cv2.fillConvexPoly(filled, approx, 1)
        iou = np.count_nonzero(original & filled) / np.count_nonzero(original | filled)
        if best is None or iou > best[0]:
            best = iou, approx.copy()
    if best is None or best[0] < .83:
        return None
    points = best[1]
    # Existing calibration expects counterclockwise image order, with the lower
    # (near) edge first. Leave it as an editable proposal for angled viewpoints.
    if cv2.contourArea(points.reshape(-1, 1, 2), oriented=True) > 0:
        points = points[::-1]
    i = max(range(4), key=lambda j: int(points[j, 1]) + int(points[(j + 1) % 4, 1]))
    points = np.roll(points, -i, axis=0)
    if points[1, 0] <= points[0, 0]:
        return None
    return (points / np.array([w, h])).tolist()


def propose(outputs, content):
    predictions = next(o for o in outputs if o.ndim == 3)[0].T
    proto = next(o for o in outputs if o.ndim == 4)[0]
    channels, mh, mw = proto.shape
    classes = predictions.shape[1] - channels - 4
    if classes != 1:
        raise ValueError('Expected a single-class table segmentation export')
    scores = predictions[:, 4]
    chosen = np.flatnonzero(np.isfinite(scores) & (scores >= MIN_SCORE))
    if not chosen.size:
        return None
    chosen = chosen[np.argsort(scores[chosen])[::-1][:40]]
    rows = predictions[chosen]
    boxes = np.column_stack((rows[:, :2] - rows[:, 2:4] / 2, rows[:, 2:4]))
    kept = cv2.dnn.NMSBoxes(boxes.tolist(), scores[chosen].tolist(), MIN_SCORE, .5)
    x, y, rw, rh = content
    for index in np.asarray(kept, dtype=np.intp).reshape(-1)[:3]:
        row, box = rows[index], boxes[index]
        logits = row[4 + classes:] @ proto.reshape(channels, -1)
        logits = logits.reshape(mh, mw)
        # Crop the upsampled mask to the predicted box and remove letterbox padding.
        mask = cv2.resize(logits, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR) > 0
        x0, y0 = np.maximum(0, np.floor(box[:2])).astype(int)
        x1, y1 = np.minimum(SIZE, np.ceil(box[:2] + box[2:])).astype(int)
        if x1 <= x0 or y1 <= y0:
            continue
        mask[:y0, :] = False
        mask[y1:, :] = False
        mask[:, :x0] = False
        mask[:, x1:] = False
        corners = corners_from_mask(mask[y:y+rh, x:x+rw])
        if corners is not None:
            return {'corners': corners, 'confidence': round(float(row[4]), 3)}
    return None


def detect_table(image):
    global _session
    if not _busy.acquire(blocking=False):
        raise BlockingIOError('Table detector is busy')
    try:
        if _session is None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            _session = ort.InferenceSession(str(MODEL_PATH), sess_options=options, providers=['CPUExecutionProvider'])
        tensor, content = prepare(image)
        return propose(_session.run(None, {_session.get_inputs()[0].name: tensor}), content)
    except (ort_errors.Fail, ort_errors.InvalidArgument, ort_errors.NoSuchFile, ort_errors.RuntimeException) as exc:
        raise RuntimeError('Table model could not run') from exc
    finally:
        _busy.release()
