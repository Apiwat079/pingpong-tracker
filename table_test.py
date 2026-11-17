"""Local validation for the real ONNX export, corner geometry and HTTP route."""
import json
import os
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

from table_detector import corners_from_mask, detect_table

root = Path(__file__).resolve().parent
os.environ['MODEL'] = str(root / 'best_960_fp32.onnx')

mask = np.zeros((400, 640), dtype=np.uint8)
points = np.array([[90, 330], [540, 350], [455, 105], [180, 95]], dtype=np.int32)
cv2.fillConvexPoly(mask, points, 1)
result = np.array(corners_from_mask(mask)) * [640, 400]
assert np.max(np.abs(result - points)) < 4, result
assert corners_from_mask(np.zeros_like(mask)) is None
clipped = np.zeros_like(mask)
cv2.fillConvexPoly(clipped, np.array([[0, 330], [540, 350], [455, 105], [180, 95]], np.int32), 1)
assert corners_from_mask(clipped) is None

with tempfile.TemporaryDirectory() as folder:
    os.environ['CAPTURE_DIR'] = folder
    from main import app
    client = app.test_client()
    assert client.get('/').status_code == 200
    assert client.post('/table/detect', data=b'bad').status_code == 415
    assert client.post('/table/detect', data=b'bad', content_type='image/jpeg').status_code == 400
    assert client.post('/table/detect', data=b'0' * (512 * 1024 + 1), content_type='image/jpeg').status_code == 413

    def check(frame):
        h, w = frame.shape[:2]
        scale = min(1, 800 / max(w, h))
        frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
        ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
        assert ok
        response = client.post('/table/detect', data=data.tobytes(), content_type='image/jpeg')
        assert response.status_code == 200, response.json
        return response.json['proposal']

    assert check(np.zeros((450, 800, 3), dtype=np.uint8)) is None
    report = []
    if len(sys.argv) > 1:
        for image in sorted(Path(sys.argv[1]).rglob('*_manual.jpg')):
            frame = cv2.imread(str(image))
            if frame is None:
                continue
            report.append({'file': image.name, 'proposal': check(frame)})
    assert client.get('/captures/list').json['total'] == 0
    print(json.dumps({'passed': True, 'real_frames': report}, indent=2))
