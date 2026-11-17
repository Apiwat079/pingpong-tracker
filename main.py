import ast
import logging
import os
import re
import socket
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

# static_folder=None: /static/ort/ is served by an explicit route below (exact MIME types)
app = Flask(__name__, static_folder=None)
logging.getLogger("werkzeug").setLevel(logging.WARNING)
cv2.setNumThreads(2)  # 4-core CPU: leave cores for HTTP/JPEG and the DML driver

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---- server-side concurrency limit (FIFO queue for the GPU) ----
# Everything reaches us from 127.0.0.1 (Tailscale Funnel), so identity is a client-generated session id.
def _env_num(name, default, cast=float):
    try:
        return cast(os.environ.get(name, default))
    except (TypeError, ValueError):
        return cast(default)


MAX_SERVER_USERS = max(1, _env_num("MAX_SERVER_USERS", 2, int))
QUEUE_IDLE_TTL = _env_num("QUEUE_IDLE_TTL", 8.0)    # active session with no heartbeat/detect for this long loses its slot
QUEUE_GRANT_TTL = _env_num("QUEUE_GRANT_TTL", 45.0)  # granted but never sent /detect for this long: slot is taken back
QUEUE_WAIT_TTL = _env_num("QUEUE_WAIT_TTL", 10.0)    # waiting session that stopped polling is dropped
SID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
ANON_SID = "anon"  # shared pseudo-session for legacy clients that send no session id (cannot match SID_RE)


class SlotQueue:
    """Thread-safe in-memory slot limiter: at most `max_active` active sessions, the rest wait in strict FIFO.
    Expiry is lazy (done at the start of every call); `clock` is injectable for tests."""

    def __init__(self, max_active, idle_ttl, grant_ttl, wait_ttl, clock=time.monotonic, log=None):
        self.max = int(max_active)
        self.idle_ttl, self.grant_ttl, self.wait_ttl = idle_ttl, grant_ttl, wait_ttl
        self._clock = clock
        self._log = log
        self._lock = threading.Lock()
        self._active = {}              # sid -> {"seen": t, "granted": t, "streaming": bool}
        self._wait = OrderedDict()     # sid -> last poll time (insertion order == FIFO order)

    def _say(self, event, sid):
        if self._log:
            self._log("[queue] %s %s active=%d/%d waiting=%d" % (event, str(sid)[:8], len(self._active), self.max, len(self._wait)))

    def _grant(self, sid, now, event):
        self._active[sid] = {"seen": now, "granted": now, "streaming": False}
        self._say(event, sid)

    def _prune(self, now):
        for sid in [s for s, a in self._active.items()
                    if now - a["seen"] > self.idle_ttl or (not a["streaming"] and now - a["granted"] > self.grant_ttl)]:
            del self._active[sid]
            self._say("expire", sid)
        for sid in [s for s, t in self._wait.items() if now - t > self.wait_ttl]:
            del self._wait[sid]
            self._say("drop", sid)
        self._promote(now)

    def _promote(self, now):
        while self._wait and len(self._active) < self.max:
            sid, _ = self._wait.popitem(last=False)
            self._grant(sid, now, "promote")

    def _info(self, sid):
        if sid in self._active:
            state, pos = "active", 0
        elif sid in self._wait:
            state, pos = "waiting", list(self._wait).index(sid) + 1
        else:
            state, pos = "none", 0
        return {"state": state, "position": pos, "active": len(self._active), "max": self.max, "waiting": len(self._wait)}

    def join(self, sid):
        with self._lock:
            now = self._clock()
            self._prune(now)
            if sid in self._active:
                self._active[sid]["seen"] = now
            elif sid in self._wait:
                self._wait[sid] = now
            elif len(self._active) < self.max and not self._wait:
                self._grant(sid, now, "admit")
            else:
                self._wait[sid] = now
                self._say("wait", sid)
            return self._info(sid)

    def status(self, sid):
        """Also the heartbeat for active and waiting sessions. Unknown sid -> state 'none'."""
        with self._lock:
            now = self._clock()
            self._prune(now)
            if sid in self._active:
                self._active[sid]["seen"] = now
            elif sid in self._wait:
                self._wait[sid] = now
            return self._info(sid)

    def leave(self, sid):
        with self._lock:
            now = self._clock()
            self._prune(now)
            if self._active.pop(sid, None) is not None:
                self._say("leave", sid)
            elif self._wait.pop(sid, None) is not None:
                self._say("leave", sid)
            self._promote(now)

    def use(self, sid, anon=False):
        """Guard for /detect: (True, info) if `sid` holds a slot (heartbeat + streaming mark), else (False, info).
        Unknown sids are never auto-admitted (rotating ids must not bypass the limit); only the shared
        'anon' pseudo-session of legacy clients is admitted, and only when a slot is free."""
        with self._lock:
            now = self._clock()
            self._prune(now)
            if anon and sid not in self._active and len(self._active) < self.max and not self._wait:
                self._grant(sid, now, "admit")
            a = self._active.get(sid)
            if a is not None:
                a["seen"] = now
                a["streaming"] = True
                return True, self._info(sid)
            return False, self._info(sid)

    def snapshot(self):
        with self._lock:
            self._prune(self._clock())
            return {"active": len(self._active), "max": self.max, "waiting": len(self._wait)}


slots = SlotQueue(MAX_SERVER_USERS, QUEUE_IDLE_TTL, QUEUE_GRANT_TTL, QUEUE_WAIT_TTL,
                  log=lambda m: print(m, flush=True))


def pick_model():
    env = os.environ.get("MODEL")
    if env:
        return env if os.path.isabs(env) else os.path.join(BASE_DIR, env)
    # Prefer the 960 export to preserve small-ball detail at inference.
    # Keep legacy exports as a server fallback when the 960 bundle is absent.
    for name in ("best_960_fp16.onnx", "best_960_fp32.onnx", "best_fp16.onnx", "best.onnx"):
        p = os.path.join(BASE_DIR, name)
        if os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, "best.onnx")


MODEL_PATH = pick_model()
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

_so = ort.SessionOptions()
_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
_so.enable_mem_pattern = False  # recommended for DirectML
_so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # recommended for DirectML
session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=_so,
    providers=[p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()],
)
input_name = session.get_inputs()[0].name
_shape = session.get_inputs()[0].shape
INPUT_H = _shape[2] if isinstance(_shape[2], int) else 640
INPUT_W = _shape[3] if isinstance(_shape[3], int) else 640
PROVIDER = session.get_providers()[0]
run_lock = threading.Lock()


def load_names():
    raw = session.get_modelmeta().custom_metadata_map.get("names")
    if raw:
        try:
            d = ast.literal_eval(raw)
            if isinstance(d, dict):
                return {int(k): str(v) for k, v in d.items()}
            if isinstance(d, (list, tuple)):
                return {i: str(v) for i, v in enumerate(d)}
        except Exception:
            pass
    return {}


NAMES = load_names()


def label_of(cid):
    return NAMES.get(cid, f"class_{cid}")


def names_list():
    n = max(NAMES) + 1 if NAMES else 0
    return [label_of(i) for i in range(n)]


def run_model(frame, conf_thr, iou_thr):
    """Returns (list of detection dicts normalized 0..1, inference_ms)."""
    t0 = time.perf_counter()
    h, w = frame.shape[:2]
    scale = min(INPUT_W / w, INPUT_H / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    top, left = (INPUT_H - nh) // 2, (INPUT_W - nw) // 2
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    boxed = cv2.copyMakeBorder(resized, top, INPUT_H - nh - top, left,
                               INPUT_W - nw - left, cv2.BORDER_CONSTANT,
                               value=(114, 114, 114))
    tensor = cv2.dnn.blobFromImage(boxed, 1 / 255.0, (INPUT_W, INPUT_H),
                                   swapRB=True, crop=False)

    with run_lock:
        output = session.run(None, {input_name: tensor})[0]

    preds = output[0]  # (4+nc, N)
    if preds.shape[0] > preds.shape[1]:  # tolerate transposed (N, 4+nc)
        preds = preds.T
    cls_scores = preds[4:]
    if cls_scores.shape[0] == 1:
        cids = np.zeros(cls_scores.shape[1], dtype=np.intp)
        scores = cls_scores[0]
    else:
        cids = np.argmax(cls_scores, axis=0)
        scores = np.take_along_axis(cls_scores, cids[None, :], axis=0)[0]
    keep = np.flatnonzero(scores >= conf_thr)

    dets = []
    if keep.size:
        bx = preds[:4, keep]
        scores, cids = scores[keep].astype(np.float32), cids[keep]
        cx, cy, bw, bh = bx[0], bx[1], bx[2], bx[3]
        x1 = np.clip((cx - bw / 2 - left) / scale, 0, w)
        y1 = np.clip((cy - bh / 2 - top) / scale, 0, h)
        x2 = np.clip((cx + bw / 2 - left) / scale, 0, w)
        y2 = np.clip((cy + bh / 2 - top) / scale, 0, h)
        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thr, iou_thr)
        idx = np.array(idx, dtype=np.intp).reshape(-1)
        for i in idx.tolist():
            cid = int(cids[i])
            dets.append({
                "x1": float(x1[i] / w), "y1": float(y1[i] / h),
                "x2": float(x2[i] / w), "y2": float(y2[i] / h),
                "conf": float(scores[i]), "cls": cid,
                "label": label_of(cid),
            })
    return dets, (time.perf_counter() - t0) * 1000


def read_image():
    if "image" in request.files:
        data = request.files["image"].read()
    else:
        data = request.get_data()
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def float_arg(name, default):
    try:
        return float(request.args.get(name, default))
    except (TypeError, ValueError):
        return default


def lan_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    return sorted(i for i in ips if not i.startswith("127."))


@app.get("/")
def home():
    return PAGE


@app.after_request
def add_isolation_headers(resp):
    # cross-origin isolation -> SharedArrayBuffer -> multithreaded WASM (everything is same-origin)
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    resp.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return resp


ORT_DIR = os.path.join(BASE_DIR, "static", "ort")
_ORT_MIME = {".wasm": "application/wasm", ".mjs": "text/javascript", ".js": "text/javascript"}


@app.get("/static/ort/<path:name>")
def ort_static(name):
    """Vendored onnxruntime-web (WebGPU bundle) for on-device inference."""
    mime = _ORT_MIME.get(os.path.splitext(name)[1].lower())
    if mime is None:
        abort(404)
    return send_from_directory(ORT_DIR, name, mimetype=mime, conditional=True, max_age=86400)


@app.get("/favicon.ico")
def favicon():
    return Response(status=204)


@app.get("/model.onnx")
def model_file():
    """The same model file the server runs, for on-device inference (ETag + cache).
    ?prec=fp32 serves best_960_fp32.onnx (for WebGPU adapters without shader-f16).
    ?variant=dyn[&prec=fp32] serves the dynamic-shape best_dyn_fp16/fp32.onnx."""
    path = MODEL_PATH
    fp32 = request.args.get("prec", "").lower() == "fp32"
    if request.args.get("variant", "").lower() == "dyn":
        # dynamic-shape model (any H, W multiple of 32): rectangular full frame + 320 ROI
        path = os.path.join(BASE_DIR, "best_dyn_fp32.onnx" if fp32 else "best_dyn_fp16.onnx")
        if not os.path.exists(path):
            abort(404)
    elif fp32:
        path = os.path.join(BASE_DIR, "best_960_fp32.onnx")
        if not os.path.exists(path):
            abort(404)
    return send_file(path, mimetype="application/octet-stream",
                     conditional=True, etag=True, max_age=0)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "provider": PROVIDER,
        "model": MODEL_PATH,
        "input_size": [INPUT_W, INPUT_H],
        "names": names_list(),
        **slots.snapshot(),
    })


@app.post("/table/detect")
def table_detect():
    # Only called when the user asks for a proposal in the calibration editor.
    if request.mimetype != "image/jpeg":
        return jsonify(error="expected_jpeg"), 415
    data = request.stream.read(512 * 1024 + 1)
    if len(data) > 512 * 1024:
        return jsonify(error="image_too_large"), 413
    from table_detector import jpeg_dimensions, detect_table
    size = jpeg_dimensions(data)
    if size is None:
        return jsonify(error="invalid_image"), 400
    if max(size) > 1024:
        return jsonify(error="image_too_large"), 413
    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None or (image.shape[1], image.shape[0]) != size:
            return jsonify(error="invalid_image"), 400
        return jsonify(proposal=detect_table(image))
    except BlockingIOError:
        return jsonify(error="busy"), 429
    except (OSError, RuntimeError, ValueError, cv2.error):
        app.logger.exception("Automatic table calibration failed")
        return jsonify(error="unavailable"), 503


def _drain_body():
    """Discard an unread (small) request body so the client can read our early 429 instead of a connection reset."""
    try:
        n = request.content_length or 0
        if 0 < n <= 4 * 1024 * 1024:
            request.stream.read(n)
    except Exception:
        pass


def _sid_from_request(body_ok=False):
    """Returns (sid_or_None, error_response_or_None). Header first, then ?sid= (sendBeacon), then a tiny text body."""
    sid = request.headers.get("X-Session-Id") or request.args.get("sid")
    if not sid and body_ok and (request.content_length or 0) <= 200:
        sid = request.get_data(as_text=True).strip().replace("sid=", "", 1) or None
    if sid is None:
        return None, None
    if not SID_RE.match(sid):
        return None, (jsonify({"error": "bad_sid"}), 400)
    return sid, None


def guard_detect():
    """Run BEFORE decoding the image / touching the model. Returns None if allowed, else a ready response.
    No X-Session-Id (legacy clients, e.g. the native iPad app) -> shared pseudo-session 'anon', which counts
    as one user and is auto-admitted only when a slot is free."""
    sid, err = _sid_from_request()
    if err:
        _drain_body()
        return err
    ok, info = slots.use(sid or ANON_SID, anon=sid is None)
    if ok:
        return None
    _drain_body()
    info = dict(info)
    info["error"] = "queue_full" if info["state"] == "waiting" or sid is None else "not_admitted"
    resp = jsonify(info)
    resp.status_code = 429
    resp.headers["Retry-After"] = "2"
    return resp


@app.post("/queue/join")
def queue_join():
    sid, err = _sid_from_request()
    if err:
        return err
    if sid is None:
        return jsonify({"error": "sid_required"}), 400
    return jsonify(slots.join(sid))


@app.get("/queue/status")
def queue_status():
    sid, err = _sid_from_request()
    if err:
        return err
    if sid is None:
        return jsonify({"error": "sid_required"}), 400
    return jsonify(slots.status(sid))


@app.post("/queue/leave")
def queue_leave():
    sid, err = _sid_from_request(body_ok=True)
    if err:
        return err
    if sid is not None:
        slots.leave(sid)
    return Response(status=204)


@app.post("/detect")
def detect():
    blocked = guard_detect()
    if blocked is not None:
        return blocked
    frame = read_image()
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400
    h, w = frame.shape[:2]
    dets, ms = run_model(
        frame, float_arg("conf", DEFAULT_CONF), float_arg("iou", DEFAULT_IOU)
    )
    return jsonify({
        "detections": dets,
        "inference_ms": ms,
        "provider": PROVIDER,
        "image_size": [w, h],
    })


@app.post("/detect_image")
def detect_image():
    """Annotated JPEG (used by the built-in web page)."""
    blocked = guard_detect()
    if blocked is not None:
        return blocked
    frame = read_image()
    if frame is None:
        return "Invalid image", 400
    h, w = frame.shape[:2]
    dets, ms = run_model(
        frame, float_arg("conf", DEFAULT_CONF), float_arg("iou", DEFAULT_IOU)
    )
    for d in dets:
        p1 = (int(d["x1"] * w), int(d["y1"] * h))
        p2 = (int(d["x2"] * w), int(d["y2"] * h))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cv2.putText(frame, f"{d['label']} {d['conf']:.2f}",
                    (p1[0], max(20, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"{PROVIDER} {ms:.0f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return "Image encoding failed", 500
    return Response(encoded.tobytes(), mimetype="image/jpeg")


CAPTURE_DIR = os.environ.get("CAPTURE_DIR") or os.path.join(BASE_DIR, "captures")
MAX_UPLOAD = 2 * 1024 * 1024
_save_lock = threading.Lock()


def _today_dir():
    return os.path.join(CAPTURE_DIR, time.strftime("%Y%m%d"))


@app.post("/save_frame")
def save_frame():
    """Store a full-res JPEG (+ optional YOLO pre-label) for later retraining."""
    if request.content_length is not None and request.content_length > MAX_UPLOAD:
        return jsonify({"error": "too large (max 2 MB)"}), 413
    data = request.stream.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        return jsonify({"error": "too large (max 2 MB)"}), 413
    if len(data) < 4 or data[:3] != b"\xff\xd8\xff":
        return jsonify({"error": "body must be a JPEG"}), 400
    reason = re.sub(r"[^a-z_]", "", request.args.get("reason", "").lower())[:24] or "unknown"
    label = None
    try:
        x1, y1, x2, y2 = (min(1.0, max(0.0, float(request.args[k]))) for k in ("x1", "y1", "x2", "y2"))
        if x2 > x1 and y2 > y1:
            label = "0 %.6f %.6f %.6f %.6f\n" % ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    except (KeyError, ValueError):
        pass
    conf = request.args.get("conf", "")
    day_dir = _today_dir()
    with _save_lock:
        os.makedirs(day_dir, exist_ok=True)
        stem = time.strftime("%Y%m%d_%H%M%S") + "_%03d_%s" % (int(time.time() * 1000) % 1000, reason)
        base, n = stem, 1
        while os.path.exists(os.path.join(day_dir, stem + ".jpg")):
            n += 1
            stem = "%s%d" % (base, n)
        with open(os.path.join(day_dir, stem + ".jpg"), "wb") as f:
            f.write(data)
        if label:
            with open(os.path.join(day_dir, stem + ".txt"), "w") as f:
                f.write(label)
        if conf:
            with open(os.path.join(day_dir, "index.csv"), "a") as f:
                f.write("%s.jpg,%s,%s,%d\n" % (stem, reason, conf[:10], 1 if label else 0))
        count = len([n for n in os.listdir(day_dir) if n.endswith(".jpg")])
    return jsonify({"saved": stem + ".jpg", "prelabel": bool(label), "count": count, "gallery": "/gallery"})


@app.get("/captures/stats")
def captures_stats():
    d = _today_dir()
    count = len([n for n in os.listdir(d) if n.endswith(".jpg")]) if os.path.isdir(d) else 0
    return jsonify({"date": time.strftime("%Y%m%d"), "count": count})


def _capture_file(day, name):
    if not re.fullmatch(r"\d{8}", day) or not re.fullmatch(r"[A-Za-z0-9_\-]+\.jpg", name):
        abort(404)
    root = os.path.realpath(CAPTURE_DIR)
    path = os.path.realpath(os.path.join(root, day, name))
    if os.path.commonpath([root, path]) != root or not os.path.isfile(path):
        abort(404)
    return path


@app.get("/gallery")
def gallery_page():
    return Response(GALLERY_PAGE, mimetype="text/html", headers={"Cache-Control": "no-store"})


@app.get("/captures/list")
def capture_list():
    try:
        limit = max(1, min(60, int(request.args.get("limit", 36))))
    except ValueError:
        return jsonify({"error": "invalid limit"}), 400
    before = request.args.get("before", "")
    records = []
    if os.path.isdir(CAPTURE_DIR):
        with os.scandir(CAPTURE_DIR) as days:
            for day in days:
                if not re.fullmatch(r"\d{8}", day.name) or not day.is_dir(follow_symlinks=False):
                    continue
                try:
                    with os.scandir(day.path) as files:
                        for file in files:
                            if not file.is_file(follow_symlinks=False) or not re.fullmatch(r"[A-Za-z0-9_\-]+\.jpg", file.name):
                                continue
                            # The public wall shows frames people explicitly save, not automatic training samples.
                            if not re.search(r"_manual\d*\.jpg$", file.name):
                                continue
                            records.append((day.name + "/" + file.name, file.stat().st_size))
                except FileNotFoundError:
                    continue
    records.sort(reverse=True)
    available = [row for row in records if not before or row[0] < before]
    selected = available[:limit]
    items = [{"key": key, "name": key.split("/")[1], "day": key.split("/")[0], "bytes": size,
              "url": "/captures/image/" + key, "thumb": "/captures/image/" + key + "?thumb=1"}
             for key, size in selected]
    return jsonify({"items": items, "total": len(records),
                    "next": selected[-1][0] if len(available) > limit else None}), 200, {"Cache-Control": "no-store"}


@app.get("/captures/image/<day>/<name>")
def capture_image(day, name):
    path = _capture_file(day, name)
    if request.args.get("thumb") != "1":
        return send_file(path, mimetype="image/jpeg", conditional=True)
    stat = os.stat(path)
    etag = "thumb-%x-%x" % (stat.st_mtime_ns, stat.st_size)
    if request.if_none_match.contains(etag):
        return Response(status=304, headers={"ETag": '"' + etag + '"', "Cache-Control": "public, max-age=3600"})
    frame = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    if frame is None:
        abort(404)
    h, w = frame.shape[:2]
    scale = min(1, 560 / max(w, h))
    if scale < 1:
        frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        abort(404)
    return Response(encoded.tobytes(), mimetype="image/jpeg", headers={"ETag": '"' + etag + '"', "Cache-Control": "public, max-age=3600"})


GALLERY_PAGE = '<!doctype html>\n<html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><meta name="theme-color" content="#101723"><title>Frame Gallery · Ping-Pong Tracker</title>\n<style>\n*{box-sizing:border-box}body{margin:0;background:#101723;color:#f7f4ee;font:15px -apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif}button,a{-webkit-tap-highlight-color:transparent}button:focus-visible,a:focus-visible{outline:3px solid #ffb46e;outline-offset:4px}button{cursor:pointer;font:inherit}a{color:inherit;text-decoration:none}[hidden]{display:none!important}\n.shell{width:min(1440px,100%);margin:auto;padding:0 clamp(20px,5vw,72px)}.nav{height:88px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #ffffff16;gap:20px}.brand{display:flex;align-items:center;gap:11px;font-weight:800;letter-spacing:-.4px}.ball{width:24px;height:24px;background:#f6934b;border-radius:50%;box-shadow:inset -5px -5px 0 #d96a2960}.nav-right{display:flex;gap:8px;align-items:center}.soft,.primary{min-height:46px;border:1px solid #ffffff25;border-radius:13px;background:#ffffff08;color:inherit;padding:11px 17px;display:inline-flex;justify-content:center;align-items:center;gap:8px}.primary{background:#f6934b;color:#17202c;border-color:#f6934b;font-weight:750}.soft:hover{background:#ffffff15}.soft:disabled{opacity:.45;cursor:default}\n.hero{display:flex;align-items:flex-end;justify-content:space-between;gap:30px;padding:52px 0 36px}.eyebrow{font-size:11px;letter-spacing:2.5px;color:#f7aa72;font-weight:800}.hero h1{font-size:clamp(34px,5vw,64px);letter-spacing:-1.6px;line-height:1.15;margin:14px 0}.hero p{color:#aab6c8;line-height:1.8;margin:0;max-width:600px}.count{font-size:clamp(36px,5vw,64px);font-weight:750;letter-spacing:-2px;line-height:1}.count-label{color:#aab6c8;font-size:13px;margin-top:10px}.feedbar{display:flex;align-items:center;justify-content:space-between;margin:0 0 20px;color:#aab6c8;font-size:13px;gap:20px}.live{display:flex;align-items:center;gap:8px}.dot{width:7px;height:7px;background:#8bdfb7;border-radius:50%}\n.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}.card{padding:0;text-align:left;overflow:hidden;border:1px solid #ffffff12;border-radius:20px;background:#192333;color:inherit;transition:transform .18s,border-color .18s}.card:hover{transform:translateY(-4px);border-color:#f6934b88}.card img{width:100%;aspect-ratio:4/3;object-fit:cover;display:block;background:#253044}.card-footer{display:flex;justify-content:space-between;gap:12px;padding:15px 17px;font-size:13px}.card-footer span:last-child{color:#92a0b4;font-size:12px}.card .arrow{color:#f6934b!important}.empty{text-align:center;padding:90px 20px;border:1px dashed #ffffff30;border-radius:22px;color:#aab6c8;line-height:1.9}.empty strong{display:block;font-size:22px;color:#f7f4ee}.empty a{color:#f7aa72}.footer{padding:32px 0 40px;text-align:center;color:#8a98ad;font-size:12px}.more{text-align:center;margin:30px 0}.note{color:#f9b185;min-height:24px}\ndialog{background:#0b111c;color:#fff;border:1px solid #ffffff20;border-radius:24px;padding:16px;width:min(1180px,94vw);max-height:94dvh;overflow:auto}dialog::backdrop{background:#04070eea}.light-head{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:0 0 12px}.light-head span{color:#aab6c8;font-size:13px}.light-img{display:block;width:100%;height:min(65dvh,900px);object-fit:contain;background:#050911;border-radius:12px}.light-foot{display:flex;align-items:center;justify-content:space-between;gap:10px;padding-top:14px;flex-wrap:wrap}.light-actions{display:flex;gap:8px}.presentation .nav,.presentation .hero,.presentation .feedbar,.presentation .footer{display:none}.presentation .grid{padding:20px 0}.presentation .shell{padding:0 20px}.presentation dialog{width:99vw;max-height:99dvh}.presentation .light-img{height:79dvh}\n@media(min-width:1500px){.grid{grid-template-columns:repeat(4,minmax(0,1fr))}}@media(max-width:700px){.shell{padding:0 16px}.nav{height:76px}.brand{font-size:13px}.nav-right .soft{padding:10px;font-size:12px}.hero{padding:32px 0;align-items:flex-start}.hero h1{letter-spacing:-1px}.hero p{font-size:13px}.hero-counter{padding-top:25px}.grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.card{border-radius:14px}.card-footer{padding:11px;font-size:11px;flex-direction:column;gap:5px}.light-img{height:57dvh}.soft,.primary{min-height:44px}.feedbar{align-items:flex-start}.feedbar>span{max-width:190px;text-align:right;font-size:11px}.nav-right{gap:5px}}@media(prefers-reduced-motion:reduce){.card{transition:none}}\n</style></head><body><main class="shell">\n<nav class="nav"><a class="brand" href="/"><i class="ball" aria-hidden="true"></i>PING-PONG TRACKER</a><div class="nav-right"><a class="soft" href="/">← กลับไปกล้อง</a><button class="soft" id="presentBtn" disabled onclick="togglePresentation()">เต็มจอ ↗</button></div></nav>\n<section class="hero"><div><div class="eyebrow">THE FRAME COLLECTION</div><h1>ทุกเฟรม มีเรื่องเล่า</h1><p>ภาพจากกล้องที่ทุกคนกดบันทึกไว้ รวมอยู่ที่นี่<br>แตะภาพเพื่อขยาย หรือเปิดสไลด์พรีเซนต์ได้ทันที</p></div><div class="hero-counter"><div class="count" id="total">—</div><div class="count-label">เฟรมที่บันทึก</div></div></section>\n<div class="feedbar"><div class="live"><i class="dot"></i><span id="feedStatus" role="status">กำลังโหลดภาพ…</span></div><button class="primary" id="playBtn" onclick="startSlides()" disabled>▶ เล่นสไลด์</button></div>\n<p class="note" id="error" role="status"></p><div id="empty" class="empty" hidden><strong>เฟรมแรกกำลังรอคุณอยู่</strong>เปิดกล้องแล้วแตะ “บันทึกภาพ”<br><a href="/">กลับไปเก็บเฟรมแรก →</a></div><section class="grid" id="grid" aria-label="ภาพที่บันทึก"></section><div class="more"><button class="soft" id="moreBtn" onclick="loadMore()" hidden>ดูภาพเพิ่มเติม ↓</button></div><footer class="footer">PING-PONG TRACKER · ภาพใหม่จะปรากฏอัตโนมัติขณะเปิดหน้านี้</footer></main>\n<dialog id="lightbox" aria-label="ภาพขยาย"><div class="light-head"><span id="lightTitle"></span><button class="soft" onclick="closeLightbox()" aria-label="ปิดภาพขยาย">ปิด ✕</button></div><img class="light-img" id="lightImage" alt="ภาพที่บันทึกจากกล้อง"><div class="light-foot"><div class="light-actions"><button class="soft" onclick="advance(-1)" aria-label="ภาพก่อนหน้า">←</button><button class="soft" onclick="advance(1)" aria-label="ภาพถัดไป">→</button><button class="soft" id="slideToggle" onclick="toggleSlides()">▶ เล่นสไลด์</button></div><a class="primary" id="download" download>ดาวน์โหลดภาพ ↓</a></div></dialog>\n<script>\nconst items=[],byKey=new Map(),nodes=new Map(),grid=document.getElementById(\'grid\'),lightbox=document.getElementById(\'lightbox\');\nlet cursor=null,loading=false,selected=null,slideTimer=null;\nconst get=id=>document.getElementById(id);\nfunction photoTime(item){const m=item.name.match(/^(\\d{4})(\\d{2})(\\d{2})_(\\d{2})(\\d{2})(\\d{2})/);return m?`${m[3]}/${m[2]}/${m[1]} · ${m[4]}:${m[5]}:${m[6]}`:item.name;}\nfunction card(item){const node=document.createElement(\'button\');node.className=\'card\';node.type=\'button\';node.setAttribute(\'aria-label\',\'เปิดภาพ \'+photoTime(item));const img=document.createElement(\'img\');img.loading=\'lazy\';img.decoding=\'async\';img.src=item.thumb;img.alt=\'เฟรม \'+photoTime(item);const foot=document.createElement(\'div\');foot.className=\'card-footer\';const date=document.createElement(\'span\');date.textContent=photoTime(item);const cue=document.createElement(\'span\');cue.className=\'arrow\';cue.textContent=\'เปิดภาพ ↗\';foot.append(date,cue);node.append(img,foot);node.onclick=()=>openPhoto(item.key);return node;}\nfunction merge(rows,head){const fresh=rows.filter(x=>!byKey.has(x.key));for(const item of fresh){byKey.set(item.key,item);nodes.set(item.key,card(item));items.push(item);}items.sort((a,b)=>a.key===b.key?0:a.key<b.key?1:-1);if(fresh.length)for(const item of items)grid.append(nodes.get(item.key));get(\'empty\').hidden=items.length>0;get(\'playBtn\').disabled=!items.length;get(\'presentBtn\').disabled=!items.length;return fresh.length;}async function fetchPage(before){const res=await fetch(\'/captures/list?limit=36\'+(before?\'&before=\'+encodeURIComponent(before):\'\'),{cache:\'no-store\'});if(!res.ok)throw Error(\'โหลดภาพไม่สำเร็จ\');return res.json();}\nasync function loadMore(){if(loading)return;loading=true;get(\'moreBtn\').disabled=true;try{const data=await fetchPage(cursor);merge(data.items,false);cursor=data.next;get(\'total\').textContent=data.total.toLocaleString(\'th-TH\');get(\'moreBtn\').hidden=!cursor;get(\'feedStatus\').textContent=\'อัปเดตภาพใหม่อัตโนมัติ\';get(\'error\').textContent=\'\';}catch(e){get(\'error\').textContent=\'โหลดภาพไม่สำเร็จ ลองอีกครั้ง\';get(\'moreBtn\').hidden=false;get(\'moreBtn\').textContent=\'ลองใหม่\';}finally{loading=false;get(\'moreBtn\').disabled=false;}}\nasync function refreshHead(){if(document.hidden||loading)return;loading=true;try{const data=await fetchPage(null);const added=merge(data.items,true);if(added&&data.next&&data.total>byKey.size){cursor=!cursor||data.next>cursor?data.next:cursor;get(\'moreBtn\').hidden=false;}get(\'total\').textContent=data.total.toLocaleString(\'th-TH\');get(\'feedStatus\').textContent=\'อัปเดตภาพใหม่อัตโนมัติ\';get(\'error\').textContent=\'\';}catch(e){get(\'feedStatus\').textContent=\'รอการเชื่อมต่อ…\';}finally{loading=false;}}\nfunction openPhoto(key){const item=byKey.get(key);if(!item)return;selected=key;get(\'lightImage\').src=item.url;get(\'download\').href=item.url;get(\'download\').download=item.name;get(\'lightTitle\').textContent=photoTime(item);if(!lightbox.open)lightbox.showModal();}\nfunction advance(direction){if(!items.length)return;const index=items.findIndex(x=>x.key===selected);openPhoto(items[(Math.max(0,index)+direction+items.length)%items.length].key);}\nfunction stopSlides(){clearInterval(slideTimer);slideTimer=null;get(\'slideToggle\').textContent=\'▶ เล่นสไลด์\';}\nfunction toggleSlides(){if(slideTimer){stopSlides();return;}slideTimer=setInterval(()=>{if(!document.hidden)advance(1);},4000);get(\'slideToggle\').textContent=\'Ⅱ หยุดสไลด์\';}\nfunction startSlides(){if(!items.length)return;openPhoto(items[0].key);if(!slideTimer)toggleSlides();}\nfunction closeLightbox(){stopSlides();lightbox.close();}\nlightbox.addEventListener(\'close\',stopSlides);lightbox.addEventListener(\'click\',e=>{if(e.target===lightbox){const r=lightbox.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)closeLightbox();}});\ndocument.addEventListener(\'keydown\',e=>{if(!lightbox.open)return;if(e.key===\'ArrowRight\'){e.preventDefault();advance(1);}else if(e.key===\'ArrowLeft\'){e.preventDefault();advance(-1);}});\nasync function togglePresentation(){const on=document.body.classList.toggle(\'presentation\');get(\'presentBtn\').textContent=on?\'ออกจากเต็มจอ\':\'เต็มจอ ↗\';if(on){if(document.documentElement.requestFullscreen)try{await document.documentElement.requestFullscreen();}catch(e){}if(items.length)openPhoto(items[0].key);}else if(document.fullscreenElement)await document.exitFullscreen();}\ndocument.addEventListener(\'fullscreenchange\',()=>{if(!document.fullscreenElement){document.body.classList.remove(\'presentation\');get(\'presentBtn\').textContent=\'เต็มจอ ↗\';}});\nlightbox.addEventListener(\'close\',()=>{if(document.body.classList.contains(\'presentation\')){document.body.classList.remove(\'presentation\');get(\'presentBtn\').textContent=\'เต็มจอ ↗\';if(document.fullscreenElement)document.exitFullscreen().catch(()=>{});}});\nloadMore();setInterval(refreshHead,5000);document.addEventListener(\'visibilitychange\',()=>{if(!document.hidden)refreshHead();});\n</script></body></html>\n\n\n\n'

PAGE = """
<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0b1424">
<title>Ping-Pong Tracker</title>
<style>
html, body { margin: 0; height: 100%; background: #000; overflow: hidden;
  font: 14px -apple-system, sans-serif; color: #fff; }
#stage { position: fixed; inset: 0; overflow: hidden; }
video, canvas#overlay { position: absolute; inset: 0; width: 100%; height: 100%; }
video { object-fit: cover; background: #111; }
#hud { position: fixed; left: 8px; top: 8px; padding: 6px 10px; border-radius: 8px;
  background: rgba(0,0,0,.55); font: 13px ui-monospace, Menlo, monospace;
  white-space: pre; z-index: 5; pointer-events: none; }
#zoomCtl { display: flex; align-items: center; justify-content: center; gap: 6px;
  width: fit-content; max-width: calc(100% - 24px); margin: 0 auto 4px; padding: 4px 8px;
  border-radius: 14px; background: rgba(11,20,36,.88); }
#zoomCtl button { min-width: 44px; min-height: 44px; margin: 0; padding: 6px 12px; }
#zoomCtl input { width: clamp(85px, 22vw, 190px); min-height: 44px; accent-color: #ed8736; }
#zoomValue { min-width: 42px; font-variant-numeric: tabular-nums; }
#ctl { position: fixed; left: 0; right: 0; bottom: 12px; text-align: center; z-index: 5; }
button { padding: 12px 20px; margin: 4px; font-size: 16px; border-radius: 10px; border: 0; }
#status { position: fixed; left: 8px; right: 8px; bottom: 70px; text-align: center;
  z-index: 5; text-shadow: 0 0 4px #000; }
#status a { color: #8cf; }
#ballhud { position: fixed; right: 8px; top: 8px; padding: 6px 10px; border-radius: 8px;
  background: rgba(0,0,0,.55); font: 13px ui-monospace, Menlo, monospace;
  white-space: pre; z-index: 5; pointer-events: none; display: none; }
#calib { position: fixed; inset: 0; z-index: 20; background: #000; display: none; }
#calCv { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; }
#calMsg { position: absolute; left: 8px; right: 8px; top: 10px; text-align: center; font-size: 20px;
  padding: 8px; border-radius: 10px; background: rgba(0,0,0,.65); pointer-events: none; }
#calBtns { position: absolute; left: 0; right: 0; bottom: calc(16px + env(safe-area-inset-bottom)); display: flex; flex-wrap: wrap; justify-content: center; gap: 6px; padding: 0 8px; }
/* ---- start screen (shown on load; the camera opens only from its button) ---- */
[hidden] { display: none !important; }
@keyframes sp-spin { to { transform: rotate(360deg); } }
@keyframes sp-ind { from { margin-left: 0; } to { margin-left: 60%; } }
#splash { position: fixed; inset: 0; z-index: 30; display: flex; box-sizing: border-box;
  overflow-y: auto; -webkit-overflow-scrolling: touch; overscroll-behavior: contain;
  padding: 16px;
  padding: max(16px, env(safe-area-inset-top)) max(16px, env(safe-area-inset-right)) max(16px, env(safe-area-inset-bottom)) max(16px, env(safe-area-inset-left));
  background: #0b1424 radial-gradient(120% 80% at 50% -10%, #17315c 0%, #0b1424 60%) no-repeat;
  color: #eaf0f8;
  font: 16px -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Sarabun', 'Noto Sans Thai', 'Thonburi', 'Leelawadee UI', Tahoma, Roboto, 'Helvetica Neue', Arial, sans-serif;
  line-height: 1.4; transition: opacity .35s ease; }
#splash.hide { opacity: 0; pointer-events: none; }
#splash button { margin: 0; font: inherit; line-height: 1.25; cursor: pointer; touch-action: manipulation;
  -webkit-tap-highlight-color: transparent; }
#splash button:focus-visible, #splash a:focus-visible { outline: 3px solid #ffd08a; outline-offset: 2px; }
#splash a { color: #9fd0ff; }
.sp-card { margin: auto; width: 100%; max-width: 460px; display: flex; flex-direction: column; gap: 20px; }
.sp-left, .sp-right { display: flex; flex-direction: column; gap: 16px; min-width: 0; }
.sp-brand { text-align: center; }
.sp-mark { width: 76px; height: 76px; display: block; margin: 0 auto 10px; }
.sp-brand h1 { margin: 0; font-size: 28px; line-height: 1.2; font-weight: 700; letter-spacing: .2px; }
.sp-sub { margin: 4px 0 0; color: #b3c1d4; font-size: 15px; }
.sp-seg { display: flex; gap: 4px; padding: 4px; border-radius: 14px;
  background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12); }
#splash .sp-seg button { flex: 1 1 0; min-height: 44px; padding: 8px 10px; border: 0; border-radius: 10px;
  background: transparent; color: #c9d4e3; font-size: 14px; }
#splash .sp-seg button[aria-pressed=true] { background: #1E4F8F; color: #fff; font-weight: 600;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.18); }
.sp-warn { padding: 12px 14px; border-radius: 12px; font-size: 14px; color: #ffe2b8;
  background: rgba(232,115,28,.16); border: 1px solid rgba(232,115,28,.65); }
.sp-steps { list-style: none; margin: 0; padding: 4px 16px; border-radius: 16px;
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1); }
.sp-steps li { display: flex; gap: 14px; align-items: flex-start; padding: 13px 0; }
.sp-steps li + li { border-top: 1px solid rgba(255,255,255,.08); }
.sp-ico { flex: none; position: relative; box-sizing: border-box; width: 28px; height: 28px; border-radius: 50%; margin-top: 1px; }
.sp-ico svg { display: none; position: absolute; inset: 0; width: 100%; height: 100%; fill: none; stroke: #fff;
  stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
.sp-steps li[data-state=pending] .sp-ico { border: 2px solid #6b7d96; }
.sp-steps li[data-state=active] .sp-ico { border: 3px solid rgba(255,255,255,.2); border-top-color: #F28A3A;
  animation: sp-spin .9s linear infinite; }
.sp-steps li[data-state=done] .sp-ico { background: #1f9d62; }
.sp-steps li[data-state=done] .ic-ok { display: block; }
.sp-steps li[data-state=error] .sp-ico { background: #d6455a; }
.sp-steps li[data-state=error] .ic-x { display: block; }
.sp-body { min-width: 0; flex: 1 1 auto; }
.sp-lbl { font-size: 16px; font-weight: 600; }
.sp-steps li[data-state=pending] .sp-lbl { color: #9aaabf; font-weight: 500; }
.sp-det { font-size: 14px; color: #b9c6d8; }
.sp-det > div { margin-top: 6px; }
.sp-sr { position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0; overflow: hidden;
  clip: rect(0 0 0 0); white-space: nowrap; border: 0; }
.sp-bar { height: 8px; border-radius: 99px; background: rgba(255,255,255,.14); overflow: hidden; }
.sp-bar > i { display: block; height: 100%; width: 0; border-radius: 99px;
  background: linear-gradient(90deg, #F28A3A, #E8731C); transition: width .2s linear; }
.sp-bar.ind > i { width: 40%; transition: none; animation: sp-ind 1.1s ease-in-out infinite alternate; }
.sp-prog { display: flex; justify-content: space-between; gap: 8px; font-size: 13px; color: #d3dcea;
  font-variant-numeric: tabular-nums; }
.sp-note { font-size: 13px; color: #9fb0c6; }
.sp-chip { display: inline-block; padding: 5px 12px; border-radius: 99px; font-size: 13px; color: #e6f0ff;
  background: rgba(30,79,143,.5); border: 1px solid #4a7fc4; }
.sp-chip.slow { background: rgba(232,115,28,.2); border-color: #E8731C; color: #ffdcc0; }
.sp-chip.pend { opacity: .7; }
.sp-err { padding: 14px; border-radius: 14px; background: rgba(214,69,90,.15); border: 1px solid rgba(255,125,145,.55); }
.sp-err h2 { margin: 0; font-size: 16px; color: #ffd0d7; }
.sp-err p { margin: 4px 0 0; font-size: 13px; color: #ecc9cf; word-break: break-word; }
.sp-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
#splash .sp-btn2 { flex: 1 1 140px; min-height: 44px; padding: 10px 14px; border-radius: 12px; font-size: 15px;
  background: rgba(255,255,255,.12); color: #fff; border: 1px solid rgba(255,255,255,.3); }
.sp-cta { display: flex; flex-direction: column; gap: 10px; text-align: center; }
#splash #spGo { width: 100%; min-height: 60px; padding: 14px 20px; border: 0; border-radius: 16px;
  font-size: 20px; font-weight: 700; background: linear-gradient(#F7A15C, #E8731C); color: #1b1004; }
#splash #spGo:disabled { background: #243145; color: #aab9cd; cursor: default; }
#splash #spGo:not(:disabled):active { transform: scale(.98); }
#spSum { font-size: 13px; color: #aab9cd; min-height: 1.4em; }
#loadpill { position: fixed; left: 50%; top: 40%; transform: translate(-50%, -50%); z-index: 6; display: flex;
  align-items: center; gap: 10px; max-width: calc(100% - 32px); box-sizing: border-box; padding: 10px 16px;
  border-radius: 99px; background: rgba(8,14,26,.85); color: #fff; font-size: 15px; pointer-events: none; }
#loadpill i { flex: none; width: 16px; height: 16px; box-sizing: border-box; border-radius: 50%;
  border: 2px solid rgba(255,255,255,.25); border-top-color: #F28A3A; animation: sp-spin .9s linear infinite; }
@media (orientation: landscape) and (max-height: 600px) {
  .sp-card { max-width: 880px; display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr);
    gap: 16px 32px; align-items: center; }
  .sp-mark { width: 56px; height: 56px; margin-bottom: 6px; }
  .sp-brand h1 { font-size: 24px; }
  .sp-steps li { padding: 8px 0; }
  #splash #spGo { min-height: 52px; font-size: 18px; }
}
@media (min-width: 700px) and (min-height: 601px) {
  .sp-card { max-width: 540px; gap: 24px; }
  .sp-mark { width: 96px; height: 96px; }
  .sp-brand h1 { font-size: 34px; }
}
@media (prefers-reduced-motion: reduce) {
  #splash { transition: none; }
  .sp-bar > i { transition: none; }
  .sp-bar.ind > i { animation: none; width: 100%; opacity: .5; }
  .sp-steps li[data-state=active] .sp-ico, #loadpill i { animation-duration: 2.8s; }
}
/* Focused camera workspace */
#appNav{position:fixed;top:0;left:0;right:0;z-index:8;display:flex;align-items:center;justify-content:space-between;gap:16px;padding:calc(14px + env(safe-area-inset-top)) clamp(16px,4vw,42px) 14px;background:linear-gradient(#0a1425dd,#0a142588,transparent)}
.app-brand{display:flex;align-items:center;gap:10px;color:#fff;text-decoration:none;font-size:12px;font-weight:700;letter-spacing:1px}.app-brand b{color:#f6a364}.app-ball{width:22px;height:22px;border-radius:50%;background:#f6934b;box-shadow:inset -5px -5px 0 #c66b2860}#appNav nav{display:flex;gap:5px;align-items:center;background:#101c2bcc;border:1px solid #ffffff20;padding:4px;border-radius:15px}#appNav nav>a,#appNav .nav-current{color:#bdc9da;font-size:13px;padding:10px 16px;border-radius:11px;text-decoration:none}#appNav .nav-current{background:#ffffff16;color:#fff}#appNav nav>a:hover{background:#ffffff10}
#sessionHint{position:fixed;top:calc(80px + env(safe-area-inset-top));left:clamp(16px,4vw,42px);z-index:5;display:flex;gap:9px;align-items:center;background:#101c2bc9;border:1px solid #ffffff20;border-radius:99px;padding:9px 13px;font-size:12px;color:#d3dce8;pointer-events:none}#sessionDot{width:7px;height:7px;border-radius:50%;background:#8392a8}body.camera-running #sessionDot{background:#8be0b3}.fps-limit{padding-left:9px;border-left:1px solid #ffffff25;color:#93a4bd;font-size:11px}
#ctl{bottom:calc(18px + env(safe-area-inset-bottom));left:50%;right:auto;transform:translateX(-50%);width:max-content;max-width:calc(100% - 24px)}#zoomCtl{width:fit-content;margin:0 auto 10px;background:#101c2bd9;border:1px solid #ffffff22;padding:2px 10px;border-radius:99px;gap:8px}#zoomCtl button{color:#e6edf6;background:transparent;padding:4px 8px;font-size:21px;min-width:38px;min-height:40px}#zoomCtl #zoomReset{font-size:12px;color:#a4b2c8}#zoomCtl input{width:clamp(65px,18vw,190px);min-height:40px;accent-color:#f6934b}#zoomValue{color:#fff;font-size:13px;min-width:38px}.zoom-caption{font-size:12px;color:#a8b6cb}.action-dock{display:flex;align-items:stretch;gap:7px;background:#101c2bf2;border:1px solid #ffffff25;box-shadow:0 16px 50px #0005;border-radius:22px;padding:8px}#ctl .action-dock button{display:flex;align-items:center;justify-content:center;gap:9px;border-radius:15px;min-height:56px;margin:0;padding:13px 22px;font-size:15px;font-weight:650;cursor:pointer}#ctl .camera-action{background:#f6934b;color:#151f2c;min-width:147px}.action-icon{font-size:18px}body.camera-running #ctl .camera-action{background:#243348;color:#ffb88f}#ctl .save-action{background:#f6f4ee;color:#172334;min-width:151px}#ctl .tools-action{background:#213047;color:#e2eaf5;padding:13px 17px!important}#ctl button:disabled{opacity:.42;cursor:default}#ctl button:focus-visible,#appNav a:focus-visible,#toolsPanel button:focus-visible{outline:3px solid #ffb678;outline-offset:3px}
#toolsPanel{position:fixed;z-index:16;right:clamp(12px,4vw,40px);bottom:calc(105px + env(safe-area-inset-bottom));width:min(340px,calc(100vw - 32px));background:#111f31;box-shadow:0 20px 80px #0008;border:1px solid #ffffff20;border-radius:23px;padding:20px;box-sizing:border-box}.tools-heading{display:flex;justify-content:space-between;align-items:center;font-size:19px}.tools-heading button{color:#a4b4cb;background:transparent;padding:5px;margin:0;font-size:17px;min-height:40px;min-width:40px}#toolsPanel p{color:#9aacc5;font-size:12px;margin:18px 0 8px}#toolsPanel>button,#toolsPanel .tools-row button{background:#22344d;color:#e7edf7;font-size:13px;min-height:44px;margin:0;padding:11px 14px}#toolsPanel>button{width:100%;text-align:left}.tools-row{display:flex;gap:8px;margin:8px 0 22px}.tools-row>button{flex:1}.stats-label{font-size:13px;display:flex;gap:10px;align-items:center;border-top:1px solid #ffffff18;padding-top:18px;min-height:36px}.stats-label input{accent-color:#f6934b;width:17px;height:17px}#toolsPanel small{display:block;color:#8194af;font-size:10px;margin-top:15px;line-height:1.7}
#hud{display:none;top:calc(120px + env(safe-area-inset-top));left:16px;max-width:calc(100vw - 48px);font-size:11px;padding:12px 14px;background:#101c2bdd;border:1px solid #ffffff1a;border-radius:15px}.show-stats #hud{display:block}#ballhud{top:calc(80px + env(safe-area-inset-top));right:20px;font-size:11px}#status{bottom:calc(170px + env(safe-area-inset-bottom));font-size:12px;pointer-events:none}#status a{pointer-events:auto}#toast{position:fixed;z-index:25;left:50%;transform:translateX(-50%);bottom:calc(180px + env(safe-area-inset-bottom));background:#f7f4ed;color:#172334;border:1px solid #fff7;border-radius:14px;padding:13px 18px;box-shadow:0 10px 35px #0005;font-size:13px;max-width:calc(100% - 50px);text-align:center}#toast a{color:#ba571e;margin-left:10px;font-weight:700}.intro-gallery{display:inline-flex;align-items:center;gap:8px;color:#1e4f8f;text-decoration:none;font-size:13px;margin-top:18px;font-weight:700}
@media(max-width:550px){#appNav{padding-left:15px;padding-right:15px}.app-brand{font-size:10px;gap:7px;letter-spacing:.2px}.app-ball{width:18px;height:18px}#appNav nav>a,#appNav .nav-current{padding:9px 12px;font-size:12px}#ctl .action-dock button{padding:12px 15px;min-height:53px;font-size:13px}#ctl .camera-action{min-width:113px}#ctl .save-action{min-width:123px}#ctl .tools-action{padding:12px!important}.fps-limit{display:none}.action-dock{gap:5px;padding:6px;border-radius:19px}#zoomCtl{gap:3px}#zoomCtl input{width:110px}#zoomCtl #zoomReset{display:none}.zoom-caption{display:none}}

</style>
</head>
<body>
<div id="stage">
<video id="video" autoplay playsinline muted></video>
<canvas id="overlay"></canvas>
</div>
<div id="ballhud"></div>
<div id="calib">
<canvas id="calCv"></canvas>
<div id="calMsg"></div>
<div id="calBtns">
<button id="calAuto" onclick="calDetectTable()" title="ส่งภาพนิ่งหนึ่งเฟรมไปตรวจบนเซิร์ฟเวอร์">จับโต๊ะอัตโนมัติ</button>
<button id="calDone" onclick="calFinish()">ยืนยันมุม</button>
<button onclick="calRedo()">Redo</button>
<button onclick="closeCalib()">Cancel</button>
</div>
</div>
<div id="hud">display fps -
detect fps  -
inference   - ms
round-trip  - ms
tracks      -
rescued     -
saved       -</div>
<div id="status">หยุดกล้องแล้ว</div>
<header id="appNav"><a class="app-brand" href="/" aria-label="Ping-Pong Tracker"><span class="app-ball"></span><span>PING-PONG <b>TRACKER</b></span></a><nav aria-label="เมนูหลัก"><span class="nav-current">กล้อง</span><a href="/gallery" onclick="stopCamera()">▦ แกลเลอรี</a></nav></header>
<div id="sessionHint"><span id="sessionDot"></span><span id="sessionLabel">พร้อมเปิดกล้อง</span><span class="fps-limit">สูงสุด 45 FPS</span></div>
<div id="ctl">
<div id="zoomCtl" role="group" aria-label="ซูมกล้อง"><span class="zoom-caption">ซูม</span><button id="zoomOut" type="button" onclick="setCameraZoom(cameraZoom - 0.25)" aria-label="ซูมออก" disabled>−</button><input id="zoomRange" type="range" min="1" max="4" step="0.05" value="1" aria-label="ระดับซูมกล้อง" oninput="setCameraZoom(this.value)" disabled><output id="zoomValue" for="zoomRange">1.0×</output><button id="zoomIn" type="button" onclick="setCameraZoom(cameraZoom + 0.25)" aria-label="ซูมเข้า" disabled>+</button><button id="zoomReset" type="button" onclick="setCameraZoom(1)" aria-label="คืนซูมเป็นหนึ่งเท่า" disabled>1×</button></div>
<div class="action-dock"><button class="camera-action" id="cameraAction" onclick="toggleCameraAction()"><span class="action-icon">●</span><span id="cameraActionLabel">เปิดกล้อง</span></button><button class="save-action" id="saveAction" onclick="saveManual()" disabled><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M4 7h4l2-3h4l2 3h4v13H4z"/><circle cx="12" cy="13" r="4"/></svg><span id="saveActionLabel">บันทึกภาพ</span></button><button class="tools-action" id="toolsAction" onclick="toggleTools()" aria-label="ตั้งค่า" aria-expanded="false"><svg width="23" height="23" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h16"/><circle cx="8" cy="6" r="2" fill="#172334"/><circle cx="16" cy="12" r="2" fill="#172334"/><circle cx="10" cy="18" r="2" fill="#172334"/></svg></button></div>
</div>
<aside id="toolsPanel" hidden aria-label="ตั้งค่ากล้อง"><div class="tools-heading"><strong>ตั้งค่า</strong><button onclick="toggleTools(false)" aria-label="ปิดตั้งค่า">✕</button></div><p>กำหนดพื้นที่โต๊ะ</p><button id="calBtn" onclick="toggleTools(false);openCalib()">กำหนดมุมโต๊ะ ↗</button><div class="tools-row"><button id="gridBtn" onclick="toggleGrid()">เส้นโต๊ะ: เปิด</button><button onclick="resetCalib();showToast('ล้างมุมโต๊ะแล้ว')">ล้างมุมโต๊ะ</button></div><label class="stats-label"><input type="checkbox" id="showStats" onchange="document.body.classList.toggle('show-stats',this.checked)"> แสดงข้อมูลประสิทธิภาพ</label><small>วาดกรอบสูงสุด 45 FPS · ประมวลผลบนเครื่องนี้</small></aside>
<!-- Retained internal hooks for fallback/collection; technical controls stay out of the camera workflow. -->
<div hidden><button id="trackBtn">Track: on</button><button id="collectBtn">Collect: off</button><button id="inferBtn">Infer: device</button><button id="roiBtn">ROI box: off</button></div>
<div id="toast" role="status" hidden></div>
<div id="loadpill" role="status" hidden><i></i><span id="loadpillT"></span></div>
<div id="splash" role="dialog" aria-modal="true" aria-labelledby="spTitle">
<div class="sp-card">
<div class="sp-left">
<div class="sp-brand">
<svg class="sp-mark" viewBox="0 0 64 64" aria-hidden="true" focusable="false">
<rect width="64" height="64" rx="16" fill="#0e2140"/>
<path d="M8 44h48l5 10H3z" fill="#1E4F8F"/>
<path d="M8 44h48" stroke="#8FB8EC" stroke-width="2.4" stroke-linecap="round"/>
<ellipse cx="32" cy="48.5" rx="10" ry="2" fill="#07101f" opacity=".6"/>
<circle cx="32" cy="28" r="11.5" fill="#E8731C"/>
<circle cx="27.8" cy="23.8" r="3.8" fill="#F8AB6A" opacity=".85"/>
<path d="M12 20q4-9 12-8" fill="none" stroke="#E8731C" stroke-opacity=".55" stroke-width="2" stroke-dasharray="1 4.5" stroke-linecap="round"/>
</svg>
<h1 id="spTitle">Ping-Pong Tracker</h1>
<p class="sp-sub">ติดตามลูกปิงปองด้วยกล้อง</p><a class="intro-gallery" href="/gallery">▦ ดูแกลเลอรีภาพที่บันทึก →</a>
</div>
<div class="sp-seg" role="group" aria-label="เลือกที่ประมวลผล">
<button type="button" id="spModeDev" data-mode="device" aria-pressed="true">ประมวลผลบนเครื่องนี้</button>
<button type="button" id="spModeSrv" data-mode="server" aria-pressed="false">ประมวลผลบน server</button>
</div>
</div>
<div class="sp-right">
<div id="spSecure" class="sp-warn" role="alert" hidden></div>
<div id="spNotice" class="sp-warn" role="status" hidden></div>
<ol class="sp-steps">
<li id="spS1" data-state="active">
<span class="sp-ico" aria-hidden="true"><svg class="ic-ok" viewBox="0 0 24 24"><path d="M6 12.5l4 4 8-9"/></svg><svg class="ic-x" viewBox="0 0 24 24"><path d="M7 7l10 10M17 7L7 17"/></svg></span>
<div class="sp-body">
<div class="sp-lbl" id="spL1">ดาวน์โหลดโมเดล</div><span class="sp-sr" id="spR1"></span>
<div class="sp-det">
<div class="sp-bar" id="spBar" role="progressbar" aria-label="ความคืบหน้าการดาวน์โหลดโมเดล" aria-valuemin="0" aria-valuemax="100"><i id="spBarI"></i></div>
<div class="sp-prog" id="spProg"><span id="spMB"></span><span id="spPct"></span></div>
<div class="sp-note" id="spNote">ครั้งแรกเท่านั้น ครั้งต่อไปจะเร็วขึ้น</div>
<div id="spT1" hidden></div>
</div>
</div>
</li>
<li id="spS2" data-state="pending">
<span class="sp-ico" aria-hidden="true"><svg class="ic-ok" viewBox="0 0 24 24"><path d="M6 12.5l4 4 8-9"/></svg><svg class="ic-x" viewBox="0 0 24 24"><path d="M7 7l10 10M17 7L7 17"/></svg></span>
<div class="sp-body">
<div class="sp-lbl" id="spL2">เตรียมการประมวลผล</div><span class="sp-sr" id="spR2"></span>
<div class="sp-det">
<div id="spChipBox" hidden><span class="sp-chip" id="spChip"></span></div>
<div id="spT2" hidden></div>
</div>
</div>
</li>
<li id="spS3" data-state="pending">
<span class="sp-ico" aria-hidden="true"><svg class="ic-ok" viewBox="0 0 24 24"><path d="M6 12.5l4 4 8-9"/></svg><svg class="ic-x" viewBox="0 0 24 24"><path d="M7 7l10 10M17 7L7 17"/></svg></span>
<div class="sp-body">
<div class="sp-lbl" id="spL3">ทดสอบโมเดล</div><span class="sp-sr" id="spR3"></span>
<div class="sp-det"><div id="spT3" hidden></div><div id="spQHint" hidden>หรือใช้ประมวลผลบนเครื่องนี้ทันทีโดยไม่ต้องรอ</div>
<div class="sp-row"><button type="button" class="sp-btn2" id="spQDev" hidden>ประมวลผลบนเครื่องนี้แทน</button></div></div>
</div>
</li>
</ol>
<div id="spErr" class="sp-err" role="alert" hidden>
<h2 id="spErrT"></h2>
<p id="spErrR"></p>
<div class="sp-row">
<button type="button" class="sp-btn2" id="spRetry">ลองใหม่</button>
<button type="button" class="sp-btn2" id="spUseSrv">ใช้ server แทน</button>
</div>
</div>
<div id="spCamErr" class="sp-err" role="alert" hidden>
<h2 id="spCamT"></h2>
<p id="spCamR"></p>
</div>
<div class="sp-cta">
<button type="button" id="spGo" disabled>กำลังเตรียม…</button>
<div id="spSum"></div>
</div>
<div class="sp-sr" id="spLive" aria-live="polite"></div>
</div>
</div>
</div>
<script>
// ==== CONFIG BEGIN (all tuning constants) ====
const FIT = 'cover';            // must match CSS object-fit of #video
const MAX_INFLIGHT = 2;
const SEND_SIDE = 960, JPEG_Q = 0.6;
let serverSendSide = SEND_SIDE;   // long side of frames POSTed to /detect = server input size (from /health), clamped [320,1280]
const LOW_CONF = 0.08;          // conf sent to /detect (blurred balls score low)
const START_CONF = 0.25;        // min conf to birth a track; also the "raw detector" threshold
const MAX_MISSES = 5;           // coast for at most this many detection updates...
const MAX_COAST_MS = 300;       // ...or this long since last hit
const TENT_MAX_MISSES = 1;      // unconfirmed track is dropped after more misses than this
const CONFIRM_HITS = 2;
const GATE_BASE = 0.03;         // gate (normalized units) = (BASE + SPEED_K*speed*dt + SIZE_K*max(w,h)) * (1 + MISS_K*misses)
const GATE_SPEED_K = 1.5;
const GATE_SIZE_K = 1.0;
const GATE_MISS_K = 0.5;
const GATE_NEW_MULT = 2.5;      // gate multiplier while velocity is still unknown (first hit only)
const ACCEL_STD = 6;            // process noise: accel std, normalized units / s^2
const MEAS_STD = 0.012;         // measurement noise std, normalized units
const LOWCONF_R_BOOST = 2;      // R multiplier for detections below START_CONF
const WH_ALPHA = 0.4;           // EMA factor for w/h
const EXTRAP_MAX_MS = 150;      // max extrapolation past last update when rendering
const MAX_SPEED_NORM = 3;       // max ball speed, screen-widths/s (re-acquire gate growth)
const REACQUIRE_MS = 400;       // a lost confirmed track can be revived (same ID) within this
const CONFIRM_CONF = 0.5;       // a single hit this confident confirms a track immediately
const ELONG_ASPECT = 1.6;       // aspect above this = motion-blur streak
const SIZE_RATIO_MAX = 3;       // max size ratio det/track for a normal detection
const ELONG_SIZE_RATIO = 6;     // relaxed size ratio for elongated detections
const MANEUVER_NIS = 9;         // normalized innovation^2 above this = impulsive maneuver
const MANEUVER_VEL_STD = 1.5;   // velocity jump std (norm/s) injected on a maneuver
const Q_BOOST_MAX = 5;          // cap on adaptive ACCEL_STD multiplier
const MISS_Q_BOOST = 3;         // ACCEL_STD multiplier while the track is missing detections
const COLLECT_MIN_MS = 334;     // data collection: max 3 uploads/s
const COLLECT_JPEG_Q = 0.9;
const SOLID_MS = 250;           // "recently matched" window for solid box
const TRAIL_LEN = 20;
const OFFSCREEN_MARGIN = 0.2;   // drop track when center is this far outside the frame
// ---- table mapping ----
const TABLE_W = 1.525, TABLE_L = 2.74, NET_Y = 1.37;   // meters; origin near-left corner, X across, Y away from camera
const IN_MARGIN = 0.03;         // "inside table" tolerance, m
const GRID_STEP = 0.25;         // verification grid spacing, m
const CALIB_KEY = 'ttCalib_v1'; // localStorage key (corners in normalized video coords)
const LOUPE_ZOOM = 2.5, LOUPE_R = 62, HANDLE_R = 15, HANDLE_HIT = 38;   // calibration UI, css px
// ---- on-device inference: rectangular full frame + ROI crop around the tracked ball ----
const INF_LONG = 640;           // one on-device profile; camera preview remains full resolution
const INF_ROI = 320;            // ROI model input (square)
const ROI_ENABLED = false;      // full-frame baseline: do not lose context in 320px crops
const ROI_MIN_PX = 320;         // min ROI side in full-resolution video pixels
const ROI_BOX_K = 6;            // ROI side >= K * ball box size (px)
const ROI_LAT_MS = 60;          // expected latency; ROI grows by |velocity| * latency * 2
const ROI_MAX_COAST = 2;        // ROI only if the track was matched within this many updates
const ROI_MISS_TO_FULL = 2;     // consecutive ROI runs without a ball -> full frame
const ROI_FULL_EVERY = 15;      // safety: every Nth run is a full-frame scan
// ==== CONFIG END ====

// ==== TRACKER BEGIN ====
// Small grayscale template follower. Display-only: never writes analytic tracks.
class PixelFollower {
  constructor(options = {}) {
    this.clock = options.clock || (() => performance.now());
    this.maxStates = Math.max(1, Math.min(8, options.maxStates || 8));
    this.minScore = options.minScore === undefined ? 0.72 : options.minScore;
    this.minMargin = options.minMargin === undefined ? 0.065 : options.minMargin;
    this.minStd = options.minStd === undefined ? 6 : options.minStd;
    this.reset();
  }
  reset() { this.frames = []; this.states = new Map(); this.W = 0; this.H = 0; this.lastReason = 'waiting'; }
  _frame(frame, copy) {
    if (!frame || !Number.isInteger(frame.W) || !Number.isInteger(frame.H) ||
        frame.W < 16 || frame.H < 16 || frame.W * frame.H > 1048576 ||
        !Number.isFinite(frame.time) || !ArrayBuffer.isView(frame.gray) ||
        frame.gray.BYTES_PER_ELEMENT !== 1 || frame.gray.length !== frame.W * frame.H) return null;
    return { W: frame.W, H: frame.H, time: frame.time,
             gray: copy ? new Uint8Array(frame.gray) : frame.gray };
  }
  _shape(frame) {
    if (this.W && (this.W !== frame.W || this.H !== frame.H)) this.reset();
    this.W = frame.W; this.H = frame.H;
  }
  _prune(now) {
    for (const [id, state] of this.states) {
      if (!state.valid || now - state.receipt > 800 || now < state.receipt - 1) this.states.delete(id);
    }
  }
  pushFrame(input) {
    const frame = this._frame(input, true);
    if (!frame) return false;
    this._shape(frame);
    const last = this.frames[this.frames.length - 1];
    if (last && frame.time < last.time - 900) return false;
    let index = this.frames.findIndex(f => f.time >= frame.time);
    if (index >= 0 && this.frames[index].time === frame.time) return false;
    if (index < 0) index = this.frames.length;
    this.frames.splice(index, 0, frame);
    const newest = this.frames[this.frames.length - 1].time;
    while (this.frames.length && (this.frames.length > 32 || this.frames[0].time < newest - 900)) this.frames.shift();
    this._prune(this.clock());
    // An out-of-order frame is retained for later exact-anchor replay, but cannot
    // update an existing state twice or roll it backwards.
    for (const state of this.states.values()) if (frame.time > state.time) {
      this._step(state, frame);
      this.lastReason = state.valid ? 'ok' : state.reason || 'lost';
    }
    return true;
  }
  _template(frame, tr, receipt) {
    const x = tr.mx * frame.W, y = tr.my * frame.H;
    const bw = tr.w * frame.W, bh = tr.h * frame.H;
    if (![x, y, bw, bh].every(Number.isFinite) || bw <= 0 || bh <= 0 ||
        tr.mx < 0 || tr.mx > 1 || tr.my < 0 || tr.my > 1) { this.lastReason = 'bad-box'; return null; }
    const ix = Math.round(x), iy = Math.round(y);
    // Eleven samples span roughly 1.25 bbox widths: include the ball outline,
    // rather than matching the uniform center of a white/orange disk.
    const sx = Math.max(0.75, bw / 8);
    const sy = Math.max(0.75, bh / 8);
    const offsets = [], values = []; let sum = 0, sumSq = 0;
    let rx = 0, ry = 0;
    for (let gy = -5; gy <= 5; gy++) for (let gx = -5; gx <= 5; gx++) {
      const dx = Math.round(gx * sx), dy = Math.round(gy * sy);
      const xx = ix + dx, yy = iy + dy;
      if (xx < 1 || yy < 1 || xx >= frame.W - 1 || yy >= frame.H - 1) { this.lastReason = 'edge'; return null; }
      rx = Math.max(rx, Math.abs(dx)); ry = Math.max(ry, Math.abs(dy));
      offsets.push(dy * frame.W + dx);
      const v = frame.gray[yy * frame.W + xx]; values.push(v); sum += v; sumSq += v * v;
    }
    const n = values.length, mean = sum / n, energy = sumSq - sum * mean;
    if (!(energy > 0) || Math.sqrt(energy / n) < this.minStd) { this.lastReason = 'flat-template'; return null; }
    const centered = new Float32Array(values.map(v => v - mean));
    return { id: tr.id, x, y, fx: x - ix, fy: y - iy, time: frame.time,
             receipt, valid: true, quality: 1, steps: 0, conf: Number.isFinite(tr.conf) ? tr.conf : 0,
             vx: Number.isFinite(tr.vx) ? tr.vx * frame.W : 0,
             vy: Number.isFinite(tr.vy) ? tr.vy * frame.H : 0,
             offsets, centered, energy, rx, ry,
             exclusion: Math.max(3, Math.min(20, Math.min(bw, bh) * 0.65)) };
  }
  anchor(tracks, tCap, exactInput) {
    const exact = this._frame(exactInput, false);
    if (!exact || exact.time !== tCap || !Number.isFinite(tCap)) return { anchored: 0, valid: 0 };
    this._shape(exact);
    const receipt = this.clock(); this._prune(receipt);
    const selected = tracks.filter(tr => tr && tr.confirmed && tr.mt === tCap &&
      (typeof tr.id === 'string' || Number.isFinite(tr.id))).sort((a, b) => (b.conf || 0) - (a.conf || 0)).slice(0, this.maxStates);
    let anchored = 0, valid = 0;
    for (const tr of selected) {
      const state = this._template(exact, tr, receipt);
      if (!state) { this.states.delete(tr.id); continue; }
      anchored++;
      // Start at the actual old camera pixels, then catch up in chronological
      // order. Matching today's pixels to a 160ms-old box is deliberately avoided.
      for (const frame of this.frames) if (frame.time > tCap && state.valid) this._step(state, frame);
      if (state.valid) { valid++; this.states.set(tr.id, state); this.lastReason = 'ok'; }
      else { this.states.delete(tr.id); this.lastReason = state.reason || 'replay-lost'; }
    }
    if (this.states.size > this.maxStates) {
      const keep = new Set([...this.states.values()].sort((a, b) => b.receipt - a.receipt || b.conf - a.conf)
        .slice(0, this.maxStates).map(s => s.id));
      for (const id of this.states.keys()) if (!keep.has(id)) this.states.delete(id);
    }
    return { anchored, valid };
  }
  _score(state, frame, x, y) {
    if (x - state.rx < 1 || y - state.ry < 1 || x + state.rx >= frame.W - 1 || y + state.ry >= frame.H - 1) return -2;
    const center = y * frame.W + x, n = state.offsets.length;
    let sum = 0, sumSq = 0, dot = 0;
    for (let i = 0; i < n; i++) {
      const v = frame.gray[center + state.offsets[i]];
      sum += v; sumSq += v * v; dot += state.centered[i] * v;
    }
    const variance = sumSq - sum * sum / n;
    if (!(variance > 0) || Math.sqrt(variance / n) < this.minStd) return -2;
    return Math.max(-1, Math.min(1, dot / Math.sqrt(state.energy * variance)));
  }
  _step(state, frame) {
    if (!state.valid || frame.time <= state.time) return;
    const dt = (frame.time - state.time) / 1000;
    if (dt > 0.125) { state.valid = false; state.reason = 'frame-gap'; return; }
    const px = state.vx * dt, py = state.vy * dt, displacement = Math.hypot(px, py);
    // A just-confirmed analytic track often has zero velocity. Its first pixel
    // displacement is unknown, so do not narrow the first search to 12px.
    const radius = state.steps === 0 ? 40 : Math.max(12, Math.min(40, Math.ceil(12 + 2 * displacement)));
    // A high-speed prediction is unreliable through a bounce. Widen the search
    // and reduce directional bias; the 40px per-frame reach remains bounded.
    const predictionWeight = Math.max(0, Math.min(1, (20 - displacement) / 8));
    const cx = Math.round(state.x - state.fx + px * predictionWeight);
    const cy = Math.round(state.y - state.fy + py * predictionWeight);
    const left = cx - radius, right = cx + radius, top = cy - radius, bottom = cy + radius;
    const scored = new Map();
    const evaluate = (x, y) => {
      if (x < left || x > right || y < top || y > bottom) return null;
      const key = x + ',' + y;
      if (scored.has(key)) return scored.get(key);
      const hit = { x, y, score: this._score(state, frame, x, y) };
      scored.set(key, hit); return hit;
    };
    for (let y = top; y <= bottom; y += 3) for (let x = left; x <= right; x += 3) evaluate(x, y);
    evaluate(right, bottom); evaluate(cx, cy);
    const coarse = [...scored.values()].sort((a, b) => b.score - a.score);
    const peaks = [];
    for (const hit of coarse) {
      if (hit.score <= -1) break;
      if (peaks.every(p => Math.hypot(p.x - hit.x, p.y - hit.y) >= 3)) peaks.push(hit);
      if (peaks.length === 3) break;
    }
    // Small disks can have a narrow correlation peak that a coarse grid misses
    // against textured surroundings. Always refine the prior/predicted centers
    // too; keep the same score/margin rejection rather than accepting a weak peak.
    const fineCenters = [...peaks, { x: cx, y: cy },
      { x: Math.round(state.x - state.fx), y: Math.round(state.y - state.fy) }];
    for (const peak of fineCenters) for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) evaluate(peak.x + dx, peak.y + dy);
    const all = [...scored.values()].sort((a, b) => b.score - a.score);
    const best = all[0];
    if (!best || best.score < this.minScore || best.x <= left + 1 || best.x >= right - 1 ||
        best.y <= top + 1 || best.y >= bottom - 1 || best.x - state.rx <= 1 || best.y - state.ry <= 1 ||
        best.x + state.rx >= frame.W - 2 || best.y + state.ry >= frame.H - 2) { state.valid = false; state.reason = !best || best.score < this.minScore ? 'weak-match' : 'search-edge'; return; }
    const second = all.find(p => Math.hypot(p.x - best.x, p.y - best.y) >= state.exclusion);
    if (second && best.score - second.score < this.minMargin) { state.valid = false; state.reason = 'ambiguous'; return; }
    const subpixel = (minus, center, plus) => {
      if (!minus || !plus || minus.score < -1 || plus.score < -1) return 0;
      const denominator = minus.score - 2 * center.score + plus.score;
      return denominator < -1e-6 ? Math.max(-0.5, Math.min(0.5, (minus.score - plus.score) / (2 * denominator))) : 0;
    };
    const ox = subpixel(evaluate(best.x - 1, best.y), best, evaluate(best.x + 1, best.y));
    const oy = subpixel(evaluate(best.x, best.y - 1), best, evaluate(best.x, best.y + 1));
    const x = best.x + state.fx + ox, y = best.y + state.fy + oy;
    if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0 || x >= frame.W || y >= frame.H) { state.valid = false; return; }
    state.vx = 0.6 * (x - state.x) / dt + 0.4 * state.vx;
    state.vy = 0.6 * (y - state.y) / dt + 0.4 * state.vy;
    state.x = x; state.y = y; state.time = frame.time; state.quality = best.score; state.steps++;
  }
  read(id, now = this.clock()) {
    const state = this.states.get(id);
    if (!state || !state.valid || !Number.isFinite(now) || now < state.time || now - state.time > 100 ||
        now < state.receipt - 1 || now - state.receipt > 800) return null;
    return { x: state.x / this.W, y: state.y / this.H, time: state.time, quality: state.quality };
  }
}
if (typeof module !== 'undefined' && module.exports) module.exports = { PixelFollower };

// Display records persist independently of detector membership; analytic tracks are never modified.
// Bounded worker bridge: template search/replay never runs on the UI thread.
class AsyncPixelFollower {
  constructor() {
    this.worker = null; this.disabled = false; this.generation = 0;
    this.reset();
  }
  reset() {
    this.generation++; this.busy = false; this.pendingFrame = null; this.pendingAnchor = null;
    this.values = new Map(); this.lastReason = this.disabled ? 'worker-off' : 'waiting';
    this.lastMs = 0;
    if (this.worker) this.worker.postMessage({ type: 'reset', gen: this.generation });
  }
  _ensure() {
    if (this.worker || this.disabled) return !!this.worker;
    let url;
    try {
      const code = PixelFollower.toString() + `
        let now = 0;
        const follower = new PixelFollower({clock: () => now});
        self.onmessage = ({data: m}) => {
          now = m.now || 0;
          if (m.type === 'reset') { follower.reset(); return; }
          const start = performance.now();
          try {
            if (m.type === 'frame') follower.pushFrame(m.frame);
            else follower.anchor(m.tracks, m.time, m.frame);
            const values = [];
            for (const state of follower.states.values()) {
              if (!state.valid) continue;
              values.push({id:state.id,x:state.x/follower.W,y:state.y/follower.H,
                time:state.time,receipt:state.receipt,quality:state.quality});
            }
            self.postMessage({gen:m.gen,values,reason:follower.lastReason,ms:performance.now()-start});
          } catch(e) { self.postMessage({gen:m.gen,values:[],reason:'worker-error',ms:0}); }
        };`;
      url = URL.createObjectURL(new Blob([code], {type: 'text/javascript'}));
      this.worker = new Worker(url);
      this.worker.onmessage = ({data}) => {
        if (data.gen !== this.generation) return;
        this.busy = false; this.values = new Map(data.values.map(v => [v.id,v]));
        this.lastReason = data.reason; this.lastMs = data.ms; this._dispatch();
      };
      this.worker.onerror = () => this._disable();
      return true;
    } catch(e) { this._disable(); return false; }
    finally { if (url) URL.revokeObjectURL(url); }
  }
  _disable() {
    if (this.worker) this.worker.terminate();
    this.worker = null; this.disabled = true; this.reset();
  }
  _dispatch() {
    if (this.busy || !this._ensure()) return;
    const job = this.pendingAnchor || this.pendingFrame;
    if (!job) return;
    if (job.type === 'anchor') this.pendingAnchor = null; else this.pendingFrame = null;
    this.busy = true;
    try {
      this.worker.postMessage({...job,gen:this.generation,now:performance.now()}, [job.frame.gray.buffer]);
    } catch(e) { this._disable(); }
  }
  pushFrame(frame) { this.pendingFrame = {type:'frame',frame}; this._dispatch(); }
  anchor(tracks, time, frame) {
    // Serialize confirmed tracks explicitly: prototype getters do not survive structured cloning.
    const selected = tracks.filter(tr => tr.confirmed && tr.mt === time).map(tr => ({
      id:tr.id,confirmed:true,mt:tr.mt,mx:tr.mx,my:tr.my,w:tr.w,h:tr.h,vx:tr.vx,vy:tr.vy,conf:tr.conf
    }));
    this.pendingAnchor = {type:'anchor',tracks:selected,time,frame}; this._dispatch();
  }
  read(id, now = performance.now()) {
    const v = this.values.get(id);
    if (!v || now < v.time || now - v.time > 100 || now < v.receipt - 1 || now - v.receipt > 800) return null;
    return v;
  }
}

class SmoothOverlay {
  constructor() { this.reset(); }
  reset() { this.records = new Map(); this.expired = new WeakMap(); this.lastNow = null; this.nextVisualId = 1; }
  sample(tracks, now, follower = null) {
    if (this.lastNow !== null && (now < this.lastNow || now - this.lastNow > 1000)) {
      // A tab/frame stall must not resurrect an already expired detection.
      for (const r of this.records.values()) this.expired.set(r.track, r.hitStamp);
      this.records.clear(); this.lastNow = null;
    }
    const dt = this.lastNow === null ? 1 / 60 : Math.max(0, (now - this.lastNow) / 1000);
    this.lastNow = now;
    const live = tracks.filter(tr => tr.confirmed);
    const liveIds = new Set(live.map(tr => tr.id));
    const claimed = new Set();
    for (const tr of live) {
      if (this.expired.get(tr) === tr.tHit) continue;
      let r = this.records.get(tr.id);
      if (!r) {
        // A nearby re-born track can keep the existing visual box during a short gap.
        // Never take a record whose source still exists, and match each record only once.
        let best = null, bestDistance = Infinity;
        for (const candidate of this.records.values()) {
          if (liveIds.has(candidate.track.id) || claimed.has(candidate) || candidate.track.label !== tr.label) continue;
          const age = now - candidate.lastReceipt;
          if (age > Math.min(700, Math.max(320, candidate.cadence * 2.2)) + 180) continue;
          const size = Math.max(tr.w, tr.h, 1e-6), oldSize = Math.max(candidate.w, candidate.h, 1e-6);
          if (Math.max(size / oldSize, oldSize / size) > 2.5) continue;
          const gate = Math.min(0.12, Math.max(0.035, 3 * size) + Math.hypot(candidate.vx, candidate.vy) * Math.min(age, 150) / 1000);
          const distance = Math.hypot(tr.x - candidate.x, tr.y - candidate.y);
          if (distance <= gate && distance < bestDistance) { best = candidate; bestDistance = distance; }
        }
        if (best) {
          this.records.delete(best.track.id); r = best; r.hitStamp = null; r.stamp = null;
        } else {
          r = { visualId: this.nextVisualId++, x: tr.x, y: tr.y, w: tr.w, h: tr.h,
                vx: 0, vy: 0, alpha: 0.15, cadence: 120, lastReceipt: now,
                hitStamp: null, stamp: null, first: true, trail: [] };
        }
        this.records.set(tr.id, r);
      }
      claimed.add(r);
      if (r.hitStamp !== tr.tHit) {
        if (r.hitStamp !== null) r.cadence = 0.65 * r.cadence + 0.35 * Math.max(16, Math.min(500, now - r.lastReceipt));
        r.hitStamp = tr.tHit; r.lastReceipt = now;
      }
      r.track = tr;
      r.sx = tr.x; r.sy = tr.y; r.stamp = tr.t;
      // Do not slow a valid fast-ball estimate just to smooth the display.
      r.vx = Number.isFinite(tr.vx) ? tr.vx : 0;
      r.vy = Number.isFinite(tr.vy) ? tr.vy : 0;
      r.tw = tr.w; r.th = tr.h;
    }
    const views = [];
    for (const [id, r] of this.records) {
      const age = Math.max(0, now - r.lastReceipt);
      const hold = Math.min(700, Math.max(320, r.cadence * 2.2));
      if (age > hold + 300) { this.expired.set(r.track, r.hitStamp); this.records.delete(id); continue; }
      // Compensate capture-to-result latency before easing a genuinely missing result.
      // Previously braking after120ms also braked fresh results arriving200–400ms late.
      const elapsed = Math.max(0, now - r.stamp) / 1000;
      const captureDelay = Math.min(0.5, Math.max(0, (r.lastReceipt - r.stamp) / 1000));
      const waitBudget = Math.max(0.10, Math.min(0.30, r.cadence * 1.5 / 1000));
      const linearUntil = captureDelay + waitBudget;
      const extra = Math.max(0, elapsed - linearUntil), damping = Math.exp(-extra / 0.08);
      const lead = elapsed <= linearUntil ? elapsed : linearUntil + 0.08 * (1 - damping);
      let tx = r.sx + r.vx * lead, ty = r.sy + r.vy * lead;
      const measuredX = Number.isFinite(r.track.mx) ? r.track.mx : r.sx;
      const measuredY = Number.isFinite(r.track.my) ? r.track.my : r.sy;
      const fallbackRadius = Math.min(0.12, Math.max(0.025, 2 * Math.max(r.tw, r.th)));
      const boundFallback = (x, y) => {
        let dx = x - measuredX, dy = y - measuredY;
        const distance = Math.hypot(dx, dy);
        if (distance > fallbackRadius) { dx *= fallbackRadius / distance; dy *= fallbackRadius / distance; }
        return [Math.max(0, Math.min(1, measuredX + dx)), Math.max(0, Math.min(1, measuredY + dy))];
      };
      [tx, ty] = boundFallback(tx, ty);
      const first = !!r.first;
      if (first) { r.x = tx; r.y = ty; r.first = false; }
      const a = 1 - Math.exp(-dt / 0.020); // short correction settling; velocity advances every frame
      let dx = first ? 0 : r.vx * damping * dt, dy = first ? 0 : r.vy * damping * dt;
      dx += (tx - r.x - dx) * a; dy += (ty - r.y - dy) * a;
      const distance = Math.hypot(dx, dy), maxStep = (Math.hypot(r.vx, r.vy) + MAX_SPEED_NORM) * dt;
      if (distance > maxStep && distance > 0) { dx *= maxStep / distance; dy *= maxStep / distance; }
      const pixel = follower ? follower.read(r.track.id, now) : null;
      // A current camera-frame match replaces extrapolation; it must not inherit smoothing delay.
      if (pixel) { r.x = pixel.x; r.y = pixel.y; }
      else {
        // A previously verified image position can already be well ahead of the
        // old detector box. Hold it on loss; never drag it back to that old box.
        if (r.hadPixel && r.hitStamp === r.pixelHitStamp) { /* hold until new evidence */ }
        else { [r.x, r.y] = boundFallback(r.x + dx, r.y + dy); }
      }
      if (pixel) { r.hadPixel = true; r.pixelHitStamp = r.hitStamp; }
      const sizeAlpha = 1 - Math.exp(-dt / 0.08);
      r.w += (r.tw - r.w) * sizeAlpha; r.h += (r.th - r.h) * sizeAlpha;
      const desiredAlpha = Math.max(0, Math.min(1, 1 - (age - hold) / 180));
      r.alpha += (desiredAlpha - r.alpha) * (1 - Math.exp(-dt / 0.05));
      // Only observed camera matches draw a path. Predictions must not invent a
      // long cross-screen trajectory; a discontinuity starts a new short path.
      if (!pixel) r.trail = [];
      const previous = r.trail[r.trail.length - 1];
      const separation = previous ? Math.hypot(r.x - previous[0], r.y - previous[1]) : 0;
      if (separation > Math.min(0.12, Math.max(0.03, 3 * Math.max(r.w, r.h)))) r.trail = [];
      if (pixel && (!previous || separation > 0.0005)) {
        r.trail.push([r.x, r.y]); if (r.trail.length > TRAIL_LEN) r.trail.shift();
      }
      views.push({ x: r.x, y: r.y, w: r.w, h: r.h, alpha: r.alpha, trail: r.trail,
                   visualId: r.visualId, track: r.track, solid: follower ? !!pixel : (liveIds.has(id) && r.track.misses === 0 && age < hold) });
    }
    return views;
  }
}




const isElong = d => (d.x2 - d.x1) / Math.max(1e-6, d.y2 - d.y1) > ELONG_ASPECT ||
                     (d.y2 - d.y1) / Math.max(1e-6, d.x2 - d.x1) > ELONG_ASPECT;

class Track {
  constructor(id, d, t) {
    this.id = id;
    this.x = (d.x1 + d.x2) / 2; this.y = (d.y1 + d.y2) / 2;
    this.vx = 0; this.vy = 0;
    this.px = [MEAS_STD * MEAS_STD, 0, 1]; this.py = [MEAS_STD * MEAS_STD, 0, 1]; // [pp, pv, vv]
    this.w = d.x2 - d.x1; this.h = d.y2 - d.y1;
    this.t = t; this.tHit = t;
    this.hits = 1; this.misses = 0;
    this.conf = d.conf; this.label = d.label || 'Ball';
    this.trail = [[this.x, this.y]];
    this.lastRaw = d.conf >= START_CONF;  // last hit had conf >= START_CONF
    this.sure = d.conf >= CONFIRM_CONF;   // one high-conf hit confirms at once
    this.qBoost = 1;                      // adaptive process-noise multiplier (maneuver)
    this.mx = this.x; this.my = this.y; this.mt = t;     // last measured point
    this.m2x = this.x; this.m2y = this.y; this.m2t = t;  // previous measured point
  }
  get confirmed() { return this.sure || this.hits >= CONFIRM_HITS; }
  predict(t) {
    const dt = Math.max(0, (t - this.t) / 1000);
    this.t = Math.max(this.t, t);
    if (dt === 0) return;
    const s = ACCEL_STD * (this.misses > 0 ? Math.max(this.qBoost, MISS_Q_BOOST) : this.qBoost);
    const q = s * s;
    for (const ax of ['x', 'y']) {
      const P = ax === 'x' ? this.px : this.py;
      const v = ax === 'x' ? 'vx' : 'vy';
      this[ax] += this[v] * dt;
      const a = P[0] + 2 * dt * P[1] + dt * dt * P[2] + q * dt * dt * dt * dt / 4;
      const b = P[1] + dt * P[2] + q * dt * dt * dt / 2;
      const c = P[2] + q * dt * dt;
      P[0] = a; P[1] = b; P[2] = c;
    }
  }
  kfUpdate(ax, z, R) {
    const P = ax === 'x' ? this.px : this.py;
    const v = ax === 'x' ? 'vx' : 'vy';
    const S = P[0] + R, k0 = P[0] / S, k1 = P[1] / S, r = z - this[ax];
    this[ax] += k0 * r; this[v] += k1 * r;
    const a = (1 - k0) * P[0], b = (1 - k0) * P[1], c = P[2] - k1 * P[1];
    P[0] = a; P[1] = b; P[2] = c;
  }
  hit(d, t) {
    const el = isElong(d);
    const R = MEAS_STD * MEAS_STD * (d.conf < START_CONF ? LOWCONF_R_BOOST : 1) * (el ? LOWCONF_R_BOOST : 1);
    const zx = (d.x1 + d.x2) / 2, zy = (d.y1 + d.y2) / 2;
    // normalized innovation squared (2 dof); large => impulsive maneuver (spike)
    const nis = (zx - this.x) ** 2 / (this.px[0] + R) + (zy - this.y) ** 2 / (this.py[0] + R);
    if (nis > MANEUVER_NIS) {
      const dt = Math.min(0.5, Math.max(0.03, (t - this.tHit) / 1000)), s2 = MANEUVER_VEL_STD * MANEUVER_VEL_STD;
      for (const P of [this.px, this.py]) { P[0] += s2 * dt * dt; P[1] += s2 * dt; P[2] += s2; }
      this.qBoost = Math.min(Q_BOOST_MAX, 1 + Math.sqrt(nis / MANEUVER_NIS));
    } else {
      this.qBoost = 1;
    }
    this.kfUpdate('x', zx, R);
    this.kfUpdate('y', zy, R);
    if (!el) {   // a blur streak says nothing reliable about ball size
      this.w += WH_ALPHA * ((d.x2 - d.x1) - this.w);
      this.h += WH_ALPHA * ((d.y2 - d.y1) - this.h);
    }
    this.m2x = this.mx; this.m2y = this.my; this.m2t = this.mt;
    this.mx = zx; this.my = zy; this.mt = t;
    this.hits++; this.misses = 0; this.tHit = t;
    this.conf = d.conf; this.lastRaw = d.conf >= START_CONF;
    if (d.conf >= CONFIRM_CONF) this.sure = true;
  }
  // continue this (lost) track with detection d: velocity from last measured point -> d
  revive(d, t) {
    const zx = (d.x1 + d.x2) / 2, zy = (d.y1 + d.y2) / 2;
    const dt = (t - this.mt) / 1000;
    if (dt > 0.02) {
      let vx = (zx - this.mx) / dt, vy = (zy - this.my) / dt;
      const sp = Math.hypot(vx, vy);
      if (sp > MAX_SPEED_NORM) { vx *= MAX_SPEED_NORM / sp; vy *= MAX_SPEED_NORM / sp; }
      this.vx = vx; this.vy = vy;
    }
    this.x = zx; this.y = zy; this.t = t;
    this.px = [MEAS_STD * MEAS_STD, 0, 1]; this.py = [MEAS_STD * MEAS_STD, 0, 1];
    if (!isElong(d)) { this.w = d.x2 - d.x1; this.h = d.y2 - d.y1; }
    this.m2x = this.mx; this.m2y = this.my; this.m2t = this.mt;
    this.mx = zx; this.my = zy; this.mt = t;
    this.hits++; this.misses = 0; this.tHit = t; this.qBoost = 1;
    this.conf = d.conf; this.lastRaw = d.conf >= START_CONF;
    this.trail.push([zx, zy]);
  }
  // circle around the predicted position
  gate(dt) {
    const speed = Math.hypot(this.vx, this.vy);
    return (GATE_BASE + GATE_SPEED_K * speed * dt + GATE_SIZE_K * Math.max(this.w, this.h)) *
           (1 + GATE_MISS_K * this.misses) * (this.hits < 2 ? GATE_NEW_MULT : 1);
  }
  // re-acquire circle around the LAST MEASURED position; grows with time since last hit
  reacqGate(t) {
    return GATE_BASE + GATE_SIZE_K * Math.max(this.w, this.h) + MAX_SPEED_NORM * Math.max(0, t - this.tHit) / 1000;
  }
  sizeOK(d) {
    const sd = Math.max(d.x2 - d.x1, d.y2 - d.y1, 1e-6), st = Math.max(this.w, this.h, 1e-6);
    return Math.max(sd / st, st / sd) <= (isElong(d) ? ELONG_SIZE_RATIO : SIZE_RATIO_MAX);
  }
}

function boxIoU(tr, d) {
  const ax1 = tr.x - tr.w / 2, ax2 = tr.x + tr.w / 2, ay1 = tr.y - tr.h / 2, ay2 = tr.y + tr.h / 2;
  const iw = Math.min(ax2, d.x2) - Math.max(ax1, d.x1), ih = Math.min(ay2, d.y2) - Math.max(ay1, d.y1);
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  return inter / (tr.w * tr.h + (d.x2 - d.x1) * (d.y2 - d.y1) - inter);
}

class Tracker {
  constructor() { this.display = new SmoothOverlay(); this.tracks = []; this.nextId = 1; this.rescued = 0; this.lastT = -Infinity; this.lost = []; this.events = []; }
  reset() { this.display.reset(); this.tracks = []; this.rescued = 0; this.lastT = -Infinity; this.lost = []; this.events = []; }
  // dets: [{x1,y1,x2,y2,conf,label}] normalized; tms: capture time in ms
  // this.events afterwards: [{type:'reacq', id}] for this update
  update(dets, tms) {
    const prevT = this.lastT;
    this.lastT = Math.max(this.lastT, tms);
    this.events = [];
    for (const tr of this.tracks) tr.predict(tms);
    const dtGate = isFinite(prevT) ? Math.max(0, (tms - prevT) / 1000) : 0;
    const pairs = [];
    this.tracks.forEach((tr, ti) => {
      const g = tr.gate(dtGate), g2 = tr.misses > 0 ? tr.reacqGate(tms) : -1;
      dets.forEach((d, di) => {
        if (d.conf < LOW_CONF || !tr.sizeOK(d)) return;
        const cx = (d.x1 + d.x2) / 2, cy = (d.y1 + d.y2) / 2;
        const dist = Math.hypot(cx - tr.x, cy - tr.y);
        if (dist <= g) { pairs.push([dist, ti, di]); return; }
        if (g2 > 0 && (d.conf >= START_CONF || isElong(d))) {
          const d2 = Math.hypot(cx - tr.mx, cy - tr.my);
          if (d2 <= g2) pairs.push([1 + d2, ti, di]);   // +1: predicted-gate matches win
        }
      });
    });
    pairs.sort((a, b) => a[0] - b[0]);
    const tUsed = new Set(), dUsed = new Set();
    for (const [, ti, di] of pairs) {
      if (tUsed.has(ti) || dUsed.has(di)) continue;
      tUsed.add(ti); dUsed.add(di);
      const tr = this.tracks[ti];
      if (tr.confirmed && tr.misses >= 2) this.events.push({ type: 'reacq', id: tr.id });
      tr.hit(dets[di], tms);
    }
    for (let ti = 0; ti < this.tracks.length; ti++) {
      const tr = this.tracks[ti];
      if (!tUsed.has(ti)) tr.misses++;
      tr.trail.push([tr.x, tr.y]);
      if (tr.trail.length > TRAIL_LEN) tr.trail.shift();
      // rescued: a shown (confirmed) track the raw detector at START_CONF would not have shown
      if (tr.confirmed && !(tUsed.has(ti) && tr.lastRaw)) this.rescued++;
    }
    // re-acquire a recently lost confirmed track (same ID) before births
    this.lost = this.lost.filter(l => tms - l.tLost <= REACQUIRE_MS);
    dets.forEach((d, di) => {
      if (dUsed.has(di) || d.conf < START_CONF || !this.lost.length) return;
      if (this.tracks.some(tr => boxIoU(tr, d) > 0.5)) return;
      const cx = (d.x1 + d.x2) / 2, cy = (d.y1 + d.y2) / 2;
      let best = -1, bd = Infinity;
      this.lost.forEach((l, li) => {
        const tr = l.tr, dist = Math.hypot(cx - tr.mx, cy - tr.my);
        const lim = MAX_SPEED_NORM * Math.max(0, tms - tr.tHit) / 1000 + GATE_BASE + GATE_SIZE_K * Math.max(tr.w, tr.h);
        if (dist <= lim && dist < bd) { bd = dist; best = li; }
      });
      if (best < 0) return;
      const tr = this.lost.splice(best, 1)[0].tr;
      tr.revive(d, tms);
      dUsed.add(di);
      this.tracks.push(tr);
      this.events.push({ type: 'reacq', id: tr.id });
    });
    dets.forEach((d, di) => {   // births: only confident, unmatched, not overlapping an existing track
      if (dUsed.has(di) || d.conf < START_CONF) return;
      if (this.tracks.some(tr => boxIoU(tr, d) > 0.5)) return;
      this.tracks.push(new Track(this.nextId++, d, tms));
    });
    this.tracks = this.tracks.filter(tr => {
      const m = OFFSCREEN_MARGIN;
      const inside = tr.x > -m && tr.x < 1 + m && tr.y > -m && tr.y < 1 + m;
      const dead = tr.misses > (tr.confirmed ? MAX_MISSES : TENT_MAX_MISSES) || tms - tr.tHit > MAX_COAST_MS;
      if (dead && tr.confirmed && inside) this.lost.push({ tr: tr, tLost: tms });
      return !dead && inside;
    });
  }
  displayFrames(now, follower = null) { return this.display.sample(this.tracks, now, follower); }
  // state of a track extrapolated to time `now` (ms)
  view(tr, now, horizon = EXTRAP_MAX_MS) {
    const e = Math.max(0, Math.min(now - tr.t, horizon)) / 1000;
    return { x: tr.x + tr.vx * e, y: tr.y + tr.vy * e,
             solid: tr.misses === 0 && (now - tr.tHit) <= SOLID_MS };
  }
}
// ==== TRACKER END ====

// ==== HOMOGRAPHY BEGIN ====
// Pure functions. Points are [x, y]; H is a row-major 9-array (h33 = 1).
function solveLinear(A, b) {          // Gaussian elimination, partial pivoting; null if singular
  const n = b.length;
  const M = A.map((r, i) => r.concat([b[i]]));
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) return null;
    if (p !== c) { const t = M[p]; M[p] = M[c]; M[c] = t; }
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      if (f === 0) continue;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = M[i][n];
    for (let k = i + 1; k < n; k++) sum -= M[i][k] * x[k];
    x[i] = sum / M[i][i];
  }
  return x;
}
function computeHomography(src4, dst4) {   // 4-point DLT; maps src -> dst; null if degenerate
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const x = src4[i][0], y = src4[i][1], u = dst4[i][0], v = dst4[i][1];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  const h = solveLinear(A, b);
  return h ? h.concat([1]) : null;
}
function applyH(H, x, y) {                 // perspective divide; null if at infinity
  const w = H[6] * x + H[7] * y + H[8];
  if (Math.abs(w) < 1e-12) return null;
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}
function det3(H) {
  return H[0] * (H[4] * H[8] - H[5] * H[7]) - H[1] * (H[3] * H[8] - H[5] * H[6]) + H[2] * (H[3] * H[7] - H[4] * H[6]);
}
function invertH(H) {                      // 3x3 inverse via adjugate; null if singular
  const d = det3(H);
  if (Math.abs(d) < 1e-12) return null;
  const [a, b, c, d2, e, f, g, h, i] = H;
  return [(e * i - f * h) / d, (c * h - b * i) / d, (b * f - c * e) / d,
          (f * g - d2 * i) / d, (a * i - c * g) / d, (c * d2 - a * f) / d,
          (d2 * h - e * g) / d, (b * g - a * h) / d, (a * e - b * d2) / d];
}
// corners in order near-left, near-right, far-right, far-left (normalized video coords, y down)
function quadError(p) {
  if (!p || p.length !== 4) return 'ต้องมี 4 มุม';
  let neg = 0, pos = 0;
  for (let i = 0; i < 4; i++) {
    const a = p[i], b = p[(i + 1) % 4], c = p[(i + 2) % 4];
    const cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (Math.abs(cr) < 1e-5) return 'มุมซ้อนกันหรืออยู่แนวเส้นตรงเดียวกัน กด Redo แล้วแตะใหม่';
    if (cr < 0) neg++; else pos++;
  }
  if (neg && pos) return 'สี่เหลี่ยมไม่นูน (มุมไขว้กัน) ตรวจลำดับมุมแล้วลองใหม่';
  if (pos === 4) return 'ลำดับมุมกลับด้าน ต้องเป็น ใกล้-ซ้าย, ใกล้-ขวา, ไกล-ขวา, ไกล-ซ้าย';
  return null;
}
// -> {ok, H (video->table m), Hinv (table m->video), msg}
function buildCalibration(corners) {
  const err = quadError(corners);
  if (err) return { ok: false, msg: err };
  const H = computeHomography(corners, [[0, 0], [TABLE_W, 0], [TABLE_W, TABLE_L], [0, TABLE_L]]);
  if (!H || !H.every(isFinite) || Math.abs(det3(H)) < 1e-9) return { ok: false, msg: 'คำนวณไม่ได้ (รูปเสื่อมสภาพ) แตะมุมใหม่' };
  const Hinv = invertH(H);
  if (!Hinv) return { ok: false, msg: 'คำนวณไม่ได้ (invert ไม่ได้) แตะมุมใหม่' };
  return { ok: true, H: H, Hinv: Hinv, msg: '' };
}
function inTable(X, Y, margin) {
  return X >= -margin && X <= TABLE_W + margin && Y >= -margin && Y <= TABLE_L + margin;
}
// ==== HOMOGRAPHY END ====

// Average overlay cadence cap; the browser/sensor may still deliver fewer frames.
class FramePacer {
  constructor(fps = 45) { this.interval = 1000 / fps; this.reset(); }
  reset() { this.next = null; this.last = null; }
  ready(now) {
    if (!Number.isFinite(now)) return false;
    if (this.last !== null && now < this.last) this.reset();
    this.last = now;
    if (this.next !== null && now + 0.01 < this.next) return false;
    if (this.next === null || now - this.next > 1000) this.next = now + this.interval;
    else this.next += (Math.floor(Math.max(0, now - this.next) / this.interval) + 1) * this.interval;
    return true;
  }
}

const displayPacer = new FramePacer(45);
let stream = null, running = false, inflight = 0;
let cameraGeneration = 0, cameraOpening = false, drawRequest = null;
let serverPumpTicket = null, serverLastFrame = null;
let dets = [];
let trackOn = true;
let lastAppliedT = -Infinity;
const tracker = new Tracker();
const pixelFollower = new AsyncPixelFollower();
let appearanceRejected = 0;
const NL = String.fromCharCode(10);
let dispFrames = 0, detCount = 0, lastInfer = 0, lastRtt = 0;
let tMark = performance.now();
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const octx = overlay.getContext('2d');
const hud = document.getElementById('hud');
const statusEl = document.getElementById('status');
const trackBtn = document.getElementById('trackBtn');
const cap = document.createElement('canvas');     // offscreen capture canvas
const cctx = cap.getContext('2d');

// ==== CAMERA ZOOM BEGIN ====
// Centered digital zoom: preview, inference, captures and calibration share this crop.
let cameraZoom = 1, cameraZoomVersion = 0;
function updateZoomUI() {
  document.getElementById('zoomRange').value = cameraZoom;
  document.getElementById('zoomRange').disabled = !running;
  document.getElementById('zoomValue').textContent = cameraZoom.toFixed(1) + '×';
  document.getElementById('zoomOut').disabled = !running || cameraZoom <= 1;
  document.getElementById('zoomIn').disabled = !running || cameraZoom >= 4;
  document.getElementById('zoomReset').disabled = !running || cameraZoom === 1;
  video.style.transform = 'scale(' + cameraZoom + ')';
  syncCameraUi();
}
function setCameraZoom(value) {
  const n = Number(value);
  if (!running || !Number.isFinite(n)) return;
  const z = Math.round(Math.max(1, Math.min(4, n)) * 100) / 100;
  if (z === cameraZoom) return;
  cameraZoom = z; cameraZoomVersion++;
  dets = []; tracker.reset(); resetPixelFollowing(); lastRoi = null; roiMiss = 0;
  lastAppliedT = performance.now();
  const hadCalibration = !!calib;
  resetCalib();
  updateZoomUI();
  statusEl.textContent = hadCalibration ? 'ปรับซูมแล้ว กรุณากำหนดมุมโต๊ะใหม่' : '';
}
// Canvas drawImage-compatible destination, with optional source region in zoomed-frame coordinates.
function drawCamera(ctx, ...args) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const sw = vw / cameraZoom, sh = vh / cameraZoom;
  const sx = (vw - sw) / 2, sy = (vh - sh) / 2;
  if (args.length === 8) {
    const [x, y, w, h, dx, dy, dw, dh] = args;
    ctx.drawImage(video, sx + x / cameraZoom, sy + y / cameraZoom,
                  w / cameraZoom, h / cameraZoom, dx, dy, dw, dh);
  } else {
    const [dx, dy, dw = vw, dh = vh] = args;
    ctx.drawImage(video, sx, sy, sw, sh, dx, dy, dw, dh);
  }
}
// ==== CAMERA ZOOM END ====

// ==== PIXEL-FRAME BEGIN ====
// Resample the exact inference capture (excluding letterbox) into the same small image used by frame following.
function pixelFrameFromRgba(data, sourceW, sourceH, time, region, aspectW, aspectH) {
  const scale = 480 / Math.max(aspectW, aspectH);
  const W = Math.max(16, Math.round(aspectW * scale)), H = Math.max(16, Math.round(aspectH * scale));
  const crop = region || { left: 0, top: 0, nw: sourceW, nh: sourceH };
  const gray = new Uint8Array(W * H);
  if (!region && sourceW === W && sourceH === H) {
    for (let i = 0, p = 0; i < gray.length; i++, p += 4) gray[i] = (77 * data[p] + 150 * data[p + 1] + 29 * data[p + 2]) >> 8;
    return { gray: gray, W: W, H: H, time: time };
  }
  for (let y = 0; y < H; y++) {
    const sy = Math.max(0, Math.min(sourceH - 1, Math.floor(crop.top + (y + 0.5) * crop.nh / H)));
    for (let x = 0; x < W; x++) {
      const sx = Math.max(0, Math.min(sourceW - 1, Math.floor(crop.left + (x + 0.5) * crop.nw / W)));
      const p = (sy * sourceW + sx) * 4;
      gray[y * W + x] = (77 * data[p] + 150 * data[p + 1] + 29 * data[p + 2]) >> 8;
    }
  }
  return { gray: gray, W: W, H: H, time: time };
}
// ==== PIXEL-FRAME END ====
// Check the *exact detector capture*. Never compare an old box with the current camera image.
function ballAppearance(d, rgba, W, H, region) {
  const rect = region || {left:0,top:0,nw:W,nh:H};
  const cx = rect.left + (d.x1+d.x2)*0.5*rect.nw;
  const cy = rect.top + (d.y1+d.y2)*0.5*rect.nh;
  const rx = (d.x2-d.x1)*0.5*rect.nw, ry = (d.y2-d.y1)*0.5*rect.nh;
  if (![cx,cy,rx,ry].every(Number.isFinite) || rx <= 0 || ry <= 0) return 'reject';
  if (Math.min(rx,ry)<2) return 'unknown'; // too few pixels to judge appearance
  const sample = (x,y) => {
    x=Math.round(x);y=Math.round(y);
    if(x<0||y<0||x>=W||y>=H)return null;
    const p=(y*W+x)*4;return [rgba[p],rgba[p+1],rgba[p+2]];
  };
  const isBallColor = c => {
    if (!c) return false;
    const [r,g,b]=c,hi=Math.max(r,g,b),lo=Math.min(r,g,b);
    return (lo>=100 && hi-lo<=Math.max(24,hi*.14)) ||
      (r>=100 && g>=45 && r>=g*1.08 && g>=b*1.25 && r-b>=40);
  };
  let colored=0,n=0;
  for(let y=-1;y<=1;y++)for(let x=-1;x<=1;x++){
    const c=sample(cx+x*.35*rx,cy+y*.35*ry);if(c){n++;if(isBallColor(c))colored++;}
  }
  // Strongly purple/blue/dark centers are not a white/orange ping-pong ball.
  if (!n || colored/n<.35) return 'reject';
  const center=sample(cx,cy);let edges=0,usable=0;
  for(let k=0;k<8;k++){
    const angle=k*Math.PI/4,dx=Math.cos(angle)*rx,dy=Math.sin(angle)*ry;
    let contrast=0,available=false;
    for(const radius of [.65,.85,1,1.2,1.4]){
      const outer=sample(cx+dx*radius,cy+dy*radius);
      if(!outer||!center)continue;available=true;
      contrast=Math.max(contrast,Math.abs(outer[0]-center[0]),Math.abs(outer[1]-center[1]),Math.abs(outer[2]-center[2]));
    }
    if(available){usable++;if(contrast>=18)edges++;}
  }
  if(usable<6)return 'unknown';
  if(edges<3)return 'reject'; // uniform background or UI panel, no enclosing edge
  return 'pass';
}
function filterBallDetections(list, rgba, W, H, region) {
  let rejected=0;
  const result=list.filter(d=>{const verdict=ballAppearance(d,rgba,W,H,region);if(verdict==='reject' || (verdict==='unknown' && d.conf < CONFIRM_CONF)){rejected++;return false;}return true;});
  appearanceRejected = rejected;
  return result;
}



let pixelLastVideoTime = null, pixelFollowFrames = 0, pixelFollowCount = 0;
function resetPixelFollowing() {
  pixelFollower.reset(); pixelLastVideoTime = null; pixelFollowFrames = 0; pixelFollowCount = 0;
}
function anchorPixelTracks(time, frame) {
  if (!frame || !trackOn) return;
  try { pixelFollower.anchor(tracker.tracks, time, frame); }
  catch (e) { resetPixelFollowing(); console.warn('Frame-follow anchor', e); }
}
function updatePixelFollowing(now) {
  if (!trackOn || !running || !video.videoWidth || pixelLastVideoTime === video.currentTime) return;
  pixelLastVideoTime = video.currentTime;
  try {
    const vw = video.videoWidth, vh = video.videoHeight, scale = 480 / Math.max(vw, vh);
    const W = Math.max(16, Math.round(vw * scale)), H = Math.max(16, Math.round(vh * scale));
    const ctx = getCtx('follow', W, H);
    drawCamera(ctx, 0, 0, W, H);
    const captureTime = performance.now();
    const frame = pixelFrameFromRgba(ctx.getImageData(0, 0, W, H).data, W, H, captureTime, null, vw, vh);
    pixelFollower.pushFrame(frame);
    pixelFollowCount = tracker.tracks.filter(tr => pixelFollower.read(tr.id, captureTime)).length;
    if (pixelFollowCount) pixelFollowFrames++;
  } catch (e) { resetPixelFollowing(); console.warn('Frame-follow capture', e); }
}

function toggleTrack() {
  trackOn = !trackOn;
  if (!trackOn) resetPixelFollowing();
  trackBtn.textContent = 'Track: ' + (trackOn ? 'on' : 'off');
}

// ==== INFER-PURE BEGIN (no DOM; pure functions, extracted by the parity test) ====
// Must match the server run_model(): letterbox (keep aspect, pad 114) -> RGB /255 NCHW,
// conf filter -> xywh->xyxy -> undo letterbox + clip -> NMS(IoU) -> normalize by image size.
const INF_S = 960, INF_PAD = 114, INF_IOU = 0.45;   // static fallback and maximum input-buffer side
const INF_LUT = (function () { const t = new Float32Array(256); for (let i = 0; i < 256; i++) t[i] = i * (1 / 255); return t; })();
const rne = x => { const f = Math.floor(x), d = x - f; return d < 0.5 ? f : (d > 0.5 ? f + 1 : (f % 2 === 0 ? f : f + 1)); };   // python round()
function lbRect(w, h, W, H) {           // letterbox w x h into W x H (centered, keep aspect)
  const scale = Math.min(W / w, H / h);
  const nw = rne(w * scale), nh = rne(h * scale);
  return { scale: scale, nw: nw, nh: nh, top: Math.floor((H - nh) / 2), left: Math.floor((W - nw) / 2) };
}
function lbParams(w, h, S) { return lbRect(w, h, S, S); }
function lbInverse(lb, x, y) { return [(x - lb.left) / lb.scale, (y - lb.top) / lb.scale]; }
// full-frame model input size for a w x h frame: long side L, short side rounded UP to a multiple of 32
function dynShape(w, h, L) {
  L = L || 640;
  const sh = Math.min(L, Math.ceil(Math.min(w, h) * L / Math.max(w, h) / 32 - 1e-9) * 32);
  return w >= h ? { W: L, H: sh } : { W: sh, H: L };
}
function decodeN(W, H) { let n = 0; for (const s of [8, 16, 32]) n += Math.ceil(H / s) * Math.ceil(W / s); return n; }
// letterboxed RGBA (W*H*4, pad already painted) -> float32 NCHW /255, one pass, reusing `out`
function rgbaToChw(d, W, H, out) {
  if (H === undefined) H = W;
  const n = W * H, g = n, b = 2 * n;
  for (let i = 0, k = 0; i < n; i++, k += 4) {
    out[i] = INF_LUT[d[k]]; out[g + i] = INF_LUT[d[k + 1]]; out[b + i] = INF_LUT[d[k + 2]];
  }
  return out;
}
// ROI: square of the full-resolution frame (px) around (cx, cy); returns {x0, y0, S} (ints), shifted to stay inside
function roiSquare(cx, cy, boxPx, speedPx, latMs, vw, vh) {
  const lim = Math.floor(Math.min(vw, vh));
  let S = Math.max(ROI_MIN_PX, ROI_BOX_K * boxPx) + speedPx * latMs / 1000 * 2;
  S = Math.min(lim, Math.round(S));
  const x0 = Math.round(Math.min(Math.max(cx - S / 2, 0), vw - S)), y0 = Math.round(Math.min(Math.max(cy - S / 2, 0), vh - S));
  return { x0: x0, y0: y0, S: S };
}
// detections normalized to the ROI square -> normalized to the full frame (same object format)
function roiMapDets(list, roi, vw, vh) {
  return list.map(d => ({ x1: (roi.x0 + d.x1 * roi.S) / vw, y1: (roi.y0 + d.y1 * roi.S) / vh,
                          x2: (roi.x0 + d.x2 * roi.S) / vw, y2: (roi.y0 + d.y2 * roi.S) / vh,
                          conf: d.conf, cls: d.cls, label: d.label }));
}
// best track for an ROI (confirmed, recently matched) -> roi or null
function pickRoi(trk, now, vw, vh) {
  let best = null;
  for (const tr of trk.tracks) if (tr.confirmed && tr.misses <= ROI_MAX_COAST && (!best || tr.tHit > best.tHit)) best = tr;
  if (!best) return null;
  const v = trk.view(best, now);
  return roiSquare(v.x * vw, v.y * vh, Math.max(best.w * vw, best.h * vh), Math.hypot(best.vx * vw, best.vy * vh), ROI_LAT_MS, vw, vh);
}
// out: model output (Float32Array), dims e.g. [1,5,8400] (or transposed [1,8400,5]);
// lb from lbParams(w,h,S); w,h = size of the image that was letterboxed.
function postprocess(out, dims, lb, w, h, conf, iou, labels) {
  const A = dims[dims.length - 2], B = dims[dims.length - 1];
  let C, N, sc, si;
  if (A <= B) { C = A; N = B; sc = N; si = 1; } else { C = B; N = A; sc = 1; si = C; }
  const nc = C - 4;
  const bx1 = [], by1 = [], bx2 = [], by2 = [], bs = [], bc = [];
  const clip = (v, hi) => (v < 0 ? 0 : (v > hi ? hi : v));
  for (let i = 0; i < N; i++) {
    let s, cid = 0;
    if (nc === 1) s = out[4 * sc + i * si];
    else { s = -Infinity; for (let c = 0; c < nc; c++) { const v = out[(4 + c) * sc + i * si]; if (v > s) { s = v; cid = c; } } }
    if (!(s >= conf)) continue;
    const cx = out[i * si], cy = out[sc + i * si], bw = out[2 * sc + i * si], bh = out[3 * sc + i * si];
    bx1.push(clip((cx - bw / 2 - lb.left) / lb.scale, w)); by1.push(clip((cy - bh / 2 - lb.top) / lb.scale, h));
    bx2.push(clip((cx + bw / 2 - lb.left) / lb.scale, w)); by2.push(clip((cy + bh / 2 - lb.top) / lb.scale, h));
    bs.push(s); bc.push(cid);
  }
  const order = bs.map((_, i) => i).sort((a, b) => (bs[b] - bs[a]) || (a - b));
  const keep = [];
  for (const i of order) {
    const ai = (bx2[i] - bx1[i]) * (by2[i] - by1[i]);
    let ok = true;
    for (const j of keep) {
      const iw = Math.min(bx2[i], bx2[j]) - Math.max(bx1[i], bx1[j]);
      const ih = Math.min(by2[i], by2[j]) - Math.max(by1[i], by1[j]);
      if (iw <= 0 || ih <= 0) continue;
      const inter = iw * ih, uni = ai + (bx2[j] - bx1[j]) * (by2[j] - by1[j]) - inter;
      if (uni > 0 && inter / uni > iou) { ok = false; break; }
    }
    if (ok) keep.push(i);
  }
  return keep.map(i => ({ x1: bx1[i] / w, y1: by1[i] / h, x2: bx2[i] / w, y2: by2[i] / h,
                          conf: bs[i], cls: bc[i], label: labels[bc[i]] || ('class_' + bc[i]) }));
}
// ==== INFER-PURE END ====

// ---- on-device inference: onnxruntime-web (WebGPU first, WASM fallback), same-origin files ----
const INFER_KEY = 'ttInfer_v1';
const ORT_LOCAL = '/static/ort/';
let ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
const MODEL_CDN = 'https://pingpong-model-delivery-20261022.sleekcandy4.chatgpt.site/pingpong-edb4204d4d8d-fp16.onnx';
const MODEL_SHA256 = 'edb4204d4d8dfd53a3d4af7da0384a37f04ed9e14ea66a6324f90793c4406c42';
let inferMode = 'device';                 // 'device' | 'server'
try { const v = localStorage.getItem(INFER_KEY); if (v === 'server' || v === 'device') inferMode = v; } catch (e) {}
let backend = '-', modelMsg = '', hudBody = hud.textContent;
let devBusy = false, devGen = 0, devFail = 0, skipGpu = false;
let ortP = null, ortSess = null, ortLib = null, ortNS = null, inBuf = null, inName = 'images', outName = 'output0';
let dynOK = true, roiMiss = 0, roiRuns = 0, roiDbg = false, lastRoi = null, modeText = '-';
const tensors = {}, ctxs = {};
let labelNames = ['Ball'], lbCtx = null;
const inferBtn = document.getElementById('inferBtn');

// ---- server queue state (server mode only; device mode never touches it) ----
// state: 'none' | 'joining' | 'waiting' | 'active'
const Q = { sid: '', state: 'none', pos: 0, active: 0, max: 0, waiting: 0, timer: 0, gen: 0, fails: 0 };
function qMakeSid() {
  let s = '';
  try { s = sessionStorage.getItem('ttSid_v1') || ''; } catch (e) {}
  if (!/^[A-Za-z0-9_-]{8,64}$/.test(s)) {
    try { s = crypto.randomUUID(); } catch (e) { s = ''; }
    if (!s) {
      const a = 'abcdefghijklmnopqrstuvwxyz0123456789';
      for (let i = 0; i < 24; i++) s += a[Math.floor(Math.random() * a.length)];
    }
    try { sessionStorage.setItem('ttSid_v1', s); } catch (e) {}
  }
  return s;
}
Q.sid = qMakeSid();
function hudHead() {
  if (inferMode === 'server') return 'backend     server' + (Q.max ? NL + 'server users ' + Q.active + '/' + Q.max + ' · waiting ' + Q.waiting : '');
  return modelMsg || ('backend     ' + backend + NL + 'build       cdn-delivery-8');
}
function updateHud() { hud.textContent = hudHead() + NL + hudBody; }

function getCtx(key, W, H) {              // reusable capture canvas per purpose ('full' | 'roi'), resized only on change
  const c = ctxs[key];
  if (c && c.w === W && c.h === H) return c.ctx;
  let cv = null, ctx = null;
  try { if (typeof OffscreenCanvas !== 'undefined') { cv = new OffscreenCanvas(W, H); ctx = cv.getContext('2d', { willReadFrequently: true }); } } catch (e) { ctx = null; }
  if (!ctx) { cv = document.createElement('canvas'); cv.width = W; cv.height = H; ctx = cv.getContext('2d', { willReadFrequently: true }); }
  ctxs[key] = { ctx: ctx, w: W, h: H };
  return ctx;
}
function getTensor(W, H) {
  const k = W + 'x' + H;
  return tensors[k] || (tensors[k] = new ortNS.Tensor('float32', inBuf.subarray(0, 3 * W * H), [1, 3, H, W]));
}
function toggleRoiDbg() { roiDbg = !roiDbg; document.getElementById('roiBtn').textContent = 'ROI box: ' + (roiDbg ? 'on' : 'off'); }

// ---- load progress (display only): written by loadDevice(), read by the start screen and the inline pill ----
// dl / rt / wu = download / runtime setup / warm-up step: 'pending' | 'active' | 'done'
const LD = { dl: 'pending', rt: 'pending', wu: 'pending', got: 0, total: 0, fast: false, ep: '', prec: '', warmMs: 0, failAt: 'dl' };
let ldToken = 0, ldAbort = null, ldLastN = 0, spOpen = true;
const pillEl = document.getElementById('loadpill'), pillTxt = document.getElementById('loadpillT');
function ldNotify(force) {
  const n = performance.now();
  if (!force && n - ldLastN < 90) return;
  ldLastN = n;
  splashRender(); inlineRender();
}
// abort a load that is still in flight (retry / mode switch / timeout); a finished session is kept
function ldCancel() {
  if (ortSess) return;
  ldToken++;
  if (ldAbort) { try { ldAbort.abort(); } catch (e) {} ldAbort = null; }
  ortP = null; modelMsg = ''; updateHud(); inlineRender();
}
// small inline loading state while the camera is running and the device model is still loading
function inlineRender() {
  const show = !spOpen && inferMode === 'device' && !ortSess && !!ortP;
  pillEl.hidden = !show;
  if (!show) return;
  let t;
  if (LD.dl === 'active') t = 'กำลังโหลดโมเดลบนเครื่อง' + (LD.total ? ' ' + Math.min(100, Math.floor(100 * LD.got / LD.total)) + '%' : '…');
  else if (LD.wu === 'active') t = 'กำลังทดสอบโมเดล…';
  else t = 'กำลังเตรียมโมเดล…';
  if (pillTxt.textContent !== t) pillTxt.textContent = t;
}

function ensureDevice() {
  if (ortSess) return Promise.resolve(true);
  if (!ortP) {
    const p = loadDevice().catch(e => { if (ortP === p) { ortP = null; inlineRender(); } throw e; });
    ortP = p; inlineRender();
  }
  return ortP;
}
async function readModelBytes(url, signal, check, progress, expectedHash) {
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (signal) { if (signal.aborted) abort(); else signal.addEventListener('abort', abort, {once:true}); }
  const timer = expectedHash ? setTimeout(abort, 45000) : null;
  try {
    const response = await fetch(url, {signal:controller.signal, credentials:'omit'});
    check();
    if (!response.ok) { const error = new Error('model ' + response.status); error.httpStatus = response.status; throw error; }
    const total = +response.headers.get('Content-Length') || 0;
    const chunks = []; let received = 0;
    progress(0,total);
    if (response.body && response.body.getReader) {
      const reader = response.body.getReader();
      for (;;) {
        const part = await reader.read(); check();
        if (part.done) break;
        chunks.push(part.value); received += part.value.length; progress(received,total);
      }
    } else { const bytes = new Uint8Array(await response.arrayBuffer()); check(); chunks.push(bytes); received=bytes.length; }
    const bytes = new Uint8Array(received); let offset=0;
    for (const chunk of chunks) { bytes.set(chunk,offset); offset += chunk.length; }
    if (expectedHash) {
      const digest = new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)); check();
      const actual = Array.from(digest,v=>v.toString(16).padStart(2,'0')).join('');
      if (actual !== expectedHash) throw new Error('model checksum mismatch');
    }
    return bytes;
  } finally { if (timer) clearTimeout(timer); if (signal) signal.removeEventListener('abort',abort); }
}
async function loadDevice() {
  try { return await loadDeviceAttempt(); }
  catch (error) {
    if (error && error.ldCancelled) throw error;
    if (ORT_BASE === ORT_LOCAL) throw error;
    console.warn('Runtime CDN unavailable; using local runtime',error);
    ORT_BASE=ORT_LOCAL; ortLib=null;
    return loadDeviceAttempt();
  }
}
async function loadDeviceAttempt() {
  const my = ++ldToken;
  const chk = () => { if (my !== ldToken) { const c = new Error('cancelled'); c.ldCancelled = true; throw c; } };   // superseded / aborted
  if (ldAbort) { try { ldAbort.abort(); } catch (e) {} }
  ldAbort = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const sig = ldAbort ? ldAbort.signal : undefined;
  Object.assign(LD, { dl: 'active', rt: 'pending', wu: 'pending', got: 0, total: 0, fast: false, ep: '', prec: '', warmMs: 0, failAt: 'dl' });
  modelMsg = 'กำลังโหลดโมเดล 0%'; updateHud(); ldNotify(true);
  if (!ortLib) ortLib = await import(ORT_BASE + 'ort.webgpu.min.mjs');
  chk();
  const ort = ortLib.InferenceSession ? ortLib : ortLib.default;
  ortNS = ort;
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2)) : 1;
  fetch('/health').then(r => r.json()).then(j => { if (j && j.names && j.names.length) labelNames = j.names; }).catch(() => {});
  const fetchModel = async prec => {
    Object.assign(LD, { dl:'active', got:0, total:0, fast:false, prec:prec }); ldNotify(true);
    const tDl=performance.now();
    const progress=(got,total)=>{ LD.got=got; LD.total=total;
      if(total){modelMsg='กำลังโหลดโมเดล '+Math.min(100,Math.floor(100*got/total))+'%';updateHud();} ldNotify(); };
    const legacy=prec==='f32'?'/model.onnx?prec=fp32':'/model.onnx';
    const dynamic='/model.onnx?variant=dyn&prec='+(prec==='f32'?'fp32':'fp16');
    let bytes;
    if(dynOK && prec==='f16') {
      try { bytes=await readModelBytes(MODEL_CDN,sig,chk,progress,MODEL_SHA256); }
      catch(error) { chk(); if(sig && sig.aborted) throw error; console.warn('Model CDN unavailable; using original server',error); }
    }
    if(!bytes) {
      try { bytes=await readModelBytes(dynOK?dynamic:legacy,sig,chk,progress); }
      catch(error) {
        chk(); if(!dynOK || error.httpStatus!==404) throw error;
        dynOK=false; bytes=await readModelBytes(legacy,sig,chk,progress);
      }
    }
    chk(); modelMsg='กำลังโหลดโมเดล 100%';updateHud();
    Object.assign(LD,{got:bytes.length,total:bytes.length,fast:performance.now()-tDl<350,dl:'done'});ldNotify(true);
    return bytes;
  };
  const models = {};
  const modelOf = async prec => models[prec] || (models[prec] = await fetchModel(prec));
  let hasF16 = false;
  if (navigator.gpu && !skipGpu) {
    try { const ad = await navigator.gpu.requestAdapter(); hasF16 = !!(ad && ad.features && ad.features.has('shader-f16')); } catch (e) {}
  }
  // attempts: webgpu with the f16 model when the adapter has shader-f16 (else f32), then webgpu f32, then wasm
  const attempts = [];
  if (navigator.gpu && !skipGpu) { if (hasF16) attempts.push(['webgpu', 'f16']); attempts.push(['webgpu', 'f32']); }
  attempts.push(['wasm', 'f16']);
  let lastErr = null;
  for (const [ep, prec] of attempts) {
    try {
      const usePrec = ep === 'wasm' && models.f32 ? 'f32' : prec;
      Object.assign(LD, { ep: ep, prec: usePrec, rt: 'pending', wu: 'pending' }); ldNotify(true);
      const bytes = await modelOf(usePrec);
      chk();
      Object.assign(LD, { dl: 'done', rt: 'active' }); ldNotify(true);
      modelMsg = 'กำลังเตรียมโมเดล (' + ep + ' ' + prec + ')…'; updateHud();
      const sess = await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: 'all' });
      chk();
      inName = sess.inputNames[0]; outName = sess.outputNames[0];
      if (!inBuf) inBuf = new Float32Array(3 * INF_S * INF_S);
      inBuf.fill(INF_PAD / 255);
      Object.assign(LD, { rt: 'done', wu: 'active' }); ldNotify(true);
      const t0 = performance.now();
      if (dynOK) { const sh = dynShape(video.videoWidth || (window.innerWidth > window.innerHeight ? 16 : 9), video.videoHeight || (window.innerWidth > window.innerHeight ? 9 : 16), INF_LONG); disposeModelOutputs(await sess.run({ [inName]: getTensor(sh.W, sh.H) })); if (ROI_ENABLED) disposeModelOutputs(await sess.run({ [inName]: getTensor(INF_ROI, INF_ROI) })); }   // warm up the selected full-frame shape
      else disposeModelOutputs(await sess.run({ [inName]: getTensor(INF_S, INF_S) }));
      chk();
      console.log('ORT ' + ep + ' ' + prec + ' warm-up ' + (performance.now() - t0).toFixed(0) + ' ms, threads ' + ort.env.wasm.numThreads);
      LD.warmMs = performance.now() - t0; LD.wu = 'done';
      ortSess = sess; backend = ep === 'wasm' ? 'wasm' : ep + ' ' + prec; modelMsg = ''; updateHud(); ldNotify(true);
      return true;
    } catch (e) {
      chk();
      console.error('ORT ' + ep + ' ' + prec + ' failed', e); lastErr = e;
      LD.failAt = LD.dl === 'active' ? 'dl' : (LD.wu === 'active' ? 'wu' : 'rt');
    }
  }
  throw lastErr || new Error('no backend');
}

function onDeviceFail(e) {
  if (inferMode !== 'device' || (e && e.ldCancelled)) return;
  console.error(e);
  inferMode = 'server'; backend = '-'; modelMsg = ''; devGen++;       // automatic fallback; not persisted
  inferBtn.textContent = 'Infer: server'; inlineRender();
  statusEl.textContent = 'รันโมเดลบนอุปกรณ์ไม่ได้ (WebGPU/WASM ใช้ไม่ได้) - สลับไปโหมดเซิร์ฟเวอร์อัตโนมัติ';
  updateHud();
  if (running) startLoop();
}

// Give the shared GPU/compositor room to present camera frames during long sessions.
function deviceRestMs(inferMs, totalMs) {
  const compute = Number.isFinite(inferMs) ? Math.max(0, inferMs) : 0;
  const total = Number.isFinite(totalMs) ? Math.max(0, totalMs) : compute;
  return Math.ceil(Math.max(24, Math.min(250, compute * 0.8), 125 - total));
}
function disposeModelOutputs(outputs) {
  for (const tensor of new Set(Object.values(outputs || {}))) {
    if (tensor && typeof tensor.dispose === 'function') tensor.dispose();
  }
}
async function deviceFrame(gen = devGen) {
  const zoomVersion = cameraZoomVersion;
  const t00 = performance.now();
  const vw = video.videoWidth, vh = video.videoHeight;
  roiRuns++;
  let roi = (ROI_ENABLED && dynOK && trackOn) ? pickRoi(tracker, t00, vw, vh) : null;
  if (roi && (roiMiss >= ROI_MISS_TO_FULL || roiRuns % ROI_FULL_EVERY === 0)) roi = null;
  let W, H, w, h, lb = null, ctx;
  if (roi) {
    W = H = INF_ROI; ctx = getCtx('roi', W, H);
    drawCamera(ctx, roi.x0, roi.y0, roi.S, roi.S, 0, 0, W, H);        // crop of the full-resolution frame
  } else {
    const s = SEND_SIDE / Math.max(vw, vh);
    w = Math.round(vw * s); h = Math.round(vh * s);                       // same size the server path uploads
    const sh = dynOK ? dynShape(w, h, INF_LONG) : { W: INF_S, H: INF_S };
    W = sh.W; H = sh.H;
    lb = lbRect(w, h, W, H);
    ctx = getCtx('full', W, H);
    ctx.fillStyle = 'rgb(' + INF_PAD + ',' + INF_PAD + ',' + INF_PAD + ')';
    ctx.fillRect(0, 0, W, H);
    drawCamera(ctx, lb.left, lb.top, lb.nw, lb.nh);
  }
  const tCap = performance.now();                              // capture timestamp, as in sendFrame
  const inputPixels = ctx.getImageData(0, 0, W, H).data;
  const pixelAnchor = trackOn && !roi ? pixelFrameFromRgba(inputPixels, W, H, tCap, lb, vw, vh) : null;
  rgbaToChw(inputPixels, W, H, inBuf);
  const t1 = performance.now();
  const res = await ortSess.run({ [inName]: getTensor(W, H) });
  const t2 = performance.now();
  try {
  const o = res[outName];
  let list;
  if (roi) {
    list = roiMapDets(postprocess(o.data, o.dims, { scale: W / roi.S, left: 0, top: 0 }, roi.S, roi.S, LOW_CONF, INF_IOU, labelNames), roi, vw, vh);
    roiMiss = list.length ? 0 : roiMiss + 1;
  } else {
    list = postprocess(o.data, o.dims, lb, w, h, LOW_CONF, INF_IOU, labelNames);
    roiMiss = 0;
  }
  lastRoi = roi; modeText = roi ? 'ROI ' + roi.S + 'px' : 'FULL ' + W + 'x' + H;
  lastInfer = t2 - t1;
  lastRtt = performance.now() - t00;
  detCount++;
  if (gen !== devGen || !running || zoomVersion !== cameraZoomVersion || inferMode !== 'device' || tCap <= lastAppliedT) return;
  lastAppliedT = tCap;
  dets = roi ? list : filterBallDetections(list, inputPixels, W, H, lb);
  tracker.update(dets, tCap);
  anchorPixelTracks(tCap, pixelAnchor);
  collectAfterUpdate();
  } finally { disposeModelOutputs(res); }
}

async function deviceStep(gen) {
  if (gen !== devGen || !running || inferMode !== 'device' || devBusy) return;
  devBusy = true;
  try {
    if (!ortSess) await ensureDevice();
    if (gen === devGen && running && inferMode === 'device' && video.videoWidth > 0) { await deviceFrame(gen); devFail = 0; }
  } catch (e) {
    if (!ortSess) onDeviceFail(e);
    else if (++devFail >= 3) {
      devFail = 0; console.error(e);
      if (backend.indexOf('webgpu') === 0) { skipGpu = true; ortSess = null; ortP = null; backend = '-'; }   // retry on WASM
      else onDeviceFail(e);
    }
  } finally { devBusy = false; const nextGen = devGen; setTimeout(() => { if (nextGen === devGen) deviceKick(); }, deviceRestMs(lastInfer, lastRtt)); }
}
// one inference in flight; the next one is scheduled when a new video frame is presented
function deviceKick() {
  if (!running || inferMode !== 'device') return;
  const gen = devGen, cb = () => { deviceStep(gen); };
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(cb); else requestAnimationFrame(cb);
}
function startLoop() { devGen++; cancelServerPump(); serverLastFrame = null; if (inferMode === 'device') deviceKick(); else pump(); }
function applyMode() {
  inferBtn.textContent = 'Infer: ' + inferMode;
  devGen++;
  if (inferMode === 'device') { qLeave(); ensureDevice().catch(onDeviceFail); }
  updateHud(); inlineRender();
  if (inferMode === 'server' && Q.state !== 'active') {      // back to server: must (re)join the queue -> splash shows the position
    if (running) stopCamera(true);
    spReopen('');
    return;
  }
  if (running) startLoop();
}
function toggleInfer() {
  inferMode = inferMode === 'device' ? 'server' : 'device';
  try { localStorage.setItem(INFER_KEY, inferMode); } catch (e) {}
  applyMode();
}

// ---- data collection (full-resolution frames for retraining) ----
let collectOn = false, saving = false, lastSaveT = -Infinity, savedCount = null;
const collectBtn = document.getElementById('collectBtn');
const fullCv = document.createElement('canvas');
function toggleCollect() {
  collectOn = !collectOn;
  collectBtn.textContent = 'Collect: ' + (collectOn ? 'on' : 'off');
  if (collectOn) refreshSaved();
}
async function refreshSaved() {
  try { savedCount = (await (await fetch('/captures/stats')).json()).count; } catch (e) {}
}
// box: {x,y,w,h} normalized center/size, or null (no pre-label)
async function saveFrame(reason, conf, box, force) {
  if (saving || video.videoWidth === 0) return;
  const now = performance.now();
  if (!force && now - lastSaveT < COLLECT_MIN_MS) return;
  saving = true; lastSaveT = now; syncCameraUi();
  try {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (fullCv.width !== vw || fullCv.height !== vh) { fullCv.width = vw; fullCv.height = vh; }
    drawCamera(fullCv.getContext('2d'), 0, 0, vw, vh);
    const blob = await new Promise(r => fullCv.toBlob(r, 'image/jpeg', COLLECT_JPEG_Q));
    if (!blob) throw new Error('เตรียมภาพไม่สำเร็จ ลองบันทึกอีกครั้ง');
    let q = '?reason=' + encodeURIComponent(reason);
    if (conf != null) q += '&conf=' + conf.toFixed(3);
    if (box) {
      const c = v => Math.min(1, Math.max(0, v)).toFixed(5);
      q += '&x1=' + c(box.x - box.w / 2) + '&y1=' + c(box.y - box.h / 2) +
           '&x2=' + c(box.x + box.w / 2) + '&y2=' + c(box.y + box.h / 2);
    }
    const resp = await fetch('/save_frame' + q, {
      method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blob });
    if (!resp.ok) throw new Error(resp.status === 413 ? 'ภาพมีขนาดใหญ่เกินไป ลองบันทึกอีกครั้ง' : 'บันทึกภาพไม่สำเร็จ ลองอีกครั้ง');
    savedCount = (await resp.json()).count;
    if (force) showToast('บันทึกภาพแล้ว', true);
  } catch (e) { console.error(e); if (force) showToast(e.message || 'บันทึกภาพไม่สำเร็จ'); }
  finally { saving = false; syncCameraUi(); }
}
function saveManual() { saveFrame('manual', null, null, true); }
function collectAfterUpdate() {
  if (!collectOn || !trackOn) return;
  const tnow = performance.now();
  const boxOf = tr => { const v = tracker.view(tr, tnow); return { x: v.x, y: v.y, w: tr.w, h: tr.h }; };
  const ev = tracker.events.find(e => e.type === 'reacq');
  if (ev) {
    const tr = tracker.tracks.find(t => t.id === ev.id);
    if (tr) return saveFrame('reacq', tr.conf, boxOf(tr), false);
  }
  const coast = tracker.tracks.find(t => t.confirmed && t.misses > 0);
  if (coast) return saveFrame('miss', coast.conf, boxOf(coast), false);
  const low = tracker.tracks.find(t => t.confirmed && t.misses === 0 && t.conf < START_CONF);
  if (low) return saveFrame('lowconf', low.conf, boxOf(low), false);
}

// ---- table mapping: calibration + ball contact point in table coordinates ----
// window.calibration = {corners, H, Hinv, tableW, tableL, netY} | null      (video-normalized <-> meters)
// window.tableCoords = latest {t (ms), x_cm, y_cm, in_table, track_id, matched} | null
let calib = null;                 // {corners, H, Hinv}
let gridOn = true;
let liveBalls = [];
let ballHudT = 0;
window.calibration = null;
window.tableCoords = null;
const ballHud = document.getElementById('ballhud');
const gridBtn = document.getElementById('gridBtn');
const clamp01 = v => Math.min(1, Math.max(0, v));

function setCalib(c) {
  calib = c;
  window.calibration = c ? { corners: c.corners, H: c.H, Hinv: c.Hinv, tableW: TABLE_W, tableL: TABLE_L, netY: NET_Y } : null;
  if (!c) window.tableCoords = null;
}
function loadCalib() {
  try {
    const raw = localStorage.getItem(CALIB_KEY);
    if (!raw) return;
    const c = JSON.parse(raw);
    if (!Array.isArray(c) || c.length !== 4 ||
        !c.every(p => Array.isArray(p) && p.length === 2 && isFinite(p[0]) && isFinite(p[1]))) return;
    const r = buildCalibration(c);
    if (r.ok) setCalib({ corners: c, H: r.H, Hinv: r.Hinv });
  } catch (e) {}
}
function saveCalib() {
  try {
    if (calib) localStorage.setItem(CALIB_KEY, JSON.stringify(calib.corners));
    else localStorage.removeItem(CALIB_KEY);
  } catch (e) {}
}
function toggleGrid() { gridOn = !gridOn; gridBtn.textContent = 'เส้นโต๊ะ: ' + (gridOn ? 'เปิด' : 'ปิด'); }
function resetCalib() { setCalib(null); liveBalls = []; saveCalib(); }

// -- calibration UI --
const calEl = document.getElementById('calib');
const calCv = document.getElementById('calCv');
const calCtx = calCv.getContext('2d');
const calMsg = document.getElementById('calMsg');
const calDoneBtn = document.getElementById('calDone');
const calAutoBtn = document.getElementById('calAuto');
const snap = document.createElement('canvas');          // frozen video frame
const CAL_PROMPTS = ['แตะมุมใกล้-ซ้าย', 'แตะมุมใกล้-ขวา', 'แตะมุมไกล-ขวา', 'แตะมุมไกล-ซ้าย'];
let calOpen = false, calCorners = [], calDrag = -1, calCur = null, calGrab = [0, 0], calFinger = [0, 0], calErr = '';
let calRequest = null;

function cancelTableDetection() {
  if (calRequest) calRequest.abort();
  calRequest = null;
  calAutoBtn.disabled = false;
  calAutoBtn.textContent = 'จับโต๊ะอัตโนมัติ';
}

async function calDetectTable() {
  if (!calOpen || calRequest || calDrag >= 0) return;
  const request = new AbortController();
  calRequest = request;
  calErr = '';
  calAutoBtn.disabled = true;
  calAutoBtn.textContent = 'กำลังตรวจโต๊ะ…';
  calRedraw();
  const timeout = setTimeout(() => request.abort(), 12000);
  try {
    const sample = document.createElement('canvas');
    const scale = Math.min(1, 800 / Math.max(snap.width, snap.height));
    sample.width = Math.max(1, Math.round(snap.width * scale));
    sample.height = Math.max(1, Math.round(snap.height * scale));
    sample.getContext('2d').drawImage(snap, 0, 0, sample.width, sample.height);
    const jpeg = await new Promise(resolve => sample.toBlob(resolve, 'image/jpeg', .84));
    if (!jpeg || request.signal.aborted) throw new Error('snapshot unavailable');
    const response = await fetch('/table/detect', {method:'POST', body:jpeg, signal:request.signal});
    if (!response.ok) throw new Error('table detection '+response.status);
    const {proposal} = await response.json();
    if (calRequest !== request || !calOpen) return;
    const corners = proposal?.corners;
    if (Array.isArray(corners) && corners.length === 4 && corners.every(p =>
        Array.isArray(p) && p.length === 2 && p.every(n => Number.isFinite(n) && n >= 0 && n <= 1)) && buildCalibration(corners).ok) {
      calCorners = corners;
    } else calErr = 'หามุมโต๊ะไม่ครบ ลองเปิดหน้าคาลิเบรตใหม่ให้เห็นทั้งโต๊ะ หรือแตะมุมเอง';
  } catch (error) {
    if (calRequest === request && calOpen) calErr = 'จับโต๊ะอัตโนมัติไม่สำเร็จ ลองใหม่หรือแตะมุมเอง';
  } finally {
    clearTimeout(timeout);
    if (calRequest === request) { cancelTableDetection(); calRedraw(); }
  }
}

function openCalib() {
  if (video.videoWidth === 0) { statusEl.textContent = 'เปิดกล้องก่อน (Start camera) แล้วกด Calibrate'; return; }
  cancelTableDetection();
  snap.width = video.videoWidth; snap.height = video.videoHeight;
  drawCamera(snap.getContext('2d'), 0, 0);
  calCorners = calib ? calib.corners.map(p => [p[0], p[1]]) : [];   // existing calibration = fine-tune
  calDrag = -1; calCur = null; calErr = ''; calOpen = true;
  calEl.style.display = 'block';
  calRedraw();
}
function closeCalib() { cancelTableDetection(); calOpen = false; calEl.style.display = 'none'; }
function calRedo() { cancelTableDetection(); calCorners = []; calDrag = -1; calCur = null; calErr = ''; calRedraw(); }
function calFinish() {
  if (calCorners.length < 4) return;
  const r = buildCalibration(calCorners);
  if (!r.ok) { calErr = 'Calibrate ไม่ผ่าน: ' + r.msg; calRedraw(); return; }
  setCalib({ corners: calCorners.map(p => [p[0], p[1]]), H: r.H, Hinv: r.Hinv });
  saveCalib();
  gridOn = true; gridBtn.textContent = 'Grid: on';
  closeCalib();
}
function calRedraw() {
  if (!calOpen) return;
  const dpr = window.devicePixelRatio || 1;
  const cw = calCv.clientWidth, ch = calCv.clientHeight;
  if (calCv.width !== Math.round(cw * dpr) || calCv.height !== Math.round(ch * dpr)) {
    calCv.width = Math.round(cw * dpr); calCv.height = Math.round(ch * dpr);
  }
  calCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  calCtx.fillStyle = '#000'; calCtx.fillRect(0, 0, cw, ch);
  const m = mapper(cw, ch);
  calCtx.drawImage(snap, m.ox, m.oy, m.dw, m.dh);
  const toS = p => [m.ox + p[0] * m.dw, m.oy + p[1] * m.dh];
  const pts = calCorners.slice();
  if (calCur) pts.push(calCur);
  if (pts.length > 1) {
    calCtx.strokeStyle = '#0ff'; calCtx.lineWidth = 2; calCtx.beginPath();
    pts.forEach((p, i) => { const s = toS(p); if (i) calCtx.lineTo(s[0], s[1]); else calCtx.moveTo(s[0], s[1]); });
    if (pts.length === 4) calCtx.closePath();
    calCtx.stroke();
  }
  calCtx.font = 'bold 16px sans-serif'; calCtx.textAlign = 'center'; calCtx.textBaseline = 'middle';
  pts.forEach((p, i) => {
    const s = toS(p);
    calCtx.beginPath(); calCtx.arc(s[0], s[1], HANDLE_R, 0, Math.PI * 2);
    calCtx.fillStyle = 'rgba(255,200,0,0.35)'; calCtx.fill();
    calCtx.strokeStyle = '#fc0'; calCtx.lineWidth = 2; calCtx.stroke();
    calCtx.fillStyle = '#fff'; calCtx.fillText(String(i + 1), s[0], s[1] + 1);
  });
  if (calDrag >= 0) {                                   // magnifier loupe above the finger
    const tgt = calDrag === 4 ? calCur : calCorners[calDrag];
    const R = LOUPE_R, vw = snap.width, vh = snap.height;
    const lx = Math.min(cw - R - 4, Math.max(R + 4, calFinger[0]));
    let ly = calFinger[1] - 110;
    if (ly < R + 4) ly = calFinger[1] + 110;
    const hs = R / ((m.dw / vw) * LOUPE_ZOOM);          // half-size of the source area, video px
    calCtx.save();
    calCtx.beginPath(); calCtx.arc(lx, ly, R, 0, Math.PI * 2); calCtx.clip();
    calCtx.fillStyle = '#000'; calCtx.fillRect(lx - R, ly - R, 2 * R, 2 * R);
    calCtx.drawImage(snap, tgt[0] * vw - hs, tgt[1] * vh - hs, 2 * hs, 2 * hs, lx - R, ly - R, 2 * R, 2 * R);
    calCtx.restore();
    calCtx.strokeStyle = '#fff'; calCtx.lineWidth = 3; calCtx.beginPath(); calCtx.arc(lx, ly, R, 0, Math.PI * 2); calCtx.stroke();
    calCtx.strokeStyle = '#f33'; calCtx.lineWidth = 1.5; calCtx.beginPath();
    calCtx.moveTo(lx - 14, ly); calCtx.lineTo(lx + 14, ly); calCtx.moveTo(lx, ly - 14); calCtx.lineTo(lx, ly + 14);
    calCtx.stroke();
  }
  if (calErr) { calMsg.textContent = calErr; calMsg.style.background = 'rgba(170,0,0,.85)'; }
  else {
    calMsg.style.background = 'rgba(0,0,0,.65)';
    calMsg.textContent = calCorners.length < 4 ? (calCorners.length + 1) + '/4  ' + CAL_PROMPTS[calCorners.length]
                                               : 'ตรวจจุด 1–4 ลากให้ตรงมุมโต๊ะ แล้วกดยืนยันมุม';
  }
  calDoneBtn.disabled = calCorners.length < 4;
  calDoneBtn.style.opacity = calCorners.length < 4 ? 0.4 : 1;
}
function calPos(e) { const r = calCv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
function calNorm(px, py, m) { return [clamp01((px - m.ox) / m.dw), clamp01((py - m.oy) / m.dh)]; }
calCv.addEventListener('pointerdown', e => {
  e.preventDefault();
  cancelTableDetection();
  calErr = '';
  const m = mapper(calCv.clientWidth, calCv.clientHeight), pos = calPos(e);
  calFinger = pos; calGrab = [0, 0];
  if (calCorners.length < 4) { calDrag = 4; calCur = calNorm(pos[0], pos[1], m); }
  else {
    let best = -1, bd = HANDLE_HIT * HANDLE_HIT;
    calCorners.forEach((c, i) => {
      const sx = m.ox + c[0] * m.dw, sy = m.oy + c[1] * m.dh;
      const d = (sx - pos[0]) ** 2 + (sy - pos[1]) ** 2;
      if (d < bd) { bd = d; best = i; calGrab = [sx - pos[0], sy - pos[1]]; }
    });
    calDrag = best;
  }
  if (calDrag >= 0) { try { calCv.setPointerCapture(e.pointerId); } catch (_) {} }
  calRedraw();
});
calCv.addEventListener('pointermove', e => {
  if (calDrag < 0) return;
  e.preventDefault();
  const m = mapper(calCv.clientWidth, calCv.clientHeight), pos = calPos(e);
  calFinger = pos;
  const n = calNorm(pos[0] + calGrab[0], pos[1] + calGrab[1], m);
  if (calDrag === 4) calCur = n; else calCorners[calDrag] = n;
  calRedraw();
});
const calUp = e => {
  if (calDrag === 4 && calCur) { calCorners.push(calCur); calCur = null; }
  calDrag = -1;
  calRedraw();
};
calCv.addEventListener('pointerup', calUp);
calCv.addEventListener('pointercancel', calUp);
window.addEventListener('resize', calRedraw);

// -- ball contact point (bottom-center of the box) in table coordinates --
function computeBalls(tnow) {
  const out = [];
  if (!calib) return out;
  if (trackOn) {
    for (const tr of tracker.tracks) {
      if (!tr.confirmed) continue;
      const v = tracker.view(tr, tnow), ny = v.y + tr.h / 2;
      const P = applyH(calib.H, v.x, ny);
      if (!P) continue;
      out.push({ nx: v.x, ny: ny, X: P[0], Y: P[1], matched: v.solid, id: tr.id, conf: tr.conf,
                 inside: inTable(P[0], P[1], IN_MARGIN) });
    }
  } else {
    for (const d of dets) {
      if (d.conf < START_CONF) continue;
      const nx = (d.x1 + d.x2) / 2, P = applyH(calib.H, nx, d.y2);
      if (!P) continue;
      out.push({ nx: nx, ny: d.y2, X: P[0], Y: P[1], matched: true, id: null, conf: d.conf,
                 inside: inTable(P[0], P[1], IN_MARGIN) });
    }
  }
  return out;
}
function publishTableCoords(tnow) {      // best ball -> window.tableCoords (for the next phase)
  if (!liveBalls.length) return null;
  const b = liveBalls.reduce((a, c) => (c.conf > a.conf ? c : a));
  window.tableCoords = { t: tnow, x_cm: b.X * 100, y_cm: b.Y * 100, in_table: b.inside, track_id: b.id, matched: b.matched };
  return b;
}
function drawGrid(m) {
  const P = (X, Y) => { const q = applyH(calib.Hinv, X, Y); return q ? [m.ox + q[0] * m.dw, m.oy + q[1] * m.dh] : null; };
  const line = (x1, y1, x2, y2) => {
    const a = P(x1, y1), b = P(x2, y2);
    if (!a || !b) return;
    octx.beginPath(); octx.moveTo(a[0], a[1]); octx.lineTo(b[0], b[1]); octx.stroke();
  };
  octx.save(); octx.setLineDash([]); octx.globalAlpha = 1;
  octx.strokeStyle = 'rgba(255,255,255,0.25)'; octx.lineWidth = 1;
  for (let x = GRID_STEP; x < TABLE_W - 1e-6; x += GRID_STEP) line(x, 0, x, TABLE_L);
  for (let y = GRID_STEP; y < TABLE_L - 1e-6; y += GRID_STEP) line(0, y, TABLE_W, y);
  octx.strokeStyle = 'rgba(255,255,0,0.85)'; octx.lineWidth = 2; line(TABLE_W / 2, 0, TABLE_W / 2, TABLE_L);
  octx.strokeStyle = 'rgba(255,70,70,0.95)'; octx.lineWidth = 3; line(0, NET_Y, TABLE_W, NET_Y);
  octx.strokeStyle = 'rgba(0,255,255,0.95)'; octx.lineWidth = 3;
  line(0, 0, TABLE_W, 0); line(TABLE_W, 0, TABLE_W, TABLE_L); line(TABLE_W, TABLE_L, 0, TABLE_L); line(0, TABLE_L, 0, 0);
  octx.restore();
}
function drawContacts(m) {
  octx.save(); octx.setLineDash([]); octx.lineWidth = 2;
  for (const b of liveBalls) {
    octx.strokeStyle = b.inside ? '#0f0' : '#f55';
    octx.globalAlpha = b.matched ? 1 : 0.6;
    octx.beginPath(); octx.arc(m.ox + b.nx * m.dw, m.oy + b.ny * m.dh, 6, 0, Math.PI * 2); octx.stroke();
  }
  octx.restore();
}
function ballHudText(b) {
  if (!calib) return '';
  if (!b) return 'ball -';
  return 'ball x ' + (b.X * 100).toFixed(0) + ' cm, y ' + (b.Y * 100).toFixed(0) + ' cm' + NL +
         (b.inside ? 'บนโต๊ะ (in)' : 'นอกโต๊ะ (out)') + (b.matched ? '' : ' (pred)');
}
// ==== START SCREEN BEGIN (start-up flow + UI only; inference / tracker / collect logic is untouched) ====
// Flow: page load -> splashInit() -> spPrepare() loads the device model (or checks /health) and reports progress here;
// the camera opens ONLY from the "เปิดกล้อง" button (spStart -> startCamera(), inside the tap gesture).
const byId = id => document.getElementById(id);
const SP = { gen: 0, ready: false, starting: false, err: null, cam: null, sv: 'pending', svInfo: null, abort: null,
             sig: '', stepT: 0, progT: 0, got: 0, errShown: false, live: '', notice: '' };
const STEP_IDX = { dl: 0, rt: 1, wu: 2 };
const spEl = {
  root: byId('splash'), go: byId('spGo'), sum: byId('spSum'), live: byId('spLive'), secure: byId('spSecure'),
  err: byId('spErr'), errT: byId('spErrT'), errR: byId('spErrR'), retry: byId('spRetry'), useSrv: byId('spUseSrv'),
  cam: byId('spCamErr'), camT: byId('spCamT'), camR: byId('spCamR'),
  modeBtns: [byId('spModeDev'), byId('spModeSrv')],
  li: [byId('spS1'), byId('spS2'), byId('spS3')], lbl: [byId('spL1'), byId('spL2'), byId('spL3')],
  sr: [byId('spR1'), byId('spR2'), byId('spR3')],
  bar: byId('spBar'), barI: byId('spBarI'), prog: byId('spProg'), mb: byId('spMB'), pct: byId('spPct'), note: byId('spNote'),
  t1: byId('spT1'), chipBox: byId('spChipBox'), chip: byId('spChip'), t2: byId('spT2'), t3: byId('spT3'),
  notice: byId('spNotice'), qHint: byId('spQHint'), qDev: byId('spQDev')
};
let spTimer = 0, spHideT = 0;
const setTxt = (el, s) => { if (el.textContent !== s) el.textContent = s; };
const fmtMB = b => (b / 1048576).toFixed(1);
const errText = e => (e && e.name === 'AbortError') ? 'หมดเวลา ไม่มีการตอบกลับ' : String((e && e.message) || e).slice(0, 240);
const baseName = p => { const s = String(p || ''); return s.slice(Math.max(s.lastIndexOf('/'), s.lastIndexOf(String.fromCharCode(92))) + 1); };
const camSecure = () => !!(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
function spBackendName(ep, prec) {
  if (ep === 'webgpu') return 'GPU ของเครื่อง (WebGPU ' + prec + ')';
  if (ep === 'wasm') return 'CPU ของเครื่อง (WASM) – ช้ากว่า';
  return '';
}

function splashRender() {
  if (!spOpen) return;
  const dev = inferMode === 'device';
  spEl.modeBtns.forEach(b => b.setAttribute('aria-pressed', b.getAttribute('data-mode') === inferMode ? 'true' : 'false'));
  const names = dev ? ['ดาวน์โหลดโมเดล', 'เตรียมการประมวลผล', 'ทดสอบโมเดล'] : ['เชื่อมต่อ server', 'พร้อมใช้งาน', 'เข้าคิวใช้งาน'];
  const qSt = Q.state === 'active' ? 'done' : (Q.state === 'waiting' || Q.state === 'joining') ? 'active' : 'pending';
  let st = dev ? [LD.dl, LD.rt, LD.wu] : [SP.sv, SP.sv === 'done' ? 'done' : 'pending', SP.sv === 'done' ? qSt : 'pending'];
  if (SP.ready) st = ['done', 'done', 'done'];
  if (SP.err) { for (let i = 0; i < 3; i++) { if (i === SP.err.step) st[i] = 'error'; else if (i > SP.err.step) st[i] = 'pending'; } }
  const stTxt = { pending: 'รอ', active: 'กำลังทำ', done: 'เสร็จแล้ว', error: 'ผิดพลาด' };
  for (let i = 0; i < 3; i++) {
    spEl.li[i].setAttribute('data-state', st[i]);
    setTxt(spEl.lbl[i], names[i]);
    setTxt(spEl.sr[i], stTxt[st[i]]);
  }
  // step 1: download progress (device) / connection (server)
  const showBar = dev && st[0] === 'active';
  spEl.bar.hidden = spEl.prog.hidden = spEl.note.hidden = !showBar;
  if (showBar) {
    const known = LD.total > 0, pct = known ? Math.min(100, Math.floor(100 * LD.got / LD.total)) : 0;
    spEl.bar.classList.toggle('ind', !known);
    spEl.barI.style.width = known ? pct + '%' : '';
    if (known) spEl.bar.setAttribute('aria-valuenow', String(pct)); else spEl.bar.removeAttribute('aria-valuenow');
    setTxt(spEl.mb, known ? fmtMB(LD.got) + ' / ' + fmtMB(LD.total) + ' MB' : (LD.got ? fmtMB(LD.got) + ' MB' : 'กำลังเชื่อมต่อ…'));
    setTxt(spEl.pct, known ? pct + '%' : '');
  }
  let t1 = '';
  if (dev) { if (st[0] === 'done') t1 = LD.fast ? 'พร้อมแล้ว' : 'ดาวน์โหลดแล้ว ' + fmtMB(LD.total) + ' MB'; }
  else if (st[0] === 'active') t1 = 'กำลังเชื่อมต่อ…';
  else if (st[0] === 'done' && SP.svInfo) t1 = [SP.svInfo.provider, SP.svInfo.model].filter(Boolean).join(' · ');
  if (st[0] === 'error') t1 = 'ไม่สำเร็จ';
  setTxt(spEl.t1, t1); spEl.t1.hidden = !t1;
  // step 2: backend chip (device) / ready (server)
  const chipOn = dev && (st[1] === 'active' || st[1] === 'done') && !!LD.ep;
  spEl.chipBox.hidden = !chipOn;
  if (chipOn) {
    setTxt(spEl.chip, spBackendName(LD.ep, LD.prec));
    spEl.chip.className = 'sp-chip' + (LD.ep === 'wasm' ? ' slow' : '') + (st[1] === 'active' ? ' pend' : '');
  }
  let t2 = '';
  if (dev) { if (st[1] === 'active') t2 = 'กำลังเริ่มระบบประมวลผล…'; else if (st[1] === 'error') t2 = 'ไม่สำเร็จ'; }
  else if (st[1] === 'done') t2 = 'ประมวลผลบน server';
  setTxt(spEl.t2, t2); spEl.t2.hidden = !t2;
  // step 3: warm-up
  let t3 = '';
  if (dev) { if (st[2] === 'active') t3 = 'กำลังทดสอบโมเดล…'; else if (st[2] === 'done') t3 = 'ใช้เวลา ' + Math.round(LD.warmMs) + ' ms'; else if (st[2] === 'error') t3 = 'ไม่สำเร็จ'; }
  else if (st[2] === 'done') t3 = 'ได้สิทธิ์ใช้งานแล้ว';
  else if (Q.state === 'waiting') t3 = 'ขณะนี้เต็ม (' + Q.active + '/' + Q.max + ') คุณอยู่คิวที่ ' + Q.pos;
  else if (Q.state === 'joining') t3 = 'กำลังเข้าคิว…';
  setTxt(spEl.t3, t3); spEl.t3.hidden = !t3;
  const qWait = !dev && Q.state === 'waiting' && !SP.err;
  spEl.qHint.hidden = spEl.qDev.hidden = !qWait;
  setTxt(spEl.notice, SP.notice || ''); spEl.notice.hidden = !SP.notice;
  // errors
  const e = SP.err;
  spEl.err.hidden = !e;
  if (e) {
    setTxt(spEl.errT, e.title); setTxt(spEl.errR, e.reason);
    spEl.useSrv.hidden = e.kind !== 'device';
    if (!SP.errShown) {
      SP.errShown = true;
      const a = document.activeElement;
      if (!a || a === document.body || spEl.root.contains(a)) { try { spEl.retry.focus({ preventScroll: true }); } catch (x) {} }
    }
  }
  spEl.cam.hidden = !SP.cam;
  if (SP.cam) { setTxt(spEl.camT, SP.cam.title); setTxt(spEl.camR, SP.cam.reason); }
  // primary button + summary
  let label, hint = '';
  if (SP.starting) label = 'กำลังเปิดกล้อง…';
  else if (SP.ready) {
    label = 'เปิดกล้อง';
    hint = dev ? spBackendName(LD.ep, LD.prec) + ' · ทดสอบ ' + Math.round(LD.warmMs) + ' ms'
               : 'server' + (SP.svInfo && SP.svInfo.provider ? ' · ' + SP.svInfo.provider : '');
  } else if (SP.err) { label = 'ยังเปิดกล้องไม่ได้'; hint = 'แก้ปัญหาด้านบนก่อน'; }
  else if (qWait) { label = 'รอคิว…'; hint = 'คุณอยู่คิวที่ ' + Q.pos; }
  else { label = 'กำลังเตรียม…'; hint = 'จะเปิดกล้องได้เมื่อเตรียมเสร็จ'; }
  spEl.go.disabled = !SP.ready || SP.starting;
  spEl.go.setAttribute('aria-busy', SP.ready || SP.err ? 'false' : 'true');
  setTxt(spEl.go, label); setTxt(spEl.sum, hint);
  const live = SP.ready ? 'พร้อมเปิดกล้อง' : (SP.err ? SP.err.title : (st.indexOf('active') >= 0 ? 'กำลัง' + names[st.indexOf('active')] : ''));
  if (live !== SP.live) { SP.live = live; spEl.live.textContent = live; }
}

// start (or restart) the preparation that matches the current Infer mode
function spPrepare() {
  const gen = ++SP.gen;
  SP.ready = false; SP.err = null; SP.cam = null; SP.errShown = false; SP.sig = ''; SP.sv = 'pending'; SP.svInfo = null;
  if (SP.abort) { try { SP.abort.abort(); } catch (e) {} SP.abort = null; }
  if (inferMode === 'server') spPrepServer(gen);
  else if (qLeave(), ortSess) SP.ready = true;                       // model already loaded: nothing to do
  else {
    ensureDevice().then(() => {
      if (gen !== SP.gen || SP.err) return;
      SP.ready = true; splashRender();
    }, e => {
      if (gen !== SP.gen || SP.err || (e && e.ldCancelled)) return;
      console.error(e);
      SP.err = { kind: 'device', step: STEP_IDX[LD.failAt] || 0, title: 'โหลดโมเดลบนเครื่องไม่สำเร็จ', reason: errText(e) };
      splashRender();
    });
  }
  splashRender();
}
async function spPrepServer(gen) {
  SP.sv = 'active';
  const ac = typeof AbortController !== 'undefined' ? new AbortController() : null;
  SP.abort = ac;
  const tm = setTimeout(() => { if (ac) ac.abort(); }, 15000);
  try {
    const r = await fetch('/health', ac ? { signal: ac.signal, cache: 'no-store' } : { cache: 'no-store' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const j = await r.json();
    if (gen !== SP.gen) return;
    SP.svInfo = { provider: String((j && j.provider) || ''), model: baseName(j && j.model) };
    { const n = Number(j && j.input_size && j.input_size[0]); serverSendSide = (n > 0 && isFinite(n)) ? Math.max(320, Math.min(1280, Math.round(n))) : SEND_SIDE; }
    SP.sv = 'done';
  } catch (e) {
    if (gen !== SP.gen) return;
    SP.sv = 'error';
    SP.err = { kind: 'server', step: 0, title: 'เชื่อมต่อ server ไม่ได้', reason: errText(e) };
  } finally { clearTimeout(tm); }
  splashRender();
  if (SP.sv === 'done' && gen === SP.gen) qJoin(gen);
}
// ---- queue client: join once, then poll /queue/status (2 s while waiting, 3 s as heartbeat when active) ----
function qReq(path, method, ms) {
  const ac = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const tm = setTimeout(() => { if (ac) ac.abort(); }, ms || 8000);
  const o = { method: method || 'GET', cache: 'no-store', headers: { 'X-Session-Id': Q.sid } };
  if (ac) o.signal = ac.signal;
  return fetch(path, o).finally(() => clearTimeout(tm));
}
function qApply(j) {
  const st = j && j.state;
  Q.state = st === 'active' || st === 'waiting' ? st : 'none';
  Q.pos = (j && j.position) | 0; Q.active = (j && j.active) | 0; Q.max = (j && j.max) | 0; Q.waiting = (j && j.waiting) | 0;
  updateHud();
}
async function qJoin(spGen) {
  const g = ++Q.gen; clearTimeout(Q.timer);
  Q.state = 'joining'; Q.fails = 0; splashRender();
  try {
    const r = await qReq('/queue/join', 'POST');
    if (g !== Q.gen) return;
    if (!r.ok) throw new Error('HTTP ' + r.status);
    qApply(await r.json());
    if (Q.state === 'none') throw new Error('queue refused');
  } catch (e) {
    if (g !== Q.gen) return;
    Q.state = 'none';
    SP.err = { kind: 'server', step: 2, title: 'เข้าคิวไม่สำเร็จ', reason: errText(e) };
    splashRender();
    return;
  }
  qGranted(spGen);
  qSchedule(g);
}
function qGranted(spGen) {                                   // splash: slot obtained -> camera button becomes enabled
  if (Q.state === 'active' && inferMode === 'server' && spOpen && !SP.err && SP.sv === 'done' && (spGen == null || spGen === SP.gen)) SP.ready = true;
  splashRender();
}
function qSchedule(g) {
  clearTimeout(Q.timer);
  Q.timer = setTimeout(() => qTick(g), Q.state === 'waiting' ? 2000 : 3000);
}
async function qTick(g) {
  if (g !== Q.gen || inferMode !== 'server') return;
  try {
    const r = await qReq('/queue/status?sid=' + encodeURIComponent(Q.sid), 'GET');
    if (g !== Q.gen) return;
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const was = Q.state;
    qApply(await r.json());
    Q.fails = 0;
    if (Q.state === 'none') {                                 // slot / queue entry expired server-side
      if (running) { qLost(); return; }
      if (spOpen) { SP.ready = false; qJoin(SP.gen); return; }   // quietly re-join from the splash
      return;
    }
    if (was === 'waiting' && Q.state === 'active') qGranted();
    else splashRender();
  } catch (e) {
    if (g !== Q.gen) return;
    if (++Q.fails >= 5 && Q.state === 'waiting' && spOpen && !SP.err) {
      SP.err = { kind: 'server', step: 2, title: 'เชื่อมต่อ server ไม่ได้', reason: errText(e) };
      splashRender();
      return;
    }
  }
  qSchedule(g);
}
function qLeave() {                                           // release slot / queue entry (idempotent, fire-and-forget)
  Q.gen++; clearTimeout(Q.timer);
  if (Q.state === 'none') return;
  Q.state = 'none'; Q.pos = 0;
  try { fetch('/queue/leave', { method: 'POST', headers: { 'X-Session-Id': Q.sid }, keepalive: true }).catch(() => {}); } catch (e) {}
  updateHud();
}
function qBeacon() {
  if (inferMode !== 'server' || Q.state === 'none') return;
  try { navigator.sendBeacon('/queue/leave?sid=' + encodeURIComponent(Q.sid)); } catch (e) {}
}
// slot lost while streaming (HTTP 429 from /detect, or status says none): close camera, go back to the queue
function qLost() {
  if (!running || inferMode !== 'server') return;
  Q.gen++; clearTimeout(Q.timer); Q.state = 'none';
  stopCamera(true);
  spReopen('หมดเวลาการใช้งาน กำลังเข้าคิวใหม่');
}
function spReopen(msg) {                                      // show the splash again (server mode: health -> join -> poll)
  SP.starting = false; SP.notice = msg || '';
  spOpen = true; clearTimeout(spHideT);
  ['ctl', 'status'].forEach(id => byId(id).setAttribute('inert', ''));
  spEl.root.hidden = false; void spEl.root.offsetWidth; spEl.root.classList.remove('hide');
  clearInterval(spTimer); spTimer = setInterval(spWatch, 1000);
  inlineRender();
  spPrepare();
}
function spQueueDevice() {                                    // "process on this device instead": same as the segmented control
  spSetMode('device');
}
window.addEventListener('pagehide', qBeacon);
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden') qBeacon(); });
// watchdog: never leave the user on a spinner (download: 60 s without new bytes or 180 s total; other steps: 60 s)
function spWatch() {
  if (!spOpen || SP.err || SP.ready || SP.starting) return;
  const now = performance.now(), dev = inferMode === 'device';
  const sig = SP.gen + inferMode + (dev ? LD.dl + LD.rt + LD.wu + LD.ep + LD.prec : SP.sv);
  if (sig !== SP.sig) { SP.sig = sig; SP.stepT = SP.progT = now; SP.got = LD.got; }
  else if (dev && LD.dl === 'active' && LD.got !== SP.got) { SP.got = LD.got; SP.progT = now; }
  const idle = now - SP.stepT;
  let step = -1;
  if (dev) {
    if (LD.dl === 'active') { if (now - SP.progT > 60000 || idle > 180000) step = 0; }
    else if (LD.rt === 'active') { if (idle > 60000) step = 1; }
    else if (LD.wu === 'active') { if (idle > 60000) step = 2; }
  } else if (SP.sv === 'active' && idle > 60000) step = 0;
  if (step < 0) return;
  if (dev) {
    const nm = ['ดาวน์โหลดโมเดล', 'เตรียมการประมวลผล', 'ทดสอบโมเดล'];
    if (step >= 1 && LD.ep === 'webgpu') skipGpu = true;       // GPU path stalled: the retry goes straight to WASM
    ldCancel();
    SP.err = { kind: 'device', step: step, title: 'ใช้เวลานานเกินไป', reason: 'ขั้นตอน ' + nm[step] + ' ค้างเกินกำหนด (60 วินาที)' };
  } else {
    if (SP.abort) { try { SP.abort.abort(); } catch (e) {} }
    SP.err = { kind: 'server', step: 0, title: 'เชื่อมต่อ server ไม่ได้', reason: 'server ไม่ตอบสนองเกิน 60 วินาที' };
  }
  splashRender();
}
function spRetry() { ldCancel(); spPrepare(); }
function spUseServer() {                                    // fallback after a device failure: not persisted (like onDeviceFail)
  ldCancel();
  inferMode = 'server'; inferBtn.textContent = 'Infer: server'; devGen++; updateHud();
  spPrepare();
}
function spSetMode(m) {                                     // segmented control: same key / variable as toggleInfer()
  if ((m !== 'device' && m !== 'server') || m === inferMode || SP.starting) return;
  inferMode = m; SP.notice = '';
  try { localStorage.setItem(INFER_KEY, m); } catch (e) {}
  inferBtn.textContent = 'Infer: ' + m;
  devGen++;
  if (m === 'device') qLeave();
  if (m === 'server') ldCancel();                           // do not keep downloading a model nobody will use
  updateHud();
  spPrepare();
}
async function spStart() {                                  // runs inside the tap: startCamera() reaches getUserMedia synchronously
  if (!SP.ready || SP.starting) return;
  SP.starting = true; SP.cam = null; splashRender();
  const secure = camSecure();
  try { await startCamera(); } catch (e) { console.error(e); }
  SP.starting = false;
  if (running) { spClose(); return; }
  SP.cam = secure ? { title: 'เปิดกล้องไม่สำเร็จ', reason: (statusEl.textContent || '') + ' – ตรวจสอบว่าอนุญาตให้ใช้กล้องแล้วลองอีกครั้ง' }
                  : { title: 'เปิดกล้องไม่ได้', reason: 'หน้านี้ต้องเปิดผ่าน HTTPS (ดูคำเตือนด้านบน)' };
  splashRender();
}
function spClose() {
  spOpen = false; clearInterval(spTimer); SP.notice = '';
  ['ctl', 'status'].forEach(id => byId(id).removeAttribute('inert'));
  spEl.root.classList.add('hide');
  const done = () => { spEl.root.hidden = true; };
  if (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) done(); else spHideT = setTimeout(done, 400);
  inlineRender();
}
function splashInit() {
  ['ctl', 'status'].forEach(id => byId(id).setAttribute('inert', ''));
  if (!camSecure()) {                                       // secure-context warning, same URL as startCamera() builds
    const url = 'https://' + location.hostname + ':5443/', a = document.createElement('a');
    a.href = url; a.textContent = url;
    spEl.secure.appendChild(document.createTextNode('เปิดกล้องไม่ได้: หน้านี้ต้องใช้ HTTPS เปิดที่ '));
    spEl.secure.appendChild(a);
    spEl.secure.appendChild(document.createTextNode(' (ยอมรับ/เชื่อถือใบรับรอง ติดตั้งได้จาก /cert)'));
    spEl.secure.hidden = false;
  }
  spEl.go.addEventListener('click', spStart);
  spEl.retry.addEventListener('click', spRetry);
  spEl.useSrv.addEventListener('click', spUseServer);
  spEl.qDev.addEventListener('click', spQueueDevice);
  spEl.modeBtns.forEach(b => b.addEventListener('click', () => spSetMode(b.getAttribute('data-mode'))));
  spTimer = setInterval(spWatch, 1000);
  spPrepare();
}
// ==== START SCREEN END ====

let toastTimeout = null;
function showToast(message, withGallery = false) {
  const toast = document.getElementById('toast');
  toast.replaceChildren(document.createTextNode(message));
  if (withGallery) { const link=document.createElement('a');link.href='/gallery';link.textContent='ดูภาพ →';link.onclick=()=>stopCamera();toast.append(link); }
  toast.hidden=false;clearTimeout(toastTimeout);toastTimeout=setTimeout(()=>{toast.hidden=true;},3500);
}
function syncCameraUi() {
  document.body.classList.toggle('camera-running',running);
  document.getElementById('cameraActionLabel').textContent = cameraOpening ? 'กำลังเปิด…' : running ? 'หยุดกล้อง' : 'เปิดกล้อง';
  document.getElementById('cameraAction').disabled = cameraOpening;
  document.getElementById('saveAction').disabled = !running || saving;
  document.getElementById('saveActionLabel').textContent = saving ? 'กำลังบันทึก…' : 'บันทึกภาพ';
  document.getElementById('sessionLabel').textContent = running ? 'กล้องกำลังทำงาน' : 'พร้อมเปิดกล้อง';
}
function toggleCameraAction() { if(running)stopCamera();else startCamera(); }
function toggleTools(force) {
  const panel=document.getElementById('toolsPanel'),open=typeof force==='boolean'?force:panel.hidden;
  panel.hidden=!open;document.getElementById('toolsAction').setAttribute('aria-expanded',String(open));
}
document.addEventListener('keydown',e=>{if(e.key==='Escape')toggleTools(false);});

syncCameraUi();
loadCalib();
inferBtn.textContent = 'Infer: ' + inferMode;     // restore Infer mode label; the start screen preloads the model / checks the server
updateHud();
splashInit();

async function startCamera() {
  if (cameraOpening) return; // coalesce repeated taps while permission/play is pending
  if (inferMode === 'server' && Q.state !== 'active') { spReopen(''); return; }   // Start pressed after Stop/expiry: queue applies again
  let openedStream = null, generation = null;
  try {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      const httpsUrl = "https://" + location.hostname + ":5443/";
      statusEl.innerHTML = 'Camera unavailable: this page needs HTTPS. Open <a href="' +
        httpsUrl + '">' + httpsUrl + '</a> (accept/trust the certificate; install from /cert if needed).';
      return;
    }
    if (stream) stopCamera(true);
    generation = ++cameraGeneration;
    cameraOpening = true;
    syncCameraUi();
    openedStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 },
               frameRate: { ideal: 30 } },
      audio: false
    });
    if (generation !== cameraGeneration) {
      openedStream.getTracks().forEach(t => t.stop());
      return;
    }
    stream = openedStream;
    video.srcObject = openedStream;
    await video.play();
    if (generation !== cameraGeneration) {
      openedStream.getTracks().forEach(t => t.stop());
      return;
    }
    statusEl.textContent = "";
    running = true;
    updateZoomUI();
    lastAppliedT = -Infinity;
    displayPacer.reset();
    drawRequest = requestAnimationFrame(now => drawLoop(now, generation));
    startLoop();
  } catch (err) {
    if (openedStream) openedStream.getTracks().forEach(t => t.stop());
    if (generation !== null && generation !== cameraGeneration) return;
    if (stream === openedStream) { stream = null; video.srcObject = null; }
    statusEl.textContent = "Camera error: " + err.name + " - " + err.message;
    console.error(err);
  } finally {
    if (generation === cameraGeneration) { cameraOpening = false; syncCameraUi(); }
  }
}

function stopCamera(keepSlot) {
  cameraGeneration++;
  cameraOpening = false;
  displayPacer.reset();
  if (drawRequest !== null) { cancelAnimationFrame(drawRequest); drawRequest = null; }
  running = false;
  updateZoomUI();
  cancelServerPump();
  if (!keepSlot && inferMode === 'server' && !spOpen) qLeave();     // user pressed Stop: free the slot for the next in line
  devGen++;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.srcObject = null;
  dets = [];
  tracker.reset(); resetPixelFollowing();
  octx.clearRect(0, 0, overlay.width, overlay.height);
  statusEl.textContent = "หยุดกล้องแล้ว";
}

// Keep the GPU busy with distinct presented camera frames, not duplicate captures.
function cancelServerPump() {
  const ticket = serverPumpTicket;
  serverPumpTicket = null;
  if (!ticket) return;
  if (ticket.kind === 'video') video.cancelVideoFrameCallback(ticket.id);
  else clearTimeout(ticket.id);
}
function pump() {
  if (!running || inferMode !== 'server' || serverPumpTicket) return;
  const ticket = { gen: devGen, kind: 'timer', id: null };
  serverPumpTicket = ticket;
  const capture = (now, metadata) => {
    if (serverPumpTicket !== ticket) return;
    serverPumpTicket = null;
    if (ticket.gen !== devGen || !running || inferMode !== 'server') return;
    let stamp;
    if (metadata && Number.isFinite(metadata.presentedFrames)) stamp = 'frame:' + metadata.presentedFrames;
    else {
      const quality = video.getVideoPlaybackQuality ? video.getVideoPlaybackQuality() : null;
      stamp = quality && quality.totalVideoFrames > 0 ? 'frame:' + quality.totalVideoFrames : 'time:' + video.currentTime;
    }
    if (inflight < MAX_INFLIGHT && video.videoWidth > 0 && stamp !== serverLastFrame) {
      serverLastFrame = stamp;
      inflight++;
      sendFrame(ticket.gen).finally(() => { inflight--; pump(); });
    }
    pump();
  };
  if (video.requestVideoFrameCallback && video.cancelVideoFrameCallback) {
    ticket.kind = 'video';
    ticket.id = video.requestVideoFrameCallback(capture);
  } else ticket.id = setTimeout(capture, 34);
}

async function sendFrame(gen = devGen) {
  const zoomVersion = cameraZoomVersion;
  try {
    const vw = video.videoWidth, vh = video.videoHeight;
    const s = serverSendSide / Math.max(vw, vh);
    const w = Math.round(vw * s), h = Math.round(vh * s);
    if (cap.width !== w || cap.height !== h) { cap.width = w; cap.height = h; }
    drawCamera(cctx, 0, 0, w, h);
    const tCap = performance.now();                  // capture timestamp for this request
    const capturedPixels = cctx.getImageData(0, 0, w, h).data;
    const pixelAnchor = trackOn ? pixelFrameFromRgba(capturedPixels, w, h, tCap, null, vw, vh) : null;
    const blob = await new Promise(r => cap.toBlob(r, 'image/jpeg', JPEG_Q));
    if (!blob || zoomVersion !== cameraZoomVersion || gen !== devGen || !running || inferMode !== 'server') return;
    const t0 = performance.now();
    const resp = await fetch('/detect?conf=' + LOW_CONF, {
      method: 'POST', headers: { 'Content-Type': 'image/jpeg', 'X-Session-Id': Q.sid }, body: blob
    });
    if (zoomVersion !== cameraZoomVersion || gen !== devGen || !running || inferMode !== 'server') return;
    if (resp.status === 429) { qLost(); return; }       // slot lost (stall / expiry): back to the queue, no retry loop here
    if (!resp.ok) throw new Error('Server error ' + resp.status);
    const j = await resp.json();
    if (zoomVersion !== cameraZoomVersion || gen !== devGen || !running || inferMode !== 'server' || tCap <= lastAppliedT) return;   // stale response / mode switched
    lastRtt = performance.now() - t0;
    lastInfer = j.inference_ms;
    detCount++;
    lastAppliedT = tCap;
    dets = filterBallDetections(j.detections || [], capturedPixels, w, h, null);
    tracker.update(dets, tCap);
    anchorPixelTracks(tCap, pixelAnchor);
    collectAfterUpdate();                            // fire-and-forget, never blocks detection
  } catch (e) {
    if (zoomVersion !== cameraZoomVersion || gen !== devGen || !running || inferMode !== 'server') return;
    console.error(e);
    statusEl.textContent = 'Error: ' + String(e);
    await new Promise(r => setTimeout(r, 500));
  }
}

// Map normalized video coords (0..1 of the frame) to canvas CSS pixels,
// replicating object-fit: cover/contain.
function mapper(cw, ch) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const k = FIT === 'cover' ? Math.max(cw / vw, ch / vh) : Math.min(cw / vw, ch / vh);
  const dw = vw * k, dh = vh * k;
  return { dw: dw, dh: dh, ox: (cw - dw) / 2, oy: (ch - dh) / 2 };
}

function drawLoop(now, generation = cameraGeneration) {
  if (!running || generation !== cameraGeneration) return;
  drawRequest = null;
  if (!displayPacer.ready(now)) { drawRequest = requestAnimationFrame(next => drawLoop(next, generation)); return; }
  const dpr = window.devicePixelRatio || 1;
  const cw = overlay.clientWidth, ch = overlay.clientHeight;
  if (overlay.width !== Math.round(cw * dpr) || overlay.height !== Math.round(ch * dpr)) {
    overlay.width = Math.round(cw * dpr); overlay.height = Math.round(ch * dpr);
  }
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.clearRect(0, 0, cw, ch);
  if (video.videoWidth > 0) {
    updatePixelFollowing(now);
    const m = mapper(cw, ch);
    if (calib && gridOn) drawGrid(m);
    octx.lineWidth = 3; octx.strokeStyle = '#0f0'; octx.fillStyle = '#0f0';
    octx.font = '16px sans-serif'; octx.textBaseline = 'bottom';
    if (roiDbg && lastRoi && inferMode === 'device') {
      octx.save(); octx.strokeStyle = '#ff0'; octx.lineWidth = 2; octx.setLineDash([6, 4]);
      octx.strokeRect(m.ox + lastRoi.x0 / video.videoWidth * m.dw, m.oy + lastRoi.y0 / video.videoHeight * m.dh,
                      lastRoi.S / video.videoWidth * m.dw, lastRoi.S / video.videoHeight * m.dh);
      octx.restore();
    }
    if (!trackOn) {
      for (const d of dets) {
        if (d.conf < START_CONF) continue;           // raw detector view
        const x = m.ox + d.x1 * m.dw, y = m.oy + d.y1 * m.dh;
        octx.strokeRect(x, y, (d.x2 - d.x1) * m.dw, (d.y2 - d.y1) * m.dh);
        octx.fillText(d.label + ' ' + d.conf.toFixed(2), x + 2, Math.max(16, y - 2));
      }
    } else {
      const tnow = performance.now();
      for (const v of tracker.displayFrames(tnow, pixelFollower)) {
        const tr = v.track;
        // fading trail
        octx.setLineDash([]);
        octx.lineWidth = 2;
        for (let i = 1; i < v.trail.length; i++) {
          octx.globalAlpha = v.alpha * 0.6 * i / v.trail.length;
          octx.beginPath();
          octx.moveTo(m.ox + v.trail[i - 1][0] * m.dw, m.oy + v.trail[i - 1][1] * m.dh);
          octx.lineTo(m.ox + v.trail[i][0] * m.dw, m.oy + v.trail[i][1] * m.dh);
          octx.stroke();
        }
        const bw = v.w * m.dw, bh = v.h * m.dh;
        const x = m.ox + v.x * m.dw - bw / 2, y = m.oy + v.y * m.dh - bh / 2;
        octx.lineWidth = 3;
        octx.globalAlpha = v.alpha;
        octx.setLineDash([]);
        octx.strokeRect(x, y, bw, bh);
        octx.fillText(tr.label + ' ' + tr.conf.toFixed(2) + (v.solid ? '' : ' (pred)'), x + 2, Math.max(16, y - 2));
        octx.globalAlpha = 1; octx.setLineDash([]);
      }
    }
    const tn = performance.now();
    liveBalls = computeBalls(tn);
    const bestBall = publishTableCoords(tn);
    if (calib) drawContacts(m);
    if (now - ballHudT >= 100) {
      ballHudT = now;
      const txt = ballHudText(bestBall);
      ballHud.textContent = txt; ballHud.style.display = txt ? 'block' : 'none';
    }
  } else liveBalls = [];
  dispFrames++;
  if (now - tMark >= 1000) {
    const dt = (now - tMark) / 1000;
    hudBody = ['display fps ' + (dispFrames / dt).toFixed(0),
      'detect fps  ' + (detCount / dt).toFixed(1),
      'follow fps  ' + (pixelFollowFrames / dt).toFixed(1),
      'follow balls ' + pixelFollowCount,
      'follow state ' + pixelFollower.lastReason,
      'worker ms   ' + pixelFollower.lastMs.toFixed(1),
      'rejected    ' + appearanceRejected,
      'mode        ' + (inferMode === 'device' ? modeText : '-'),
      'device ms   ' + lastInfer.toFixed(0),
      'round-trip  ' + lastRtt.toFixed(0) + ' ms',
      'gpu rest    ' + deviceRestMs(lastInfer, lastRtt) + ' ms',
      'tracks      ' + tracker.tracks.filter(t => t.confirmed).length,
      'rescued     ' + tracker.rescued,
      'saved       ' + (savedCount == null ? '-' : savedCount) + (collectOn ? ' (collecting)' : '')].join(NL);
    updateHud();
    dispFrames = 0; detCount = 0; pixelFollowFrames = 0; tMark = now;
  }
  drawRequest = requestAnimationFrame(next => drawLoop(next, generation));
}
</script>
</body>
</html>
"""

CERT_PATH = os.path.join(BASE_DIR, "cert.pem")
KEY_PATH = os.path.join(BASE_DIR, "key.pem")


@app.get("/cert")
def download_cert():
    with open(CERT_PATH, "rb") as f:
        data = f.read()
    return Response(data, mimetype="application/x-x509-ca-cert", headers={
        "Content-Disposition": 'attachment; filename="yolo-server.crt"'})


def _wanted_sans():
    import ipaddress
    ips = ["127.0.0.1"] + lan_ips()
    names = ["localhost"]
    try:
        names.append(socket.gethostname())
    except Exception:
        pass
    return names, [ipaddress.ip_address(i) for i in dict.fromkeys(ips)]


def ensure_cert():
    import datetime
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    names, ips = _wanted_sans()
    if os.path.exists(CERT_PATH) and os.path.exists(KEY_PATH):
        try:
            with open(CERT_PATH, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read())
            san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            have = set(san.get_values_for_type(x509.IPAddress))
            if all(i in have for i in ips) and cert.not_valid_after_utc > (
                    datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)):
                return
        except Exception:
            pass
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "yolo-server")])
    now = datetime.datetime.now(datetime.timezone.utc)
    alt = [x509.DNSName(n) for n in dict.fromkeys(names)] + [x509.IPAddress(i) for i in ips]
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=825))
            .add_extension(x509.SubjectAlternativeName(alt), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, content_commitment=False,
                key_encipherment=False, data_encipherment=False, key_agreement=False,
                crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(key, hashes.SHA256()))
    with open(KEY_PATH, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM,
                                  serialization.PrivateFormat.TraditionalOpenSSL,
                                  serialization.NoEncryption()))
    with open(CERT_PATH, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print("Generated self-signed certificate:", CERT_PATH)


if __name__ == "__main__":
    from werkzeug.serving import make_server

    print("Active provider:", PROVIDER, "| available:", session.get_providers())
    print("Input size:", [INPUT_W, INPUT_H], "| names:", names_list())
    _w = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
    _, _ms = run_model(_w, DEFAULT_CONF, DEFAULT_IOU)
    print(f"Warm-up inference done in {_ms:.1f} ms")
    HTTP_PORT = int(os.environ.get("PORT", 5000))
    HTTPS_PORT = int(os.environ.get("HTTPS_PORT", 5443))
    ensure_cert()
    for ip in lan_ips():
        print(f"App URL (http):      http://{ip}:{HTTP_PORT}")
        print(f"Safari camera page:  https://{ip}:{HTTPS_PORT}")
        print(f"Install cert:        http://{ip}:{HTTP_PORT}/cert")
    http_srv = make_server("0.0.0.0", HTTP_PORT, app, threaded=True)
    https_srv = make_server("0.0.0.0", HTTPS_PORT, app, threaded=True,
                            ssl_context=(CERT_PATH, KEY_PATH))
    threading.Thread(target=https_srv.serve_forever, daemon=True).start()
    http_srv.serve_forever()
