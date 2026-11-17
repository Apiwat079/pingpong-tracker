"""Check Linux startup and routes using temporary captures and an isolated port."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np

root = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix="pingpong-smoke-") as data:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ, CAPTURE_DIR=data, MODEL=str(root / "best_960_fp32.onnx"))

    def call(path, body=None, headers=None):
        return urlopen(Request(base + path, data=body, headers=headers or {}), timeout=10)

    with subprocess.Popen([sys.executable, "-m", "gunicorn", "--workers", "1", "--threads", "8",
                           "--bind", f"127.0.0.1:{port}", "main:app"], cwd=root, env=env) as server:
        try:
            for _ in range(100):
                assert server.poll() is None, "Server stopped unexpectedly"
                try:
                    with call("/health") as response:
                        health = json.load(response)
                    break
                except OSError:
                    time.sleep(.25)
            else:
                raise RuntimeError("Server did not become ready")
            assert health["status"] == "ok" and health["provider"] == "CPUExecutionProvider"
            assert health["input_size"] == [960, 960]
            with call("/") as response:
                assert response.headers["Cross-Origin-Embedder-Policy"] == "require-corp"
                assert b"pingpong-model-delivery-20261022" in response.read()
            with call("/gallery") as response:
                assert b"Frame Gallery" in response.read()
            for route, size in [("/model.onnx?variant=dyn&prec=fp16", 19096239),
                                ("/model.onnx?variant=dyn&prec=fp32", 38340063),
                                ("/static/ort/ort-wasm-simd-threaded.asyncify.wasm", 24254953)]:
                with call(route, headers={"Range": "bytes=0-63"}) as response:
                    assert response.status == 206 and response.headers["Content-Range"] == f"bytes 0-63/{size}"
                    assert len(response.read()) == 64
            img = np.zeros((240, 320, 3), np.uint8)
            cv2.circle(img, (160, 120), 20, (255, 255, 255), -1)
            valid, jpg = cv2.imencode(".jpg", img)
            assert valid
            headers = {"Content-Type": "image/jpeg"}
            with call("/save_frame?reason=manual", jpg.tobytes(), headers) as response:
                assert json.load(response)["gallery"] == "/gallery"
            with call("/captures/list") as response:
                gallery = json.load(response)
            assert gallery["total"] == 1 and len(list(Path(data).rglob("*_manual.jpg"))) == 1
            with call(gallery["items"][0]["thumb"]) as response:
                assert response.headers.get_content_type() == "image/jpeg" and response.read(2) == b"\xff\xd8"
            with call("/detect", jpg.tobytes(), headers) as response:
                detection = json.load(response)
            assert detection["provider"] == "CPUExecutionProvider" and detection["image_size"] == [320, 240]
            print(json.dumps({"passed": True, "health": health["status"], "provider": health["provider"],
                              "cpu_inference_ms": round(detection["inference_ms"], 1), "routes_checked": 9}))
        finally:
            server.terminate()
            server.wait(timeout=15)
