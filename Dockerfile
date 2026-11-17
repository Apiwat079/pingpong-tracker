FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8080 \
    MODEL=/app/best_960_fp32.onnx CAPTURE_DIR=/data/captures TZ=Asia/Bangkok

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 tracker \
    && useradd --uid 10001 --gid tracker --no-create-home tracker \
    && mkdir -p /app /data/captures \
    && chown tracker:tracker /data/captures
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py best_960_fp32.onnx best_dyn_fp16.onnx best_dyn_fp32.onnx ./
COPY table_detector.py table_seg_640.onnx ./
COPY static/ort/ ./static/ort/
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --start-period=30s --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/health',timeout=3).close()"
CMD ["sh", "-c", "exec gunicorn --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:${PORT:-8080} main:app"]
