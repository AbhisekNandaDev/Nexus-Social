# ── base image ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System libraries:
#   libgl1 / libglib2.0-0  — required by OpenCV (cv2) and DeepFace
#   ffmpeg                 — required by faster-whisper to decode audio from video files
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ─────────────────────────────────────────────────────────
# Copy requirements first so this layer is cached unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

# ── Application code ────────────────────────────────────────────────────────────
COPY . .

# Ensure the log directory exists (logger writes to logs/app.log)
RUN mkdir -p logs

# ── Runtime ─────────────────────────────────────────────────────────────────────
EXPOSE 8000

# Liveness probe — /health is a fast, dependency-free endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Run from /app so that src.*, utils.*, config.* are all on sys.path
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
