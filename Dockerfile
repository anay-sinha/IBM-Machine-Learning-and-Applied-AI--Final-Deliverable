# ---------------------------------------------------------------------------
# Fraud Detection Service – Multi-stage Dockerfile
# ---------------------------------------------------------------------------
# Stage 1: builder  – installs heavy ML dependencies into a venv
# Stage 2: runtime  – lean image with only what's needed to serve
#
# Build:
#   docker build -t fraud-detection:latest .
#
# Run (CPU):
#   docker run -p 8000:8000 \
#     -v $(pwd)/models:/app/models:ro \
#     -v $(pwd)/data:/app/data:ro \
#     fraud-detection:latest
#
# Run (GPU – requires nvidia-container-toolkit):
#   docker run --gpus all -p 8000:8000 \
#     -v $(pwd)/models:/app/models:ro \
#     -v $(pwd)/data:/app/data:ro \
#     fraud-detection:latest

# ---------------------------------------------------------------------------
# Stage 1 – Builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first to leverage Docker layer cache
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 – Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="ml-platform@yourcompany.com"
LABEL description="Fraud Detection Real-Time Scoring Service"
LABEL version="1.0.0"

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user for security
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy application source
COPY --chown=appuser:appgroup fraud_detection/ /app/

# Mount points (models and data are injected at runtime)
RUN mkdir -p /app/models /app/data /app/reports \
 && chown -R appuser:appgroup /app

USER appuser

# Health-check – poll /health every 30 s; 3 retries before unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

EXPOSE 8000

# Gunicorn + Uvicorn workers for production throughput
# Override CMD with `docker run ... gunicorn app:app ...` for multi-worker
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]
