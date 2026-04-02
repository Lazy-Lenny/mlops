# syntax=docker/dockerfile:1
# Multi-stage: build heavy wheels in full image, ship slim runtime.

FROM python:3.11-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# --- Runtime (minimal) ---
FROM python:3.11-slim-bookworm

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /opt/mlops

COPY src/ ./src/
COPY config/ ./config/
COPY ci/ ./ci/
COPY dvc.yaml ./
COPY dvc.lock ./

# Data and models are expected via bind mount or DVC at run time
ENTRYPOINT []
CMD ["python", "src/train.py", "--help"]
