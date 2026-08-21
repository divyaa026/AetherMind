# Federated Mental Health Prediction
# Multi-stage Docker build for production deployment

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Builder stage - install dependencies
# ============================================
FROM base as builder

# Copy requirements first for layer caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ============================================
# Production stage
# ============================================
FROM base as production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/data /app/experiments /app/models && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1

# Default command
CMD ["python", "experiments/run_experiment.py", "--config", "config.yaml"]

# ============================================
# Development stage
# ============================================
FROM production as development

USER root

# Install development dependencies
RUN pip install pytest pytest-cov black flake8 isort mypy jupyter

# Switch back to appuser
USER appuser

# Expose Jupyter port
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

# ============================================
# GPU stage (requires NVIDIA Docker runtime)
# ============================================
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04 as gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install torch --index-url https://download.pytorch.org/whl/cu118

# Create user
RUN useradd --create-home --shell /bin/bash appuser

# Copy application
COPY --chown=appuser:appuser . .

# Create directories
RUN mkdir -p /app/data /app/experiments /app/models && \
    chown -R appuser:appuser /app

USER appuser

CMD ["python", "experiments/run_experiment.py", "--config", "config.yaml"]
