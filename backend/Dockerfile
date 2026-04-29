# Churn Predictor - Backend Dockerfile (multi-stage)
# Stage 1: build wheels with compilers available
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps for any wheel that needs compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cached layer)
COPY requirements.txt .

# Install to a /install prefix so we can copy only the result to stage 2
RUN pip install --no-cache-dir --prefix=/install \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu


# Stage 2: slim runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy only the installed packages, not build tools
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Cloud Run injects PORT env var (default 8080)
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# Production server
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
