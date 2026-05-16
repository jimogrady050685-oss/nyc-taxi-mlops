# Dockerfile — NYC Taxi Fare Prediction Service
# ==============================================
# Mirrors the structure of the lecturer's student-performance template.
# Single image used for both training and serving.
#
# Build:    docker build -t USERNAME/nyctaxi-fare:latest .
# Run API:  docker run -p 5000:5000 USERNAME/nyctaxi-fare:latest
# Train:    docker run --rm -v $(pwd)/models:/app/models USERNAME/nyctaxi-fare:latest python train.py
#
# Video talking point:
#   "The Dockerfile builds a self-contained image with my Python code,
#    the trained model file, and gunicorn as the WSGI server. The image
#    is pushed to Docker Hub by the GitHub Actions workflow. Argo CD then
#    pulls this image down into the Kind Kubernetes cluster whenever the
#    deployment manifest changes."

FROM python:3.11-slim

# Minimal system tools — curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies first (cache layer) ───────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────────────────────
COPY preprocess.py .
COPY train.py .
COPY app.py .

# ── Copy the pre-trained model file into the image ────────────────────────────
# In the GitHub Actions pipeline, train.py runs first as a separate job,
# produces models/gbt_model.joblib, uploads it as an artifact, then the
# Docker build job downloads the artifact into models/ before building.
# This way the deployed image already has the model baked in — no volume
# mounting needed in Kubernetes.
COPY models/ ./models/

ENV MODEL_DIR=/app/models \
    MODEL_PATH=/app/models/gbt_model.joblib \
    METRICS_PATH=/app/models/metrics.json \
    PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Use gunicorn (production WSGI) — two workers fit a small Kind cluster fine
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
