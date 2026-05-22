# Dockerfile
# ===========
# Builds a single image used for both training and serving.
# The trained model is baked into the image at build time rather than mounted at runtime. 
# This means the image that gets tested in CI is exactly the same image that runs in Kubernetes.
#
# Build:   docker build -t jcogrady/nyctaxi-fare:latest .
# Run API: docker run -p 5000:5000 jcogrady/nyctaxi-fare:latest
# Train:   docker run --rm -v $(pwd)/models:/app/models jcogrady/nyctaxi-fare:latest python train.py

FROM python:3.11-slim

# curl is needed for the healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer gets cached between builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY preprocess.py .
COPY train.py .
COPY app.py .

# Copy the trained model into the image.
# In the GitHub Actions pipeline the train job runs first, uploads the model as an artifact, and the build-image job downloads it into models/ before
# running docker build. 
# No volume mounts needed in Kubernetes.
COPY models/ ./models/

ENV MODEL_DIR=/app/models \
    MODEL_PATH=/app/models/gbt_model.joblib \
    METRICS_PATH=/app/models/metrics.json \
    PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Two gunicorn workers is enough for a Kind cluster demo
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]