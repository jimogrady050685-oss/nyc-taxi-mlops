"""
app.py — Flask API for NYC Taxi Fare Prediction
================================================
This Flask application loads the trained GBT model (gbt_model.joblib) and
exposes two endpoints:

  POST /predict  — accepts trip features, returns predicted fare in dollars
  GET  /health   — returns model status and last recorded validation metrics

Why Flask? The assignment requires "a Flask API or equivalent". Flask is the
simplest choice for a solo developer wrapping a single scikit-learn model.
It runs inside the Docker container on GCP VM port 5000.

Video talking point:
  "app.py is the serving layer. When GitHub Actions deploys a new model, it
   restarts this container with the updated .joblib file. The /predict endpoint
   takes the same features the model was trained on — trip distance, duration,
   hour of day, borough etc. — and returns the predicted fare in dollars.
   The /health endpoint is what the CI pipeline calls to confirm the container
   is up and responding before marking a deployment as successful."
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# ── Paths — same defaults as train.py ────────────────────────────────────────
MODEL_PATH   = os.environ.get("MODEL_PATH", "models/gbt_model.joblib")
METRICS_PATH = os.environ.get("METRICS_PATH", "models/metrics.json")

app = Flask(__name__)

# ── Load model at startup — fail fast if the file isn't there ────────────────
# joblib.load() deserialises the sklearn Pipeline (preprocessor + GBT) that
# was saved at the end of train.py.
try:
    model = joblib.load(MODEL_PATH)
    print(f"[app] Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"[app] WARNING: model file not found at {MODEL_PATH}. "
          "Run train.py first, then restart the container.")


# ── Feature columns — must match train.py exactly ────────────────────────────
# The model's ColumnTransformer expects these columns in this order.
NUM_COLS = [
    "hour_of_day", "day_of_week", "is_weekend",
    "trip_distance", "trip_duration_mins", "passenger_count",
    "has_congestion", "has_airport_fee", "has_cbd_fee", "RatecodeID",
]
CAT_COLS = ["PU_Borough_str"]
ALL_FEATURE_COLS = NUM_COLS + CAT_COLS


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.

    Returns 200 if the model is loaded and ready.
    Returns 503 if the model file wasn't found (e.g. before first training run).

    GitHub Actions CD workflow calls GET /health after deploying to confirm
    the container came up successfully.
    """
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    if model is None:
        return jsonify({
            "status":  "degraded",
            "message": "Model not loaded — run train.py to generate the model file",
            "metrics": metrics,
        }), 503

    return jsonify({
        "status":  "ok",
        "model":   "GradientBoostingRegressor",
        "metrics": metrics,
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint.

    Accepts JSON body with trip feature fields. Returns predicted fare in dollars.

    Example request body:
    {
        "hour_of_day":        14,
        "day_of_week":        3,
        "is_weekend":         0,
        "trip_distance":      3.5,
        "trip_duration_mins": 18.0,
        "passenger_count":    1,
        "has_congestion":     1,
        "has_airport_fee":    0,
        "has_cbd_fee":        1,
        "RatecodeID":         1,
        "PU_Borough_str":     "Manhattan"
    }

    Example curl:
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"trip_distance": 3.5, "trip_duration_mins": 18.0,
              "hour_of_day": 14, "day_of_week": 3, "is_weekend": 0,
              "passenger_count": 1, "has_congestion": 1,
              "has_airport_fee": 0, "has_cbd_fee": 1,
              "RatecodeID": 1, "PU_Borough_str": "Manhattan"}'
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON in request body"}), 400

    # ── Build a single-row DataFrame with all required features ──────────────
    # Default values are sensible fallbacks — same as the fillna() calls in train.py
    defaults = {
        "hour_of_day":        12,
        "day_of_week":        3,
        "is_weekend":         0,
        "trip_distance":      2.0,
        "trip_duration_mins": 10.0,
        "passenger_count":    1,
        "has_congestion":     0,
        "has_airport_fee":    0,
        "has_cbd_fee":        0,
        "RatecodeID":         1,
        "PU_Borough_str":     "Unknown",
    }
    defaults.update(data)

    # Validate that the required high-signal features are present
    required = ["trip_distance", "trip_duration_mins"]
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({
            "error":   "Missing required fields",
            "missing": missing,
            "hint":    "trip_distance and trip_duration_mins are required",
        }), 422

    # Build the feature row in the order the pipeline expects
    row = pd.DataFrame([{col: defaults[col] for col in ALL_FEATURE_COLS}])

    # ── Run prediction ────────────────────────────────────────────────────────
    try:
        log_pred    = model.predict(row)[0]
        dollar_pred = float(np.expm1(log_pred))   # inverse of log1p — back to $
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "predicted_fare_usd": round(dollar_pred, 2),
        "inputs":             {col: defaults[col] for col in ALL_FEATURE_COLS},
        "model":              "GradientBoostingRegressor",
        "note":               "Predicted fare_amount (metered base fare, excludes tip and tolls)",
    }), 200


if __name__ == "__main__":
    # 0.0.0.0 binds to all interfaces — required inside Docker so the host
    # can reach the container via port forwarding (-p 5000:5000).
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
