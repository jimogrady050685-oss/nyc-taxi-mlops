"""
app.py
======
Flask API for the NYC Taxi Fare prediction model.

Two endpoints:
  POST /predict  — takes trip features, returns a predicted fare in dollars
  GET  /health   — returns model status and the last recorded validation metrics

The model is loaded once at startup from models/gbt_model.joblib.
If the file is not there the app still starts but returns 503 until a trained
model is available.
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

MODEL_PATH   = os.environ.get("MODEL_PATH",   "models/gbt_model.joblib")
METRICS_PATH = os.environ.get("METRICS_PATH", "models/metrics.json")

app = Flask(__name__)

# Load the model when the container starts. 
# If the file is missing the app keeps running but every endpoint will return 503 until a model is present.
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"WARNING: no model file found at {MODEL_PATH}. Run train.py first.")

# These must match the column order used in train.py exactly.
# The ColumnTransformer inside the pipeline expects them in this order.
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
    Returns 200 if the model is loaded and ready to serve predictions.
    Returns 503 if the model file was not found at startup.
    Also returns whatever metrics were written by the last training run.
    """
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    if model is None:
        return jsonify({
            "status":  "degraded",
            "message": "Model not loaded. Run train.py to generate the model file.",
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
    Accepts a JSON body with trip feature fields and returns a predicted fare.

    trip_distance and trip_duration_mins are required.
    Everything else has a sensible default so partial requests still work.

    Example request:
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

    # Start with sensible defaults and overlay whatever was sent in the request
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

    # These two fields carry the most signal so they are required
    required = ["trip_distance", "trip_duration_mins"]
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({
            "error":   "Missing required fields",
            "missing": missing,
            "hint":    "trip_distance and trip_duration_mins are required",
        }), 422

    row = pd.DataFrame([{col: defaults[col] for col in ALL_FEATURE_COLS}])

    try:
        log_pred    = model.predict(row)[0]
        dollar_pred = float(np.expm1(log_pred))  # model was trained on log1p(fare)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "predicted_fare_usd": round(dollar_pred, 2),
        "inputs":             {col: defaults[col] for col in ALL_FEATURE_COLS},
        "model":              "GradientBoostingRegressor",
        "note":               "Predicted base fare only. Excludes tip and tolls.",
    }), 200


if __name__ == "__main__":
    # 0.0.0.0 binds to all interfaces so the host can reach the container
    # through Docker port forwarding
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)