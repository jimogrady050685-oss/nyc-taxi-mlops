"""
train.py — Model Training with MLflow Tracking
================================================
Converted from Databricks PySpark Cells 6-11 of the original notebook,
with MLflow added for model versioning, hyperparameter logging, and lineage.

MLflow rationale (per lecturer's advice on MLOps tasks vs ML engineering tasks):
  - Logs every training run with a unique run_id
  - Captures hyperparameters, metrics, and the model artifact in one place
  - Provides model lineage — you can trace any deployed model back to the
    exact code, data sample, and hyperparameters that produced it
  - The /mlruns directory is uploaded as a GitHub artifact, so model history
    is preserved across pipeline runs

The trained model is also saved with joblib for the Flask API to load.
metrics.json is written for the /health endpoint and quick CI inspection.

Video talking point:
  "train.py wraps the original GBT pipeline in MLflow tracking. Every run
   logs the hyperparameters, validation metrics, and the serialised model
   to the mlruns directory. This gives me model versioning and lineage —
   I can audit exactly which run produced any deployed model. The model
   artifact is also dumped to joblib for the Flask API to load at startup."
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.compose      import ColumnTransformer
from sklearn.ensemble     import GradientBoostingRegressor
from sklearn.pipeline     import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics      import mean_absolute_error, mean_squared_error, r2_score

import preprocess

# ── Column definitions — identical to original Databricks notebook Cell 6 ─────
NUM_COLS = [
    "hour_of_day", "day_of_week", "is_weekend",
    "trip_distance", "trip_duration_mins", "passenger_count",
    "has_congestion", "has_airport_fee", "has_cbd_fee", "RatecodeID",
]
CAT_COLS = ["PU_Borough_str"]
TARGET   = "log_spend_per_trip"   # log1p(fare_amount)

# ── Output paths ──────────────────────────────────────────────────────────────
MODEL_DIR    = os.environ.get("MODEL_DIR", "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "gbt_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# ── MLflow configuration ──────────────────────────────────────────────────────
# Default: local file backend at ./mlruns. This works inside the GitHub Actions
# runner without needing a separate MLflow server. The mlruns/ directory is
# uploaded as an artifact after training.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT   = os.environ.get("MLFLOW_EXPERIMENT", "nyctaxi-fare")


def build_pipeline() -> Pipeline:
    """
    PySpark MLlib pipeline mapped to sklearn:
      StringIndexer + OneHotEncoder + VectorAssembler  →  ColumnTransformer
      GBTRegressor(maxIter=150, stepSize=0.05, maxDepth=8)
        →  GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=8)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ],
        remainder="drop",
    )

    gbt = GradientBoostingRegressor(
        n_estimators     = 150,   # was maxIter=150
        learning_rate    = 0.05,  # was stepSize=0.05
        max_depth        = 8,     # was maxDepth=8
        min_samples_leaf = 5,
        random_state     = 42,
        verbose          = 1,
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",        gbt),
    ])


def evaluate(model: Pipeline, X: pd.DataFrame, y_log: pd.Series,
             y_dollars: pd.Series) -> dict:
    """Evaluate on dollar scale (expm1 of log predictions)."""
    log_preds    = model.predict(X)
    dollar_preds = np.expm1(log_preds)

    mae  = mean_absolute_error(y_dollars, dollar_preds)
    rmse = np.sqrt(mean_squared_error(y_dollars, dollar_preds))
    r2   = r2_score(y_dollars, dollar_preds)

    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def train(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Full training run with MLflow tracking."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Configure MLflow ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    feature_cols = NUM_COLS + CAT_COLS
    X_train = train_df[feature_cols].fillna({"passenger_count": 1, "RatecodeID": 1, "PU_Borough_str": "Unknown"})
    y_train = train_df[TARGET]
    X_val   = val_df[feature_cols].fillna({"passenger_count": 1, "RatecodeID": 1, "PU_Borough_str": "Unknown"})
    y_val   = val_df[TARGET]
    y_val_dollars = val_df["spend_per_trip"]

    print(f"[train] X_train: {X_train.shape}  X_val: {X_val.shape}")

    # ── Build pipeline ───────────────────────────────────────────────────────
    pipeline = build_pipeline()
    gbt_params = pipeline.named_steps["model"].get_params()

    # ── MLflow run — captures params, metrics, model in one place ────────────
    # set_tag values give us model lineage: which git SHA, which data sample,
    # which environment produced this model.
    with mlflow.start_run() as run:
        # Tags — model lineage metadata
        mlflow.set_tag("model_type", "GradientBoostingRegressor")
        mlflow.set_tag("framework",  "scikit-learn")
        mlflow.set_tag("dataset",    "nyc-tlc-yellow-2025-01")
        mlflow.set_tag("git_sha",    os.environ.get("GITHUB_SHA", "local")[:8])
        mlflow.set_tag("sample_fraction", os.environ.get("SAMPLE_FRACTION", "0.2"))

        # Log hyperparameters — these are exactly what the lecturer wants
        # to see for model versioning / experiment tracking
        for param_name in ["n_estimators", "learning_rate", "max_depth",
                           "min_samples_leaf", "random_state"]:
            mlflow.log_param(param_name, gbt_params[param_name])
        mlflow.log_param("target_transform", "log1p")
        mlflow.log_param("train_records",    len(X_train))
        mlflow.log_param("val_records",      len(X_val))

        # ── Fit ──────────────────────────────────────────────────────────────
        print("[train] Fitting GBT pipeline ...")
        pipeline.fit(X_train, y_train)

        # ── Naïve baseline for comparison ────────────────────────────────────
        global_mean_log = float(y_train.mean())
        baseline_pred   = np.full(len(y_val_dollars), np.expm1(global_mean_log))
        baseline_mae    = mean_absolute_error(y_val_dollars, baseline_pred)

        # ── Evaluate on dollar scale ─────────────────────────────────────────
        metrics = evaluate(pipeline, X_val, y_val, y_val_dollars)
        metrics["baseline_mae"] = round(baseline_mae, 4)
        metrics["model"]        = "GradientBoostingRegressor"

        # Log metrics to MLflow — these show up in the MLflow UI side-by-side
        # across all runs so you can compare model versions visually
        mlflow.log_metric("mae_dollars",   metrics["mae"])
        mlflow.log_metric("rmse_dollars",  metrics["rmse"])
        mlflow.log_metric("r2",            metrics["r2"])
        mlflow.log_metric("baseline_mae",  metrics["baseline_mae"])

        # Log the model itself as an MLflow artifact — gives you a versioned,
        # reproducible model that can be loaded later with mlflow.sklearn.load_model()
        mlflow.sklearn.log_model(pipeline, artifact_path="model",
                                 registered_model_name=None)

        run_id = run.info.run_id
        print(f"[mlflow] run_id: {run_id}")
        metrics["mlflow_run_id"] = run_id

    # ── Also save with joblib for Flask app to load at startup ────────────────
    # We keep this in addition to MLflow so app.py doesn't need an MLflow
    # client at runtime — it just loads a plain .joblib file.
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[train] Model saved → {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Metrics saved → {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 60)
    print("NYC Taxi Fare — Training with MLflow Tracking")
    print("=" * 60)

    # Allow training from a pre-computed features parquet (artifact handoff)
    # or run the full pipeline from raw data
    features_path = os.environ.get("FEATURES_PARQUET")
    if features_path and os.path.exists(features_path):
        print(f"[train] Loading pre-computed features from {features_path}")
        df = pd.read_parquet(features_path)
        # Re-apply time split on the pre-computed features
        train_df = df[df["tpep_pickup_datetime"] <  pd.Timestamp("2025-01-25")]
        val_df   = df[df["tpep_pickup_datetime"] >= pd.Timestamp("2025-01-25")]
    else:
        train_df, val_df = preprocess.run_pipeline(parquet_path)

    metrics = train(train_df, val_df)
    print(f"\n✓ Training complete. MAE=${metrics['mae']}  R2={metrics['r2']}")
