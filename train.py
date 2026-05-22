"""
train.py
========
Model training with MLflow tracking.

Ported from the original Databricks notebook (Cells 6 to 11) with MLflow added around the training loop. 
The model itself is the same Gradient Boosting pipeline. 
The difference here is that every run is tracked so you can compare runs, audit what produced any deployed model, and trace
a live deployment back to the exact code and data that built it.

If FEATURES_PARQUET is set in the environment, train.py skips the download and loads the pre-processed features directly. 
This is how the GitHub Actions pipeline works: the preprocess job produces features.parquet and passes it as an artifact 
so the train job does not have to download the dataset again.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.compose       import ColumnTransformer
from sklearn.ensemble      import GradientBoostingRegressor
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score

import preprocess

# Must match the column definitions in preprocess.py and app.py exactly.
# The ColumnTransformer inside the pipeline expects them in this order.
NUM_COLS = [
    "hour_of_day", "day_of_week", "is_weekend",
    "trip_distance", "trip_duration_mins", "passenger_count",
    "has_congestion", "has_airport_fee", "has_cbd_fee", "RatecodeID",
]
CAT_COLS = ["PU_Borough_str"]
TARGET   = "log_spend_per_trip"   # log1p(fare_amount) — reverse with np.expm1

MODEL_DIR    = os.environ.get("MODEL_DIR", "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "gbt_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Local file backend works fine inside GitHub Actions without needing
# a separate MLflow server. The mlruns directory gets uploaded as an artifact.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT   = os.environ.get("MLFLOW_EXPERIMENT", "nyctaxi-fare")


def build_pipeline() -> Pipeline:
    """
    Builds the sklearn pipeline that mirrors the original PySpark MLlib setup.

    PySpark                                          sklearn equivalent
    StringIndexer + OneHotEncoder + VectorAssembler  ColumnTransformer
    GBTRegressor(maxIter=150, stepSize=0.05,         GradientBoostingRegressor(
      maxDepth=8)                                      n_estimators=150,
                                                       learning_rate=0.05,
                                                       max_depth=8)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ],
        remainder="drop",
    )

    gbt = GradientBoostingRegressor(
        n_estimators     = 150,
        learning_rate    = 0.05,
        max_depth        = 8,
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
    """
    Evaluates on the dollar scale so metrics are human readable.
    Predictions come out as log values and are converted back with expm1.
    """
    log_preds    = model.predict(X)
    dollar_preds = np.expm1(log_preds)

    mae  = mean_absolute_error(y_dollars, dollar_preds)
    rmse = np.sqrt(mean_squared_error(y_dollars, dollar_preds))
    r2   = r2_score(y_dollars, dollar_preds)

    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def train(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """
    Runs the full training loop with MLflow tracking.
    Returns the metrics dict which gets written to metrics.json
    and picked up by the /health endpoint in app.py.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    feature_cols = NUM_COLS + CAT_COLS
    fill_vals    = {"passenger_count": 1, "RatecodeID": 1, "PU_Borough_str": "Unknown"}

    X_train       = train_df[feature_cols].fillna(fill_vals)
    y_train       = train_df[TARGET]
    X_val         = val_df[feature_cols].fillna(fill_vals)
    y_val         = val_df[TARGET]
    y_val_dollars = val_df["spend_per_trip"]

    print(f"X_train: {X_train.shape}  X_val: {X_val.shape}")

    pipeline   = build_pipeline()
    gbt_params = pipeline.named_steps["model"].get_params()

    with mlflow.start_run() as run:

        # Tags give you the lineage link: which commit, which dataset,
        # which sample fraction produced this model
        mlflow.set_tag("model_type",      "GradientBoostingRegressor")
        mlflow.set_tag("framework",       "scikit-learn")
        mlflow.set_tag("dataset",         "nyc-tlc-yellow-2025-01")
        mlflow.set_tag("git_sha",         os.environ.get("GITHUB_SHA", "local")[:8])
        mlflow.set_tag("sample_fraction", os.environ.get("SAMPLE_FRACTION", "0.2"))

        # Log hyperparameters so every run is fully reproducible
        for param_name in ["n_estimators", "learning_rate", "max_depth",
                           "min_samples_leaf", "random_state"]:
            mlflow.log_param(param_name, gbt_params[param_name])
        mlflow.log_param("target_transform", "log1p")
        mlflow.log_param("train_records",    len(X_train))
        mlflow.log_param("val_records",      len(X_val))

        print("Fitting GBT pipeline...")
        pipeline.fit(X_train, y_train)

        # Baseline: predict the global mean fare for every trip
        # Used as a sanity check that the model is actually learning something
        global_mean_log = float(y_train.mean())
        baseline_pred   = np.full(len(y_val_dollars), np.expm1(global_mean_log))
        baseline_mae    = mean_absolute_error(y_val_dollars, baseline_pred)

        metrics = evaluate(pipeline, X_val, y_val, y_val_dollars)
        metrics["baseline_mae"] = round(baseline_mae, 4)
        metrics["model"]        = "GradientBoostingRegressor"

        mlflow.log_metric("mae_dollars",  metrics["mae"])
        mlflow.log_metric("rmse_dollars", metrics["rmse"])
        mlflow.log_metric("r2",           metrics["r2"])
        mlflow.log_metric("baseline_mae", metrics["baseline_mae"])

        # Log the full sklearn pipeline as an MLflow artifact
        # so it can be loaded and compared against future runs
        mlflow.sklearn.log_model(pipeline, artifact_path="model",
                                 registered_model_name=None)

        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")
        metrics["mlflow_run_id"] = run_id

    # Also save as a plain joblib file so app.py can load it at startup
    # without needing an MLflow client in the container
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    parquet_path  = sys.argv[1] if len(sys.argv) > 1 else None
    features_path = os.environ.get("FEATURES_PARQUET")

    print("=" * 60)
    print("NYC Taxi Fare  —  Training with MLflow Tracking")
    print("=" * 60)

    if features_path and os.path.exists(features_path):
        # Load the pre-processed artifact from the preprocess job
        print(f"Loading pre-computed features from {features_path}")
        df       = pd.read_parquet(features_path)
        train_df = df[df["tpep_pickup_datetime"] <  pd.Timestamp("2025-01-25")]
        val_df   = df[df["tpep_pickup_datetime"] >= pd.Timestamp("2025-01-25")]
    else:
        train_df, val_df = preprocess.run_pipeline(parquet_path)

    metrics = train(train_df, val_df)
    print(f"\nTraining complete.  MAE=${metrics['mae']}  R2={metrics['r2']}")