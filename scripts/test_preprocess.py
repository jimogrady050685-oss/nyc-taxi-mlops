"""
scripts/test_preprocess.py
==========================
Smoke test for the preprocessing pipeline. Used by ci.yml on every push.

Creates a small synthetic DataFrame with the same column structure as the
real NYC TLC dataset, runs it through feature_engineering() and time_split(),
and checks the output looks correct. No real data downloaded, no model trained.
Completes in about 5 seconds and blocks a merge if anything is broken.
"""

import sys
import os

# Add the parent directory to the path so preprocess.py can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from preprocess import feature_engineering, time_split


def make_synthetic_df(n: int = 500) -> pd.DataFrame:
    """
    Builds a synthetic DataFrame that matches the NYC TLC schema.
    All values are within the valid ranges that would survive the cleaning funnel.
    """
    rng = np.random.default_rng(42)

    base = pd.Timestamp("2025-01-01")
    pickup_offsets = pd.to_timedelta(rng.integers(0, 31 * 24 * 3600, n), unit="s")
    duration_secs  = rng.integers(300, 3600, n)

    df = pd.DataFrame({
        "tpep_pickup_datetime":  base + pickup_offsets,
        "tpep_dropoff_datetime": base + pickup_offsets + pd.to_timedelta(duration_secs, unit="s"),
        "trip_distance":         rng.uniform(0.5, 15.0, n),
        "fare_amount":           rng.uniform(5.0, 80.0, n),
        "total_amount":          rng.uniform(8.0, 90.0, n),
        "passenger_count":       rng.integers(1, 5, n).astype(float),
        "RatecodeID":            rng.choice([1, 2, 3, 4], n).astype(float),
        "congestion_surcharge":  rng.choice([0.0, 2.5], n),
        "Airport_fee":           rng.choice([0.0, 1.75], n),
        "cbd_congestion_fee":    rng.choice([0.0, 2.25], n),
        "PULocationID":          rng.integers(1, 264, n),
        "PU_Borough":            rng.choice(
            ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"], n
        ),
        "PU_Zone":               "Test Zone",
    })

    df["trip_duration_mins"] = duration_secs / 60.0

    return df


def test_feature_engineering():
    df = make_synthetic_df(500)
    result = feature_engineering(df)

    # Check all expected columns are present after feature engineering
    required_cols = [
        "hour_of_day", "day_of_week", "is_weekend",
        "spend_per_trip", "log_spend_per_trip",
        "has_congestion", "has_airport_fee", "has_cbd_fee",
        "PU_Borough_str",
    ]
    for col in required_cols:
        assert col in result.columns, f"Missing column: {col}"

    # log_spend_per_trip is the model target so it must be finite and positive
    assert result["log_spend_per_trip"].notna().all(), "NaN in log_spend_per_trip"
    assert (result["log_spend_per_trip"] > 0).all(), "Non-positive log target"

    # Basic sanity checks on engineered features
    assert result["hour_of_day"].between(0, 23).all(), "hour_of_day out of range"
    assert set(result["is_weekend"].unique()).issubset({0, 1}), "is_weekend not binary"

    print(f"  feature_engineering passed: {len(result)} rows, {len(result.columns)} columns")


def test_time_split():
    df = make_synthetic_df(500)
    df = feature_engineering(df)
    train_df, val_df = time_split(df)

    # Both splits must have rows
    assert len(train_df) > 0, "Empty training split"
    assert len(val_df)   > 0, "Empty validation split"

    # Validation set must not contain any records from before the split date
    assert (val_df["tpep_pickup_datetime"] >= pd.Timestamp("2025-01-25")).all(), \
        "Temporal leakage: val set contains pre-split-date records"

    print(f"  time_split passed: train={len(train_df)}, val={len(val_df)}")


if __name__ == "__main__":
    print("Running preprocessing smoke tests...")
    test_feature_engineering()
    test_time_split()
    print("All smoke tests passed.")