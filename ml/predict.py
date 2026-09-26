"""
predict.py

Generates the `sales_forecast` table: one row per (region, category) for the
month immediately after the most recent month in gold_forecast_input.

Uses whichever approach won during training (see ml/train.py) — read from
ml/artifacts/best_model.joblib. If that artifact says the baseline won (or
doesn't exist at all, e.g. train.py hasn't been run yet), this script falls
back to the same lag-1 persistence forecast, so there is always a safe,
honest default rather than a silent failure or an unjustified prediction.

Usage:
    python ml/predict.py --input <path to gold_forecast_input csv> --output <output csv path>
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import add_lag_and_rolling_features, add_categorical_encoding  # noqa: E402

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_model.joblib")


def resolve_csv(path: str) -> str:
    if os.path.isfile(path):
        return path
    matches = glob.glob(os.path.join(path, "part-*.csv"))
    if not matches:
        raise FileNotFoundError(f"No CSV found at or under {path}")
    return matches[0]


def load_model_artifact():
    if not os.path.exists(MODEL_PATH):
        print(f"No trained model artifact found at {MODEL_PATH}. "
              f"Run ml/train.py first. Falling back to lag-1 persistence with no measured error margin.")
        return {"model": None, "feature_cols": None, "run_name": "lag1_persistence_untested", "test_metrics": None}
    import joblib
    return joblib.load(MODEL_PATH)


def build_next_month_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Builds one feature row per (region, category) for the month AFTER the
    latest month present in the data — i.e. the actual forecast target."""
    raw_df = raw_df.sort_values(["region", "category", "month"]).reset_index(drop=True)
    last_month = raw_df["month"].max()
    forecast_month = (last_month + pd.offsets.MonthBegin(1))

    segments = raw_df[["region", "category"]].drop_duplicates()
    future_rows = segments.copy()
    future_rows["month"] = forecast_month
    future_rows["revenue"] = np.nan  # unknown — this is what we're predicting
    future_rows["orders"] = np.nan
    future_rows["units_sold"] = np.nan

    combined = pd.concat([raw_df, future_rows], ignore_index=True)
    combined = add_lag_and_rolling_features(combined)
    combined, cat_cols = add_categorical_encoding(combined)

    future = combined[combined["month"] == forecast_month].reset_index(drop=True)
    return future, forecast_month, cat_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to gold_forecast_input CSV (file or Spark output dir)")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "dbfs_local", "gold", "sales_forecast.csv"))
    args = parser.parse_args()

    csv_path = resolve_csv(args.input)
    raw_df = pd.read_csv(csv_path)
    raw_df["month"] = pd.to_datetime(raw_df["month"])

    artifact = load_model_artifact()
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    run_name = artifact["run_name"]
    test_metrics = artifact["test_metrics"]

    future, forecast_month, _ = build_next_month_features(raw_df)
    print(f"Forecasting month: {forecast_month.date()}  ({len(future)} region x category segments)")

    if model is not None:
        predicted = model.predict(future[feature_cols])
        model_version = f"{run_name}"
    else:
        # lag-1 persistence: predicted revenue = last known month's revenue
        predicted = future["lag_1"].to_numpy()
        model_version = run_name

    future["predicted_revenue"] = np.round(predicted, 2)

    # Confidence-style bounds derived from the MEASURED test MAPE of whichever
    # approach we're using — never an invented number. If no measured error
    # exists (model never trained/evaluated), bounds are omitted entirely
    # rather than guessed.
    if test_metrics is not None and "mape" in test_metrics:
        margin = test_metrics["mape"] / 100.0
        future["lower_bound"] = np.round(future["predicted_revenue"] * (1 - margin), 2)
        future["upper_bound"] = np.round(future["predicted_revenue"] * (1 + margin), 2)
    else:
        future["lower_bound"] = np.nan
        future["upper_bound"] = np.nan

    future["forecast_month"] = forecast_month.strftime("%Y-%m-%d")
    future["model_version"] = model_version

    sales_forecast = future[[
        "forecast_month", "region", "category", "predicted_revenue",
        "lower_bound", "upper_bound", "model_version",
    ]].sort_values(["region", "category"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sales_forecast.to_csv(args.output, index=False)

    print(f"\nsales_forecast written -> {args.output}")
    print(sales_forecast.to_string(index=False))


if __name__ == "__main__":
    main()
