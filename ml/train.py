"""
train.py

Trains and compares two approaches to predicting next month's revenue for
each (region, category) pair:

  1. Baseline  - "next month = same as last month" (lag_1). Every forecasting
                 project needs this: if a fancier model can't beat this, the
                 fancier model isn't earning its complexity.
  2. XGBoost   - gradient-boosted trees on the lag/rolling/calendar features.

Both runs (plus an optional third "tuned" run) are logged to MLflow under a
single experiment so they can be compared side by side.

Usage:
    python ml/train.py --input <path to gold_forecast_input csv>
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(__file__))
from features import build_feature_matrix, chronological_split, TARGET_COLUMN  # noqa: E402

EXPERIMENT_NAME = os.environ.get("RETAIL_MLFLOW_EXPERIMENT", "Retail-Sales-Forecasting")
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# On Databricks, leave the tracking URI alone — the workspace already
# provides a managed MLflow tracking server, and overriding it here would
# silently disconnect runs from the workspace UI. Locally, pin an explicit
# SQLite store at the project root rather than relying on whatever a given
# MLflow version defaults to (this changed between major versions — MLflow
# 2.x defaulted to a flat-file ./mlruns store, MLflow 3.x defaults to
# sqlite:///mlflow.db — pinning it keeps `mlflow ui` pointed at the same
# place regardless of which version is installed).
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    _default_tracking_db = os.path.join(PROJECT_ROOT, "mlflow.db")
    mlflow.set_tracking_uri(os.environ.get("RETAIL_MLFLOW_TRACKING_URI", f"sqlite:///{_default_tracking_db}"))


def resolve_csv(path: str) -> str:
    """gold_forecast_input can be either a direct CSV file, or a Spark
    coalesce(1)-written directory containing a single part-*.csv file."""
    if os.path.isfile(path):
        return path
    matches = glob.glob(os.path.join(path, "part-*.csv"))
    if not matches:
        raise FileNotFoundError(f"No CSV found at or under {path}")
    return matches[0]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE is safe here since revenue is always well above zero at this grain
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}


def plot_predictions_vs_actual(y_true, y_pred, title: str, out_path: str):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual revenue")
    plt.ylabel("Predicted revenue")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_feature_importance(model: XGBRegressor, feature_cols: list[str], out_path: str):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh([feature_cols[i] for i in order][::-1], importances[order][::-1])
    plt.xlabel("Importance")
    plt.title("XGBoost feature importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def run_baseline(test_df: pd.DataFrame) -> dict:
    """Baseline prediction = last month's revenue (lag_1)."""
    y_true = test_df[TARGET_COLUMN].to_numpy()
    y_pred = test_df["lag_1"].to_numpy()
    metrics = evaluate(y_true, y_pred)

    with mlflow.start_run(run_name="baseline_last_month"):
        mlflow.log_param("model_type", "naive_baseline_lag1")
        mlflow.log_metrics(metrics)
        out_path = os.path.join(ARTIFACT_DIR, "baseline_pred_vs_actual.png")
        plot_predictions_vs_actual(y_true, y_pred, "Baseline: predicted vs actual (test set)", out_path)
        mlflow.log_artifact(out_path)

    print(f"[baseline]        MAE={metrics['mae']:>10,.2f}  RMSE={metrics['rmse']:>10,.2f}  MAPE={metrics['mape']:>6.2f}%")
    return metrics


def run_xgboost(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    feature_cols: list[str], run_name: str, params: dict,
) -> tuple[dict, XGBRegressor]:
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COLUMN]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COLUMN]

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        eval_metric="mae",
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred_test = model.predict(X_test)
    metrics = evaluate(y_test.to_numpy(), y_pred_test)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "xgboost")
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("feature_list", ", ".join(feature_cols))
        mlflow.log_metrics(metrics)

        pred_path = os.path.join(ARTIFACT_DIR, f"{run_name}_pred_vs_actual.png")
        plot_predictions_vs_actual(y_test.to_numpy(), y_pred_test, f"{run_name}: predicted vs actual (test set)", pred_path)
        mlflow.log_artifact(pred_path)

        fi_path = os.path.join(ARTIFACT_DIR, f"{run_name}_feature_importance.png")
        plot_feature_importance(model, feature_cols, fi_path)
        mlflow.log_artifact(fi_path)

        mlflow.xgboost.log_model(model, name="model")

    print(f"[{run_name:<16}] MAE={metrics['mae']:>10,.2f}  RMSE={metrics['rmse']:>10,.2f}  MAPE={metrics['mape']:>6.2f}%")
    return metrics, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to gold_forecast_input CSV (file or Spark output dir)")
    args = parser.parse_args()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    mlflow.set_experiment(EXPERIMENT_NAME)

    csv_path = resolve_csv(args.input)
    print(f"Loading forecast input from: {csv_path}")
    raw_df = pd.read_csv(csv_path)
    raw_df["month"] = pd.to_datetime(raw_df["month"])
    raw_df = raw_df.sort_values(["region", "category", "month"]).reset_index(drop=True)

    df, feature_cols = build_feature_matrix(raw_df)
    train_df, val_df, test_df = chronological_split(df)

    print(f"\nRows: total={len(df)}  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"Train months: {train_df['month'].min().date()} -> {train_df['month'].max().date()}")
    print(f"Val months:   {val_df['month'].min().date()} -> {val_df['month'].max().date()}")
    print(f"Test months:  {test_df['month'].min().date()} -> {test_df['month'].max().date()}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}\n")

    print(f"MLflow experiment: {EXPERIMENT_NAME}\n")
    print("=" * 70)

    baseline_metrics = run_baseline(test_df)

    default_params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1}
    xgb_metrics, xgb_model = run_xgboost(
        train_df, val_df, test_df, feature_cols, "xgboost_default", default_params
    )

    tuned_params = {"n_estimators": 350, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9}
    tuned_metrics, tuned_model = run_xgboost(
        train_df, val_df, test_df, feature_cols, "xgboost_tuned", tuned_params
    )

    print("=" * 70)
    print("\nComparison vs. baseline (negative = worse than baseline):")
    for name, m in [("xgboost_default", xgb_metrics), ("xgboost_tuned", tuned_metrics)]:
        mae_improvement = (baseline_metrics["mae"] - m["mae"]) / baseline_metrics["mae"] * 100
        print(f"  {name:<16} MAE improvement vs baseline: {mae_improvement:+.1f}%")

    # keep the best-performing model (lowest test MAE) as the "production" candidate
    candidates = [
        ("baseline", baseline_metrics, None),
        ("xgboost_default", xgb_metrics, xgb_model),
        ("xgboost_tuned", tuned_metrics, tuned_model),
    ]
    best_name, best_metrics, best_model = min(candidates, key=lambda c: c[1]["mae"])
    print(f"\nBest model by test MAE: {best_name} (MAE={best_metrics['mae']:,.2f})")

    # Always save an artifact describing the winner, even when the winner is
    # the baseline (model=None). predict.py reads this so it always knows
    # which approach to use and what historical error margin to quote —
    # never guessing at accuracy it hasn't measured.
    import joblib
    model_path = os.path.join(ARTIFACT_DIR, "best_model.joblib")
    joblib.dump(
        {
            "model": best_model,
            "feature_cols": feature_cols,
            "run_name": best_name,
            "test_metrics": best_metrics,
        },
        model_path,
    )
    if best_model is not None:
        print(f"Best model saved to {model_path} (used by ml/predict.py)")
    else:
        print(f"Baseline won on test MAE — {model_path} records that decision "
              f"(and its measured MAPE) so ml/predict.py uses the lag-1 "
              f"persistence forecast instead of an unjustified ML model.")


if __name__ == "__main__":
    main()
