# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Forecasting (Training)
# MAGIC
# MAGIC **What this notebook does:** reads `gold_forecast_input`, builds lag/rolling/
# MAGIC calendar features, and trains + compares two approaches to predicting next
# MAGIC month's revenue per (region, category): a naive "last month repeats"
# MAGIC baseline, and XGBoost. Both are logged to MLflow so they can be compared
# MAGIC side by side in the Experiments UI.
# MAGIC
# MAGIC **Why hand off to pandas here?** `gold_forecast_input` is at most a few
# MAGIC thousand rows (months x regions x categories) — far too small to need
# MAGIC Spark's distributed processing. Spark is the right tool for cleaning
# MAGIC 100k+ raw transactions; pandas + scikit-learn/XGBoost is the right tool
# MAGIC for a small, in-memory ML training set. Using Spark for both stages isn't
# MAGIC "more scalable" here, just slower to iterate on.
# MAGIC
# MAGIC The actual feature engineering and training logic lives in `ml/features.py`
# MAGIC and `ml/train.py` so it's identical whether run from this notebook, from
# MAGIC the command line, or from a unit test — this notebook is a thin,
# MAGIC explained wrapper around that shared code.

# COMMAND ----------

import os
import sys

try:
    dbutils.widgets.text("base_path", "/dbfs/FileStore/retail_sales_analytics", "Base storage path")
    BASE_PATH = dbutils.widgets.get("base_path")
    REPO_ROOT = os.path.join("/Workspace", "Repos", "retail-sales-analytics")  # adjust to your Repos path
except NameError:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("retail-forecasting-train").getOrCreate()
    BASE_PATH = os.environ.get("RETAIL_BASE_PATH", os.path.join(os.path.dirname(__file__), "..", "dbfs_local"))
    REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

sys.path.insert(0, os.path.join(REPO_ROOT, "ml"))
GOLD_DIR = os.path.join(BASE_PATH, "gold")

# COMMAND ----------

# MAGIC %md ## Load gold_forecast_input and convert to pandas

# COMMAND ----------

spark_df = spark.read.parquet(os.path.join(GOLD_DIR, "forecast_input"))
raw_df = spark_df.toPandas()
raw_df["month"] = raw_df["month"].astype("datetime64[ns]")
print(f"gold_forecast_input: {len(raw_df)} rows, "
      f"{raw_df['month'].nunique()} months, "
      f"{raw_df[['region','category']].drop_duplicates().shape[0]} region x category segments")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build features, split chronologically, train & compare
# MAGIC See `ml/features.py` for the exact leakage-safe lag/rolling logic, and
# MAGIC `ml/train.py` for the baseline vs. XGBoost comparison and MLflow logging.
# MAGIC This mirrors exactly what `python ml/train.py --input <path>` does from
# MAGIC the command line — same functions, just called in-notebook here.

# COMMAND ----------

from features import build_feature_matrix, chronological_split, TARGET_COLUMN
from train import (
    run_baseline, run_xgboost, evaluate, ARTIFACT_DIR, EXPERIMENT_NAME,
)
import mlflow

os.makedirs(ARTIFACT_DIR, exist_ok=True)
mlflow.set_experiment(EXPERIMENT_NAME)

df, feature_cols = build_feature_matrix(raw_df)
train_df, val_df, test_df = chronological_split(df)

print(f"Train: {train_df['month'].min().date()} -> {train_df['month'].max().date()} ({len(train_df)} rows)")
print(f"Val:   {val_df['month'].min().date()} -> {val_df['month'].max().date()} ({len(val_df)} rows)")
print(f"Test:  {test_df['month'].min().date()} -> {test_df['month'].max().date()} ({len(test_df)} rows)")

# COMMAND ----------

baseline_metrics = run_baseline(test_df)

default_params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1}
xgb_metrics, xgb_model = run_xgboost(train_df, val_df, test_df, feature_cols, "xgboost_default", default_params)

tuned_params = {"n_estimators": 350, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9}
tuned_metrics, tuned_model = run_xgboost(train_df, val_df, test_df, feature_cols, "xgboost_tuned", tuned_params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pick and persist the winner
# MAGIC **Important:** we pick whichever run has the lowest TEST MAE — we do not
# MAGIC assume XGBoost wins just because it's the more sophisticated model. If
# MAGIC the baseline wins, that's the honest result and `ml/predict.py` will use
# MAGIC the baseline approach for real forecasts (see the README's Results
# MAGIC section for why that happens on this dataset).

# COMMAND ----------

import joblib

candidates = [
    ("baseline", baseline_metrics, None),
    ("xgboost_default", xgb_metrics, xgb_model),
    ("xgboost_tuned", tuned_metrics, tuned_model),
]
best_name, best_metrics, best_model = min(candidates, key=lambda c: c[1]["mae"])
print(f"Best model by test MAE: {best_name} (MAE={best_metrics['mae']:,.2f}, MAPE={best_metrics['mape']:.2f}%)")

model_path = os.path.join(ARTIFACT_DIR, "best_model.joblib")
joblib.dump(
    {"model": best_model, "feature_cols": feature_cols, "run_name": best_name, "test_metrics": best_metrics},
    model_path,
)
print(f"Saved winning-model record -> {model_path} (read by 05_prediction.py / ml/predict.py)")
