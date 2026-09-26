# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Prediction
# MAGIC
# MAGIC **What this notebook does:** uses the winning model recorded by
# MAGIC `04_forecasting.py` to generate next month's revenue forecast for every
# MAGIC (region, category) segment, and writes it as a proper Gold table —
# MAGIC `sales_forecast` — so Power BI and the AI analyst can both read it the
# MAGIC same way they read any other Gold table.

# COMMAND ----------

import os
import sys

try:
    dbutils.widgets.text("base_path", "/dbfs/FileStore/retail_sales_analytics", "Base storage path")
    BASE_PATH = dbutils.widgets.get("base_path")
    REPO_ROOT = os.path.join("/Workspace", "Repos", "retail-sales-analytics")  # adjust to your Repos path
except NameError:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("retail-forecasting-predict").getOrCreate()
    BASE_PATH = os.environ.get("RETAIL_BASE_PATH", os.path.join(os.path.dirname(__file__), "..", "dbfs_local"))
    REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

sys.path.insert(0, os.path.join(REPO_ROOT, "ml"))
GOLD_DIR = os.path.join(BASE_PATH, "gold")

# COMMAND ----------

from predict import load_model_artifact, build_next_month_features
import numpy as np
import pandas as pd

spark_df = spark.read.parquet(os.path.join(GOLD_DIR, "forecast_input"))
raw_df = spark_df.toPandas()
raw_df["month"] = raw_df["month"].astype("datetime64[ns]")

artifact = load_model_artifact()
model, feature_cols, run_name, test_metrics = (
    artifact["model"], artifact["feature_cols"], artifact["run_name"], artifact["test_metrics"]
)
print(f"Using model: {run_name}"
      f"{' (measured test MAPE ' + str(test_metrics['mape']) + '%)' if test_metrics else ' (no measured error available)'}")

# COMMAND ----------

future, forecast_month, _ = build_next_month_features(raw_df)
print(f"Forecasting {forecast_month.date()} for {len(future)} region x category segments")

if model is not None:
    predicted = model.predict(future[feature_cols])
else:
    predicted = future["lag_1"].to_numpy()  # persistence fallback

future["predicted_revenue"] = np.round(predicted, 2)

if test_metrics is not None and "mape" in test_metrics:
    margin = test_metrics["mape"] / 100.0
    future["lower_bound"] = np.round(future["predicted_revenue"] * (1 - margin), 2)
    future["upper_bound"] = np.round(future["predicted_revenue"] * (1 + margin), 2)
else:
    future["lower_bound"] = np.nan
    future["upper_bound"] = np.nan

future["forecast_month"] = forecast_month.strftime("%Y-%m-%d")
future["model_version"] = run_name

sales_forecast_pd = future[[
    "forecast_month", "region", "category", "predicted_revenue", "lower_bound", "upper_bound", "model_version",
]].sort_values(["region", "category"]).reset_index(drop=True)

print(sales_forecast_pd.to_string(index=False))

# COMMAND ----------

# MAGIC %md ## Write sales_forecast as a Gold table (Parquet + CSV, same pattern as the other Gold tables)

# COMMAND ----------

sales_forecast_spark = spark.createDataFrame(sales_forecast_pd)
sales_forecast_spark.write.mode("overwrite").parquet(os.path.join(GOLD_DIR, "sales_forecast"))

csv_dir = os.path.join(GOLD_DIR, "csv_export", "gold_sales_forecast")
sales_forecast_spark.coalesce(1).write.mode("overwrite").option("header", True).csv(csv_dir)

# also drop a flat CSV the AI analyst reads directly (see ai_analyst/analyst.py FORECAST_PATH)
sales_forecast_pd.to_csv(os.path.join(GOLD_DIR, "sales_forecast.csv"), index=False)

print(f"sales_forecast written -> {os.path.join(GOLD_DIR, 'sales_forecast')} (Parquet) "
      f"and {os.path.join(GOLD_DIR, 'sales_forecast.csv')} (flat CSV)")
