# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Silver Cleaning
# MAGIC
# MAGIC **What this notebook does:** reads `bronze_sales` and produces a clean,
# MAGIC trustworthy `silver_sales` table, plus a data-quality report documenting
# MAGIC exactly what was fixed or removed and why.
# MAGIC
# MAGIC Every transformation below is intentionally explicit (no "magic" cleaning
# MAGIC library) so each rule is easy to justify to a business stakeholder.

# COMMAND ----------

import os

try:
    dbutils.widgets.text("base_path", "/dbfs/FileStore/retail_sales_analytics", "Base storage path")
    BASE_PATH = dbutils.widgets.get("base_path")
except NameError:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("retail-silver-cleaning").getOrCreate()
    BASE_PATH = os.environ.get("RETAIL_BASE_PATH", os.path.join(os.path.dirname(__file__), "..", "dbfs_local"))

BRONZE_PATH = os.path.join(BASE_PATH, "bronze", "sales")
SILVER_PATH = os.path.join(BASE_PATH, "silver", "sales")
DQ_REPORT_PATH = os.path.join(BASE_PATH, "silver", "data_quality_report.json")

print(f"Reading bronze from: {BRONZE_PATH}")
print(f"Writing silver to:   {SILVER_PATH}")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

df = spark.read.parquet(BRONZE_PATH)
total_rows = df.count()
print(f"Bronze rows read: {total_rows:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Standardize text fields
# MAGIC Bronze can contain inconsistent casing (`"south"` vs `"South"`) from
# MAGIC upstream systems. We title-case `region`, `category`, and `sub_category`
# MAGIC so the same real-world value always groups together in aggregates —
# MAGIC otherwise `"south"` and `"South"` would silently become two different
# MAGIC regions in every downstream chart.

# COMMAND ----------

df = (
    df
    .withColumn("region", F.initcap(F.trim(F.col("region"))))
    .withColumn("category", F.initcap(F.trim(F.col("category"))))
    .withColumn("sub_category", F.initcap(F.trim(F.col("sub_category"))))
    .withColumn("order_id", F.trim(F.col("order_id")))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Parse `order_date` into a real DateType
# MAGIC Bronze keeps dates as strings (safer for raw ingestion — a malformed date
# MAGIC string won't fail the whole file load). Here we parse it properly and
# MAGIC drop rows where parsing fails, since a transaction with no valid date is
# MAGIC unusable for time-series analysis or forecasting.

# COMMAND ----------

df = df.withColumn("order_date_parsed", F.to_date("order_date", "yyyy-MM-dd"))

unparseable_dates = df.filter(F.col("order_date_parsed").isNull()).count()
print(f"Rows with unparseable order_date: {unparseable_dates:,}")

df = df.filter(F.col("order_date_parsed").isNotNull())
df = df.drop("order_date").withColumnRenamed("order_date_parsed", "order_date")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Handle nulls
# MAGIC `discount` nulls are treated as "no discount was applied" (0.0) rather than
# MAGIC dropped — a missing discount is a reasonable default, not a reason to lose
# MAGIC an otherwise-valid transaction. Nulls in identifying fields (`order_id`,
# MAGIC `product_id`) are not recoverable and those rows are dropped.

# COMMAND ----------

null_discount_count = df.filter(F.col("discount").isNull()).count()
df = df.fillna({"discount": 0.0})

null_id_count = df.filter(F.col("order_id").isNull() | F.col("product_id").isNull()).count()
df = df.filter(F.col("order_id").isNotNull() & F.col("product_id").isNotNull())

print(f"Nulls in discount (filled with 0.0): {null_discount_count:,}")
print(f"Rows dropped for missing order_id/product_id: {null_id_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Validate numeric fields
# MAGIC Business rule: quantity must be positive, unit_price and revenue cannot be
# MAGIC negative. A negative quantity almost always indicates a data entry error
# MAGIC or an unhandled return/refund code from the source system — since this
# MAGIC feed has no return-transaction semantics, we treat these as invalid rather
# MAGIC than guessing at intent.

# COMMAND ----------

invalid_quantity = df.filter(F.col("quantity") <= 0).count()
invalid_price = df.filter(F.col("unit_price") < 0).count()
invalid_revenue = df.filter(F.col("revenue") < 0).count()

df_invalid = df.filter((F.col("quantity") <= 0) | (F.col("unit_price") < 0) | (F.col("revenue") < 0))
df = df.filter((F.col("quantity") > 0) & (F.col("unit_price") >= 0) & (F.col("revenue") >= 0))

print(f"Invalid quantity (<=0): {invalid_quantity:,}")
print(f"Invalid unit_price (<0): {invalid_price:,}")
print(f"Invalid revenue (<0): {invalid_revenue:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Recalculate revenue where it disagrees with quantity × unit_price
# MAGIC Revenue should always equal `quantity * unit_price` before discount. If a
# MAGIC source-system rounding error causes a mismatch, we trust the raw
# MAGIC quantity/price (the more granular fields) and recompute revenue, rather
# MAGIC than trusting a pre-aggregated number we can't audit.

# COMMAND ----------

df = df.withColumn("expected_revenue", F.round(F.col("quantity") * F.col("unit_price"), 2))
revenue_mismatch = df.filter(F.abs(F.col("revenue") - F.col("expected_revenue")) > 0.01).count()
print(f"Rows where revenue disagreed with quantity*unit_price: {revenue_mismatch:,} (recalculated)")

df = df.withColumn("revenue", F.col("expected_revenue")).drop("expected_revenue")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Remove duplicate orders
# MAGIC The same `order_id` should never appear twice. We keep the first
# MAGIC occurrence (by ingestion_timestamp) and drop the rest.

# COMMAND ----------

dup_window = Window.partitionBy("order_id").orderBy("ingestion_timestamp")
df = (
    df.withColumn("_row_num", F.row_number().over(dup_window))
)
duplicate_rows = df.filter(F.col("_row_num") > 1).count()
df = df.filter(F.col("_row_num") == 1).drop("_row_num")

print(f"Duplicate order_id rows removed: {duplicate_rows:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Derived time & revenue fields
# MAGIC These make Gold-layer aggregation and the ML feature pipeline much
# MAGIC simpler — every downstream consumer needs `year`/`month`/`quarter`
# MAGIC eventually, so we compute them once here.
# MAGIC
# MAGIC `net_revenue` is what the business actually earns: `revenue * (1 - discount)`.
# MAGIC `revenue` (gross, before discount) is kept too, since some analyses
# MAGIC (e.g. discount effectiveness) need both.

# COMMAND ----------

df_silver = (
    df
    .withColumn("year", F.year("order_date"))
    .withColumn("month", F.month("order_date"))
    .withColumn("month_start", F.trunc("order_date", "month"))
    .withColumn("quarter", F.quarter("order_date"))
    .withColumn("week", F.weekofyear("order_date"))
    .withColumn("net_revenue", F.round(F.col("revenue") * (1 - F.col("discount")), 2))
)

silver_row_count = df_silver.count()
print(f"Final silver_sales row count: {silver_row_count:,}")
df_silver.printSchema()

# COMMAND ----------

# MAGIC %md ## Write silver_sales

# COMMAND ----------

df_silver.write.mode("overwrite").parquet(SILVER_PATH)
print(f"silver_sales written -> {SILVER_PATH}")

# COMMAND ----------

# MAGIC %md ## Data quality report
# MAGIC A simple, auditable summary — not a dedicated DQ platform, just numbers a
# MAGIC stakeholder (or a future engineer debugging a discrepancy) can trust.

# COMMAND ----------

import json

report = {
    "bronze_rows": total_rows,
    "silver_rows": silver_row_count,
    "rows_dropped_total": total_rows - silver_row_count,
    "unparseable_dates_dropped": unparseable_dates,
    "missing_id_rows_dropped": null_id_count,
    "invalid_quantity_dropped": invalid_quantity,
    "invalid_unit_price_dropped": invalid_price,
    "invalid_revenue_dropped": invalid_revenue,
    "duplicate_rows_dropped": duplicate_rows,
    "null_discount_filled_with_zero": null_discount_count,
    "revenue_recalculated_rows": revenue_mismatch,
}

print(json.dumps(report, indent=2))

os.makedirs(os.path.dirname(DQ_REPORT_PATH), exist_ok=True)
with open(DQ_REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nData quality report written -> {DQ_REPORT_PATH}")
