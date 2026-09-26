# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Gold Business Aggregates
# MAGIC
# MAGIC **What this notebook does:** turns `silver_sales` (one row per order line)
# MAGIC into the business-facing tables that Power BI, SQL analytics, and the ML
# MAGIC forecasting step all consume. Gold tables answer specific business
# MAGIC questions directly — nobody downstream should have to re-derive "revenue
# MAGIC by month" from raw transactions.
# MAGIC
# MAGIC Tables produced:
# MAGIC - `gold_monthly_sales` — revenue/orders/units by month × region × category
# MAGIC - `gold_product_performance` — per-product totals + month-over-month growth
# MAGIC - `gold_regional_performance` — per-region totals + growth rate
# MAGIC - `gold_forecast_input` — the monthly region×category series the ML model trains on

# COMMAND ----------

import os

try:
    dbutils.widgets.text("base_path", "/dbfs/FileStore/retail_sales_analytics", "Base storage path")
    BASE_PATH = dbutils.widgets.get("base_path")
except NameError:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("retail-gold-tables").getOrCreate()
    BASE_PATH = os.environ.get("RETAIL_BASE_PATH", os.path.join(os.path.dirname(__file__), "..", "dbfs_local"))

SILVER_PATH = os.path.join(BASE_PATH, "silver", "sales")
GOLD_DIR = os.path.join(BASE_PATH, "gold")

from pyspark.sql import functions as F
from pyspark.sql.window import Window

df = spark.read.parquet(SILVER_PATH)
print(f"silver_sales rows: {df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## gold_monthly_sales
# MAGIC The core table almost every Power BI page reads from. We use `net_revenue`
# MAGIC (after discount) as "revenue" throughout Gold, since that's what the
# MAGIC business actually books — gross revenue before discount is only useful
# MAGIC for discount-effectiveness analysis, not headline reporting.

# COMMAND ----------

gold_monthly_sales = (
    df.groupBy("month_start", "region", "category")
    .agg(
        F.round(F.sum("net_revenue"), 2).alias("revenue"),
        F.countDistinct("order_id").alias("orders"),
        F.sum("quantity").alias("units_sold"),
    )
    .withColumn("average_order_value", F.round(F.col("revenue") / F.col("orders"), 2))
    .withColumnRenamed("month_start", "month")
    .orderBy("month", "region", "category")
)

gold_monthly_sales.write.mode("overwrite").parquet(os.path.join(GOLD_DIR, "monthly_sales"))
print(f"gold_monthly_sales: {gold_monthly_sales.count():,} rows")
gold_monthly_sales.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## gold_product_performance
# MAGIC `sales_growth` compares each product's most recent full month of revenue
# MAGIC against the month before it — a simple, explainable momentum signal
# MAGIC (not a full regression), which is exactly what a business user asking
# MAGIC "which products are declining?" wants to see.

# COMMAND ----------

product_monthly = (
    df.groupBy("product_id", "product_name", "category", "month_start")
    .agg(F.sum("net_revenue").alias("month_revenue"))
)

latest_month = product_monthly.agg(F.max("month_start")).collect()[0][0]
prev_month = product_monthly.filter(F.col("month_start") < latest_month).agg(F.max("month_start")).collect()[0][0]

latest_rev = product_monthly.filter(F.col("month_start") == latest_month) \
    .select("product_id", F.col("month_revenue").alias("latest_month_revenue"))
prev_rev = product_monthly.filter(F.col("month_start") == prev_month) \
    .select("product_id", F.col("month_revenue").alias("prev_month_revenue"))

growth_lookup = (
    latest_rev.join(prev_rev, "product_id", "outer")
    .fillna(0, subset=["latest_month_revenue", "prev_month_revenue"])
    .withColumn(
        "sales_growth",
        F.when(F.col("prev_month_revenue") > 0,
               F.round((F.col("latest_month_revenue") - F.col("prev_month_revenue")) / F.col("prev_month_revenue"), 4))
         .otherwise(None)
    )
    .select("product_id", "sales_growth")
)

gold_product_performance = (
    df.groupBy("product_id", "product_name", "category")
    .agg(
        F.round(F.sum("net_revenue"), 2).alias("total_revenue"),
        F.sum("quantity").alias("units_sold"),
        F.round(F.avg("unit_price"), 2).alias("average_price"),
    )
    .join(growth_lookup, "product_id", "left")
    .orderBy(F.desc("total_revenue"))
)

gold_product_performance.write.mode("overwrite").parquet(os.path.join(GOLD_DIR, "product_performance"))
print(f"gold_product_performance: {gold_product_performance.count():,} rows "
      f"(growth = {prev_month} -> {latest_month})")
gold_product_performance.orderBy(F.desc("total_revenue")).show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## gold_regional_performance
# MAGIC Same month-over-month growth idea, applied at the region level.

# COMMAND ----------

region_monthly = df.groupBy("region", "month_start").agg(F.sum("net_revenue").alias("month_revenue"))

latest_region_rev = region_monthly.filter(F.col("month_start") == latest_month) \
    .select("region", F.col("month_revenue").alias("latest_month_revenue"))
prev_region_rev = region_monthly.filter(F.col("month_start") == prev_month) \
    .select("region", F.col("month_revenue").alias("prev_month_revenue"))

region_growth_lookup = (
    latest_region_rev.join(prev_region_rev, "region", "outer")
    .fillna(0, subset=["latest_month_revenue", "prev_month_revenue"])
    .withColumn(
        "growth_rate",
        F.when(F.col("prev_month_revenue") > 0,
               F.round((F.col("latest_month_revenue") - F.col("prev_month_revenue")) / F.col("prev_month_revenue"), 4))
         .otherwise(None)
    )
    .select("region", "growth_rate")
)

gold_regional_performance = (
    df.groupBy("region")
    .agg(
        F.round(F.sum("net_revenue"), 2).alias("total_revenue"),
        F.countDistinct("order_id").alias("orders"),
        F.sum("quantity").alias("units_sold"),
    )
    .join(region_growth_lookup, "region", "left")
    .orderBy(F.desc("total_revenue"))
)

gold_regional_performance.write.mode("overwrite").parquet(os.path.join(GOLD_DIR, "regional_performance"))
print(f"gold_regional_performance: {gold_regional_performance.count():,} rows")
gold_regional_performance.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## gold_forecast_input
# MAGIC The exact grain the ML model trains on: one row per
# MAGIC (month, region, category). This is intentionally a *narrower* table than
# MAGIC `gold_monthly_sales` conceptually — same grain, but we keep it as its own
# MAGIC table so the ML pipeline has a stable, purpose-built contract that won't
# MAGIC break if `gold_monthly_sales` gains new columns later.

# COMMAND ----------

gold_forecast_input = (
    df.groupBy("month_start", "region", "category")
    .agg(
        F.round(F.sum("net_revenue"), 2).alias("revenue"),
        F.countDistinct("order_id").alias("orders"),
        F.sum("quantity").alias("units_sold"),
    )
    .withColumnRenamed("month_start", "month")
    .orderBy("month", "region", "category")
)

gold_forecast_input.write.mode("overwrite").parquet(os.path.join(GOLD_DIR, "forecast_input"))
print(f"gold_forecast_input: {gold_forecast_input.count():,} rows "
      f"({gold_forecast_input.select('month').distinct().count()} distinct months)")

# also drop a flat CSV copy for Power BI / pandas / the AI analyst to read without Spark
for name, spark_df in [
    ("gold_monthly_sales", gold_monthly_sales),
    ("gold_product_performance", gold_product_performance),
    ("gold_regional_performance", gold_regional_performance),
    ("gold_forecast_input", gold_forecast_input),
]:
    csv_dir = os.path.join(GOLD_DIR, "csv_export", name)
    (
        spark_df.coalesce(1).write.mode("overwrite")
        .option("header", True)
        .csv(csv_dir)
    )
print(f"\nCSV exports written under {os.path.join(GOLD_DIR, 'csv_export')} (for Power BI / pandas consumption)")
