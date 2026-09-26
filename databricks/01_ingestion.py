# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Bronze Ingestion
# MAGIC
# MAGIC **What this notebook does:** loads the raw retail sales CSV as-is into a
# MAGIC `bronze_sales` table. Bronze is a raw landing zone — we do **not** clean,
# MAGIC dedupe, or transform business data here. We only add metadata that lets us
# MAGIC audit *when* and *from what file* a row was ingested. That way, if a
# MAGIC downstream bug is found later, we can always re-run Silver/Gold from an
# MAGIC unmodified copy of what we received.
# MAGIC
# MAGIC **Why keep raw data untouched?** If the source system sends a bad file, or a
# MAGIC business rule changes, you want to be able to re-process history without
# MAGIC re-extracting from the source. Bronze is that insurance policy.

# COMMAND ----------

import os
from datetime import datetime, timezone

# On Databricks, `spark` and `dbutils` already exist in the notebook context.
# Locally (for testing/demoing this pipeline outside Databricks) we create a
# local SparkSession and skip dbutils widgets. This lets the exact same code
# run in both places.
try:
    dbutils.widgets.text("base_path", "/dbfs/FileStore/retail_sales_analytics", "Base storage path")
    BASE_PATH = dbutils.widgets.get("base_path")
    RUNNING_ON_DATABRICKS = True
except NameError:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("retail-bronze-ingestion").getOrCreate()
    BASE_PATH = os.environ.get("RETAIL_BASE_PATH", os.path.join(os.path.dirname(__file__), "..", "dbfs_local"))
    RUNNING_ON_DATABRICKS = False

RAW_CSV_PATH = os.environ.get(
    "RETAIL_RAW_CSV",
    os.path.join(os.path.dirname(__file__), "..", "data", "raw", "retail_sales_raw.csv"),
)
BRONZE_PATH = os.path.join(BASE_PATH, "bronze", "sales")

print(f"Running on Databricks: {RUNNING_ON_DATABRICKS}")
print(f"Reading raw CSV from:  {RAW_CSV_PATH}")
print(f"Writing bronze to:     {BRONZE_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the expected raw schema
# MAGIC We set an explicit schema instead of `inferSchema=True`. Schema inference
# MAGIC requires an extra full pass over the file and can silently guess wrong
# MAGIC types (e.g. reading `discount` as a string if a row is malformed). Being
# MAGIC explicit here also documents the contract we expect from upstream.

# COMMAND ----------

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)

raw_schema = StructType([
    StructField("order_id", StringType(), True),
    StructField("order_date", StringType(), True),   # parsed to DateType in Silver, not Bronze
    StructField("product_id", StringType(), True),
    StructField("product_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("sub_category", StringType(), True),
    StructField("region", StringType(), True),
    StructField("city", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("unit_price", DoubleType(), True),
    StructField("discount", DoubleType(), True),
    StructField("revenue", DoubleType(), True),
])

# COMMAND ----------

# MAGIC %md ## Load and stamp with ingestion metadata

# COMMAND ----------

from pyspark.sql import functions as F

df_raw = (
    spark.read
    .option("header", True)
    .schema(raw_schema)
    .csv(RAW_CSV_PATH)
)

source_file_name = os.path.basename(RAW_CSV_PATH)
ingestion_ts = datetime.now(timezone.utc).isoformat()

df_bronze = (
    df_raw
    .withColumn("ingestion_timestamp", F.lit(ingestion_ts))
    .withColumn("source_file", F.lit(source_file_name))
)

row_count = df_bronze.count()
print(f"Loaded {row_count:,} raw rows.")
df_bronze.printSchema()
df_bronze.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md ## Write bronze_sales
# MAGIC We write as Parquet, partitioned by nothing (bronze is small enough and we
# MAGIC want a single scan-friendly source of truth). In a production system on
# MAGIC Databricks you'd typically use Delta Lake here for ACID writes and time
# MAGIC travel — Parquet is used for this project to keep the stack minimal.

# COMMAND ----------

df_bronze.write.mode("overwrite").parquet(BRONZE_PATH)
print(f"bronze_sales written: {row_count:,} rows -> {BRONZE_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify
# MAGIC Quick sanity read-back — row count should match what we just wrote, and no
# MAGIC business logic has been applied yet (duplicates, bad casing, negative
# MAGIC quantities, and nulls from the source file are all still present on
# MAGIC purpose — Silver is responsible for handling them).

# COMMAND ----------

check_df = spark.read.parquet(BRONZE_PATH)
assert check_df.count() == row_count, "Row count mismatch after write!"
print("Bronze ingestion verified OK.")
