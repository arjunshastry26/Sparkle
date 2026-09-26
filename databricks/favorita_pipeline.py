"""Build Gold tables from the Corporacion Favorita competition data.

Run with Spark (locally or as a Databricks notebook):

    python databricks/favorita_pipeline.py

The pipeline keeps the existing analyst contract:
  - monthly: month, region, category, revenue, orders, units_sold
  - product: product_id, product_name, category, total_revenue, units_sold
  - regional: region, total_revenue, orders, units_sold
  - forecast_input: month, region, category, revenue, orders, units_sold

For Favorita, ``revenue`` means unit sales because the source does not provide
prices. ``region`` is the store state and ``category`` is the item family.
"""

from __future__ import annotations

import os
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)


PROJECT_ROOT = Path("/tmp")
FAVORITA_DIR = Path(
    os.environ.get("FAVORITA_DATA_DIR", "/Volumes/workspace/default/favorita")
)
BASE_PATH = Path(
    os.environ.get("RETAIL_BASE_PATH", "/Volumes/workspace/default/favorita_gold")
)
GOLD_DIR = BASE_PATH / "gold"


def _read_csv(spark: SparkSession, name: str, schema: StructType):
    path = FAVORITA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing Favorita file: {path}")
    return spark.read.option("header", True).schema(schema).csv(str(path))


def _write_gold(df, name: str) -> None:
    parquet_path = GOLD_DIR / name
    csv_path = GOLD_DIR / "csv_export" / f"gold_{name}"
    df.write.mode("overwrite").parquet(str(parquet_path))
    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(str(csv_path))
    )


def main() -> None:
    spark = (
        SparkSession.builder.appName("favorita-retail-sales-pipeline")
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "8"))
        .config("spark.default.parallelism", os.environ.get("SPARK_DEFAULT_PARALLELISM", "8"))
        .getOrCreate()
    )
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    train_schema = StructType(
        [
            StructField("id", LongType(), True),
            StructField("date", StringType(), True),
            StructField("store_nbr", IntegerType(), True),
            StructField("item_nbr", IntegerType(), True),
            StructField("unit_sales", DoubleType(), True),
            StructField("onpromotion", StringType(), True),
        ]
    )
    item_schema = StructType(
        [
            StructField("item_nbr", IntegerType(), True),
            StructField("family", StringType(), True),
            StructField("class", IntegerType(), True),
            StructField("perishable", IntegerType(), True),
        ]
    )
    store_schema = StructType(
        [
            StructField("store_nbr", IntegerType(), True),
            StructField("city", StringType(), True),
            StructField("state", StringType(), True),
            StructField("type", StringType(), True),
            StructField("cluster", IntegerType(), True),
        ]
    )
    train = _read_csv(spark, "train.csv", train_schema)
    items = _read_csv(spark, "items.csv", item_schema)
    stores = _read_csv(spark, "stores.csv", store_schema)

    sales = (
        train.withColumn("sale_date", F.to_date("date", "yyyy-MM-dd"))
        .withColumn(
            "onpromotion",
            F.when(F.lower(F.col("onpromotion")) == "true", F.lit(True))
            .when(F.lower(F.col("onpromotion")) == "false", F.lit(False))
            .otherwise(F.lit(False)),
        )
        .filter(F.col("sale_date").isNotNull())
        .join(items, "item_nbr", "left")
        .join(stores, "store_nbr", "left")
        .withColumn("family", F.coalesce(F.col("family"), F.lit("Unknown")))
        .withColumn("state", F.coalesce(F.col("state"), F.lit("Unknown")))
        .withColumn("month", F.trunc("sale_date", "month"))
    )
    summary = (
        sales.groupBy("month", "state", "family", "item_nbr")
        .agg(
            F.round(F.sum("unit_sales"), 2).alias("revenue"),
            F.countDistinct("id").alias("orders"),
            F.round(F.sum("unit_sales"), 2).alias("units_sold"),
            F.round(F.avg(F.col("onpromotion").cast("int")) * 100, 2).alias("promotion_rate"),
        )
    )

    monthly = (
        summary.groupBy("month", "state", "family")
        .agg(
            F.round(F.sum("revenue"), 2).alias("revenue"),
            F.sum("orders").alias("orders"),
            F.round(F.sum("units_sold"), 2).alias("units_sold"),
            F.round(F.avg("promotion_rate"), 2).alias("promotion_rate"),
        )
        .withColumnRenamed("state", "region")
        .withColumnRenamed("family", "category")
        .withColumn(
            "average_order_value",
            F.round(F.col("revenue") / F.col("orders"), 2),
        )
        .orderBy("month", "region", "category")
    )

    product_monthly = summary.groupBy("item_nbr", "family", "month").agg(
        F.sum("revenue").alias("month_revenue")
    )
    latest_month = product_monthly.agg(F.max("month")).first()[0]
    previous_month = (
        product_monthly.filter(F.col("month") < latest_month)
        .agg(F.max("month"))
        .first()[0]
    )
    latest_product = product_monthly.filter(F.col("month") == latest_month).select(
        "item_nbr", F.col("month_revenue").alias("latest_revenue")
    )
    previous_product = product_monthly.filter(
        F.col("month") == previous_month
    ).select("item_nbr", F.col("month_revenue").alias("previous_revenue"))
    product_growth = (
        latest_product.join(previous_product, "item_nbr", "outer")
        .fillna(0, subset=["latest_revenue", "previous_revenue"])
        .withColumn(
            "sales_growth",
            F.when(
                F.col("previous_revenue") > 0,
                (F.col("latest_revenue") - F.col("previous_revenue"))
                / F.col("previous_revenue"),
            ),
        )
        .select("item_nbr", "sales_growth")
    )

    product = (
        summary.groupBy("item_nbr", "family")
        .agg(
            F.round(F.sum("revenue"), 2).alias("total_revenue"),
            F.round(F.sum("units_sold"), 2).alias("units_sold"),
            F.round(F.avg("promotion_rate"), 2).alias("promotion_rate"),
        )
        .withColumnRenamed("item_nbr", "product_id")
        .withColumnRenamed("family", "category")
        .withColumn("product_name", F.col("product_id").cast("string"))
        .join(
            product_growth.withColumnRenamed("item_nbr", "product_id"),
            "product_id",
            "left",
        )
        .select(
            "product_id",
            "product_name",
            "category",
            "total_revenue",
            "units_sold",
            "promotion_rate",
            "sales_growth",
        )
        .orderBy(F.desc("total_revenue"))
    )

    regional = (
        summary.groupBy("state")
        .agg(
            F.round(F.sum("revenue"), 2).alias("total_revenue"),
            F.sum("orders").alias("orders"),
            F.round(F.sum("units_sold"), 2).alias("units_sold"),
        )
        .withColumnRenamed("state", "region")
        .orderBy(F.desc("total_revenue"))
    )

    forecast_input = (
        monthly.select(
            "month",
            "region",
            "category",
            "revenue",
            "orders",
            "units_sold",
        )
        .orderBy("month", "region", "category")
    )

    for name, frame in [
        ("monthly_sales", monthly),
        ("product_performance", product),
        ("regional_performance", regional),
        ("forecast_input", forecast_input),
    ]:
        _write_gold(frame, name)
        print(f"{name}: {frame.count():,} rows")

    spark.stop()


if __name__ == "__main__":
    main()
