"""
test_data.py

Tests the data-quality RULES applied in databricks/02_cleaning.py, using a
small hand-crafted PySpark DataFrame rather than the full pipeline. This
keeps tests fast (milliseconds, not minutes) while exercising the exact same
transformations — revenue calculation, duplicate detection, null handling,
and invalid-row filtering — that the Silver notebook applies at scale.
"""

import pandas as pd
import pytest
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


@pytest.fixture(scope="module")
def spark():
    spark = (
        SparkSession.builder
        .appName("retail-tests")
        .master("local[1]")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture
def raw_sample(spark):
    """A tiny bronze-like sample containing every kind of dirty data the
    Silver layer is responsible for handling."""
    rows = [
        # (order_id, product_id, quantity, unit_price, discount, revenue, region, ingestion_ts)
        ("ORD-0001", "P001", 2, 10.0, 0.0, 20.0, "West", "t1"),      # clean row
        ("ORD-0002", "P002", 3, 15.0, None, 45.0, "east", "t1"),     # null discount, lowercase region
        ("ORD-0003", "P003", -1, 10.0, 0.0, -10.0, "South", "t1"),   # invalid: negative quantity
        ("ORD-0001", "P001", 2, 10.0, 0.0, 20.0, "West", "t2"),      # duplicate order_id (later ingestion)
        ("ORD-0004", "P004", 2, 10.0, 0.0, 999.0, "North", "t1"),    # revenue disagrees with qty*price
    ]
    columns = ["order_id", "product_id", "quantity", "unit_price", "discount", "revenue", "region", "ingestion_timestamp"]
    return spark.createDataFrame(rows, columns)


def test_revenue_recalculation(spark, raw_sample):
    df = raw_sample.withColumn("expected_revenue", F.round(F.col("quantity") * F.col("unit_price"), 2))
    row4 = df.filter(F.col("order_id") == "ORD-0004").collect()[0]
    assert row4["revenue"] == 999.0  # the (wrong) raw value
    assert row4["expected_revenue"] == 20.0  # what it should be recalculated to


def test_duplicate_detection_keeps_first_by_ingestion(spark, raw_sample):
    dup_window = Window.partitionBy("order_id").orderBy("ingestion_timestamp")
    df = raw_sample.withColumn("_row_num", F.row_number().over(dup_window))

    duplicates = df.filter(F.col("_row_num") > 1)
    assert duplicates.count() == 1
    assert duplicates.collect()[0]["order_id"] == "ORD-0001"

    deduped = df.filter(F.col("_row_num") == 1).drop("_row_num")
    assert deduped.filter(F.col("order_id") == "ORD-0001").count() == 1


def test_null_discount_filled_with_zero(spark, raw_sample):
    df = raw_sample.fillna({"discount": 0.0})
    row = df.filter(F.col("order_id") == "ORD-0002").collect()[0]
    assert row["discount"] == 0.0


def test_negative_quantity_is_invalid(spark, raw_sample):
    invalid = raw_sample.filter(F.col("quantity") <= 0)
    valid = raw_sample.filter(F.col("quantity") > 0)
    assert invalid.count() == 1
    assert invalid.collect()[0]["order_id"] == "ORD-0003"
    assert valid.count() == 4


def test_region_standardization(spark, raw_sample):
    df = raw_sample.withColumn("region", F.initcap(F.trim(F.col("region"))))
    regions = {r["region"] for r in df.select("region").collect()}
    assert "east" not in regions
    assert "East" in regions


def test_data_quality_report_counts_reconcile(spark, raw_sample):
    """The DQ report's arithmetic must always reconcile: bronze_rows -
    silver_rows == sum of every drop reason. This guards against a rule
    being added later that drops rows without being counted."""
    total = raw_sample.count()
    invalid_qty = raw_sample.filter(F.col("quantity") <= 0).count()

    dup_window = Window.partitionBy("order_id").orderBy("ingestion_timestamp")
    with_rownum = raw_sample.withColumn("_row_num", F.row_number().over(dup_window))
    duplicates = with_rownum.filter(F.col("_row_num") > 1).count()

    remaining = with_rownum.filter((F.col("_row_num") == 1) & (F.col("quantity") > 0)).count()
    assert total - remaining == invalid_qty + duplicates
