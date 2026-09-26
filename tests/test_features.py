"""
test_features.py

Tests for:
  - lag/rolling feature generation (ml/features.py)
  - chronological train/val/test split (ml/features.py)
  - forecast output schema (ml/predict.py)
  - AI analyst graceful handling when data is unavailable (ai_analyst/analyst.py)
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ml"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ai_analyst"))

from features import (  # noqa: E402
    add_lag_and_rolling_features,
    build_feature_matrix,
    chronological_split,
)
from predict import build_next_month_features  # noqa: E402
import analyst  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_series_df():
    """8 months of one region/category series with a known, simple pattern
    (revenue = 100 * month_index) so lag values are trivially predictable."""
    months = pd.date_range("2024-01-01", periods=8, freq="MS")
    return pd.DataFrame({
        "month": months,
        "region": "West",
        "category": "Technology",
        "revenue": [100 * (i + 1) for i in range(8)],
        "orders": [10] * 8,
        "units_sold": [50] * 8,
    })


@pytest.fixture
def multi_segment_df():
    """24 months x 2 segments, for split-fraction tests."""
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    rows = []
    for region in ["West", "East"]:
        for i, m in enumerate(months):
            rows.append({"month": m, "region": region, "category": "Technology",
                         "revenue": 1000 + i * 10, "orders": 5, "units_sold": 20})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lag / rolling feature tests
# ---------------------------------------------------------------------------

def test_lag_1_equals_previous_month(single_series_df):
    df = add_lag_and_rolling_features(single_series_df)
    row = df[df["month"] == "2024-06-01"].iloc[0]  # revenue = 600
    prev_row = df[df["month"] == "2024-05-01"].iloc[0]  # revenue = 500
    assert row["lag_1"] == prev_row["revenue"] == 500


def test_lag_features_do_not_leak_current_month(single_series_df):
    """The defining leakage check: a row's lag/rolling features must never
    equal (or be derived from) its own revenue value."""
    df = add_lag_and_rolling_features(single_series_df)
    for _, row in df.dropna(subset=["lag_1"]).iterrows():
        assert row["lag_1"] != row["revenue"]
        if pd.notna(row["rolling_mean_3"]):
            assert row["revenue"] not in [row["rolling_mean_3"]]  # current value shouldn't appear in its own average


def test_rolling_mean_3_excludes_current_month(single_series_df):
    df = add_lag_and_rolling_features(single_series_df)
    row = df[df["month"] == "2024-04-01"].iloc[0]  # revenue = 400
    # rolling_mean_3 should average months Jan/Feb/Mar (100, 200, 300) = 200
    assert row["rolling_mean_3"] == pytest.approx(200.0)


def test_first_months_have_null_lag6(single_series_df):
    """The first 5 months of any series can't have a valid lag_6 — this is
    expected and is exactly what build_feature_matrix() drops."""
    df = add_lag_and_rolling_features(single_series_df)
    early = df[df["month"] < "2024-07-01"]
    assert early["lag_6"].isna().all()


def test_build_feature_matrix_drops_incomplete_rows(single_series_df):
    df, feature_cols = build_feature_matrix(single_series_df)
    assert df[feature_cols].isna().sum().sum() == 0
    assert "lag_6" in feature_cols
    assert "region_West" in feature_cols
    assert "category_Technology" in feature_cols


# ---------------------------------------------------------------------------
# Chronological split tests
# ---------------------------------------------------------------------------

def test_chronological_split_no_month_overlap(multi_segment_df):
    df, _ = build_feature_matrix(multi_segment_df)
    train, val, test = chronological_split(df)

    train_months = set(train["month"])
    val_months = set(val["month"])
    test_months = set(test["month"])
    assert train_months.isdisjoint(val_months)
    assert val_months.isdisjoint(test_months)
    assert train_months.isdisjoint(test_months)


def test_chronological_split_preserves_time_order(multi_segment_df):
    df, _ = build_feature_matrix(multi_segment_df)
    train, val, test = chronological_split(df)
    assert train["month"].max() <= val["month"].min()
    assert val["month"].max() <= test["month"].min()


def test_chronological_split_roughly_matches_fractions(multi_segment_df):
    df, _ = build_feature_matrix(multi_segment_df)
    all_months = sorted(df["month"].unique())
    train, val, test = chronological_split(df, train_frac=0.7, val_frac=0.15)

    n_train_months = len(set(train["month"]))
    n_total_months = len(all_months)
    # allow slack for integer rounding on a small number of months
    assert abs(n_train_months / n_total_months - 0.7) < 0.15


# ---------------------------------------------------------------------------
# Forecast output schema
# ---------------------------------------------------------------------------

def test_forecast_output_schema(multi_segment_df):
    future, forecast_month, _ = build_next_month_features(multi_segment_df)
    expected_cols = {"lag_1", "lag_2", "lag_3", "lag_6", "rolling_mean_3", "rolling_mean_6",
                      "month_number", "quarter", "region", "category", "month"}
    assert expected_cols.issubset(set(future.columns))
    # one forecast row per distinct (region, category) segment
    assert len(future) == multi_segment_df[["region", "category"]].drop_duplicates().shape[0]
    assert (future["month"] == forecast_month).all()
    assert forecast_month == multi_segment_df["month"].max() + pd.offsets.MonthBegin(1)


# ---------------------------------------------------------------------------
# AI analyst: graceful handling of unavailable data
# ---------------------------------------------------------------------------

def test_analyst_functions_report_unavailable_on_empty_data():
    empty_data = {}
    for fn, args in [
        (analyst.get_total_revenue, (empty_data,)),
        (analyst.get_top_products, (empty_data,)),
        (analyst.get_declining_products, (empty_data,)),
        (analyst.get_forecast, (empty_data,)),
        (analyst.get_forecast_growth_region, (empty_data,)),
    ]:
        result = fn(*args)
        assert result["available"] is False
        assert "reason" in result


def test_deterministic_fallback_message_on_unavailable_data():
    result = {"available": False, "reason": "no data loaded"}
    message = analyst.format_deterministic_answer("total_revenue", result)
    assert "can't answer" in message.lower()
    assert "no data loaded" in message


def test_intent_identification_is_deterministic():
    """Same question must always route to the same intent — this routing
    step must never depend on an LLM or any randomness."""
    q = "Why did sales decrease in July 2024?"
    results = {analyst.identify_intent(q) for _ in range(5)}
    assert results == {"why_decrease"}


@pytest.mark.parametrize(
    ("question", "expected_intent"),
    [
        ("Which region performed worst?", "region_performance"),
        ("Which region had the highest revenue?", "region_performance"),
        ("Which products performed best?", "top_products"),
        ("What are the highest revenue products?", "top_products"),
        ("Which products are declining?", "declining_products"),
        ("What was revenue in December 2025?", "monthly_trend"),
        ("Which category grew the most?", "category_growth"),
    ],
)
def test_overlapping_question_phrases_route_to_specific_intent(question, expected_intent):
    assert analyst.identify_intent(question) == expected_intent


def test_forecast_missing_returns_helpful_reason():
    result = analyst.get_forecast({"monthly": pd.DataFrame()})
    assert result["available"] is False
    assert "predict.py" in result["reason"]
