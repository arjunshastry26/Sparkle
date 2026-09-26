"""
features.py

Feature engineering for the "predict next month's revenue" problem, at the
(month, region, category) grain produced by gold_forecast_input.

Pure pandas — no Spark dependency — so this is easy to unit test and easy to
reuse from both the Databricks forecasting notebook and a plain local script.
"""

from __future__ import annotations

import pandas as pd

# Columns fed into the model. Kept as a single source of truth so training,
# prediction, and MLflow logging all agree on exactly what the model expects.
LAG_COLUMNS = ["lag_1", "lag_2", "lag_3", "lag_6"]
ROLLING_COLUMNS = ["rolling_mean_3", "rolling_mean_6"]
CALENDAR_COLUMNS = ["month_number", "quarter"]
TARGET_COLUMN = "revenue"


def load_forecast_input(path: str) -> pd.DataFrame:
    """Loads the gold_forecast_input export (CSV) and ensures correct dtypes."""
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"])
    return df.sort_values(["region", "category", "month"]).reset_index(drop=True)


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds lag_1/2/3/6 and rolling_mean_3/6, computed independently within
    each (region, category) time series.

    Leakage note: every rolling window and lag is built from `.shift(1)`
    onward — i.e. only months strictly BEFORE the target month are used.
    The target month's own revenue is never part of its own features.
    """
    df = df.copy()
    group_cols = ["region", "category"]

    df = df.sort_values(group_cols + ["month"])
    grouped = df.groupby(group_cols)[TARGET_COLUMN]

    df["lag_1"] = grouped.shift(1)
    df["lag_2"] = grouped.shift(2)
    df["lag_3"] = grouped.shift(3)
    df["lag_6"] = grouped.shift(6)

    # rolling means computed on the ALREADY-SHIFTED series so the current
    # month is excluded from its own rolling average
    shifted = df.groupby(group_cols)[TARGET_COLUMN].shift(1)
    df["rolling_mean_3"] = shifted.groupby([df["region"], df["category"]]).rolling(3).mean().reset_index(drop=True)
    df["rolling_mean_6"] = shifted.groupby([df["region"], df["category"]]).rolling(6).mean().reset_index(drop=True)

    df["month_number"] = df["month"].dt.month
    df["quarter"] = df["month"].dt.quarter

    return df


def add_categorical_encoding(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encodes region and category. Returns the encoded dataframe and
    the list of encoded column names, so callers always have a consistent
    feature list regardless of which categories are present in a given
    slice of data."""
    df = df.copy()
    region_dummies = pd.get_dummies(df["region"], prefix="region")
    category_dummies = pd.get_dummies(df["category"], prefix="category")
    df = pd.concat([df, region_dummies, category_dummies], axis=1)
    encoded_cols = list(region_dummies.columns) + list(category_dummies.columns)
    return df, encoded_cols


def build_feature_matrix(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """End-to-end feature build: lags + rolling + calendar + one-hot encoding.
    Drops rows without a full lag_6 history (the first 6 months of each
    region×category series), since those rows can't be fairly evaluated
    against a model trained on complete feature vectors."""
    df = add_lag_and_rolling_features(raw_df)
    df, cat_cols = add_categorical_encoding(df)

    feature_cols = LAG_COLUMNS + ROLLING_COLUMNS + CALENDAR_COLUMNS + cat_cols

    df_complete = df.dropna(subset=LAG_COLUMNS + ROLLING_COLUMNS).reset_index(drop=True)
    return df_complete, feature_cols


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits by UNIQUE MONTH (not by row) so that no month's data leaks
    across the train/validation/test boundary. Never shuffle time-series
    data before splitting — order must be preserved.
    """
    months = sorted(df["month"].unique())
    n = len(months)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_months = set(months[:train_end])
    val_months = set(months[train_end:val_end])
    test_months = set(months[val_end:])

    train_df = df[df["month"].isin(train_months)].reset_index(drop=True)
    val_df = df[df["month"].isin(val_months)].reset_index(drop=True)
    test_df = df[df["month"].isin(test_months)].reset_index(drop=True)

    return train_df, val_df, test_df
