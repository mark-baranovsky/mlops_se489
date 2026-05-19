"""Feature engineering utilities for the retail demand forecasting pipeline.

This module provides functions for building time-series features from
weekly aggregated demand data including lag features, rolling averages,
and calendar-based features.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

ENTITY_COLS = ["warehouse", "product_code"]


def add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Add lag demand features to the DataFrame.

    Each lag feature contains the demand value N weeks before the current week
    for the same warehouse-product combination. No data leakage occurs because
    lag(N) always looks strictly backward.

    Args:
        df: DataFrame sorted by warehouse, product_code, week_start_date.
        lags: List of lag periods in weeks (e.g. [1, 2, 3, 4, 52]).

    Returns:
        DataFrame with new lag columns added.
    """
    for n in lags:
        col = f"lag_{n}_week_demand"
        df[col] = df.groupby(ENTITY_COLS)["weekly_order_demand"].shift(n)
        logger.debug("Added %s", col)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean and standard deviation features.

    Rolling features are computed over a window that ends one week before
    the current row to prevent data leakage.

    Args:
        df: DataFrame sorted by warehouse, product_code, week_start_date.

    Returns:
        DataFrame with rolling_4wk_avg_demand, rolling_4wk_std_demand,
        rolling_8wk_avg_demand, and demand_momentum columns added.
    """
    grp = df.groupby(ENTITY_COLS)["weekly_order_demand"]

    df["rolling_4wk_avg_demand"] = grp.transform(lambda x: x.shift(1).rolling(4).mean())
    df["rolling_4wk_std_demand"] = grp.transform(lambda x: x.shift(1).rolling(4).std())
    df["rolling_8wk_avg_demand"] = grp.transform(lambda x: x.shift(1).rolling(8).mean())
    df["demand_momentum"] = df["lag_1_week_demand"] - df["rolling_4wk_avg_demand"]

    logger.debug("Added rolling features")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based seasonality features.

    These features describe the target week's position in the calendar and
    carry no leakage risk since they do not depend on demand values.

    Args:
        df: DataFrame with a week_start_date column.

    Returns:
        DataFrame with quarter, is_quarter_end_week, and is_peak_month columns.
    """
    df["quarter"] = pd.to_datetime(df["week_start_date"]).dt.quarter
    df["is_quarter_end_week"] = df["week_of_year"].isin([13, 26, 39, 52]).astype(int)
    df["is_peak_month"] = df["month"].isin([11, 12, 1]).astype(int)
    logger.debug("Added calendar features")
    return df