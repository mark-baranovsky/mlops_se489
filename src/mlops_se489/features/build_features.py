"""Feature engineering pipeline for retail demand forecasting.

This module handles the Gold aggregation and feature engineering layers.
It transforms cleaned Silver data into model-ready features including
lag features, rolling averages, and calendar signals.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
SILVER_PATH = BASE_DIR / "data" / "interim" / "silver_product_demand_clean.parquet"
GOLD_PATH = BASE_DIR / "data" / "processed" / "gold_weekly_product_demand.parquet"
FEATURES_PATH = BASE_DIR / "data" / "processed" / "gold_weekly_product_demand_features.parquet"

ENTITY_COLS = ["warehouse", "product_code"]


def build_gold(silver_path: Path = SILVER_PATH, output_path: Path = GOLD_PATH) -> pd.DataFrame:
    """Aggregate Silver daily rows into Gold weekly demand totals.

    Groups by warehouse, product_code, and week_start_date to produce
    one row per product per warehouse per week. Separates returns from
    regular orders and computes weekly statistics.

    Args:
        silver_path: Path to the Silver parquet file.
        output_path: Destination path for the Gold parquet file.

    Returns:
        DataFrame with weekly aggregated demand per warehouse-product pair.

    Raises:
        FileNotFoundError: If the Silver parquet file does not exist.
    """
    logger.info("Starting Gold aggregation from %s", silver_path)

    if not silver_path.exists():
        raise FileNotFoundError(f"Silver file not found: {silver_path}")

    df = pd.read_parquet(silver_path)
    logger.info("Loaded %d rows from Silver", len(df))
    logger.info("Distinct warehouses: %d", df["warehouse"].nunique())
    logger.info("Distinct products: %d", df["product_code"].nunique())

    df["demand_type"] = df["order_demand"].apply(lambda x: "return" if x < 0 else "order")

    group_cols = ["warehouse", "product_code", "product_category", "week_start_date", "year", "month", "week_of_year"]

    orders = df[df["demand_type"] == "order"]
    returns = df[df["demand_type"] == "return"]

    weekly_orders = (
        orders.groupby(group_cols)["order_demand"]
        .agg(
            weekly_order_demand="sum",
            avg_daily_order_demand="mean",
            max_daily_order_demand="max",
            min_daily_order_demand="min",
            positive_order_count="count",
        )
        .reset_index()
    )

    weekly_returns = returns.groupby(group_cols)["order_demand"].agg(weekly_return_demand="sum").reset_index()

    weekly_all = (
        df.groupby(group_cols)["order_demand"]
        .agg(
            weekly_net_demand="sum",
            order_count="count",
        )
        .reset_index()
    )

    has_returns = returns.groupby(group_cols).size().reset_index(name="return_count")
    has_returns["has_returns"] = True

    df_gold = weekly_all.merge(weekly_orders, on=group_cols, how="left")
    df_gold = df_gold.merge(weekly_returns, on=group_cols, how="left")
    df_gold = df_gold.merge(has_returns[group_cols + ["has_returns"]], on=group_cols, how="left")

    df_gold["weekly_order_demand"] = df_gold["weekly_order_demand"].fillna(0)
    df_gold["weekly_return_demand"] = df_gold["weekly_return_demand"].fillna(0)
    df_gold["has_returns"] = df_gold["has_returns"].fillna(False)
    df_gold = df_gold.sort_values(["warehouse", "product_code", "week_start_date"]).reset_index(drop=True)
    df_gold["gold_processed_at"] = datetime.utcnow().isoformat()

    neg_demand = (df_gold["weekly_order_demand"] < 0).sum()
    if neg_demand > 0:
        logger.warning("Found %d rows with negative weekly_order_demand", neg_demand)

    logger.info("Gold file written to %s with %d rows", output_path, len(df_gold))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_gold.to_parquet(output_path, index=False)
    return df_gold


def build_features(gold_path: Path = GOLD_PATH, output_path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Engineer time-series features from the Gold weekly table.

    Creates the following features for model training:
    - Lag features: demand 1, 2, 3, 4, and 52 weeks ago
    - Rolling features: 4-week and 8-week rolling average and std
    - Demand momentum: lag_1 minus 4-week rolling average
    - Year-over-year change: absolute and percentage
    - Calendar features: quarter, is_quarter_end_week, is_peak_month

    All features use strictly historical data to prevent data leakage.

    Args:
        gold_path: Path to the Gold parquet file.
        output_path: Destination path for the features parquet file.

    Returns:
        DataFrame with all engineered features ready for model training.

    Raises:
        FileNotFoundError: If the Gold parquet file does not exist.
    """
    logger.info("Starting feature engineering from %s", gold_path)

    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    df = pd.read_parquet(gold_path)
    df = df.sort_values(["warehouse", "product_code", "week_start_date"]).reset_index(drop=True)
    logger.info("Loaded %d rows from Gold", len(df))

    for n in [1, 2, 3, 4, 52]:
        df[f"lag_{n}_week_demand"] = df.groupby(ENTITY_COLS)["weekly_order_demand"].shift(n)

    df["rolling_4wk_avg_demand"] = df.groupby(ENTITY_COLS)["weekly_order_demand"].transform(
        lambda x: x.shift(1).rolling(4).mean()
    )
    df["rolling_4wk_std_demand"] = df.groupby(ENTITY_COLS)["weekly_order_demand"].transform(
        lambda x: x.shift(1).rolling(4).std()
    )
    df["rolling_8wk_avg_demand"] = df.groupby(ENTITY_COLS)["weekly_order_demand"].transform(
        lambda x: x.shift(1).rolling(8).mean()
    )
    df["demand_momentum"] = df["lag_1_week_demand"] - df["rolling_4wk_avg_demand"]
    df["yoy_demand_change"] = df["lag_1_week_demand"] - df["lag_52_week_demand"]
    df["yoy_demand_pct_change"] = df.apply(
        lambda row: (
            (row["lag_1_week_demand"] - row["lag_52_week_demand"]) / row["lag_52_week_demand"] * 100
            if pd.notna(row["lag_52_week_demand"]) and row["lag_52_week_demand"] != 0
            else None
        ),
        axis=1,
    )

    df["quarter"] = pd.to_datetime(df["week_start_date"]).dt.quarter
    df["is_quarter_end_week"] = df["week_of_year"].isin([13, 26, 39, 52]).astype(int)
    df["is_peak_month"] = df["month"].isin([11, 12, 1]).astype(int)

    feature_cols = [
        "warehouse",
        "product_code",
        "product_category",
        "week_start_date",
        "weekly_order_demand",
        "lag_1_week_demand",
        "lag_2_week_demand",
        "lag_3_week_demand",
        "lag_4_week_demand",
        "lag_52_week_demand",
        "rolling_4wk_avg_demand",
        "rolling_4wk_std_demand",
        "rolling_8wk_avg_demand",
        "demand_momentum",
        "yoy_demand_change",
        "yoy_demand_pct_change",
        "year",
        "month",
        "quarter",
        "week_of_year",
        "is_quarter_end_week",
        "is_peak_month",
        "order_count",
        "has_returns",
    ]

    df_features = df[feature_cols].copy()
    df_features["features_processed_at"] = datetime.utcnow().isoformat()

    total = len(df_features)
    for col in feature_cols:
        n = df_features[col].isna().sum()
        pct = n / total * 100
        if pct > 50:
            logger.warning("High null rate in %s: %.1f%%", col, pct)
        else:
            logger.debug("Column %s: %.1f%% nulls", col, pct)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path, index=False)
    logger.info(
        "Features file written to %s with %d rows and %d columns",
        output_path,
        len(df_features),
        len(df_features.columns),
    )
    return df_features


if __name__ == "__main__":
    build_gold()
    build_features()
