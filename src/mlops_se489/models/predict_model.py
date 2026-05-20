"""Batch prediction pipeline for retail demand forecasting.

This module loads the champion model and generates next-week demand
forecasts for every active warehouse-product combination.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
FEATURES_PATH = BASE_DIR / "data" / "processed" / "gold_weekly_product_demand_features.parquet"
PRED_PATH = BASE_DIR / "data" / "processed" / "demand_predictions.parquet"
CHAMPION_PATH = BASE_DIR / "models" / "champion_model.pkl"

FEATURE_COLS = [
    "lag_1_week_demand", "lag_2_week_demand", "lag_3_week_demand",
    "lag_4_week_demand", "lag_52_week_demand",
    "rolling_4wk_avg_demand", "rolling_4wk_std_demand",
    "rolling_8wk_avg_demand", "demand_momentum",
    "yoy_demand_change", "yoy_demand_pct_change",
    "year", "month", "quarter", "week_of_year",
    "is_quarter_end_week", "is_peak_month", "order_count",
]


def run_batch_prediction(
    features_path: Path = FEATURES_PATH,
    champion_path: Path = CHAMPION_PATH,
    output_path: Path = PRED_PATH,
) -> pd.DataFrame:
    """Generate next-week demand forecasts for all active products.

    Loads the champion model and applies it to the most recent week's
    features to predict demand for the following week. Only warehouse-product
    combinations that were active in the latest week are forecast.

    Args:
        features_path: Path to the features parquet file.
        champion_path: Path to the saved champion model pickle file.
        output_path: Destination path for the predictions parquet file.

    Returns:
        DataFrame containing predictions with metadata columns.

    Raises:
        FileNotFoundError: If the features or champion model files do not exist.
        AssertionError: If no predictions are generated or negative predictions exist.
    """
    logger.info("Starting batch prediction pipeline")

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not champion_path.exists():
        raise FileNotFoundError(f"Champion model not found: {champion_path}")

    model = joblib.load(champion_path)
    logger.info("Champion model loaded: %s", type(model.named_steps["model"]).__name__)

    df = pd.read_parquet(features_path)
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])

    logger.info("Features rows: %d", len(df))
    logger.info("Date range: %s to %s", df["week_start_date"].min(), df["week_start_date"].max())

    latest_week = df["week_start_date"].max()
    next_week = latest_week + timedelta(days=7)

    logger.info("Latest week in features: %s", latest_week.date())
    logger.info("Generating predictions for: %s", next_week.date())

    df_latest = df[df["week_start_date"] == latest_week].copy()
    logger.info("Active products in latest week: %d", len(df_latest))

    X = df_latest[FEATURE_COLS]
    raw_preds = model.predict(X)

    df_latest["predicted_demand"] = np.maximum(np.round(raw_preds).astype(int), 0)
    df_latest["prediction_week_start_date"] = next_week
    df_latest["features_as_of_week"] = latest_week
    df_latest["model_name"] = type(model.named_steps["model"]).__name__
    df_latest["predicted_at"] = datetime.utcnow().isoformat()

    df_preds = df_latest[[
        "warehouse", "product_code", "product_category",
        "prediction_week_start_date", "predicted_demand",
        "features_as_of_week", "weekly_order_demand",
        "model_name", "predicted_at",
    ]].copy()

    df_preds = df_preds.sort_values(["warehouse", "product_code"]).reset_index(drop=True)

    assert len(df_preds) > 0, "No predictions generated"
    neg = (df_preds["predicted_demand"] < 0).sum()
    assert neg == 0, f"Found {neg} negative predictions"

    logger.info("Predictions generated: %d", len(df_preds))
    logger.info("Min predicted_demand: %d", df_preds["predicted_demand"].min())
    logger.info("Max predicted_demand: %d", df_preds["predicted_demand"].max())
    logger.info("Avg predicted_demand: %.1f", df_preds["predicted_demand"].mean())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_preds.to_parquet(output_path, index=False)
    logger.info("Predictions written to %s", output_path)

    return df_preds


if __name__ == "__main__":
    run_batch_prediction()
