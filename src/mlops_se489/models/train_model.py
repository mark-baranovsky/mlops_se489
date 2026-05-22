"""Model training pipeline for retail demand forecasting.

This module trains and compares multiple forecasting models using scikit-learn
and Prophet, tracks all experiments with MLflow, and saves the champion model.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import ipdb
ipdb.set_trace()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
FEATURES_PATH = BASE_DIR / "data" / "processed" / "gold_weekly_product_demand_features.parquet"
MODELS_DIR = BASE_DIR / "models"
SPLIT_DATE = "2016-10-01"
EXPERIMENT_NAME = "demand_forecast_v1"

FEATURE_COLS = [
    "lag_1_week_demand", "lag_2_week_demand", "lag_3_week_demand",
    "lag_4_week_demand", "lag_52_week_demand",
    "rolling_4wk_avg_demand", "rolling_4wk_std_demand",
    "rolling_8wk_avg_demand", "demand_momentum",
    "yoy_demand_change", "yoy_demand_pct_change",
    "year", "month", "quarter", "week_of_year",
    "is_quarter_end_week", "is_peak_month", "order_count",
]
TARGET_COL = "weekly_order_demand"


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted values.

    Returns:
        RMSE as a float.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_baseline(df_train: pd.DataFrame, df_val: pd.DataFrame) -> tuple[float, str]:
    """Train and evaluate the mean-lag baseline model.

    Predicts next week demand as the average of the last 4 weeks.

    Args:
        df_train: Training DataFrame with feature columns.
        df_val: Validation DataFrame with feature columns.

    Returns:
        Tuple of (val_rmse, run_id).
    """
    logger.info("Training baseline mean-lag model")
    lag_cols = ["lag_1_week_demand", "lag_2_week_demand", "lag_3_week_demand", "lag_4_week_demand"]

    with mlflow.start_run(run_name="baseline_mean_lag"):
        preds_train = df_train[lag_cols].fillna(0).mean(axis=1)
        preds_val = df_val[lag_cols].fillna(0).mean(axis=1)

        train_rmse = rmse(df_train[TARGET_COL], preds_train)
        val_rmse = rmse(df_val[TARGET_COL], preds_val)
        val_mae = mean_absolute_error(df_val[TARGET_COL], preds_val)

        mlflow.log_param("model_type", "baseline_mean_lag")
        mlflow.log_param("split_date", SPLIT_DATE)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.set_tag("is_champion", "false")

        run_id = mlflow.active_run().info.run_id
        logger.info("Baseline — train RMSE: %.1f | val RMSE: %.1f | val MAE: %.1f", train_rmse, val_rmse, val_mae)
        return val_rmse, run_id


def train_random_forest(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    params: dict | None = None,
) -> tuple[float, str, Pipeline]:
    """Train and evaluate a Random Forest model.

    Uses a scikit-learn Pipeline with median imputation for null lag features
    followed by a RandomForestRegressor.

    Args:
        df_train: Training DataFrame with feature columns.
        df_val: Validation DataFrame with feature columns.
        params: Optional hyperparameter dictionary. Defaults to
            n_estimators=100, max_depth=6, random_state=42.

    Returns:
        Tuple of (val_rmse, run_id, fitted_pipeline).
    """
    if params is None:
        params = {"n_estimators": 100, "max_depth": 6, "random_state": 42}

    logger.info("Training Random Forest with params: %s", params)

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL]
    X_val = df_val[FEATURE_COLS]
    y_val = df_val[TARGET_COL]

    with mlflow.start_run(run_name="random_forest"):
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(**params)),
        ])
        pipeline.fit(X_train, y_train)

        train_rmse = rmse(y_train, pipeline.predict(X_train))
        val_rmse = rmse(y_val, pipeline.predict(X_val))
        val_mae = mean_absolute_error(y_val, pipeline.predict(X_val))

        mlflow.log_param("model_type", "random_forest")
        mlflow.log_param("split_date", SPLIT_DATE)
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.sklearn.log_model(pipeline, "random_forest_model")
        mlflow.set_tag("is_champion", "false")

        run_id = mlflow.active_run().info.run_id
        logger.info("Random Forest — train RMSE: %.1f | val RMSE: %.1f | val MAE: %.1f", train_rmse, val_rmse, val_mae)
        return val_rmse, run_id, pipeline


def train_gradient_boosting(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    params: dict | None = None,
) -> tuple[float, str, Pipeline]:
    """Train and evaluate a Gradient Boosting model.

    Uses a scikit-learn Pipeline with median imputation for null lag features
    followed by a GradientBoostingRegressor.

    Args:
        df_train: Training DataFrame with feature columns.
        df_val: Validation DataFrame with feature columns.
        params: Optional hyperparameter dictionary. Defaults to
            n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42.

    Returns:
        Tuple of (val_rmse, run_id, fitted_pipeline).
    """
    if params is None:
        params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "random_state": 42}

    logger.info("Training Gradient Boosting with params: %s", params)

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL]
    X_val = df_val[FEATURE_COLS]
    y_val = df_val[TARGET_COL]

    with mlflow.start_run(run_name="gradient_boosting"):
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(**params)),
        ])
        pipeline.fit(X_train, y_train)

        train_rmse = rmse(y_train, pipeline.predict(X_train))
        val_rmse = rmse(y_val, pipeline.predict(X_val))
        val_mae = mean_absolute_error(y_val, pipeline.predict(X_val))

        mlflow.log_param("model_type", "gradient_boosting")
        mlflow.log_param("split_date", SPLIT_DATE)
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.sklearn.log_model(pipeline, "gradient_boosting_model")
        mlflow.set_tag("is_champion", "false")

        run_id = mlflow.active_run().info.run_id
        logger.info("GBT — train RMSE: %.1f | val RMSE: %.1f | val MAE: %.1f", train_rmse, val_rmse, val_mae)
        return val_rmse, run_id, pipeline


def train_prophet(df_train: pd.DataFrame, df_val: pd.DataFrame) -> tuple[float, str]:
    """Train and evaluate a Prophet model on the most active product.

    Prophet is the required third-party package for this project. It is
    applied to the single product with the most historical weeks to demonstrate
    how a dedicated time-series model performs vs general-purpose tree models.

    Args:
        df_train: Training DataFrame with feature columns.
        df_val: Validation DataFrame with feature columns.

    Returns:
        Tuple of (val_rmse, run_id).
    """
    top_product = df_train.groupby("product_code")[TARGET_COL].count().idxmax()
    logger.info("Training Prophet on top product: %s", top_product)

    df_prophet_train = (
        df_train[df_train["product_code"] == top_product][["week_start_date", TARGET_COL]]
        .rename(columns={"week_start_date": "ds", TARGET_COL: "y"})
        .dropna()
    )
    df_prophet_val = (
        df_val[df_val["product_code"] == top_product][["week_start_date", TARGET_COL]]
        .rename(columns={"week_start_date": "ds", TARGET_COL: "y"})
        .dropna()
    )

    with mlflow.start_run(run_name="prophet_top_product"):
        model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
        model.fit(df_prophet_train)

        future = model.make_future_dataframe(periods=len(df_prophet_val), freq="W")
        forecast = model.predict(future)
        val_preds = forecast.tail(len(df_prophet_val))["yhat"].values

        if len(val_preds) == len(df_prophet_val):
            prophet_rmse = rmse(df_prophet_val["y"].values, val_preds)
            prophet_mae = mean_absolute_error(df_prophet_val["y"].values, val_preds)
        else:
            prophet_rmse = float("nan")
            prophet_mae = float("nan")

        mlflow.log_param("model_type", "prophet")
        mlflow.log_param("product", top_product)
        mlflow.log_param("split_date", SPLIT_DATE)
        mlflow.log_metric("val_rmse", prophet_rmse)
        mlflow.log_metric("val_mae", prophet_mae)
        mlflow.set_tag("is_champion", "false")
        mlflow.set_tag("scope", "single_product_demo")

        run_id = mlflow.active_run().info.run_id
        logger.info("Prophet — product=%s | val RMSE: %.1f | val MAE: %.1f", top_product, prophet_rmse, prophet_mae)
        return prophet_rmse, run_id


def run_training(
    features_path: Path = FEATURES_PATH,
    models_dir: Path = MODELS_DIR,
    split_date: str = SPLIT_DATE,
) -> str:
    """Run the full model training pipeline.

    Trains baseline, Random Forest, Gradient Boosting, and Prophet models.
    Selects the champion based on lowest validation RMSE and saves it to disk.

    Args:
        features_path: Path to the features parquet file.
        models_dir: Directory to save the champion model.
        split_date: Date string used to split train and validation sets.

    Returns:
        Path to the saved champion model file.

    Raises:
        FileNotFoundError: If the features parquet file does not exist.
        AssertionError: If the train or validation set is empty.
    """
    logger.info("Starting model training pipeline")

    logger.info("Expected features file: {features_path}")
      # Debugging: Check file paths and existence
    if not features_path.exists():
        breakpoint()
        logger.error("Features file not found: {features_path}")
        raise FileNotFoundError(f"Features file not found: {features_path}. Run pipeline previous stage.")

    df = pd.read_parquet(features_path)
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])

    logger.info("Total rows: %d", len(df))
    logger.info("Date range: %s to %s", df["week_start_date"].min(), df["week_start_date"].max())

    df_train = df[df["week_start_date"] < split_date].copy()
    df_val = df[df["week_start_date"] >= split_date].copy()

    assert len(df_train) > 0, "Training set is empty — move SPLIT_DATE later"
    assert len(df_val) > 0, "Validation set is empty — move SPLIT_DATE earlier"

    logger.info("Train rows: %d | Val rows: %d", len(df_train), len(df_val))

    mlflow.set_experiment(EXPERIMENT_NAME)

    baseline_rmse, _ = train_baseline(df_train, df_val)
    rf_rmse, rf_run_id, rf_pipeline = train_random_forest(df_train, df_val)
    gbt_rmse, gbt_run_id, gbt_pipeline = train_gradient_boosting(df_train, df_val)
    train_prophet(df_train, df_val)

    logger.info("=" * 50)
    logger.info("MODEL COMPARISON")
    logger.info("Baseline       val RMSE: %.1f", baseline_rmse)
    logger.info("Random Forest  val RMSE: %.1f", rf_rmse)
    logger.info("GBT            val RMSE: %.1f", gbt_rmse)

    results = {
        "random_forest": (rf_rmse, rf_run_id, rf_pipeline),
        "gradient_boosting": (gbt_rmse, gbt_run_id, gbt_pipeline),
    }

    champion_name = min(results, key=lambda k: results[k][0])
    champion_rmse = results[champion_name][0]
    champion_run_id = results[champion_name][1]
    champion_model = results[champion_name][2]

    logger.info("Champion: %s (val RMSE: %.1f)", champion_name, champion_rmse)

    models_dir.mkdir(parents=True, exist_ok=True)
    champion_path = models_dir / "champion_model.pkl"
    joblib.dump(champion_model, champion_path)
    logger.info("Champion model saved to %s", champion_path)

    client = mlflow.tracking.MlflowClient()
    client.set_tag(champion_run_id, "is_champion", "true")

    return str(champion_path)


if __name__ == "__main__":
    run_training()
