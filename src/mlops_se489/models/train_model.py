"""Model training pipeline for retail demand forecasting.

This module trains and compares multiple forecasting models using scikit-learn
and Prophet, tracks all experiments with MLflow, and saves the champion model.
"""

from __future__ import annotations

from pathlib import Path

import os
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

from mlops_se489.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
FEATURES_PATH: Path | str

FEATURES_PATH_STR = os.getenv("DATA_PATH")
if FEATURES_PATH_STR:
    # If it's a cloud URI string, keep it as a string or use a Cloud Path library
    FEATURES_PATH = FEATURES_PATH_STR  
else:
    FEATURES_PATH = BASE_DIR / "data" / "processed" / "gold_weekly_product_demand_features.parquet"
MODELS_DIR = BASE_DIR / "models"
SPLIT_DATE = "2016-10-01"
EXPERIMENT_NAME = "demand_forecast_v1"

FEATURE_COLS = [
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
]

TARGET_COL = "weekly_order_demand"


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_baseline(df_train: pd.DataFrame, df_val: pd.DataFrame) -> tuple[float, str]:
    """Train and evaluate the mean-lag baseline model.

    Args:
        df_train: Training DataFrame.
        df_val: Validation DataFrame.

    Returns:
        Tuple containing validation RMSE and MLflow run ID.
    """
    logger.info("Training baseline mean-lag model")

    lag_cols = [
        "lag_1_week_demand",
        "lag_2_week_demand",
        "lag_3_week_demand",
        "lag_4_week_demand",
    ]

    with mlflow.start_run(run_name="baseline_mean_lag"):
        preds_train = df_train[lag_cols].fillna(0).mean(axis=1)
        preds_val = df_val[lag_cols].fillna(0).mean(axis=1)

        train_rmse = rmse(df_train[TARGET_COL], preds_train)
        val_rmse = rmse(df_val[TARGET_COL], preds_val)
        val_mae = float(mean_absolute_error(df_val[TARGET_COL], preds_val))

        mlflow.log_param("model_type", "baseline_mean_lag")
        mlflow.log_param("split_date", SPLIT_DATE)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.set_tag("is_champion", "false")

        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("MLflow active run was not found.")
        run_id = active_run.info.run_id

    logger.info(
        "Baseline — train RMSE: %.1f | val RMSE: %.1f | val MAE: %.1f",
        train_rmse,
        val_rmse,
        val_mae,
    )
    return val_rmse, run_id


def train_random_forest(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    params: dict[str, int] | None = None,
) -> tuple[float, str, Pipeline]:
    """Train and evaluate a Random Forest model.

    Args:
        df_train: Training DataFrame.
        df_val: Validation DataFrame.
        params: Optional Random Forest hyperparameters.

    Returns:
        Tuple containing validation RMSE, MLflow run ID, and fitted pipeline.
    """
    if params is None:
        params = {"n_estimators": 100, "max_depth": 6, "random_state": 42}

    logger.info("Training Random Forest with params: %s", params)

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL]
    X_val = df_val[FEATURE_COLS]
    y_val = df_val[TARGET_COL]

    with mlflow.start_run(run_name="random_forest"):
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(**params)),
            ]
        )
        pipeline.fit(X_train, y_train)

        train_preds = pipeline.predict(X_train)
        val_preds = pipeline.predict(X_val)

        train_rmse = rmse(y_train, train_preds)
        val_rmse = rmse(y_val, val_preds)
        val_mae = float(mean_absolute_error(y_val, val_preds))

        mlflow.log_param("model_type", "random_forest")
        mlflow.log_param("split_date", SPLIT_DATE)
        mlflow.log_params(params)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.sklearn.log_model(pipeline, "random_forest_model")
        mlflow.set_tag("is_champion", "false")

        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("MLflow active run was not found.")
        run_id = active_run.info.run_id

    logger.info(
        "Random Forest — train RMSE: %.1f | val RMSE: %.1f | val MAE: %.1f",
        train_rmse,
        val_rmse,
        val_mae,
    )
    return val_rmse, run_id, pipeline


def train_gradient_boosting(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    params: dict[str, int | float] | None = None,
) -> tuple[float, str, Pipeline]:
    """Train and evaluate a Gradient Boosting model.

    Args:
        df_train: Training DataFrame.
        df_val: Validation DataFrame.
        params: Optional Gradient Boosting hyperparameters.

    Returns:
        Tuple containing validation RMSE, MLflow run ID, and fitted pipeline.
    """
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "random_state": 42,
        }

    logger.info("Training Gradient Boosting with params: %s", params)

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL]
    X_val = df_val[FEATURE_COLS]
    y_val = df_val[TARGET_COL]

    with mlflow.start_run(run_name="gradient_boosting"):
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(**params)),
            ]
        )
        pipeline.fit(X_train, y_train)

        train_preds = pipeline.predict(X_train)
        val_preds = pipeline.predict(X_val)

        train_rmse = rmse(y_train, train_preds)
        val_rmse = rmse(y_val, val_preds)
        val_mae = float(mean_absolute_error(y_val, val_preds))

        mlflow.log_param("model_type", "gradient_boosting")
        mlflow.log_param("split_date", SPLIT_DATE)
        mlflow.log_params(params)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.sklearn.log_model(pipeline, "gradient_boosting_model")
        mlflow.set_tag("is_champion", "false")

        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("MLflow active run was not found.")
        run_id = active_run.info.run_id

    logger.info(
        "GBT — train RMSE: %.1f | val RMSE: %.1f | val MAE: %.1f",
        train_rmse,
        val_rmse,
        val_mae,
    )
    return val_rmse, run_id, pipeline


def train_prophet(df_train: pd.DataFrame, df_val: pd.DataFrame) -> tuple[float, str]:
    """Train and evaluate a Prophet model on the most active product.

    Args:
        df_train: Training DataFrame.
        df_val: Validation DataFrame.

    Returns:
        Tuple containing validation RMSE and MLflow run ID.
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
            prophet_mae = float(mean_absolute_error(df_prophet_val["y"].values, val_preds))
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

        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("MLflow active run was not found.")
        run_id = active_run.info.run_id

    logger.info(
        "Prophet — product=%s | val RMSE: %.1f | val MAE: %.1f",
        top_product,
        prophet_rmse,
        prophet_mae,
    )
    return prophet_rmse, run_id


def run_training(
    features_path: Path | str = FEATURES_PATH,
    models_dir: Path = MODELS_DIR,
    split_date: str = SPLIT_DATE,
) -> str:
    """Run the full model training pipeline.

    Args:
        features_path: Path to the features parquet file.
        models_dir: Directory where the champion model will be saved.
        split_date: Date used to split training and validation data.

    Returns:
        Path to the saved champion model.
    """
    logger.info("Starting model training pipeline")
    logger.info("Expected features file: %s", features_path)

    if isinstance(features_path, Path) and not features_path.exists():
        logger.error("Features file not found: %s", features_path)
        raise FileNotFoundError(f"Features file not found: {features_path}. Run pipeline previous stage.")

    df = pd.read_parquet(features_path)
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])

    logger.info("Total rows: %d", len(df))
    logger.info(
        "Date range: %s to %s",
        df["week_start_date"].min(),
        df["week_start_date"].max(),
    )

    df_train = df[df["week_start_date"] < split_date].copy()
    df_val = df[df["week_start_date"] >= split_date].copy()

    if len(df_train) == 0:
        raise ValueError("Training set is empty — move SPLIT_DATE later")
    if len(df_val) == 0:
        raise ValueError("Validation set is empty — move SPLIT_DATE earlier")

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

    if os.getenv("CLOUD_ML_JOB_ID"):
        # If running on Vertex AI, save directly to your persistent bucket URI string
        champion_path = "gs://mlops489-retail-bucket/models/champion_model.pkl"
        logger.info("Vertex AI detected. Saving champion model directly to GCS...")
    else:
        # Fallback for local testing on your workstation
        models_dir.mkdir(parents=True, exist_ok=True)
        champion_path = str(models_dir / "champion_model.pkl")
        logger.info("Local environment detected. Saving champion locally...")

    joblib.dump(champion_model, champion_path)
    logger.info("Champion model saved to %s", champion_path)

    client = mlflow.tracking.MlflowClient()
    client.set_tag(champion_run_id, "is_champion", "true")

    return str(champion_path)

if __name__ == "__main__":
    run_training()
