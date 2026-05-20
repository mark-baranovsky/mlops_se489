"""Hydra-configured training script for retail demand forecasting.

This script uses Hydra for configuration management, allowing hyperparameters
and paths to be overridden from the command line without changing code.

Example usage:
    python src/mlops_se489/models/train_with_config.py
    python src/mlops_se489/models/train_with_config.py model=gradient_boosting
    python src/mlops_se489/models/train_with_config.py experiment.split_date=2016-06-01
    python src/mlops_se489/models/train_with_config.py model.n_estimators=200
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import joblib

log = logging.getLogger(__name__)


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        RMSE as a float.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Run model training using Hydra configuration.

    Args:
        cfg: Hydra configuration object containing experiment, model, and path settings.
    """
    log.info("Starting training with config: model=%s split_date=%s", cfg.model.name, cfg.experiment.split_date)

    base_dir = Path(__file__).resolve().parents[3]
    features_path = base_dir / cfg.paths.features
    models_dir = base_dir / cfg.paths.models

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    df = pd.read_parquet(features_path)
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])

    df_train = df[df["week_start_date"] < cfg.experiment.split_date].copy()
    df_val = df[df["week_start_date"] >= cfg.experiment.split_date].copy()

    log.info("Train rows: %d | Val rows: %d", len(df_train), len(df_val))

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL]
    X_val = df_val[FEATURE_COLS]
    y_val = df_val[TARGET_COL]

    mlflow.set_experiment(cfg.experiment.name)

    if cfg.model.name == "random_forest":
        model_cls = RandomForestRegressor
        params = {
            "n_estimators": cfg.model.n_estimators,
            "max_depth": cfg.model.max_depth,
            "random_state": cfg.model.random_state,
        }
    elif cfg.model.name == "gradient_boosting":
        model_cls = GradientBoostingRegressor
        params = {
            "n_estimators": cfg.model.n_estimators,
            "max_depth": cfg.model.max_depth,
            "learning_rate": cfg.model.learning_rate,
            "random_state": cfg.model.random_state,
        }
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    with mlflow.start_run(run_name=f"hydra_{cfg.model.name}"):
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model_cls(**params)),
        ])
        pipeline.fit(X_train, y_train)

        train_rmse = rmse(y_train, pipeline.predict(X_train))
        val_rmse = rmse(y_val, pipeline.predict(X_val))
        val_mae = mean_absolute_error(y_val, pipeline.predict(X_val))

        mlflow.log_param("model_name", cfg.model.name)
        mlflow.log_param("split_date", cfg.experiment.split_date)
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.sklearn.log_model(pipeline, f"hydra_{cfg.model.name}_model")

        log.info("train RMSE: %.1f | val RMSE: %.1f | val MAE: %.1f", train_rmse, val_rmse, val_mae)

        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"hydra_{cfg.model.name}.pkl"
        joblib.dump(pipeline, model_path)
        log.info("Model saved to %s", model_path)


if __name__ == "__main__":
    train()
