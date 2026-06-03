"""Model training and evaluation utilities for the retail demand forecasting pipeline.

This module provides a simple Model scaffold used by tests, plus helper
functions for evaluating model performance and selecting the champion model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mlops_se489.models.base import BaseModel

logger = logging.getLogger(__name__)


class Model(BaseModel):
    """Default model scaffold.

    This class is intentionally minimal. It exists as a reusable base scaffold
    for tests and future model implementations.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the model.

        Args:
            config: Optional model configuration dictionary.
        """
        self.config = config or {}

    def fit(self, X: Any, y: Any) -> None:
        """Fit the model.

        Raises:
            NotImplementedError: This scaffold does not implement training.
        """
        raise NotImplementedError("Model.fit() must be implemented by a subclass.")

    def predict(self, X: Any) -> Any:
        """Generate predictions.

        Raises:
            NotImplementedError: This scaffold does not implement prediction.
        """
        raise NotImplementedError("Model.predict() must be implemented by a subclass.")

    def save(self, path: str | Path) -> None:
        """Save the model object to disk.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "Model":
        """Load a saved Model object from disk.

        Args:
            path: Path to the saved model file.

        Returns:
            Loaded Model instance.

        Raises:
            TypeError: If the loaded object is not a Model instance.
        """
        loaded = joblib.load(path)

        if not isinstance(loaded, cls):
            raise TypeError(f"Expected loaded object to be {cls.__name__}, got {type(loaded).__name__}")

        return loaded


def compute_rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted values from the model.

    Returns:
        RMSE as a float.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mae(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted values from the model.

    Returns:
        MAE as a float.
    """
    return float(mean_absolute_error(y_true, y_pred))


def select_champion(results: dict[str, tuple[float, str, object]]) -> str:
    """Select the champion model based on lowest validation RMSE.

    Args:
        results: Dictionary mapping model name to a tuple of
            (val_rmse, run_id, fitted_pipeline).

    Returns:
        Name of the champion model.
    """
    champion = min(results, key=lambda k: results[k][0])
    logger.info("Champion model selected: %s (val RMSE=%.1f)", champion, results[champion][0])
    return champion