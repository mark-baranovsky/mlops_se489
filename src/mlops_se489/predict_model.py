"""Model training and evaluation utilities for the retail demand forecasting pipeline.

This module provides helper functions for training scikit-learn models,
evaluating their performance, and selecting the champion model.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


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