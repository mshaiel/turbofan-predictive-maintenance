"""
Evaluation module for Remaining Useful Life (RUL) predictions.

Implements standard regression metrics (RMSE, MAE, MAPE) and the domain-specific
NASA Asymmetric Scoring Function from the PHM'08 prognostics competition.
"""

import logging
from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def nasa_score(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> float:
    """
    Compute the NASA PHM'08 asymmetric scoring function.

    Penalizes late predictions (y_pred > y_true, failure occurs sooner than predicted)
    exponentially more severely than early predictions (y_pred < y_true), reflecting
    the catastrophic operational cost of unexpected engine failure.

    Mathematical formulation:
        d = y_pred - y_true
        s_i = exp(-d / 13) - 1   if d < 0  (early prediction)
        s_i = exp(d / 10) - 1    if d >= 0 (late prediction)
        S = sum(s_i)

    Parameters
    ----------
    y_true : Union[np.ndarray, pd.Series]
        Ground-truth Remaining Useful Life values.
    y_pred : Union[np.ndarray, pd.Series]
        Predicted Remaining Useful Life values.

    Returns
    -------
    float
        Total NASA asymmetric penalty score.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    d = y_p - y_t

    score = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1.0,
        np.exp(d / 10.0) - 1.0
    )
    total_score = float(np.sum(score))
    return total_score


def evaluate_predictions(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for RUL forecaster.

    Parameters
    ----------
    y_true : Union[np.ndarray, pd.Series]
        Ground-truth RUL values.
    y_pred : Union[np.ndarray, pd.Series]
        Predicted RUL values.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys 'rmse', 'mae', 'mape', and 'nasa_score'.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))

    # Safe MAPE calculation avoiding division by zero
    non_zero_mask = y_t > 0
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs((y_t[non_zero_mask] - y_p[non_zero_mask]) / y_t[non_zero_mask])) * 100.0)
    else:
        mape = 0.0

    score = nasa_score(y_t, y_p)

    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 4),
        "nasa_score": round(score, 2),
    }

    logger.info("Evaluation metrics: %s", metrics)
    return metrics
