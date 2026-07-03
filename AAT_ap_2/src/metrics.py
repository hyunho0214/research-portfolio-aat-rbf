"""Metric helpers for forecasting and Duffing reconstruction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute scalar forecasting metrics in original physical units."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def duffing_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute per-state reconstruction metrics for x and v."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 2 or y_true.shape[1] != 2:
        raise ValueError("Duffing targets must have shape (samples, 2).")
    return {
        "mse_x": float(mean_squared_error(y_true[:, 0], y_pred[:, 0])),
        "mse_v": float(mean_squared_error(y_true[:, 1], y_pred[:, 1])),
        "r2_x": float(r2_score(y_true[:, 0], y_pred[:, 0])),
        "r2_v": float(r2_score(y_true[:, 1], y_pred[:, 1])),
    }
