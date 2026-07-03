"""
Utility functions for RBF2 project.
Contains metrics computation and sigma selection.
"""

import numpy as np


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R-squared (coefficient of determination)
    R² = 1 - SS_res / SS_tot
    Edge case: zero variance y_true -> return 0
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def select_n_sigmas_from_range(sigma_candidates: np.ndarray, N: int) -> np.ndarray:
    """
    STEP 6.1: Select N distinct sigma values via log-spacing.
    Uses logarithmic spacing between sigma_min and sigma_max,
    matching the paper's LOG_SPACED_SIGMAS specification.
    """
    if N <= 0:
        return np.array([])
    if N == 1:
        return np.array([np.sqrt(sigma_candidates.min() * sigma_candidates.max())])

    sigma_set = np.logspace(
        np.log10(sigma_candidates.min()),
        np.log10(sigma_candidates.max()),
        N
    )
    return sigma_set


def check_condition_number(Phi: np.ndarray, threshold: float = 1e10) -> bool:
    """Check if design matrix is ill-conditioned."""
    try:
        cond = np.linalg.cond(Phi)
        return cond < threshold
    except np.linalg.LinAlgError:
        return False