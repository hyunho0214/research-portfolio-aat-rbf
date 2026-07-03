"""
STEP 2: Build supervised datasets via sliding window
Based on Algorithm 1, STEP 2
"""

import numpy as np


def create_dataset(sequence: np.ndarray, L: int):
    """
    Build supervised datasets via sliding window.

    Args:
        sequence: 1D time-series
        L: input window length (number of consecutive samples)

    Returns:
        X: matrix of L consecutive samples, shape (n_samples, L)
        y: next sample (target), shape (n_samples,)
    """
    X_list = []
    y_list = []

    for t in range(len(sequence) - L):
        X_list.append(sequence[t:t+L])
        y_list.append(sequence[t+L])

    return np.array(X_list), np.array(y_list)