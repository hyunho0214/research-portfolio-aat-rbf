"""Data preparation helpers that preserve chronological time-series order."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class StandardizedData:
    """Container for train-fitted standardization artifacts."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    x_scaler: StandardScaler
    y_scaler: StandardScaler


def load_electricity_series(
    data_path: str | Path,
    datetime_column: str = "datetime",
    target_column: str = "nat_demand",
) -> pd.DataFrame:
    """Load, sort, and validate the hourly electricity demand series."""
    frame = pd.read_csv(data_path)
    required = {datetime_column, target_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame[[datetime_column, target_column]].copy()
    frame[datetime_column] = pd.to_datetime(frame[datetime_column])
    frame = frame.sort_values(datetime_column).reset_index(drop=True)
    if frame[datetime_column].duplicated().any():
        raise ValueError("Duplicate datetime values were found.")
    if frame[target_column].isna().any():
        raise ValueError(f"Missing values were found in {target_column}.")
    return frame


def split_series_by_year(
    frame: pd.DataFrame,
    train_years: tuple[int, int] = (2016, 2019),
    test_year: int = 2020,
    datetime_column: str = "datetime",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by calendar year without shuffling samples."""
    years = frame[datetime_column].dt.year
    train_mask = (years >= train_years[0]) & (years <= train_years[1])
    test_mask = years == test_year
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    test_frame = frame.loc[test_mask].reset_index(drop=True)
    if train_frame.empty or test_frame.empty:
        raise ValueError("Calendar split produced an empty train or test set.")
    return train_frame, test_frame


def chronological_fraction_split(
    values: np.ndarray,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an ordered array into train and test portions."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    split_index = int(len(values) * train_fraction)
    if split_index <= 0 or split_index >= len(values):
        raise ValueError("train_fraction produced an empty split.")
    return values[:split_index], values[split_index:]


def make_sliding_windows(
    values: np.ndarray,
    window_length: int,
    target_width: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create supervised samples from a 1D or 2D ordered sequence."""
    if window_length <= 0:
        raise ValueError("window_length must be positive.")
    if target_width <= 0:
        raise ValueError("target_width must be positive.")

    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        series = values.reshape(-1, 1)
    elif values.ndim == 2:
        series = values
    else:
        raise ValueError("values must be 1D or 2D.")

    n_samples = len(series) - window_length
    if n_samples <= 0:
        raise ValueError("Not enough samples to create sliding windows.")

    x_rows = []
    y_rows = []
    for start in range(n_samples):
        stop = start + window_length
        # Flattening keeps Duffing [x, v] windows compatible with RBF vectors.
        x_rows.append(series[start:stop].reshape(-1))
        y_rows.append(series[stop, :target_width])
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)


def standardize_train_test(
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
) -> StandardizedData:
    """Fit scalers on train only, then transform train and test arrays."""
    y_train_2d = np.asarray(y_train_raw, dtype=float)
    y_test_2d = np.asarray(y_test_raw, dtype=float)
    if y_train_2d.ndim == 1:
        y_train_2d = y_train_2d.reshape(-1, 1)
        y_test_2d = y_test_2d.reshape(-1, 1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(np.asarray(x_train_raw, dtype=float))
    x_test = x_scaler.transform(np.asarray(x_test_raw, dtype=float))
    y_train = y_scaler.fit_transform(y_train_2d)
    y_test = y_scaler.transform(y_test_2d)
    return StandardizedData(x_train, x_test, y_train, y_test, x_scaler, y_scaler)
