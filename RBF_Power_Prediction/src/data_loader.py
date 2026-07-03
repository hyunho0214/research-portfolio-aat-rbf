"""
Panama Electricity Demand Data Loader and Preprocessing
Loads and preprocesses real-world electricity demand data for forecasting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import os


def load_real_panama_data(filepath: str = "data/continuous_dataset.csv",
                          start_year: int = 2016,
                          end_year: int = 2020,
                          aggregate_to_daily: bool = False) -> pd.DataFrame:
    """
    Load real Panama electricity demand data from CSV.

    The dataset contains hourly electricity demand data with weather features.
    Default returns HOURLY data to match the paper's approach.

    Args:
        filepath: Path to the continuous dataset CSV.
        start_year: Start year for filtering.
        end_year: End year for filtering.
        aggregate_to_daily: If True, aggregate to daily (sum). Default False (hourly).

    Returns:
        DataFrame with datetime index and demand column.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found at '{filepath}'. "
            f"Please ensure the Panama dataset is available at this location."
        )

    df = pd.read_csv(filepath, parse_dates=['datetime'])

    # Filter by year range
    df['year'] = df['datetime'].dt.year
    df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

    # Set datetime as index
    df = df.set_index('datetime').sort_index()

    if aggregate_to_daily:
        # Aggregate to daily demand (sum of hourly demand)
        daily_demand = df['nat_demand'].resample('D').sum()
        result_df = pd.DataFrame({'demand': daily_demand})
        result_df = result_df.dropna()
    else:
        # Use hourly data directly (paper's approach)
        result_df = pd.DataFrame({'demand': df['nat_demand']})

    return result_df


def _create_sliding_windows_vectorized(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sliding window data X and y from a time series array in a vectorized manner.
    """
    n = len(data)
    if n <= window_size:
        return np.empty((0, window_size)), np.empty((0, 1))

    shape = (n - window_size, window_size)
    strides = (data.strides[0], data.strides[0])
    X = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    y = data[window_size:].reshape(-1, 1)

    return X, y


class DataScaler:
    """
    Separate scalers for X (inputs) and y (targets) as required by STEP 3.
    """
    def __init__(self):
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.y_mean: Optional[float] = None
        self.y_std: Optional[float] = None

    def fit_X(self, X: np.ndarray):
        """Fit scaler for inputs X (STEP 3)"""
        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0) + 1e-8

    def fit_y(self, y: np.ndarray):
        """Fit scaler for targets y (STEP 3)"""
        self.y_mean = np.mean(y)
        self.y_std = np.std(y) + 1e-8

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """Transform X using fitted scaler"""
        return (X - self.x_mean) / self.x_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Transform y using fitted scaler"""
        return (y - self.y_mean) / self.y_std

    def inverse_transform_X(self, X_norm: np.ndarray) -> np.ndarray:
        """Inverse transform X to original scale"""
        return X_norm * self.x_std + self.x_mean

    def inverse_transform_y(self, y_norm: np.ndarray) -> np.ndarray:
        """Inverse transform y to original scale"""
        return y_norm * self.y_std + self.y_mean


def preprocess_data(df: pd.DataFrame,
                    window_size: int = 10,
                    normalize: bool = True,
                    train_stats: Optional[Tuple] = None
                    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float]]:
    """
    Preprocess electricity demand data for RBF network (STEP 3).

    Fits scaler_X on X_train and scaler_y on y_train separately.
    Stores inverse transformations for later recovery.

    Args:
        df: DataFrame with 'demand' column.
        window_size: Number of consecutive time steps for input.
        normalize: Whether to normalize data.
        train_stats: If provided as (y_mean, y_std) [old 2-tuple] — use those for
                     normalization. If provided as (x_mean, x_std, y_mean, y_std)
                     [new 4-tuple] — use x_mean/x_std for X and y_mean/y_std for y.
                     If None, fit from df.

    Returns:
        Tuple of (X, y, x_mean, x_std, y_mean, y_std).
        x_mean/x_std are arrays, y_mean/y_std are scalars.
    """
    demand = df['demand'].values
    x_mean, x_std, y_mean, y_std = None, None, None, None

    if normalize:
        if train_stats is not None:
            if len(train_stats) == 2:
                # Old format: (y_mean, y_std) — shared normalization for X and y
                y_mean, y_std = train_stats
                demand_for_X = (demand - y_mean) / y_std
                X_raw, y_raw = _create_sliding_windows_vectorized(demand_for_X, window_size)
                # No separate X scaling in old format; x_mean/x_std stay None
                y_raw = (y_raw - y_mean) / y_std
            elif len(train_stats) == 4:
                # New format: (x_mean, x_std, y_mean, y_std)
                x_mean, x_std, y_mean, y_std = train_stats
                demand_for_X = (demand - y_mean) / y_std
                X_raw, y_raw = _create_sliding_windows_vectorized(demand_for_X, window_size)
                X = (X_raw - x_mean) / x_std
                y = (y_raw - y_mean) / y_std
                return X, y, x_mean, x_std, y_mean, y_std
            else:
                raise ValueError(f"train_stats must have 2 or 4 elements, got {len(train_stats)}")
        else:
            # Fit scaler_X and scaler_y separately (STEP 3)
            scaler_X = DataScaler()
            scaler_y = DataScaler()

            X_raw, y_raw = _create_sliding_windows_vectorized(demand, window_size)
            scaler_X.fit_X(X_raw)
            scaler_y.fit_y(y_raw)

            X = scaler_X.transform_X(X_raw)
            y = scaler_y.transform_y(y_raw)

            x_mean = scaler_X.x_mean
            x_std = scaler_X.x_std
            y_mean = scaler_y.y_mean
            y_std = scaler_y.y_std
    else:
        X, y = _create_sliding_windows_vectorized(demand, window_size)

    return X, y, x_mean, x_std, y_mean, y_std


def split_data_by_date(df: pd.DataFrame,
                       split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into training and testing sets based on a split date.

    Args:
        df: The DataFrame to split, with a DatetimeIndex.
        split_date: The date to split on (e.g., '2020-01-01'). Data before this
                    date becomes the training set.

    Returns:
        A tuple of (train_df, test_df).
    """
    train_df = df[df.index < pd.to_datetime(split_date)]
    test_df = df[df.index >= pd.to_datetime(split_date)]
    return train_df, test_df


def inverse_normalize(y_normalized: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Inverse transform normalized data back to original scale.

    Args:
        y_normalized: Normalized values.
        mean: Original mean.
        std: Original standard deviation.

    Returns:
        Original scale values.
    """
    return y_normalized * std + mean
