"""
Panama Electricity Demand Data Loader
STEP 1: Train-test split (time-based)
"""

import pandas as pd
from typing import Tuple
import os
from src.config import (DATA_FILEPATH, START_YEAR, END_YEAR,
                         SPLIT_DATE, MIN_TRAIN_SAMPLES, MIN_TEST_SAMPLES)


def load_panama_data(filepath: str = None) -> pd.DataFrame:
    """
    Load Panama electricity demand data from CSV.

    Args:
        filepath: Path to CSV. Defaults to config DATA_FILEPATH.

    Returns:
        DataFrame with datetime index and 'demand' column

    Raises:
        FileNotFoundError: If data file not found
        ValueError: If required columns missing
    """
    if filepath is None:
        filepath = DATA_FILEPATH

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found at '{filepath}'. "
            f"Please ensure the dataset is available."
        )

    # Read CSV - only keep lines that look like valid data
    # (datetime header or lines starting with valid year 2015-2020)
    good_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('datetime'):
                good_lines.append(line)
            elif len(line) > 4 and line[0:4].isdigit():
                year = int(line[0:4])
                if 2015 <= year <= 2020:
                    good_lines.append(line)

    # Parse from good lines only
    from io import StringIO
    df = pd.read_csv(StringIO(''.join(good_lines)))

    # Validate required columns
    if 'nat_demand' not in df.columns:
        raise ValueError(f"Required column 'nat_demand' not found. Available: {df.columns.tolist()}")

    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Filter by year range BEFORE setting index
    df['year'] = df['datetime'].dt.year
    df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)]

    # Set datetime as index and sort
    df = df.set_index('datetime').sort_index()

    return pd.DataFrame({'demand': df['nat_demand']})


def split_data_by_date(df: pd.DataFrame, split_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    STEP 1: Train-test split (time-based)

    Chronological split at split_date (fixed date, not ratio):
        s_train = before split_date
        s_test  = after split_date

    Args:
        df: DataFrame with DatetimeIndex
        split_date: Date to split on. Defaults to config SPLIT_DATE.

    Returns:
        Tuple of (train_df, test_df)

    Raises:
        ValueError: If split results in too few samples
    """
    if split_date is None:
        split_date = SPLIT_DATE

    train_df = df[df.index < pd.to_datetime(split_date)]
    test_df = df[df.index >= pd.to_datetime(split_date)]

    if len(train_df) < MIN_TRAIN_SAMPLES:
        raise ValueError(f"Train samples ({len(train_df)}) < MIN_TRAIN_SAMPLES ({MIN_TRAIN_SAMPLES})")
    if len(test_df) < MIN_TEST_SAMPLES:
        raise ValueError(f"Test samples ({len(test_df)}) < MIN_TEST_SAMPLES ({MIN_TEST_SAMPLES})")

    return train_df, test_df