"""
Configuration file for RBF2 project.
Contains all constants, paths, and reproducibility controls.
"""

import numpy as np
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILEPATH = DATA_DIR / "continuous_dataset.csv"

# Algorithm constants (from Code 1.py)
L = 10                      # Input window length
C_TOTAL = 200              # Total number of RBF centers
SIGMA_MIN = 2.9            # Experimental sigma range (AAT devices)
SIGMA_MAX = 19.8
N_VALUES = [1, 3, 10, 20, 40, 60, 80]  # N values to test (now with 5 years of train data)

# Data split - chronological 80/20
# Paper: Train 2016-2019, Test 2020
# Note: Actual data is from 2015-01-03 to 2020-06-27
SPLIT_DATE = '2020-01-01'
START_YEAR = 2016
END_YEAR = 2020

# MLP baseline parameters (STEP 4)
MLP_HIDDEN_UNITS = 200
MLP_ACTIVATION = 'logistic'
MLP_LEARNING_RATE = 0.01
MLP_MOMENTUM = 0.9
MLP_MAX_ITER = 1000

# Sigma candidates (STEP 5)
SIGMA_CANDIDATES_COUNT = 2000

# Ridge regularization for numerical stability (Codex feedback)
RIDGE_ALPHA = 1e-4

# Reproducibility (Codex feedback)
RANDOM_STATE = 42

# Edge-case thresholds (Codex feedback)
MIN_TRAIN_SAMPLES = 20
MIN_TEST_SAMPLES = 10


def get_sigma_candidates() -> np.ndarray:
    """STEP 5: Precompute experimental sigma candidates (log-spaced per paper)"""
    return np.logspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_CANDIDATES_COUNT)