"""
MLP Baseline Implementation
Based on Algorithm 1, STEP 4
"""

from sklearn.neural_network import MLPRegressor
from src.config import (MLP_HIDDEN_UNITS, MLP_ACTIVATION,
                         MLP_LEARNING_RATE, MLP_MOMENTUM,
                         MLP_MAX_ITER, RANDOM_STATE)


def create_mlp_baseline():
    """Create MLP baseline as specified in Algorithm 1 STEP 4."""
    return MLPRegressor(
        hidden_layer_sizes=(MLP_HIDDEN_UNITS,),
        activation=MLP_ACTIVATION,
        solver='sgd',
        max_iter=MLP_MAX_ITER,
        random_state=RANDOM_STATE,
        learning_rate_init=MLP_LEARNING_RATE,
        momentum=MLP_MOMENTUM
    )