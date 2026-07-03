"""
Training and Evaluation Module for RBF Networks
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from src.rbf_network import RBFNetwork, RBFNetworkFactory


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Coefficient of Determination (R²)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def train_and_evaluate(network: RBFNetwork,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       n_kernels: int,
                       sigma_values: Optional[np.ndarray] = None,
                       sigma_strategy: str = "fixed") -> Dict:
    """
    Train and evaluate an RBF network.

    Args:
        network: RBF network instance
        X_train: Training inputs
        y_train: Training targets
        X_test: Test inputs
        y_test: Test targets
        n_kernels: Number of kernels
        sigma_values: Sigma values (optional)
        sigma_strategy: Sigma determination strategy

    Returns:
        Dictionary with training and test metrics
    """
    # Fit
    network.fit(X_train, y_train, n_kernels=n_kernels,
                sigma_values=sigma_values, sigma_strategy=sigma_strategy)

    # Predict
    y_train_pred = network.predict(X_train)
    y_test_pred = network.predict(X_test)

    # Calculate metrics
    train_mse = calculate_mse(y_train, y_train_pred)
    test_mse = calculate_mse(y_test, y_test_pred)
    train_r2 = calculate_r2(y_train, y_train_pred)
    test_r2 = calculate_r2(y_test, y_test_pred)

    return {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }


def run_kernel_count_experiment(X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  kernel_counts: List[int],
                                  mode: str = 'original',
                                  use_ridge: bool = False,
                                  ridge_alpha: float = 0.01) -> Dict:
    """
    Run experiment varying number of kernels.

    This reproduces Figure 4j and 4k from the paper.

    Args:
        X_train: Training inputs
        y_train: Training targets
        X_test: Test inputs
        y_test: Test targets
        kernel_counts: List of kernel counts to try (e.g., [1, 3, 10, 80])
        mode: 'original' or 'improved'
        use_ridge: Use Ridge regression
        ridge_alpha: Ridge alpha value

    Returns:
        Dictionary with results for each kernel count
    """
    n_inputs = X_train.shape[1]
    results = {
        'kernel_counts': kernel_counts,
        'test_mse': [],
        'test_r2': [],
        'train_mse': [],
        'train_r2': []
    }

    for n_kernels in kernel_counts:
        # Create network
        if mode == 'original':
            network = RBFNetworkFactory.create_paper_version(n_inputs)
        else:
            network = RBFNetworkFactory.create_improved_version(
                n_inputs, ridge_alpha=ridge_alpha
            )

        # Train and evaluate
        metrics = train_and_evaluate(
            network, X_train, y_train, X_test, y_test,
            n_kernels=n_kernels
        )

        results['test_mse'].append(metrics['test_mse'])
        results['test_r2'].append(metrics['test_r2'])
        results['train_mse'].append(metrics['train_mse'])
        results['train_r2'].append(metrics['train_r2'])

        print(f"  N={n_kernels:3d}: Test MSE={metrics['test_mse']:.4f}, "
              f"Test R²={metrics['test_r2']:.4f}")

    return results


def compare_paper_vs_improved(X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                kernel_counts: List[int]) -> Dict:
    """
    Compare original paper version vs improved version.

    Args:
        X_train: Training inputs
        y_train: Training targets
        X_test: Test inputs
        y_test: Test targets
        kernel_counts: List of kernel counts

    Returns:
        Dictionary with comparison results
    """
    print("\n=== Paper Version (Original) ===")
    paper_results = run_kernel_count_experiment(
        X_train, y_train, X_test, y_test,
        kernel_counts, mode='original'
    )

    print("\n=== Improved Version (Normalized + Ridge) ===")
    improved_results = run_kernel_count_experiment(
        X_train, y_train, X_test, y_test,
        kernel_counts, mode='improved'
    )

    return {
        'paper': paper_results,
        'improved': improved_results
    }
