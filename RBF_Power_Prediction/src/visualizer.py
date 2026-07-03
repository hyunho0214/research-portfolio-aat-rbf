"""
Visualization Module - Reproduces Figure 4j and 4k from the paper
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_mse_vs_kernels(results: Dict,
                          title: str = "MSE vs Number of Kernels",
                          save_path: Optional[str] = None,
                          paper_reference: Optional[Dict] = None):
    """
    Plot MSE vs number of kernels (reproduces Figure 4j).

    Args:
        results: Dictionary with 'kernel_counts' and 'test_mse'
        title: Plot title
        save_path: Path to save figure
        paper_reference: Dict with 'n' and 'mse' for paper reference points
    """
    plt.figure(figsize=(10, 6))

    kernel_counts = results['kernel_counts']
    test_mse = results['test_mse']

    plt.plot(kernel_counts, test_mse, 'b-o', linewidth=2, markersize=8, label='RBF Network')

    # Paper reference points
    if paper_reference is not None:
        plt.scatter(paper_reference['n'], paper_reference['mse'],
                   color='red', s=100, zorder=5, label='Paper Reference')

    plt.xlabel('Number of Kernels (N)', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_r2_vs_kernels(results: Dict,
                         title: str = "R² vs Number of Kernels",
                         save_path: Optional[str] = None,
                         paper_reference: Optional[Dict] = None):
    """
    Plot R² vs number of kernels (reproduces Figure 4k).

    Args:
        results: Dictionary with 'kernel_counts' and 'test_r2'
        title: Plot title
        save_path: Path to save figure
        paper_reference: Dict with 'n' and 'r2' for paper reference points
    """
    plt.figure(figsize=(10, 6))

    kernel_counts = results['kernel_counts']
    test_r2 = results['test_r2']

    plt.plot(kernel_counts, test_r2, 'g-o', linewidth=2, markersize=8, label='RBF Network')

    # Paper reference points
    if paper_reference is not None:
        plt.scatter(paper_reference['n'], paper_reference['r2'],
                   color='red', s=100, zorder=5, label='Paper Reference')

    plt.xlabel('Number of Kernels (N)', fontsize=12)
    plt.ylabel('Coefficient of Determination (R²)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_figure_4jk(results: Dict,
                      save_dir: Optional[str] = None,
                      paper_reference: Optional[Dict] = None):
    """
    Reproduce both Figure 4j and 4k from the paper.

    Args:
        results: Dictionary with kernel count experiment results
        save_dir: Directory to save figures
        paper_reference: Reference points from paper (optional)
    """
    if save_dir:
        mse_path = f"{save_dir}/figure_4j_mse.png"
        r2_path = f"{save_dir}/figure_4k_r2.png"
    else:
        mse_path = None
        r2_path = None

    plot_mse_vs_kernels(
        results,
        title="Electricity Demand Forecasting: MSE vs Number of Kernels",
        save_path=mse_path,
        paper_reference=paper_reference
    )

    plot_r2_vs_kernels(
        results,
        title="Electricity Demand Forecasting: R² vs Number of Kernels",
        save_path=r2_path,
        paper_reference=paper_reference
    )


def plot_prediction_comparison(y_true: np.ndarray,
                                 y_pred_dict: Dict[str, np.ndarray],
                                 start_idx: int = 0,
                                 n_points: int = 100,
                                 save_path: Optional[str] = None):
    """
    Plot comparison of predictions from different configurations.

    Args:
        y_true: True values
        y_pred_dict: Dictionary of {name: predictions}
        start_idx: Starting index for plotting
        n_points: Number of points to plot
        save_path: Path to save figure
    """
    plt.figure(figsize=(14, 6))

    end_idx = min(start_idx + n_points, len(y_true))
    x_range = np.arange(start_idx, end_idx)

    plt.plot(x_range, y_true[start_idx:end_idx], 'k-', linewidth=2, label='True', alpha=0.7)

    colors = ['r', 'g', 'b', 'm', 'c']
    for i, (name, y_pred) in enumerate(y_pred_dict.items()):
        plt.plot(x_range, y_pred[start_idx:end_idx], '-', color=colors[i % len(colors)],
                linewidth=1.5, label=name, alpha=0.7)

    plt.xlabel('Time Index', fontsize=12)
    plt.ylabel('Electricity Demand (Normalized)', fontsize=12)
    plt.title('Prediction Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_comparison_paper_vs_improved(paper_results: Dict,
                                        improved_results: Dict,
                                        save_dir: Optional[str] = None):
    """
    Plot comparison between paper and improved versions.

    Args:
        paper_results: Results from paper version
        improved_results: Results from improved version
        save_dir: Directory to save figures
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    kernel_counts = paper_results['kernel_counts']

    # MSE comparison
    axes[0].plot(kernel_counts, paper_results['test_mse'], 'b-o', label='Paper Version', linewidth=2)
    axes[0].plot(kernel_counts, improved_results['test_mse'], 'r--s', label='Improved Version', linewidth=2)
    axes[0].set_xlabel('Number of Kernels (N)')
    axes[0].set_ylabel('Test MSE')
    axes[0].set_title('MSE Comparison')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # R² comparison
    axes[1].plot(kernel_counts, paper_results['test_r2'], 'b-o', label='Paper Version', linewidth=2)
    axes[1].plot(kernel_counts, improved_results['test_r2'], 'r--s', label='Improved Version', linewidth=2)
    axes[1].set_xlabel('Number of Kernels (N)')
    axes[1].set_ylabel('Test R²')
    axes[1].set_title('R² Comparison')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/comparison_paper_vs_improved.png", dpi=150)
        print(f"Figure saved to {save_dir}/comparison_paper_vs_improved.png")

    plt.show()
