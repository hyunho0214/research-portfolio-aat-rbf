"""
Figure 4h-k Visualization Script
Generates:
- h: Date vs demand for N=1,3,10,80 (actual vs predicted)
- i: 4 subplots for N=1,3,10,80, days 0-15 comparison
- j: MSE vs N (log scale)
- k: R² vs N
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_panama_data, split_data_by_date
from src.data_utils import create_dataset
from src.config import L, C_TOTAL, N_VALUES, get_sigma_candidates
from src.utils import compute_mse, compute_r2, select_n_sigmas_from_range
from src.rbf_network import RBFNetwork


def run_rbf_analysis():
    """Run full RBF analysis and return results."""
    df = load_panama_data()
    train_df, test_df = split_data_by_date(df)

    s_train = train_df['demand'].values
    s_test = test_df['demand'].values

    X_train_raw, y_train_raw = create_dataset(s_train, L)
    X_test_raw, y_test_raw = create_dataset(s_test, L)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train_raw)
    y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))
    X_test = scaler_X.transform(X_test_raw)
    y_test = scaler_y.transform(y_test_raw.reshape(-1, 1))

    y_test_orig = scaler_y.inverse_transform(y_test)

    sigma_candidates = get_sigma_candidates()

    results = {'y_test': y_test_orig, 'predictions': {}, 'metrics': []}
    test_dates = test_df.index[L:]

    for N in N_VALUES:
        sigma_set = select_n_sigmas_from_range(sigma_candidates, N)
        rbf = RBFNetwork.create_with_clustering(X_train, N, C_TOTAL, sigma_set)
        rbf.fit(X_train, y_train)
        y_pred_norm = rbf.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_norm)

        mse = compute_mse(y_test_orig, y_pred)
        mae = np.mean(np.abs(y_test_orig - y_pred))
        r2 = compute_r2(y_test_orig, y_pred)

        results['predictions'][N] = y_pred
        results['metrics'].append({'N': N, 'mse': mse, 'mae': mae, 'r2': r2})

    return results, test_dates


def plot_figure_4(results, test_dates, save_path=None):
    """Generate Figure 4h-k style plots."""
    N_values_plot = [1, 3, 10, 80]
    metrics_df = pd.DataFrame(results['metrics'])

    fig = plt.figure(figsize=(16, 14))

    # =========================================================================
    # h: Date vs demand for N=1,3,10,80
    # =========================================================================
    ax_h = fig.add_subplot(3, 2, 1)
    y_test = results['y_test'].ravel()

    # Plot actual values (black line)
    ax_h.plot(test_dates, y_test, 'k-', linewidth=1.5, label='Actual', alpha=0.8)

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    for idx, N in enumerate(N_values_plot):
        if N in results['predictions']:
            y_pred = results['predictions'][N].ravel()
            ax_h.plot(test_dates, y_pred, color=colors[idx],
                     linewidth=1.0, label=f'N={N}', alpha=0.7)

    ax_h.set_xlabel('Date', fontsize=11)
    ax_h.set_ylabel('Electricity Demand (MW)', fontsize=11)
    ax_h.set_title('(h) Electricity Demand Prediction (Test Period)', fontsize=12, fontweight='bold')
    ax_h.legend(loc='upper right')
    ax_h.tick_params(axis='x', rotation=30)
    ax_h.grid(True, alpha=0.3)

    # =========================================================================
    # i: 4 subplots for N=1,3,10,80, days 0-15
    # =========================================================================
    days_to_show = 15 * 24  # 15 days of hourly data

    for idx, N in enumerate(N_values_plot):
        ax_i = fig.add_subplot(3, 4, 9 + idx)  # Third row, 4 subplots

        y_test_short = results['y_test'].ravel()[:days_to_show]
        y_pred_short = results['predictions'][N].ravel()[:days_to_show]
        days = np.arange(len(y_test_short)) / 24  # Convert to days

        # Actual (solid black)
        ax_i.plot(days, y_test_short, 'k-', linewidth=1.5, label='Actual')
        # Predicted (dashed)
        ax_i.plot(days, y_pred_short, '--', color=colors[idx], linewidth=1.2, label=f'N={N}')

        ax_i.set_xlabel('Days', fontsize=9)
        ax_i.set_ylabel('Demand (MW)', fontsize=9)
        ax_i.set_title(f'N={N}', fontsize=10)
        ax_i.legend(loc='upper right', fontsize=7)
        ax_i.grid(True, alpha=0.3)
        ax_i.set_xlim([0, 15])

    # =========================================================================
    # j: MSE vs N (log scale)
    # =========================================================================
    ax_j = fig.add_subplot(3, 2, 3)

    mse_values = metrics_df['mse'].values
    n_values = metrics_df['N'].values

    ax_j.semilogy(n_values, mse_values, 'bo-', linewidth=2, markersize=8)
    ax_j.set_xlabel('Number of Gaussians (N)', fontsize=11)
    ax_j.set_ylabel('MSE (MW²)', fontsize=11)
    ax_j.set_title('(j) MSE vs Number of Gaussians', fontsize=12, fontweight='bold')
    ax_j.set_ylim([1e2, 1e5])
    ax_j.grid(True, alpha=0.3, which='both')
    ax_j.set_xticks(n_values)

    # Paper reference points
    ax_j.semilogy([1, 80], [9800, 120], 'ks', markersize=8, alpha=0.5, label='Paper')
    ax_j.legend()

    # =========================================================================
    # k: R² vs N
    # =========================================================================
    ax_k = fig.add_subplot(3, 2, 4)

    r2_values = metrics_df['r2'].values

    ax_k.plot(n_values, r2_values, 'ro-', linewidth=2, markersize=8)
    ax_k.set_xlabel('Number of Gaussians (N)', fontsize=11)
    ax_k.set_ylabel('R² Score', fontsize=11)
    ax_k.set_title('(k) R² vs Number of Gaussians', fontsize=12, fontweight='bold')
    ax_k.set_ylim([0.0, 1.0])
    ax_k.grid(True, alpha=0.3)
    ax_k.set_xticks(n_values)

    # Paper reference points
    ax_k.plot([1, 80], [0.41, 0.94], 'ks', markersize=8, alpha=0.5, label='Paper')
    ax_k.legend()

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.suptitle('RBF Network Performance: Multi-Gaussian Analysis', fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    print("Running RBF analysis for Figure 4...")
    results, test_dates = run_rbf_analysis()
    print("Generating Figure 4h-k...")
    plot_figure_4(results, test_dates, save_path='figure_4.png')
    print("Done!")