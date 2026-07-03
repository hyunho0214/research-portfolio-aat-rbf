"""
Main execution script for RBF2 project.
Implements Algorithm 1 from Code 1.py with Codex feedback applied.

Changes from original plan (Codex feedback):
- Ridge regularization for numerical stability
- Edge-case guards
- Clear split rule (fixed date, not ratio)
- Reproducibility controls (random_state)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from src.config import (L, C_TOTAL, N_VALUES, get_sigma_candidates, RANDOM_STATE)
from src.data_loader import load_panama_data, split_data_by_date
from src.data_utils import create_dataset
from src.utils import compute_mse, compute_mae, compute_r2, select_n_sigmas_from_range
from src.rbf_network import RBFNetwork
from src.mlp_baseline import create_mlp_baseline


def main():
    """Main function implementing Algorithm 1 from Code 1.py."""
    print("=" * 60)
    print("RBF2 - Multi-Gaussian RBF forecasting")
    print("Algorithm 1: Based on Code 1.py with Codex improvements")
    print("=" * 60)

    # ==========================================================================
    # STEP 1: Load data and train-test split (time-based)
    # ==========================================================================
    print("\n[STEP 1] Loading Panama data...")
    df = load_panama_data()
    print(f"  Data shape: {df.shape}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Demand range: {df['demand'].min():.2f} - {df['demand'].max():.2f} MW")

    train_df, test_df = split_data_by_date(df)
    print(f"  Train: {len(train_df)} samples ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"  Test:  {len(test_df)} samples ({test_df.index[0]} to {test_df.index[-1]})")

    # ==========================================================================
    # STEP 2: Build supervised datasets via sliding window
    # ==========================================================================
    print("\n[STEP 2] Building sliding window datasets...")
    s_train = train_df['demand'].values
    s_test = test_df['demand'].values

    X_train_raw, y_train_raw = create_dataset(s_train, L)
    X_test_raw, y_test_raw = create_dataset(s_test, L)

    # Edge-case: verify post-window sample counts
    if len(X_train_raw) < max(N_VALUES):
        raise ValueError(f"Train windows ({len(X_train_raw)}) < max N_values ({max(N_VALUES)})")
    if len(X_test_raw) < 1:
        raise ValueError(f"No test windows available (test samples={len(s_test)}, L={L})")

    print(f"  X_train: {X_train_raw.shape}, y_train: {y_train_raw.shape}")
    print(f"  X_test:  {X_test_raw.shape}, y_test:  {y_test_raw.shape}")

    # ==========================================================================
    # STEP 3: Standardization (separate scalers for X and y)
    # ==========================================================================
    print("\n[STEP 3] Standardizing inputs and targets...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Ensure 2D input for StandardScaler (works with 1D via reshape)
    X_train = scaler_X.fit_transform(X_train_raw)
    y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))

    X_test = scaler_X.transform(X_test_raw)
    y_test = scaler_y.transform(y_test_raw.reshape(-1, 1))
    print(f"  X_train scaled: mean={X_train.mean():.4f}, std={X_train.std():.4f}")
    print(f"  y_train scaled: mean={y_train.mean():.4f}, std={y_train.std():.4f}")

    # Store original scale for inverse transform (keep 2D)
    y_test_orig = scaler_y.inverse_transform(y_test)

    # ==========================================================================
    # STEP 4: Train MLP baseline
    # ==========================================================================
    print("\n[STEP 4] Training MLP baseline...")
    mlp = create_mlp_baseline()
    mlp.fit(X_train, y_train.ravel())

    y_pred_mlp_norm = mlp.predict(X_test).reshape(-1, 1)
    y_pred_mlp = scaler_y.inverse_transform(y_pred_mlp_norm)

    mlp_mse = compute_mse(y_test_orig, y_pred_mlp)
    mlp_mae = compute_mae(y_test_orig, y_pred_mlp)
    mlp_r2 = compute_r2(y_test_orig, y_pred_mlp)
    print(f"  MLP Test MSE: {mlp_mse:.4f}")
    print(f"  MLP Test MAE: {mlp_mae:.4f}")
    print(f"  MLP Test R²:  {mlp_r2:.4f}")

    # ==========================================================================
    # STEP 5: Precompute experimental sigma candidates
    # ==========================================================================
    print("\n[STEP 5] Precomputing sigma candidates...")
    sigma_candidates = get_sigma_candidates()
    print(f"  Sigma range: [{sigma_candidates.min():.4f}, {sigma_candidates.max():.4f}]")
    print(f"  Total candidates: {len(sigma_candidates)}")

    # ==========================================================================
    # STEP 6: Loop over N values
    # ==========================================================================
    print("\n[STEP 6] Testing RBF with different N values...")
    RBF_metrics = []
    best_mse = np.inf
    best_N = None
    best_pred = None

    for N in N_VALUES:
        # STEP 6.1: Select N distinct sigma values
        sigma_set = select_n_sigmas_from_range(sigma_candidates, N)

        # STEP 6.2-6.3: Create RBF network with clustering
        rbf = RBFNetwork.create_with_clustering(X_train, N, C_TOTAL, sigma_set)

        # STEP 6.4: Train RBF
        rbf.fit(X_train, y_train)

        # STEP 6.5: Evaluate RBF
        y_pred_rbf_norm = rbf.predict(X_test)
        y_pred_rbf = scaler_y.inverse_transform(y_pred_rbf_norm)

        mse = compute_mse(y_test_orig, y_pred_rbf)
        mae = compute_mae(y_test_orig, y_pred_rbf)
        r2 = compute_r2(y_test_orig, y_pred_rbf)

        RBF_metrics.append({'N': N, 'mse': mse, 'mae': mae, 'r2': r2, 'y_pred': y_pred_rbf})
        print(f"  N={N:3d}: MSE={mse:10.4f}, MAE={mae:8.4f}, R²={r2:.4f}")

        # STEP 6.6: Track best
        if mse < best_mse:
            best_mse = mse
            best_N = N
            best_pred = y_pred_rbf

    # ==========================================================================
    # STEP 7: Return results
    # ==========================================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Algorithm 1)")
    print("=" * 60)
    print(f"\nMLP Baseline: MSE={mlp_mse:.4f}, MAE={mlp_mae:.4f}, R²={mlp_r2:.4f}")
    print(f"\nBest RBF: N={best_N}, MSE={best_mse:.4f}")

    print("\n--- All RBF Metrics ---")
    print(f"{'N':>4}  {'MSE':>12}  {'MAE':>12}  {'R²':>8}")
    for m in RBF_metrics:
        print(f"{m['N']:4d}  {m['mse']:12.4f}  {m['mae']:12.4f}  {m['r2']:8.4f}")

    print("\n--- Comparison with Paper Reference ---")
    print("Paper: N=1  MSE=9.8e3 MW², R²=0.41")
    print("Paper: N=80 MSE=1.2e2 MW², R²=0.94")
    print("\nOur results:")
    for m in RBF_metrics:
        if m['N'] in [1, 80]:
            print(f"  N={m['N']}: MSE={m['mse']:.4f}, R²={m['r2']:.4f}")


if __name__ == "__main__":
    main()