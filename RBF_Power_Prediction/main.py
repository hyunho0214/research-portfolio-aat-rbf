"""
Main execution script for RBF Neural Network Power Prediction
Implements Algorithm 1 from Code 1.py precisely.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.data_loader import load_real_panama_data, split_data_by_date
from src.rbf_network import RBFNetwork, select_n_sigmas_from_range
from src.trainer import calculate_mse, calculate_r2


# =============================================================================
# STEP 5: Precompute experimental sigma candidates
# Experimental sigma values from AAT devices: sigma_min=2.9, sigma_max=19.8
# =============================================================================
SIGMA_MIN = 2.9
SIGMA_MAX = 19.8
C_TOTAL_PANAMA = 200   # Total RBF centers for Panama data


def create_dataset(sequence: np.ndarray, L: int):
    """
    Build supervised datasets via sliding window (STEP 2).

    Args:
        sequence: 1D time-series
        L: input window length

    Returns:
        X_list: matrix of L consecutive samples
        y_list: next sample
    """
    X_list = []
    y_list = []
    for t in range(len(sequence) - L):
        X_list.append(sequence[t:t+L])
        y_list.append(sequence[t+L])
    return np.array(X_list), np.array(y_list)


def main():
    """
    Main function implementing Algorithm 1 from Code 1.py.
    """
    print("=" * 60)
    print("RBF Neural Network for Power Demand Prediction")
    print("Algorithm 1: Multi-Gaussian RBF forecasting")
    print("=" * 60)

    # ==========================================================================
    # Load Panama data
    # ==========================================================================
    DATA_FILEPATH = "data/continuous_dataset.csv"

    print(f"\nLoading Panama electricity data from '{DATA_FILEPATH}'...")
    try:
        df = load_real_panama_data(
            filepath=DATA_FILEPATH,
            start_year=2016,
            end_year=2020
        )
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Demand range: {df['demand'].min():.2f} - {df['demand'].max():.2f} MW")
    except FileNotFoundError as e:
        print(e)
        print("\nPlease ensure the dataset is available in the 'data' directory.")
        return

    # ==========================================================================
    # STEP 1: Train–test split (time-based) — 80/20
    # ==========================================================================
    L = 10  # input window length
    train_df, test_df = split_data_by_date(df, split_date='2020-01-01')
    print(f"\nTrain date range: {train_df.index[0]} to {train_df.index[-1]}")
    print(f"Test date range: {test_df.index[0]} to {test_df.index[-1]}")

    # ==========================================================================
    # STEP 2: Build supervised datasets via sliding window
    # ==========================================================================
    s_train = train_df['demand'].values
    s_test = test_df['demand'].values

    X_train_raw, y_train_raw = create_dataset(s_train, L)
    X_test_raw, y_test_raw = create_dataset(s_test, L)

    print(f"\nTraining samples: {len(X_train_raw)}")
    print(f"Test samples: {len(X_test_raw)}")

    # ==========================================================================
    # STEP 3: Standardization — scaler_X for X, scaler_y for y (separate)
    # ==========================================================================
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train_raw)
    y_train = scaler_y.fit_transform(y_train_raw.reshape(-1)).reshape(-1, 1)

    X_test = scaler_X.transform(X_test_raw)
    y_test = scaler_y.transform(y_test_raw.reshape(-1)).reshape(-1, 1)

    print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # ==========================================================================
    # STEP 4: Train MLP baseline (hidden=200, logistic sigmoid, SGD)
    # ==========================================================================
    print("\n--- MLP Baseline (STEP 4) ---")
    mlp = MLPRegressor(
        hidden_layer_sizes=(200,),
        activation='logistic',
        solver='sgd',
        max_iter=1000,
        random_state=42,
        learning_rate_init=0.01,
        momentum=0.9
    )
    mlp.fit(X_train, y_train.ravel())

    y_pred_mlp_raw = mlp.predict(X_test)
    y_pred_mlp = scaler_y.inverse_transform(y_pred_mlp_raw.reshape(-1)).reshape(-1, 1)
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1)).reshape(-1, 1)

    mlp_mse = calculate_mse(y_test_orig, y_pred_mlp)
    mlp_mae = np.mean(np.abs(y_test_orig - y_pred_mlp))
    mlp_r2 = calculate_r2(y_test_orig, y_pred_mlp)

    print(f"MLP Test MSE: {mlp_mse:.4f}")
    print(f"MLP Test MAE: {mlp_mae:.4f}")
    print(f"MLP Test R²: {mlp_r2:.4f}")

    # ==========================================================================
    # STEP 5: Precompute experimental sigma candidates
    # sigma range from AAT device measurements: [2.9, 19.8]
    # ==========================================================================
    sigma_candidates = np.linspace(SIGMA_MIN, SIGMA_MAX, 1000)

    # ==========================================================================
    # STEP 6: Loop over N (number of distinct sigma values)
    # ==========================================================================
    N_values = [1, 3, 10, 20, 40, 60, 80]

    RBF_metrics = []
    best_mse = np.inf
    best_N = None

    print("\n--- RBF Network Results (STEP 6) ---")

    for N in N_values:
        # STEP 6.1: Select N distinct widths from experimental distribution
        sigma_set = select_n_sigmas_from_range(sigma_candidates, N)

        # STEP 6.2: Determine RBF centers via k-means clustering on training inputs
        kmeans = KMeans(n_clusters=N, random_state=42, n_init=10)
        kmeans.fit(X_train)
        cluster_centers = kmeans.cluster_centers_

        # STEP 6.3: Expand to C_total centers with assigned widths
        # Replicate each cluster center based on C_total/N distribution
        base_count = C_TOTAL_PANAMA // N
        extra_count = C_TOTAL_PANAMA % N

        centers_list = []
        sigma_list = []

        for k in range(N):
            count_k = base_count + (1 if k < extra_count else 0)
            for _ in range(count_k):
                centers_list.append(cluster_centers[k])
                sigma_list.append(sigma_set[k])

        centers_array = np.array(centers_list)  # shape: (C_TOTAL_PANAMA, L)
        sigma_array = np.array(sigma_list)        # shape: (C_TOTAL_PANAMA,)

        # STEP 6.4: Train RBF network
        # Construct design matrix Φ_train[n, j] = exp(-||X_train[n] - c[j]||² / (2σ[j]²))
        distances = np.linalg.norm(
            X_train[:, np.newaxis, :] - centers_array[np.newaxis, :, :],
            axis=2
        )
        Phi_train = np.exp(-(distances ** 2) / (2 * sigma_array ** 2))

        # Augment Φ_train with bias column of ones (STEP 6.4)
        Phi_train_bias = np.hstack([Phi_train, np.ones((Phi_train.shape[0], 1))])

        # Solve for weights w in least-squares sense
        w = np.linalg.lstsq(Phi_train_bias, y_train, rcond=None)[0]

        # STEP 6.5: Evaluate RBF network
        distances_test = np.linalg.norm(
            X_test[:, np.newaxis, :] - centers_array[np.newaxis, :, :],
            axis=2
        )
        Phi_test = np.exp(-(distances_test ** 2) / (2 * sigma_array ** 2))
        Phi_test_bias = np.hstack([Phi_test, np.ones((Phi_test.shape[0], 1))])

        y_pred_rbf_norm = Phi_test_bias @ w
        y_pred_rbf = scaler_y.inverse_transform(y_pred_rbf_norm.reshape(-1)).reshape(-1, 1)

        mse = calculate_mse(y_test_orig, y_pred_rbf)
        mae = np.mean(np.abs(y_test_orig - y_pred_rbf))
        r2 = calculate_r2(y_test_orig, y_pred_rbf)

        RBF_metrics.append({
            'N': N,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'y_pred': y_pred_rbf
        })

        print(f"  N={N:3d}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # STEP 6.6: Track best-performing N
        if mse < best_mse:
            best_mse = mse
            best_N = N

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

    # Paper reference: N=1 MSE=9.8e3, R²=0.41; N=80 MSE=1.2e2, R²=0.94
    print("\n--- Comparison with Paper Reference ---")
    print("Paper: N=1  MSE=9.8e3 MW², R²=0.41")
    print("Paper: N=80 MSE=1.2e2 MW², R²=0.94")
    print("\nOur results (best match):")
    for m in RBF_metrics:
        if m['N'] in [1, 80]:
            print(f"  N={m['N']}: MSE={m['mse']:.4f}, R²={m['r2']:.4f}")


if __name__ == "__main__":
    main()
