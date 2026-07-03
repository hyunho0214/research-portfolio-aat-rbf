"""
Figure 4 Replication (h, i, j, k)
Precisely implements Algorithm 1 from Code 1.py with:
  - STEP 3: Separate scaler_X and scaler_y normalization
  - STEP 4: MLP baseline (hidden=200, logistic, SGD)
  - STEP 5: Experimental sigma range [2.9, 19.8]
  - STEP 6: Cluster-based sigma assignment + bias column
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch execution
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from src.data_loader import load_real_panama_data, split_data_by_date
from src.trainer import calculate_mse, calculate_r2
from src.rbf_network import select_n_sigmas_from_range

# ============================================================================
# Paper Parameters (Algorithm 1)
# ============================================================================
SIGMA_MIN = 2.9
SIGMA_MAX = 19.8
C_TOTAL = 200   # Total RBF centers (Panama)
L = 10           # Input window length
N_VALUES = [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80]

# Paper reference (original MW² scale)
PAPER_REF = {
    'N': [1, 80],
    'MSE': [9.8e3, 1.2e2],
    'R2': [0.41, 0.94]
}


def create_dataset(sequence: np.ndarray, L: int):
    """Build supervised datasets via sliding window (STEP 2 of Algorithm 1)."""
    X_list, y_list = [], []
    for t in range(len(sequence) - L):
        X_list.append(sequence[t:t+L])
        y_list.append(sequence[t+L])
    return np.array(X_list), np.array(y_list)


def train_rbf_algorithm1(X_train, y_train, X_test, N, sigma_min, sigma_max, C_total):
    """
    Train RBF network precisely following Algorithm 1 (Code 1.py).

    Steps:
      6.1: Select N distinct widths from [sigma_min, sigma_max] via quantiles
      6.2: Run k-means with N clusters on X_train
      6.3: Expand each cluster center to C_total with cluster-based sigma assignment
      6.4: Compute Φ design matrix with bias column; solve least-squares
      6.5: Predict and inverse-transform
    """
    # STEP 6.1: Select N distinct sigma values via quantile sampling
    sigma_candidates = np.linspace(sigma_min, sigma_max, 1000)
    sigma_set = select_n_sigmas_from_range(sigma_candidates, N)

    # STEP 6.2: Determine RBF centers via k-means with N clusters
    kmeans = KMeans(n_clusters=N, random_state=42, n_init=10)
    kmeans.fit(X_train)
    cluster_centers = kmeans.cluster_centers_

    # STEP 6.3: Expand to C_total centers with cluster-based sigma assignment
    base_count = C_total // N
    extra_count = C_total % N

    centers_list = []
    sigma_list = []
    for k in range(N):
        count_k = base_count + (1 if k < extra_count else 0)
        for _ in range(count_k):
            centers_list.append(cluster_centers[k])
            sigma_list.append(sigma_set[k])

    centers_array = np.array(centers_list)
    sigma_array = np.array(sigma_list)

    # STEP 6.4: Construct Φ_train design matrix + bias column
    distances = np.linalg.norm(
        X_train[:, np.newaxis, :] - centers_array[np.newaxis, :, :],
        axis=2
    )
    Phi_train = np.exp(-(distances ** 2) / (2 * sigma_array ** 2))
    Phi_train_bias = np.hstack([Phi_train, np.ones((Phi_train.shape[0], 1))])

    # Solve for weights in least-squares sense
    w = np.linalg.lstsq(Phi_train_bias, y_train, rcond=None)[0]

    # STEP 6.5: Evaluate on test set
    distances_test = np.linalg.norm(
        X_test[:, np.newaxis, :] - centers_array[np.newaxis, :, :],
        axis=2
    )
    Phi_test = np.exp(-(distances_test ** 2) / (2 * sigma_array ** 2))
    Phi_test_bias = np.hstack([Phi_test, np.ones((Phi_test.shape[0], 1))])

    y_pred_norm = Phi_test_bias @ w
    return y_pred_norm, sigma_set


# ============================================================================
# Load Data
# ============================================================================
print("Loading Panama data...")
df = load_real_panama_data(
    filepath='C:/Users/HYUNHO/Desktop/Panama/continuous dataset.csv',
    start_year=2016, end_year=2020
)
print(f"Data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# ============================================================================
# STEP 1: Time-based 80/20 train-test split
# ============================================================================
train_df, test_df = split_data_by_date(df, '2020-01-01')
print(f"\nTrain: {train_df.index[0]} to {train_df.index[-1]}")
print(f"Test:  {test_df.index[0]} to {test_df.index[-1]}")

s_train = train_df['demand'].values
s_test = test_df['demand'].values
test_demand_original = s_test.copy()
test_time_index = np.arange(len(test_demand_original))

# ============================================================================
# STEP 2: Build sliding-window datasets
# ============================================================================
X_train_raw, y_train_raw = create_dataset(s_train, L)
X_test_raw, y_test_raw = create_dataset(s_test, L)
print(f"\nSliding window dataset: X_train={X_train_raw.shape}, y_train={y_train_raw.shape}")
print(f"                        X_test={X_test_raw.shape}, y_test={y_test_raw.shape}")

# ============================================================================
# STEP 3: Separate standardization — scaler_X and scaler_y
# ============================================================================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train_raw)
y_train_2d = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))
y_train = y_train_2d.ravel()

X_test = scaler_X.transform(X_test_raw)
y_test_scaled_2d = scaler_y.transform(y_test_raw.reshape(-1, 1))
y_test_scaled = y_test_scaled_2d.ravel()

# y_test in original scale for metric calculation
y_test_original = scaler_y.inverse_transform(y_test_scaled_2d).ravel()

print(f"\nNormalization (STEP 3): scaler_X (L={L} dims), scaler_y (scalar)")

# ============================================================================
# STEP 4: MLP Baseline (hidden=200, logistic, SGD)
# ============================================================================
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

y_pred_mlp_norm = mlp.predict(X_test)
y_pred_mlp = scaler_y.inverse_transform(y_pred_mlp_norm.reshape(-1, 1)).ravel()

mlp_mse = calculate_mse(y_test_original, y_pred_mlp)
mlp_mae = np.mean(np.abs(y_test_original - y_pred_mlp))
mlp_r2 = calculate_r2(y_test_original, y_pred_mlp)

print(f"MLP  MSE={mlp_mse:.2f} MW², MAE={mlp_mae:.2f} MW, R²={mlp_r2:.4f}")

# ============================================================================
# STEP 5 + 6: Loop over N and train RBF networks
# ============================================================================
print("\n--- RBF Network Results (STEP 6) ---")
predictions = {}
metrics = {'N': [], 'MSE': [], 'MAE': [], 'R2': []}

for N in N_VALUES:
    y_pred_norm, sigma_set = train_rbf_algorithm1(
        X_train, y_train, X_test, N, SIGMA_MIN, SIGMA_MAX, C_TOTAL
    )

    # Inverse-transform to original scale
    y_pred_rbf = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

    # Metrics in original scale
    mse = calculate_mse(y_test_original, y_pred_rbf)
    mae = np.mean(np.abs(y_test_original - y_pred_rbf))
    r2 = calculate_r2(y_test_original, y_pred_rbf)

    predictions[N] = {
        'y_pred': y_pred_rbf,
        'sigma_set': sigma_set
    }
    metrics['N'].append(N)
    metrics['MSE'].append(mse)
    metrics['MAE'].append(mae)
    metrics['R2'].append(r2)

    print(f"  N={N:3d}: MSE={mse:10.2f} MW², MAE={mae:8.2f} MW, R²={r2:.4f}")

# ============================================================================
# Find best N
# ============================================================================
best_idx = np.argmin(metrics['MSE'])
best_N = metrics['N'][best_idx]
best_MSE = metrics['MSE'][best_idx]
best_R2 = metrics['R2'][best_idx]
print(f"\nBest RBF: N={best_N}, MSE={best_MSE:.2f} MW², R²={best_R2:.4f}")

# ============================================================================
# Figure 4h: Measured daily electricity demand for test year 2020
# ============================================================================
plt.figure(figsize=(14, 5))
plt.plot(test_time_index, test_demand_original, 'k-', linewidth=0.8, label='Measured')
plt.xlabel('Day of 2020', fontsize=12)
plt.ylabel('Daily Electricity Demand (MW)', fontsize=12)
plt.title('Figure 4h: Measured Daily Electricity Demand (Panama, 2020)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('C:/Users/HYUNHO/Desktop/RBF_Power_Prediction/Fig4h_Measured_Demand.png', dpi=150, bbox_inches='tight')
print("\nFigure 4h saved.")

# ============================================================================
# Figure 4i: 15-day prediction segments — compare N=1, 3, 10, 80
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
start_day = 50  # start of the 15-day window

for idx, N in enumerate([1, 3, 10, 80]):
    ax = axes[idx // 2, idx % 2]
    y_pred = predictions[N]['y_pred']

    # Measured demand — original scale (sliding window starts from day 0)
    days_measured = np.arange(start_day, start_day + 15)
    ax.plot(
        days_measured,
        test_demand_original[start_day:start_day + 15],
        'k-', linewidth=1.5, label='Measured', marker='o', markersize=3
    )

    # Predictions — only the portion starting from window_size offset
    # y_pred[i] corresponds to day i + window_size
    pred_start = start_day + L  # first predicted day in the window
    pred_days = np.arange(pred_start, pred_start + len(y_pred[start_day:]))
    ax.plot(
        pred_days,
        y_pred[start_day:],
        'r--', linewidth=1.5, label=f'N={N}', alpha=0.85
    )

    ax.set_xlabel('Day of 2020', fontsize=10)
    ax.set_ylabel('Demand (MW)', fontsize=10)
    ax.set_title(f'N = {N}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Figure 4i: 15-Day Prediction Segments with Varying Gaussian Kernels', fontsize=14)
plt.tight_layout()
plt.savefig('C:/Users/HYUNHO/Desktop/RBF_Power_Prediction/Fig4i_Prediction_Segments.png', dpi=150, bbox_inches='tight')
print("Figure 4i saved.")

# ============================================================================
# Figure 4j: MSE vs N (with MLP baseline)
# ============================================================================
plt.figure(figsize=(10, 6))
plt.plot(metrics['N'], metrics['MSE'], 'b-o', linewidth=2, markersize=8, label='RBF Network')

# MLP baseline horizontal line
plt.axhline(y=mlp_mse, color='orange', linestyle='--', linewidth=1.5,
            label=f'MLP Baseline (MSE={mlp_mse:.0f})')

# Paper reference
plt.scatter(PAPER_REF['N'], PAPER_REF['MSE'],
            color='red', s=150, zorder=5, label='Paper Reference', marker='s')

plt.xlabel('Number of Distinct σ Values (N)', fontsize=12)
plt.ylabel('Mean Squared Error (MW²)', fontsize=12)
plt.title('Figure 4j: MSE vs Number of Kernels', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('C:/Users/HYUNHO/Desktop/RBF_Power_Prediction/Fig4j_MSE_vs_N.png', dpi=150, bbox_inches='tight')
print("Figure 4j saved.")

# ============================================================================
# Figure 4k: R² vs N (with MLP baseline)
# ============================================================================
plt.figure(figsize=(10, 6))
plt.plot(metrics['N'], metrics['R2'], 'g-o', linewidth=2, markersize=8, label='RBF Network')

# MLP baseline horizontal line
plt.axhline(y=mlp_r2, color='orange', linestyle='--', linewidth=1.5,
            label=f'MLP Baseline (R²={mlp_r2:.3f})')

# Paper reference
plt.scatter(PAPER_REF['N'], PAPER_REF['R2'],
            color='red', s=150, zorder=5, label='Paper Reference', marker='s')

plt.xlabel('Number of Distinct σ Values (N)', fontsize=12)
plt.ylabel('Coefficient of Determination (R²)', fontsize=12)
plt.title('Figure 4k: R² vs Number of Kernels', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('C:/Users/HYUNHO/Desktop/RBF_Power_Prediction/Fig4k_R2_vs_N.png', dpi=150, bbox_inches='tight')
print("Figure 4k saved.")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("SUMMARY: Figure 4 Replication (Algorithm 1)")
print("="*60)
print(f"Sigma range: [{SIGMA_MIN}, {SIGMA_MAX}]  (AAT device measurements)")
print(f"Total RBF centers C_total: {C_TOTAL}")
print(f"Input window L: {L}")
print()
print(f"{'N':>4}  {'MSE (MW²)':>12}  {'MAE (MW)':>10}  {'R²':>8}")
print("-" * 42)
for i, N in enumerate(metrics['N']):
    print(f"{N:4d}  {metrics['MSE'][i]:12.2f}  {metrics['MAE'][i]:10.2f}  {metrics['R2'][i]:8.4f}")
print()
print(f"MLP Baseline: MSE={mlp_mse:.2f} MW², R²={mlp_r2:.4f}")
print()
print("Paper Reference:")
print(f"  N={1}:  MSE={PAPER_REF['MSE'][0]:.2e} MW², R²={PAPER_REF['R2'][0]:.2f}")
print(f"  N={80}: MSE={PAPER_REF['MSE'][1]:.2e} MW², R²={PAPER_REF['R2'][1]:.2f}")
print()
print("Figures saved to RBF_Power_Prediction/:")
print("  Fig4h_Measured_Demand.png")
print("  Fig4i_Prediction_Segments.png")
print("  Fig4j_MSE_vs_N.png")
print("  Fig4k_R2_vs_N.png")
