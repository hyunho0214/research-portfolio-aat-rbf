Algorithm 2: Duffing oscillator reconstruction with multi-Gaussian RBF network

Goal:
    Reconstruct the chaotic Duffing oscillator dynamics using an RBF network
    with multiple Gaussian kernels whose widths σ are chosen based on
    experimentally measured Gaussian responses (Fig. 3d–g, Fig. S10).

Inputs:
    - Δt = 1 ms: time step for numerical integration
    - T_total: total simulation time
    - T_trans: transient duration to discard
    - L = 10: input window length (10 consecutive samples = 10 ms)
    - C_total: total number of Gaussian kernels (= 300 for main text Duffing)
    - N_values: list of numbers of distinct σ values (e.g., N = 1, 2, …, 80)
    - sigma_data: experimentally measured σ distribution from AAT devices
                  (multi-Gaussian fitting results; Fig. 3d–g, Fig. S10)
    - Duffing parameters (α, β, γ, ω) and initial state (x0, v0)

Outputs:
    - Reconstruction performance (MSE, R²) as a function of N
    - Best-performing RBF model’s predictions for x(t) and v(t)
    - Optionally, phase-space and time-domain plots as in Fig. 4c–f, S11, S12

---------------------------------------------------------------
1:  # STEP 1: Generate Duffing oscillator time series
2:  Define Duffing ODE:
3:      dx/dt = v
4:      dv/dt = -β v + x - α x³ + γ cos(ω t)
5:
6:  Initialize:
7:      t = 0
8:      state = [x0, v0]
9:
10: Use a 4th-order Runge–Kutta-type solver (e.g., RK4/RK45) with time step Δt:
11:     Simulate from t = 0 to t = T_total + T_trans
12:     Collect state(t) = [x(t), v(t)] at each step
13:
14: Discard initial transient:
15:     data = state samples after time T_trans
16:     # data shape: (num_points, 2) containing [x(t), v(t)]

17: # STEP 2: Construct input–output pairs via sliding window
18: function CREATE_DATASET(data, L):
19:     X_list = empty list
20:     y_list = empty list
21:     for n from 1 to length(data) - L:
22:         window = data[n : n+L-1]           # 10 consecutive [x, v]
23:         X_list.append( flatten(window) )   # shape: 2L (stack x,v)
24:         y_list.append( data[n+L] )         # next state [x, v]
25:     return matrix(X_list), matrix(y_list)
26:
27: X_raw, y_raw = CREATE_DATASET(data, L)
28: # X_raw: (num_samples, 2L), y_raw: (num_samples, 2)

29: # STEP 3: Standardize inputs and outputs
30: Fit input scaler on X_raw and transform:
31:     X = STANDARDIZE(X_raw)
32: Fit output scaler on y_raw and transform:
33:     y = STANDARDIZE(y_raw)
34:
35: # STEP 4: Split into training and test sets (chronological)
36: Choose a chronological split ratio (e.g., 50% train, 50% test):
37:     num_train = floor(0.5 * num_samples)
38:     X_train = X[1 : num_train]
39:     y_train = y[1 : num_train]
40:     X_test  = X[num_train+1 : end]
41:     y_test  = y[num_train+1 : end]
42:
43: Store original-scale test targets:
44:     y_test_orig = INVERSE_STANDARDIZE(y_test)

45: # STEP 5: Obtain experimental σ range from device data
46: (sigma_min, sigma_max) = PROCESS_EXPERIMENTAL_SIGMA_DATA(sigma_data)
47:     # sigma_data: set of Gaussian widths extracted from AAT measurements
48:     # (multi-Gaussian fits in Fig. 3d–g and Fig. S10)
49:     # Example:
50:     #   sigma_min = min(σ_exp)
51:     #   sigma_max = max(σ_exp)

52: Initialize:
53:     metrics_list = empty list    # to store (N, MSE_x, MSE_v, R²_x, R²_v)
54:     predictions = empty dict     # map from N → predicted time series
55:
56: # STEP 6: Loop over number of distinct σ values
57: for each N in N_values:
58:     # STEP 6.1: Select N σ values within experimental range
59:     sigma_set = LOG_SPACED_SIGMAS(sigma_min, sigma_max, N)
60:         # N logarithmically spaced values between sigma_min and sigma_max

61:     # STEP 6.2: Find N cluster centers in input space
62:     Run k-means clustering on X_train with N clusters
63:     Let cluster_centers[1..N] be the N centroids

64:     # STEP 6.3: Expand to C_total Gaussian kernels
65:     base_count  = floor(C_total / N)
66:     extra_count = C_total mod N
67:     centers_list = empty list
68:     sigma_list   = empty list
69:     for k from 1 to N:
70:         if k ≤ extra_count:
71:             count_k = base_count + 1
72:         else:
73:             count_k = base_count
74:         Replicate cluster_centers[k] exactly count_k times
75:         Append these replicated centers to centers_list
76:         Append sigma_set[k] to sigma_list count_k times
77:     # centers_list: length C_total, sigma_list: length C_total

78:     # STEP 6.4: Train RBF network (linear least-squares)
79:     Define RBF feature map Φ for an input u:
80:         for j from 1 to C_total:
81:             Φ_j(u) = exp(- ||u - centers_list[j]||² / (2 * sigma_list[j]²))
82:
83:     Construct design matrix Φ_train:
84:         Φ_train[n, j] = Φ_j(X_train[n])
85:     Augment with bias:
86:         Φ̃_train = [Φ_train | 1]      # add column of ones
87:
88:     Solve for weights W (size (C_total+1) × 2) using least squares:
89:         W = argmin_W || Φ̃_train * W - y_train ||²

90:     # STEP 6.5: Evaluate on test set
91:     Construct Φ_test from X_test similarly
92:     Φ̃_test = [Φ_test | 1]
93:     y_pred = Φ̃_test * W          # standardized predictions
94:     y_pred_orig = INVERSE_STANDARDIZE(y_pred)
95:
96:     # Compute error metrics separately for x and v
97:     MSE_x = MSE(y_test_orig[:, x], y_pred_orig[:, x])
98:     MSE_v = MSE(y_test_orig[:, v], y_pred_orig[:, v])
99:     R²_x  = R2(y_test_orig[:, x], y_pred_orig[:, x])
100:    R²_v  = R2(y_test_orig[:, v], y_pred_orig[:, v])
101:
102:    Append (N, MSE_x, MSE_v, R²_x, R²_v) to metrics_list
103:    Store predictions[N] = y_pred_orig

104: # STEP 7: (Optional) Select representative N for visualization
105: Choose representative N values (e.g., N = 1, 5, 10, 30) for figures:
106:     - Phase-space reconstructions (x(t) vs v(t))
107:     - Time-domain overlays x(t), v(t) for ground truth vs prediction
108:
109: # Note: For the main-text Duffing example in Fig. 4c–f,
110: #       a fixed total number of kernels C_total = 300 is used
111: #       while N is varied. For SI Fig. S11, the same procedure
112: #       is applied with C_total = 720, as stated in the caption.
