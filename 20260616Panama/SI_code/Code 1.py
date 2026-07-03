Algorithm 1: Multi-Gaussian RBF forecasting using experimental Gaussian widths

Input:
    - s[1..T]: 1D time-series (e.g., electrical demand)
    - L: input window length (here L = 10)
    - N_values: list of number of distinct Gaussian widths (e.g., {1, 2, ..., 80})
    - C_total: total number of RBF centers (here C_total = 200)
    - sigma_data: experimentally measured Gaussian widths from AAT devices
                  (e.g., extracted from multi-Gaussian fits to device conductance curves)

Output:
    - Performance metrics (MSE, MAE, R²) for each N in N_values
    - Best-performing RBF model and its predictions on the test set
    - MLP baseline metrics and predictions

-------------------------------------------------------------
1:  # STEP 1: Train–test split (time-based)
2:  Split s[1..T] into:
3:      s_train = first 80% of samples (chronological)
4:      s_test  = last  20% of samples

5:  # STEP 2: Build supervised datasets via sliding window
6:  function CREATE_DATASET(sequence, L):
7:      X_list = empty list
8:      y_list = empty list
9:      for t from 1 to length(sequence) - L:
10:         X_list.append( sequence[t : t+L-1] )   # L consecutive samples
11:         y_list.append( sequence[t+L] )        # next sample
12:     return matrix(X_list), vector(y_list)

13: X_train_raw, y_train_raw = CREATE_DATASET(s_train, L)
14: X_test_raw,  y_test_raw  = CREATE_DATASET(s_test,  L)

15: # STEP 3: Standardization of inputs and targets
16: Fit scaler_X on X_train_raw and transform X_train_raw, X_test_raw
17: Fit scaler_y on y_train_raw and transform y_train_raw, y_test_raw
18: Store inverse transformations for later (to recover original units)

19: # STEP 4: Train MLP baseline
20: Initialize 1-hidden-layer MLP:
21:     - hidden units: 200
22:     - activation: logistic (sigmoid)
23:     - optimizer: stochastic gradient descent
24:     - learning rate, momentum, and max iterations as in Experimental section
25: Train MLP on (X_train, y_train)
26: Predict y_pred_mlp (standardized) for X_test
27: Inverse-transform y_pred_mlp and y_test to original scale
28: Compute MLP MSE, MAE, R² on test set

29: # STEP 5: Precompute experimental sigma candidates
30: sigma_candidates = PROCESS_EXPERIMENTAL_SIGMA_DATA(sigma_data)
31:     # e.g., collect all fitted Gaussian widths from measured multi-Gaussian
32:     # device responses and summarize them as a continuous range:
33:     # [sigma_min, sigma_max] matching the experimental distribution

34: # STEP 6: Loop over N (number of distinct sigma values)
35: Initialize empty list RBF_metrics
36: best_mse  = +∞
37: best_N    = None
38: best_pred = None

39: for each N in N_values:
40:     # STEP 6.1: Select N distinct widths from experimental distribution
41:     sigma_set = SELECT_N_SIGMAS_FROM_DATA(sigma_candidates, N)
42:         # e.g., choose N widths that span [sigma_min, sigma_max]
43:         # or choose N representative quantiles of the empirical distribution

44:     # STEP 6.2: Determine RBF centers via clustering on training inputs
45:     Run k-means clustering on X_train with N clusters
46:     Let cluster_centers[1..N] be the N cluster centroids in input space

47:     # STEP 6.3: Expand to C_total centers with assigned widths
48:     centers_list = empty list
49:     sigma_list   = empty list
50:     base_count   = floor(C_total / N)
51:     extra_count  = C_total mod N
52:     for k from 1 to N:
53:         if k ≤ extra_count:
54:             count_k = base_count + 1
55:         else:
56:             count_k = base_count
57:         Replicate cluster_centers[k] exactly count_k times
58:         Append these replicated centers to centers_list
59:         Append sigma_set[k] to sigma_list count_k times
60:     # Now centers_list has length C_total and sigma_list has length C_total

61:     # STEP 6.4: Train RBF network
62:     Initialize RBFNetwork with:
63:         input_dim   = L
64:         num_centers = C_total
65:         centers     = centers_list
66:         sigma_values= sigma_list
67:         output_dim  = 1

68:     Fit RBFNetwork on (X_train, y_train) using linear least squares:
69:         - Construct design matrix Φ_train:
70:               Φ_train[n, j] = exp( -||X_train[n] - centers_list[j]||²
71:                                         / (2 * sigma_list[j]²) )
72:         - Augment Φ_train with bias column of ones
73:         - Solve for weights w in least-squares sense:
74:               w = argmin_w || Φ_train * w - y_train ||²

75:     # STEP 6.5: Evaluate RBF network
76:     Compute Φ_test analogously on X_test
77:     Predict y_pred_rbf (standardized) = Φ_test * w
78:     Inverse-transform y_pred_rbf and y_test to original scale
79:     Compute MSE, MAE, R² for this N
80:     Store (N, MSE, MAE, R²) in RBF_metrics

81:     # STEP 6.6: Track best-performing N
82:     if current MSE < best_mse:
83:         best_mse  = current MSE
84:         best_N    = N
85:         best_pred = y_pred_rbf (in original scale)

86: # STEP 7: Return results
87: Return:
88:     - MLP metrics (MSE, MAE, R²)
89:     - RBF_metrics for all N in N_values
90:     - best_N and best_pred
