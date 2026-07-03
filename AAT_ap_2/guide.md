# Codex Implementation Guide: Multi-Gaussian RBF Forecasting Based on AAT-Derived Gaussian Widths

## Purpose

This guide defines the implementation target for the RBF-based forecasting/reconstruction code so that it stays as close as possible to the published paper and supplementary pseudocode, while resolving underspecified parts in the most technically defensible way.

The goal is **not** to reproduce the authors’ hidden source code line by line. The goal is to implement a pipeline that:

1. preserves the paper’s core idea,
2. follows the supplementary pseudocode wherever it is explicit,
3. resolves missing details using choices that are mathematically sound and easy to defend in a manuscript.

---

## Core scientific idea to preserve

Use **experimentally derived Gaussian widths** from AAT devices as the source of diversity for the hidden-layer Gaussian kernels in an RBF network.

The key experiment is:

- keep the **total number of kernels** fixed,
- vary the **number of distinct Gaussian widths** `N`,
- evaluate how increasing `sigma` diversity affects prediction/reconstruction quality.

This is the conceptual link between the device paper and the machine-learning simulation.

---

## High-level conclusion for implementation

### Adopt this principle

**Use `C_total` distinct RBF centers, and assign the selected `N` sigma values evenly across those centers.**

### Why this is the preferred interpretation

The supplementary pseudocode describes a version where k-means is run with `N` clusters and the resulting centers are replicated until the model reaches `C_total` kernels. That is useful as a conceptual explanation, but literal replication creates repeated basis functions when both center and sigma are duplicated. That leads to redundant columns in the design matrix and weakens the meaning of “fixed total kernels.”

Using **distinct `C_total` centers** is more defensible because it:

- preserves the paper’s idea of a fixed kernel budget,
- preserves the sweep over the number of distinct `sigma` values,
- avoids duplicated Gaussian basis functions,
- gives a cleaner least-squares system,
- is easier to justify in a paper or reviewer response.

### Recommended statement for Methods

> To preserve a fixed model capacity while varying Gaussian-width diversity, the total number of RBF kernels was held constant, and a selected set of `N` experimentally derived width values was evenly assigned across a fixed set of cluster-defined centers in the training-input space.

---

## Required implementation targets

Two related pipelines should be supported.

### 1. Electricity-demand forecasting

- data type: 1D time series
- input window length: `L = 10`
- target: next value after the window
- total kernels: `C_total = 200`
- metric sweep: vary `N` (number of distinct sigma values)

### 2. Duffing oscillator reconstruction

- data type: 2D state sequence `[x(t), v(t)]`
- input window length: `L = 10`
- input vector shape: flattened `2L`
- target: next state `[x(t+1), v(t+1)]`
- total kernels: `C_total = 300`
- metric sweep: vary `N`

---

## Implementation rules

## Rule 1. Preserve chronological splitting

Never randomly shuffle time-series samples.

### Electricity forecasting
Preferred split for paper-consistent reproduction:

- train: years 2016–2019
- test: year 2020

If a generic mode is needed for other datasets, allow chronological fraction split as an option, but the default for the paper workflow should be the calendar-based split above.

### Duffing
Use chronological train/test splitting after transient removal.
A 50/50 split is acceptable because the supplementary pseudocode explicitly uses that as an example.

---

## Rule 2. Standardize both inputs and targets

Apply standardization to:

- `X_train`, `X_test`
- `y_train`, `y_test`

Fit the scalers on training data only.
Use inverse transformation before reporting predictions and metrics in original units.

This is not optional.
It is explicitly consistent with the supplementary pseudocode and is also necessary for a meaningful interaction between Euclidean distances and Gaussian widths.

---

## Rule 3. Sigma values must come from experimental AAT width information

The RBF width values must be tied to the experimentally measured Gaussian widths.

### Recommended processing of sigma data

Given an experimental sigma array `sigma_data`:

1. remove invalid or nonpositive values,
2. optionally remove obvious fit failures/outliers only if there is a documented rule,
3. obtain a representative range:
   - `sigma_min = min(valid_sigma)`
   - `sigma_max = max(valid_sigma)`

### Recommended selection of `N` distinct sigmas

Use logarithmic spacing between `sigma_min` and `sigma_max`:

```python
sigma_set = np.geomspace(sigma_min, sigma_max, N)
```

This is the preferred choice because it is the clearest match to the paper’s description for the simulation section.

### Important note

If the experimental sigma values exist in a device-domain scale that is not directly compatible with standardized input distances, preserve the **relative span/distribution** of the experimental sigmas while using a normalized version for the actual RBF computation.

Best-practice option:

- normalize experimental sigmas by a fixed reference,
- then rescale into the standardized-input distance regime.

However, unless this conversion is absolutely necessary for numerical stability, start with the direct standardized-space implementation using the processed experimental sigma range.

---

## Rule 4. Use `C_total` distinct centers

### Recommended center construction

Run k-means with:

- `n_clusters = C_total`
- on `X_train` after standardization

Then use the resulting `C_total` centroids as the RBF centers.

Example:

```python
kmeans = KMeans(n_clusters=C_total, random_state=seed, n_init=10)
kmeans.fit(X_train)
centers = kmeans.cluster_centers_
```

### Sigma assignment across centers

After choosing `sigma_set` of length `N`, assign those width values evenly across the `C_total` centers.

Example logic:

- `base_count = C_total // N`
- `extra = C_total % N`
- first `extra` sigma values get `base_count + 1` centers
- remaining sigma values get `base_count` centers

Then build `sigma_per_center` of length `C_total`.

### Recommended ordering

The simplest acceptable implementation is block assignment:

- first group of centers gets `sigma_set[0]`
- second group gets `sigma_set[1]`
- ...

A slightly better implementation is to shuffle the sigma assignments across centers with a fixed random seed after even allocation. This avoids ordering artifacts without changing the overall distribution.

Preferred default:

- create even sigma counts,
- randomly permute assignments with a fixed seed.

---

## Rule 5. RBF layer definition

For an input vector `u` and center `c_j` with width `sigma_j`:

```python
phi_j(u) = exp(-||u - c_j||^2 / (2 * sigma_j^2))
```

Construct the design matrix `Phi` with shape:

- forecasting: `(num_samples, C_total)`
- Duffing: `(num_samples, C_total)`

Then append a bias column of ones.

---

## Rule 6. Train output weights by linear least squares

Use a linear readout fitted in closed form or via `np.linalg.lstsq`.

### Single-output case
Electricity demand:

```python
w, *_ = np.linalg.lstsq(Phi_aug_train, y_train, rcond=None)
```

### Multi-output case
Duffing `[x, v]` prediction:

```python
W, *_ = np.linalg.lstsq(Phi_aug_train, y_train, rcond=None)
```

No nonlinear output training is needed.
This linear readout is central to the paper-consistent implementation.

---

## Rule 7. Metrics

### Electricity forecasting
Report at least:

- MSE
- MAE
- R²

### Duffing reconstruction
Report at least:

- `MSE_x`
- `MSE_v`
- `R²_x`
- `R²_v`

Where needed, also provide phase-space and time-domain visualizations.

---

## Pipeline specification

## A. Electricity-demand forecasting pipeline

### Step A1. Load and sort data

- load daily electricity-demand data
- sort chronologically
- confirm a single scalar target per day

### Step A2. Split

Preferred paper-aligned split:

- training years: 2016–2019
- test year: 2020

### Step A3. Build supervised windows

For each sequence:

- input: 10 consecutive daily observations
- target: next day demand

Pseudoform:

```python
X[t] = s[t:t+10]
y[t] = s[t+10]
```

### Step A4. Standardize

- fit `scaler_X` on `X_train_raw`
- transform train/test inputs
- fit `scaler_y` on `y_train_raw`
- transform train/test targets

### Step A5. Build sigma candidates from experimental data

- load `sigma_data`
- clean it
- determine `sigma_min`, `sigma_max`

### Step A6. Sweep over `N`

For each `N` in `N_values`:

1. select `N` sigmas using log spacing,
2. compute `C_total` distinct centers using k-means on `X_train`,
3. assign the `N` sigmas evenly across the `C_total` centers,
4. construct `Phi_train`,
5. fit least-squares output weights,
6. construct `Phi_test`,
7. inverse-transform predictions,
8. compute metrics.

### Step A7. Track best model

Store:

- best `N`
- best metrics
- best predictions
- optionally best centers and sigma assignment

---

## B. Duffing reconstruction pipeline

### Step B1. Generate Duffing data

Use RK4 or an equivalent fixed-step fourth-order Runge–Kutta solver.

State equations:

```python
x_dot = v
v_dot = -beta * v + x - alpha * x**3 + gamma * cos(omega * t)
```

### Step B2. Remove transient

Discard the initial transient region before building the dataset.

### Step B3. Build supervised windows

Each input is 10 consecutive states.
Flatten each window into length `2L`.
Each target is the next state vector `[x, v]`.

### Step B4. Standardize

Apply the same standardization philosophy as in the electricity pipeline.

### Step B5. Train/test split

Use chronological split.
Default: 50/50 after transient removal.

### Step B6. Sweep over `N`

For each `N`:

1. choose `N` log-spaced sigmas,
2. compute `C_total = 300` distinct centers using k-means,
3. assign sigmas evenly,
4. fit least-squares multi-output readout,
5. inverse-transform predictions,
6. compute per-dimension metrics.

### Step B7. Visualization

Generate:

- phase-space plots `x(t)` vs `v(t)`
- time-domain overlays of true vs predicted `x(t)` and `v(t)`

---

## Why the distinct-center approach is the default

This section is meant to guide Codex when there is ambiguity.

### The supplementary-style literal approach

A literal reading would be:

- run k-means with `N` clusters,
- get `N` centers,
- replicate them until reaching `C_total` kernels,
- assign one sigma per replicated center group.

### Problem with literal replication

If both the center and sigma are replicated exactly, then the corresponding Gaussian basis functions are identical. This creates repeated columns in `Phi`, which:

- does not truly increase hidden-layer expressivity,
- can reduce matrix rank,
- makes `C_total` partly cosmetic rather than functional.

### Better paper-ready interpretation

Use:

- `C_total` genuinely distinct centers,
- `N` distinct sigma values,
- even sigma reuse across the distinct centers.

This preserves the paper’s intended ablation:

- fixed kernel count,
- varied width diversity.

This should be treated as the preferred implementation unless there is an explicit reason to reproduce the supplementary pseudocode literally.

---

## What Codex should implement

## File structure recommendation

```text
project/
  guide.md
  data/
  src/
    data_utils.py
    sigma_utils.py
    rbf.py
    forecasting.py
    duffing.py
    metrics.py
    plotting.py
    main_forecasting.py
    main_duffing.py
  outputs/
  notebooks/
```

## Minimal required components

### `sigma_utils.py`
Should implement:

- `clean_sigma_data(sigma_data)`
- `get_sigma_range(sigma_data)`
- `select_log_spaced_sigmas(sigma_data, N)`
- `assign_sigmas_evenly(C_total, sigma_set, shuffle=True, seed=...)`

### `data_utils.py`
Should implement:

- chronological split helpers
- sliding-window dataset creation
- standardization helpers

### `rbf.py`
Should implement:

- center extraction with k-means
- Gaussian feature construction
- least-squares fit/predict
- support for single-output and multi-output

### `forecasting.py`
Should implement:

- end-to-end electricity-demand experiment
- sweep over `N`
- best-model tracking
- optional MLP baseline

### `duffing.py`
Should implement:

- Duffing solver
- transient removal
- dataset creation
- RBF sweep
- plotting hooks

---

## Recommended API design

### RBF model class

```python
class RBFNetwork:
    def __init__(self, centers, sigma_per_center):
        ...

    def _compute_features(self, X):
        ...

    def fit(self, X, y):
        ...

    def predict(self, X):
        ...
```

### Experiment runner

```python
def run_rbf_sigma_sweep(
    X_train,
    y_train,
    X_test,
    y_test,
    sigma_data,
    N_values,
    C_total,
    seed=0,
):
    ...
```

Return:

- per-`N` metrics table
- best model info
- predictions
- sigma assignments
- centers

---

## MLP baseline policy

Implement the MLP baseline as optional.
It is useful for completeness, but the main scientific contribution is the multi-Gaussian RBF formulation.

Suggested default:

```python
MLPRegressor(
    hidden_layer_sizes=(200,),
    activation="logistic",
    solver="sgd",
    random_state=seed,
    max_iter=...,
)
```

If exact optimizer hyperparameters from the original source are unavailable, use a reasonable documented default and state that the baseline was tuned under the same train/test conditions.

---

## Reproducibility requirements

Codex should make the code reproducible.

### Always fix random seeds for:

- k-means
- any sigma-assignment shuffle
- MLP baseline

### Save outputs

For each run, save:

- metrics CSV
- best predictions CSV
- configuration JSON or YAML
- plots

### Log configuration

At minimum log:

- `L`
- `C_total`
- `N_values`
- seed
- sigma range
- split description
- scaler choice
- whether sigma assignments were shuffled

---

## Manuscript-friendly rationale text

Use language consistent with the following.

### Short rationale

> The multi-Gaussian RBF model was implemented to reflect the experimentally observed diversity of AAT Gaussian widths. To isolate the effect of width diversity, the total number of kernels was fixed while the number of distinct width values was varied.

### Longer rationale

> Because the supplementary pseudocode conceptually fixes the total kernel budget while sweeping the number of distinct Gaussian widths, we implemented the hidden layer using a constant number of cluster-defined centers and evenly assigned the selected width values across those centers. This preserves the intended ablation on sigma diversity while avoiding redundant basis duplication.

---

## Do not do the following

- do not randomly shuffle time-series samples before splitting,
- do not fit scalers on train+test together,
- do not use randomly initialized RBF centers as the default,
- do not vary both `C_total` and `N` at the same time in the main experiment,
- do not replace experimental sigma information with arbitrary hand-tuned widths unless clearly labeled as an auxiliary analysis,
- do not replicate identical center-sigma pairs as the default implementation.

---

## Final instruction to Codex

When ambiguity exists, prefer the choice that best preserves all three of the following:

1. fidelity to the paper’s central claim,
2. mathematical defensibility,
3. clarity in a Methods section and reviewer response.

If there is a conflict between literal pseudocode replication and a cleaner implementation of the same scientific idea, choose the cleaner implementation **but keep the scientific idea unchanged** and document the decision in comments.
