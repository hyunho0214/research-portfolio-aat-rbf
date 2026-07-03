# Project Map

This file explains what each folder contributes to the portfolio and how a
reviewer can inspect the work efficiently.

## 1. `20260616Panama`

Most complete end-to-end workflow.

Purpose:

- fit measured AAT transfer curves with Gaussian parameters,
- extract sigma, mu, amplitude, baseline, and fit quality,
- feed extracted sigma values into RBF simulations,
- reproduce Panama electricity-demand forecasting and Duffing oscillator
  reconstruction workflows,
- save metrics, predictions, static figures, and an interactive HTML plot.

Key files:

- `extract_sigmas.py`: transfer-curve Gaussian fitting entry point.
- `run_panama.py`: Panama electricity-demand RBF simulation.
- `run_duffing.py`: Duffing oscillator RBF reconstruction simulation.
- `make_interactive_panama.py`: interactive browser visualization.
- `src/transfer_sigma.py`: sigma extraction utilities.
- `src/rbf.py`: RBF feature construction, clustering, fitting, and metrics.
- `manual.md`: hands-on execution guide.
- `flow.md`: conceptual end-to-end workflow.
- `tests/`: regression and utility tests.

Representative outputs:

- `output/my_sigmas/figures/gaussian_fit_overview.png`
- `output/panama_my_sigmas/figures/panama_15_day_segments.png`
- `output/panama_my_sigmas/figures/panama_r2.png`
- `output/duffing_selected/figures/duffing_phase_space.png`
- `output/duffing_selected/duffing_rbf_metrics.csv`

## 2. `new_AAT`

Most software-engineered package.

Purpose:

- package the hardware-aware AAT kernel concept as an installable Python module,
- calibrate long-format transfer curves into a kernel library,
- compare kernel variants using measured AAT parameters:
  `sigma_only`, `a_sigma`, `full_gaussian`, and `direct_curve`,
- support CLI workflows and pytest-based validation.

Key files:

- `pyproject.toml`: package metadata and console scripts.
- `src/aat_kernel/calibration.py`: transfer-curve calibration and kernel-library export.
- `src/aat_kernel/hardware_kernel.py`: measured-parameter kernel bank and linear readout.
- `src/aat_kernel/cli.py`: command-line interface.
- `configs/hardware_experiment.yaml`: experiment configuration template.
- `tests/`: unit and smoke tests.

Why it matters:

This folder shows the transition from a reproduction script to a reusable
research package with clearer APIs, test coverage, and documented commands.

## 3. `AAT_ap`, `AAT_ap_2`, `AAT_ap_scaled`

Iterative RBF experiment branches.

Purpose:

- explore how to interpret the number of distinct sigma values `N`,
- compare direct vs scaled sigma handling,
- evaluate center replication versus distinct-center RBF implementations,
- save metrics and plots across multiple experimental settings.

Key files:

- `src/sigma_utils.py`: sigma cleaning, selection, and assignment.
- `src/rbf.py`: RBF model implementation.
- `src/forecasting.py`: electricity-demand runner.
- `src/duffing.py`: Duffing runner.
- `compare.md`, `guide.md`, `manual.md`, `report.md`: experiment rationale and notes.
- `outputs/`: metrics, prediction overlays, model arrays, and configuration JSON.

Why it matters:

These folders document the development history and the scientific reasoning
behind implementation choices such as distinct centers, fixed kernel budget,
and device-to-model sigma scaling.

## 4. `RBFRQ`

Transfer-curve model comparison.

Purpose:

- fit transfer curves with Gaussian and rational-quadratic bell-shaped models,
- compare fit quality using metrics such as R2, RMSE, AIC, and BIC,
- summarize fitted parameter trends across drain-voltage conditions.

Key files:

- `RBFRQfit.py`: fitting and plotting script.
- `data.xlsx`: transfer-curve source workbook.
- `fit_results/`: per-voltage fits, aggregate summaries, and trend plots.

Why it matters:

This folder supports the device-modeling side of the portfolio by showing that
transfer-curve shape analysis was considered beyond a single Gaussian fit.

## 5. `RBF_Power_Prediction`

Early paper-reproduction prototype.

Purpose:

- implement the initial RBF neural-network workflow for electricity-demand
  prediction and Duffing reconstruction,
- reproduce Figure 4-style plots from the AAT paper,
- organize the first version of data loading, Gaussian kernels, training, and
  visualization modules.

Key files:

- `PLAN.md`: original implementation plan.
- `Fig4_Replication.py`: figure reproduction script.
- `src/rbf_network.py`, `src/trainer.py`, `src/visualizer.py`: modular prototype code.
- `Fig4h_*`, `Fig4i_*`, `Fig4j_*`, `Fig4k_*`: generated figure outputs.

## 6. `RBF2`

Baseline and reproducibility refinement.

Purpose:

- refine the RBF forecasting implementation with fixed random state,
  date-based splitting, ridge stabilization, and MLP baseline comparison,
- keep a compact implementation of the paper-style algorithm.

Key files:

- `main.py`: full forecasting workflow.
- `figure_4.py`: figure generation.
- `src/rbf_network.py`: RBF network implementation.
- `src/mlp_baseline.py`: baseline model.
- `data/continuous_dataset.csv`: input dataset.

## Suggested Interview Talking Points

- How experimental transfer-curve parameters were converted into machine-learning
  kernel parameters.
- Why chronological splitting and train-only standardization matter for
  time-series forecasting.
- Why repeated center-sigma pairs can create redundant RBF basis functions.
- How `new_AAT` generalizes sigma-only simulations into a measured kernel
  library using amplitude, center, width, and direct curves.
- What tradeoffs exist when mapping device-domain voltage parameters into a
  standardized input feature space.
