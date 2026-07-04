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

## 3. `FEDL_Data`

Raw-data preprocessing and plotting tool for FEDL measurement files.

Purpose:

- consolidate Agilent CSV or Keithley Excel measurement outputs into a
  plot-ready `raw_data.csv`,
- allow manual column selection, column ordering, and head/tail preview before
  export,
- build Series, Group, and Plot definitions from the consolidated table,
- generate MATLAB plotting scripts for log-scale FEDL device curves,
- keep representative PNG/SVG/FIG outputs and distributable Windows executables
  as portfolio evidence.

Key files:

- `README.md`: curated overview and commands.
- `data preprocessing/raw_data_gui.py`: standalone raw-data extraction GUI.
- `data preprocessing/build_raw_data.ps1`: scriptable CSV consolidation.
- `plotting/plotting_gui.py`: standalone plotting GUI.
- `plotting/matlab_generator.py`: MATLAB script generator.
- `final/app.py`: integrated two-tab preprocessing plus plotting GUI.
- `final/outputs/Plot1_V1_Abs_Id_log.png`: representative plotted output.
- `project_plans/2026-03-20_fedl_plotting_plan.md`: development plan and iteration log.

Why it matters:

This folder demonstrates practical lab automation: converting raw instrument
exports into traceable plotting input, then generating quick visual checks that
make repeated device sweeps easier to review and share.

## 4. `memT_BO`

Bayesian optimization workflow for TFT and memtransistor process-condition
screening.

Purpose:

- use Gaussian-process regression and Expected Improvement to recommend the
  next process condition,
- validate the optimization approach first on TFT mobility data,
- extend the workflow to memtransistor log on/off-ratio optimization,
- compare kernel choices such as RBF, Constant x RBF, Rational Quadratic, and
  Matern,
- save prediction grids, confidence intervals, experimental points, and
  next-point recommendations as auditable CSV/Excel artifacts.

Key files:

- `README.md`: curated overview and commands.
- `CURATION.md`: what was kept, what was left out, and why.
- `TFT/250723 TFT BO_HH/simple TFT BO(ratio-mobility)(Cons&RBF)_csv input.py`:
  final TFT mobility optimizer using CSV input.
- `TFT/250723 TFT BO_HH/mobility_prediction_iter_0.csv`: saved TFT prediction
  grid with confidence intervals.
- `TFT/250723 TFT BO_HH/next_point_iter_0.csv`: next recommended TFT process
  condition.
- `memT/csv input/(ratio,thickness)-retention, excel (ConRBF)_file load_NEW_1.py`:
  final Constant x RBF memtransistor optimizer.
- `memT/csv input/(ratio,thickness)-retention, excel (RBF)_file load_NEW_1.py`:
  RBF comparison workflow.
- `memT/csv input/onoff_ratio_prediction_iter_0.csv`: saved memtransistor
  prediction grid.
- `memT/csv input/next_point_iter_0.csv`: next recommended memtransistor
  process condition.
- `memT/TEST/TFT test.py`: TFT process FOM pre-test using mobility, Vth, and
  subthreshold swing.
- `memT/RBF 결과/`: saved RBF and Constant x RBF parameter-sweep outputs.

Why it matters:

This folder shows how experimental-device work was turned into an iterative
optimization loop: select a candidate grid, fit a surrogate model, score
Expected Improvement, recommend the next experiment, and preserve each
iteration for review.

## 5. `SECOM_Defect_Prediction`

Semiconductor defect-prediction and hyperparameter-optimization portfolio
project.

Purpose:

- model rare semiconductor fail samples from SECOM process-sensor data,
- handle class imbalance with recall/F1/PR-AUC-oriented evaluation,
- compare median and KNN imputation choices,
- keep SMOTE and feature selection inside the modeling pipeline to avoid
  leakage,
- compare Logistic Regression, Random Forest, SVM, Gradient Boosting, and
  XGBoost,
- document hyperparameter tuning and threshold tradeoffs for missed-fail versus
  false-alarm decisions.

Key files:

- `README.md`: project overview and run commands.
- `MODEL_CARD.md`: final model snapshot, limitations, and next steps.
- `docs/REQUIREMENT_TRACEABILITY.md`: maps original requirements to code and reports.
- `docs/PORTFOLIO_BRIEF_KR.md`: Korean interview brief.
- `run_experiment.py`: model-comparison entry point.
- `tune_model.py`: RandomizedSearchCV tuning entry point.
- `threshold_analysis.py`: recall/false-alarm threshold analysis.
- `reports/final/`: curated figures, CSVs, selected features, and tuning outputs.

Why it matters:

This folder turns a common Kaggle-style semiconductor dataset into a
manufacturing-oriented portfolio story: accuracy is not enough under class
imbalance, preprocessing choices matter, and operating thresholds should be
chosen with missed-fail cost in mind.

## 6. `MES_SQLD_Practice`

SQLD-level semiconductor MES practice project using SQLite.

Purpose:

- practice basic SQL through a semiconductor MES scenario,
- filter low-yield or defective wafers,
- compare yield before and after equipment PM events,
- create simple wafer-map defect preprocessing features,
- keep the SQL easy enough to explain in interviews.

Key files:

- `README.md`: project overview, run commands, and query map.
- `sql/schema.sql`: sample MES table definitions.
- `sql/seed_data.sql`: synthetic practice data.
- `queries/01_defect_wafer_filter.sql`: low-yield wafer filtering.
- `queries/02_pm_yield_relation.sql`: PM-before/after yield comparison.
- `queries/03_wafer_map_defect_preprocess.sql`: wafer-map feature creation.
- `docs/INTERVIEW_BRIEF_KR.md`: Korean interview talking points.
- `docs/SQLD_SCOPE.md`: SQL concepts used and intentionally avoided.
- `outputs/`: generated CSV and Markdown query results.

Why it matters:

This folder shows that SQLD fundamentals can be connected to semiconductor
manufacturing data without pretending to be a senior DBA. It emphasizes
explainable joins, filtering, grouping, and CASE logic.

## 7. `AAT_ap`, `AAT_ap_2`, `AAT_ap_scaled`

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

## 8. `RBFRQ`

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

## 9. `RBF_Power_Prediction`

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

## 10. `RBF2`

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
- How Bayesian optimization converted TFT and memtransistor process grids into
  next-experiment recommendations.
- Why chronological splitting and train-only standardization matter for
  time-series forecasting.
- Why repeated center-sigma pairs can create redundant RBF basis functions.
- How `new_AAT` generalizes sigma-only simulations into a measured kernel
  library using amplitude, center, width, and direct curves.
- How `FEDL_Data` turns raw measurement exports into auditable plotting input
  and fast visual review outputs.
- Why class imbalance changes the evaluation policy for semiconductor defect
  prediction.
- How simple SQL joins and CASE expressions can support MES-style wafer and PM
  screening.
- What tradeoffs exist when mapping device-domain voltage parameters into a
  standardized input feature space.
