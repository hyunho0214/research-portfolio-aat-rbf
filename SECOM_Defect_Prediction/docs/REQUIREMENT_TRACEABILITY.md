# Requirement Traceability

This document maps the original project requirements to the implemented code,
validation commands, and portfolio artifacts.

## Requirement 1: Extreme Class Imbalance

Original problem:

- SECOM is a high-yield semiconductor dataset.
- Pass samples dominate the dataset.
- A naive all-pass classifier can show high accuracy while detecting no fails.

Implemented response:

- Treat fail label `1` as the positive class.
- Avoid accuracy as the primary metric.
- Evaluate fail recall, fail F1, PR-AUC, balanced accuracy, and confusion
  matrix counts.
- Compare a dummy all-pass classifier against real models.
- Support SMOTE inside the training fold only.
- Support class-weighted models.
- Add threshold analysis to expose recall/false-alarm tradeoffs.

Evidence:

- Code: `src/secom_defect/modeling.py`
- Code: `threshold_analysis.py`
- Report: `reports/final/metrics_summary.csv`
- Report: `reports/final/confusion_matrices.csv`
- Report: `reports/final/threshold_tradeoff.csv`
- Report: `reports/final/recommended_threshold.json`
- Figure: `reports/final/class_distribution.png`
- Figure: `reports/final/model_metric_comparison.png`
- Figure: `reports/final/threshold_tradeoff.png`

Current result:

- Dummy classifier fail recall: `0.000`
- Final XGBoost benchmark fail recall: `0.615`
- Threshold-tuned XGBoost fail recall: `0.810` at threshold `0.15`

## Requirement 2: Many Missing Values

Original problem:

- Sensor values can be missing due to equipment delay, different sampling
  cycles, or incomplete measurement.
- Very sparse sensor columns should be dropped.
- Remaining missing values should be imputed with a stronger strategy than
  naive global averaging.

Implemented response:

- Drop columns whose missing rate is above the configured threshold.
- Default missing-column threshold: `0.50`.
- Support multiple imputers:
  - median,
  - mean,
  - KNN,
  - forward-fill style fallback.
- Save missingness summaries and figures for audit.
- Compare median and KNN imputation in benchmark reports.

Evidence:

- Code: `src/secom_defect/modeling.py`
- Code: `src/secom_defect/reporting.py`
- Config: `configs/experiment_matrix.yaml`
- Report: `reports/final/missingness_summary.csv`
- Figure: `reports/final/missingness_distribution.png`
- Report: `reports/final/experiment_best_summary.csv`
- Figure: `reports/final/preprocessing_comparison.png`

Current result:

- SMOTE + median top-30 benchmark best fail F1: `0.221`
- SMOTE + KNN top-30 benchmark best fail F1: `0.193`
- Final median top-50 5-fold benchmark fail F1: `0.229`

## Requirement 3: High Dimensionality, Multicollinearity, and Sensor Noise

Original problem:

- SECOM has hundreds of anonymized sensor features.
- Some sensors are nearly constant or weakly related to fail outcome.
- A smaller set of influential sensor features is needed for a cleaner
  portfolio result.

Implemented response:

- Apply variance-threshold filtering.
- Use Random Forest feature importance for top-k feature selection.
- Support top-k settings such as 30, 50, and 80.
- Save selected feature lists and feature-importance tables.
- Plot top selected sensor importances.

Evidence:

- Code: `src/secom_defect/modeling.py`
- Config: `configs/experiment_matrix.yaml`
- Report: `reports/final/selected_features.csv`
- Report: `reports/final/feature_importance.csv`
- Figure: `reports/final/top_feature_importance.png`

Current result:

- Final benchmark uses top-50 selected features.
- Top selected features include `sensor_059`, `sensor_033`, `sensor_103`,
  `sensor_510`, and `sensor_351`.

## Requirement 4: Compare Multiple Machine-Learning Models

Original goal:

- Build a Python-based semiconductor defect prediction project.
- Compare multiple model families.
- Identify which model and preprocessing path is most effective.

Implemented response:

- Compare:
  - dummy classifier,
  - Logistic Regression,
  - Random Forest,
  - RBF SVM,
  - Gradient Boosting,
  - XGBoost.
- Run baseline, SMOTE, KNN-imputation, and 5-fold top-50 experiments.
- Summarize model and preprocessing comparisons in final reports.

Evidence:

- Code: `run_experiment.py`
- Code: `src/secom_defect/modeling.py`
- Script: `scripts/compare_reports.py`
- Report: `reports/final/metrics_summary.csv`
- Report: `reports/final/experiment_best_summary.csv`
- Figure: `reports/final/model_metric_comparison.png`
- Figure: `reports/final/preprocessing_comparison.png`

Current result:

| Experiment | Best model | Fail recall | Fail F1 | Balanced accuracy |
| --- | --- | ---: | ---: | ---: |
| No SMOTE + median top-30 | Logistic Regression | 0.548 | 0.219 | 0.651 |
| SMOTE + median top-30 | Logistic Regression | 0.558 | 0.221 | 0.655 |
| SMOTE + KNN top-30 | XGBoost | 0.510 | 0.193 | 0.621 |
| SMOTE + median top-50 5-fold | XGBoost | 0.615 | 0.229 | 0.674 |

## Requirement 5: Hyperparameter Optimization and Improvement Direction

Original goal:

- Build a project about defect prediction and hyperparameter optimization.
- Use results to suggest improvement directions.

Implemented response:

- Add `tune_model.py` using `RandomizedSearchCV`.
- Add a first logistic-regression tuning smoke run.
- Add `threshold_analysis.py` for operating-point selection.
- Document next improvement paths in the model card.

Evidence:

- Code: `tune_model.py`
- Code: `threshold_analysis.py`
- Doc: `BASELINE_NOTES.md`
- Doc: `MODEL_CARD.md`
- Report: `reports/final/threshold_tradeoff.csv`
- Report: `reports/final/recommended_threshold.json`
- Report: `reports/final/tuning_logistic_best_params.json`
- Report: `reports/final/tuning_logistic_cv_results.csv`
- Report: `reports/final/tuning_logistic_holdout_classification_report.json`
- Report: `reports/final/tuning_logistic_holdout_confusion_matrix.csv`

Current result:

- Logistic tuning smoke run confirmed the tuning harness works.
- Best tuned logistic parameter: `model__C = 0.00195`
- Tuned logistic holdout fail recall: `0.667`
- Tuned logistic holdout fail F1: `0.212`
- Threshold analysis selected threshold `0.15` for target fail recall `>= 0.70`.
- Recommended next steps include longer XGBoost tuning, feature-count
  comparison, cost-sensitive thresholding, and probability calibration.

## AI-Assisted Development Requirement

Additional portfolio goal:

- Make the project read as active AI-assisted development.
- Show what the user asked Codex to do.
- Show the validation harness and command history.
- Avoid presenting the work as passive dependence on Codex.

Implemented response:

- Add human-in-the-loop workflow documentation.
- Add prompt harness documentation.
- Add command log and validation evidence.
- Record the approval-gate workflow: Codex proposes, user approves or corrects,
  Codex implements, Codex validates, user decides the next direction.

Evidence:

- Doc: `docs/CODEX_WORKFLOW.md`
- Doc: `docs/PROMPT_HARNESS.md`
- Doc: `docs/COMMAND_LOG.md`
- Doc: `docs/PORTFOLIO_BRIEF_KR.md`
