# Model Card: SECOM Semiconductor Defect Prediction

## Intended Use

This project builds a rare-defect prediction workflow for semiconductor
manufacturing sensor data. The model is intended for portfolio demonstration
and process-monitoring research, not direct production deployment.

The core objective is to detect the rare fail class (`1`) from hundreds of
anonymized process sensor signals while avoiding the common accuracy trap in
high-yield manufacturing data.

## Data

- Dataset: SECOM semiconductor manufacturing data.
- Primary reproducible source in this repository: UCI SECOM.
- Labels: `-1` for pass and `1` for fail.
- Shape described by the source metadata: 1567 samples and 591 features.
- Class distribution: 1463 pass samples and 104 fail samples.

Raw data is excluded from Git. The project can download the public UCI archive
or load locally downloaded Kaggle-style SECOM files from `data/raw/`.

## Preprocessing

The workflow is designed around semiconductor data problems:

- Class imbalance: fail samples are rare, so accuracy is not used as the main
  success metric.
- Missing values: columns with excessive missingness are removed, and remaining
  missing values can be handled by median, KNN, mean, or forward-fill style
  imputers.
- High-dimensional noise: low-variance features are removed before model
  training.
- Feature selection: tree-based feature importance is used to select a smaller
  set of dominant sensor features.
- Data leakage control: SMOTE and feature selection are placed inside the
  cross-validation pipeline.

## Models Compared

- Dummy most-frequent classifier.
- Logistic Regression with class weighting.
- Random Forest with class weighting.
- RBF-kernel SVM with class weighting.
- Gradient Boosting.
- XGBoost with class weighting.

## Evaluation Policy

The primary evaluation metrics are:

- fail recall,
- fail F1,
- PR-AUC,
- balanced accuracy,
- confusion matrix counts, especially missed fail samples.

Accuracy is intentionally not emphasized because an all-pass classifier can
look strong under severe class imbalance while failing the engineering task.

## Current Final Snapshot

The current curated snapshot uses:

- 5-fold stratified cross-validation,
- median imputation,
- SMOTE inside the training fold,
- top-50 selected features,
- weighted XGBoost as the best model by fail-class F1 in the final benchmark.

| Model | Fail recall | Fail precision | Fail F1 | PR-AUC | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost weighted | 0.615 | 0.141 | 0.229 | 0.134 | 0.674 |

Final confusion matrix for the XGBoost benchmark:

- true pass predicted pass: 1072
- true pass predicted fail: 391
- true fail predicted pass: 40
- true fail predicted fail: 64

## Threshold Tradeoff

For rare-defect screening, the default 0.50 probability threshold is not always
the most useful operating point. A threshold analysis was added for the final
XGBoost workflow.

Selected threshold for target fail recall `>= 0.70`:

- threshold: 0.15
- fail recall: 0.810
- fail precision: 0.117
- fail F1: 0.205
- detected fails: 17
- missed fails: 4
- false alarms: 128

This shows the production tradeoff clearly: increasing fail recall reduces
missed defects but increases false alarms.

## Hyperparameter Tuning Snapshot

A focused `RandomizedSearchCV` smoke run was added for the class-weighted
Logistic Regression baseline under the SMOTE + median-imputation + top-30
feature workflow.

Best tuning result:

- best CV fail F1: `0.176`
- best parameter: `model__C = 0.00195`
- holdout fail recall: `0.667`
- holdout fail precision: `0.126`
- holdout fail F1: `0.212`
- holdout confusion matrix: 14 detected fails, 7 missed fails, and 97 false
  alarms

The tuning snapshot is intentionally presented as a reproducible optimization
harness rather than a final claim that Logistic Regression is the best model.
The next stronger optimization pass should focus on the final XGBoost workflow.

## Limitations

- The SECOM feature names are anonymized, so sensor-level physical
  interpretation is limited.
- The final model is a portfolio benchmark, not a qualified fab deployment
  model.
- False-alarm cost is not known, so threshold selection uses recall/F1 rather
  than a real economic cost function.
- Additional validation would be needed on line-specific or time-split data
  before production use.

## Next Improvements

- Run longer XGBoost hyperparameter tuning with more iterations.
- Compare top-30, top-50, and top-80 feature settings under identical splits.
- Add cost-sensitive threshold selection if false-alarm and missed-defect costs
  are available.
- Add model calibration analysis for probability reliability.

## AI-Assisted Development Record

This project is intentionally documented as AI-assisted development. The user
defined the semiconductor problem, evaluation priorities, and approval gates.
Codex assisted with implementation, validation commands, result summarization,
and documentation after each approved direction.

See:

- `docs/CODEX_WORKFLOW.md`
- `docs/PROMPT_HARNESS.md`
- `docs/COMMAND_LOG.md`
