# Project Plan: Python-Based Semiconductor Defect Prediction and Hyperparameter Optimization

## Portfolio Positioning

This project should read as an applied semiconductor manufacturing analytics
case study rather than a generic Kaggle classification notebook.

It should also read as an AI-assisted development case study. The intended
message is that the project owner used Codex actively and critically: defining
the manufacturing problem, approving the direction, requiring validation
harnesses, and interpreting the results instead of passively accepting generated
code.

Recommended Korean title:

> Python 기반 반도체 불량 예측 및 하이퍼파라미터 최적화

Recommended English subtitle:

> Rare-failure classification on SECOM process-sensor data with leakage-safe
> preprocessing, imbalance handling, feature selection, and model comparison.

## Why This Project Matters

SECOM is a good portfolio dataset because it resembles real process-monitoring
work:

- yield is high, so the fail class is rare,
- sensor measurements contain missing values,
- hundreds of anonymous process variables contain redundant and noisy signals,
- the cost of missing a defect is usually higher than the cost of a false alarm,
- feature ranking can be discussed as a process-engineering decision-support
  tool, not just as a machine-learning trick.

## Core Technical Challenges

### 1. Extreme Class Imbalance

Problem:

- Pass labels dominate the dataset.
- A model can obtain high accuracy by predicting every sample as pass.

Implementation:

- Treat fail (`1`) as the positive class.
- Report recall, F1, PR-AUC, balanced accuracy, and confusion matrix.
- Compare class-weighted models against SMOTE-based training.
- Keep a dummy baseline to expose the accuracy trap.

### 2. Missing Values

Problem:

- Some sensors are unavailable for many observations.
- Naive mean imputation may erase process structure.

Implementation:

- Drop columns above a configurable missing-rate threshold, starting at 50%.
- Compare median imputation and KNN imputation.
- Keep a forward-fill option as a process-flow-inspired baseline, but use it
  carefully because SECOM timestamps are attached to labels rather than a fully
  documented time-series sequence.

### 3. High Dimensionality, Multicollinearity, and Noise

Problem:

- Many sensors are constant, near-constant, redundant, or unrelated to failure.

Implementation:

- Apply `VarianceThreshold`.
- Optionally drop highly correlated features.
- Use Random Forest feature importance to select top 30, 50, and 80 sensors.
- Save selected sensor IDs and feature importance as report artifacts.

## Experiment Matrix

Recommended first pass:

| ID | Imputer | Resampling | Feature selection | Models |
| --- | --- | --- | --- | --- |
| A | median | none | variance only | dummy, logistic, random forest |
| B | median | SMOTE | top 50 RF importance | logistic, random forest, SVM |
| C | KNN | SMOTE | top 50 RF importance | logistic, random forest, gradient boosting |
| D | KNN | class weights | top 30/50/80 RF importance | random forest, XGBoost |

## Hyperparameter Search Plan

Use `RandomizedSearchCV` before moving to heavier Bayesian optimization.

Implemented entry point:

```powershell
py -3 tune_model.py --download --model logistic_balanced --output-dir reports\tuning_logistic_smote_median_top30 --splits 3 --n-iter 8 --top-k 30 --imputer median --resampling smote
```

Target search spaces:

- Logistic Regression: `C`, penalty, solver, class weight.
- Random Forest: `n_estimators`, `max_depth`, `min_samples_leaf`,
  `max_features`, class weight.
- SVM: `C`, `gamma`, kernel, class weight.
- XGBoost: `n_estimators`, `max_depth`, `learning_rate`, `subsample`,
  `colsample_bytree`, `scale_pos_weight`.

Primary scoring:

- `f1` for fail class during broad screening,
- recall-constrained selection for final model, e.g. choose the highest F1 among
  models with fail recall above a defined threshold.

## Deliverables

Minimum GitHub-ready outputs:

- clean `README.md` with problem framing and result table,
- `BASELINE_NOTES.md` or equivalent result notes that explain the first
  benchmark,
- reproducible CLI script,
- hyperparameter-tuning CLI script,
- threshold-analysis CLI script for recall/false-alarm tradeoff,
- saved `metrics_summary.csv`,
- confusion-matrix plot,
- PR curve plot,
- top sensor importance chart,
- preprocessing comparison plot across no-SMOTE, SMOTE+median, and SMOTE+KNN
  experiments,
- short discussion of which preprocessing choice helped most,
- short discussion of the false-negative tradeoff.

## Suggested Final README Story

1. Explain why 94% accuracy is a trap.
2. Show the class-distribution plot.
3. Show missingness distribution and dropped columns.
4. Compare preprocessing variants.
5. Compare models using recall/F1/PR-AUC.
6. Present the best model and confusion matrix.
7. Interpret top sensors as process-monitoring candidates.
8. State limitations: anonymized sensors, small fail count, public benchmark
   data, and limited process-context interpretability.

## AI-Assisted Development Evidence

The GitHub project should include:

- the prompt structure used to guide Codex,
- a command log showing what was run and what was observed,
- a validation harness that can compile or rerun benchmarks,
- notes showing where the user corrected Codex's direction,
- final discussion that separates human process judgment from AI-generated
  implementation support.

## Completion Criteria

The project is complete when:

- the pipeline runs from raw SECOM files to final reports,
- SMOTE is applied only on training folds,
- at least three model families are compared,
- at least two imputation strategies are compared,
- at least one feature-selection strategy is used,
- final metrics include recall, F1, PR-AUC, ROC-AUC, balanced accuracy, and a
  confusion matrix,
- reports and figures are saved under `reports/`,
- the README explains practical semiconductor meaning rather than only model
  scores.
