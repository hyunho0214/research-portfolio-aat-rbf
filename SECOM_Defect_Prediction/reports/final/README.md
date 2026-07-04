# Final Report Snapshot

This folder contains the curated report artifacts selected for the GitHub
portfolio snapshot. Raw SECOM files and intermediate experiment folders are not
tracked.

## Main Benchmark

Configuration:

- 5-fold stratified cross-validation
- SMOTE inside the training fold
- median imputation
- 50 selected sensor features
- model comparison across dummy, Logistic Regression, Random Forest, SVM,
  Gradient Boosting, and XGBoost

Best model in this snapshot:

| Model | Fail recall | Fail precision | Fail F1 | PR-AUC | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost weighted | 0.615 | 0.141 | 0.229 | 0.134 | 0.674 |

Confusion matrix summary:

- detected fails: 64
- missed fails: 40
- false alarms: 391
- correctly passed samples: 1072

## Threshold Tradeoff

The default 0.50 threshold is not necessarily the best production decision.
With the XGBoost model, a threshold of 0.15 was selected for a target fail
recall of 0.70:

- fail recall: 0.810
- fail precision: 0.117
- fail F1: 0.205
- detected fails: 17
- missed fails: 4
- false alarms: 128

This demonstrates the key process-engineering tradeoff: catching more rare
defects requires accepting more false alarms unless model precision is further
improved.

For the full portfolio-level interpretation, see `../../MODEL_CARD.md`.

## Files

- `metrics_summary.csv`: cross-validated model comparison.
- `confusion_matrices.csv`: model-level confusion matrix counts.
- `experiment_best_summary.csv`: preprocessing-strategy comparison.
- `model_metric_comparison.png`: fail-class recall/F1/PR-AUC by model.
- `best_model_confusion_matrix.png`: confusion matrix for the best model.
- `preprocessing_comparison.png`: best model across preprocessing strategies.
- `threshold_tradeoff.csv`: threshold-level recall/precision/F1 tradeoff.
- `threshold_tradeoff.png`: threshold tradeoff plot.
- `recommended_threshold.json`: selected threshold and reason.
- `selected_features.csv`: selected final top-k sensor list.
- `feature_importance.csv`: selected feature-importance table.
- `class_distribution.csv`: pass/fail count table.
- `missingness_summary.csv`: sensor missing-rate table.
- `tuning_logistic_best_params.json`: focused RandomizedSearchCV best parameter.
- `tuning_logistic_cv_results.csv`: tuning CV results table.
- `tuning_logistic_holdout_classification_report.json`: tuned holdout report.
- `tuning_logistic_holdout_confusion_matrix.csv`: tuned holdout confusion matrix.
- `top_feature_importance.png`: top selected sensor importances.
- `class_distribution.png`: SECOM pass/fail imbalance.
- `missingness_distribution.png`: missing-value rate distribution.
