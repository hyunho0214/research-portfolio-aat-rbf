# Baseline Notes

Smoke run executed with the current scaffold:

```powershell
py -3 run_experiment.py --download --output-dir reports/smoke --splits 3 --top-k 30 --resampling none
```

Environment note:

- `imbalanced-learn` was not installed during this first smoke run, so SMOTE was
  not evaluated yet.
- `xgboost` was not installed during this first smoke run, so the XGBoost
  benchmark remains a planned follow-up.

## First Smoke Result

| Model | Fail recall | Fail precision | Fail F1 | PR-AUC | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression, class-weighted | 0.548 | 0.137 | 0.219 | 0.163 | 0.651 |
| SVM RBF, class-weighted | 0.240 | 0.160 | 0.192 | 0.113 | 0.575 |
| Gradient Boosting | 0.029 | 0.214 | 0.051 | 0.155 | 0.511 |
| Random Forest, class-weighted | 0.019 | 1.000 | 0.038 | 0.194 | 0.510 |
| Dummy most-frequent | 0.000 | 0.000 | 0.000 | 0.066 | 0.500 |

## Interpretation

The initial result already demonstrates the main SECOM lesson:

- The dummy model catches zero fail samples.
- Class-weighted Logistic Regression catches the most fail samples in this
  preliminary run, but precision is low.
- Random Forest is conservative in this configuration: it avoids false alarms
  but misses most fail samples.
- The next meaningful experiment is SMOTE inside cross-validation, then a
  threshold-tuning pass to choose a practical recall/precision tradeoff.

## Next Experiment

Dependencies were installed and two SMOTE experiments were executed:

```powershell
py -3 -m pip install imbalanced-learn xgboost
py -3 run_experiment.py --download --output-dir reports/smote_median_top30 --splits 3 --top-k 30 --imputer median --resampling smote
py -3 run_experiment.py --download --output-dir reports/smote_knn_top30 --splits 3 --top-k 30 --imputer knn --resampling smote
py -3 scripts\compare_reports.py --reports reports\smoke reports\smote_median_top30 reports\smote_knn_top30 --names no_smote_median smote_median smote_knn --output-dir reports\comparison
```

## SMOTE and Imputation Comparison

| Preprocessing strategy | Best model | Fail recall | Fail precision | Fail F1 | PR-AUC | Balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| No SMOTE + median imputer | Logistic Regression, class-weighted | 0.548 | 0.137 | 0.219 | 0.163 | 0.651 |
| SMOTE + median imputer | Logistic Regression, class-weighted | 0.558 | 0.138 | 0.221 | 0.174 | 0.655 |
| SMOTE + KNN imputer | XGBoost, weighted | 0.510 | 0.119 | 0.193 | 0.121 | 0.621 |

Current interpretation:

- SMOTE with median imputation slightly improved fail recall, fail F1, PR-AUC,
  and balanced accuracy in this 3-fold top-30 feature run.
- KNN imputation did not improve the best F1 in this first pass, even though it
  changed the best model from Logistic Regression to XGBoost.
- The next serious improvement path is not simply "use KNN"; it is threshold
  tuning plus hyperparameter search under a recall-first scoring policy.

Generated but ignored report artifacts:

- `reports/smote_median_top30/model_metric_comparison.png`
- `reports/smote_median_top30/best_model_confusion_matrix.png`
- `reports/smote_median_top30/top_feature_importance.png`
- `reports/comparison/preprocessing_comparison.png`

## Stronger 5-Fold Top-50 Benchmark

Command:

```powershell
py -3 run_experiment.py --download --output-dir reports\smote_median_top50_5fold --splits 5 --top-k 50 --imputer median --resampling smote
```

Best result:

| Model | Fail recall | Fail precision | Fail F1 | PR-AUC | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost, weighted | 0.615 | 0.141 | 0.229 | 0.134 | 0.674 |

Confusion matrix:

- true pass predicted pass: 1072
- true pass predicted fail: 391
- true fail predicted pass: 40
- true fail predicted fail: 64

Interpretation:

- XGBoost improved fail recall compared with the earlier top-30 runs.
- The tradeoff is high false alarms, which motivates threshold analysis and
  production-cost-based model selection.

## Next Optimization Pass

Short hyperparameter-tuning smoke run:

```powershell
py -3 tune_model.py --download --model logistic_balanced --output-dir reports\tuning_logistic_smote_median_top30 --splits 3 --n-iter 8 --top-k 30 --imputer median --resampling smote
```

Observed result:

- best CV fail F1: 0.176
- best parameter: `model__C = 0.00195`
- holdout fail recall: 0.667
- holdout fail precision: 0.126
- holdout fail F1: 0.212
- holdout confusion matrix: 196 true pass, 97 false alarms, 7 missed fails,
  and 14 detected fails

This confirms that the tuning path works, but also shows the central production
tradeoff: the tuned model catches more fail samples, while false alarms remain
high.

Recommended stronger next run:

```powershell
py -3 run_experiment.py --download --output-dir reports/smote_median_top50 --splits 5 --top-k 50 --imputer median --resampling smote
py -3 tune_model.py --download --model xgboost_weighted --output-dir reports\tuning_xgboost_smote_median_top50 --splits 5 --n-iter 30 --top-k 50 --imputer median --resampling smote
```

Further refinement after this:

- compare top-30, top-50, and top-80 feature settings under the same CV split,
- run a longer XGBoost tuning pass with more iterations,
- add a cost-sensitive selection rule if false-alarm cost is known.

Threshold analysis was added with:

```powershell
py -3 threshold_analysis.py --download --model xgboost_weighted --output-dir reports\threshold_xgboost_smote_median_top50 --top-k 50 --imputer median --resampling smote --target-recall 0.70
```

The selected threshold was 0.15:

- fail recall: 0.810
- fail precision: 0.117
- fail F1: 0.205
- missed fails: 4
- detected fails: 17
- false alarms: 128

Selected final outputs were copied into `reports/final/`.
