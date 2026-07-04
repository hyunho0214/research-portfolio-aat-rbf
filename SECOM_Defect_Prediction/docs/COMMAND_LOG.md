# Command Log

This log summarizes the commands used while building the SECOM project with
Codex. It is not a full terminal transcript; it is a curated record of the
prompts, checks, and evidence that matter for a portfolio review.

## User Direction and Approval Pattern

The project was intentionally framed as active AI-assisted development rather
than passive code generation. The user direction was:

```text
AI-assisted semiconductor yield-defect detection program으로 정리하라.
Codex에 수동적으로 의존한 것이 아니라, Codex가 진행 방향을 제안하고
사용자가 허락하거나 수정한 뒤 다음 단계로 넘어가는 방식으로 기록하라.
```

This created the operating rule used for the rest of the project:

- Codex proposes an experiment or implementation direction.
- The user approves, corrects, or narrows that direction.
- Codex implements only the approved scope.
- Codex runs the validation harness and records the result.
- The next step is selected from the evidence, not from generated code alone.

## Environment Check

```powershell
py -3 -c "import importlib.util; mods=['numpy','pandas','sklearn','scipy','matplotlib','imblearn','xgboost','yaml']; [print(m, bool(importlib.util.find_spec(m))) for m in mods]"
```

Initial finding:

- Available: NumPy, pandas, scikit-learn, SciPy, matplotlib, PyYAML.
- Missing initially: imbalanced-learn, XGBoost.

## Dependency Installation

```powershell
py -3 -m pip install imbalanced-learn xgboost
```

Installed:

- `imbalanced-learn 0.14.2`
- `xgboost 3.3.0`

## XGBoost Label Handling Check

```powershell
py -3 -c "from xgboost import XGBClassifier; import numpy as np; X=np.random.default_rng(0).normal(size=(20,3)); y=np.array([-1,1]*10); clf=XGBClassifier(n_estimators=5, eval_metric='logloss'); clf.fit(X,y)"
```

Finding:

- XGBoost rejected SECOM labels `[-1, 1]` because it expects `[0, 1]`.
- The project added `BinaryLabelMappingClassifier` so model internals can use
  zero-based labels while project-level metrics still use `-1=pass`, `1=fail`.

## Baseline Benchmark

```powershell
py -3 run_experiment.py --download --output-dir reports/smoke --splits 3 --top-k 30 --resampling none
```

Best result:

- Logistic Regression, class-weighted.
- Fail recall: 0.548.
- Fail F1: 0.219.

## SMOTE and Imputation Benchmarks

```powershell
py -3 run_experiment.py --download --output-dir reports/smote_median_top30 --splits 3 --top-k 30 --imputer median --resampling smote
py -3 run_experiment.py --download --output-dir reports/smote_knn_top30 --splits 3 --top-k 30 --imputer knn --resampling smote
py -3 scripts\compare_reports.py --reports reports\smoke reports\smote_median_top30 reports\smote_knn_top30 --names no_smote_median smote_median smote_knn --output-dir reports\comparison
```

Finding:

- SMOTE + median imputation slightly improved fail F1 and PR-AUC in the first
  top-30 run.
- KNN imputation did not automatically outperform median imputation.

## Stronger 5-Fold Benchmark

```powershell
py -3 run_experiment.py --download --output-dir reports\smote_median_top50_5fold --splits 5 --top-k 50 --imputer median --resampling smote
```

Best result:

- XGBoost weighted.
- Fail recall: 0.615.
- Fail F1: 0.229.
- Confusion matrix: 64 detected fails, 40 missed fails, 391 false alarms.

## Hyperparameter Tuning Smoke

```powershell
py -3 tune_model.py --download --model logistic_balanced --output-dir reports\tuning_logistic_smote_median_top30 --splits 3 --n-iter 8 --top-k 30 --imputer median --resampling smote
```

Observed result:

- Best CV fail F1: 0.176.
- Best parameter: `model__C = 0.00195`.
- Holdout fail recall: 0.667.
- Holdout fail F1: 0.212.

## Threshold Tradeoff Analysis

```powershell
py -3 threshold_analysis.py --download --model xgboost_weighted --output-dir reports\threshold_xgboost_smote_median_top50 --top-k 50 --imputer median --resampling smote --target-recall 0.70
```

Observed result:

- Selected threshold: 0.15.
- Fail recall: 0.810.
- Fail precision: 0.117.
- Missed fails: 4.
- False alarms: 128.

## Final Snapshot

Selected report artifacts were copied into:

```text
reports/final/
```

These artifacts are intentionally allowed by `.gitignore` so the GitHub version
can show final results without committing raw SECOM data or every intermediate
report folder.

## Git Hygiene Checks

```powershell
git check-ignore -v -- "SECOM_Defect_Prediction\data\raw\secom.data" "SECOM_Defect_Prediction\reports\smote_median_top30\model_metric_comparison.png" "SECOM_Defect_Prediction\reports\final\metrics_summary.csv"
git diff --check -- SECOM_Defect_Prediction
py -3 -m py_compile SECOM_Defect_Prediction\run_experiment.py SECOM_Defect_Prediction\tune_model.py SECOM_Defect_Prediction\threshold_analysis.py SECOM_Defect_Prediction\scripts\compare_reports.py SECOM_Defect_Prediction\src\secom_defect\data.py SECOM_Defect_Prediction\src\secom_defect\modeling.py SECOM_Defect_Prediction\src\secom_defect\reporting.py
```
