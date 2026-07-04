# Prompt Harness

This file records the reusable prompt structure used to guide Codex. It is
written as a portfolio artifact: the goal is to show how the user directed the
AI system, not to expose hidden system prompts.

## Initial Problem Prompt

```text
Kaggle SECOM 데이터를 이용해 "Python 기반 반도체 불량 예측 및
하이퍼파라미터 최적화" 프로젝트를 구성하고 싶다.

핵심 요구사항:
1. 정상(-1)이 대부분이고 불량(1)이 매우 적은 class imbalance를
   F1-score, recall, PR-AUC 중심으로 다룰 것.
2. 결측치가 많은 sensor column은 50% threshold로 drop하고, 남은 결측치는
   median, KNN imputer, forward-fill 후보로 비교할 것.
3. 590개 수준의 고차원 sensor feature에서 variance threshold와
   Random Forest / XGBoost importance 기반 feature selection을 구현할 것.
4. SMOTE는 data leakage가 없도록 training fold 내부에만 적용할 것.
5. 다양한 모델을 비교하고, 어떤 전처리와 모델 조합이 불량 검출에
   효과적인지 포트폴리오용 report로 정리할 것.
```

## Human-in-the-Loop Refinement Prompt

```text
이 프로젝트는 내가 직접 손코딩한 것이 아니라 Codex를 이용한 vibe coding
프로젝트로도 보여주고 싶다. GitHub에 어떤 명령을 Codex에게 줬고,
어떤 validation harness로 검증했는지도 함께 남겨라.

단, 수동적으로 Codex에 의존한 것처럼 보이면 안 된다. Codex가 진행 방향을
제안하고, 사용자가 허락하거나 수정한 뒤 다음 단계로 넘어가는 방식으로
정리하라.
```

## Direction-Approval Harness

```text
앞으로는 Codex가 혼자 모든 결정을 내려서 진행한 것처럼 쓰지 말고,
각 단계마다 다음 구조로 기록하라.

1. Codex가 다음 실험/구현 방향을 제안한다.
2. 사용자가 그 방향을 승인하거나 수정한다.
3. Codex가 승인된 범위만 구현한다.
4. Codex가 validation harness를 실행하고 결과를 요약한다.
5. 사용자가 결과를 보고 다음 방향을 다시 결정한다.

특히 데이터 전처리, SMOTE 적용 위치, 모델군 선택, 평가 지표, threshold
정책, GitHub에 남길 최종 산출물은 사용자 승인 게이트를 거친 것으로
문서화하라.
```

## Implementation Harness

```text
Build a reproducible Python project with:
- src/secom_defect/data.py for SECOM loading and UCI download fallback
- src/secom_defect/modeling.py for leakage-safe preprocessing and model comparison
- src/secom_defect/reporting.py for figures and report artifacts
- run_experiment.py for model comparison
- tune_model.py for RandomizedSearchCV
- threshold_analysis.py for recall/false-alarm tradeoff analysis
- scripts/compare_reports.py for preprocessing comparison
- scripts/run_validation.ps1 for compile/smoke/benchmark validation

Validation gates:
- Python files must compile.
- Raw data and generated reports must be ignored by Git.
- SMOTE benchmark must run and save metrics_summary.csv.
- The workflow must compare multiple model families.
- Final reports must include fail recall, fail F1, PR-AUC, balanced accuracy,
  confusion matrix, and feature importance.
```

## Validation Commands

Compile-only:

```powershell
.\scripts\run_validation.ps1 -Mode compile
```

Smoke benchmark:

```powershell
.\scripts\run_validation.ps1 -Mode smoke
```

Benchmark comparison:

```powershell
.\scripts\run_validation.ps1 -Mode benchmark
```

Focused tuning example:

```powershell
py -3 tune_model.py --download --model logistic_balanced --output-dir reports\tuning_logistic_smote_median_top30 --splits 3 --n-iter 8 --top-k 30 --imputer median --resampling smote
```

Threshold analysis example:

```powershell
py -3 threshold_analysis.py --download --model xgboost_weighted --output-dir reports\threshold_xgboost_smote_median_top50 --top-k 50 --imputer median --resampling smote --target-recall 0.70
```
