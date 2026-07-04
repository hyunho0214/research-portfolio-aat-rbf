# 포트폴리오 설명 브리프

## 프로젝트 한 줄 설명

SECOM 반도체 공정 센서 데이터를 이용해 희귀 불량 샘플을 탐지하고, 전처리
전략과 머신러닝 모델을 비교한 AI-assisted 반도체 수율 불량 탐지 프로젝트.

## 왜 이 프로젝트를 만들었나

반도체 양산 라인은 정상 샘플이 대부분이기 때문에 단순 Accuracy는 쉽게
착시를 만든다. SECOM 데이터도 정상(`-1`)이 대부분이고 불량(`1`)은 매우
적어서, 모든 샘플을 정상으로 예측해도 높은 정확도가 나올 수 있다.

이 프로젝트는 그런 통계적 함정을 피하기 위해 fail recall, fail F1, PR-AUC,
balanced accuracy, confusion matrix를 중심으로 모델을 평가한다.

## 해결하려는 핵심 문제

- 극단적인 class imbalance 때문에 Accuracy가 신뢰하기 어려움.
- 센서 측정 지연이나 누락으로 결측치가 많음.
- 590개 수준의 고차원 센서 feature 안에 고정값 센서와 노이즈 feature가 섞여 있음.
- 불량을 더 많이 잡으려면 false alarm이 증가하는 공정 엔지니어링 tradeoff가 있음.

## 구현한 접근

- 결측치 비율이 높은 feature를 제거.
- median, KNN 등 여러 imputation 전략을 비교.
- Variance Threshold로 거의 변하지 않는 feature를 제거.
- Random Forest 기반 feature importance로 상위 sensor feature를 선택.
- SMOTE를 cross-validation의 training fold 내부에만 적용해 data leakage를 방지.
- Logistic Regression, Random Forest, SVM, Gradient Boosting, XGBoost를 비교.
- threshold analysis로 recall과 false alarm 사이의 tradeoff를 분석.

## 현재 결과 요약

최종 스냅샷에서는 5-fold stratified cross-validation, median imputation,
SMOTE, top-50 feature selection 조건에서 weighted XGBoost가 가장 강한
fail-class F1을 보였다.

| 항목 | 값 |
| --- | ---: |
| Fail recall | 0.615 |
| Fail precision | 0.141 |
| Fail F1 | 0.229 |
| PR-AUC | 0.134 |
| Balanced accuracy | 0.674 |

Threshold를 0.15로 낮추면 fail recall은 0.810까지 올라갔지만 false alarm도
함께 증가했다. 이 결과는 실제 공정에서 불량 미검출 비용과 false alarm 비용을
함께 고려해야 한다는 개선 방향을 보여준다.

## AI 활용 방식

이 프로젝트는 Codex를 수동적으로 의존한 결과물이 아니라, 사용자가 문제 정의와
진행 방향을 주도한 AI-assisted 개발 사례로 정리했다.

작업 방식은 다음과 같다.

1. 사용자가 반도체 공정 데이터의 문제와 평가 기준을 정의한다.
2. Codex가 다음 구현 또는 실험 방향을 제안한다.
3. 사용자가 그 방향을 승인하거나 수정한다.
4. Codex가 승인된 범위만 구현하고 validation harness를 실행한다.
5. 사용자가 결과를 보고 다음 실험 방향을 다시 결정한다.

이 기록은 `docs/CODEX_WORKFLOW.md`, `docs/PROMPT_HARNESS.md`,
`docs/COMMAND_LOG.md`에 남겨 GitHub에서 AI 활용 능력도 함께 증빙할 수
있도록 했다.

## 면접에서 강조할 포인트

- 반도체 수율 데이터에서는 Accuracy보다 fail recall과 missed-fail count가
  더 중요하다는 점을 명확히 인식했다.
- SMOTE를 전체 데이터에 먼저 적용하지 않고 cross-validation pipeline 내부에
  넣어 data leakage를 방지했다.
- 단일 모델 성능만 보여주지 않고 전처리, feature selection, threshold까지
  함께 비교했다.
- AI를 단순 코드 생성기가 아니라 실험 설계, 검증 자동화, 문서화 파트너로
  활용했다.
