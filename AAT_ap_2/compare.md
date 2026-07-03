# RBF2 vs AAT_ap 프로젝트 비교

## 1. 요약

`RBF2`와 `AAT_ap`는 모두 전력 수요 시계열을 sliding window로 바꾸고, Gaussian RBF network로 다음 시점의 `nat_demand`를 예측한다는 점에서 같은 계열의 프로젝트입니다. 두 프로젝트 모두 시간 순서를 유지한 train/test split, 입력/타깃 표준화, k-means 기반 RBF center, sigma diversity sweep, MSE/MAE/R2 평가를 사용합니다.

가장 큰 차이는 `RBF2`가 Figure 4 재현과 단일 forecasting 실험에 가까운 구조인 반면, `AAT_ap`는 논문용으로 재사용 가능한 CLI 기반 실험 파이프라인이며 forecasting과 Duffing reconstruction을 모두 지원한다는 점입니다.

또한 sigma 처리 방식이 크게 다릅니다. `RBF2`는 experimental sigma range를 모델 sigma로 직접 사용하고, `AAT_ap`는 experimental sigma의 상대적 diversity를 보존하되 standardized input distance scale에 맞게 재스케일합니다.

## 2. 공통점

| 항목 | 공통 내용 |
|---|---|
| 목적 | AAT-derived Gaussian width diversity가 RBF 예측 성능에 미치는 영향 확인 |
| 전력 데이터 | Panama electricity demand 계열의 `nat_demand` 사용 |
| 입력 구성 | 길이 `L=10` sliding window |
| 타깃 | 다음 시점의 전력 수요 |
| split 철학 | random shuffle 없이 시간 순서 유지 |
| 표준화 | `X_train`, `y_train` 기준으로 scaler fit 후 test transform |
| RBF feature | `exp(-||x-c||^2 / (2*sigma^2))` |
| center 생성 | k-means 기반 |
| metric | MSE, MAE, R2 |
| sigma sweep | 서로 다른 sigma 개수 `N`을 바꿔 성능 비교 |
| 기본 kernel 수 | 전력 예측에서 `C_total=200` |

## 3. 주요 차이점

| 항목 | RBF2 | AAT_ap |
|---|---|---|
| 실행 구조 | `main.py`, `figure_4.py` 중심 단일 실행 | `main_forecasting.py`, `main_duffing.py` CLI 기반 |
| 지원 task | 전력 수요 forecasting만 구현 | 전력 forecasting + Duffing reconstruction |
| 데이터 위치 | `data/continuous_dataset.csv` | 루트의 `continuous dataset.csv` |
| 데이터 로딩 | CSV에서 valid-looking line을 선별해 로딩 | pandas로 직접 로딩 후 required column 검증 |
| train/test split | `2020-01-01` 기준 split, 2016-2020 필터 | 기본 2016-2019 train, 2020 test |
| `N_values` | `[1, 3, 10, 20, 40, 60, 80]` | 기본 `1,2,3,...,80` |
| sigma source | config에 `SIGMA_MIN=2.9`, `SIGMA_MAX=19.8` 하드코딩 | `sigma.csv` 또는 `--sigma-values` 필수 입력 |
| N=1 sigma | geometric mean 사용 | 현재 scaling 후 selected median 기준으로 model sigma가 `d_ref`가 됨 |
| sigma scale | experimental sigma를 RBF sigma로 직접 사용 | experimental sigma ratio를 `d_ref`로 재스케일 |
| center 수 처리 | k-means를 `N`개 center로 수행 후 center를 복제해 `C_total`로 확장 | k-means를 `C_total`개 center로 수행해 distinct center 유지 |
| 중복 basis 가능성 | center와 sigma가 함께 복제되어 동일 basis function 반복 가능 | center는 distinct라 동일 center-sigma pair 반복을 피함 |
| output weight 학습 | ridge-regularized least squares | 기본 `np.linalg.lstsq` closed-form least squares |
| MLP baseline | 구현 및 main에서 실행 | 구현하지 않음 |
| plot | Figure 4h-k style 통합 그림 | metrics plot, best prediction overlay, Duffing phase/time plot |
| 결과 저장 | figure 중심, metrics CSV 저장 구조는 제한적 | `metrics.csv`, `best_predictions.csv`, `config.json`, `best_model_arrays.npz`, PNG 저장 |
| 문서 | 코드 주석 중심 | `guide.md`, `report.md`, `manua.md`, `compare.md` |

## 4. Sigma 처리 비교

### RBF2

`RBF2`는 `src/config.py`에 sigma range를 직접 둡니다.

```python
SIGMA_MIN = 2.9
SIGMA_MAX = 19.8
SIGMA_CANDIDATES_COUNT = 2000
```

`src/utils.py`에서는 `N=1`일 때 geometric mean을 사용합니다.

```python
if N == 1:
    return np.array([np.sqrt(sigma_candidates.min() * sigma_candidates.max())])
```

`N>1`에서는 sigma range를 log-space로 나눕니다. 이렇게 선택된 sigma는 표준화된 `X_train`의 RBF 계산에 그대로 들어갑니다.

### AAT_ap

`AAT_ap`는 sigma 값을 코드에 하드코딩하지 않습니다. 실행 시 `sigma.csv` 또는 `--sigma-values`로 experimental sigma를 입력해야 합니다.

현재 방식은 다음과 같습니다.

```text
sigma_exp   = geomspace(sigma_min_exp, sigma_max_exp, N)
sigma_rel   = sigma_exp / median(sigma_exp)
d_ref       = median nearest-center distance in standardized X_train
sigma_model = sigma_rel * d_ref
```

이 방식은 AAT 실험에서 관찰된 width diversity와 ratio를 유지하면서, 실제 RBF 계산은 표준화된 입력 공간의 거리 scale에 맞춥니다.

## 5. Center 구성 비교

### RBF2

`RBF2`는 `N`개 k-means center를 먼저 구한 뒤, 이를 반복 복제해 `C_total=200`개 center로 확장합니다.

장점:

- supplementary pseudocode의 문자적 흐름과 가깝습니다.
- `N`개의 Gaussian group이라는 해석이 직관적입니다.

주의점:

- 같은 center와 같은 sigma가 반복되면 RBF design matrix에 동일 column이 생길 수 있습니다.
- `C_total`이 실제 표현력 증가보다 반복 count에 가까워질 수 있습니다.

### AAT_ap

`AAT_ap`는 처음부터 `C_total`개 k-means center를 구합니다. 그 다음 선택된 `N`개 sigma를 `C_total` center에 균등 배정합니다.

장점:

- 전체 kernel budget이 실제 distinct center로 유지됩니다.
- `N` 변화가 center 수 변화가 아니라 sigma diversity 변화로 분리됩니다.
- 동일 center-sigma basis 반복 문제를 피합니다.

주의점:

- supplementary pseudocode의 문자적 구현과는 다릅니다.
- Methods에서 “fixed center budget에 sigma diversity를 할당했다”는 설명이 필요합니다.

## 6. RBF 학습 방식 비교

| 항목 | RBF2 | AAT_ap |
|---|---|---|
| 선형 readout | 있음 | 있음 |
| bias column | 있음 | 있음 |
| solver | ridge: `(Phi^T Phi + alpha I)^-1 Phi^T y` | `np.linalg.lstsq` |
| regularization | `RIDGE_ALPHA=1e-4` | 없음 |
| multi-output | 기본 구조는 single-output 중심 | Duffing `[x, v]` multi-output 지원 |

`RBF2`의 ridge regularization은 numerical stability에 유리합니다. `AAT_ap`는 guide의 “linear least squares”를 더 직접적으로 따릅니다. 다만 향후 `AAT_ap`에서도 optional ridge를 추가하면 large `C_total`, large `N`, ill-conditioned feature matrix에서 더 안정적일 수 있습니다.

## 7. RBF2에만 있는 항목

- `figure_4.py`: Figure 4h-k style plot 생성.
- `figure_4.png`: 생성된 통합 figure.
- `src/mlp_baseline.py`: MLPRegressor baseline.
- config에 실험 상수 집중 관리.
- ridge regularization.
- paper reference point 출력.
- `N=1`에서 sigma geometric mean 사용.

## 8. AAT_ap에만 있는 항목

- `src/main_forecasting.py`, `src/main_duffing.py`: CLI 실행 구조.
- Duffing oscillator reconstruction 전체 파이프라인.
- sigma CSV/inline 입력 필수화.
- experimental sigma ratio를 standardized input distance scale로 재스케일하는 방식.
- `C_total` distinct center 방식.
- `outputs/` 아래 실행별 결과 저장.
- `config.json`, `best_predictions.csv`, `best_model_arrays.npz` 저장.
- `report.md`, `manua.md`, `guide.md`.
- forecasting과 Duffing에 공통으로 쓰는 modular utility 구조.

## 9. 논문 타당성 관점의 해석

`RBF2`는 supplementary pseudocode를 더 문자적으로 따라간 구현입니다. 특히 `N`개 center를 만들고 center를 복제하는 구조는 pseudocode 설명과 가깝습니다. 하지만 동일 basis function이 반복될 수 있어 fixed kernel budget의 의미가 약해질 수 있습니다.

`AAT_ap`는 논문 Methods에서 방어하기 쉬운 방향으로 정리한 구현입니다. `C_total`개의 distinct center를 유지하면서 sigma diversity만 바꾸므로, 실험의 독립변수가 `N`에 더 명확히 묶입니다. 또한 experimental sigma의 절대 단위를 그대로 표준화된 입력 공간에 넣지 않고, 상대적 diversity를 `d_ref`로 옮기는 방식은 device-domain과 model-domain의 scale mismatch를 줄입니다.

따라서 Figure 4j의 모양을 재현하는 데에는 `RBF2`의 구조가 더 가까울 수 있지만, 논문용으로 “왜 이렇게 구현했는가”를 설명하기에는 `AAT_ap`의 현재 구조가 더 방어적입니다.

## 10. 향후 통합 또는 검증 제안

두 프로젝트를 더 공정하게 비교하려면 아래 ablation을 같은 데이터와 같은 sigma 입력으로 실행하는 것이 좋습니다.

1. `RBF2` 방식: `N`개 center 후 복제, direct sigma, ridge.
2. `AAT_ap` 방식: `C_total` distinct center, scaled sigma, no ridge.
3. Hybrid A: `C_total` distinct center + direct sigma.
4. Hybrid B: `N` center 복제 + scaled sigma.
5. Hybrid C: `C_total` distinct center + scaled sigma + ridge.

이렇게 나누면 성능 차이가 center 전략 때문인지, sigma scaling 때문인지, ridge 때문인지 분리해서 볼 수 있습니다.
