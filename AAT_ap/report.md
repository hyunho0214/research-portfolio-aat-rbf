# AAT Multi-Gaussian RBF 구현 보고서

## 1. 전체 로직 흐름

이 프로젝트는 AAT 실험에서 얻은 Gaussian width 값의 다양성이 RBF 예측 성능에 어떤 영향을 주는지 확인하기 위한 코드입니다. 핵심 실험 조건은 전체 RBF kernel 수 `C_total`은 고정하고, 서로 다른 sigma 값의 개수 `N`만 바꾸는 것입니다.

전력 수요 예측 파이프라인은 `continuous dataset.csv`에서 시간별 `nat_demand`만 사용합니다. 데이터는 시간순으로 정렬한 뒤 2016-2019년을 train, 2020년의 사용 가능한 구간을 test로 나눕니다. 각 입력은 길이 10의 과거 전력 수요 window이고, target은 바로 다음 시간의 `nat_demand`입니다.

Duffing reconstruction 파이프라인은 RK4 방법으로 Duffing oscillator를 생성합니다. 초기 transient를 제거한 뒤, 10개 연속 상태 `[x, v]`를 flatten하여 입력으로 만들고 다음 상태 `[x(t+1), v(t+1)]`를 예측합니다. train/test split은 시간 순서를 유지하는 50/50 split입니다.

두 파이프라인 모두 `X`와 `y`를 train split에 대해서만 fit한 `StandardScaler`로 표준화합니다. 예측값은 metric 계산 전에 원래 단위로 inverse transform합니다. 이 처리는 Euclidean distance 기반 RBF와 sigma scale의 상호작용을 안정적으로 만들고, train/test data leakage를 피하기 위한 필수 단계입니다.

## 2. 주요 구현 결정과 이유

### C_total개의 distinct center 사용

보조 pseudocode를 문자 그대로 따르면 `N`개 center를 반복 복제해 `C_total` kernel을 만들 수 있습니다. 하지만 center와 sigma가 모두 같은 basis function이 반복되면 design matrix의 column이 중복되어 rank와 표현력이 불필요하게 약해집니다.

따라서 본 구현은 `KMeans(n_clusters=C_total)`로 `C_total`개의 distinct center를 만들고, 선택된 `N`개의 sigma를 center들에 균등 배정합니다. 이 방식은 kernel budget을 고정하면서 sigma diversity만 변화시키는 실험 목적을 더 명확하게 보존합니다.

### Sigma 값 처리

sigma 값은 코드 내부에 임의로 넣지 않았습니다. 논문용 시뮬레이션에서 RBF width는 AAT 실험값과 연결되어야 하므로, 실행 시 `--sigma-csv` 또는 `--sigma-values`로 반드시 제공해야 합니다.

입력 sigma에서 NaN, infinite, nonpositive 값을 제거한 뒤, 남은 실험값의 `min`과 `max` 사이를 `np.geomspace`로 나누어 `N`개의 experimental width를 선택합니다. 이는 width가 scale 변화에 민감한 양수 파라미터라는 점에서 선형 간격보다 방어하기 쉬운 선택입니다.

선택된 experimental sigma의 절대값은 RBF 계산에 그대로 넣지 않습니다. AAT device-domain sigma와 표준화된 입력 window의 Euclidean distance scale이 같다고 가정하기 어렵기 때문입니다. 대신 선택된 sigma를 그 집합의 median으로 나누어 상대 폭 `sigma_rel`로 만들고, train split에서 각 sample이 nearest k-means center까지 갖는 거리의 median `d_ref`를 곱합니다.

```text
sigma_exp   = geomspace(sigma_min_exp, sigma_max_exp, N)
sigma_rel   = sigma_exp / median(sigma_exp)
d_ref       = median nearest-center distance in standardized X_train
sigma_model = sigma_rel * d_ref
```

이 방식은 AAT 실험에서 관찰된 width diversity와 ratio를 보존하면서, 실제 RBF 계산은 표준화된 입력 공간의 거리 단위에 맞춥니다.

### Chronological split

시계열 데이터는 random shuffle을 하지 않습니다. 전력 수요는 calendar split을 사용하고, Duffing은 transient 제거 후 chronological fraction split을 사용합니다. 이 결정은 미래 정보를 train에 섞는 data leakage를 피하기 위한 것입니다.

### Least-squares readout

RBF hidden feature는 고정하고, output weight는 `np.linalg.lstsq`로 닫힌 형태의 선형 readout을 학습합니다. 이는 guide의 요구와 RBF network의 기본 구조에 맞으며, 불필요한 nonlinear optimizer 변수를 도입하지 않습니다.

## 3. 출력 파일

각 실행은 지정한 `--output-dir` 아래에 다음 파일을 저장합니다.

- `metrics.csv`: 각 `N`별 metric.
- `best_predictions.csv`: 가장 낮은 MSE 기준 best model의 true/predicted 값.
- `config.json`: 실행 파라미터, experimental sigma range, `d_ref`, split 정보, best N.
- `best_model_arrays.npz`: centers, experimental sigma set, relative sigma set, model-space sigma set, center별 sigma assignment, least-squares weights.
- PNG plots: metric sweep과 best prediction/reconstruction plot.

## 4. 코드 구조

- `src/data_utils.py`: 시간 정렬, split, sliding window, 표준화.
- `src/sigma_utils.py`: sigma loading/cleaning/selection/assignment.
- `src/rbf.py`: k-means center, RBF feature, least-squares model.
- `src/forecasting.py`: 전력 수요 예측 실험 runner.
- `src/duffing.py`: Duffing simulation과 reconstruction runner.
- `src/metrics.py`: forecasting 및 Duffing metric.
- `src/plotting.py`: 결과 plot 저장.
- `src/main_forecasting.py`: forecasting CLI.
- `src/main_duffing.py`: Duffing CLI.

## 5. 재현성

`seed`는 k-means와 sigma assignment shuffle에 사용됩니다. 기본 `N_values`는 `1, 2, 3, ..., 80`입니다. 전력 예측의 기본 `C_total`은 200, Duffing reconstruction의 기본 `C_total`은 300입니다.
