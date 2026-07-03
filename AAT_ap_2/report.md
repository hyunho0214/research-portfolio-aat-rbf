# AAT_ap_2 Multi-Gaussian RBF 구현 보고서

## 1. 전체 로직 흐름

`AAT_ap_2`는 AAT 실험에서 얻은 Gaussian width sigma 값의 다양성이 RBF 예측 성능에 어떤 영향을 주는지 확인하기 위한 코드입니다. 현재 이 폴더의 기본 정책은 Forecasting과 Duffing reconstruction 모두 `direct sigma only`입니다.

전력 수요 예측은 `continuous dataset.csv`에서 시간별 `nat_demand` 단일 시계열만 사용합니다. 데이터는 시간순으로 정렬되고, 기본값으로 2016-2019년을 train, CSV에 존재하는 2020년 구간을 test로 사용합니다. 입력은 길이 10의 과거 전력 수요 window이고, target은 다음 1시간의 `nat_demand`입니다.

Duffing reconstruction은 RK4로 Duffing oscillator를 생성한 뒤 transient를 제거합니다. 입력은 10개 연속 상태 `[x, v]`를 flatten한 vector이고, target은 다음 상태 `[x(t+1), v(t+1)]`입니다. 기본 split은 chronological 50/50입니다.

두 파이프라인 모두 `X`와 `y`를 train split에 대해서만 fit한 `StandardScaler`로 표준화합니다. 예측 후 metric은 inverse transform한 원 단위 값으로 계산합니다. 이 처리는 train/test leakage를 피하고, RBF 계산이 안정적인 수치 범위에서 이루어지도록 하기 위한 단계입니다.

## 2. Sigma 처리

sigma 값은 코드 내부에 임의로 넣지 않고, 실행 시 `--sigma-csv` 또는 `--sigma-values`로 반드시 입력받습니다. 입력 sigma에서 NaN, infinite, 0 이하 값을 제거한 뒤, cleaned sigma의 최소값과 최대값을 사용합니다.

`N=1`일 때는 한쪽 끝값을 쓰지 않고 geometric mean을 사용합니다.

```text
sigma = sqrt(sigma_min * sigma_max)
```

`N > 1`일 때는 experimental sigma range에서 log-space로 `N`개를 선택합니다.

```text
sigma_set = np.geomspace(sigma_min, sigma_max, N)
```

선택된 sigma는 별도 scale 변환 없이 RBF width로 직접 사용합니다.

```text
sigma_model = sigma_set
```

따라서 `AAT_ap_2`에서는 선택된 experimental sigma를 그대로 model sigma로 사용합니다. 이 결정은 RBF2/Figure-style 실험 흐름과 맞추기 위한 것이며, 현재 폴더에서는 direct sigma 결과를 중심으로 비교합니다.

## 3. Center와 Kernel Budget

Forecasting 기본값은 `center_mode=replicated`입니다. 각 `N`에 대해 `KMeans(n_clusters=N)`를 수행한 뒤, 얻은 center와 sigma를 `C_total`까지 복제합니다. 이 방식은 RBF2의 Figure 4식 흐름에 가깝게 `N` 증가에 따라 unique center 수와 sigma diversity가 함께 변하게 만듭니다.

비교용으로 `center_mode=distinct`도 남아 있습니다. 이 경우 `KMeans(n_clusters=C_total)`로 고정된 center budget을 만든 뒤, 선택된 sigma를 center들에 균등 배정합니다.

Duffing은 `C_total`개의 distinct k-means center를 기본으로 사용하고, 선택된 direct sigma를 center들에 균등 배정합니다.

## 4. Output Weight 학습

RBF hidden feature를 계산한 뒤, bias column을 추가하고 output weight를 `np.linalg.lstsq`로 학습합니다. 기본 `ridge_alpha=0.0`이므로 ordinary least squares입니다.

```text
Phi_aug @ W ~= y
W = np.linalg.lstsq(Phi_aug, y)
```

양수 `--ridge-alpha`를 주면 ridge readout을 사용할 수 있지만, 기본 논문용 실행은 OLS입니다.

## 5. Figure Script 정리

기존 `figure_4.py`는 삭제했습니다. 현재 figure 생성은 `figure_panama.py`가 담당합니다.

`figure_panama.py`는 전처리, sigma 선택, center 생성, RBF 학습, 예측, metric 계산을 `src/forecasting.py`의 공통 helper로 수행합니다. 따라서 같은 옵션으로 실행하면 `src/main_forecasting.py`와 같은 metric 결과가 나와야 하며, 차이는 plot 구성뿐입니다.

## 6. 출력 파일

Forecasting과 Duffing 실행 결과는 지정한 `--output-dir` 아래에 저장됩니다.

- `metrics.csv`: 각 `N`별 성능 지표
- `best_predictions.csv`: best N의 실제값과 예측값
- `config.json`: 실행 파라미터와 sigma range
- `best_model_arrays.npz`: centers, sigma, weights 배열
- PNG plots: metric sweep과 prediction/reconstruction plot

`figure_panama.py`는 지정한 PNG와 같은 이름의 metrics CSV를 저장합니다.

## 7. 재현성

기본 `N_values`는 `1,2,3,...,80`입니다. k-means와 sigma assignment shuffle은 `--seed`로 제어합니다. Forecasting 기본 `C_total`은 200이고, Duffing 기본 `C_total`은 300입니다.
