# AAT_ap_2 실행 매뉴얼

## 1. 기본 방향

`AAT_ap_2`는 Forecasting과 Duffing reconstruction 모두 sigma를 `direct` 방식으로만 사용합니다. 즉, 입력된 experimental sigma 값을 표준화된 입력 거리 기준으로 다시 scale하지 않습니다.

필수 입력은 AAT 실험에서 얻은 sigma 값입니다. 예시 `sigma.csv` 형식은 아래처럼 한 열이면 충분합니다.

```csv
sigma
4.8
5.3
6.1
7.0
8.4
9.7
11.2
13.0
15.5
18.1
```

명령줄에서 직접 넣을 수도 있습니다.

```powershell
--sigma-values 4.8,5.3,6.1,7.0,8.4,9.7,11.2,13.0,15.5,18.1
```

## 2. Sigma 선택 방식

코드는 먼저 NaN, infinite, 0 이하 sigma를 제거합니다. 그 다음 cleaned sigma의 최소값과 최대값 사이에서 `N`개의 값을 log-space로 선택합니다.

```text
N = 1: sigma = sqrt(sigma_min * sigma_max)
N > 1: sigma_set = np.geomspace(sigma_min, sigma_max, N)
```

선택된 `sigma_set`은 그대로 RBF Gaussian width로 사용됩니다.

```text
sigma_model = sigma_set
```

따라서 `AAT_ap_2`에는 `--sigma-mode` 옵션이 없습니다. 이 폴더는 direct sigma 실험만 실행하도록 정리했습니다.

## 3. Forecasting 실행

기본 설정은 아래와 같습니다.

```text
N sweep       = 1,2,3,...,80
sigma mode   = direct only
N=1 sigma    = sqrt(sigma_min * sigma_max)
center mode  = replicated
readout      = ordinary least squares, np.linalg.lstsq
ridge alpha  = 0.0
window L     = 10
C_total      = 200
train split  = 2016-2019
test split   = available 2020 period
```

기본 실행:

```powershell
py src/main_forecasting.py --data "continuous dataset.csv" --sigma-csv sigma.csv --sigma-column sigma --output-dir outputs/forecasting
```

작은 smoke test:

```powershell
py src/main_forecasting.py --data "continuous dataset.csv" --sigma-csv sigma.csv --sigma-column sigma --n-values 1,2,3 --c-total 20 --output-dir outputs/smoke_forecasting
```

자주 바꾸는 옵션:

```text
--n-values          비교할 N 값. 생략하면 1..80.
--window-length     입력 window 길이. 기본 10.
--c-total           전체 RBF kernel 수. 기본 200.
--seed              k-means seed. 기본 0.
--center-mode       replicated 또는 distinct. 기본 replicated.
--ridge-alpha       output weight ridge alpha. 기본 0.0, 즉 OLS.
--train-start-year  기본 2016.
--train-end-year    기본 2019.
--test-year         기본 2020.
```

## 4. Figure 생성

권장 figure script는 `figure_panama.py`입니다. 이 파일은 전처리, sigma 선택, center 생성, RBF 학습, 예측, metric 계산을 `src/forecasting.py`의 공통 로직으로 수행합니다. 차이는 plot 구성뿐입니다.

PNG 저장 및 matplotlib UI 열기:

```powershell
py figure_panama.py --sigma-csv sigma.csv --sigma-column sigma --output figure_panama.png
```

PNG와 metrics CSV만 저장:

```powershell
py figure_panama.py --sigma-csv sigma.csv --sigma-column sigma --output figure_panama.png --show false
```

기본 MSE/R2 plot은 `N=1,2,3,...,80` 전체를 표시합니다. 예측 overlay subplot은 기본적으로 `N=1,3,10,80`을 보여줍니다.

## 5. Duffing Reconstruction 실행

Duffing도 Forecasting과 동일하게 direct sigma만 사용합니다. 기본 설정은 아래와 같습니다.

```text
N sweep         = 1,2,3,...,80
sigma mode     = direct only
N=1 sigma      = sqrt(sigma_min * sigma_max)
center mode    = distinct C_total k-means centers
readout        = ordinary least squares, np.linalg.lstsq
C_total        = 300
train fraction = 0.5
```

기본 실행:

```powershell
py src/main_duffing.py --sigma-csv sigma.csv --sigma-column sigma --output-dir outputs/duffing
```

작은 smoke test:

```powershell
py src/main_duffing.py --sigma-csv sigma.csv --sigma-column sigma --n-values 1,2 --c-total 20 --n-steps 3000 --transient-steps 300 --output-dir outputs/smoke_duffing
```

자주 바꾸는 Duffing 옵션:

```text
--n-values          기본 1..80
--window-length     기본 10
--c-total           기본 300
--train-fraction    기본 0.5
--transient-steps   기본 2000
--alpha             기본 1.0
--beta              기본 0.2
--gamma             기본 0.3
--omega             기본 1.2
--dt                기본 0.01
--n-steps           기본 20000
--x0                기본 0.1
--v0                기본 0.0
```

## 6. 결과 파일

`src/main_forecasting.py`와 `src/main_duffing.py` 실행 결과는 `--output-dir` 아래에 저장됩니다.

```text
metrics.csv              N별 metric
best_predictions.csv     best N의 실제값과 예측값
config.json              실행 설정
best_model_arrays.npz    center, sigma, weight 배열
*.png                    metric plot과 prediction/reconstruction plot
```

`figure_panama.py`는 지정한 PNG와 같은 이름의 metrics CSV를 함께 저장합니다.

```text
figure_panama.png
figure_panama_metrics.csv
```

## 7. 주의사항

sigma 값을 입력하지 않으면 실행은 중단됩니다. 논문용 코드에서 sigma를 임의 placeholder로 채우지 않기 위한 의도적인 설계입니다.

전력 데이터는 원본 시간별 해상도를 유지합니다. 기본 test set은 CSV에 존재하는 2020년 구간, 즉 사용 가능한 2020 period입니다.

현재 `AAT_ap_2`의 핵심 실험 조건은 `replicated center + direct sigma + OLS`입니다.
