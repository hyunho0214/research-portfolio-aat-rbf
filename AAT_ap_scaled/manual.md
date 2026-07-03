# AAT_ap_3 실행 매뉴얼

## 1. 프로젝트 기본 원칙

`AAT_ap_3`는 forecasting과 Duffing 모두 sigma를 **scaled 방식으로만** 사용합니다. direct sigma 옵션은 없습니다.

scaled 방식은 AAT experimental sigma의 상대적 폭 비율을 보존하되, 실제 RBF 계산에 들어가는 sigma를 standardized input space의 거리 scale에 맞춥니다.

```text
sigma_exp   = geomspace(sigma_min_exp, sigma_max_exp, N)
sigma_rel   = sigma_exp / median(sigma_exp)
d_ref       = median nearest-center distance in standardized X_train
sigma_model = sigma_rel * d_ref
```

## 2. Sigma 입력 형식

sigma CSV는 한 열로 작성합니다.

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

또는 명령어에서 직접 입력할 수 있습니다.

```powershell
--sigma-values 4.8,5.3,6.1,7.0,8.4,9.7,11.2,13.0,15.5,18.1
```

논문용 실행에서는 실제 AAT 실험 sigma 값을 사용하세요.

## 3. 전력 수요 예측

현재 forecasting 기본 설정은 아래와 같습니다.

```text
N sweep       = 1,2,3,...,80
sigma mode    = scaled only
center mode   = replicated
readout       = ordinary least squares, np.linalg.lstsq
ridge alpha   = 0.0
window L      = 10
C_total       = 200
train split   = 2016-2019
test split    = available 2020 period
```

기본 실행:

```powershell
py src/main_forecasting.py --data "continuous dataset.csv" --sigma-csv sigma.csv --sigma-column sigma --output-dir outputs/forecasting
```

sigma 값을 직접 넣는 실행:

```powershell
py src/main_forecasting.py --data "continuous dataset.csv" --sigma-values 4.8,5.3,6.1,7.0,8.4,9.7,11.2,13.0,15.5,18.1 --output-dir outputs/forecasting
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

## 4. Panama Figure 생성

`figure_panama.py`는 `src/main_forecasting.py`와 같은 전처리, 학습, 예측 로직을 사용합니다. 차이는 그래프를 Figure 4 스타일로 구성한다는 점뿐입니다.

기본 실행:

```powershell
py figure_panama.py --sigma-csv sigma.csv --sigma-column sigma --output figure_panama.png
```

창 없이 PNG와 metrics CSV만 저장:

```powershell
py figure_panama.py --sigma-csv sigma.csv --sigma-column sigma --output figure_panama.png --show false
```

기본 MSE/R2 plot은 `N=1,2,3,...,80` 전체를 표시합니다. 예측 overlay와 작은 subplot은 기본적으로 `N=1,3,10,80`을 표시합니다.

작은 테스트:

```powershell
py figure_panama.py --sigma-csv sigma.csv --sigma-column sigma --n-values 1,2,3 --plot-n-values 1,3 --c-total 20 --output outputs/test_figure_panama.png --show false
```

## 5. Duffing Reconstruction

Duffing도 scaled sigma만 사용합니다.

기본 실행:

```powershell
py src/main_duffing.py --sigma-csv sigma.csv --sigma-column sigma --output-dir outputs/duffing
```

sigma 값을 직접 넣는 실행:

```powershell
py src/main_duffing.py --sigma-values 4.8,5.3,6.1,7.0,8.4,9.7,11.2,13.0,15.5,18.1 --output-dir outputs/duffing
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

작은 smoke test:

```powershell
py src/main_duffing.py --sigma-csv sigma.csv --sigma-column sigma --n-values 1,2 --c-total 20 --n-steps 3000 --transient-steps 300 --output-dir outputs/smoke_duffing
```

## 6. 결과 파일

`src/main_forecasting.py`와 `src/main_duffing.py` 실행 결과는 `--output-dir` 아래에 저장됩니다.

```text
metrics.csv              N별 성능 지표
best_predictions.csv     best N의 실제값과 예측값
config.json              실행 설정
best_model_arrays.npz    center, sigma, weight 배열
*.png                    metric plot과 prediction plot
```

`figure_panama.py`는 지정한 PNG와 같은 이름의 metrics CSV를 저장합니다.

```text
figure_panama.png
figure_panama_metrics.csv
```

## 7. 주의사항

sigma 값을 입력하지 않으면 실행이 중단됩니다.

전력 데이터는 원본 시간별 해상도를 유지합니다. 기본 test set은 CSV에 존재하는 2020년 구간, 즉 2020-01-01부터 2020-06-27까지의 사용 가능한 기간입니다.
