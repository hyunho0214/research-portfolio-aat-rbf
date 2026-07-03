# 전체 Workflow 설명

이 문서는 현재 프로젝트 코드가 어떤 순서로 동작하는지, transfer curve에서 sigma를 어떻게 추출하는지, 그 sigma가 Panama 전력 예측 RBF 시뮬레이션에서 어떻게 사용되는지 설명합니다.

## 1. 전체 흐름 요약

```text
소자 transfer curve Excel
-> 각 ID-VG curve를 Gaussian fitting
-> curve별 sigma 추출
-> sigma_scale로 RBF 입력공간 sigma로 변환
-> Panama DEMAND time-series 로드
-> sliding window 데이터셋 생성
-> train/test split 및 표준화
-> k-means로 RBF center 생성
-> multi-sigma Gaussian RBF feature 생성
-> least-squares로 output weight 학습
-> test 구간 예측
-> Figure 4i/j/k 스타일 결과 생성
-> interactive HTML 그래프 생성
```

## 2. Transfer Curve 입력

입력 파일은 Excel만 사용합니다.

```text
templates\transfer_curve_template.xlsx
```

기본 구조:

```text
VG, ID_#1, ID_#2, ID_#3, ...
30.00, ...
29.75, ...
29.50, ...
...
0.00, ...
```

- `VG`는 gate voltage입니다.
- `ID_#1`, `ID_#2`, ... 각 열은 서로 다른 transfer curve입니다.
- 한 열에서 sigma 하나가 추출됩니다.
- 각 열이 서로 다른 `VD` 조건이라면, column별로 서로 다른 sigma가 나옵니다.
- `ID_#21`처럼 오른쪽에 열을 추가해도 됩니다.
- `VD_6V_ID`처럼 열 이름에 VD 값을 넣으면 결과의 `vd` 열에 `6.0`이 기록됩니다.

## 3. Gaussian Fitting으로 Sigma 추출

실행:

```powershell
py -3 extract_sigmas.py templates\transfer_curve_template.xlsx --output-dir output\my_sigmas
```

관련 코드:

```text
extract_sigmas.py
src\transfer_sigma.py
```

각 ID curve는 다음 함수로 fitting됩니다.

```text
ID(VG) = baseline + A * exp(-(VG - mu)^2 / (2 * sigma^2))
```

추출되는 parameter:

```text
A        Gaussian amplitude
mu       peak가 나타나는 VG 위치
sigma    Gaussian width
baseline background current
R2       fitting 품질
```

출력 파일:

```text
output\my_sigmas\sigma_values.xlsx
output\my_sigmas\gaussian_fit_results.xlsx
output\my_sigmas\figures\gaussian_fit_overview.png
output\my_sigmas\figures\sigma_summary.png
```

`sigma_values.xlsx`가 Panama RBF 시뮬레이션에 들어가는 sigma 입력 파일입니다.

## 4. Sigma의 의미

소자 transfer curve에서 추출한 sigma는 `VG` 축 기준입니다.

```text
sigma_device 단위 = V
```

하지만 Panama RBF 모델에서 쓰는 입력은 전력 demand window를 `StandardScaler`로 표준화한 값입니다.

```text
sigma_RBF 단위 = standardized input unit
```

따라서 코드에서는 다음 변환을 사용합니다.

```text
sigma_RBF = sigma_device * sigma_scale
```

즉 `sigma_scale`은 소자의 전압-domain sigma를 RBF 입력공간의 sigma로 바꾸는 calibration factor입니다.

논문은 이 변환식을 명시하지 않았습니다. 따라서 `sigma_scale`은 소자 transfer curve와 Panama feature space를 연결하는 실험적 encoding scale로 취급해야 합니다.

## 5. Sigma Selection

Panama RBF 실행에서는 `sigma_values.xlsx`의 `sigma` 열을 읽습니다.

기본 선택 방식:

```text
sigma_selection = logspace
```

즉, 추출된 sigma들의 scaled min/max를 구한 뒤, N개 sigma를 log-spaced로 뽑습니다.

예:

```text
raw sigma min/max = 3.0 / 4.0 V
sigma_scale = 0.1
scaled sigma min/max = 0.3 / 0.4
N = 3
sigma_set = logspace(0.3, 0.4, 3)
```

분포 자체의 분위수를 쓰고 싶으면 다음 옵션을 사용할 수 있습니다.

```powershell
--sigma-selection quantile
```

## 6. Panama 데이터 준비

Panama 원본 데이터는 Excel 파일을 사용합니다.

```text
Panama\train_dataframes.xlsx
```

관련 코드:

```text
run_panama.py
src\panama.py
```

동작:

1. workbook 안의 sheet를 스캔합니다.
2. `datetime`과 `DEMAND` 열을 가진 sheet만 후보로 봅니다.
3. 가장 긴 sheet를 자동 선택합니다.
4. 현재 파일에서는 보통 `Week 24, Jun 2020` sheet가 선택됩니다.
5. `DEMAND` 열을 시간 순서대로 1D time-series로 사용합니다.

## 7. Sliding Window 생성

논문 SI pseudocode와 같이 window length는 `L=10`입니다.

전력 수요 time-series가 다음과 같다면:

```text
s[0], s[1], s[2], ..., s[t]
```

입력과 정답은 이렇게 구성됩니다.

```text
X = [s[t], s[t+1], ..., s[t+9]]
y = s[t+10]
```

즉 과거 10개 시간 step으로 다음 1개 demand 값을 예측합니다.

## 8. Train/Test Split과 표준화

현재 구현은 local `SI_code/Code 1.py`를 따라 chronological 80/20 split을 사용합니다.

```text
앞 80% = train
뒤 20% = test
```

표준화는 train set 기준으로만 fitting합니다.

```text
X_train 기준으로 X scaler fit
y_train 기준으로 y scaler fit
X_train, X_test transform
y_train transform
```

test set 정보가 scaler fitting에 들어가지 않도록 합니다.

## 9. RBF Network 구성

관련 코드:

```text
src\rbf.py
```

RBF kernel:

```text
phi_j(x) = exp(-||x - center_j||^2 / (2 * sigma_j^2))
```

Panama 기본 설정:

```text
L = 10
C_total = 200
N = 사용자가 지정, 예: 1, 3, 10, 80 또는 1:80
```

동작 순서:

1. `N`개의 sigma 값을 준비합니다.
2. `X_train`에 대해 k-means clustering을 수행합니다.
3. k-means center도 `N`개 생성됩니다.
4. 전체 kernel 수가 `C_total=200`이 되도록 center와 sigma를 반복 복제합니다.
5. RBF feature matrix `Phi_train`을 만듭니다.
6. bias column을 추가합니다.
7. output weight는 least-squares로 풉니다.

즉 neural network를 gradient descent로 오래 학습하는 구조가 아니라:

```text
RBF feature 계산 -> 선형 least-squares로 output weight 계산
```

입니다.

## 10. Prediction과 Metric 계산

RBF에서 나온 예측값은 표준화된 y 공간에 있습니다.

따라서 마지막에 원래 MW 단위로 되돌립니다.

```text
standardized prediction
-> inverse_transform
-> MW prediction
```

그 다음 test true value와 비교해 다음 metric을 계산합니다.

```text
MSE
MAE
R2
```

## 11. Panama 실행 예시

대표 N만 빠르게 실행:

```powershell
py -3 run_panama.py --sigma-file output\my_sigmas\sigma_values.xlsx --sigma-scale 0.1 --n-values 1,3,10,80 --selected-n 1,3,10,80 --output-dir output\panama_my_sigmas
```

전체 `N=1`부터 `N=80`까지 실행:

```powershell
py -3 run_panama.py --sigma-file output\my_sigmas\sigma_values.xlsx --sigma-scale 0.1 --n-values 1:80 --selected-n 1,3,10,80 --output-dir output\panama_my_sigmas_full
```

## 12. 결과 파일

Panama 결과 폴더:

```text
output\panama_my_sigmas
```

주요 파일:

```text
output\panama_my_sigmas\panama_rbf_metrics.csv
output\panama_my_sigmas\figures\panama_15_day_segments.png
output\panama_my_sigmas\figures\panama_full_test.png
output\panama_my_sigmas\figures\panama_mse.png
output\panama_my_sigmas\figures\panama_r2.png
output\panama_my_sigmas\predictions\
```

논문 Figure 4i에 해당하는 파일:

```text
output\panama_my_sigmas\figures\panama_15_day_segments.png
```

## 13. Interactive Graph

정적인 PNG 대신 zoom/pan 가능한 HTML 그래프를 만들 수 있습니다.

```powershell
py -3 make_interactive_panama.py --output-dir output\panama_my_sigmas
```

결과:

```text
output\panama_my_sigmas\interactive\panama_interactive.html
```

열기:

```powershell
explorer output\panama_my_sigmas\interactive\panama_interactive.html
```

지원 기능:

- 마우스 휠: x축 확대/축소
- 드래그: 좌우 이동
- 더블클릭: reset
- checkbox: true/predicted line 켜기/끄기

## 14. 중요한 해석 포인트

- 추출 sigma 자체는 소자의 물리적 transfer curve 폭입니다.
- `sigma_scale`은 물리 소자 sigma와 RBF 입력 feature space를 연결하는 보정 factor입니다.
- 논문은 이 변환식을 명시하지 않았으므로, `sigma_scale=1`과 Figure 4i 형태에 맞춘 보정값을 비교하는 것이 가장 안전합니다.
- `N=1`에서 예측선이 거의 직선으로 무너지는지, `N=80`에서 충분히 따라가는지가 Figure 4i 재현성의 중요한 check point입니다.
