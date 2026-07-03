# Figure 4 Reproduction Manual

이 문서는 transfer curve 데이터에서 Gaussian sigma를 추출하고, 추출한 sigma를 Panama RBF 시뮬레이션에 넣어 실행하는 방법을 정리한 매뉴얼입니다.

## 1. PowerShell 시작 위치

항상 먼저 프로젝트 폴더로 이동합니다.

```powershell
cd C:\Users\HYUNHO\Desktop\20260616Panama
```

## 2. Transfer Curve 입력 파일 작성

입력 템플릿은 아래 파일을 사용합니다.

```text
templates\transfer_curve_template.xlsx
```

기본 형식은 다음과 같습니다.

```text
VG,ID_#1,ID_#2,ID_#3,...
30.00,...
29.75,...
29.50,...
...
0.00,...
```

작성 규칙:

- `VG` 열은 30 V에서 0 V까지 0.25 V step으로 내려갑니다.
- `ID_#1`, `ID_#2`, ... 각 열에 서로 다른 transfer curve의 `ID` 값을 입력합니다.
- 한 열은 한 조건의 curve입니다. 즉 한 열에서 sigma 하나가 추출됩니다.
- 열은 오른쪽으로 더 추가해도 됩니다.
- 추가 열의 1행 header는 비워두지 마세요.
- VD 값을 자동으로 기록하고 싶으면 열 이름을 `VD_6V_ID`, `VD_7V_ID`처럼 쓰면 됩니다.
- `ID`가 음수여도 기본 실행은 `abs(ID)`로 fitting합니다.

## 3. Sigma 추출 실행

Excel 템플릿에서 sigma를 추출합니다.

```powershell
py -3 extract_sigmas.py templates\transfer_curve_template.xlsx --output-dir output\my_sigmas
```

실행하면 PowerShell에 다음 정보가 표시됩니다.

```text
VG column
ID columns fitted
ID column names
sigma_min / sigma_max / sigma_median
fit_R2_min / fit_R2_median
curve별 sigma, mu, R2 표
```

생성되는 주요 파일:

```text
output\my_sigmas\sigma_values.xlsx
output\my_sigmas\gaussian_fit_results.xlsx
output\my_sigmas\figures\gaussian_fit_overview.png
output\my_sigmas\figures\sigma_summary.png
```

`sigma_values.xlsx`가 Panama 시뮬레이션에 들어가는 sigma 입력 파일입니다.

## 4. Sigma 추출 결과 확인

가장 먼저 확인할 파일:

```text
output\my_sigmas\sigma_values.xlsx
```

중요한 열:

```text
column    원래 ID curve 열 이름
vd        열 이름에서 파싱된 VD 값, 없으면 빈칸
sigma     Gaussian fitting으로 추출된 sigma
mu        Gaussian peak 위치
amplitude Gaussian amplitude
baseline  baseline current
r2        fitting 품질
```

`r2`가 낮으면 해당 curve가 단일 Gaussian으로 잘 맞지 않는다는 뜻입니다.

## 5. Panama 시뮬레이션 실행

추출한 sigma를 이용해 대표 N 값만 빠르게 실행합니다.

```powershell
py -3 run_panama.py --sigma-file output\my_sigmas\sigma_values.xlsx --sigma-scale 0.1 --n-values 1,3,10,80 --selected-n 1,3,10,80 --output-dir output\panama_my_sigmas
```

전체 `N=1`부터 `N=80`까지 실행하려면:

```powershell
py -3 run_panama.py --sigma-file output\my_sigmas\sigma_values.xlsx --sigma-scale 0.1 --n-values 1:80 --selected-n 1,3,10,80 --output-dir output\panama_my_sigmas_full
```

실행하면 PowerShell에 다음 정보가 표시됩니다.

```text
Panama data source
sample 수
raw sigma min/max
sigma scale
scaled sigma min/max
effective sigma min/max
sigma selection 방식
N values
```

생성되는 주요 파일:

```text
output\panama_my_sigmas\panama_rbf_metrics.csv
output\panama_my_sigmas\figures\panama_15_day_segments.png
output\panama_my_sigmas\figures\panama_full_test.png
output\panama_my_sigmas\figures\panama_mse.png
output\panama_my_sigmas\figures\panama_r2.png
output\panama_my_sigmas\predictions\
```

Figure 4i에 해당하는 그림은 다음 파일입니다.

```text
output\panama_my_sigmas\figures\panama_15_day_segments.png
```

## 5-1. Interactive Panama 그래프 생성

정적인 PNG 대신 브라우저에서 확대/축소, pan, reset, legend on/off가 가능한 HTML 그래프를 만들 수 있습니다.

```powershell
py -3 make_interactive_panama.py --output-dir output\panama_my_sigmas
```

생성 파일:

```text
output\panama_my_sigmas\interactive\panama_interactive.html
```

PowerShell에서 바로 열기:

```powershell
explorer output\panama_my_sigmas\interactive\panama_interactive.html
```

사용 방법:

- 마우스 휠: x축 확대/축소
- 드래그: 좌우 이동
- 더블클릭: view reset
- checkbox: true/predicted line 켜기/끄기

## 6. Sigma Scale 의미

`extract_sigmas.py`가 뽑는 sigma는 transfer curve의 `VG` 축에서 얻은 값입니다.

```text
sigma_device 단위 = V
```

하지만 `run_panama.py`의 RBF 입력은 `StandardScaler`로 표준화된 전력 데이터 window입니다.

```text
sigma_RBF 단위 = standardized input unit
```

코드에서는 다음처럼 변환합니다.

```text
sigma_RBF = sigma_device * sigma_scale
```

즉 `sigma_scale`은 소자 transfer curve의 전압-domain sigma를 RBF 입력공간 sigma로 바꾸는 calibration factor입니다.

논문은 이 변환식을 직접 명시하지 않으므로, 아래 두 실행을 비교하는 것을 권장합니다.

보정 없는 실행:

```powershell
py -3 run_panama.py --sigma-file output\my_sigmas\sigma_values.xlsx --sigma-scale 1 --n-values 1,3,10,80 --selected-n 1,3,10,80 --output-dir output\panama_sigma_scale_1
```

Figure 4i의 `N=1` 직선형 거동에 맞춘 시작값:

```powershell
py -3 run_panama.py --sigma-file output\my_sigmas\sigma_values.xlsx --sigma-scale 0.1 --n-values 1,3,10,80 --selected-n 1,3,10,80 --output-dir output\panama_sigma_scale_0p1
```

더 체계적으로 잡으려면:

```text
sigma_scale = 목표 effective sigma_min / 추출된 sigma_min
```

현재 Panama 재현에서는 Figure 4i의 `N=1`을 거의 직선으로 만들기 위한 목표 effective sigma_min이 약 `0.3`이었습니다.

예를 들어 추출된 sigma_min이 3.0 V이면:

```text
sigma_scale = 0.3 / 3.0 = 0.1
```

## 7. 자주 나는 오류

### 파일 경로 오류

오류 예:

```text
FileNotFoundError: No such file or directory
```

해결:

- `cd C:\Users\HYUNHO\Desktop\20260616Panama`를 먼저 실행했는지 확인합니다.
- 실제 파일명이 명령에 적은 파일명과 같은지 확인합니다.

### Sigma 출력이 안 보이는 경우

먼저 아래 파일이 생성됐는지 확인합니다.

```text
output\my_sigmas\sigma_values.xlsx
```

생성되지 않았다면 PowerShell 오류 메시지를 확인합니다.

### `Unnamed: 21` 같은 열이 나오는 경우

추가 데이터 열의 1행 header가 비어 있다는 뜻입니다.

해결:

```text
ID_#21
```

또는 VD 값을 알고 있으면:

```text
VD_6V_ID
```

처럼 1행에 이름을 넣습니다.

### Fitting R2가 낮은 경우

단일 Gaussian으로 curve가 잘 설명되지 않는다는 뜻입니다.

가능한 원인:

- curve가 비대칭임
- peak가 measurement range 밖에 있음
- baseline이 크거나 noise가 큼
- 하나의 curve에 여러 peak가 있음

이 경우 `gaussian_fit_overview.png`를 확인해야 합니다.

## 8. Duffing 실행

Duffing Figure 4c-f 대표 N 실행:

```powershell
py -3 run_duffing.py --n-values 3,80 --selected-n 3,80 --output-dir output\duffing_selected
```

전체 `N=1`부터 `N=80`까지 실행:

```powershell
py -3 run_duffing.py --n-values 1:80 --selected-n 3,80 --output-dir output\duffing_full
```

주요 결과:

```text
output\duffing_selected\duffing_rbf_metrics.csv
output\duffing_selected\figures\duffing_phase_space.png
output\duffing_selected\figures\duffing_time_domain.png
```
