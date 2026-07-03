# 실행 매뉴얼

## 1. 필요한 파일

현재 폴더에는 전력 수요 데이터 `continuous dataset.csv`가 있습니다. 추가로 AAT 실험에서 얻은 Gaussian width sigma 값이 필요합니다.

sigma는 두 방식 중 하나로 입력할 수 있습니다.

CSV 예시:

```csv
sigma
0.12
0.18
0.25
0.40
```

또는 명령어에서 직접 입력:

```powershell
--sigma-values 0.12,0.18,0.25,0.40
```

논문용 실행에서는 실제 AAT 실험 sigma 값을 사용해야 합니다. 코드에는 임의 placeholder sigma가 기본값으로 들어 있지 않습니다.

입력한 sigma의 절대값은 RBF 계산에 그대로 들어가지 않습니다. 코드는 experimental sigma의 상대적 폭 비율을 보존한 뒤, 표준화된 `X_train`에서 계산한 nearest-center 거리 기준값 `d_ref`에 맞게 재스케일합니다.

```text
sigma_exp   = geomspace(sigma_min_exp, sigma_max_exp, N)
sigma_rel   = sigma_exp / median(sigma_exp)
sigma_model = sigma_rel * d_ref
```

따라서 `sigma.csv`는 AAT 실험에서 얻은 width diversity/range를 제공하는 역할을 합니다.

## 2. 전력 수요 예측 실행

기본 실행:

```powershell
py src/main_forecasting.py --data "continuous dataset.csv" --sigma-csv sigma.csv --sigma-column sigma --output-dir outputs/forecasting
```

sigma 값을 직접 넣는 실행:

```powershell
py src/main_forecasting.py --data "continuous dataset.csv" --sigma-values 0.12,0.18,0.25,0.40 --output-dir outputs/forecasting
```

자주 바꿀 파라미터:

- `--n-values`: 비교할 distinct sigma 개수. 기본값은 `1,2,3,...,80`입니다.
- `--window-length 10`: 입력 window 길이.
- `--c-total 200`: 전체 RBF kernel 개수.
- `--seed 0`: k-means와 sigma assignment seed.
- `--shuffle-sigmas true`: center별 sigma 배정을 섞을지 여부.
- `--train-start-year 2016`
- `--train-end-year 2019`
- `--test-year 2020`

작은 smoke test 예시:

```powershell
py src/main_forecasting.py --data "continuous dataset.csv" --sigma-values 0.1,0.2,0.4 --n-values 1,2 --c-total 20 --output-dir outputs/smoke_forecasting
```

## 3. Duffing reconstruction 실행

기본 실행:

```powershell
py src/main_duffing.py --sigma-csv sigma.csv --sigma-column sigma --output-dir outputs/duffing
```

sigma 값을 직접 넣는 실행:

```powershell
py src/main_duffing.py --sigma-values 0.12,0.18,0.25,0.40 --output-dir outputs/duffing
```

자주 바꿀 파라미터:

- `--n-values`: 기본값은 `1,2,3,...,80`입니다.
- `--window-length 10`
- `--c-total 300`
- `--train-fraction 0.5`
- `--transient-steps 2000`
- `--alpha 1.0`
- `--beta 0.2`
- `--gamma 0.3`
- `--omega 1.2`
- `--dt 0.01`
- `--n-steps 20000`
- `--x0 0.1`
- `--v0 0.0`

작은 smoke test 예시:

```powershell
py src/main_duffing.py --sigma-values 0.1,0.2,0.4 --n-values 1,2 --c-total 20 --n-steps 3000 --transient-steps 300 --output-dir outputs/smoke_duffing
```

## 4. 결과 확인

실행 후 `--output-dir`에 다음 파일이 생성됩니다.

- `metrics.csv`: `N`별 성능 지표.
- `best_predictions.csv`: best `N`의 실제값과 예측값.
- `config.json`: 실행 설정과 sigma range.
- `best_model_arrays.npz`: center, sigma assignment, weight 배열.
- PNG 파일: metric 변화와 예측 overlay plot.

전력 예측에서는 `mse`, `mae`, `r2`를 확인합니다. Duffing에서는 `mse_x`, `mse_v`, `r2_x`, `r2_v`를 확인합니다.

## 5. 주의사항

sigma를 입력하지 않으면 실행이 중단됩니다. 이는 AAT 실험값과 RBF width 사이의 연결을 유지하기 위한 의도적인 설계입니다.

전력 데이터는 원본 시간별 해상도를 유지합니다. 기본 test set은 CSV에 존재하는 2020년 구간, 즉 2020-01-01부터 2020-06-27까지의 사용 가능한 기간입니다.
