# Figure 4 재현 이슈 및 가정

이 파일은 논문과 로컬 SI pseudocode만으로 정확히 결정할 수 없는 부분, 또는 논문 본문/그림/SI 사이에 서로 맞지 않는 부분을 기록합니다.

## 원본 자료 접근

- ACS DOI 페이지에는 공식 Supporting Information 파일 `am5c19999_si_001.pdf`와 `am5c19999_si_002.zip`가 표시됩니다.
- 하지만 현재 환경에서 해당 파일을 직접 다운로드하려고 하면 HTTP 403 오류가 발생했습니다.
- 따라서 구현은 프로젝트 폴더 안의 `SI_code/Code 1.py`와 `SI_code/Code 2.py`를 실행 명세로 사용합니다.
- 참고 DOI: https://pubs.acs.org/doi/10.1021/acsami.5c19999

## Gaussian 폭 sigma 값

- 로컬 프로젝트에는 논문 소자에서 추출한 실제 Gaussian width 숫자 데이터가 없습니다.
- Duffing 기본 RBF 폭 범위는 `sigma_min=3`, `sigma_max=20`으로 유지합니다.
- Panama 기본 RBF 폭 범위는 `sigma_min=0.3`, `sigma_max=20`으로 설정합니다.
- 이유는 Panama 입력이 `StandardScaler`로 표준화된 10차원 window이기 때문에, Figure 3에서 읽히는 전압-domain sigma 값을 그대로 쓰면 `N=1` 예측선이 논문 Figure 4i보다 과도하게 잘 맞기 때문입니다.
- `N=1` 민감도 실험에서 `sigma=0.3`일 때 첫 15일 예측선 표준편차가 약 `3.46 MW`로 거의 직선이 되었고, 기존 `sigma=3`일 때는 약 `96.4 MW`로 하루 주기를 상당히 따라갔습니다.
- 따라서 논문 Figure 4i의 거의 일직선인 `N=1` 노란색선은 작은 effective sigma 또는 입력 정규화/스케일링 차이에서 비롯된 것으로 보는 것이 가장 그럴듯합니다.
- 향후 사용자 실험 소자의 Gaussian fitting sigma 데이터가 준비되면 이 범위를 교체하거나, 실측 sigma 리스트를 직접 넣는 방식으로 확장해야 합니다.
- 현재 구현은 `N=1`일 때 log-spaced sigma 범위의 첫 값인 `sigma_min`을 사용합니다.
- `output/panama_sigma_sensitivity/n1_sigma_sensitivity.csv`와 `output/panama_sigma_sensitivity/figures/n1_sigma_comparison_with_flat_case.png`에 sigma 민감도 검증 결과를 저장했습니다.
- `extract_sigmas.py`로 추출되는 sigma는 transfer curve의 `VG` 전압 단위(V)입니다.
- 반면 `run_panama.py`에서 RBF에 들어가는 sigma는 표준화된 입력 window 공간의 폭입니다.
- 논문은 전압-domain sigma를 RBF 입력공간 sigma로 어떻게 스케일 변환했는지 명시하지 않습니다.
- 따라서 `run_panama.py --sigma-file ... --sigma-scale ...` 옵션으로 실측 sigma에 배율을 곱해 재현성을 조정할 수 있게 했습니다.

## Duffing oscillator

- 논문/SI는 Duffing 방정식, RK4 또는 RK45 계열 적분, `dt=1 ms`, `L=10`, `C_total=300`은 제공합니다.
- 하지만 초기조건, 전체 시뮬레이션 길이, transient 제거 길이는 명시하지 않습니다.
- 현재 기본값은 `x0=0.1`, `v0=0.0`, `transient_steps=100000`, `post_steps=200000`입니다.
- 로컬 SI pseudocode에 맞춰 전체 생성 데이터에 대해 표준화를 수행한 뒤 chronological 50/50 train/test split을 적용합니다.
- 논문 본문은 Duffing 시각화를 `N=3`과 `N=10`으로 설명하지만, 실제 Figure 4d/f의 라벨은 `N=80`으로 보입니다.
- 따라서 현재 그림 생성 기본값은 실제 Figure 4 라벨을 따라 `N=3`과 `N=80`입니다.

## Panama 전력 예측

- 논문 Methods는 2016-2019 데이터를 train, 2020 데이터를 hold-out test로 사용했다고 설명합니다.
- 반면 로컬 `SI_code/Code 1.py`는 chronological 80/20 split을 사용한다고 설명합니다.
- 현재 구현은 SI_code의 80/20 split을 따릅니다.
- 데이터 원본은 `Panama/train_dataframes.xlsx`의 실제 수요값 `DEMAND` 열을 기본으로 사용합니다.
- Excel 파일에는 여러 sheet가 있으므로, 기본 loader는 `datetime`과 `DEMAND` 열을 가진 sheet들을 스캔한 뒤 가장 긴 sheet를 선택합니다.
- 현재 파일에서는 `Week 24, Jun 2020` sheet가 기본 선택됩니다.
- 시간별 샘플은 daily 평균으로 집계하지 않고 그대로 사용합니다.
- 그림의 x축은 sample index를 24로 나누어 day 단위로 표시합니다.
- 이 방식이 Figure 4h/i에 보이는 일주기 패턴과 더 잘 맞습니다.

## MLP baseline

- 논문은 비교용 MLP baseline을 언급하지만 learning rate, momentum, max iteration 등 정확한 hyperparameter를 제공하지 않습니다.
- 따라서 MLP baseline은 `run_panama.py --include-mlp` 옵션으로만 실행되며, script 안에 문서화된 기본값을 사용합니다.
- MLP baseline은 현재 재현의 핵심 목표가 아니라 보조 비교 대상으로 취급합니다.
