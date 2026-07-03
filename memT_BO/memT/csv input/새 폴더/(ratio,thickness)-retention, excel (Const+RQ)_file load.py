import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, ConstantKernel
from scipy.stats import norm
from matplotlib import cm

# 1. 실험 가능한 O2/Ar 비율 후보 생성 (0/50부터 20/30까지 21가지)
o2_values = np.arange(0, 21)  # 0, 1, 2, ..., 20
ar_values = 50 - o2_values    # 50, 49, 48, ..., 30
o2_ar_ratio = o2_values / ar_values  # 비율 계산

# 비율을 분수 형태로 표시하기 위한 레이블 생성
ratio_labels = [f"{o2}/{ar}" for o2, ar in zip(o2_values, ar_values)]

# 2. 두께 후보 생성 (5nm-80nm, 5nm 간격)
thickness_nm = np.arange(5, 85, 5)  # 5, 10, 15, ..., 80 nm
thickness_um = thickness_nm / 1000.0  # 0.005, 0.01, 0.015, ..., 0.08 μm

# 3. CSV 파일에서 초기 실험 데이터 읽어오기
def load_experiment_data_from_csv(filename):
    """
    CSV 파일에서 실험 데이터를 읽어와서 학습용 데이터로 변환

    Parameters:
    -----------
    filename : str
        CSV 파일 경로 (예: 'experiment_data.csv')
        Expected CSV columns:
        - O2_ratio: O2 비율 (0-20 범위)
        - thickness: 두께 (nm 단위)
        - on_off_ratio: on/off 비율 (log 값)

    Returns:
    --------
    X_train : numpy array
        입력 데이터 (O2/Ar 비율, 두께)
    y_train : numpy array
        출력 데이터 (on/off ratio)
    """
    print(f"{filename} 파일에서 실험 데이터를 읽어옵니다.")

    try:
        # CSV 파일 읽기
        df = pd.read_csv(filename)
        print("CSV 데이터:")
        print(df.head())

        # 필수 컬럼 확인
        required_columns = ['O2_ratio', 'thickness', 'on_off_ratio']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")

        init_x = []
        init_y = []

        for _, row in df.iterrows():
            # O2 비율 읽기
            o2_ratio = row['O2_ratio']

            # 두께 읽기 (nm 단위를 μm으로 변환)
            thickness_nm_val = row['thickness']
            thickness_um_val = thickness_nm_val / 1000.0

            # on/off ratio 읽기
            on_off_ratio = row['on_off_ratio']

            # O2/Ar 비율 계산 (O2 비율이 0-20 범위라고 가정)
            if o2_ratio < 0 or o2_ratio > 20:
                print(f"경고: O2 비율 {o2_ratio}이 유효 범위(0-20)를 벗어남")
                continue

            ar_ratio = 50 - o2_ratio
            if ar_ratio <= 0:
                print(f"경고: Ar 비율이 0 이하입니다. O2: {o2_ratio}, Ar: {ar_ratio}")
                continue

            o2_ar_ratio_val = o2_ratio / ar_ratio

            init_x.append([o2_ar_ratio_val, thickness_um_val])
            init_y.append(on_off_ratio)

            print(f"데이터: O2={o2_ratio}, Ar={ar_ratio}, O2/Ar={o2_ar_ratio_val:.3f}, "
                  f"두께={thickness_nm_val}nm ({thickness_um_val:.4f}μm), on/off ratio={on_off_ratio}")

        if len(init_x) == 0:
            raise ValueError("유효한 데이터가 없습니다.")

        X_train = np.array(init_x)
        y_train = np.array(init_y)

        print(f"총 {len(X_train)}개의 실험 데이터를 읽어왔습니다.")
        return X_train, y_train

    except FileNotFoundError:
        print(f"{filename} 파일을 찾을 수 없습니다.")
        return None, None
    except Exception as e:
        print(f"데이터 읽기 중 오류 발생: {e}")
        return None, None

# 수동 입력 대체 함수
def manual_input_data():
    """
    CSV 파일을 찾을 수 없을 때 수동으로 데이터 입력
    """
    print("수동으로 초기 데이터를 입력합니다.")

    init_x = []
    init_y = []
    n_init = 6  # 초기 실험 데이터 개수

    print("\nO2/Ar 비율 후보:")
    for i, label in enumerate(ratio_labels):
        print(f"{i}: {label} (실제 비율: {o2_ar_ratio[i]:.3f})")

    print("\n두께 후보 (nm):")
    for thick in thickness_nm:
        print(f"{thick} nm ({thick/1000:.4f} μm)")

    for i in range(n_init):
        print(f"\n[초기 데이터 {i+1}/{n_init}]")
        ratio_idx = int(input("O2/Ar 비율 (위 목록에서 번호 선택): "))
        ratio = o2_ar_ratio[ratio_idx]

        # 두께 직접 선택 (5nm 단위)
        thick_nm = int(input("두께 선택 (5, 10, 15, ..., 80 중 선택): "))
        thick_um = thick_nm / 1000.0
        print(f"선택된 두께: {thick_nm} nm ({thick_um:.4f} μm)")

        y = float(input("측정된 on/off ratio (상용로그 값): "))

        init_x.append([ratio, thick_um])
        init_y.append(y)

    return np.array(init_x), np.array(init_y)

# 4. RQ 커널 기반 GPR 모델 생성 함수 (신뢰구간 최적화 적용)
def create_optimized_rq_gpr():
    """
    산화물 반도체 멤트랜지스터 최적화에 특화된 RQ 커널 GPR 모델 생성
    신뢰구간 축소를 위한 최적화된 하이퍼파라미터 적용
    """
    # WO3/IGZO 멤트랜지스터 최적화용 RQ 커널
    kernel = ConstantKernel(
        constant_value=3.0,
        constant_value_bounds=(0.5, 30.0)
    ) * RationalQuadratic(
        length_scale=[0.025, 0.0025],    # O2/Ar 비율: 0.025, 두께: 0.0025μm
        alpha=0.15,                      # 제한적 스케일 혼합으로 신뢰구간 축소
        length_scale_bounds=[(1e-3, 0.2), (1e-4, 0.02)],
        alpha_bounds=(0.02, 0.8)
    ) + WhiteKernel(
        noise_level=1e-6,                # 측정 노이즈 최소화
        noise_level_bounds=(1e-9, 1e-5)
    )

    # GPR 모델 생성 (정규화 및 최적화 강화)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,                      # 수치적 안정성 강화
        normalize_y=True,                # 출력 정규화 활성화
        n_restarts_optimizer=25,         # 최적화 시도 증가
        random_state=42
    )

    return gpr

# 5. Expected Improvement 계산 함수
def expected_improvement(mu, sigma, y_max, xi=0.5):
    """
    Expected Improvement 계산 (RQ 커널에 최적화된 xi 값 사용)
    """
    imp = mu - y_max - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    return ei

# 6. CSV 데이터 저장 함수
def save_prediction_to_csv(iteration, ratio_mesh, thick_mesh, mu, sigma, X_train, y_train, next_point=None):
    """
    예측값, 신뢰구간, 실험 데이터, 다음 추천 포인트를 CSV로 저장
    """
    # 실험 데이터가 있는 위치에 값을 넣고, 없으면 NaN으로 채움
    exp_data_array = np.full(mu.shape, np.nan)

    # 학습 데이터 좌표를 메시 그리드 인덱스로 변환
    for x, y, val in zip(X_train[:,0], X_train[:,1], y_train):
        idx_ratio = (np.abs(ratio_mesh[0,:] - x)).argmin()
        idx_thick = (np.abs(thick_mesh[:,0] - y)).argmin()
        exp_data_array[idx_thick, idx_ratio] = val

    # 다음 추천 포인트 위치에 값을 표시
    next_point_array = np.full(mu.shape, np.nan)
    if next_point is not None:
        next_ratio, next_thick = next_point
        idx_ratio = (np.abs(ratio_mesh[0,:] - next_ratio)).argmin()
        idx_thick = (np.abs(thick_mesh[:,0] - next_thick)).argmin()
        next_point_array[idx_thick, idx_ratio] = mu[idx_thick, idx_ratio]

    # 예측 표면 데이터 CSV
    df_prediction = pd.DataFrame({
        'O2_Ar_Ratio': ratio_mesh.ravel(),
        'Thickness_nm': (thick_mesh.ravel() * 1000),
        'OnOff_Ratio_Prediction': mu.ravel(),
        'OnOff_Ratio_CI_Lower': (mu - 1.96 * sigma).ravel(),
        'OnOff_Ratio_CI_Upper': (mu + 1.96 * sigma).ravel(),
        'Experimental_OnOff_Ratio': exp_data_array.ravel(),
        'Next_Point_OnOff_Ratio': next_point_array.ravel(),
        'Standard_Deviation': sigma.ravel(),
        'Thickness_um': thick_mesh.ravel(),
        'Confidence_Interval_Width': (2 * 1.96 * sigma).ravel()  # 신뢰구간 폭 추가
    })

    # 실험 데이터 CSV
    df_experiments = pd.DataFrame({
        'O2_Ar_Ratio': X_train[:, 0],
        'Thickness_um': X_train[:, 1],
        'Thickness_nm': X_train[:, 1] * 1000,
        'OnOff_Ratio_Measured': y_train
    })

    # CSV 파일로 저장
    prediction_filename = f'onoff_ratio_prediction_RQ_iter_{iteration}.csv'
    experiment_filename = f'experiment_data_RQ_iter_{iteration}.csv'

    df_prediction.to_csv(prediction_filename, index=False)
    df_experiments.to_csv(experiment_filename, index=False)

    print(f"\nCSV 파일 저장 완료:")
    print(f"- 예측 데이터: {prediction_filename}")
    print(f"- 실험 데이터: {experiment_filename}")

    # 신뢰구간 통계 출력
    avg_ci_width = np.mean(2 * 1.96 * sigma)
    print(f"- 평균 95% 신뢰구간 폭: {avg_ci_width:.4f}")

    # 다음 추천 포인트 (있는 경우) 별도 CSV로 저장
    if next_point is not None:
        next_ratio, next_thick = next_point
        next_mu = mu[(np.abs(thick_mesh[:,0] - next_thick)).argmin(),
                     (np.abs(ratio_mesh[0,:] - next_ratio)).argmin()]

        df_next = pd.DataFrame({
            'O2_Ar_Ratio': [next_ratio],
            'Thickness_um': [next_thick],
            'Thickness_nm': [next_thick * 1000],
            'Predicted_OnOff_Ratio': [next_mu]
        })

        next_filename = f'next_point_RQ_iter_{iteration}.csv'
        df_next.to_csv(next_filename, index=False)
        print(f"- 다음 추천 포인트: {next_filename}")

    return prediction_filename

# 7. 메인 실행부 - CSV 파일에서 데이터 로딩 시도
csv_filename = 'experiment_data.csv'  # CSV 파일명
X_train, y_train = load_experiment_data_from_csv(csv_filename)

# CSV 파일 로딩 실패 시 수동 입력
if X_train is None or y_train is None:
    X_train, y_train = manual_input_data()

print(f"\n=== RQ 커널 기반 베이지안 최적화 시작 ===")
print(f"초기 실험 데이터: {len(X_train)}개")
print("목표: 산화물 반도체 멤트랜지스터 on/off ratio 최적화")

# 8. 베이지안 최적화 루프 (RQ 커널 적용)
iteration = 0

while True:
    print(f"\n--- 반복 {iteration + 1} ---")

    # RQ 커널 기반 GPR 모델 생성 및 학습
    gpr = create_optimized_rq_gpr()
    gpr.fit(X_train, y_train)

    # 학습된 커널 파라미터 출력
    print(f"학습된 커널 파라미터:")
    print(f"- Constant: {gpr.kernel_.k1.k1.constant_value:.4f}")
    print(f"- RQ length_scale: {gpr.kernel_.k1.k2.length_scale}")
    print(f"- RQ alpha: {gpr.kernel_.k1.k2.alpha:.4f}")
    print(f"- White noise: {gpr.kernel_.k2.noise_level:.2e}")

    # 모든 가능한 조합에 대한 예측
    ratio_mesh, thick_mesh = np.meshgrid(o2_ar_ratio, thickness_um)
    X_candidates = np.column_stack([ratio_mesh.ravel(), thick_mesh.ravel()])

    # 예측 및 EI 계산
    mu, sigma = gpr.predict(X_candidates, return_std=True)
    y_max = np.max(y_train)

    # RQ 커널에 최적화된 EI 계산 (xi=0.5로 조정)
    ei = expected_improvement(mu, sigma, y_max, xi=0.5)

    # 다음 추천 조합 (EI 최대)
    ei_2d = ei.reshape(len(thickness_um), len(o2_ar_ratio))
    next_2d_idx = np.unravel_index(np.argmax(ei_2d), ei_2d.shape)
    next_thick_idx, next_ratio_idx = next_2d_idx

    next_ratio = o2_ar_ratio[next_ratio_idx]
    next_thick = thickness_um[next_thick_idx]
    next_ratio_label = ratio_labels[next_ratio_idx]
    next_thick_nm = thickness_nm[next_thick_idx]

    # 예측 정확도 및 신뢰구간 분석
    mu_2d = mu.reshape(thick_mesh.shape)
    sigma_2d = sigma.reshape(thick_mesh.shape)
    avg_ci_width = np.mean(2 * 1.96 * sigma_2d)
    max_prediction = np.max(mu_2d)
    max_pred_location = np.unravel_index(np.argmax(mu_2d), mu_2d.shape)

    print(f"\n=== 예측 결과 분석 ===")
    print(f"평균 95% 신뢰구간 폭: {avg_ci_width:.4f}")
    print(f"최대 예측 on/off ratio: {max_prediction:.4f}")
    print(f"현재 최고 실험값: {y_max:.4f}")

    print(f"\n[추천] 다음 실험 조건:")
    print(f"O2/Ar 비율: {next_ratio_label} (실제 비율: {next_ratio:.3f})")
    print(f"두께: {next_thick_nm} nm ({next_thick:.4f} μm)")
    print(f"예상 EI 값: {np.max(ei):.6f}")

    # 예측 데이터를 CSV로 저장
    csv_file = save_prediction_to_csv(
        iteration,
        ratio_mesh,
        thick_mesh,
        mu_2d,
        sigma_2d,
        X_train,
        y_train,
        (next_ratio, next_thick)
    )

    # 시각화 (RQ 커널 특화)
    fig = plt.figure(figsize=(20, 6))

    # 1. 3D 표면 플롯 (RQ 커널 예측)
    ax1 = fig.add_subplot(131, projection='3d')

    surf1 = ax1.plot_surface(ratio_mesh, thick_mesh, mu_2d,
                            cmap=cm.viridis, alpha=0.8)
    ax1.scatter(X_train[:, 0], X_train[:, 1], y_train,
               color='red', s=60, marker='o', label='Real data')
    ax1.scatter([next_ratio], [next_thick], [mu_2d[next_thick_idx, next_ratio_idx]],
               color='orange', s=120, marker='*', label='Next point')

    ax1.set_xlabel('O2/Ar Ratio')
    ax1.set_ylabel('Thickness (μm)')
    ax1.set_zlabel('on/off ratio (log)')
    ax1.set_title(f'RQ Kernel Prediction (α={gpr.kernel_.k1.k2.alpha:.3f})')
    ax1.legend()

    # 신뢰구간 시각화 (더 투명하게)
    surf_upper = ax1.plot_surface(ratio_mesh, thick_mesh,
                                 (mu_2d + 1.96*sigma_2d),
                                 color='lightblue', alpha=0.15, linewidth=0)
    surf_lower = ax1.plot_surface(ratio_mesh, thick_mesh,
                                 (mu_2d - 1.96*sigma_2d),
                                 color='lightblue', alpha=0.15, linewidth=0)

    # 2. EI 시각화
    ax2 = fig.add_subplot(132)
    ei_contour = ax2.contourf(ratio_mesh, thick_mesh, ei_2d, levels=25, cmap='hot')
    ax2.scatter(X_train[:, 0], X_train[:, 1], color='white', s=40,
               marker='o', edgecolors='black', linewidth=1.5)
    ax2.scatter([next_ratio], [next_thick], color='orange', s=120, marker='*')
    ax2.set_xlabel('O2/Ar Ratio')
    ax2.set_ylabel('Thickness (μm)')
    ax2.set_title('Expected Improvement (RQ Kernel)')
    plt.colorbar(ei_contour, ax=ax2, label='EI Value')

    # 3. 신뢰구간 폭 시각화 (RQ 커널 특화)
    ax3 = fig.add_subplot(133)
    ci_width_2d = 2 * 1.96 * sigma_2d
    ci_contour = ax3.contourf(ratio_mesh, thick_mesh, ci_width_2d,
                             levels=20, cmap='Blues_r')
    ax3.scatter(X_train[:, 0], X_train[:, 1], color='red', s=40,
               marker='o', edgecolors='white', linewidth=1.5)
    ax3.scatter([next_ratio], [next_thick], color='orange', s=120, marker='*')
    ax3.set_xlabel('O2/Ar Ratio')
    ax3.set_ylabel('Thickness (μm)')
    ax3.set_title('95% Confidence Interval Width')
    plt.colorbar(ci_contour, ax=ax3, label='CI Width')

    plt.tight_layout()
    plt.show()

    # 계속 진행 여부 확인
    cont = input("\n계속 실험을 진행하려면 Enter, 종료하려면 q 입력: ")
    if cont.lower() == 'q':
        print("실험 추천을 종료합니다.")
        print(f"총 {len(X_train)}개의 실험 데이터로 학습 완료")
        break

    # 다음 실험 데이터 추가 입력
    print("\nO2/Ar 비율 후보:")
    for i, label in enumerate(ratio_labels):
        print(f"{i}: {label} (실제 비율: {o2_ar_ratio[i]:.3f})")

    print("\n두께 후보 (nm):")
    for thick in thickness_nm:
        print(f"{thick} nm ({thick/1000:.4f} μm)")

    print(f"\n[추천 조건] O2/Ar: {next_ratio_label}, 두께: {next_thick_nm} nm")

    ratio_idx = int(input("O2/Ar 비율 (위 목록에서 번호 선택): "))
    ratio = o2_ar_ratio[ratio_idx]

    thick_nm = int(input("두께 선택 (5, 10, 15, ..., 80 중 선택): "))
    thick_um = thick_nm / 1000.0
    print(f"선택된 두께: {thick_nm} nm ({thick_um:.4f} μm)")

    y_new = float(input("측정된 on/off ratio (상용로그 값): "))

    # 새로운 데이터 추가
    X_train = np.vstack([X_train, [[ratio, thick_um]]])
    y_train = np.append(y_train, y_new)

    iteration += 1

print("\n=== 최종 결과 요약 ===")
print(f"총 실험 횟수: {len(X_train)}회")
print(f"최고 on/off ratio: {np.max(y_train):.4f}")
best_idx = np.argmax(y_train)
print(f"최적 조건: O2/Ar={X_train[best_idx, 0]:.3f}, 두께={X_train[best_idx, 1]*1000:.0f}nm")
