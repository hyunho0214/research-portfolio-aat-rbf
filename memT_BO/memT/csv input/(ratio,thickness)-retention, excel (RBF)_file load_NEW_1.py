import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.stats import norm
from matplotlib import cm

# 1. 실험 가능한 O2/(Ar+O2) 퍼센트 비율 후보 생성 (0%부터 32%까지, 4%씩 증가)
o2_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # 9가지 O2 비율
ar_values = 50 - o2_values  # 50, 48, 46, 44, 42, 40, 38, 36, 34
total_gas = o2_values + ar_values  # 총 가스량 (항상 50)

# O2/(Ar+O2) 비율을 퍼센트로 계산
o2_percent_ratio = (o2_values / total_gas) * 100  # 0%, 4%, 8%, 12%, 16%, 20%, 24%, 28%, 32%

# 비율을 퍼센트 형태로 표시하기 위한 레이블 생성
ratio_labels = [f"{o2_percent:.1f}%" for o2_percent in o2_percent_ratio]

print("O2/(Ar+O2) 퍼센트 비율 후보 (9가지):")
for i, (o2, ar, percent) in enumerate(zip(o2_values, ar_values, o2_percent_ratio)):
    print(f"{i}: O2={o2}, Ar={ar} → {percent:.1f}%")

# 2. 두께 후보 생성 (10nm-90nm, 10nm 간격, 9가지) - nm 단위 그대로 사용
thickness_nm = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130])  # 9가지 두께 (nm 단위)

print(f"\n두께 후보 (9가지): {thickness_nm} nm")
print(f"총 후보군 개수: {len(o2_values)} × {len(thickness_nm)} = {len(o2_values) * len(thickness_nm)}개")

# 3. CSV 파일에서 초기 실험 데이터 읽어오기 (O2_percent 전용)
def load_experiment_data_from_csv(filename):
    """
    CSV 파일에서 실험 데이터를 읽어와서 학습용 데이터로 변환

    Parameters:
    -----------
    filename : str
        CSV 파일 경로 (예: 'experiment_data.csv')

    Expected CSV columns:
    - O2_percent: O2 퍼센트 (0-32% 범위)
    - thickness: 두께 (nm 단위)
    - on_off_ratio: on/off 비율 (log 값)

    Returns:
    --------
    X_train : numpy array
        입력 데이터 (O2 퍼센트, 두께 nm)
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
        required_columns = ['O2_percent', 'thickness', 'on_off_ratio']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")

        print("O2_percent 컬럼을 사용합니다.")

        init_x = []
        init_y = []

        for _, row in df.iterrows():
            # O2 퍼센트 직접 읽기
            o2_percent = row['O2_percent']

            # 두께 읽기 (nm 단위 그대로 사용)
            thickness_nm_val = row['thickness']

            # on/off ratio 읽기
            on_off_ratio = row['on_off_ratio']

            # 유효성 검사
            if o2_percent < 0 or o2_percent > 32:
                print(f"경고: O2 퍼센트 {o2_percent:.1f}%가 유효 범위(0-32%)를 벗어남")
                continue

            if thickness_nm_val < 50 or thickness_nm_val > 130:
                print(f"경고: 두께 {thickness_nm_val}nm가 유효 범위(10-90nm)를 벗어남")
                continue

            init_x.append([o2_percent, thickness_nm_val])
            init_y.append(on_off_ratio)

            print(f"데이터: O2={o2_percent:.1f}%, "
                  f"두께={thickness_nm_val}nm, "
                  f"on/off ratio={on_off_ratio}")

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

# 수동 입력 함수 (nm 단위 기반으로 수정)
def manual_input_data():
    """
    CSV 파일을 찾을 수 없을 때 수동으로 데이터 입력
    """
    print("수동으로 초기 데이터를 입력합니다.")

    init_x = []
    init_y = []
    n_init = 6  # 초기 실험 데이터 개수

    print("\nO2/(Ar+O2) 퍼센트 비율 후보 (9가지):")
    for i, label in enumerate(ratio_labels):
        o2 = o2_values[i]
        ar = ar_values[i]
        print(f"{i}: {label} (O2={o2}, Ar={ar})")

    print(f"\n두께 후보 (9가지): {thickness_nm} nm")

    for i in range(n_init):
        print(f"\n[초기 데이터 {i+1}/{n_init}]")

        ratio_idx = int(input("O2 퍼센트 비율 (0-8 중 선택): "))
        if ratio_idx < 0 or ratio_idx >= len(o2_percent_ratio):
            print("유효하지 않은 인덱스입니다. 0으로 설정합니다.")
            ratio_idx = 0
        o2_percent = o2_percent_ratio[ratio_idx]

        # 두께 직접 선택 (nm 단위)
        thick_nm = int(input("두께 선택 (10, 20, 30, 40, 50, 60, 70, 80, 90 중 선택): "))
        if thick_nm not in thickness_nm:
            print("유효하지 않은 두께입니다. 10nm로 설정합니다.")
            thick_nm = 10

        print(f"선택된 조건: O2={o2_percent:.1f}%, 두께={thick_nm}nm")

        y = float(input("측정된 on/off ratio (상용로그 값): "))

        init_x.append([o2_percent, thick_nm])
        init_y.append(y)

    return np.array(init_x), np.array(init_y)

# CSV 데이터 저장 함수 (nm 단위 기반으로 수정)
def save_prediction_to_csv(iteration, percent_mesh, thick_mesh, mu, sigma, X_train, y_train, next_point=None):
    """
    예측값, 신뢰구간, 실험 데이터, 다음 추천 포인트를 CSV로 저장
    """
    # 실험 데이터가 있는 위치에 값을 넣고, 없으면 NaN으로 채움
    exp_data_array = np.full(mu.shape, np.nan)

    # 학습 데이터 좌표를 메시 그리드 인덱스로 변환
    for x, y, val in zip(X_train[:,0], X_train[:,1], y_train):
        idx_percent = (np.abs(percent_mesh[0,:] - x)).argmin()
        idx_thick = (np.abs(thick_mesh[:,0] - y)).argmin()
        exp_data_array[idx_thick, idx_percent] = val

    # 다음 추천 포인트 위치에 값을 표시
    next_point_array = np.full(mu.shape, np.nan)
    if next_point is not None:
        next_percent, next_thick = next_point
        idx_percent = (np.abs(percent_mesh[0,:] - next_percent)).argmin()
        idx_thick = (np.abs(thick_mesh[:,0] - next_thick)).argmin()
        next_point_array[idx_thick, idx_percent] = mu[idx_thick, idx_percent]

    # 예측 표면 데이터 CSV
    df_prediction = pd.DataFrame({
        'O2_Percent': percent_mesh.ravel(),
        'Thickness_nm': thick_mesh.ravel(),
        'OnOff_Ratio_Prediction': mu.ravel(),
        'OnOff_Ratio_CI_Lower': (mu - 1.96 * sigma).ravel(),
        'OnOff_Ratio_CI_Upper': (mu + 1.96 * sigma).ravel(),
        'Experimental_OnOff_Ratio': exp_data_array.ravel(),
        'Next_Point_OnOff_Ratio': next_point_array.ravel(),
        'Standard_Deviation': sigma.ravel()
    })

    # 실험 데이터 CSV
    df_experiments = pd.DataFrame({
        'O2_Percent': X_train[:, 0],
        'Thickness_nm': X_train[:, 1],
        'OnOff_Ratio_Measured': y_train
    })

    # CSV 파일로 저장
    prediction_filename = f'onoff_ratio_prediction_iter_{iteration}.csv'
    experiment_filename = f'experiment_data_iter_{iteration}.csv'

    df_prediction.to_csv(prediction_filename, index=False)
    df_experiments.to_csv(experiment_filename, index=False)

    print(f"\nCSV 파일 저장 완료:")
    print(f"- 예측 데이터: {prediction_filename}")
    print(f"- 실험 데이터: {experiment_filename}")

    # 다음 추천 포인트 (있는 경우) 별도 CSV로 저장
    if next_point is not None:
        next_percent, next_thick = next_point
        next_mu = mu[(np.abs(thick_mesh[:,0] - next_thick)).argmin(),
                     (np.abs(percent_mesh[0,:] - next_percent)).argmin()]

        df_next = pd.DataFrame({
            'O2_Percent': [next_percent],
            'Thickness_nm': [next_thick],
            'Predicted_OnOff_Ratio': [next_mu]
        })

        next_filename = f'next_point_iter_{iteration}.csv'
        df_next.to_csv(next_filename, index=False)
        print(f"- 다음 추천 포인트: {next_filename}")

    return prediction_filename

def print_hyperparameters(gpr, iteration):
    """
    GPR 모델의 최적화된 하이퍼파라미터 출력

    Parameters:
    -----------
    gpr : GaussianProcessRegressor
        학습된 GPR 모델
    iteration : int
        현재 반복 횟수
    """
    print(f"\n=== 반복 {iteration + 1}: GPR 하이퍼파라미터 ===")

    # 1. 전체 커널 구조 출력
    print(f"전체 커널 구조: {gpr.kernel_}")

    # 2. RBF 커널 파라미터
    try:
        # RBF 커널이 첫 번째 커널인 경우
        rbf_kernel = gpr.kernel_.k1 if hasattr(gpr.kernel_, 'k1') else gpr.kernel_
        if hasattr(rbf_kernel, 'length_scale'):
            length_scales = rbf_kernel.length_scale
            print(f"RBF Length Scale:")
            print(f"  O2 퍼센트: {length_scales[0]:.4f}")
            print(f"  두께 (nm): {length_scales[1]:.4f}")

        # Length scale bounds
        if hasattr(rbf_kernel, 'length_scale_bounds'):
            bounds = rbf_kernel.length_scale_bounds
            print(f"RBF Length Scale Bounds:")
            print(f"  O2 퍼센트: ({bounds[0][0]:.2f}, {bounds[0][1]:.2f})")
            print(f"  두께 (nm): ({bounds[1][0]:.2f}, {bounds[1][1]:.2f})")
    except Exception as e:
        print(f"RBF 커널 파라미터 추출 중 오류: {e}")

    # 3. White Kernel 파라미터
    try:
        # White 커널이 두 번째 커널인 경우
        white_kernel = gpr.kernel_.k2 if hasattr(gpr.kernel_, 'k2') else None
        if white_kernel and hasattr(white_kernel, 'noise_level'):
            noise_level = white_kernel.noise_level
            print(f"White Kernel Noise Level: {noise_level:.6f}")

            if hasattr(white_kernel, 'noise_level_bounds'):
                noise_bounds = white_kernel.noise_level_bounds
                if noise_bounds != 'fixed':
                    print(f"White Kernel Noise Bounds: {noise_bounds}")
                else:
                    print(f"White Kernel Noise Bounds: fixed")
    except Exception as e:
        print(f"White 커널 파라미터 추출 중 오류: {e}")

    # 4. GPR 모델 파라미터
    print(f"GPR 모델 파라미터:")
    print(f"  Alpha (노이즈 분산): {gpr.alpha}")
    print(f"  Normalize Y: {gpr.normalize_y}")
    print(f"  N Restarts Optimizer: {gpr.n_restarts_optimizer}")
    print(f"  Random State: {gpr.random_state}")

    # 5. 학습된 모델 통계
    print(f"학습된 모델 통계:")
    print(f"  학습 데이터 개수: {len(gpr.X_train_) if hasattr(gpr, 'X_train_') else 'N/A'}")
    print(f"  Log Marginal Likelihood: {gpr.log_marginal_likelihood():.6f}")

    # 6. 커널 파라미터의 gradient (있는 경우)
    try:
        if hasattr(gpr, 'kernel_') and hasattr(gpr.kernel_, 'theta'):
            theta = gpr.kernel_.theta
            print(f"최적화된 커널 파라미터 (theta): {theta}")
    except Exception as e:
        print(f"Theta 파라미터 추출 중 오류: {e}")

    print("=" * 50)


# 4. 메인 실행부 - CSV 파일에서 데이터 로딩 시도
csv_filename = 'experiment_data.csv'  # CSV 파일명
X_train, y_train = load_experiment_data_from_csv(csv_filename)

# CSV 파일 로딩 실패 시 수동 입력
if X_train is None or y_train is None:
    X_train, y_train = manual_input_data()

# 5. 베이지안 최적화 루프
iteration = 0

while True:
    # GPR 모델 정의 및 학습 (nm 단위에 맞게 커널 파라미터 조정)
    kernel = RBF(
        length_scale=[2.2311, 50.0486],  # O2 퍼센트(0-16%), 두께(10-90nm)에 맞게 조정
        length_scale_bounds=('fixed')  # nm 범위에 맞는 경계값
    ) + WhiteKernel(
        noise_level=0.001,
        noise_level_bounds=('fixed')
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=False,
        n_restarts_optimizer=40,
        random_state=0
    )

    # GPR 모델 학습
    gpr.fit(X_train, y_train)

    print_hyperparameters(gpr, iteration)

    # 모든 가능한 조합에 대한 예측 (81개 후보군 기반, nm 단위)
    percent_mesh, thick_mesh = np.meshgrid(o2_percent_ratio, thickness_nm)
    X_candidates = np.column_stack([percent_mesh.ravel(), thick_mesh.ravel()])

    print(f"후보군 확인: {len(X_candidates)}개 (9×9={len(o2_percent_ratio)}×{len(thickness_nm)})")

    # 예측 및 EI 계산
    mu, sigma = gpr.predict(X_candidates, return_std=True)
    y_max = np.max(y_train)

    def expected_improvement(mu, sigma, y_max, xi=0.01):
        imp = mu - y_max - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        return ei

    ei = expected_improvement(mu, sigma, y_max)

    # 다음 추천 조합 (EI 최대)
    ei_2d = ei.reshape(len(thickness_nm), len(o2_percent_ratio))
    next_2d_idx = np.unravel_index(np.argmax(ei_2d), ei_2d.shape)
    next_thick_idx, next_percent_idx = next_2d_idx

    next_percent = o2_percent_ratio[next_percent_idx]
    next_thick = thickness_nm[next_thick_idx]
    next_percent_label = ratio_labels[next_percent_idx]
    next_thick_nm = thickness_nm[next_thick_idx]

    print(f"\n=== 반복 {iteration + 1} ===")
    print(f"현재 학습 데이터: {len(X_train)}개")
    print(f"현재 최고 on/off ratio: {y_max:.4f}")
    print(f"\n[추천] 다음 실험 조건:")
    print(f"O2 퍼센트: {next_percent_label} (O2={o2_values[next_percent_idx]}, Ar={ar_values[next_percent_idx]})")
    print(f"두께: {next_thick_nm}nm")
    print(f"예상 EI: {np.max(ei):.6f}")

    # 예측 데이터를 CSV로 저장
    csv_file = save_prediction_to_csv(
        iteration,
        percent_mesh,
        thick_mesh,
        mu.reshape(thick_mesh.shape),
        sigma.reshape(thick_mesh.shape),
        X_train,
        y_train,
        (next_percent, next_thick)
    )

    # 시각화 (9×9 그리드, nm 단위로 업데이트)
    fig = plt.figure(figsize=(18, 6))

    # 1. 3D 표면 플롯
    ax1 = fig.add_subplot(131, projection='3d')
    mu_2d = mu.reshape(thick_mesh.shape)
    sigma_2d = sigma.reshape(thick_mesh.shape)

    surf1 = ax1.plot_surface(percent_mesh, thick_mesh, mu_2d, cmap=cm.viridis, alpha=0.7)
    ax1.scatter(X_train[:, 0], X_train[:, 1], y_train, color='red', s=50, marker='o', label='Real data')
    ax1.scatter([next_percent], [next_thick], [mu_2d[next_thick_idx, next_percent_idx]],
                color='orange', s=100, marker='*', label='Next point')

    ax1.set_xlabel('O2 Percent (%)')
    ax1.set_ylabel('Thickness (nm)')  # nm 단위로 표시
    ax1.set_zlabel('on/off ratio (log)')
    ax1.set_title('on/off ratio Prediction (81 candidates)')
    ax1.legend()

    # 신뢰구간 상한/하한 (투명한 표면)
    ax1.plot_surface(percent_mesh, thick_mesh, (mu_2d + 1.96*sigma_2d),
                     color='skyblue', alpha=0.2, linewidth=0)
    ax1.plot_surface(percent_mesh, thick_mesh, (mu_2d - 1.96*sigma_2d),
                     color='skyblue', alpha=0.2, linewidth=0)

    # 2. EI 시각화
    ax2 = fig.add_subplot(132)
    ei_contour = ax2.contourf(percent_mesh, thick_mesh, ei_2d, levels=15, cmap='hot')
    ax2.scatter(X_train[:, 0], X_train[:, 1], color='white', s=30, marker='o', edgecolors='black')
    ax2.scatter([next_percent], [next_thick], color='orange', s=100, marker='*')
    ax2.set_xlabel('O2 Percent (%)')
    ax2.set_ylabel('Thickness (nm)')  # nm 단위로 표시
    ax2.set_title('Expected Improvement (EI)')
    plt.colorbar(ei_contour, ax=ax2, label='EI Value')

    # 3. 예측값 시각화
    ax3 = fig.add_subplot(133)
    pred_contour = ax3.contourf(percent_mesh, thick_mesh, mu_2d, levels=15, cmap='plasma')
    ax3.scatter(X_train[:, 0], X_train[:, 1], color='white', s=30, marker='o', edgecolors='black')
    ax3.scatter([next_percent], [next_thick], color='orange', s=100, marker='*')
    ax3.set_xlabel('O2 Percent (%)')
    ax3.set_ylabel('Thickness (nm)')  # nm 단위로 표시
    ax3.set_title('on/off ratio Prediction (2D Contour)')
    plt.colorbar(pred_contour, ax=ax3, label='on/off ratio (log)')

    plt.tight_layout()
    plt.show()

    # 계속 진행 여부 확인
    cont = input("\n계속 실험을 진행하려면 Enter, 종료하려면 q 입력: ")
    if cont.lower() == 'q':
        print("실험 추천을 종료합니다.")
        break

    # 다음 실험 데이터 추가 입력
    print("\nO2/(Ar+O2) 퍼센트 비율 후보 (9가지):")
    for i, label in enumerate(ratio_labels):
        o2 = o2_values[i]
        ar = ar_values[i]
        print(f"{i}: {label} (O2={o2}, Ar={ar})")

    print(f"\n두께 후보 (9가지): {thickness_nm} nm")
    print(f"\n[추천 조건] O2: {next_percent_label}, 두께: {next_thick_nm}nm")

    percent_idx = int(input("O2 퍼센트 비율 (0-8 중 선택): "))
    if percent_idx < 0 or percent_idx >= len(o2_percent_ratio):
        print("유효하지 않은 인덱스입니다. 0으로 설정합니다.")
        percent_idx = 0
    o2_percent = o2_percent_ratio[percent_idx]

    thick_nm = int(input("두께 선택 (10, 20, 30, 40, 50, 60, 70, 80, 90 중 선택): "))
    if thick_nm not in thickness_nm:
        print("유효하지 않은 두께입니다. 10nm로 설정합니다.")
        thick_nm = 10

    print(f"선택된 조건: O2={o2_percent:.1f}%, 두께={thick_nm}nm")

    y_new = float(input("측정된 on/off ratio (상용로그 값): "))

    # 새로운 데이터 추가
    X_train = np.vstack([X_train, [[o2_percent, thick_nm]]])
    y_train = np.append(y_train, y_new)

    iteration += 1

print("\n베이지안 최적화가 완료되었습니다.")
print(f"최종 학습 데이터: {len(X_train)}개")
print(f"최고 on/off ratio: {np.max(y_train):.4f}")

# 최종 최적 조건 출력
best_idx = np.argmax(y_train)
best_o2_percent = X_train[best_idx, 0]
best_thickness_nm = X_train[best_idx, 1]

print(f"\n=== 최적 조건 ===")
print(f"O2 퍼센트: {best_o2_percent:.1f}%")
print(f"두께: {best_thickness_nm:.0f}nm")
print(f"on/off ratio: {np.max(y_train):.4f}")

print(f"\n=== 후보군 요약 ===")
print(f"O2 비율: {len(o2_values)}가지 ({o2_percent_ratio[0]:.1f}% ~ {o2_percent_ratio[-1]:.1f}%)")
print(f"두께: {len(thickness_nm)}가지 ({thickness_nm[0]}nm ~ {thickness_nm[-1]}nm)")
print(f"총 후보군: {len(o2_values)} × {len(thickness_nm)} = {len(o2_values) * len(thickness_nm)}개")
