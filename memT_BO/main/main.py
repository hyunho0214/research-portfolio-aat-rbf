import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel,ConstantKernel
from scipy.stats import norm
import pandas as pd

# 1. 실험 가능한 O2/Ar 비율 후보 생성 (0/20, 1/19, ..., 10/10)
o2_values = np.arange(0, 11)  # O2 값: 0, 1, 2, ..., 10
ar_values = 20 - o2_values    # Ar 값: 20, 19, 18, ..., 10
ratio_candidates = o2_values / ar_values  # 실제 비율 값 (0.0, 0.053, ..., 1.0)
ratio_labels = [f"{o2}/{ar}" for o2, ar in zip(o2_values, ar_values)]

# 두께 정보
thickness = [(10 + 5*i) for i in range(10)]

# 2D 후보 생성 (모든 ratio와 thickness 조합)
X_candidates = []
candidate_labels = []
for i, ratio in enumerate(ratio_candidates):
    for j, thick in enumerate(thickness):
        X_candidates.append([ratio, thick])
        candidate_labels.append(f"Ratio:{ratio_labels[i]}, Thick:{thick}")
X_candidates = np.array(X_candidates)

# 엑셀 저장 함수
def save_prediction_to_excel(iteration, X_candidates, candidate_labels, mu, sigma, X_train, y_train, next_idx=None):
    # 실험 데이터가 있는 위치에만 값을 넣고, 나머지는 NaN
    exp_data = np.full_like(mu, np.nan, dtype=float)
    for i, (x_train, y_train_val) in enumerate(zip(X_train, y_train)):
        # 가장 가까운 후보 찾기
        distances = np.sqrt(np.sum((X_candidates - x_train)**2, axis=1))
        closest_idx = np.argmin(distances)
        exp_data[closest_idx] = y_train_val

    # 다음 추천 포인트 위치에만 값을 넣고, 나머지는 NaN
    next_point = np.full_like(mu, np.nan, dtype=float)
    if next_idx is not None:
        next_point[next_idx] = mu[next_idx]

    df = pd.DataFrame({
        'Ratio': X_candidates[:, 0],
        'Thickness': X_candidates[:, 1],
        'Candidate_Label': candidate_labels,
        'Mobility_Prediction': mu,
        'Mobility_CI_Lower': mu - 1.96 * sigma,
        'Mobility_CI_Upper': mu + 1.96 * sigma,
        'Standard_Deviation': sigma,
        'Experimental_Mobility': exp_data,
        'Next_Point_Mobility': next_point
    })
    filename = f'mobility_prediction_iter_{iteration}.xlsx'
    df.to_excel(filename, index=False)
    print(f'엑셀 파일 저장 완료: {filename}')

# 2. 초기 실험 데이터 입력
print("초기 실험값을 입력하세요.")
init_x = []
init_y = []
n_init = 3  # 초기 실험 데이터 개수
print("O2/Ar 비율 후보:")
for i, label in enumerate(ratio_labels):
    print(f"{i}: {label} (실제 비율: {ratio_candidates[i]:.3f})")
print("두께 후보:")
for i, thick in enumerate(thickness):
    print(f"{i}: {thick}")

for i in range(n_init):
    ratio_idx = int(input(f"실험 {i+1} O2/Ar 비율 (위 목록에서 번호 선택): "))
    ratio = ratio_candidates[ratio_idx]
    thick_idx = int(input(f"실험 {i+1} 두께 (위 목록에서 번호 선택): "))
    thick = thickness[thick_idx]
    y = float(input(f"실험 {i+1} mobility (1~100): "))
    init_x.append([ratio, thick])
    init_y.append(y)

X_train = np.array(init_x)
y_train = np.array(init_y)

# 3. 반복 실험 추천 루프
iteration = 0
while True:
    # GPR 모델 정의 및 학습 (2D 입력용)
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.1, 1000.0)) * RBF(length_scale=[0.2, 5.0], length_scale_bounds=(1e-5, 10.0)) + WhiteKernel(noise_level=0.001, noise_level_bounds="fixed")
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False, n_restarts_optimizer=10, random_state=0)
    gpr.fit(X_train, y_train)

    # 후보 조합 예측 및 EI 계산
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
    next_idx = np.argmax(ei)
    next_ratio = X_candidates[next_idx, 0]
    next_thick = X_candidates[next_idx, 1]
    next_ratio_idx = np.argmin(np.abs(ratio_candidates - next_ratio))
    next_thick_idx = np.argmin(np.abs(np.array(thickness) - next_thick))
    next_label = candidate_labels[next_idx]

    # 엑셀로 예측 결과 저장
    save_prediction_to_excel(
        iteration=iteration,
        X_candidates=X_candidates,
        candidate_labels=candidate_labels,
        mu=mu,
        sigma=sigma,
        X_train=X_train,
        y_train=y_train,
        next_idx=next_idx
    )

    print(f"\n[추천] 다음 실험 추천: {next_label} (EI 최대)")

    # 기존 2D 그래프 복원 (O2/Ar ratio vs Mobility)
    # 부드러운 곡선용: 촘촘한 x축 (0~1 범위로 확장)
    x_fine = np.linspace(0, 1.0, 500).reshape(-1, 1)
    # 두께는 평균값으로 고정하여 2D 예측
    thick_mean = np.mean(thickness)
    X_fine_2d = np.column_stack([x_fine.flatten(), np.full(len(x_fine), thick_mean)])
    mu_fine, sigma_fine = gpr.predict(X_fine_2d, return_std=True)

    # 2D 그래프용 EI 계산 (각 ratio에 대해 최대 EI)
    ei_2d = []
    for ratio in ratio_candidates:
        # 해당 ratio의 모든 thickness 조합에서 최대 EI 선택
        ratio_mask = np.abs(X_candidates[:, 0] - ratio) < 1e-10
        ei_2d.append(np.max(ei[ratio_mask]))
    ei_2d = np.array(ei_2d)

    # Figure 6 스타일 plot (기존 2D 그래프)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})

    # 상단 그래프 (mobility 예측)
    ax1.plot(x_fine, mu_fine, 'b-', label='Prediction')
    ax1.fill_between(x_fine.flatten(), mu_fine-1.96*sigma_fine, mu_fine+1.96*sigma_fine, color='skyblue', alpha=0.5, label='95% Confidence Interval')
    ax1.scatter(X_train[:, 0], y_train, color='green', marker='D', label='Real data')
    ax1.set_ylabel('Mobility (cm$^2$/Vs)')
    ax1.legend(loc='upper left')
    ax1.set_title('Bayesian Optimization for IGZO TFT (O2/Ar ratio)')

    # 하단 그래프 (EI)
    ax2.plot(ratio_candidates, ei_2d, 'k-', label='EI')
    next_ratio_ei = ei_2d[next_ratio_idx]
    ax2.scatter([next_ratio], [next_ratio_ei], color='orange', marker='*', s=150, label='Max EI (Next)')
    ax2.set_xlabel('O2/Ar ratio')
    ax2.set_ylabel('EI')
    ax2.set_ylim(0, max(ei_2d) * 1.1)
    plt.xticks(ratio_candidates, ratio_labels, rotation=45)
    ax2.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    # 3D 시각화 (ratio vs thickness vs mobility)
    fig = plt.figure(figsize=(15, 5))

    # 첫 번째 서브플롯: 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(X_candidates[:, 0], X_candidates[:, 1], mu, c=mu, cmap='viridis', alpha=0.6)
    ax1.scatter(X_train[:, 0], X_train[:, 1], y_train, color='red', s=100, marker='D', label='Real data')
    ax1.scatter([next_ratio], [next_thick], [mu[next_idx]], color='orange', s=200, marker='*', label='Next')
    ax1.set_xlabel('O2/Ar ratio')
    ax1.set_ylabel('Thickness')
    ax1.set_zlabel('Mobility')
    ax1.set_title('3D Mobility Prediction')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, shrink=0.5)

    # 두 번째 서브플롯: EI heatmap
    ax2 = fig.add_subplot(132)
    ei_matrix = ei.reshape(len(ratio_candidates), len(thickness))
    im = ax2.imshow(ei_matrix.T, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
    ax2.set_xlabel('O2/Ar ratio index')
    ax2.set_ylabel('Thickness index')
    ax2.set_title('Expected Improvement Heatmap')
    ax2.set_xticks(range(len(ratio_candidates)))
    ax2.set_xticklabels([f"{i}" for i in range(len(ratio_candidates))])
    ax2.set_yticks(range(len(thickness)))
    ax2.set_yticklabels([f"{i}" for i in range(len(thickness))])
    plt.colorbar(im, ax=ax2)

    # 세 번째 서브플롯: Mobility heatmap
    ax3 = fig.add_subplot(133)
    mu_matrix = mu.reshape(len(ratio_candidates), len(thickness))
    im2 = ax3.imshow(mu_matrix.T, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    ax3.set_xlabel('O2/Ar ratio index')
    ax3.set_ylabel('Thickness index')
    ax3.set_title('Mobility Prediction Heatmap')
    ax3.set_xticks(range(len(ratio_candidates)))
    ax3.set_xticklabels([f"{i}" for i in range(len(ratio_candidates))])
    ax3.set_yticks(range(len(thickness)))
    ax3.set_yticklabels([f"{i}" for i in range(len(thickness))])

    # 실험 데이터 점 표시
    for x_train, y_train_val in zip(X_train, y_train):
        ratio_idx = np.argmin(np.abs(ratio_candidates - x_train[0]))
        thick_idx = np.argmin(np.abs(np.array(thickness) - x_train[1]))
        ax3.scatter(ratio_idx, thick_idx, color='red', s=100, marker='D')

    # 추천 점 표시
    ax3.scatter(next_ratio_idx, next_thick_idx, color='orange', s=200, marker='*')

    plt.colorbar(im2, ax=ax3)
    plt.tight_layout()
    plt.show()

    # 4. 사용자 입력: 실험 종료 여부
    cont = input("계속 실험을 진행하려면 Enter, 종료하려면 q 입력: ")
    if cont.lower() == 'q':
        print("실험 추천을 종료합니다.")
        break

    # 5. 다음 실험 데이터 추가 입력
    print("O2/Ar 비율 후보:")
    for i, label in enumerate(ratio_labels):
        print(f"{i}: {label} (실제 비율: {ratio_candidates[i]:.3f})")
    print("두께 후보:")
    for i, thick in enumerate(thickness):
        print(f"{i}: {thick}")

    ratio_idx = int(input(f"실험 O2/Ar 비율 (추천: {ratio_labels[next_ratio_idx]}, 위 목록에서 번호 선택): "))
    ratio = ratio_candidates[ratio_idx]
    thick_idx = int(input(f"실험 두께 (추천: {thickness[next_thick_idx]}, 위 목록에서 번호 선택): "))
    thick = thickness[thick_idx]
    y_new = float(input("실험 mobility (1~100): "))

    X_train = np.vstack([X_train, [[ratio, thick]]])
    y_train = np.append(y_train, y_new)
    iteration += 1