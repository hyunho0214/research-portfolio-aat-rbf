import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.impute import SimpleImputer

class BayesianOptimizationTFTFullCSV:
    def __init__(self, csv_filepath, nan_strategy='drop'):
        """
        nan_strategy: 'drop', 'mean', 'median', 'most_frequent'
        """
        self.kernel = (
            C(1.0, (1e-3, 1e3)) *
            Matern(length_scale=[1.0,1.0,1.0],
                   length_scale_bounds=(1e-2,1e2),
                   nu=2.5)
            + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10,1e1))
        )
        self.gpr = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=0.0,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=42
        )
        self.csv_filepath = csv_filepath
        self.nan_strategy = nan_strategy
        self.X_sample = None
        self.Y_sample = None
        self._load_data()

    def _load_data(self):
        """CSV에서 데이터 읽기 + NaN 값 처리"""
        try:
            df = pd.read_csv(self.csv_filepath)
            print(f"원본 데이터 크기: {len(df)} 행")

            # 필수 공정 컬럼 확인
            proc_cols = ['power', 'pressure', 'Gas ratio']
            for col in proc_cols:
                if col not in df.columns:
                    raise ValueError(f"CSV에 '{col}' 컬럼이 필요합니다.")

            # NaN 값 확인
            print("\n=== NaN 값 확인 ===")
            nan_info = df.isnull().sum()
            if nan_info.sum() > 0:
                print("각 컬럼별 NaN 개수:")
                print(nan_info[nan_info > 0])
                print(f"총 NaN 값 개수: {df.isnull().sum().sum()}")

                # NaN이 포함된 행 출력
                nan_rows = df[df.isnull().any(axis=1)]
                if len(nan_rows) > 0:
                    print("\nNaN이 포함된 행:")
                    print(nan_rows)
            else:
                print("NaN 값이 없습니다.")

            # NaN 값 처리
            df_cleaned = self._handle_nan_values(df)
            print(f"처리 후 데이터 크기: {len(df_cleaned)} 행")

            # 공정 조건 추출
            self.X_sample = df_cleaned[proc_cols].values

            # 전기적 특성으로 FoM 계산 또는 FoM 컬럼 직접 사용
            if set(['mu_fe','vth','ss']).issubset(df_cleaned.columns):
                print("전기적 특성에서 FoM 계산")
                mu = df_cleaned['mu_fe'].tolist()
                vth = df_cleaned['vth'].tolist()
                ss = df_cleaned['ss'].tolist()
                self.Y_sample = np.array(self._calc_fom(mu, vth, ss))
            elif 'fom' in df_cleaned.columns:
                print("FoM 컬럼 직접 사용")
                self.Y_sample = df_cleaned['fom'].values
            else:
                raise ValueError("CSV에 mu_fe,vth,ss 또는 fom 컬럼이 필요합니다.")

            # 최종 데이터 확인
            print(f"\n최종 학습 데이터: X={self.X_sample.shape}, y={self.Y_sample.shape}")
            print(f"FoM 범위: {self.Y_sample.min():.4f} ~ {self.Y_sample.max():.4f}")

            # 최종적으로 NaN 확인
            if np.isnan(self.X_sample).any() or np.isnan(self.Y_sample).any():
                raise ValueError("처리 후에도 NaN 값이 남아있습니다.")

        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            raise

    def _handle_nan_values(self, df):
        """NaN 값 처리 전략"""
        if self.nan_strategy == 'drop':
            # NaN이 포함된 행 삭제
            df_cleaned = df.dropna()
            dropped_rows = len(df) - len(df_cleaned)
            if dropped_rows > 0:
                print(f"NaN이 포함된 {dropped_rows}개 행을 삭제했습니다.")

        elif self.nan_strategy in ['mean', 'median', 'most_frequent']:
            # Imputer를 사용한 결측값 보간
            print(f"NaN 값을 {self.nan_strategy} 방법으로 보간합니다.")

            # 숫자형 컬럼만 보간
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if self.nan_strategy == 'most_frequent':
                imputer = SimpleImputer(strategy='most_frequent')
            else:
                imputer = SimpleImputer(strategy=self.nan_strategy)

            df_cleaned = df.copy()
            df_cleaned[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        else:
            raise ValueError("nan_strategy는 'drop', 'mean', 'median', 'most_frequent' 중 하나여야 합니다.")

        return df_cleaned

    def _calc_fom(self, mu_list, vth_list, ss_list):
        """논문과 동일한 FoM 계산"""
        def normalize(arr):
            mn, mx = min(arr), max(arr)
            if mx == mn:
                return [0.5]*len(arr)
            return [(v-mn)/(mx-mn) for v in arr]

        mu_n = normalize(mu_list)
        vth_n = normalize([abs(v) for v in vth_list])
        ss_n = normalize(ss_list)
        fom = [
            0.4*mu_n[i] + 0.4*(1 - vth_n[i]) + 0.2*(1 - ss_n[i])
            for i in range(len(mu_list))
        ]
        return fom

    def fit(self):
        """GPR 모델 학습 + 추가 안전성 검사"""
        if self.X_sample is None or self.Y_sample is None:
            raise ValueError("데이터가 로드되지 않았습니다.")

        # 최종 NaN 확인
        if np.isnan(self.X_sample).any():
            raise ValueError("X_sample에 NaN 값이 있습니다.")
        if np.isnan(self.Y_sample).any():
            raise ValueError("Y_sample에 NaN 값이 있습니다.")

        print("GPR 모델 학습 시작...")
        self.gpr.fit(self.X_sample, self.Y_sample)
        print("GPR 모델 학습 완료")

    def expected_improvement(self, X_cand):
        """논문 원본 EI 수식 구현"""
        mu, sigma = self.gpr.predict(X_cand, return_std=True)
        sigma = sigma.reshape(-1,1)
        mu_opt = np.max(self.Y_sample)
        with np.errstate(divide='warn'):
            z = (mu - mu_opt) / sigma
            ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
            ei[sigma == 0.0] = 0.0
        return ei

    def recommend_next(self, n_candidates=1000):
        """다음 실험 조건 추천"""
        X_cand = np.random.uniform(
            low=[50,1,10],
            high=[200,10,50],
            size=(n_candidates,3)
        )
        ei = self.expected_improvement(X_cand)
        idx = np.argmax(ei)
        next_x = X_cand[idx]
        return {
            'Plasma Power': round(next_x[0],2),
            'Pressure': round(next_x[1],2),
            'Gas Ratio': round(next_x[2],2),
            'Expected Improvement': float(ei[idx])
        }

    def display_data_summary(self):
        """데이터 요약 정보 출력"""
        print("\n=== 데이터 요약 ===")
        print(f"총 샘플 수: {len(self.X_sample)}")
        print("공정 조건 범위:")
        print(f"  Power: {self.X_sample[:,0].min():.1f} ~ {self.X_sample[:,0].max():.1f}")
        print(f"  Pressure: {self.X_sample[:,1].min():.1f} ~ {self.X_sample[:,1].max():.1f}")
        print(f"  Gas Ratio: {self.X_sample[:,2].min():.1f} ~ {self.X_sample[:,2].max():.1f}")
        print(f"FoM 범위: {self.Y_sample.min():.4f} ~ {self.Y_sample.max():.4f}")

# 사용 예시
if __name__ == "__main__":
    try:
        # 방법 1: NaN 행 삭제 (기본값)
        bo = BayesianOptimizationTFTFullCSV("full_data.csv", nan_strategy='drop')

        # 방법 2: 평균값으로 보간
        # bo = BayesianOptimizationTFTFullCSV("full_data.csv", nan_strategy='mean')

        # 방법 3: 중앙값으로 보간
        # bo = BayesianOptimizationTFTFullCSV("full_data.csv", nan_strategy='median')

        bo.display_data_summary()
        bo.fit()
        recommendation = bo.recommend_next()

        print("\n=== 다음 추천 실험 조건 ===")
        for k,v in recommendation.items():
            print(f"{k}: {v}")

    except Exception as e:
        print(f"실행 오류: {e}")
        print("\nCSV 파일을 확인해주세요:")
        print("1. 파일 경로가 올바른지")
        print("2. 필수 컬럼이 있는지 (power, pressure, Gas ratio)")
        print("3. 데이터 형식이 올바른지")
