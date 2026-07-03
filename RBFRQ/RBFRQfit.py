import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =========================================================
# 1) 모델 정의
# =========================================================

def gaussian(vg, b, A, mu, sigma):
    return b + A * np.exp(-((vg - mu) ** 2) / (2.0 * sigma ** 2))


def rq_bell(vg, b, A, mu, ell, alpha):
    return b + A * (1.0 + ((vg - mu) ** 2) / (2.0 * alpha * ell ** 2)) ** (-alpha)


def calc_aic(n, rss, k):
    rss = max(rss, 1e-30)
    return n * np.log(rss / n) + 2 * k


def calc_bic(n, rss, k):
    rss = max(rss, 1e-30)
    return n * np.log(rss / n) + k * np.log(n)


# =========================================================
# 2) wide-format 엑셀 로드
# =========================================================

def load_transfer_wide_xlsx(
    xlsx_path,
    sheet_name=0,
    use_abs_current=False,
):
    """
    네 데이터 형식:
      - 첫 row: Vd 값들 (10~30)
      - 첫 col: Vg 값들 (30~0)
      - 각 나머지 column: 한 개의 transfer curve (Id)

    반환:
      curves = [
        {"Vd": 10, "Vg": np.array(...), "Id": np.array(...)},
        ...
      ]
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    # 첫 row, 첫 col 분리
    vd_values = raw.iloc[0, 1:].to_numpy(dtype=float)
    vg_values = raw.iloc[1:, 0].to_numpy(dtype=float)

    curves = []
    for j, vd in enumerate(vd_values, start=1):
        Id = raw.iloc[1:, j].to_numpy(dtype=float)

        valid = np.isfinite(vg_values) & np.isfinite(Id)
        vg = vg_values[valid]
        current = Id[valid]

        if use_abs_current:
            current = np.abs(current)

        # 혹시 Vg가 descending이면 ascending으로 뒤집어도 되고, 그대로 써도 fit은 됨
        # 보기 편하게 ascending 정렬
        order = np.argsort(vg)
        vg = vg[order]
        current = current[order]

        curves.append({
            "Vd": float(vd),
            "Vg": vg,
            "Id": current,
        })

    return curves


# =========================================================
# 3) fitting 보조 함수
# =========================================================

def select_fit_region(vg, Id, peak_fraction=0.05, min_points=8):
    """
    peak 주변 주된 bell-shape 구간만 선택
    """
    peak = np.max(Id)
    if peak <= 0:
        mask = np.ones_like(Id, dtype=bool)
        return vg, Id, mask

    mask = Id >= (peak_fraction * peak)

    if np.sum(mask) < min_points:
        mask = np.ones_like(Id, dtype=bool)

    return vg[mask], Id[mask], mask


def initial_sigma_from_fwhm(vg, Id):
    peak_idx = np.argmax(Id)
    peak_val = Id[peak_idx]
    half_val = np.min(Id) + 0.5 * (peak_val - np.min(Id))

    above = np.where(Id >= half_val)[0]
    if len(above) >= 2:
        fwhm = vg[above[-1]] - vg[above[0]]
        sigma0 = max(fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))), 1e-6)
    else:
        sigma0 = max((vg.max() - vg.min()) / 6.0, 1e-6)

    return sigma0


def fit_gaussian(vg, Id):
    b0 = float(np.min(Id))
    A0 = float(np.max(Id) - np.min(Id))
    mu0 = float(vg[np.argmax(Id)])
    sigma0 = initial_sigma_from_fwhm(vg, Id)

    p0 = [b0, max(A0, 1e-20), mu0, sigma0]
    bounds = (
        [-np.inf, 0.0, vg.min(), 1e-9],
        [ np.inf, np.inf, vg.max(), np.inf]
    )

    popt, _ = curve_fit(
        gaussian,
        vg,
        Id,
        p0=p0,
        bounds=bounds,
        maxfev=30000,
    )

    pred = gaussian(vg, *popt)
    rss = float(np.sum((Id - pred) ** 2))

    return {
        "pred": pred,
        "params": {
            "b": popt[0],
            "A": popt[1],
            "mu": popt[2],
            "sigma": popt[3],
        },
        "rmse": float(np.sqrt(mean_squared_error(Id, pred))),
        "mae": float(mean_absolute_error(Id, pred)),
        "r2": float(r2_score(Id, pred)),
        "aic": float(calc_aic(len(vg), rss, 4)),
        "bic": float(calc_bic(len(vg), rss, 4)),
        "rss": rss,
    }


def fit_rq(vg, Id, alpha_init_list=(0.5, 1.0, 2.0, 5.0, 10.0)):
    b0 = float(np.min(Id))
    A0 = float(np.max(Id) - np.min(Id))
    mu0 = float(vg[np.argmax(Id)])
    ell0 = initial_sigma_from_fwhm(vg, Id)

    best = None

    for alpha0 in alpha_init_list:
        p0 = [b0, max(A0, 1e-20), mu0, ell0, alpha0]
        bounds = (
            [-np.inf, 0.0, vg.min(), 1e-9, 1e-6],
            [ np.inf, np.inf, vg.max(), np.inf, np.inf]
        )

        try:
            popt, _ = curve_fit(
                rq_bell,
                vg,
                Id,
                p0=p0,
                bounds=bounds,
                maxfev=50000,
            )

            pred = rq_bell(vg, *popt)
            rss = float(np.sum((Id - pred) ** 2))

            result = {
                "pred": pred,
                "params": {
                    "b": popt[0],
                    "A": popt[1],
                    "mu": popt[2],
                    "ell": popt[3],
                    "alpha": popt[4],
                },
                "rmse": float(np.sqrt(mean_squared_error(Id, pred))),
                "mae": float(mean_absolute_error(Id, pred)),
                "r2": float(r2_score(Id, pred)),
                "aic": float(calc_aic(len(vg), rss, 5)),
                "bic": float(calc_bic(len(vg), rss, 5)),
                "rss": rss,
            }

            if best is None or result["rss"] < best["rss"]:
                best = result

        except Exception:
            continue

    if best is None:
        raise RuntimeError("RQ fit failed.")

    return best


# =========================================================
# 4) curve 하나 분석
# =========================================================

def analyze_curve(vg, Id, peak_fraction=0.05):
    vg_fit, Id_fit, fit_mask = select_fit_region(vg, Id, peak_fraction=peak_fraction)

    g = fit_gaussian(vg_fit, Id_fit)
    r = fit_rq(vg_fit, Id_fit)

    winner_aic = "gaussian" if g["aic"] < r["aic"] else "rq"
    winner_bic = "gaussian" if g["bic"] < r["bic"] else "rq"

    return {
        "vg_fit": vg_fit,
        "Id_fit": Id_fit,
        "fit_mask": fit_mask,
        "gaussian": g,
        "rq": r,
        "winner_aic": winner_aic,
        "winner_bic": winner_bic,
    }


# =========================================================
# 5) 전체 Vd sweep 분석
# =========================================================

def analyze_wide_transfer_file(
    xlsx_path,
    output_dir="fit_results",
    sheet_name=0,
    peak_fraction=0.05,
    use_abs_current=False,
):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    curves = load_transfer_wide_xlsx(
        xlsx_path=xlsx_path,
        sheet_name=sheet_name,
        use_abs_current=use_abs_current,
    )

    summary_rows = []

    for curve in curves:
        vd = curve["Vd"]
        vg = curve["Vg"]
        Id = curve["Id"]

        try:
            result = analyze_curve(vg, Id, peak_fraction=peak_fraction)
            g = result["gaussian"]
            r = result["rq"]

            summary_rows.append({
                "Vd": vd,
                "n_points_raw": len(vg),
                "n_points_fit": len(result["vg_fit"]),

                "gaussian_rmse": g["rmse"],
                "gaussian_mae": g["mae"],
                "gaussian_r2": g["r2"],
                "gaussian_aic": g["aic"],
                "gaussian_bic": g["bic"],
                "gaussian_b": g["params"]["b"],
                "gaussian_A": g["params"]["A"],
                "gaussian_mu": g["params"]["mu"],
                "gaussian_sigma": g["params"]["sigma"],

                "rq_rmse": r["rmse"],
                "rq_mae": r["mae"],
                "rq_r2": r["r2"],
                "rq_aic": r["aic"],
                "rq_bic": r["bic"],
                "rq_b": r["params"]["b"],
                "rq_A": r["params"]["A"],
                "rq_mu": r["params"]["mu"],
                "rq_ell": r["params"]["ell"],
                "rq_alpha": r["params"]["alpha"],

                "winner_aic": result["winner_aic"],
                "winner_bic": result["winner_bic"],
            })

            # curve figure 저장
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(vg, Id, "ko", ms=3, label="raw")
            ax.plot(result["vg_fit"], g["pred"], "-", lw=2, label="Gaussian fit")
            ax.plot(result["vg_fit"], r["pred"], "--", lw=2, label="RQ fit")
            ax.set_title(f"Transfer curve fit | Vd={vd:.0f} V")
            ax.set_xlabel("Vg (V)")
            ax.set_ylabel("Id (A)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / f"fit_Vd_{int(vd)}.png", dpi=200)
            plt.close(fig)

        except Exception as e:
            summary_rows.append({
                "Vd": vd,
                "error": str(e),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "fit_summary.csv", index=False)

    valid = summary_df.dropna(subset=["gaussian_aic", "rq_aic"]).copy()

    if len(valid) > 0:
        # aggregate 요약
        agg = {
            "n_curves": len(valid),
            "gaussian_win_aic_count": int(np.sum(valid["winner_aic"] == "gaussian")),
            "rq_win_aic_count": int(np.sum(valid["winner_aic"] == "rq")),
            "gaussian_win_bic_count": int(np.sum(valid["winner_bic"] == "gaussian")),
            "rq_win_bic_count": int(np.sum(valid["winner_bic"] == "rq")),
            "mean_gaussian_rmse": float(valid["gaussian_rmse"].mean()),
            "mean_rq_rmse": float(valid["rq_rmse"].mean()),
            "mean_gaussian_r2": float(valid["gaussian_r2"].mean()),
            "mean_rq_r2": float(valid["rq_r2"].mean()),
            "mean_gaussian_sigma": float(valid["gaussian_sigma"].mean()),
            "std_gaussian_sigma": float(valid["gaussian_sigma"].std(ddof=1)) if len(valid) > 1 else 0.0,
            "mean_rq_alpha": float(valid["rq_alpha"].mean()),
            "std_rq_alpha": float(valid["rq_alpha"].std(ddof=1)) if len(valid) > 1 else 0.0,
        }
        pd.DataFrame([agg]).to_csv(outdir / "aggregate_summary.csv", index=False)

        # sigma / alpha trend
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(valid["Vd"], valid["gaussian_sigma"], "o-")
        axes[0].set_xlabel("Vd (V)")
        axes[0].set_ylabel("Gaussian sigma")
        axes[0].set_title("Gaussian sigma vs Vd")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(valid["Vd"], valid["rq_alpha"], "o-")
        axes[1].set_xlabel("Vd (V)")
        axes[1].set_ylabel("RQ alpha")
        axes[1].set_title("RQ alpha vs Vd")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(outdir / "parameter_trends.png", dpi=200)
        plt.close(fig)

        # AIC/BIC 비교
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(valid["Vd"], valid["gaussian_aic"], "o-", label="Gaussian")
        axes[0].plot(valid["Vd"], valid["rq_aic"], "s--", label="RQ")
        axes[0].set_xlabel("Vd (V)")
        axes[0].set_ylabel("AIC")
        axes[0].set_title("AIC vs Vd")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(valid["Vd"], valid["gaussian_bic"], "o-", label="Gaussian")
        axes[1].plot(valid["Vd"], valid["rq_bic"], "s--", label="RQ")
        axes[1].set_xlabel("Vd (V)")
        axes[1].set_ylabel("BIC")
        axes[1].set_title("BIC vs Vd")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(outdir / "aic_bic_vs_vd.png", dpi=200)
        plt.close(fig)

    return summary_df


# =========================================================
# 6) 실행 예시
# =========================================================

if __name__ == "__main__":
    summary = analyze_wide_transfer_file(
        xlsx_path="data.xlsx",
        output_dir="fit_results",
        sheet_name=0,
        peak_fraction=0.05,
        use_abs_current=False,   # p-type라서 음전류면 True로 바꿔서 먼저 확인
    )

    print(summary.head())
    print("Done.")