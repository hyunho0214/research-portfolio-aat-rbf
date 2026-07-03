"""Calibration utilities for measured AAT transfer curves."""

from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit


REQUIRED_CURVE_COLUMNS = ("curve_id", "Vg", "response")
DEFAULT_METADATA_COLUMNS = (
    "device_id",
    "operating_mode",
    "Vd",
    "light_intensity",
    "wavelength",
    "pulse_width",
    "state_tag",
)
EPS = 1e-12


def gaussian_response(vg: np.ndarray, baseline: float, A: float, mu: float, sigma: float) -> np.ndarray:
    """Evaluate the fitted transfer-curve Gaussian response."""

    sigma_safe = max(float(abs(sigma)), EPS)
    return baseline + A * np.exp(-((vg - mu) ** 2) / (2.0 * sigma_safe**2))


def load_transfer_curves(path: str | Path) -> pd.DataFrame:
    """Load long-format transfer curves and validate required columns."""

    path = Path(path)
    curves = pd.read_csv(path)
    missing = [col for col in REQUIRED_CURVE_COLUMNS if col not in curves.columns]
    if missing:
        raise ValueError(f"Missing required curve columns: {missing}")

    curves = curves.copy()
    curves["Vg"] = pd.to_numeric(curves["Vg"], errors="coerce")
    curves["response"] = pd.to_numeric(curves["response"], errors="coerce")
    curves = curves.dropna(subset=["curve_id", "Vg", "response"])
    if curves.empty:
        raise ValueError("No valid curve rows remain after dropping invalid Vg/response values.")

    return curves.sort_values(["curve_id", "Vg"]).reset_index(drop=True)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= EPS:
        return 1.0 if ss_res <= EPS else float("nan")
    return 1.0 - ss_res / ss_tot


def _fallback_fit(vg: np.ndarray, response: np.ndarray) -> dict[str, float]:
    baseline = float(np.median(response))
    peak_idx = int(np.argmax(np.abs(response - baseline)))
    A = float(response[peak_idx] - baseline)
    mu = float(vg[peak_idx])
    vg_span = max(float(np.ptp(vg)), EPS)
    sigma = vg_span / 6.0
    y_fit = gaussian_response(vg, baseline, A, mu, sigma)
    mse = float(np.mean((response - y_fit) ** 2))
    return {
        "baseline": baseline,
        "A": A,
        "mu": mu,
        "sigma": sigma,
        "fit_mse": mse,
        "fit_r2": _r2_score(response, y_fit),
        "fit_success": False,
    }


def fit_gaussian_curve(vg: np.ndarray, response: np.ndarray) -> dict[str, float]:
    """Fit response = baseline + A * exp(-(Vg - mu)^2 / (2 * sigma^2))."""

    vg = np.asarray(vg, dtype=float)
    response = np.asarray(response, dtype=float)
    valid = np.isfinite(vg) & np.isfinite(response)
    vg = vg[valid]
    response = response[valid]

    if vg.size < 4:
        raise ValueError("At least four valid points are required for Gaussian fitting.")

    order = np.argsort(vg)
    vg = vg[order]
    response = response[order]

    vg_min = float(np.min(vg))
    vg_max = float(np.max(vg))
    vg_span = max(vg_max - vg_min, EPS)
    y_min = float(np.min(response))
    y_max = float(np.max(response))
    y_span = max(y_max - y_min, EPS)

    if y_span <= EPS:
        result = _fallback_fit(vg, response)
        result["fit_success"] = True
        return result

    baseline_low = y_min - 2.0 * y_span
    baseline_high = y_max + 2.0 * y_span
    bounds = (
        [baseline_low, -5.0 * y_span, vg_min, vg_span / 10000.0],
        [baseline_high, 5.0 * y_span, vg_max, max(2.0 * vg_span, vg_span / 10000.0)],
    )
    sigma0 = max(vg_span / 6.0, vg_span / 1000.0)
    p0_candidates = [
        [y_min, y_max - y_min, float(vg[int(np.argmax(response))]), sigma0],
        [y_max, y_min - y_max, float(vg[int(np.argmin(response))]), sigma0],
        [float(np.median(response)), float(response[np.argmax(np.abs(response - np.median(response)))] - np.median(response)), float(vg[int(np.argmax(np.abs(response - np.median(response))))]), sigma0],
    ]

    best: tuple[float, np.ndarray] | None = None
    for p0 in p0_candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                params, _ = curve_fit(
                    gaussian_response,
                    vg,
                    response,
                    p0=p0,
                    bounds=bounds,
                    maxfev=20000,
                )
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        y_fit = gaussian_response(vg, *params)
        mse = float(np.mean((response - y_fit) ** 2))
        if best is None or mse < best[0]:
            best = (mse, params)

    if best is None:
        return _fallback_fit(vg, response)

    mse, params = best
    baseline, A, mu, sigma = [float(x) for x in params]
    sigma = max(abs(sigma), EPS)
    y_fit = gaussian_response(vg, baseline, A, mu, sigma)
    return {
        "baseline": baseline,
        "A": A,
        "mu": mu,
        "sigma": sigma,
        "fit_mse": mse,
        "fit_r2": _r2_score(response, y_fit),
        "fit_success": True,
    }


def _first_metadata_row(group: pd.DataFrame, metadata_columns: list[str]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    first = group.iloc[0]
    for col in metadata_columns:
        if col in group.columns:
            metadata[col] = first[col]
    return metadata


def calibrate_kernel_library(
    curves: pd.DataFrame,
    metadata_columns: list[str] | None = None,
    amplitude_normalization: str = "median_abs",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit all curves and return library, fit metrics, and normalized curves."""

    missing = [col for col in REQUIRED_CURVE_COLUMNS if col not in curves.columns]
    if missing:
        raise ValueError(f"Missing required curve columns: {missing}")

    if metadata_columns is None:
        metadata_columns = [col for col in DEFAULT_METADATA_COLUMNS if col in curves.columns]

    rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    for curve_id, group in curves.groupby("curve_id", sort=True):
        group = group.sort_values("Vg")
        fit = fit_gaussian_curve(group["Vg"].to_numpy(), group["response"].to_numpy())
        metadata = _first_metadata_row(group, metadata_columns)
        row: dict[str, object] = {
            "kernel_id": curve_id,
            "curve_id": curve_id,
            "n_points": int(len(group)),
            "Vg_min": float(group["Vg"].min()),
            "Vg_max": float(group["Vg"].max()),
            **metadata,
            **fit,
        }
        rows.append(row)
        metrics_rows.append(
            {
                "kernel_id": curve_id,
                "curve_id": curve_id,
                "fit_mse": fit["fit_mse"],
                "fit_r2": fit["fit_r2"],
                "fit_success": fit["fit_success"],
                "n_points": int(len(group)),
            }
        )

    library = pd.DataFrame(rows)
    if library.empty:
        raise ValueError("No transfer curves were available for calibration.")

    if amplitude_normalization != "median_abs":
        raise ValueError("Only amplitude_normalization='median_abs' is currently supported.")

    abs_A = np.abs(pd.to_numeric(library["A"], errors="coerce").to_numpy(dtype=float))
    positive = abs_A[np.isfinite(abs_A) & (abs_A > EPS)]
    scale = float(np.median(positive)) if positive.size else 1.0
    library["A_tilde"] = pd.to_numeric(library["A"], errors="coerce") / scale
    library["amplitude_normalization"] = amplitude_normalization
    library["amplitude_scale"] = scale

    metrics = pd.DataFrame(metrics_rows)
    normalized_curves = build_normalized_transfer_curves(curves, library)
    return library, metrics, normalized_curves


def build_normalized_transfer_curves(curves: pd.DataFrame, library: pd.DataFrame) -> pd.DataFrame:
    """Normalize measured curves for direct measured-curve kernels."""

    fit_cols = ["curve_id", "baseline", "A", "A_tilde"]
    merged = curves.merge(library[fit_cols], on="curve_id", how="inner")
    denom = pd.to_numeric(merged["A"], errors="coerce").to_numpy(dtype=float)
    scale = pd.to_numeric(merged["A_tilde"], errors="coerce").to_numpy(dtype=float)
    centered = pd.to_numeric(merged["response"], errors="coerce").to_numpy(dtype=float) - pd.to_numeric(
        merged["baseline"], errors="coerce"
    ).to_numpy(dtype=float)
    response_norm = np.zeros_like(centered, dtype=float)
    valid = np.isfinite(centered) & np.isfinite(denom) & (np.abs(denom) > EPS) & np.isfinite(scale)
    response_norm[valid] = centered[valid] / denom[valid] * scale[valid]
    response_norm[~valid] = centered[~valid]

    keep_cols = ["curve_id", "Vg", "response"]
    metadata_cols = [col for col in DEFAULT_METADATA_COLUMNS if col in merged.columns]
    out = merged[keep_cols + metadata_cols].copy()
    out = out.rename(columns={"response": "response_raw"})
    out["response_norm"] = response_norm
    return out.sort_values(["curve_id", "Vg"]).reset_index(drop=True)


def save_calibration_outputs(
    curves_path: str | Path,
    out_dir: str | Path,
    metadata_columns: list[str] | None = None,
    make_plots: bool = True,
    max_plot_curves: int = 16,
) -> dict[str, Path]:
    """Run calibration from a CSV path and save all planned artifacts."""

    curves_path = Path(curves_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = load_transfer_curves(curves_path)
    library, metrics, normalized_curves = calibrate_kernel_library(curves, metadata_columns=metadata_columns)

    library_path = out_dir / "kernel_library.csv"
    metrics_path = out_dir / "curve_fit_metrics.csv"
    normalized_path = out_dir / "normalized_transfer_curves.csv"
    config_path = out_dir / "calibration_config.json"

    library.to_csv(library_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    normalized_curves.to_csv(normalized_path, index=False)

    config = {
        "curves_path": str(curves_path),
        "required_columns": list(REQUIRED_CURVE_COLUMNS),
        "metadata_columns": metadata_columns,
        "amplitude_normalization": "median_abs",
        "n_kernels": int(len(library)),
        "outputs": {
            "kernel_library": str(library_path),
            "curve_fit_metrics": str(metrics_path),
            "normalized_transfer_curves": str(normalized_path),
        },
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    paths = {
        "kernel_library": library_path,
        "curve_fit_metrics": metrics_path,
        "normalized_transfer_curves": normalized_path,
        "calibration_config": config_path,
    }

    if make_plots:
        plot_path = out_dir / "curve_fit_examples.png"
        plot_curve_fits(curves, library, plot_path, max_curves=max_plot_curves)
        paths["curve_fit_examples"] = plot_path

    return paths


def plot_curve_fits(
    curves: pd.DataFrame,
    library: pd.DataFrame,
    out_path: str | Path,
    max_curves: int = 16,
) -> Path:
    """Save a compact grid of raw curves and Gaussian fits."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    plot_library = library.head(max_curves)
    n = len(plot_library)
    if n == 0:
        raise ValueError("No calibrated curves available to plot.")

    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, row in zip(axes.ravel(), plot_library.to_dict("records")):
        group = curves[curves["curve_id"] == row["curve_id"]].sort_values("Vg")
        vg = group["Vg"].to_numpy(dtype=float)
        response = group["response"].to_numpy(dtype=float)
        fit_vg = np.linspace(float(np.min(vg)), float(np.max(vg)), 300)
        fit_response = gaussian_response(fit_vg, row["baseline"], row["A"], row["mu"], row["sigma"])
        ax.axis("on")
        ax.scatter(vg, response, s=12, alpha=0.75, label="measured")
        ax.plot(fit_vg, fit_response, color="black", linewidth=1.5, label="fit")
        ax.set_title(f"{row['curve_id']} R2={row['fit_r2']:.3f}", fontsize=9)
        ax.set_xlabel("Vg")
        ax.set_ylabel("response")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path
