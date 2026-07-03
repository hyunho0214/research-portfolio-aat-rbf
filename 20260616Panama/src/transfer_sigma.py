"""Extract Gaussian sigma values from transfer-curve tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


@dataclass
class GaussianFit:
    column: str
    vd: float | None
    amplitude: float
    mu: float
    sigma: float
    baseline: float
    fwhm: float
    r2: float
    rmse: float
    n_points: int
    success: bool
    message: str


def gaussian_with_baseline(x: np.ndarray, amplitude: float, mu: float, sigma: float, baseline: float) -> np.ndarray:
    """Gaussian current model used to extract A, mu, and sigma."""

    return baseline + amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def load_transfer_table(path: Path, *, sheet_name: str | int | None = None) -> pd.DataFrame:
    """Load transfer-curve data from an Excel workbook."""

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path, sheet_name=0 if sheet_name is None else sheet_name)
    raise ValueError(f"Only Excel input is supported for this workflow, not: {path.suffix}")


def infer_vg_column(columns: Sequence[str], requested: str | None = None) -> str:
    """Infer the gate-voltage column name."""

    column_names = [str(col) for col in columns]
    if requested:
        if requested not in column_names:
            raise ValueError(f"Requested VG column not found: {requested}")
        return requested

    normalized = {col: re.sub(r"[^a-z0-9]", "", col.lower()) for col in column_names}
    candidates = {"vg", "vgs", "vgate", "gatevoltage", "gatev", "voltage"}
    for col, norm in normalized.items():
        if norm in candidates:
            return col
    return column_names[0]


def infer_id_columns(df: pd.DataFrame, vg_column: str, requested: Sequence[str] | None = None) -> list[str]:
    """Infer ID/current columns to fit."""

    if requested:
        missing = [col for col in requested if col not in df.columns]
        if missing:
            raise ValueError(f"Requested ID columns not found: {missing}")
        return [str(col) for col in requested]

    id_columns: list[str] = []
    for col in df.columns:
        col_name = str(col)
        if col_name == vg_column:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= 5:
            id_columns.append(col_name)
    if not id_columns:
        raise ValueError("No numeric ID columns found.")
    return id_columns


def parse_vd_from_column(column: str) -> float | None:
    """Best-effort VD value extraction from a column label."""

    text = str(column)
    patterns = [
        r"V[_\s-]*D\s*[=:]?\s*(-?\d+(?:\.\d+)?)",
        r"VD\s*(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*V",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def estimate_initial_sigma(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate sigma from FWHM; fall back to one sixth of the x span."""

    baseline = float(np.nanmin(y))
    peak = float(np.nanmax(y))
    half = baseline + 0.5 * (peak - baseline)
    above = x[y >= half]
    if len(above) >= 2:
        fwhm = float(np.max(above) - np.min(above))
        if fwhm > 0:
            return fwhm / 2.354820045
    span = float(np.max(x) - np.min(x))
    return max(span / 6.0, np.finfo(float).eps)


def fit_transfer_curve(
    vg: Sequence[float],
    current: Sequence[float],
    *,
    column: str,
    use_abs: bool = True,
    min_sigma: float | None = None,
    max_sigma: float | None = None,
) -> GaussianFit:
    """Fit one transfer curve to a Gaussian and return extracted parameters."""

    x = np.asarray(vg, dtype=float).reshape(-1)
    y = np.asarray(current, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if use_abs:
        y = np.abs(y)
    if len(x) < 5:
        return _failed_fit(column, "not enough finite points", len(x))

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = float(np.max(x) - np.min(x))
    step = float(np.median(np.diff(np.unique(x)))) if len(np.unique(x)) > 1 else span
    if span <= 0:
        return _failed_fit(column, "VG span must be positive", len(x))

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_scale = max(float(np.max(np.abs(y))), np.finfo(float).eps)
    y_fit_data = y / y_scale
    y_min_fit = float(np.min(y_fit_data))
    y_max_fit = float(np.max(y_fit_data))
    amplitude0 = max(y_max - y_min, np.finfo(float).eps)
    amplitude0_fit = max(y_max_fit - y_min_fit, np.finfo(float).eps)
    mu0 = float(x[int(np.argmax(y))])
    sigma0 = estimate_initial_sigma(x, y)
    min_sigma_value = min_sigma if min_sigma is not None else max(step / 5.0, span / 1000.0)
    max_sigma_value = max_sigma if max_sigma is not None else span * 2.0
    sigma0 = float(np.clip(sigma0, min_sigma_value, max_sigma_value))

    p0 = [amplitude0_fit, mu0, sigma0, y_min_fit]
    lower = [0.0, float(np.min(x)), min_sigma_value, -np.inf]
    upper = [np.inf, float(np.max(x)), max_sigma_value, np.inf]

    try:
        params, _ = curve_fit(
            gaussian_with_baseline,
            x,
            y_fit_data,
            p0=p0,
            bounds=(lower, upper),
            maxfev=50_000,
        )
        amplitude_fit, mu, sigma, baseline_fit = [float(value) for value in params]
        amplitude = amplitude_fit * y_scale
        baseline = baseline_fit * y_scale
        pred = gaussian_with_baseline(x, amplitude, mu, sigma, baseline)
        r2 = float(r2_score(y, pred))
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        return GaussianFit(
            column=str(column),
            vd=parse_vd_from_column(str(column)),
            amplitude=amplitude,
            mu=mu,
            sigma=abs(sigma),
            baseline=baseline,
            fwhm=2.354820045 * abs(sigma),
            r2=r2,
            rmse=rmse,
            n_points=int(len(x)),
            success=True,
            message="ok",
        )
    except Exception as exc:  # scipy raises several fit-specific subclasses
        return _failed_fit(column, f"{type(exc).__name__}: {exc}", len(x))


def _failed_fit(column: str, message: str, n_points: int) -> GaussianFit:
    return GaussianFit(
        column=str(column),
        vd=parse_vd_from_column(str(column)),
        amplitude=np.nan,
        mu=np.nan,
        sigma=np.nan,
        baseline=np.nan,
        fwhm=np.nan,
        r2=np.nan,
        rmse=np.nan,
        n_points=int(n_points),
        success=False,
        message=message,
    )


def fit_transfer_table(
    df: pd.DataFrame,
    *,
    vg_column: str | None = None,
    id_columns: Sequence[str] | None = None,
    use_abs: bool = True,
    min_sigma: float | None = None,
    max_sigma: float | None = None,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Fit all requested ID columns in a transfer table."""

    df = df.copy()
    df.columns = [str(col) for col in df.columns]
    vg_name = infer_vg_column([str(col) for col in df.columns], vg_column)
    selected_id_columns = infer_id_columns(df, vg_name, id_columns)
    vg = pd.to_numeric(df[vg_name], errors="coerce").to_numpy(dtype=float)
    fits = [
        fit_transfer_curve(
            vg,
            pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float),
            column=str(col),
            use_abs=use_abs,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
        )
        for col in selected_id_columns
    ]
    rows = [fit.__dict__ for fit in fits]
    result = pd.DataFrame(rows)
    result.insert(0, "curve_index", np.arange(1, len(result) + 1))
    return result, vg_name, selected_id_columns


def load_sigma_values(path: Path, *, column: str = "sigma") -> np.ndarray:
    """Load positive sigma values from an Excel result file."""

    df = load_transfer_table(path)
    if column not in df.columns:
        raise ValueError(f"Sigma column not found: {column}")
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        raise ValueError(f"No positive sigma values found in {path}")
    return values
