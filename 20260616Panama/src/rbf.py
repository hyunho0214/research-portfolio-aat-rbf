"""Multi-Gaussian RBF utilities following the local SI pseudocode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class RBFRunResult:
    """Predictions and metrics for one value of N."""

    n_distinct: int
    sigma_set: np.ndarray
    centers: np.ndarray
    sigma_list: np.ndarray
    weights: np.ndarray
    y_pred: np.ndarray
    metrics: dict[str, float]


def parse_n_values(value: str | Sequence[int]) -> list[int]:
    """Parse N values from forms like "1:80", "1..80", or "1,3,10"."""

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("N value list cannot be empty.")
        if ".." in text:
            start_s, end_s = text.split("..", 1)
            return list(range(int(start_s), int(end_s) + 1))
        if ":" in text and "," not in text:
            parts = [int(part) for part in text.split(":")]
            if len(parts) == 2:
                start, end = parts
                step = 1
            elif len(parts) == 3:
                start, end, step = parts
            else:
                raise ValueError(f"Could not parse N values: {value}")
            if step == 0:
                raise ValueError("N range step cannot be zero.")
            return list(range(start, end + (1 if step > 0 else -1), step))
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    return [int(item) for item in value]


def create_sliding_window_1d(sequence: Sequence[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Create one-step-ahead supervised data from a 1D time series."""

    values = np.asarray(sequence, dtype=float).reshape(-1)
    if window < 1:
        raise ValueError("window must be positive.")
    if len(values) <= window:
        raise ValueError("sequence length must be larger than window.")

    x = np.empty((len(values) - window, window), dtype=float)
    y = np.empty((len(values) - window, 1), dtype=float)
    for i in range(len(values) - window):
        x[i] = values[i : i + window]
        y[i, 0] = values[i + window]
    return x, y


def create_sliding_window_multivariate(data: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Create one-step-ahead supervised data from a multivariate time series."""

    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("data must have shape (num_samples, num_features).")
    if window < 1:
        raise ValueError("window must be positive.")
    if len(values) <= window:
        raise ValueError("data length must be larger than window.")

    n_samples, n_features = values.shape
    x = np.empty((n_samples - window, window * n_features), dtype=float)
    y = np.empty((n_samples - window, n_features), dtype=float)
    for i in range(n_samples - window):
        x[i] = values[i : i + window].reshape(-1)
        y[i] = values[i + window]
    return x, y


def log_spaced_sigmas(sigma_min: float, sigma_max: float, n_distinct: int) -> np.ndarray:
    """Select N log-spaced sigma values within the configured experimental range."""

    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must be positive.")
    if sigma_min > sigma_max:
        raise ValueError("sigma_min must be <= sigma_max.")
    if n_distinct < 1:
        raise ValueError("n_distinct must be positive.")
    if n_distinct == 1:
        return np.asarray([sigma_min], dtype=float)
    return np.geomspace(sigma_min, sigma_max, n_distinct).astype(float)


def select_sigma_set(
    n_distinct: int,
    *,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    sigma_candidates: Sequence[float] | None = None,
    method: str = "logspace",
) -> np.ndarray:
    """Select N sigma values from a range or measured candidate list."""

    if n_distinct < 1:
        raise ValueError("n_distinct must be positive.")
    method = method.lower()
    candidates = None
    if sigma_candidates is not None:
        candidates = np.asarray(sigma_candidates, dtype=float).reshape(-1)
        candidates = candidates[np.isfinite(candidates) & (candidates > 0)]
        if len(candidates) == 0:
            raise ValueError("sigma_candidates must contain at least one positive finite value.")

    if candidates is not None and method in {"quantile", "empirical"}:
        if n_distinct == 1:
            return np.asarray([float(np.min(candidates))])
        quantiles = np.linspace(0.0, 1.0, n_distinct)
        return np.quantile(candidates, quantiles).astype(float)

    if candidates is not None and method in {"logspace", "range"}:
        range_min = float(np.min(candidates)) if sigma_min is None else float(sigma_min)
        range_max = float(np.max(candidates)) if sigma_max is None else float(sigma_max)
        return log_spaced_sigmas(range_min, range_max, n_distinct)

    if method not in {"logspace", "range", "quantile", "empirical"}:
        raise ValueError("method must be 'logspace' or 'quantile'.")
    if sigma_min is None or sigma_max is None:
        raise ValueError("sigma_min and sigma_max are required when sigma_candidates is not provided.")
    return log_spaced_sigmas(float(sigma_min), float(sigma_max), n_distinct)


def replicate_centers_and_sigmas(
    cluster_centers: np.ndarray,
    sigma_set: Sequence[float],
    total_centers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evenly replicate N centers/sigmas to C_total kernels as described in SI."""

    centers = np.asarray(cluster_centers, dtype=float)
    sigmas = np.asarray(sigma_set, dtype=float).reshape(-1)
    n_distinct = len(sigmas)
    if centers.ndim != 2:
        raise ValueError("cluster_centers must be a 2D array.")
    if len(centers) != n_distinct:
        raise ValueError("cluster_centers and sigma_set must have the same length.")
    if total_centers < n_distinct:
        raise ValueError("total_centers must be >= number of distinct sigmas.")

    base_count = total_centers // n_distinct
    extra_count = total_centers % n_distinct

    centers_list: list[np.ndarray] = []
    sigma_list: list[float] = []
    for k in range(n_distinct):
        count = base_count + (1 if k < extra_count else 0)
        centers_list.extend([centers[k]] * count)
        sigma_list.extend([float(sigmas[k])] * count)

    return np.asarray(centers_list, dtype=float), np.asarray(sigma_list, dtype=float)


def build_design_matrix(
    x: np.ndarray,
    centers: np.ndarray,
    sigma_list: np.ndarray,
    chunk_size: int = 20_000,
    include_bias: bool = True,
) -> np.ndarray:
    """Build the Gaussian RBF design matrix with chunked distance evaluation."""

    x = np.asarray(x, dtype=float)
    centers = np.asarray(centers, dtype=float)
    sigma_list = np.asarray(sigma_list, dtype=float).reshape(-1)
    if x.ndim != 2 or centers.ndim != 2:
        raise ValueError("x and centers must be 2D arrays.")
    if x.shape[1] != centers.shape[1]:
        raise ValueError("x and centers must have the same feature dimension.")
    if len(centers) != len(sigma_list):
        raise ValueError("centers and sigma_list must have the same length.")
    if np.any(sigma_list <= 0):
        raise ValueError("all sigma values must be positive.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    n_samples = x.shape[0]
    n_centers = centers.shape[0]
    n_columns = n_centers + (1 if include_bias else 0)
    phi = np.empty((n_samples, n_columns), dtype=float)
    center_norm = np.sum(centers * centers, axis=1)[None, :]
    denom = 2.0 * sigma_list[None, :] ** 2

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        block = x[start:end]
        distances = np.sum(block * block, axis=1)[:, None] + center_norm - 2.0 * block @ centers.T
        np.maximum(distances, 0.0, out=distances)
        distances /= -denom
        np.exp(distances, out=distances)
        phi[start:end, :n_centers] = distances

    if include_bias:
        phi[:, -1] = 1.0
    return phi


def fit_predict_rbf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_distinct: int,
    total_centers: int,
    sigma_min: float,
    sigma_max: float,
    *,
    sigma_candidates: Sequence[float] | None = None,
    sigma_selection: str = "logspace",
    cluster_method: str = "kmeans",
    random_state: int = 0,
    n_init: int = 10,
    max_iter: int = 300,
    batch_size: int = 4096,
    chunk_size: int = 20_000,
    output_names: Sequence[str] | None = None,
) -> RBFRunResult:
    """Train and evaluate one SI-style multi-Gaussian RBF model."""

    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_test.ndim == 1:
        y_test = y_test[:, None]
    if n_distinct > len(x_train):
        raise ValueError("n_distinct cannot exceed number of training samples.")

    sigma_set = select_sigma_set(
        n_distinct,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_candidates=sigma_candidates,
        method=sigma_selection,
    )
    cluster_method = cluster_method.lower()
    if cluster_method == "kmeans":
        clusterer = KMeans(
            n_clusters=n_distinct,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
    elif cluster_method in {"minibatch", "mini_batch"}:
        clusterer = MiniBatchKMeans(
            n_clusters=n_distinct,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            batch_size=batch_size,
        )
    else:
        raise ValueError("cluster_method must be 'kmeans' or 'minibatch'.")

    clusterer.fit(x_train)
    centers, sigma_list = replicate_centers_and_sigmas(
        clusterer.cluster_centers_, sigma_set, total_centers
    )
    phi_train = build_design_matrix(x_train, centers, sigma_list, chunk_size=chunk_size)
    weights = np.linalg.lstsq(phi_train, y_train, rcond=None)[0]
    phi_test = build_design_matrix(x_test, centers, sigma_list, chunk_size=chunk_size)
    y_pred = phi_test @ weights
    metrics = regression_metrics(y_test, y_pred, output_names=output_names)

    return RBFRunResult(
        n_distinct=n_distinct,
        sigma_set=sigma_set,
        centers=centers,
        sigma_list=sigma_list,
        weights=weights,
        y_pred=y_pred,
        metrics=metrics,
    )


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    output_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute MSE, MAE, and R2 for each output plus output averages."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    n_outputs = y_true.shape[1]
    if output_names is None:
        output_names = [f"y{i}" for i in range(n_outputs)]
    if len(output_names) != n_outputs:
        raise ValueError("output_names length must match number of outputs.")

    metrics: dict[str, float] = {}
    mse_values: list[float] = []
    mae_values: list[float] = []
    r2_values: list[float] = []
    for idx, name in enumerate(output_names):
        mse = float(mean_squared_error(y_true[:, idx], y_pred[:, idx]))
        mae = float(mean_absolute_error(y_true[:, idx], y_pred[:, idx]))
        r2 = float(r2_score(y_true[:, idx], y_pred[:, idx]))
        metrics[f"MSE_{name}"] = mse
        metrics[f"MAE_{name}"] = mae
        metrics[f"R2_{name}"] = r2
        mse_values.append(mse)
        mae_values.append(mae)
        r2_values.append(r2)

    metrics["MSE_mean"] = float(np.mean(mse_values))
    metrics["MAE_mean"] = float(np.mean(mae_values))
    metrics["R2_mean"] = float(np.mean(r2_values))
    return metrics


def metrics_row(n_distinct: int, metrics: dict[str, float]) -> dict[str, float | int]:
    """Return a CSV-friendly metrics row."""

    return {"N": int(n_distinct), **metrics}


def ensure_n_values(values: Iterable[int], total_centers: int) -> list[int]:
    """Validate and sort unique N values."""

    parsed = sorted({int(value) for value in values})
    if not parsed:
        raise ValueError("At least one N value is required.")
    if parsed[0] < 1:
        raise ValueError("N values must be positive.")
    if parsed[-1] > total_centers:
        raise ValueError("N values cannot exceed total_centers.")
    return parsed
