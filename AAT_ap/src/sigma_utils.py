"""Utilities for handling experimentally measured AAT Gaussian widths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_N_VALUES = list(range(1, 81))


def parse_n_values(value: str | None) -> list[int]:
    """Parse comma-separated sigma-diversity values used in the sweep."""
    if value is None or value.strip() == "":
        return DEFAULT_N_VALUES.copy()
    n_values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not n_values:
        raise ValueError("At least one N value must be provided.")
    if any(n <= 0 for n in n_values):
        raise ValueError("All N values must be positive integers.")
    return n_values


def parse_float_list(value: str | None) -> np.ndarray | None:
    """Parse comma-separated floats; return None when no value was supplied."""
    if value is None or value.strip() == "":
        return None
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        return None
    return np.asarray(values, dtype=float)


def load_sigma_data(
    sigma_csv: str | None = None,
    sigma_column: str | None = None,
    sigma_values: str | None = None,
) -> np.ndarray:
    """Load sigma values from either a CSV column or a comma-separated list.

    The project intentionally has no built-in placeholder sigma values because
    the RBF widths must remain tied to the AAT experimental measurements.
    """
    inline_values = parse_float_list(sigma_values)
    if inline_values is not None:
        if sigma_csv is not None:
            raise ValueError("Use either --sigma-values or --sigma-csv, not both.")
        return inline_values

    if sigma_csv is None:
        raise ValueError(
            "AAT sigma values are required. Provide --sigma-csv with --sigma-column "
            "or provide --sigma-values such as 0.1,0.2,0.5."
        )

    csv_path = Path(sigma_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Sigma CSV was not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if sigma_column is None:
        if frame.shape[1] != 1:
            raise ValueError(
                "Provide --sigma-column because the sigma CSV has more than one column."
            )
        sigma_column = str(frame.columns[0])
    if sigma_column not in frame.columns:
        raise ValueError(f"Sigma column {sigma_column!r} was not found in {csv_path}.")
    return frame[sigma_column].to_numpy(dtype=float)


def clean_sigma_data(sigma_data: np.ndarray) -> np.ndarray:
    """Remove invalid, NaN, infinite, and nonpositive sigma measurements."""
    sigma = np.asarray(sigma_data, dtype=float).reshape(-1)
    valid = sigma[np.isfinite(sigma) & (sigma > 0.0)]
    if valid.size == 0:
        raise ValueError("No valid positive sigma values remain after cleaning.")
    return valid


def get_sigma_range(sigma_data: np.ndarray) -> tuple[float, float]:
    """Return the cleaned experimental sigma range."""
    valid = clean_sigma_data(sigma_data)
    sigma_min = float(np.min(valid))
    sigma_max = float(np.max(valid))
    if sigma_min <= 0.0 or sigma_max <= 0.0:
        raise ValueError("Sigma range must be strictly positive.")
    return sigma_min, sigma_max


def select_log_spaced_sigmas(sigma_data: np.ndarray, n_sigmas: int) -> np.ndarray:
    """Select N distinct widths by log spacing across the experimental range."""
    if n_sigmas <= 0:
        raise ValueError("n_sigmas must be positive.")
    sigma_min, sigma_max = get_sigma_range(sigma_data)
    return np.geomspace(sigma_min, sigma_max, n_sigmas)


def rescale_sigmas_by_reference_distance(
    sigma_set_exp: np.ndarray,
    reference_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map experimental sigma ratios into standardized-input distance units.

    The experimental values define relative width diversity. Dividing by their
    median makes the central selected width equal to one, and multiplying by a
    train-only distance reference sets the RBF width scale in model space.
    """
    sigma_set_exp = np.asarray(sigma_set_exp, dtype=float).reshape(-1)
    if sigma_set_exp.size == 0:
        raise ValueError("sigma_set_exp must contain at least one value.")
    if np.any(sigma_set_exp <= 0.0):
        raise ValueError("All experimental sigma values must be positive.")
    if not np.isfinite(reference_distance) or reference_distance <= 0.0:
        raise ValueError("reference_distance must be a positive finite value.")

    sigma_relative = sigma_set_exp / np.median(sigma_set_exp)
    sigma_model = sigma_relative * reference_distance
    return sigma_relative.astype(float), sigma_model.astype(float)


def assign_sigmas_evenly(
    c_total: int,
    sigma_set: np.ndarray,
    shuffle: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Assign selected sigma values evenly across a fixed center budget."""
    sigma_set = np.asarray(sigma_set, dtype=float).reshape(-1)
    if c_total <= 0:
        raise ValueError("c_total must be positive.")
    if sigma_set.size == 0:
        raise ValueError("sigma_set must contain at least one value.")
    if sigma_set.size > c_total:
        raise ValueError("The number of selected sigmas cannot exceed c_total.")

    base_count = c_total // sigma_set.size
    extra = c_total % sigma_set.size
    counts = np.full(sigma_set.size, base_count, dtype=int)
    counts[:extra] += 1

    # Even reuse isolates sigma diversity while keeping the kernel count fixed.
    sigma_per_center = np.repeat(sigma_set, counts)
    if shuffle:
        rng = np.random.default_rng(seed)
        sigma_per_center = rng.permutation(sigma_per_center)
    return sigma_per_center.astype(float)
