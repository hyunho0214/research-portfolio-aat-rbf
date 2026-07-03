"""End-to-end hourly electricity-demand RBF sigma-diversity experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from data_utils import (
    load_electricity_series,
    make_sliding_windows,
    split_series_by_year,
    standardize_train_test,
)
from metrics import regression_metrics
from plotting import plot_forecasting_predictions, plot_metric_sweep
from rbf import (
    RBFNetwork,
    compute_kmeans_centers,
    compute_replicated_centers_and_sigmas,
)
from sigma_utils import (
    assign_sigmas_evenly,
    clean_sigma_data,
    select_log_spaced_sigmas_with_geomean_single,
)


def _window_target_timestamps(frame: pd.DataFrame, window_length: int) -> np.ndarray:
    """Return timestamps aligned to the one-step-ahead targets."""
    return frame["datetime"].iloc[window_length:].to_numpy()


def prepare_forecasting_data(
    data_path: str | Path,
    window_length: int,
    train_start_year: int = 2016,
    train_end_year: int = 2019,
    test_year: int = 2020,
) -> dict[str, object]:
    """Prepare chronological train/test windows for the hourly demand series."""
    frame = load_electricity_series(data_path)
    train_frame, test_frame = split_series_by_year(
        frame,
        train_years=(train_start_year, train_end_year),
        test_year=test_year,
    )

    x_train_raw, y_train_raw = make_sliding_windows(
        train_frame["nat_demand"].to_numpy(),
        window_length=window_length,
        target_width=1,
    )
    x_test_raw, y_test_raw = make_sliding_windows(
        test_frame["nat_demand"].to_numpy(),
        window_length=window_length,
        target_width=1,
    )
    standardized = standardize_train_test(x_train_raw, y_train_raw, x_test_raw, y_test_raw)
    return {
        "standardized": standardized,
        "train_frame": train_frame,
        "test_frame": test_frame,
        "test_timestamps": _window_target_timestamps(test_frame, window_length),
        "y_train_raw": y_train_raw,
        "y_test_raw": y_test_raw,
    }


def run_rbf_sigma_sweep(
    prepared_data: dict[str, object],
    sigma_data: np.ndarray,
    n_values: list[int],
    c_total: int = 200,
    seed: int = 0,
    shuffle_sigmas: bool = True,
    center_mode: str = "replicated",
    ridge_alpha: float = 0.0,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, object]:
    """Run the shared RBF sweep used by CLI experiments and figure scripts."""
    data = prepared_data
    standardized = data["standardized"]
    sigma_clean = clean_sigma_data(sigma_data)

    if center_mode not in {"distinct", "replicated"}:
        raise ValueError("center_mode must be either 'distinct' or 'replicated'.")

    # Distinct mode fixes center capacity; replicated mode recomputes centers for each N.
    centers_fixed = None
    if center_mode == "distinct":
        centers_fixed = compute_kmeans_centers(standardized.x_train, c_total=c_total, seed=seed)
    metrics_rows: list[dict[str, float]] = []
    predictions: dict[int, np.ndarray] = {}
    best: dict[str, object] | None = None

    for n_sigmas in n_values:
        if n_sigmas > c_total:
            raise ValueError("Each N value must be less than or equal to c_total.")
        sigma_set_exp = select_log_spaced_sigmas_with_geomean_single(sigma_clean, n_sigmas)
        sigma_set_model = sigma_set_exp

        if center_mode == "replicated":
            centers, sigma_per_center = compute_replicated_centers_and_sigmas(
                standardized.x_train,
                n_clusters=n_sigmas,
                c_total=c_total,
                sigma_set=sigma_set_model,
                seed=seed,
            )
        else:
            assert centers_fixed is not None
            centers = centers_fixed
            sigma_per_center = assign_sigmas_evenly(
                c_total,
                sigma_set_model,
                shuffle=shuffle_sigmas,
                seed=seed + n_sigmas,
            )

        model = RBFNetwork(
            centers=centers,
            sigma_per_center=sigma_per_center,
            ridge_alpha=ridge_alpha,
        )
        model.fit(standardized.x_train, standardized.y_train)

        y_pred_standardized = model.predict(standardized.x_test)
        y_pred = standardized.y_scaler.inverse_transform(y_pred_standardized).reshape(-1)
        y_true = np.asarray(data["y_test_raw"]).reshape(-1)
        row = {"N": int(n_sigmas), **regression_metrics(y_true, y_pred)}
        metrics_rows.append(row)
        predictions[int(n_sigmas)] = y_pred
        if progress_callback is not None:
            progress_callback(row)

        if best is None or row["mse"] < best["metrics"]["mse"]:
            best = {
                "N": int(n_sigmas),
                "metrics": row,
                "model": model,
                "sigma_set_exp": sigma_set_exp,
                "sigma_set_model": sigma_set_model,
                "sigma_per_center": sigma_per_center,
                "centers": centers,
                "y_pred": y_pred,
                "y_true": y_true,
            }

    assert best is not None
    metrics_frame = pd.DataFrame(metrics_rows)
    return {
        "metrics": metrics_frame,
        "predictions": predictions,
        "best": best,
        "y_true": np.asarray(data["y_test_raw"]).reshape(-1),
        "test_timestamps": data["test_timestamps"],
        "sigma_clean": sigma_clean,
    }


def run_forecasting_experiment(
    data_path: str | Path,
    sigma_data: np.ndarray,
    output_dir: str | Path,
    n_values: list[int],
    window_length: int = 10,
    c_total: int = 200,
    seed: int = 0,
    shuffle_sigmas: bool = True,
    center_mode: str = "replicated",
    ridge_alpha: float = 0.0,
    train_start_year: int = 2016,
    train_end_year: int = 2019,
    test_year: int = 2020,
) -> dict[str, object]:
    """Run the RBF sweep and save metrics, predictions, config, and plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_forecasting_data(
        data_path,
        window_length,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        test_year=test_year,
    )
    sweep = run_rbf_sigma_sweep(
        data,
        sigma_data,
        n_values,
        c_total=c_total,
        seed=seed,
        shuffle_sigmas=shuffle_sigmas,
        center_mode=center_mode,
        ridge_alpha=ridge_alpha,
    )
    metrics_frame = sweep["metrics"]
    best = sweep["best"]
    sigma_clean = sweep["sigma_clean"]
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)

    prediction_frame = pd.DataFrame(
        {
            "datetime": data["test_timestamps"],
            "y_true": best["y_true"],
            "y_pred": best["y_pred"],
        }
    )
    prediction_frame.to_csv(output_dir / "best_predictions.csv", index=False)

    np.savez(
        output_dir / "best_model_arrays.npz",
        centers=best["centers"],
        sigma_set_exp=best["sigma_set_exp"],
        sigma_set_model=best["sigma_set_model"],
        sigma_per_center=best["sigma_per_center"],
        weights=best["model"].weights,
    )

    config = {
        "task": "hourly_electricity_forecasting",
        "data_path": str(data_path),
        "target_column": "nat_demand",
        "window_length": window_length,
        "c_total": c_total,
        "n_values": n_values,
        "best_N": best["N"],
        "seed": seed,
        "shuffle_sigmas": shuffle_sigmas,
        "center_mode": center_mode,
        "sigma_mode": "direct",
        "ridge_alpha": ridge_alpha,
        "sigma_min": float(np.min(sigma_clean)),
        "sigma_max": float(np.max(sigma_clean)),
        "sigma_scaling": "direct experimental sigma values",
        "train_years": [train_start_year, train_end_year],
        "test_year": test_year,
        "test_note": "The available 2020 period ends at the final row in the CSV.",
        "standardization": "X and y scalers fitted on train split only.",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plot_metric_sweep(metrics_frame, output_dir / "metric_mse.png", metric="mse")
    plot_metric_sweep(metrics_frame, output_dir / "metric_r2.png", metric="r2")
    plot_forecasting_predictions(
        data["test_timestamps"],
        best["y_true"],
        best["y_pred"],
        output_dir / "best_prediction_overlay.png",
    )
    return {"metrics": metrics_frame, "predictions": sweep["predictions"], "best": best, "config": config}
