"""End-to-end hourly electricity-demand RBF sigma-diversity experiment."""

from __future__ import annotations

import json
from pathlib import Path

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
from rbf import RBFNetwork, compute_kmeans_centers, median_nearest_center_distance
from sigma_utils import (
    assign_sigmas_evenly,
    clean_sigma_data,
    rescale_sigmas_by_reference_distance,
    select_log_spaced_sigmas,
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


def run_forecasting_experiment(
    data_path: str | Path,
    sigma_data: np.ndarray,
    output_dir: str | Path,
    n_values: list[int],
    window_length: int = 10,
    c_total: int = 200,
    seed: int = 0,
    shuffle_sigmas: bool = True,
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
    standardized = data["standardized"]
    sigma_clean = clean_sigma_data(sigma_data)

    # Centers are fixed across N so the sweep isolates width diversity only.
    centers = compute_kmeans_centers(standardized.x_train, c_total=c_total, seed=seed)
    distance_reference = median_nearest_center_distance(standardized.x_train, centers)
    metrics_rows: list[dict[str, float]] = []
    best: dict[str, object] | None = None

    for n_sigmas in n_values:
        if n_sigmas > c_total:
            raise ValueError("Each N value must be less than or equal to c_total.")
        sigma_set_exp = select_log_spaced_sigmas(sigma_clean, n_sigmas)
        sigma_set_rel, sigma_set_model = rescale_sigmas_by_reference_distance(
            sigma_set_exp,
            distance_reference,
        )
        sigma_per_center = assign_sigmas_evenly(
            c_total,
            sigma_set_model,
            shuffle=shuffle_sigmas,
            seed=seed + n_sigmas,
        )
        model = RBFNetwork(centers=centers, sigma_per_center=sigma_per_center)
        model.fit(standardized.x_train, standardized.y_train)

        y_pred_scaled = model.predict(standardized.x_test)
        y_pred = standardized.y_scaler.inverse_transform(y_pred_scaled).reshape(-1)
        y_true = np.asarray(data["y_test_raw"]).reshape(-1)
        row = {"N": int(n_sigmas), **regression_metrics(y_true, y_pred)}
        metrics_rows.append(row)

        if best is None or row["mse"] < best["metrics"]["mse"]:
            best = {
                "N": int(n_sigmas),
                "metrics": row,
                "model": model,
                "sigma_set_exp": sigma_set_exp,
                "sigma_set_rel": sigma_set_rel,
                "sigma_set_model": sigma_set_model,
                "sigma_per_center": sigma_per_center,
                "y_pred": y_pred,
                "y_true": y_true,
            }

    assert best is not None
    metrics_frame = pd.DataFrame(metrics_rows)
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
        centers=centers,
        sigma_set_exp=best["sigma_set_exp"],
        sigma_set_rel=best["sigma_set_rel"],
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
        "sigma_min": float(np.min(sigma_clean)),
        "sigma_max": float(np.max(sigma_clean)),
        "sigma_scaling": "experimental geomspace -> divide by selected median -> multiply by train median nearest-center distance",
        "sigma_distance_reference": distance_reference,
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
    return {"metrics": metrics_frame, "best": best, "config": config}
