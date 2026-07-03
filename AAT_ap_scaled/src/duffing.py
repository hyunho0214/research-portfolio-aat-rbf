"""Duffing oscillator reconstruction with the same multi-Gaussian RBF method."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import chronological_fraction_split, make_sliding_windows, standardize_train_test
from metrics import duffing_metrics
from plotting import plot_duffing_phase, plot_duffing_time, plot_metric_sweep
from rbf import RBFNetwork, compute_kmeans_centers, median_nearest_center_distance
from sigma_utils import (
    assign_sigmas_evenly,
    clean_sigma_data,
    rescale_sigmas_by_reference_distance,
    select_log_spaced_sigmas,
)


def duffing_derivative(
    t: float,
    state: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    omega: float,
) -> np.ndarray:
    """Evaluate x_dot=v and v_dot=-beta*v+x-alpha*x^3+gamma*cos(omega*t)."""
    x_value, v_value = state
    x_dot = v_value
    v_dot = -beta * v_value + x_value - alpha * x_value**3 + gamma * np.cos(omega * t)
    return np.asarray([x_dot, v_dot], dtype=float)


def simulate_duffing(
    alpha: float = 1.0,
    beta: float = 0.2,
    gamma: float = 0.3,
    omega: float = 1.2,
    dt: float = 0.01,
    n_steps: int = 20000,
    x0: float = 0.1,
    v0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Duffing dynamics with a fixed-step fourth-order RK method."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if n_steps <= 1:
        raise ValueError("n_steps must be greater than 1.")

    times = np.arange(n_steps, dtype=float) * dt
    states = np.zeros((n_steps, 2), dtype=float)
    states[0] = [x0, v0]
    for index in range(n_steps - 1):
        t = times[index]
        y = states[index]
        # RK4 is deterministic and accurate enough for a defensible simulation baseline.
        k1 = duffing_derivative(t, y, alpha, beta, gamma, omega)
        k2 = duffing_derivative(t + 0.5 * dt, y + 0.5 * dt * k1, alpha, beta, gamma, omega)
        k3 = duffing_derivative(t + 0.5 * dt, y + 0.5 * dt * k2, alpha, beta, gamma, omega)
        k4 = duffing_derivative(t + dt, y + dt * k3, alpha, beta, gamma, omega)
        states[index + 1] = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return times, states


def prepare_duffing_data(
    window_length: int,
    train_fraction: float,
    transient_steps: int,
    alpha: float,
    beta: float,
    gamma: float,
    omega: float,
    dt: float,
    n_steps: int,
    x0: float,
    v0: float,
) -> dict[str, object]:
    """Generate Duffing samples, remove transient, split chronologically, standardize."""
    times, states = simulate_duffing(alpha, beta, gamma, omega, dt, n_steps, x0, v0)
    if transient_steps < 0 or transient_steps >= len(states) - window_length - 1:
        raise ValueError("transient_steps leaves too few samples for windowing.")

    kept_times = times[transient_steps:]
    kept_states = states[transient_steps:]
    train_states, test_states = chronological_fraction_split(kept_states, train_fraction)
    train_times, test_times = chronological_fraction_split(kept_times, train_fraction)

    x_train_raw, y_train_raw = make_sliding_windows(train_states, window_length, target_width=2)
    x_test_raw, y_test_raw = make_sliding_windows(test_states, window_length, target_width=2)
    standardized = standardize_train_test(x_train_raw, y_train_raw, x_test_raw, y_test_raw)
    return {
        "standardized": standardized,
        "train_times": train_times,
        "test_times": test_times[window_length:],
        "y_train_raw": y_train_raw,
        "y_test_raw": y_test_raw,
        "states_after_transient": kept_states,
    }


def run_duffing_experiment(
    sigma_data: np.ndarray,
    output_dir: str | Path,
    n_values: list[int],
    window_length: int = 10,
    c_total: int = 300,
    seed: int = 0,
    shuffle_sigmas: bool = True,
    train_fraction: float = 0.5,
    transient_steps: int = 2000,
    alpha: float = 1.0,
    beta: float = 0.2,
    gamma: float = 0.3,
    omega: float = 1.2,
    dt: float = 0.01,
    n_steps: int = 20000,
    x0: float = 0.1,
    v0: float = 0.0,
) -> dict[str, object]:
    """Run the Duffing reconstruction sweep and save diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_duffing_data(
        window_length,
        train_fraction,
        transient_steps,
        alpha,
        beta,
        gamma,
        omega,
        dt,
        n_steps,
        x0,
        v0,
    )
    standardized = data["standardized"]
    sigma_clean = clean_sigma_data(sigma_data)
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
        y_pred = standardized.y_scaler.inverse_transform(y_pred_scaled)
        y_true = np.asarray(data["y_test_raw"])
        row = {"N": int(n_sigmas), **duffing_metrics(y_true, y_pred)}
        metrics_rows.append(row)
        ranking_score = row["mse_x"] + row["mse_v"]
        if best is None or ranking_score < best["ranking_score"]:
            best = {
                "N": int(n_sigmas),
                "metrics": row,
                "ranking_score": ranking_score,
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
            "time": data["test_times"],
            "x_true": best["y_true"][:, 0],
            "v_true": best["y_true"][:, 1],
            "x_pred": best["y_pred"][:, 0],
            "v_pred": best["y_pred"][:, 1],
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
        "task": "duffing_reconstruction",
        "window_length": window_length,
        "c_total": c_total,
        "n_values": n_values,
        "best_N": best["N"],
        "seed": seed,
        "shuffle_sigmas": shuffle_sigmas,
        "train_fraction": train_fraction,
        "transient_steps": transient_steps,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "omega": omega,
        "dt": dt,
        "n_steps": n_steps,
        "x0": x0,
        "v0": v0,
        "sigma_min": float(np.min(sigma_clean)),
        "sigma_max": float(np.max(sigma_clean)),
        "sigma_mode": "scaled",
        "sigma_scaling": "experimental geomspace -> divide by selected median -> multiply by train median nearest-center distance",
        "sigma_distance_reference": distance_reference,
        "standardization": "X and y scalers fitted on train split only.",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plot_metric_sweep(metrics_frame, output_dir / "metric_mse_x.png", metric="mse_x")
    plot_metric_sweep(metrics_frame, output_dir / "metric_mse_v.png", metric="mse_v")
    plot_duffing_time(data["test_times"], best["y_true"], best["y_pred"], output_dir / "duffing_time.png")
    plot_duffing_phase(best["y_true"], best["y_pred"], output_dir / "duffing_phase.png")
    return {"metrics": metrics_frame, "best": best, "config": config}
