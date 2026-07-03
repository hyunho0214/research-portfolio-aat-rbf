"""Run the Figure 4c-f Duffing oscillator RBF simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.duffing import integrate_duffing_rk4
from src.plotting import save_duffing_phase_space, save_duffing_time_domain, save_metric_plots
from src.rbf import (
    create_sliding_window_multivariate,
    ensure_n_values,
    fit_predict_rbf,
    metrics_row,
    parse_n_values,
    regression_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--transient-steps", type=int, default=100_000)
    parser.add_argument("--post-steps", type=int, default=200_000)
    parser.add_argument("--x0", type=float, default=0.1)
    parser.add_argument("--v0", type=float, default=0.0)
    parser.add_argument("--damping", type=float, default=0.2)
    parser.add_argument("--drive", type=float, default=0.3)
    parser.add_argument("--omega", type=float, default=1.2)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--split-ratio", type=float, default=0.5)
    parser.add_argument("--total-centers", type=int, default=300)
    parser.add_argument("--n-values", default="1:80")
    parser.add_argument("--selected-n", default="3,80")
    parser.add_argument("--sigma-min", type=float, default=3.0)
    parser.add_argument("--sigma-max", type=float, default=20.0)
    parser.add_argument("--cluster-method", choices=["kmeans", "minibatch"], default="kmeans")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--output-dir", type=Path, default=Path("output") / "duffing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    (args.output_dir / "predictions").mkdir(exist_ok=True)

    print("[Duffing] Integrating oscillator")
    times, states = integrate_duffing_rk4(
        dt=args.dt,
        transient_steps=args.transient_steps,
        post_steps=args.post_steps,
        x0=args.x0,
        v0=args.v0,
        damping=args.damping,
        drive=args.drive,
        omega=args.omega,
    )
    x_raw, y_raw = create_sliding_window_multivariate(states, args.window)

    # The SI pseudocode standardizes the full generated Duffing dataset before splitting.
    scaler_x = StandardScaler().fit(x_raw)
    scaler_y = StandardScaler().fit(y_raw)
    x_all = scaler_x.transform(x_raw)
    y_all = scaler_y.transform(y_raw)
    split_idx = int(args.split_ratio * len(x_all))
    x_train = x_all[:split_idx]
    y_train = y_all[:split_idx]
    x_test = x_all[split_idx:]
    y_test_scaled = y_all[split_idx:]
    y_test_orig = scaler_y.inverse_transform(y_test_scaled)
    test_times = times[args.window + split_idx : args.window + split_idx + len(y_test_orig)]

    n_values = ensure_n_values(parse_n_values(args.n_values), args.total_centers)
    selected_n = ensure_n_values(parse_n_values(args.selected_n), args.total_centers)
    missing_selected = [n for n in selected_n if n not in n_values]
    if missing_selected:
        raise ValueError(f"selected N values must also be in --n-values: {missing_selected}")

    metrics_rows: list[dict[str, float]] = []
    selected_predictions: dict[int, np.ndarray] = {}
    for n in n_values:
        print(f"[Duffing] Training RBF N={n}")
        result = fit_predict_rbf(
            x_train,
            y_train,
            x_test,
            y_test_scaled,
            n,
            args.total_centers,
            args.sigma_min,
            args.sigma_max,
            cluster_method=args.cluster_method,
            random_state=args.random_state,
            n_init=args.n_init,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            output_names=["x", "v"],
        )
        y_pred_orig = scaler_y.inverse_transform(result.y_pred)
        metrics = regression_metrics(y_test_orig, y_pred_orig, output_names=["x", "v"])
        metrics_rows.append(metrics_row(n, metrics))
        if n in selected_n:
            selected_predictions[n] = y_pred_orig
            pred_df = pd.DataFrame(
                {
                    "time": test_times,
                    "true_x": y_test_orig[:, 0],
                    "true_v": y_test_orig[:, 1],
                    "predicted_x": y_pred_orig[:, 0],
                    "predicted_v": y_pred_orig[:, 1],
                }
            )
            pred_df.to_csv(args.output_dir / "predictions" / f"duffing_predictions_N{n:03d}.csv", index=False)

    pd.DataFrame(metrics_rows).to_csv(args.output_dir / "duffing_rbf_metrics.csv", index=False)
    save_duffing_phase_space(
        y_test_orig,
        selected_predictions,
        args.output_dir / "figures" / "duffing_phase_space.png",
        selected_n=selected_n,
    )
    save_duffing_time_domain(
        y_test_orig,
        selected_predictions,
        args.output_dir / "figures" / "duffing_time_domain.png",
        selected_n=selected_n,
    )
    save_metric_plots(metrics_rows, args.output_dir / "figures" / "duffing")

    metadata = {
        "dt": args.dt,
        "transient_steps": args.transient_steps,
        "post_steps": args.post_steps,
        "x0": args.x0,
        "v0": args.v0,
        "damping": args.damping,
        "drive": args.drive,
        "omega": args.omega,
        "window": args.window,
        "split_ratio": args.split_ratio,
        "total_centers": args.total_centers,
        "n_values": n_values,
        "selected_n": selected_n,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "cluster_method": args.cluster_method,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[Duffing] Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
