"""Run the Figure 4g-k Panama electricity-demand RBF simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.panama import load_excel_demand_series
from src.plotting import save_metric_plots, save_panama_15_day_segments, save_panama_full_test
from src.rbf import create_sliding_window_1d, ensure_n_values, fit_predict_rbf, metrics_row, parse_n_values, regression_metrics
from src.transfer_sigma import load_sigma_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", type=Path, default=Path("Panama") / "train_dataframes.xlsx")
    parser.add_argument("--sheet", default=None, help="Excel sheet name. Defaults to the longest DEMAND sheet.")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--total-centers", type=int, default=200)
    parser.add_argument("--n-values", default="1:80")
    parser.add_argument("--sigma-min", type=float, default=None)
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument("--sigma-file", type=Path, default=None, help="Excel file with extracted sigma values.")
    parser.add_argument("--sigma-column", default="sigma", help="Column name to read from --sigma-file.")
    parser.add_argument(
        "--sigma-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to sigma values loaded from --sigma-file.",
    )
    parser.add_argument(
        "--sigma-selection",
        choices=["logspace", "quantile"],
        default="logspace",
        help="How to select N sigma values from --sigma-file. logspace uses min/max; quantile uses empirical quantiles.",
    )
    parser.add_argument("--cluster-method", choices=["kmeans", "minibatch"], default="kmeans")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--selected-n", default="1,3,10,80")
    parser.add_argument("--include-mlp", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("output") / "panama")
    return parser.parse_args()


def load_series(args: argparse.Namespace) -> pd.DataFrame:
    return load_excel_demand_series(args.excel, sheet_name=args.sheet, start_date=args.start_date)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    (args.output_dir / "predictions").mkdir(exist_ok=True)

    df = load_series(args)
    values = df["demand"].to_numpy(dtype=float)
    split_idx = int(args.split_ratio * len(values))
    train_values = values[:split_idx]
    test_values = values[split_idx:]

    x_train_raw, y_train_raw = create_sliding_window_1d(train_values, args.window)
    x_test_raw, y_test_raw = create_sliding_window_1d(test_values, args.window)
    scaler_x = StandardScaler().fit(x_train_raw)
    scaler_y = StandardScaler().fit(y_train_raw)
    x_train = scaler_x.transform(x_train_raw)
    x_test = scaler_x.transform(x_test_raw)
    y_train = scaler_y.transform(y_train_raw)

    sigma_candidates = None
    if args.sigma_file is not None:
        sigma_candidates = load_sigma_values(args.sigma_file, column=args.sigma_column) * args.sigma_scale
        sigma_min = float(np.min(sigma_candidates)) if args.sigma_min is None else float(args.sigma_min)
        sigma_max = float(np.max(sigma_candidates)) if args.sigma_max is None else float(args.sigma_max)
    else:
        sigma_min = 0.3 if args.sigma_min is None else float(args.sigma_min)
        sigma_max = 20.0 if args.sigma_max is None else float(args.sigma_max)

    n_values = ensure_n_values(parse_n_values(args.n_values), args.total_centers)
    selected_n = ensure_n_values(parse_n_values(args.selected_n), args.total_centers)
    missing_selected = [n for n in selected_n if n not in n_values]
    if missing_selected:
        raise ValueError(f"selected N values must also be in --n-values: {missing_selected}")

    print("[Panama] Data summary")
    print("  source: excel")
    print(f"  source detail: {df.attrs.get('sheet_name', '')}")
    print(f"  samples: {len(values)} total, {len(train_values)} train, {len(test_values)} test")
    print(f"  window: {args.window}")
    print("[Panama] Sigma configuration")
    if args.sigma_file is not None:
        raw_sigmas = load_sigma_values(args.sigma_file, column=args.sigma_column)
        print(f"  sigma file: {args.sigma_file}")
        print(f"  sigma column: {args.sigma_column}")
        print(f"  raw sigma count: {len(raw_sigmas)}")
        print(f"  raw sigma min/max: {np.min(raw_sigmas):.6g} / {np.max(raw_sigmas):.6g}")
        print(f"  sigma scale: {args.sigma_scale:.6g}")
        print(f"  scaled sigma min/max: {np.min(sigma_candidates):.6g} / {np.max(sigma_candidates):.6g}")
    else:
        print("  sigma file: none")
    print(f"  effective sigma min/max used: {sigma_min:.6g} / {sigma_max:.6g}")
    print(f"  sigma selection: {args.sigma_selection}")
    print(f"  N values: {n_values}")

    metrics_rows: list[dict[str, float]] = []
    selected_predictions: dict[int, np.ndarray] = {}
    for n in n_values:
        print(f"[Panama] Training RBF N={n}")
        result = fit_predict_rbf(
            x_train,
            y_train,
            x_test,
            scaler_y.transform(y_test_raw),
            n,
            args.total_centers,
            sigma_min,
            sigma_max,
            sigma_candidates=sigma_candidates,
            sigma_selection=args.sigma_selection,
            cluster_method=args.cluster_method,
            random_state=args.random_state,
            n_init=args.n_init,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            output_names=["demand"],
        )
        y_pred_orig = scaler_y.inverse_transform(result.y_pred)
        metrics = regression_metrics(y_test_raw, y_pred_orig, output_names=["demand"])
        row = metrics_row(n, metrics)
        metrics_rows.append(row)
        if n in selected_n:
            selected_predictions[n] = y_pred_orig
            pred_df = pd.DataFrame(
                {
                    "sample": np.arange(len(y_pred_orig)),
                    "day": np.arange(len(y_pred_orig), dtype=float) / 24.0,
                    "true_demand": y_test_raw.reshape(-1),
                    "predicted_demand": y_pred_orig.reshape(-1),
                }
            )
            pred_df.to_csv(args.output_dir / "predictions" / f"panama_predictions_N{n:03d}.csv", index=False)

    pd.DataFrame(metrics_rows).to_csv(args.output_dir / "panama_rbf_metrics.csv", index=False)

    if args.include_mlp:
        print("[Panama] Training optional MLP baseline")
        mlp = MLPRegressor(
            hidden_layer_sizes=(200,),
            activation="logistic",
            solver="sgd",
            learning_rate_init=0.01,
            momentum=0.9,
            max_iter=2000,
            random_state=args.random_state,
        )
        mlp.fit(x_train, y_train.ravel())
        mlp_pred = scaler_y.inverse_transform(mlp.predict(x_test).reshape(-1, 1))
        mlp_metrics = regression_metrics(y_test_raw, mlp_pred, output_names=["demand"])
        pd.DataFrame([mlp_metrics]).to_csv(args.output_dir / "panama_mlp_metrics.csv", index=False)

    save_panama_full_test(
        y_test_raw,
        selected_predictions,
        args.output_dir / "figures" / "panama_full_test.png",
        selected_n=selected_n,
    )
    save_panama_15_day_segments(
        y_test_raw,
        selected_predictions,
        args.output_dir / "figures" / "panama_15_day_segments.png",
        selected_n=selected_n,
    )
    save_metric_plots(metrics_rows, args.output_dir / "figures" / "panama")

    metadata = {
        "source": "excel",
        "source_detail": df.attrs.get("sheet_name", ""),
        "start_date": args.start_date,
        "samples": int(len(values)),
        "train_samples": int(len(train_values)),
        "test_samples": int(len(test_values)),
        "window": args.window,
        "split_ratio": args.split_ratio,
        "total_centers": args.total_centers,
        "n_values": n_values,
        "selected_n": selected_n,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "sigma_file": str(args.sigma_file) if args.sigma_file else None,
        "sigma_column": args.sigma_column,
        "sigma_scale": args.sigma_scale,
        "sigma_selection": args.sigma_selection,
        "sigma_candidate_count": int(len(sigma_candidates)) if sigma_candidates is not None else 0,
        "cluster_method": args.cluster_method,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[Panama] Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
