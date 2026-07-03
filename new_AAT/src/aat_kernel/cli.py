"""Command-line interfaces for AAT hardware-aware kernel experiments."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from .calibration import DEFAULT_METADATA_COLUMNS, save_calibration_outputs
from .hardware_kernel import (
    VALID_KERNEL_MODES,
    GateEquivalentEncoder,
    HardwareAwareKernelBank,
    HardwareAwareKernelRegressor,
    HardwareKernelLibrary,
)


def calibrate_curves_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an AAT hardware kernel library from raw transfer curves.")
    parser.add_argument("--curves", required=True, help="Long-format raw transfer-curve CSV.")
    parser.add_argument("--out", required=True, help="Output directory for calibration artifacts.")
    parser.add_argument(
        "--metadata-cols",
        nargs="*",
        default=None,
        help="Optional metadata columns to preserve. Defaults to known AAT metadata columns when present.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip curve-fit example plot generation.")
    parser.add_argument("--max-plot-curves", type=int, default=16, help="Maximum curves shown in the fit plot.")
    args = parser.parse_args(argv)

    paths = save_calibration_outputs(
        curves_path=args.curves,
        out_dir=args.out,
        metadata_columns=args.metadata_cols,
        make_plots=not args.no_plot,
        max_plot_curves=args.max_plot_curves,
    )
    print("Saved calibration artifacts:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


def run_hardware_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run hardware-aware AAT kernel regression variants.")
    parser.add_argument("--config", required=True, help="YAML or JSON experiment config.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    outputs = run_hardware_experiment(config)
    print("Saved hardware experiment artifacts:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("YAML config support requires PyYAML. Install with: py -m pip install -e .") from exc
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("Experiment config must be a mapping.")
    return loaded


def run_hardware_experiment(config: dict[str, Any]) -> dict[str, Path]:
    seed = int(config.get("seed", 0))
    np.random.seed(seed)

    output_dir = Path(config.get("output_dir", "outputs/hardware_experiment"))
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config.get("data", {})
    if "supervised_csv" not in data_cfg:
        raise ValueError("Config requires data.supervised_csv.")
    feature_columns = _as_list(data_cfg.get("feature_columns"))
    target_columns = _as_list(data_cfg.get("target_columns"))
    if not feature_columns or not target_columns:
        raise ValueError("Config requires non-empty data.feature_columns and data.target_columns.")

    supervised = pd.read_csv(data_cfg["supervised_csv"])
    time_col = data_cfg.get("time_col") or config.get("split", {}).get("time_col")
    if time_col:
        if time_col not in supervised.columns:
            raise ValueError(f"time_col={time_col!r} is not present in supervised CSV.")
        supervised = supervised.sort_values(time_col).reset_index(drop=True)
    supervised = supervised.dropna(subset=feature_columns + target_columns).reset_index(drop=True)

    split_cfg = config.get("split", {})
    train_fraction = float(split_cfg.get("train_fraction", 0.8))
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("split.train_fraction must be between 0 and 1.")
    split_idx = int(np.floor(len(supervised) * train_fraction))
    if split_idx <= 0 or split_idx >= len(supervised):
        raise ValueError("Train/test split produced an empty train or test set.")

    X = supervised[feature_columns].to_numpy(dtype=float)
    y = supervised[target_columns].to_numpy(dtype=float)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    standardize_targets = bool(data_cfg.get("standardize_targets", True))
    if standardize_targets:
        y_mean = np.mean(y_train, axis=0)
        y_scale = np.std(y_train, axis=0)
        y_scale[y_scale <= 1e-12] = 1.0
        y_train_fit = (y_train - y_mean) / y_scale
    else:
        y_mean = np.zeros(y_train.shape[1], dtype=float)
        y_scale = np.ones(y_train.shape[1], dtype=float)
        y_train_fit = y_train

    kernel_cfg = config.get("kernel_library", {})
    if "path" not in kernel_cfg:
        raise ValueError("Config requires kernel_library.path.")
    library = HardwareKernelLibrary.from_csv(kernel_cfg["path"], curves_path=kernel_cfg.get("curves_path"))
    filters = kernel_cfg.get("filter") or {}
    if filters:
        library = library.filter(**filters)

    variants = config.get("models", {}).get("variants", list(VALID_KERNEL_MODES))
    variants = _as_list(variants)
    unknown = [variant for variant in variants if variant not in VALID_KERNEL_MODES]
    if unknown:
        raise ValueError(f"Unknown model variants: {unknown}. Expected values: {VALID_KERNEL_MODES}")

    ridge_alpha = float(config.get("readout", {}).get("ridge_alpha", 0.0))
    encoder_mode = config.get("encoder", {}).get("mode", "pca_sigmoid")

    metrics_rows: list[dict[str, Any]] = []
    outputs: dict[str, Path] = {}
    for variant in variants:
        encoder = GateEquivalentEncoder(library.vg_min, library.vg_max, mode=encoder_mode)
        bank = HardwareAwareKernelBank(library=library, encoder=encoder, mode=variant)
        regressor = HardwareAwareKernelRegressor(bank, ridge_alpha=ridge_alpha)
        regressor.fit(X_train, y_train_fit)
        pred_fit = regressor.predict(X_test)
        if pred_fit.ndim == 1:
            pred_fit = pred_fit[:, None]
        y_pred = pred_fit * y_scale + y_mean

        predictions = _prediction_frame(supervised.iloc[split_idx:], target_columns, y_test, y_pred, time_col)
        pred_path = output_dir / f"hardware_kernel_predictions_{variant}.csv"
        predictions.to_csv(pred_path, index=False)
        outputs[f"predictions_{variant}"] = pred_path

        plot_path = output_dir / f"hardware_kernel_predictions_{variant}.png"
        plot_predictions(predictions, target_columns, variant, plot_path)
        outputs[f"prediction_plot_{variant}"] = plot_path

        metrics_rows.extend(
            _metric_rows(
                variant=variant,
                target_columns=target_columns,
                y_true=y_test,
                y_pred=y_pred,
                n_train=len(X_train),
                n_test=len(X_test),
                n_kernels=library.n_kernels,
                feature_rank=regressor.train_feature_rank_,
                ridge_alpha=ridge_alpha,
            )
        )

    metrics = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "hardware_kernel_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    outputs["metrics"] = metrics_path

    config_path = output_dir / "hardware_kernel_config.json"
    serializable_config = _json_ready(config)
    serializable_config["resolved"] = {
        "n_rows": int(len(supervised)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_kernels": int(library.n_kernels),
        "vg_min": float(library.vg_min),
        "vg_max": float(library.vg_max),
        "target_standardization": bool(standardize_targets),
        "random_seed": seed,
    }
    config_path.write_text(json.dumps(serializable_config, indent=2), encoding="utf-8")
    outputs["config"] = config_path

    return outputs


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _prediction_frame(
    source_rows: pd.DataFrame,
    target_columns: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_col: str | None,
) -> pd.DataFrame:
    frame = pd.DataFrame({"row_index": source_rows.index.to_numpy()})
    if time_col:
        frame[time_col] = source_rows[time_col].to_numpy()
    for idx, target in enumerate(target_columns):
        frame[f"{target}_true"] = y_true[:, idx]
        frame[f"{target}_pred"] = y_pred[:, idx]
        frame[f"{target}_error"] = y_pred[:, idx] - y_true[:, idx]
    return frame


def _metric_rows(
    variant: str,
    target_columns: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_train: int,
    n_test: int,
    n_kernels: int,
    feature_rank: int,
    ridge_alpha: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, target in enumerate(target_columns):
        truth = y_true[:, idx]
        pred = y_pred[:, idx]
        error = pred - truth
        mse = float(np.mean(error**2))
        mae = float(np.mean(np.abs(error)))
        ss_res = float(np.sum(error**2))
        ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
        r2 = float("nan") if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
        rows.append(
            {
                "variant": variant,
                "target": target,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "n_train": n_train,
                "n_test": n_test,
                "n_kernels": n_kernels,
                "feature_rank": feature_rank,
                "ridge_alpha": ridge_alpha,
            }
        )
    return rows


def plot_predictions(predictions: pd.DataFrame, target_columns: list[str], variant: str, out_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    n_targets = len(target_columns)
    fig, axes = plt.subplots(n_targets, 1, figsize=(8.0, max(2.8, 2.6 * n_targets)), squeeze=False)
    x_axis = np.arange(len(predictions))
    for ax, target in zip(axes[:, 0], target_columns):
        ax.plot(x_axis, predictions[f"{target}_true"], label="true", linewidth=1.5)
        ax.plot(x_axis, predictions[f"{target}_pred"], label="pred", linewidth=1.2)
        ax.set_title(f"{variant}: {target}")
        ax.set_xlabel("test row")
        ax.set_ylabel(target)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AAT hardware-aware kernel tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate_parser = subparsers.add_parser("calibrate-curves")
    calibrate_parser.add_argument("--curves", required=True)
    calibrate_parser.add_argument("--out", required=True)
    calibrate_parser.add_argument("--metadata-cols", nargs="*", default=None)
    calibrate_parser.add_argument("--no-plot", action="store_true")
    calibrate_parser.add_argument("--max-plot-curves", type=int, default=16)

    run_parser = subparsers.add_parser("run-hardware")
    run_parser.add_argument("--config", required=True)

    args = parser.parse_args(argv)
    if args.command == "calibrate-curves":
        forwarded = ["--curves", args.curves, "--out", args.out, "--max-plot-curves", str(args.max_plot_curves)]
        if args.metadata_cols:
            forwarded.extend(["--metadata-cols", *args.metadata_cols])
        if args.no_plot:
            forwarded.append("--no-plot")
        return calibrate_curves_cli(forwarded)
    return run_hardware_cli(["--config", args.config])


if __name__ == "__main__":
    raise SystemExit(main())
