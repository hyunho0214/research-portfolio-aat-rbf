"""Panama electricity-demand Figure 4 style plots using src forecasting logic."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from forecasting import prepare_forecasting_data, run_rbf_sigma_sweep  # noqa: E402
from sigma_utils import load_sigma_data, parse_n_values  # noqa: E402


def str_to_bool(value: str) -> bool:
    """Parse shell-friendly boolean strings."""
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser mirroring the forecasting entry point."""
    parser = argparse.ArgumentParser(description="Create Panama Figure 4 style forecasting plots.")
    parser.add_argument("--data", default="continuous dataset.csv", help="Path to the electricity CSV.")
    parser.add_argument("--sigma-csv", default="sigma.csv", help="CSV containing experimental AAT sigma values.")
    parser.add_argument("--sigma-column", default="sigma", help="Column name containing sigma values.")
    parser.add_argument("--sigma-values", default=None, help="Comma-separated sigma values.")
    parser.add_argument("--output", default="figure_panama.png", help="Saved PNG path.")
    parser.add_argument("--pickle-output", default=None, help="Saved matplotlib figure pickle path.")
    parser.add_argument("--load-pkl", default=None, help="Open a saved matplotlib figure pickle without retraining.")
    parser.add_argument("--n-values", default=None, help="MSE/R2 N sweep values. Default: 1..80.")
    parser.add_argument("--plot-n-values", default="1,3,10,80", help="Prediction overlay/subplot N values.")
    parser.add_argument("--window-length", type=int, default=10, help="Input window length.")
    parser.add_argument("--c-total", type=int, default=200, help="Total number of RBF kernels.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for k-means and sigma assignment.")
    parser.add_argument("--shuffle-sigmas", type=str_to_bool, default=True, help="Shuffle sigma assignments.")
    parser.add_argument(
        "--center-mode",
        choices=["distinct", "replicated"],
        default="replicated",
        help="replicated clusters into N centers and expands to C_total; distinct fixes C_total unique centers.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=0.0, help="Optional ridge alpha for output weights.")
    parser.add_argument("--train-start-year", type=int, default=2016, help="First training year.")
    parser.add_argument("--train-end-year", type=int, default=2019, help="Last training year.")
    parser.add_argument("--test-year", type=int, default=2020, help="Calendar year used for testing.")
    parser.add_argument("--days-to-show", type=int, default=15, help="Days shown in small prediction panels.")
    parser.add_argument("--show", type=str_to_bool, default=True, help="Open matplotlib UI after saving PNG.")
    return parser


def default_pickle_path(output_path: str | Path) -> Path:
    """Use the PNG output stem as the default pickle path."""
    return Path(output_path).with_suffix(".pkl")


def save_figure_pickle(fig: plt.Figure, pickle_path: str | Path) -> Path:
    """Persist a matplotlib Figure so the UI can be reopened without retraining."""
    pickle_path = Path(pickle_path)
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with pickle_path.open("wb") as file:
        pickle.dump(fig, file)
    return pickle_path


def load_figure_pickle(pickle_path: str | Path) -> plt.Figure:
    """Load a previously pickled matplotlib Figure."""
    pickle_path = Path(pickle_path)
    with pickle_path.open("rb") as file:
        fig = pickle.load(file)
    if not isinstance(fig, plt.Figure):
        raise TypeError(f"Pickle does not contain a matplotlib Figure: {pickle_path}")
    return fig


def run_panama_figure_analysis(args: argparse.Namespace) -> dict[str, object]:
    """Run the same sweep logic used by src/main_forecasting.py."""
    n_values = parse_n_values(args.n_values)
    plot_n_values = parse_n_values(args.plot_n_values)
    n_values_for_run = sorted(set(n_values).union(plot_n_values))

    print("[1/4] Sigma loading started.")
    sigma_data = load_sigma_data(args.sigma_csv, args.sigma_column, args.sigma_values)
    print(f"[1/4] Sigma loading complete. Count={len(sigma_data)}")

    print("[2/4] Preprocessing started.")
    prepared = prepare_forecasting_data(
        args.data,
        args.window_length,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_year=args.test_year,
    )
    print("[2/4] Preprocessing complete.")
    print(f"[3/4] Sweep started. N values={n_values_for_run}")

    def report_progress(row: dict[str, float]) -> None:
        """Print one concise status line after each N is trained and tested."""
        print(
            f"[3/4] N={int(row['N'])} train/test complete | "
            f"MSE={row['mse']:.6g} | R2={row['r2']:.6g}"
        )

    sweep = run_rbf_sigma_sweep(
        prepared,
        sigma_data,
        n_values_for_run,
        c_total=args.c_total,
        seed=args.seed,
        shuffle_sigmas=args.shuffle_sigmas,
        center_mode=args.center_mode,
        ridge_alpha=args.ridge_alpha,
        progress_callback=report_progress,
    )
    sweep["config"] = {
        "center_mode": args.center_mode,
        "sigma_mode": "direct",
        "ridge_alpha": args.ridge_alpha,
        "seed": args.seed,
        "c_total": args.c_total,
        "n_values": n_values_for_run,
    }
    print("[3/4] Sweep complete.")
    return sweep


def plot_panama_figure(
    sweep: dict[str, object],
    output_path: str | Path,
    plot_n_values: list[int],
    days_to_show: int,
) -> plt.Figure:
    """Plot Figure-4-style panels using predictions from the shared sweep."""
    metrics_df = pd.DataFrame(sweep["metrics"]).sort_values("N")
    predictions = sweep["predictions"]
    y_true = np.asarray(sweep["y_true"]).reshape(-1)
    test_dates = sweep["test_timestamps"]

    fig = plt.figure(figsize=(16, 14))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

    ax_h = fig.add_subplot(3, 2, 1)
    ax_h.plot(test_dates, y_true, "k-", linewidth=1.5, label="Actual", alpha=0.85)
    for index, n_value in enumerate(plot_n_values):
        if n_value in predictions:
            ax_h.plot(
                test_dates,
                predictions[n_value],
                color=colors[index % len(colors)],
                linewidth=1.0,
                label=f"N={n_value}",
                alpha=0.75,
            )
    ax_h.set_xlabel("Date")
    ax_h.set_ylabel("Electricity Demand")
    ax_h.set_title("(h) Electricity Demand Prediction", fontweight="bold")
    ax_h.legend(loc="upper right")
    ax_h.tick_params(axis="x", rotation=30)
    ax_h.grid(True, alpha=0.3)

    points_to_show = min(days_to_show * 24, len(y_true))
    for index, n_value in enumerate(plot_n_values[:4]):
        ax_i = fig.add_subplot(3, 4, 9 + index)
        days = np.arange(points_to_show) / 24.0
        ax_i.plot(days, y_true[:points_to_show], "k-", linewidth=1.3, label="Actual")
        if n_value in predictions:
            ax_i.plot(
                days,
                predictions[n_value][:points_to_show],
                "--",
                color=colors[index % len(colors)],
                linewidth=1.2,
                label=f"N={n_value}",
            )
        ax_i.set_xlabel("Days")
        ax_i.set_ylabel("Demand")
        ax_i.set_title(f"N={n_value}")
        ax_i.legend(loc="upper right", fontsize=7)
        ax_i.grid(True, alpha=0.3)
        ax_i.set_xlim([0, days_to_show])

    ax_j = fig.add_subplot(3, 2, 3)
    ax_j.semilogy(metrics_df["N"], metrics_df["mse"], "bo-", linewidth=2, markersize=5)
    ax_j.semilogy([1, 80], [9800, 120], "ks", markersize=8, alpha=0.5, label="Paper")
    ax_j.set_xlabel("Number of different Gaussians (N)")
    ax_j.set_ylabel("MSE")
    ax_j.set_title("(j) MSE vs Number of Gaussians", fontweight="bold")
    ax_j.grid(True, alpha=0.3, which="both")
    ax_j.legend()

    ax_k = fig.add_subplot(3, 2, 4)
    ax_k.plot(metrics_df["N"], metrics_df["r2"], "ro-", linewidth=2, markersize=5)
    ax_k.plot([1, 80], [0.41, 0.94], "ks", markersize=8, alpha=0.5, label="Paper")
    ax_k.set_xlabel("Number of different Gaussians (N)")
    ax_k.set_ylabel("R2 Score")
    ax_k.set_title("(k) R2 vs Number of Gaussians", fontweight="bold")
    ax_k.set_ylim([0.0, 1.0])
    ax_k.grid(True, alpha=0.3)
    ax_k.legend()

    config = sweep.get("config", {})
    fig.suptitle(
        "Panama Electricity Demand: Multi-Gaussian RBF Analysis\n"
        f"center={config.get('center_mode')}, sigma={config.get('sigma_mode')}, "
        f"ridge={config.get('ridge_alpha')}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    metrics_df.to_csv(output_path.with_name(output_path.stem + "_metrics.csv"), index=False)
    print(f"Saved PNG: {output_path}")
    print(f"Saved metrics: {output_path.with_name(output_path.stem + '_metrics.csv')}")
    return fig


def main() -> None:
    """Run shared forecasting analysis, save figure, and optionally show UI."""
    args = build_parser().parse_args()
    if args.load_pkl is not None:
        print(f"Loading saved figure pickle: {args.load_pkl}")
        fig = load_figure_pickle(args.load_pkl)
        print("Pickle loaded. Opening matplotlib UI.")
        plt.figure(fig.number)
        plt.show()
        return

    sweep = run_panama_figure_analysis(args)
    plot_n_values = parse_n_values(args.plot_n_values)
    fig = plot_panama_figure(sweep, args.output, plot_n_values, args.days_to_show)
    pickle_path = Path(args.pickle_output) if args.pickle_output else default_pickle_path(args.output)
    saved_pickle = save_figure_pickle(fig, pickle_path)
    print(f"Saved pickle: {saved_pickle}")
    print("[4/4] Figure outputs complete.")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
