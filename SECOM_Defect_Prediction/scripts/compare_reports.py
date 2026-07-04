"""Compare multiple SECOM experiment report folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SECOM experiment metrics.")
    parser.add_argument("--reports", nargs="+", required=True, help="Report directories to compare.")
    parser.add_argument("--names", nargs="+", required=True, help="Short names for each report directory.")
    parser.add_argument("--output-dir", default="reports/comparison", help="Directory for comparison outputs.")
    return parser.parse_args()


def load_metrics(report_dir: Path, experiment_name: str) -> pd.DataFrame:
    metrics_path = report_dir / "metrics_summary.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    metrics.insert(0, "experiment", experiment_name)
    return metrics


def main() -> None:
    args = parse_args()
    if len(args.reports) != len(args.names):
        raise ValueError("--reports and --names must have the same length.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = pd.concat(
        [load_metrics(Path(report), name) for report, name in zip(args.reports, args.names)],
        ignore_index=True,
    )
    all_metrics.to_csv(output_dir / "experiment_model_metrics.csv", index=False)

    best_rows = (
        all_metrics.sort_values(["experiment", "f1_fail", "recall_fail"], ascending=[True, False, False])
        .groupby("experiment", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_rows.to_csv(output_dir / "experiment_best_summary.csv", index=False)
    plot_best_experiments(best_rows, output_dir / "preprocessing_comparison.png")

    print(best_rows[["experiment", "model", "recall_fail", "precision_fail", "f1_fail", "pr_auc"]])
    print(f"Wrote comparison outputs to: {output_dir}")


def plot_best_experiments(best_rows: pd.DataFrame, output_png: Path) -> None:
    best_rows = best_rows.sort_values("f1_fail")
    labels = best_rows["experiment"] + "\n" + best_rows["model"].str.replace("_", " ", regex=False)
    metrics = ["recall_fail", "f1_fail", "pr_auc"]
    colors = ["#e45756", "#4c78a8", "#72b7b2"]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = range(len(best_rows))
    width = 0.23
    for idx, metric in enumerate(metrics):
        offsets = [value + (idx - 1) * width for value in x]
        ax.bar(offsets, best_rows[metric], width=width, label=metric, color=colors[idx])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Best Model by Preprocessing Strategy")
    ax.set_ylim(0, max(0.75, best_rows[metrics].to_numpy().max() * 1.15))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
