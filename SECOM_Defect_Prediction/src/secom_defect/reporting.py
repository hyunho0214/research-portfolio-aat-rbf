"""Report-table and figure generation for SECOM experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_dataset_profile(X: pd.DataFrame, y: pd.Series, output_dir: str | Path) -> None:
    """Save class-distribution and missingness summaries."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    counts = y.value_counts().sort_index()
    profile = pd.DataFrame(
        {
            "label": counts.index,
            "meaning": ["pass" if label == -1 else "fail" for label in counts.index],
            "count": counts.values,
            "fraction": counts.values / len(y),
        }
    )
    profile.to_csv(output_path / "class_distribution.csv", index=False)

    missingness = X.isna().mean().rename("missing_rate").reset_index()
    missingness = missingness.rename(columns={"index": "feature"}).sort_values(
        "missing_rate",
        ascending=False,
    )
    missingness.to_csv(output_path / "missingness_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    labels = [f"{row.meaning}\n({row.label})" for row in profile.itertuples()]
    ax.bar(labels, profile["count"], color=["#4c78a8", "#e45756"])
    ax.set_title("SECOM Class Distribution")
    ax.set_ylabel("Samples")
    for idx, row in profile.iterrows():
        ax.text(idx, row["count"], f"{row['fraction']:.1%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path / "class_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(missingness["missing_rate"], bins=30, color="#72b7b2", edgecolor="white")
    ax.axvline(0.5, color="#e45756", linestyle="--", linewidth=1.4, label="50% drop threshold")
    ax.set_title("Sensor Missing-Value Rate Distribution")
    ax.set_xlabel("Missing rate")
    ax.set_ylabel("Sensor count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "missingness_distribution.png", dpi=180)
    plt.close(fig)


def generate_report_figures(output_dir: str | Path) -> None:
    """Generate standard figures from saved experiment report CSVs."""

    output_path = Path(output_dir)
    metrics_path = output_path / "metrics_summary.csv"
    confusion_path = output_path / "confusion_matrices.csv"
    importance_path = output_path / "feature_importance.csv"

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        _plot_metric_comparison(metrics, output_path / "model_metric_comparison.png")

        if confusion_path.exists() and not metrics.empty:
            confusion = pd.read_csv(confusion_path)
            best_model = metrics.sort_values(["f1_fail", "recall_fail"], ascending=False).iloc[0]["model"]
            _plot_best_confusion(confusion, best_model, output_path / "best_model_confusion_matrix.png")

    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        _plot_feature_importance(importance, output_path / "top_feature_importance.png")


def _plot_metric_comparison(metrics: pd.DataFrame, output_png: Path) -> None:
    ordered = metrics.sort_values("f1_fail", ascending=True)
    model_labels = ordered["model"].str.replace("_", " ", regex=False)
    metric_cols = ["recall_fail", "f1_fail", "pr_auc"]
    colors = ["#e45756", "#4c78a8", "#72b7b2"]

    fig, ax = plt.subplots(figsize=(8.2, max(4.0, 0.48 * len(ordered))))
    y_pos = np.arange(len(ordered))
    height = 0.22
    for offset, metric, color in zip([-height, 0, height], metric_cols, colors):
        ax.barh(y_pos + offset, ordered[metric], height=height, label=metric, color=color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_labels)
    ax.set_xlim(0, max(0.75, ordered[metric_cols].to_numpy().max() * 1.15))
    ax.set_xlabel("Score")
    ax.set_title("SECOM Fail-Class Model Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def _plot_best_confusion(confusion: pd.DataFrame, best_model: str, output_png: Path) -> None:
    row = confusion.loc[confusion["model"] == best_model].iloc[0]
    matrix = np.array(
        [
            [row["true_pass_pred_pass"], row["true_pass_pred_fail"]],
            [row["true_fail_pred_pass"], row["true_fail_pred_fail"]],
        ],
        dtype=int,
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"Best Model Confusion Matrix\n{best_model}")
    ax.set_xticks([0, 1], labels=["Pred pass", "Pred fail"])
    ax.set_yticks([0, 1], labels=["True pass", "True fail"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def _plot_feature_importance(importance: pd.DataFrame, output_png: Path, top_n: int = 20) -> None:
    importance = importance.dropna(subset=["importance"]).head(top_n).sort_values("importance")
    if importance.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.barh(importance["feature"], importance["importance"], color="#59a14f")
    ax.set_title(f"Top {len(importance)} Selected Sensor Importances")
    ax.set_xlabel("Random Forest importance")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
