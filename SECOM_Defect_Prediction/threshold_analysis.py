"""Analyze fail-detection threshold tradeoffs for a selected SECOM model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from secom_defect.data import load_secom
from secom_defect.modeling import ExperimentConfig, build_pipeline, make_models
from secom_defect.reporting import save_dataset_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze threshold tradeoffs for SECOM fail detection.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="reports/threshold_analysis")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model", default="xgboost_weighted")
    parser.add_argument("--imputer", choices=["median", "mean", "knn", "forward_fill"], default="median")
    parser.add_argument("--missing-threshold", type=float, default=0.5)
    parser.add_argument("--variance-threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--resampling", choices=["none", "smote"], default="smote")
    parser.add_argument("--target-recall", type=float, default=0.70)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_secom(PROJECT_ROOT / args.data_dir, download=args.download)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dataset_profile(data.features, data.labels, output_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        data.features,
        data.labels,
        test_size=0.2,
        stratify=data.labels,
        random_state=args.random_state,
    )
    config = ExperimentConfig(
        missing_threshold=args.missing_threshold,
        imputer=args.imputer,
        variance_threshold=args.variance_threshold,
        top_k=args.top_k,
        resampling=args.resampling,
        random_state=args.random_state,
    )
    model = make_models(args.random_state)[args.model]
    pipeline = build_pipeline(model, config)
    pipeline.fit(X_train, y_train)

    if not hasattr(pipeline, "predict_proba"):
        raise ValueError(f"Model pipeline for {args.model} does not expose predict_proba.")
    fail_score = pipeline.predict_proba(X_test)[:, 1]

    rows = []
    y_binary = (y_test.to_numpy() == 1).astype(int)
    for threshold in np.round(np.arange(0.05, 0.96, 0.05), 2):
        pred = np.where(fail_score >= threshold, 1, -1)
        cm = confusion_matrix(y_test, pred, labels=[-1, 1])
        rows.append(
            {
                "threshold": threshold,
                "recall_fail": recall_score(y_test, pred, pos_label=1, zero_division=0),
                "precision_fail": precision_score(y_test, pred, pos_label=1, zero_division=0),
                "f1_fail": f1_score(y_test, pred, pos_label=1, zero_division=0),
                "false_alarm_count": int(cm[0, 1]),
                "missed_fail_count": int(cm[1, 0]),
                "detected_fail_count": int(cm[1, 1]),
                "predicted_fail_count": int((pred == 1).sum()),
            }
        )

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(output_dir / "threshold_tradeoff.csv", index=False)
    recommended = choose_threshold(threshold_df, args.target_recall)
    with open(output_dir / "recommended_threshold.json", "w", encoding="utf-8") as handle:
        json.dump(recommended, handle, indent=2)
    plot_threshold_tradeoff(threshold_df, args.target_recall, output_dir / "threshold_tradeoff.png")

    print(pd.DataFrame([recommended]))
    print(f"Reports written to: {output_dir}")


def choose_threshold(threshold_df: pd.DataFrame, target_recall: float) -> dict[str, float]:
    feasible = threshold_df[threshold_df["recall_fail"] >= target_recall]
    if feasible.empty:
        selected = threshold_df.sort_values(["recall_fail", "f1_fail"], ascending=False).iloc[0]
        reason = "highest available recall; target recall was not reached"
    else:
        selected = feasible.sort_values(["f1_fail", "precision_fail"], ascending=False).iloc[0]
        reason = "highest F1 while meeting target recall"
    result = selected.to_dict()
    result["selection_reason"] = reason
    result["target_recall"] = target_recall
    return result


def plot_threshold_tradeoff(threshold_df: pd.DataFrame, target_recall: float, output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(threshold_df["threshold"], threshold_df["recall_fail"], marker="o", label="Fail recall")
    ax.plot(threshold_df["threshold"], threshold_df["precision_fail"], marker="o", label="Fail precision")
    ax.plot(threshold_df["threshold"], threshold_df["f1_fail"], marker="o", label="Fail F1")
    ax.axhline(target_recall, color="#e45756", linestyle="--", linewidth=1.2, label="Target recall")
    ax.set_title("Fail-Detection Threshold Tradeoff")
    ax.set_xlabel("Fail probability threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
