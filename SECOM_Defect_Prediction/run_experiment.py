"""Run the first SECOM defect-prediction benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from secom_defect.data import load_secom
from secom_defect.modeling import ExperimentConfig, run_model_comparison, train_holdout_feature_report
from secom_defect.reporting import generate_report_figures, save_dataset_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SECOM defect-prediction model comparison.")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing SECOM files.")
    parser.add_argument("--output-dir", default="reports/baseline", help="Directory for report artifacts.")
    parser.add_argument("--download", action="store_true", help="Download UCI SECOM files if missing.")
    parser.add_argument("--imputer", choices=["median", "mean", "knn", "forward_fill"], default="median")
    parser.add_argument("--missing-threshold", type=float, default=0.5)
    parser.add_argument("--variance-threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--resampling", choices=["none", "smote"], default="none")
    parser.add_argument("--splits", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_secom(PROJECT_ROOT / args.data_dir, download=args.download)
    config = ExperimentConfig(
        missing_threshold=args.missing_threshold,
        imputer=args.imputer,
        variance_threshold=args.variance_threshold,
        top_k=args.top_k,
        resampling=args.resampling,
    )

    output_dir = PROJECT_ROOT / args.output_dir
    save_dataset_profile(data.features, data.labels, output_dir)
    summary = run_model_comparison(data.features, data.labels, output_dir, config, args.splits)
    train_holdout_feature_report(data.features, data.labels, output_dir, config)
    generate_report_figures(output_dir)

    print("\nSECOM model comparison complete.")
    print(summary[["model", "recall_fail", "precision_fail", "f1_fail", "pr_auc", "balanced_accuracy"]])
    print(f"\nReports written to: {output_dir}")


if __name__ == "__main__":
    main()
