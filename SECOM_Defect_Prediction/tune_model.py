"""Hyperparameter tuning entry point for SECOM defect prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from secom_defect.data import load_secom
from secom_defect.modeling import ExperimentConfig, build_pipeline, make_models
from secom_defect.reporting import save_dataset_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune one SECOM defect-prediction model.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="reports/tuning")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--model",
        choices=["logistic_balanced", "random_forest_balanced", "svm_rbf_balanced", "xgboost_weighted"],
        default="logistic_balanced",
    )
    parser.add_argument("--imputer", choices=["median", "mean", "knn", "forward_fill"], default="median")
    parser.add_argument("--missing-threshold", type=float, default=0.5)
    parser.add_argument("--variance-threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--resampling", choices=["none", "smote"], default="smote")
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def param_distributions(model_name: str) -> dict[str, object]:
    if model_name == "logistic_balanced":
        return {
            "model__C": loguniform(1e-3, 1e2),
        }
    if model_name == "random_forest_balanced":
        return {
            "model__n_estimators": randint(200, 800),
            "model__max_depth": [None, 4, 6, 10, 14],
            "model__min_samples_leaf": randint(1, 8),
            "model__max_features": ["sqrt", "log2", 0.3, 0.5],
        }
    if model_name == "svm_rbf_balanced":
        return {
            "model__C": loguniform(1e-2, 1e2),
            "model__gamma": loguniform(1e-4, 1e0),
        }
    if model_name == "xgboost_weighted":
        return {
            "model__estimator__n_estimators": randint(100, 500),
            "model__estimator__max_depth": randint(2, 6),
            "model__estimator__learning_rate": loguniform(1e-2, 2e-1),
            "model__estimator__subsample": uniform(0.65, 0.35),
            "model__estimator__colsample_bytree": uniform(0.55, 0.45),
            "model__estimator__scale_pos_weight": [8, 10, 12, 14, 16, 20],
        }
    raise ValueError(f"Unsupported model: {model_name}")


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

    scorer = make_scorer(f1_score, pos_label=1, zero_division=0)
    cv = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.random_state)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions(args.model),
        n_iter=args.n_iter,
        scoring=scorer,
        cv=cv,
        random_state=args.random_state,
        n_jobs=1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results.to_csv(output_dir / "tuning_cv_results.csv", index=False)

    y_pred = search.best_estimator_.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.Series(report).to_json(output_dir / "holdout_classification_report.json", indent=2)

    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
    pd.DataFrame(
        [
            {
                "model": args.model,
                "true_pass_pred_pass": int(cm[0, 0]),
                "true_pass_pred_fail": int(cm[0, 1]),
                "true_fail_pred_pass": int(cm[1, 0]),
                "true_fail_pred_fail": int(cm[1, 1]),
            }
        ]
    ).to_csv(output_dir / "holdout_confusion_matrix.csv", index=False)

    best_params = {
        "model": args.model,
        "best_cv_f1_fail": float(search.best_score_),
        "best_params": search.best_params_,
    }
    with open(output_dir / "best_params.json", "w", encoding="utf-8") as handle:
        json.dump(best_params, handle, indent=2)

    print(f"Best CV fail F1: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    print(f"Reports written to: {output_dir}")


if __name__ == "__main__":
    main()
