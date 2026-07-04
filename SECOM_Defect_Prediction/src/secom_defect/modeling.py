"""Leakage-safe SECOM preprocessing, model comparison, and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:  # pragma: no cover - handled by runtime configuration
    SMOTE = None
    ImbPipeline = Pipeline

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - xgboost is optional
    XGBClassifier = None


class BinaryLabelMappingClassifier(BaseEstimator, ClassifierMixin):
    """Map SECOM labels -1/1 to 0/1 for estimators that require zero-based labels."""

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BinaryLabelMappingClassifier":
        self.estimator_ = clone(self.estimator)
        y_binary = (pd.Series(y).to_numpy() == 1).astype(int)
        self.estimator_.fit(X, y_binary)
        self.classes_ = np.array([-1, 1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.estimator_.predict(X)
        return np.where(pred == 1, 1, -1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict_proba(X)


class MissingRateDropper(BaseEstimator, TransformerMixin):
    """Drop columns whose missing rate exceeds the configured threshold."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MissingRateDropper":
        missing_rate = X.isna().mean()
        self.keep_columns_ = missing_rate[missing_rate <= self.threshold].index.to_list()
        self.dropped_columns_ = missing_rate[missing_rate > self.threshold].index.to_list()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.keep_columns_]


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """Impute missing values while preserving SECOM sensor column names."""

    def __init__(self, kind: str = "median"):
        self.kind = kind

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DataFrameImputer":
        self.columns_ = list(X.columns)
        if self.kind == "median":
            self.imputer_ = SimpleImputer(strategy="median")
        elif self.kind == "mean":
            self.imputer_ = SimpleImputer(strategy="mean")
        elif self.kind == "knn":
            self.imputer_ = KNNImputer(n_neighbors=5, weights="distance")
        elif self.kind == "forward_fill":
            self.fill_values_ = X.ffill().bfill().median(numeric_only=True)
            return self
        else:
            raise ValueError(f"Unknown imputer: {self.kind}")
        self.imputer_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.kind == "forward_fill":
            transformed = X.ffill().bfill().fillna(self.fill_values_)
            return transformed.loc[:, self.columns_]
        transformed = self.imputer_.transform(X)
        return pd.DataFrame(transformed, columns=self.columns_, index=X.index)


class DataFrameVarianceThreshold(BaseEstimator, TransformerMixin):
    """VarianceThreshold wrapper that preserves selected feature names."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DataFrameVarianceThreshold":
        self.selector_ = VarianceThreshold(threshold=self.threshold)
        self.selector_.fit(X)
        self.feature_names_in_ = np.asarray(X.columns)
        self.selected_features_ = self.feature_names_in_[self.selector_.get_support()].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = self.selector_.transform(X)
        return pd.DataFrame(transformed, columns=self.selected_features_, index=X.index)


class RandomForestTopKSelector(BaseEstimator, TransformerMixin):
    """Select top-k features using Random Forest feature importance."""

    def __init__(self, top_k: int | None = 50, random_state: int = 42):
        self.top_k = top_k
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestTopKSelector":
        if self.top_k is None:
            self.selected_features_ = list(X.columns)
            self.feature_importance_ = pd.DataFrame(
                {"feature": X.columns, "importance": np.nan}
            )
            return self

        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X, y)
        importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
        self.feature_importance_ = importance
        self.selected_features_ = importance.head(self.top_k)["feature"].to_list()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.selected_features_]


@dataclass(frozen=True)
class ExperimentConfig:
    missing_threshold: float = 0.5
    imputer: str = "median"
    variance_threshold: float = 0.0
    top_k: int | None = 50
    resampling: str = "none"
    random_state: int = 42


def make_imputer(kind: str) -> Any:
    return DataFrameImputer(kind)


def make_models(random_state: int = 42) -> dict[str, Any]:
    models: dict[str, Any] = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_balanced": LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=random_state,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "svm_rbf_balanced": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            C=3.0,
            gamma="scale",
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }
    if XGBClassifier is not None:
        models["xgboost_weighted"] = BinaryLabelMappingClassifier(
            XGBClassifier(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                scale_pos_weight=14,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )
        )
    return models


def build_pipeline(model: Any, config: ExperimentConfig) -> Any:
    steps: list[tuple[str, Any]] = [
        ("missing_drop", MissingRateDropper(config.missing_threshold)),
        ("imputer", make_imputer(config.imputer)),
        ("variance", DataFrameVarianceThreshold(config.variance_threshold)),
        ("feature_select", RandomForestTopKSelector(config.top_k, config.random_state)),
        ("scale", StandardScaler()),
    ]

    if config.resampling == "smote":
        if SMOTE is None:
            raise ImportError("Install imbalanced-learn to use SMOTE.")
        steps.append(("smote", SMOTE(random_state=config.random_state, k_neighbors=5)))
        pipeline_cls = ImbPipeline
    elif config.resampling == "none":
        pipeline_cls = Pipeline
    else:
        raise ValueError(f"Unknown resampling strategy: {config.resampling}")

    steps.append(("model", model))
    return pipeline_cls(steps)


def _positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, list(model.classes_).index(1)]
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        return np.asarray(score)
    return model.predict(X)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    metrics = {
        "recall_fail": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_fail": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_fail": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    try:
        metrics["pr_auc"] = average_precision_score((y_true == 1).astype(int), y_score)
    except ValueError:
        metrics["pr_auc"] = np.nan
    try:
        metrics["roc_auc"] = roc_auc_score((y_true == 1).astype(int), y_score)
    except ValueError:
        metrics["roc_auc"] = np.nan
    return metrics


def run_model_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str | Path,
    config: ExperimentConfig,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Run leakage-safe CV model comparison and save report artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    models = make_models(config.random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    rows = []
    reports = {}
    confusion_rows = []

    for name, model in models.items():
        pipeline = build_pipeline(model, config)
        y_pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict", n_jobs=None)

        scoring_pipeline = build_pipeline(model, config)
        y_score = cross_val_predict(
            scoring_pipeline,
            X,
            y,
            cv=cv,
            method="predict_proba" if hasattr(model, "predict_proba") else "decision_function",
            n_jobs=None,
        )
        if y_score.ndim == 2:
            y_score = y_score[:, 1]

        metrics = evaluate_predictions(y, y_pred, y_score)
        metrics["model"] = name
        rows.append(metrics)
        reports[name] = classification_report(y, y_pred, output_dict=True, zero_division=0)

        cm = confusion_matrix(y, y_pred, labels=[-1, 1])
        confusion_rows.append(
            {
                "model": name,
                "true_pass_pred_pass": int(cm[0, 0]),
                "true_pass_pred_fail": int(cm[0, 1]),
                "true_fail_pred_pass": int(cm[1, 0]),
                "true_fail_pred_fail": int(cm[1, 1]),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["f1_fail", "recall_fail"], ascending=False)
    summary.to_csv(output_path / "metrics_summary.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output_path / "confusion_matrices.csv", index=False)
    pd.Series(reports).to_json(output_path / "classification_reports.json", indent=2)

    return summary


def train_holdout_feature_report(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str | Path,
    config: ExperimentConfig,
) -> None:
    """Fit one holdout pipeline and save selected-feature diagnostics."""

    output_path = Path(output_dir)
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=config.random_state,
    )
    pipeline = build_pipeline(make_models(config.random_state)["random_forest_balanced"], config)
    pipeline.fit(X_train, y_train)

    selector = pipeline.named_steps["feature_select"]
    pd.DataFrame({"feature": selector.selected_features_}).to_csv(
        output_path / "selected_features.csv",
        index=False,
    )
    if hasattr(selector, "feature_importance_"):
        selector.feature_importance_.to_csv(output_path / "feature_importance.csv", index=False)
