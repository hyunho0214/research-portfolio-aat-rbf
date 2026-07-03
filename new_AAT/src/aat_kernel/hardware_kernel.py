"""Hardware-aware kernel banks and linear readout models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12
VALID_KERNEL_MODES = ("sigma_only", "a_sigma", "full_gaussian", "direct_curve")


class HardwareKernelLibrary:
    """Container for measured AAT kernel primitives and optional raw curves."""

    required_columns = ("kernel_id", "curve_id", "A_tilde", "mu", "sigma")

    def __init__(self, table: pd.DataFrame, curves: pd.DataFrame | None = None):
        self.table = table.copy().reset_index(drop=True)
        self.curves = None if curves is None else curves.copy().reset_index(drop=True)
        self._validate()

    @classmethod
    def from_csv(cls, path: str | Path, curves_path: str | Path | None = None) -> "HardwareKernelLibrary":
        path = Path(path)
        table = pd.read_csv(path)
        curves = None
        if curves_path is not None:
            curves = pd.read_csv(curves_path)
        else:
            sibling = path.with_name("normalized_transfer_curves.csv")
            if sibling.exists():
                curves = pd.read_csv(sibling)
        return cls(table=table, curves=curves)

    def _validate(self) -> None:
        missing = [col for col in self.required_columns if col not in self.table.columns]
        if missing:
            raise ValueError(f"Missing required kernel-library columns: {missing}")

        for col in ("A_tilde", "mu", "sigma"):
            self.table[col] = pd.to_numeric(self.table[col], errors="coerce")
        self.table = self.table.dropna(subset=["A_tilde", "mu", "sigma"]).reset_index(drop=True)
        self.table = self.table[self.table["sigma"].abs() > EPS].reset_index(drop=True)
        if self.table.empty:
            raise ValueError("Kernel library contains no valid kernels after validation.")

        if self.curves is not None:
            missing_curve_cols = [col for col in ("curve_id", "Vg", "response_norm") if col not in self.curves.columns]
            if missing_curve_cols:
                raise ValueError(f"Missing required normalized-curve columns: {missing_curve_cols}")
            self.curves["Vg"] = pd.to_numeric(self.curves["Vg"], errors="coerce")
            self.curves["response_norm"] = pd.to_numeric(self.curves["response_norm"], errors="coerce")
            self.curves = self.curves.dropna(subset=["curve_id", "Vg", "response_norm"])

    def filter(self, **kwargs: object) -> "HardwareKernelLibrary":
        """Filter the library by exact metadata matches or membership lists."""

        mask = pd.Series(True, index=self.table.index)
        for key, value in kwargs.items():
            if value is None or value == "":
                continue
            if key not in self.table.columns:
                raise ValueError(f"Cannot filter on unknown kernel-library column: {key}")
            if isinstance(value, (list, tuple, set)):
                mask &= self.table[key].isin(list(value))
            else:
                mask &= self.table[key] == value
        filtered_table = self.table.loc[mask].reset_index(drop=True)
        if filtered_table.empty:
            raise ValueError("Kernel library filter removed all kernels.")

        filtered_curves = self.curves
        if filtered_curves is not None:
            curve_ids = set(filtered_table["curve_id"].astype(str))
            filtered_curves = filtered_curves[filtered_curves["curve_id"].astype(str).isin(curve_ids)].reset_index(drop=True)
        return HardwareKernelLibrary(filtered_table, filtered_curves)

    def get_parameters(self, kernel_ids: list[object] | None = None) -> dict[str, np.ndarray]:
        table = self.table
        if kernel_ids is not None:
            table = table[table["kernel_id"].isin(kernel_ids)].reset_index(drop=True)
            if table.empty:
                raise ValueError("No requested kernel_ids were found in the library.")

        return {
            "kernel_id": table["kernel_id"].to_numpy(),
            "curve_id": table["curve_id"].to_numpy(),
            "A_tilde": table["A_tilde"].to_numpy(dtype=float),
            "mu": table["mu"].to_numpy(dtype=float),
            "sigma": np.abs(table["sigma"].to_numpy(dtype=float)),
        }

    @property
    def vg_min(self) -> float:
        if "Vg_min" in self.table.columns:
            return float(pd.to_numeric(self.table["Vg_min"], errors="coerce").min())
        if self.curves is not None and not self.curves.empty:
            return float(self.curves["Vg"].min())
        return float((self.table["mu"] - 4.0 * self.table["sigma"].abs()).min())

    @property
    def vg_max(self) -> float:
        if "Vg_max" in self.table.columns:
            return float(pd.to_numeric(self.table["Vg_max"], errors="coerce").max())
        if self.curves is not None and not self.curves.empty:
            return float(self.curves["Vg"].max())
        return float((self.table["mu"] + 4.0 * self.table["sigma"].abs()).max())

    @property
    def n_kernels(self) -> int:
        return int(len(self.table))


class GateEquivalentEncoder:
    """Map arbitrary model inputs into a scalar gate-voltage coordinate."""

    def __init__(self, vg_min: float, vg_max: float, mode: str = "pca_sigmoid"):
        if mode != "pca_sigmoid":
            raise ValueError("Only mode='pca_sigmoid' is implemented in the first version.")
        if not np.isfinite(vg_min) or not np.isfinite(vg_max) or vg_max <= vg_min:
            raise ValueError("Expected finite vg_min < vg_max.")
        self.vg_min = float(vg_min)
        self.vg_max = float(vg_max)
        self.mode = mode
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "GateEquivalentEncoder":
        X = _as_2d_float(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ <= EPS] = 1.0
        Xs = (X - self.mean_) / self.scale_

        if Xs.shape[1] == 1:
            component = np.array([1.0])
        else:
            _, _, vh = np.linalg.svd(Xs, full_matrices=False)
            component = vh[0].astype(float)
            pivot = int(np.argmax(np.abs(component)))
            if component[pivot] < 0:
                component = -component

        raw_score = Xs @ component
        self.component_ = component
        self.score_mean_ = float(np.mean(raw_score))
        self.score_scale_ = float(np.std(raw_score))
        if self.score_scale_ <= EPS:
            self.score_scale_ = 1.0
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("GateEquivalentEncoder must be fitted before transform().")
        X = _as_2d_float(X)
        Xs = (X - self.mean_) / self.scale_
        score = (Xs @ self.component_ - self.score_mean_) / self.score_scale_
        score = np.clip(score, -50.0, 50.0)
        sigmoid = _stable_sigmoid(score)
        return self.vg_min + (self.vg_max - self.vg_min) * sigmoid

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class HardwareAwareKernelBank:
    """Feature bank using measured AAT transfer-curve parameters."""

    def __init__(
        self,
        library: HardwareKernelLibrary,
        encoder: GateEquivalentEncoder,
        mode: str = "full_gaussian",
        mu_ref: float | None = None,
    ):
        if mode not in VALID_KERNEL_MODES:
            raise ValueError(f"Unknown kernel mode {mode!r}. Expected one of {VALID_KERNEL_MODES}.")
        self.library = library
        self.encoder = encoder
        self.mode = mode
        self.mu_ref = mu_ref

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "HardwareAwareKernelBank":
        self.encoder.fit(X, y)
        self.params_ = self.library.get_parameters()
        if self.mu_ref is None:
            self.mu_ref_ = float(np.median(self.params_["mu"]))
        else:
            self.mu_ref_ = float(self.mu_ref)
        self.is_fitted = True
        return self

    def compute_features(self, X: np.ndarray) -> np.ndarray:
        if not getattr(self, "is_fitted", False):
            raise RuntimeError("HardwareAwareKernelBank must be fitted before compute_features().")

        z = self.encoder.transform(X)
        if self.mode == "direct_curve":
            return self._compute_direct_curve_features(z)

        A_tilde = self.params_["A_tilde"]
        mu = self.params_["mu"]
        sigma = np.maximum(np.abs(self.params_["sigma"]), EPS)

        if self.mode == "sigma_only":
            gain = np.ones_like(A_tilde)
            center = np.full_like(mu, self.mu_ref_, dtype=float)
        elif self.mode == "a_sigma":
            gain = A_tilde
            center = np.full_like(mu, self.mu_ref_, dtype=float)
        else:
            gain = A_tilde
            center = mu

        delta = z[:, None] - center[None, :]
        features = gain[None, :] * np.exp(-(delta**2) / (2.0 * sigma[None, :] ** 2))
        return np.asarray(features, dtype=float)

    def _compute_direct_curve_features(self, z: np.ndarray) -> np.ndarray:
        if self.library.curves is None:
            raise ValueError("direct_curve mode requires normalized transfer curves.")

        features = np.empty((len(z), self.library.n_kernels), dtype=float)
        curves = self.library.curves.copy()
        curves["curve_id_str"] = curves["curve_id"].astype(str)
        for j, curve_id in enumerate(self.params_["curve_id"]):
            group = curves[curves["curve_id_str"] == str(curve_id)].sort_values("Vg")
            if group.empty:
                raise ValueError(f"No normalized transfer curve found for curve_id={curve_id!r}.")
            vg = group["Vg"].to_numpy(dtype=float)
            values = group["response_norm"].to_numpy(dtype=float)
            features[:, j] = np.interp(z, vg, values, left=values[0], right=values[-1])
        return features


class HardwareAwareKernelRegressor:
    """Linear readout fitted on top of a hardware-aware kernel bank."""

    def __init__(self, kernel_bank: HardwareAwareKernelBank, ridge_alpha: float = 0.0):
        if ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative.")
        self.kernel_bank = kernel_bank
        self.ridge_alpha = float(ridge_alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HardwareAwareKernelRegressor":
        y_array = np.asarray(y, dtype=float)
        self.output_was_1d_ = y_array.ndim == 1
        if self.output_was_1d_:
            y_array = y_array[:, None]
        if y_array.ndim != 2:
            raise ValueError("y must be a one-dimensional or two-dimensional array.")

        self.kernel_bank.fit(X, y_array)
        features = self.kernel_bank.compute_features(X)
        self.train_feature_rank_ = int(np.linalg.matrix_rank(features))
        design = _append_bias(features)
        if self.ridge_alpha == 0.0:
            weights, *_ = np.linalg.lstsq(design, y_array, rcond=None)
        else:
            penalty = np.eye(design.shape[1]) * self.ridge_alpha
            penalty[-1, -1] = 0.0
            lhs = design.T @ design + penalty
            rhs = design.T @ y_array
            try:
                weights = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                weights, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        self.weights_ = weights
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not getattr(self, "is_fitted", False):
            raise RuntimeError("HardwareAwareKernelRegressor must be fitted before predict().")
        features = self.kernel_bank.compute_features(X)
        pred = _append_bias(features) @ self.weights_
        if self.output_was_1d_:
            return pred[:, 0]
        return pred


def _as_2d_float(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("Expected a one-dimensional or two-dimensional input array.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Input array contains non-finite values.")
    return X


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _append_bias(features: np.ndarray) -> np.ndarray:
    return np.column_stack([features, np.ones(features.shape[0])])
