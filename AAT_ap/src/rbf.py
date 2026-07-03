"""Multi-Gaussian RBF network with least-squares linear readout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


def compute_kmeans_centers(x_train: np.ndarray, c_total: int, seed: int = 0) -> np.ndarray:
    """Compute C_total distinct centers in standardized input space."""
    if c_total <= 0:
        raise ValueError("c_total must be positive.")
    if c_total > len(x_train):
        raise ValueError("c_total cannot exceed the number of training samples.")
    kmeans = KMeans(n_clusters=c_total, random_state=seed, n_init=10)
    kmeans.fit(x_train)
    return kmeans.cluster_centers_


def median_nearest_center_distance(x_train: np.ndarray, centers: np.ndarray) -> float:
    """Return median distance from each train sample to its nearest RBF center."""
    x_train = np.asarray(x_train, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if x_train.ndim != 2 or centers.ndim != 2:
        raise ValueError("x_train and centers must be 2D arrays.")
    if x_train.shape[1] != centers.shape[1]:
        raise ValueError("x_train and centers must have the same feature dimension.")

    diff = x_train[:, None, :] - centers[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    nearest_distances = np.min(distances, axis=1)
    reference_distance = float(np.median(nearest_distances))
    if reference_distance <= 0.0:
        raise ValueError("Median nearest-center distance must be positive.")
    return reference_distance


@dataclass
class RBFNetwork:
    """Gaussian RBF hidden layer plus a closed-form linear output layer."""

    centers: np.ndarray
    sigma_per_center: np.ndarray
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=float)
        self.sigma_per_center = np.asarray(self.sigma_per_center, dtype=float).reshape(-1)
        if self.centers.ndim != 2:
            raise ValueError("centers must have shape (C_total, input_dim).")
        if len(self.sigma_per_center) != len(self.centers):
            raise ValueError("sigma_per_center length must match number of centers.")
        if np.any(self.sigma_per_center <= 0.0):
            raise ValueError("All sigma values must be strictly positive.")

    def compute_features(self, x: np.ndarray) -> np.ndarray:
        """Evaluate phi_j(u)=exp(-||u-c_j||^2/(2*sigma_j^2))."""
        x = np.asarray(x, dtype=float)
        diff = x[:, None, :] - self.centers[None, :, :]
        squared_distance = np.sum(diff * diff, axis=2)
        denominator = 2.0 * (self.sigma_per_center[None, :] ** 2)
        return np.exp(-squared_distance / denominator)

    def _augment_with_bias(self, phi: np.ndarray) -> np.ndarray:
        """Append an intercept column for the linear readout."""
        return np.hstack([phi, np.ones((phi.shape[0], 1), dtype=phi.dtype)])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RBFNetwork":
        """Fit output weights using least squares on fixed RBF features."""
        phi_aug = self._augment_with_bias(self.compute_features(x))
        self.weights, *_ = np.linalg.lstsq(phi_aug, np.asarray(y, dtype=float), rcond=None)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict standardized targets for standardized input vectors."""
        if self.weights is None:
            raise ValueError("The RBFNetwork must be fitted before prediction.")
        phi_aug = self._augment_with_bias(self.compute_features(x))
        return phi_aug @ self.weights
