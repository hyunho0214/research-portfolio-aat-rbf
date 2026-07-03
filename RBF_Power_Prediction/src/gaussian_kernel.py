"""
Gaussian Kernel Functions for RBF Neural Network
Implements both original paper version and improved version with regularization
"""

import numpy as np
from typing import Optional, List


class GaussianKernel:
    """
    Single Gaussian kernel for RBF network.

    Attributes:
        center: Kernel center vector (k-dimensional)
        sigma: Standard deviation (width) of the Gaussian
        amplitude: Amplitude factor A (default=1.0, can be absorbed into weights)
    """

    def __init__(self, center: np.ndarray, sigma: float, amplitude: float = 1.0):
        self.center = center
        self.sigma = sigma
        self.amplitude = amplitude

    def __call__(self, x: np.ndarray) -> float:
        """
        Compute Gaussian activation for input x.

        G(x) = A * exp(-||x - center||^2 / (2 * sigma^2))

        Args:
            x: Input vector

        Returns:
            Gaussian activation value
        """
        distance = np.linalg.norm(x - self.center)
        return self.amplitude * np.exp(-(distance ** 2) / (2 * self.sigma ** 2))

    def compute_activation(self, x: np.ndarray) -> float:
        """Alias for __call__"""
        return self(x)


class GaussianKernelLayer:
    """
    Layer of Gaussian kernels for RBF network.
    Supports both original paper version and improved version.

    Note: Normalization is handled externally by RBFNetwork to avoid double-normalization.
    This layer expects pre-normalized data when use_normalization=True.
    """

    def __init__(self, sigma_strategy: str = "original", use_normalization: bool = False):
        """
        Args:
            sigma_strategy: 'original' (paper) or 'improved' (with sigma determination)
            use_normalization: Whether normalization params are stored (actual normalization
                              is done externally in RBFNetwork)
        """
        self.kernels: list[GaussianKernel] = []
        self.sigma_strategy = sigma_strategy
        self.use_normalization = use_normalization
        self.input_mean = None
        self.input_std = None
        self._centers_matrix: Optional[np.ndarray] = None
        self._sigmas_vector: Optional[np.ndarray] = None
        self._amplitudes_vector: Optional[np.ndarray] = None
        self._is_dirty = True

    def set_normalization_params(self, mean: np.ndarray, std: np.ndarray):
        """Set normalization parameters for numerical stability"""
        self.input_mean = mean
        self.input_std = std

    def clear_kernels(self):
        """Clear all kernels and invalidate cache"""
        self.kernels = []
        self._is_dirty = True

    def add_kernel(self, center: np.ndarray, sigma: float, amplitude: float = 1.0):
        """Add a Gaussian kernel to the layer"""
        self.kernels.append(GaussianKernel(center, sigma, amplitude))
        self._is_dirty = True

    def _prepare_batch_arrays(self):
        """Precompute centers, sigmas, and amplitudes as arrays for vectorized computation"""
        if self._is_dirty or self._centers_matrix is None or len(self._centers_matrix) != len(self.kernels):
            self._centers_matrix = np.array([k.center for k in self.kernels])
            self._sigmas_vector = np.array([k.sigma for k in self.kernels])
            self._amplitudes_vector = np.array([k.amplitude for k in self.kernels])
            self._is_dirty = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute activations for all kernels using VECTORIZED computation.

        Args:
            x: Input vector (or matrix of shape (n_samples, n_features))
               NOTE: Expected to be already normalized if use_normalization=True

        Returns:
            Array of kernel activations (n_samples, n_kernels)
        """
        # Handle single sample
        if x.ndim == 1:
            x = x.reshape(1, -1)

        self._prepare_batch_arrays()

        if len(self.kernels) == 0:
            return np.zeros((x.shape[0], 0))

        # Vectorized distance computation using broadcasting
        # x: (n_samples, n_features), centers: (n_kernels, n_features)
        # Result: (n_samples, n_kernels)
        distances = np.linalg.norm(x[:, np.newaxis, :] - self._centers_matrix[np.newaxis, :, :], axis=2)

        # Vectorized Gaussian activation
        # activations[j, i] = amplitude[i] * exp(-||x[j] - center[i]||^2 / (2 * sigma[i]^2))
        activations = self._amplitudes_vector * np.exp(-(distances ** 2) / (2 * self._sigmas_vector ** 2))

        return activations

    def forward_legacy(self, x: np.ndarray) -> np.ndarray:
        """
        Legacy forward method with nested loops (kept for reference).
        Use forward() instead for better performance.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        activations = np.zeros((x.shape[0], len(self.kernels)))
        for i, kernel in enumerate(self.kernels):
            for j in range(x.shape[0]):
                activations[j, i] = kernel(x[j])

        return activations

    def get_activations_single(self, x: np.ndarray) -> np.ndarray:
        """Get activations for a single input"""
        return self.forward(x).flatten()


def determine_sigma_kneighbors(centers: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Determine sigma based on k-nearest neighbors (Improved version only).

    For each center, find k nearest neighbors and compute average distance.
    This is used in the improved version to set sigma values.

    Args:
        centers: Array of kernel centers (n_centers, n_features)
        k: Number of nearest neighbors to consider

    Returns:
        Array of sigma values for each center
    """
    n_centers = centers.shape[0]
    sigmas = np.zeros(n_centers)

    for i in range(n_centers):
        # Compute distances to all other centers
        distances = np.linalg.norm(centers - centers[i], axis=1)
        # Exclude self (set self-distance to infinity)
        distances[i] = np.inf
        # Get k nearest
        k_nearest = np.partition(distances, k)[:k]
        # Sigma is average distance to k nearest neighbors
        sigmas[i] = np.mean(k_nearest)

    return sigmas


def compute_sigma_heuristic(centers: np.ndarray, method: str = "mean_distance") -> float:
    """
    Compute a heuristic sigma value for all kernels.

    Args:
        centers: Array of kernel centers
        method: Method to use ('mean_distance', 'std_distance', 'max_distance')

    Returns:
        Single sigma value
    """
    n_centers = centers.shape[0]

    # Compute all pairwise distances
    distances = []
    for i in range(n_centers):
        for j in range(i + 1, n_centers):
            dist = np.linalg.norm(centers[i] - centers[j])
            distances.append(dist)

    distances = np.array(distances)

    if method == "mean_distance":
        return np.mean(distances)
    elif method == "std_distance":
        return np.std(distances)
    elif method == "max_distance":
        return np.max(distances) / np.sqrt(n_centers)
    else:
        raise ValueError(f"Unknown method: {method}")
