"""
RBF Neural Network Implementation
Implements both original paper version and improved version with regularization.
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, Tuple
from src.gaussian_kernel import GaussianKernelLayer, determine_sigma_kneighbors, compute_sigma_heuristic


class RBFNetwork:
    """
    Radial Basis Function Neural Network.

    Supports two modes:
    - 'original': Paper version with fixed sigma values
    - 'improved': With normalization, sigma determination, and ridge regression

    Attributes:
        n_inputs: Number of input features
        n_outputs: Number of output units
        kernel_layer: Gaussian kernel layer
        weights: Output layer weights
        mode: 'original' or 'improved'
    """

    def __init__(self, n_inputs: int, n_outputs: int = 1,
                 mode: str = 'original',
                 use_normalization: bool = False,
                 use_ridge: bool = False,
                 ridge_alpha: float = 0.01):
        """
        Args:
            n_inputs: Number of input features
            n_outputs: Number of output units
            mode: 'original' (paper) or 'improved'
            use_normalization: Normalize inputs for numerical stability
            use_ridge: Use Ridge regression instead of plain least-squares
            ridge_alpha: Ridge regularization strength
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.mode = mode
        self.use_normalization = use_normalization
        self.use_ridge = use_ridge
        self.ridge_alpha = ridge_alpha

        self.kernel_layer = GaussianKernelLayer(
            sigma_strategy=mode,
            use_normalization=use_normalization
        )
        self.weights: Optional[np.ndarray] = None
        self.centers: Optional[np.ndarray] = None

    def set_normalization_params(self, mean: np.ndarray, std: np.ndarray):
        """Set normalization parameters for numerical stability"""
        self.kernel_layer.set_normalization_params(mean, std)

    def fit(self, X: np.ndarray, y: np.ndarray,
            n_kernels: int = 80,
            n_sigma_distinct: int = None,
            sigma_values: Optional[np.ndarray] = None,
            sigma_strategy: str = "fixed"):
        """
        Fit the RBF network.

        Paper approach:
        - For Panama: total_kernels=200, n_sigma_distinct varies 1~80
        - For Duffing: total_kernels=300, n_sigma_distinct varies 1~80

        Args:
            X: Training inputs (n_samples, n_features)
            y: Training targets (n_samples, n_outputs)
            n_kernels: Total number of Gaussian kernels (e.g., 200 for Panama)
            n_sigma_distinct: Number of DISTINCT sigma values (sigma diversity).
                             If None, defaults to n_kernels (each kernel unique sigma).
                             Paper uses: total=200, n_sigma_distinct=1~80.
                             When n_sigma_distinct < n_kernels, sigma values are SHARED.
            sigma_values: Pre-defined sigma values array (length = n_sigma_distinct)
                         If None, sigma values are generated based on sigma_strategy.
            sigma_strategy: How to generate sigma values ('fixed', 'kneighbors', 'heuristic')
        """
        n_samples = X.shape[0]

        # Normalize if needed
        if self.use_normalization:
            self.input_mean = np.mean(X, axis=0)
            self.input_std = np.std(X, axis=0) + 1e-8
            X_normalized = (X - self.input_mean) / self.input_std
            self.kernel_layer.set_normalization_params(self.input_mean, self.input_std)
        else:
            X_normalized = X
            self.input_mean = None
            self.input_std = None

        # Determine kernel centers using k-means clustering
        n_kernels = min(n_kernels, n_samples)
        kmeans = KMeans(n_clusters=n_kernels, random_state=42, n_init=10)
        kmeans.fit(X_normalized)
        self.centers = kmeans.cluster_centers_

        # Determine number of distinct sigma values
        if n_sigma_distinct is None:
            n_sigma_distinct = n_kernels

        # Generate sigma values for each DISTINCT sigma group
        if sigma_values is None:
            if sigma_strategy == "kneighbors":
                sigma_base = determine_sigma_kneighbors(self.centers, k=3)
                sigma_distinct = np.array(sorted(set(sigma_base)))[:n_sigma_distinct]
            elif sigma_strategy == "heuristic":
                sigma = compute_sigma_heuristic(self.centers, method="mean_distance")
                sigma_distinct = np.array([sigma])
            else:  # 'fixed' - use logarithmic distribution across sigma range
                sigma_distinct = np.logspace(np.log10(0.5), np.log10(10.0), n_sigma_distinct)

        # Ensure we have n_sigma_distinct values
        if len(sigma_distinct) < n_sigma_distinct:
            sigma_distinct = np.tile(sigma_distinct, n_sigma_distinct // len(sigma_distinct) + 1)[:n_sigma_distinct]

        # ASSIGN sigma values to each kernel (cycling through distinct sigma values)
        # This is the key: multiple kernels share the same sigma value
        kernel_sigmas = np.array([sigma_distinct[i % n_sigma_distinct] for i in range(n_kernels)])

        # Add kernels
        self.kernel_layer.clear_kernels()
        for i in range(n_kernels):
            self.kernel_layer.add_kernel(self.centers[i], kernel_sigmas[i])

        # Store info for debugging
        self.n_sigma_distinct = n_sigma_distinct

        # Compute activations
        activations = self.kernel_layer.forward(X_normalized)

        # Solve for weights using least-squares (or Ridge regression)
        if self.use_ridge:
            # Ridge regression: W = (X^T X + alpha*I)^(-1) X^T y
            I = np.eye(activations.shape[1])
            XTX = activations.T @ activations
            XTy = activations.T @ y
            self.weights = np.linalg.solve(XTX + self.ridge_alpha * I, XTy)
        else:
            # Plain least-squares
            self.weights = np.linalg.lstsq(activations, y, rcond=None)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the RBF network.

        Args:
            X: Input samples (n_samples, n_features)

        Returns:
            Predictions (n_samples, n_outputs)
        """
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Normalize if needed
        if self.use_normalization and self.input_mean is not None:
            X_normalized = (X - self.input_mean) / self.input_std
        else:
            X_normalized = X

        # Compute activations
        activations = self.kernel_layer.forward(X_normalized)

        # Compute output
        y_pred = activations @ self.weights

        return y_pred

    def get_kernel_info(self) -> dict:
        """Get information about the kernels"""
        if self.centers is None:
            return {}

        sigma_values = [k.sigma for k in self.kernel_layer.kernels]

        return {
            'n_kernels': len(self.centers),
            'sigma_min': np.min(sigma_values),
            'sigma_max': np.max(sigma_values),
            'sigma_mean': np.mean(sigma_values),
            'input_dim': self.n_inputs
        }


class RBFNetworkFactory:
    """Factory for creating RBF networks with different configurations."""

    @staticmethod
    def create_paper_version(n_inputs: int, n_outputs: int = 1) -> RBFNetwork:
        """Create original paper version"""
        return RBFNetwork(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            mode='original',
            use_normalization=False,
            use_ridge=False
        )

    @staticmethod
    def create_improved_version(n_inputs: int, n_outputs: int = 1,
                                 ridge_alpha: float = 0.01) -> RBFNetwork:
        """Create improved version with regularization and normalization"""
        return RBFNetwork(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            mode='improved',
            use_normalization=True,
            use_ridge=True,
            ridge_alpha=ridge_alpha
        )

    @staticmethod
    def create_custom_version(n_inputs: int, n_outputs: int = 1,
                               use_normalization: bool = True,
                               use_ridge: bool = True,
                               ridge_alpha: float = 0.01) -> RBFNetwork:
        """Create custom version with specified options"""
        return RBFNetwork(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            mode='improved' if (use_normalization or use_ridge) else 'original',
            use_normalization=use_normalization,
            use_ridge=use_ridge,
            ridge_alpha=ridge_alpha
        )


def select_n_sigmas_from_range(sigma_candidates: np.ndarray, N: int) -> np.ndarray:
    """
    STEP 6.1: Select N distinct sigma values spanning the experimental range.

    Selects N representative quantiles from the experimental sigma distribution,
    spanning [sigma_min, sigma_max] range from AAT device measurements.

    Args:
        sigma_candidates: Array of sigma candidates from experimental distribution
        N: Number of distinct sigma values to select

    Returns:
        Array of N distinct sigma values sorted in ascending order
    """
    if N <= 0:
        return np.array([])
    if N == 1:
        return np.array([np.median(sigma_candidates)])
    # Select N evenly spaced quantiles
    quantiles = np.linspace(0, 1, N)
    sigma_set = np.quantile(sigma_candidates, quantiles)
    return np.sort(sigma_set)
