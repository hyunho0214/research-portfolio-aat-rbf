"""
RBF Network Implementation
Based on Algorithm 1, STEP 6.2 - 6.5

Key changes based on Codex feedback:
- Ridge regularization for numerical stability
- Edge-case guards
"""

import numpy as np
from sklearn.cluster import KMeans
from src.config import RIDGE_ALPHA, RANDOM_STATE


class RBFNetwork:
    """
    Radial Basis Function Neural Network.
    Follows Algorithm 1 with ridge regularization for stability.
    """

    def __init__(self, input_dim: int, num_centers: int,
                 centers: np.ndarray, sigma_values: np.ndarray,
                 output_dim: int = 1):
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.centers = centers
        self.sigma_values = sigma_values
        self.output_dim = output_dim
        self.weights = None

    @staticmethod
    def create_with_clustering(X_train: np.ndarray, N: int, C_total: int,
                                sigma_set: np.ndarray) -> 'RBFNetwork':
        """
        Factory: Create RBF network with k-means clustering.
        STEP 6.2: K-means on X_train with N clusters
        STEP 6.3: Expand to C_total centers
        """
        # STEP 6.2: K-means clustering with reproducibility
        kmeans = KMeans(n_clusters=N, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_train)
        cluster_centers = kmeans.cluster_centers_

        # STEP 6.3: Expand to C_total centers
        base_count = C_total // N
        extra_count = C_total % N

        centers_list = []
        sigma_list = []

        for k in range(N):
            count_k = base_count + (1 if k < extra_count else 0)
            for _ in range(count_k):
                centers_list.append(cluster_centers[k])
                sigma_list.append(sigma_set[k])

        return RBFNetwork(
            input_dim=X_train.shape[1],
            num_centers=C_total,
            centers=np.array(centers_list),
            sigma_values=np.array(sigma_list),
            output_dim=1
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        STEP 6.4: Train RBF network using ridge regularized least squares.
        Design matrix Phi with bias column, solve with ridge regression.
        """
        # Compute design matrix
        distances = np.linalg.norm(
            X_train[:, np.newaxis, :] - self.centers[np.newaxis, :, :],
            axis=2
        )
        Phi_train = np.exp(-(distances ** 2) / (2 * self.sigma_values ** 2))

        # Augment with bias column
        Phi_train_bias = np.hstack([Phi_train, np.ones((Phi_train.shape[0], 1))])

        # Ridge regularized least squares: w = (Phi^T Phi + alpha*I)^-1 Phi^T y
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)

        I = np.eye(Phi_train_bias.shape[1])
        I[-1, -1] = 0  # Don't regularize bias term
        XTX = Phi_train_bias.T @ Phi_train_bias
        XTy = Phi_train_bias.T @ y_train
        self.weights = np.linalg.solve(XTX + RIDGE_ALPHA * I, XTy)

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """STEP 6.5: Evaluate RBF network."""
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")

        distances = np.linalg.norm(
            X_test[:, np.newaxis, :] - self.centers[np.newaxis, :, :],
            axis=2
        )
        Phi_test = np.exp(-(distances ** 2) / (2 * self.sigma_values ** 2))
        Phi_test_bias = np.hstack([Phi_test, np.ones((Phi_test.shape[0], 1))])

        return Phi_test_bias @ self.weights