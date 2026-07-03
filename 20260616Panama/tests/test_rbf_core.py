import unittest

import numpy as np

from src.rbf import (
    build_design_matrix,
    create_sliding_window_1d,
    create_sliding_window_multivariate,
    ensure_n_values,
    fit_predict_rbf,
    log_spaced_sigmas,
    parse_n_values,
    replicate_centers_and_sigmas,
    select_sigma_set,
)


class RBFCoreTests(unittest.TestCase):
    def test_parse_n_values(self):
        self.assertEqual(parse_n_values("1:3"), [1, 2, 3])
        self.assertEqual(parse_n_values("1,3,10"), [1, 3, 10])
        self.assertEqual(parse_n_values("2..4"), [2, 3, 4])

    def test_sliding_window_1d(self):
        x, y = create_sliding_window_1d([1, 2, 3, 4, 5], 2)
        np.testing.assert_array_equal(x, np.array([[1, 2], [2, 3], [3, 4]], dtype=float))
        np.testing.assert_array_equal(y.ravel(), np.array([3, 4, 5], dtype=float))

    def test_sliding_window_multivariate(self):
        data = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=float)
        x, y = create_sliding_window_multivariate(data, 2)
        self.assertEqual(x.shape, (2, 4))
        np.testing.assert_array_equal(y, np.array([[3, 30], [4, 40]], dtype=float))

    def test_log_sigmas_and_replication(self):
        sigmas = log_spaced_sigmas(3, 12, 3)
        self.assertEqual(len(sigmas), 3)
        centers = np.array([[0], [1], [2]], dtype=float)
        expanded_centers, expanded_sigmas = replicate_centers_and_sigmas(centers, sigmas, 8)
        self.assertEqual(expanded_centers.shape, (8, 1))
        self.assertEqual(expanded_sigmas.shape, (8,))
        self.assertEqual((expanded_centers.ravel() == 0).sum(), 3)
        self.assertEqual((expanded_centers.ravel() == 1).sum(), 3)
        self.assertEqual((expanded_centers.ravel() == 2).sum(), 2)

    def test_select_sigmas_from_candidates(self):
        candidates = [0.3, 1.0, 3.0, 20.0]
        np.testing.assert_allclose(select_sigma_set(3, sigma_candidates=candidates, method="logspace"), [0.3, np.sqrt(6), 20.0])
        np.testing.assert_allclose(select_sigma_set(3, sigma_candidates=candidates, method="quantile"), [0.3, 2.0, 20.0])

    def test_design_matrix_shape_and_bias(self):
        x = np.array([[0.0], [1.0]])
        centers = np.array([[0.0], [1.0], [2.0]])
        sigmas = np.array([1.0, 1.0, 1.0])
        phi = build_design_matrix(x, centers, sigmas, chunk_size=1)
        self.assertEqual(phi.shape, (2, 4))
        np.testing.assert_array_equal(phi[:, -1], np.ones(2))

    def test_fit_predict_smoke(self):
        rng = np.random.default_rng(0)
        x_train = rng.normal(size=(40, 3))
        y_train = x_train[:, :1] * 0.5
        x_test = rng.normal(size=(10, 3))
        y_test = x_test[:, :1] * 0.5
        result = fit_predict_rbf(
            x_train,
            y_train,
            x_test,
            y_test,
            n_distinct=3,
            total_centers=6,
            sigma_min=1.0,
            sigma_max=3.0,
            cluster_method="minibatch",
            n_init=1,
            max_iter=20,
            output_names=["target"],
        )
        self.assertEqual(result.y_pred.shape, y_test.shape)
        self.assertTrue(np.isfinite(result.metrics["MSE_target"]))

    def test_ensure_n_values(self):
        self.assertEqual(ensure_n_values([3, 1, 3], 10), [1, 3])
        with self.assertRaises(ValueError):
            ensure_n_values([0], 10)


if __name__ == "__main__":
    unittest.main()
