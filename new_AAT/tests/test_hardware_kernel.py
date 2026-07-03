from __future__ import annotations

import numpy as np
import pandas as pd

from aat_kernel.hardware_kernel import (
    GateEquivalentEncoder,
    HardwareAwareKernelBank,
    HardwareAwareKernelRegressor,
    HardwareKernelLibrary,
)


def _library() -> HardwareKernelLibrary:
    table = pd.DataFrame(
        {
            "kernel_id": ["k0", "k1", "k2"],
            "curve_id": ["c0", "c1", "c2"],
            "A_tilde": [1.0, 0.8, 1.2],
            "mu": [-0.5, 0.0, 0.6],
            "sigma": [0.25, 0.35, 0.45],
            "Vg_min": [-1.0, -1.0, -1.0],
            "Vg_max": [1.0, 1.0, 1.0],
        }
    )
    curve_rows = []
    vg = np.linspace(-1.0, 1.0, 51)
    for curve_id, amp, mu, sigma in zip(table["curve_id"], table["A_tilde"], table["mu"], table["sigma"]):
        response = amp * np.exp(-((vg - mu) ** 2) / (2 * sigma**2))
        for x, y in zip(vg, response):
            curve_rows.append({"curve_id": curve_id, "Vg": x, "response_norm": y})
    return HardwareKernelLibrary(table, pd.DataFrame(curve_rows))


def test_encoder_maps_to_gate_range() -> None:
    X = np.column_stack([np.linspace(-2, 2, 50), np.linspace(1, -1, 50)])
    encoder = GateEquivalentEncoder(vg_min=-1.2, vg_max=0.8)
    z = encoder.fit_transform(X)
    assert z.shape == (50,)
    assert np.all(z >= -1.2)
    assert np.all(z <= 0.8)


def test_all_kernel_modes_make_finite_feature_matrices() -> None:
    X = np.column_stack([np.linspace(-2, 2, 20), np.sin(np.linspace(0, 1, 20))])
    library = _library()
    for mode in ("sigma_only", "a_sigma", "full_gaussian", "direct_curve"):
        encoder = GateEquivalentEncoder(library.vg_min, library.vg_max)
        bank = HardwareAwareKernelBank(library, encoder, mode=mode).fit(X)
        features = bank.compute_features(X)
        assert features.shape == (20, library.n_kernels)
        assert np.all(np.isfinite(features))


def test_regressor_handles_multi_output() -> None:
    X = np.column_stack([np.linspace(-2, 2, 60), np.cos(np.linspace(0, 2, 60))])
    y = np.column_stack([np.sin(X[:, 0]), X[:, 0] + 0.2 * X[:, 1]])
    library = _library()
    bank = HardwareAwareKernelBank(library, GateEquivalentEncoder(library.vg_min, library.vg_max), mode="full_gaussian")
    model = HardwareAwareKernelRegressor(bank, ridge_alpha=1e-6).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))
    assert model.train_feature_rank_ > 0
