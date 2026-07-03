from __future__ import annotations

import numpy as np
import pandas as pd

from aat_kernel.calibration import calibrate_kernel_library, gaussian_response


def test_calibration_recovers_synthetic_gaussian_parameters() -> None:
    rows = []
    vg = np.linspace(-2.0, 2.0, 121)
    for idx, (A, mu, sigma) in enumerate([(2.0, -0.4, 0.35), (1.4, 0.6, 0.5), (-1.8, 0.2, 0.42)]):
        response = gaussian_response(vg, baseline=0.1, A=A, mu=mu, sigma=sigma)
        for x, y in zip(vg, response):
            rows.append({"curve_id": f"c{idx}", "Vg": x, "response": y, "device_id": "d0"})

    curves = pd.DataFrame(rows)
    library, metrics, normalized = calibrate_kernel_library(curves)

    assert len(library) == 3
    assert len(metrics) == 3
    assert set(["A", "A_tilde", "mu", "sigma", "baseline", "fit_r2", "fit_mse"]).issubset(library.columns)
    assert np.all(library["fit_r2"] > 0.999)
    assert np.allclose(sorted(np.abs(library["A"])), sorted([2.0, 1.4, 1.8]), atol=1e-3)
    assert np.allclose(sorted(library["sigma"]), sorted([0.35, 0.5, 0.42]), atol=1e-3)
    assert {"curve_id", "Vg", "response_raw", "response_norm"}.issubset(normalized.columns)
