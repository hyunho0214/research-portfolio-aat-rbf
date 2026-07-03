from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aat_kernel.calibration import gaussian_response, save_calibration_outputs
from aat_kernel.cli import run_hardware_experiment


def test_calibration_and_hardware_experiment_smoke(tmp_path: Path) -> None:
    curves_path = tmp_path / "curves.csv"
    rows = []
    vg = np.linspace(-1.5, 1.5, 80)
    for idx, (A, mu, sigma) in enumerate([(1.0, -0.5, 0.25), (1.3, 0.1, 0.35), (0.9, 0.55, 0.3)]):
        response = gaussian_response(vg, 0.05, A, mu, sigma)
        for x, y in zip(vg, response):
            rows.append({"curve_id": f"c{idx}", "Vg": x, "response": y, "Vd": 0.1 * idx})
    pd.DataFrame(rows).to_csv(curves_path, index=False)

    calibration_dir = tmp_path / "kernel_library"
    calibration_paths = save_calibration_outputs(curves_path, calibration_dir, make_plots=False)

    supervised_path = tmp_path / "supervised.csv"
    t = np.linspace(0, 4, 100)
    supervised = pd.DataFrame(
        {
            "time": np.arange(len(t)),
            "x1": np.sin(t),
            "x2": np.cos(t),
            "y": 0.7 * np.sin(t) + 0.2 * np.cos(t),
        }
    )
    supervised.to_csv(supervised_path, index=False)

    config = {
        "seed": 0,
        "output_dir": str(tmp_path / "experiment"),
        "data": {
            "supervised_csv": str(supervised_path),
            "feature_columns": ["x1", "x2"],
            "target_columns": ["y"],
            "time_col": "time",
            "standardize_targets": True,
        },
        "split": {"train_fraction": 0.8},
        "kernel_library": {
            "path": str(calibration_paths["kernel_library"]),
            "curves_path": str(calibration_paths["normalized_transfer_curves"]),
            "filter": {},
        },
        "encoder": {"mode": "pca_sigmoid"},
        "models": {"variants": ["sigma_only", "a_sigma", "full_gaussian", "direct_curve"]},
        "readout": {"ridge_alpha": 1e-6},
    }

    outputs = run_hardware_experiment(config)
    assert outputs["metrics"].exists()
    metrics = pd.read_csv(outputs["metrics"])
    assert set(metrics["variant"]) == {"sigma_only", "a_sigma", "full_gaussian", "direct_curve"}
    for variant in ("sigma_only", "a_sigma", "full_gaussian", "direct_curve"):
        assert outputs[f"predictions_{variant}"].exists()
        assert outputs[f"prediction_plot_{variant}"].exists()

    saved_config = json.loads(outputs["config"].read_text(encoding="utf-8"))
    assert saved_config["resolved"]["n_kernels"] == 3
