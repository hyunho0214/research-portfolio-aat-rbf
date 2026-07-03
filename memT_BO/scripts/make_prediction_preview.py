"""Generate a portfolio preview image from saved memtransistor BO outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm


PROJECT_DIR = Path(__file__).resolve().parents[1]
PREDICTION_CSV = PROJECT_DIR / "memT" / "csv input" / "onoff_ratio_prediction_iter_0.csv"
NEXT_POINT_CSV = PROJECT_DIR / "memT" / "csv input" / "next_point_iter_0.csv"
OUTPUT_PNG = PROJECT_DIR / "assets" / "memT_bo_generated_prediction_preview.png"


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    improvement = mu - y_best - xi
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def main() -> None:
    df = pd.read_csv(PREDICTION_CSV)
    next_point = pd.read_csv(NEXT_POINT_CSV).iloc[0]

    mu_grid = df.pivot(
        index="Thickness_nm",
        columns="O2_Percent",
        values="OnOff_Ratio_Prediction",
    ).sort_index()
    sigma_grid = df.pivot(
        index="Thickness_nm",
        columns="O2_Percent",
        values="Standard_Deviation",
    ).sort_index()

    x_values = mu_grid.columns.to_numpy(dtype=float)
    y_values = mu_grid.index.to_numpy(dtype=float)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    z_mesh = mu_grid.to_numpy(dtype=float)

    measured = df.dropna(subset=["Experimental_OnOff_Ratio"])
    y_best = measured["Experimental_OnOff_Ratio"].max()
    ei_mesh = expected_improvement(z_mesh, sigma_grid.to_numpy(dtype=float), y_best)

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 6.8), constrained_layout=True)

    ax_surface = fig.add_subplot(1, 2, 1, projection="3d")
    ax_surface.plot_surface(
        x_mesh,
        y_mesh,
        z_mesh,
        cmap=cm.viridis,
        edgecolor="k",
        linewidth=0.12,
        alpha=0.88,
    )
    ax_surface.scatter(
        measured["O2_Percent"],
        measured["Thickness_nm"],
        measured["Experimental_OnOff_Ratio"],
        color="#e63946",
        s=42,
        label="Measured",
        depthshade=False,
    )
    ax_surface.scatter(
        [next_point["O2_Percent"]],
        [next_point["Thickness_nm"]],
        [next_point["Predicted_OnOff_Ratio"]],
        color="#ffbe0b",
        edgecolor="black",
        marker="*",
        s=170,
        label="Next condition",
        depthshade=False,
    )
    ax_surface.set_title("Memtransistor BO: 3D Prediction Surface", pad=12)
    ax_surface.set_xlabel("O2 percent (%)")
    ax_surface.set_ylabel("Thickness (nm)")
    ax_surface.set_zlabel("log on/off ratio")
    ax_surface.view_init(elev=28, azim=-135)
    ax_surface.legend(loc="upper left")

    ax_ei = fig.add_subplot(1, 2, 2)
    contour = ax_ei.contourf(x_mesh, y_mesh, ei_mesh, levels=30, cmap="viridis")
    ax_ei.scatter(
        [next_point["O2_Percent"]],
        [next_point["Thickness_nm"]],
        color="#ffbe0b",
        edgecolor="black",
        marker="*",
        s=190,
        label="Recommended condition",
    )
    ax_ei.scatter(
        measured["O2_Percent"],
        measured["Thickness_nm"],
        color="#e63946",
        edgecolor="white",
        s=45,
        label="Measured",
    )
    ax_ei.set_title("Expected Improvement Score")
    ax_ei.set_xlabel("O2 percent (%)")
    ax_ei.set_ylabel("Thickness (nm)")
    ax_ei.legend(loc="upper right")
    fig.colorbar(contour, ax=ax_ei, label="EI score")

    fig.savefig(OUTPUT_PNG, dpi=180)
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
