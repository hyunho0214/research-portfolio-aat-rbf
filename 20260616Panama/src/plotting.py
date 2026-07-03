"""Plotting helpers for Figure 4-style outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_duffing_phase_space(
    y_true: np.ndarray,
    predictions: Mapping[int, np.ndarray],
    path: Path,
    *,
    selected_n: Sequence[int],
) -> None:
    """Save phase-space reconstructions for selected N values."""

    fig, axes = plt.subplots(1, len(selected_n), figsize=(5.2 * len(selected_n), 4.4), squeeze=False)
    for ax, n in zip(axes.ravel(), selected_n):
        pred = predictions[n]
        ax.plot(y_true[:, 0], y_true[:, 1], color="0.8", linewidth=0.7, label="True")
        ax.plot(pred[:, 0], pred[:, 1], linewidth=0.8, label=f"Predicted, N = {n}")
        ax.set_title(f"N = {n}")
        ax.set_xlabel("X(t)")
        ax.set_ylabel("Y(t)")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 1)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_duffing_time_domain(
    y_true: np.ndarray,
    predictions: Mapping[int, np.ndarray],
    path: Path,
    *,
    selected_n: Sequence[int],
    max_points: int = 100_000,
) -> None:
    """Save x/v time-domain overlays for selected N values."""

    n_points = min(max_points, len(y_true))
    x_axis = np.arange(n_points)
    fig, axes = plt.subplots(len(selected_n), 2, figsize=(12, 3.2 * len(selected_n)), squeeze=False)
    for row, n in enumerate(selected_n):
        pred = predictions[n]
        for col, (name, idx) in enumerate([("Y(t)", 1), ("X(t)", 0)]):
            ax = axes[row, col]
            ax.plot(x_axis, y_true[:n_points, idx], color="0.25", linewidth=0.8, label="True Value")
            ax.plot(
                x_axis,
                pred[:n_points, idx],
                linestyle="--",
                linewidth=0.8,
                label=f"Predicted, N = {n}",
            )
            ax.set_xlabel("Time Step")
            ax.set_ylabel(name)
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.94)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_panama_full_test(
    y_true: np.ndarray,
    predictions: Mapping[int, np.ndarray],
    path: Path,
    *,
    selected_n: Sequence[int],
) -> None:
    """Save a Figure 4h-style 3D line stack for the full test interval."""

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    y = y_true.reshape(-1)
    days = np.arange(len(y), dtype=float) / 24.0
    fig = plt.figure(figsize=(10.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    y_levels = np.arange(len(selected_n) + 1, dtype=float)
    ax.plot(days, np.full_like(days, y_levels[-1]), y, color="0.25", linewidth=0.7, label="True Value")
    colors = plt.cm.YlOrRd(np.linspace(0.25, 0.85, len(selected_n)))
    for level, n, color in zip(y_levels[:-1], selected_n, colors):
        ax.plot(days, np.full_like(days, level), predictions[n].reshape(-1), color=color, linewidth=0.7, label=f"N = {n}")
    ax.set_xlabel("Days")
    ax.set_ylabel("# Different Gaussian")
    ax.set_zlabel("Electric Demand (MW)")
    ax.set_yticks(list(y_levels))
    ax.set_yticklabels([f"N={n}" for n in selected_n] + ["True"])
    ax.view_init(elev=22, azim=-67)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.94)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_panama_15_day_segments(
    y_true: np.ndarray,
    predictions: Mapping[int, np.ndarray],
    path: Path,
    *,
    selected_n: Sequence[int],
    days: int = 15,
) -> None:
    """Save Figure 4i-style 15-day forecast segments."""

    n_points = min(days * 24, len(y_true))
    x_axis = np.arange(n_points, dtype=float) / 24.0
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    colors = plt.cm.YlOrRd(np.linspace(0.25, 0.85, len(selected_n)))
    for ax, n, color in zip(axes.ravel(), selected_n, colors):
        ax.plot(x_axis, y_true[:n_points].reshape(-1), color="0.25", linewidth=0.8, label="True Value")
        ax.plot(
            x_axis,
            predictions[n][:n_points].reshape(-1),
            color=color,
            linestyle="--",
            linewidth=0.9,
            label=f"N = {n}",
        )
        ax.set_xlabel("Days")
        ax.set_ylabel("Electric Demand (MW)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_metric_plots(metrics_rows: Sequence[dict[str, float]], path_prefix: Path, *, r2_key: str = "R2_mean") -> None:
    """Save MSE and R2 versus N plots."""

    n = np.asarray([row["N"] for row in metrics_rows], dtype=float)
    mse = np.asarray([row["MSE_mean"] for row in metrics_rows], dtype=float)
    r2 = np.asarray([row[r2_key] for row in metrics_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(n, mse, marker="o", markersize=3.0, linewidth=0.9, color="#d9821f")
    ax.set_xlabel("# of Different Gaussian")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path_prefix.with_name(path_prefix.name + "_mse.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(n, r2, marker="o", markersize=3.0, linewidth=0.9, color="#b46b38")
    ax.set_xlabel("# of Different Gaussian")
    ax.set_ylabel("R2")
    ax.set_ylim(min(-0.05, float(np.nanmin(r2)) - 0.05), min(1.05, max(1.0, float(np.nanmax(r2)) + 0.05)))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path_prefix.with_name(path_prefix.name + "_r2.png"), dpi=300)
    plt.close(fig)
