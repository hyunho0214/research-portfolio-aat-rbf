"""Plotting helpers for saved experiment diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metric_sweep(metrics: pd.DataFrame, output_path: str | Path, metric: str) -> None:
    """Save a line plot showing how a metric changes with sigma diversity N."""
    if metric not in metrics.columns:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(metrics["N"], metrics[metric], marker="o")
    ax.set_xlabel("Number of distinct sigma values (N)")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_forecasting_predictions(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    max_points: int = 500,
) -> None:
    """Save a compact true-vs-predicted electricity demand overlay."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_points = min(len(y_true), max_points)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps[:n_points], y_true[:n_points], label="true", linewidth=1.4)
    ax.plot(timestamps[:n_points], y_pred[:n_points], label="predicted", linewidth=1.2)
    ax.set_xlabel("time")
    ax.set_ylabel("nat_demand")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_duffing_time(
    t: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    max_points: int = 1000,
) -> None:
    """Save time-domain overlays for x(t) and v(t)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_points = min(len(y_true), max_points)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    labels = ["x", "v"]
    for index, ax in enumerate(axes):
        ax.plot(t[:n_points], y_true[:n_points, index], label=f"true {labels[index]}")
        ax.plot(t[:n_points], y_pred[:n_points, index], label=f"pred {labels[index]}")
        ax.set_ylabel(labels[index])
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_duffing_phase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    max_points: int = 3000,
) -> None:
    """Save phase-space reconstruction plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_points = min(len(y_true), max_points)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(y_true[:n_points, 0], y_true[:n_points, 1], label="true", linewidth=1.0)
    ax.plot(y_pred[:n_points, 0], y_pred[:n_points, 1], label="predicted", linewidth=1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
