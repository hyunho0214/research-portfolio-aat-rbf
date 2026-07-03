"""Duffing oscillator generation for the Figure 4 reproduction."""

from __future__ import annotations

import numpy as np


def duffing_rhs(
    t: float,
    state: np.ndarray,
    *,
    damping: float = 0.2,
    drive: float = 0.3,
    omega: float = 1.2,
) -> np.ndarray:
    """Right-hand side for x'' + 0.2 x' - x + x^3 = 0.3 cos(1.2 t)."""

    x, v = state
    return np.asarray([v, -damping * v + x - x**3 + drive * np.cos(omega * t)], dtype=float)


def integrate_duffing_rk4(
    *,
    dt: float = 0.001,
    transient_steps: int = 100_000,
    post_steps: int = 200_000,
    x0: float = 0.1,
    v0: float = 0.0,
    damping: float = 0.2,
    drive: float = 0.3,
    omega: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the Duffing system with fixed-step RK4 and discard transient."""

    if dt <= 0:
        raise ValueError("dt must be positive.")
    if transient_steps < 0 or post_steps <= 0:
        raise ValueError("transient_steps must be >= 0 and post_steps must be positive.")

    total_steps = transient_steps + post_steps
    states = np.empty((total_steps + 1, 2), dtype=float)
    states[0] = [x0, v0]
    t = 0.0
    state = states[0].copy()

    for idx in range(total_steps):
        k1 = duffing_rhs(t, state, damping=damping, drive=drive, omega=omega)
        k2 = duffing_rhs(t + 0.5 * dt, state + 0.5 * dt * k1, damping=damping, drive=drive, omega=omega)
        k3 = duffing_rhs(t + 0.5 * dt, state + 0.5 * dt * k2, damping=damping, drive=drive, omega=omega)
        k4 = duffing_rhs(t + dt, state + dt * k3, damping=damping, drive=drive, omega=omega)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        states[idx + 1] = state

    start = transient_steps
    end = transient_steps + post_steps
    post_states = states[start:end]
    post_times = np.arange(start, end, dtype=float) * dt
    return post_times, post_states
