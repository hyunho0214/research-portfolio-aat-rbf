"""
Duffing Oscillator Data Generation
Generates chaotic time-series data using 4th-order Runge-Kutta integration.
"""

import numpy as np
from typing import Tuple


def duffing_oscillator(t: float, x: float, v: float,
                       alpha: float = -1.0, beta: float = 1.0,
                       delta: float = 0.2, gamma: float = 0.3,
                       omega: float = 1.2) -> Tuple[float, float]:
    """
    Duffing oscillator equation: x'' + delta*x' + beta*x³ = gamma*cos(omega*t)

    The system is: x' = v, v' = -delta*v - beta*x³ + gamma*cos(omega*t)

    Args:
        t: Time
        x: Position
        v: Velocity
        alpha, beta: Stiffness parameters (default: alpha=-1, beta=1)
        delta: Damping coefficient (default: 0.2)
        gamma: Forcing amplitude (default: 0.3)
        omega: Forcing frequency (default: 1.2)

    Returns:
        (dx/dt, dv/dt) tuple
    """
    dxdt = v
    dvdt = -delta * v - beta * x**3 + gamma * np.cos(omega * t)
    return dxdt, dvdt


def runge_kutta_4th_order(t: float, x: float, v: float, dt: float, **params) -> Tuple[float, float]:
    """
    4th-order Runge-Kutta integration step.

    Args:
        t: Current time
        x: Current position
        v: Current velocity
        dt: Time step
        **params: Additional parameters for duffing_oscillator

    Returns:
        (next_x, next_v) tuple
    """
    # k1
    k1_x, k1_v = duffing_oscillator(t, x, v, **params)
    # k2
    k2_x, k2_v = duffing_oscillator(t + dt/2, x + k1_x*dt/2, v + k1_v*dt/2, **params)
    # k3
    k3_x, k3_v = duffing_oscillator(t + dt/2, x + k2_x*dt/2, v + k2_v*dt/2, **params)
    # k4
    k4_x, k4_v = duffing_oscillator(t + dt, x + k3_x*dt, v + k3_v*dt, **params)

    next_x = x + (dt/6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
    next_v = v + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return next_x, next_v


def generate_duffing_data(n_samples: int = 10000, dt: float = 0.001,
                          x0: float = 0.0, v0: float = 0.0,
                          discard_transient: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Duffing oscillator time-series data.

    The equation is: x'' + 0.2x' + x³ = 0.3cos(1.2t)

    Args:
        n_samples: Number of samples to generate
        dt: Time step in seconds (default: 1 ms)
        x0, v0: Initial conditions
        discard_transient: Number of initial samples to discard (transient response)

    Returns:
        (t, x, v) where t is time array, x is position, v is velocity
    """
    t = np.zeros(n_samples + discard_transient)
    x = np.zeros(n_samples + discard_transient)
    v = np.zeros(n_samples + discard_transient)

    # Initial conditions
    x[0] = x0
    v[0] = v0

    # Default parameters from paper: x'' + 0.2x' + x³ = 0.3cos(1.2t)
    params = {
        'alpha': -1.0,
        'beta': 1.0,
        'delta': 0.2,
        'gamma': 0.3,
        'omega': 1.2
    }

    # Generate data
    for i in range(n_samples + discard_transient - 1):
        t[i+1] = t[i] + dt
        x[i+1], v[i+1] = runge_kutta_4th_order(t[i], x[i], v[i], dt, **params)

    # Discard transient
    t = t[discard_transient:]
    x = x[discard_transient:]
    v = v[discard_transient:]

    return t, x, v


def create_sliding_window_data(x: np.ndarray, v: np.ndarray,
                                 window_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-output pairs using sliding window method.

    Input: 10 consecutive samples (10 ms)
    Output: Next oscillator state (position and velocity)

    Args:
        x: Position time-series
        v: Velocity time-series
        window_size: Number of consecutive samples for input

    Returns:
        (X, y) where X is input matrix (n_samples, window_size*2)
               and y is output vector (n_samples, 2)
    """
    n = len(x)
    n_pairs = n - window_size

    X = np.zeros((n_pairs, window_size * 2))  # Both position and velocity
    y = np.zeros((n_pairs, 2))  # Next state: [position, velocity]

    for i in range(n_pairs):
        # Input: 10 consecutive samples (both x and v)
        input_window = np.concatenate([
            x[i:i+window_size],
            v[i:i+window_size]
        ])
        X[i] = input_window

        # Output: next state
        y[i] = [x[i+window_size], v[i+window_size]]

    return X, y


def generate_training_test_data(test_ratio: float = 0.2,
                                  window_size: int = 10,
                                  n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate train/test split for Duffing oscillator prediction.

    Args:
        test_ratio: Ratio of data to use for testing
        window_size: Sliding window size
        n_samples: Total samples to generate

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    t, x, v = generate_duffing_data(n_samples=n_samples, dt=0.001)
    X, y = create_sliding_window_data(x, v, window_size)

    # Split
    n = len(X)
    n_test = int(n * test_ratio)
    n_train = n - n_test

    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test
