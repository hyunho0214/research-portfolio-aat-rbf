"""Command-line entry point for Duffing reconstruction."""

from __future__ import annotations

import argparse

from duffing import run_duffing_experiment
from main_forecasting import str_to_bool
from sigma_utils import load_sigma_data, parse_n_values


def build_parser() -> argparse.ArgumentParser:
    """Create the Duffing CLI parser."""
    parser = argparse.ArgumentParser(description="Run AAT-derived multi-Gaussian RBF Duffing reconstruction.")
    parser.add_argument("--sigma-csv", default=None, help="CSV containing experimental AAT sigma values.")
    parser.add_argument("--sigma-column", default=None, help="Column name containing sigma values.")
    parser.add_argument("--sigma-values", default=None, help="Comma-separated sigma values.")
    parser.add_argument("--output-dir", default="outputs/duffing", help="Directory for saved outputs.")
    parser.add_argument("--n-values", default=None, help="Comma-separated N sweep values. Default: 1..80.")
    parser.add_argument("--window-length", type=int, default=10, help="Input state window length.")
    parser.add_argument("--c-total", type=int, default=300, help="Total number of RBF kernels.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for k-means and sigma assignment.")
    parser.add_argument("--shuffle-sigmas", type=str_to_bool, default=True, help="Shuffle sigma assignments.")
    parser.add_argument("--train-fraction", type=float, default=0.5, help="Chronological training fraction.")
    parser.add_argument("--transient-steps", type=int, default=2000, help="Initial RK4 steps discarded.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Duffing cubic stiffness coefficient.")
    parser.add_argument("--beta", type=float, default=0.2, help="Duffing damping coefficient.")
    parser.add_argument("--gamma", type=float, default=0.3, help="Duffing forcing amplitude.")
    parser.add_argument("--omega", type=float, default=1.2, help="Duffing forcing angular frequency.")
    parser.add_argument("--dt", type=float, default=0.01, help="RK4 step size.")
    parser.add_argument("--n-steps", type=int, default=20000, help="Total RK4 simulation steps.")
    parser.add_argument("--x0", type=float, default=0.1, help="Initial displacement.")
    parser.add_argument("--v0", type=float, default=0.0, help="Initial velocity.")
    return parser


def main() -> None:
    """Load CLI inputs and run the Duffing reconstruction experiment."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        sigma_data = load_sigma_data(args.sigma_csv, args.sigma_column, args.sigma_values)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    result = run_duffing_experiment(
        sigma_data=sigma_data,
        output_dir=args.output_dir,
        n_values=parse_n_values(args.n_values),
        window_length=args.window_length,
        c_total=args.c_total,
        seed=args.seed,
        shuffle_sigmas=args.shuffle_sigmas,
        train_fraction=args.train_fraction,
        transient_steps=args.transient_steps,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        omega=args.omega,
        dt=args.dt,
        n_steps=args.n_steps,
        x0=args.x0,
        v0=args.v0,
    )
    best_n = result["config"]["best_N"]
    print(f"Duffing reconstruction complete. Best N={best_n}. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
