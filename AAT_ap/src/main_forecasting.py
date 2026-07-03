"""Command-line entry point for hourly electricity-demand forecasting."""

from __future__ import annotations

import argparse

from forecasting import run_forecasting_experiment
from sigma_utils import load_sigma_data, parse_n_values


def str_to_bool(value: str) -> bool:
    """Parse argparse boolean strings consistently across shells."""
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def build_parser() -> argparse.ArgumentParser:
    """Create the forecasting CLI parser."""
    parser = argparse.ArgumentParser(description="Run AAT-derived multi-Gaussian RBF forecasting.")
    parser.add_argument("--data", default="continuous dataset.csv", help="Path to the electricity CSV.")
    parser.add_argument("--sigma-csv", default=None, help="CSV containing experimental AAT sigma values.")
    parser.add_argument("--sigma-column", default=None, help="Column name containing sigma values.")
    parser.add_argument("--sigma-values", default=None, help="Comma-separated sigma values.")
    parser.add_argument("--output-dir", default="outputs/forecasting", help="Directory for saved outputs.")
    parser.add_argument("--n-values", default=None, help="Comma-separated N sweep values. Default: 1..80.")
    parser.add_argument("--window-length", type=int, default=10, help="Input window length.")
    parser.add_argument("--c-total", type=int, default=200, help="Total number of RBF kernels.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for k-means and sigma assignment.")
    parser.add_argument("--shuffle-sigmas", type=str_to_bool, default=True, help="Shuffle sigma assignments.")
    parser.add_argument("--train-start-year", type=int, default=2016, help="First training year.")
    parser.add_argument("--train-end-year", type=int, default=2019, help="Last training year.")
    parser.add_argument("--test-year", type=int, default=2020, help="Calendar year used for testing.")
    return parser


def main() -> None:
    """Load CLI inputs and run the forecasting experiment."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        sigma_data = load_sigma_data(args.sigma_csv, args.sigma_column, args.sigma_values)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    result = run_forecasting_experiment(
        data_path=args.data,
        sigma_data=sigma_data,
        output_dir=args.output_dir,
        n_values=parse_n_values(args.n_values),
        window_length=args.window_length,
        c_total=args.c_total,
        seed=args.seed,
        shuffle_sigmas=args.shuffle_sigmas,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_year=args.test_year,
    )
    best_n = result["config"]["best_N"]
    print(f"Forecasting complete. Best N={best_n}. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
