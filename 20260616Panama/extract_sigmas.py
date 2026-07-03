"""Extract Gaussian sigma values from VG-ID transfer-curve data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.transfer_sigma import fit_transfer_table, gaussian_with_baseline, load_transfer_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Excel transfer-curve table.")
    parser.add_argument("--sheet", default=None, help="Excel sheet name or index. Default: first sheet.")
    parser.add_argument("--vg-column", default=None, help="VG column name. Default: inferred or first column.")
    parser.add_argument("--id-columns", default=None, help="Comma-separated ID columns. Default: all numeric non-VG columns.")
    parser.add_argument("--no-abs-current", action="store_true", help="Fit signed current instead of abs(ID).")
    parser.add_argument("--min-sigma", type=float, default=None, help="Lower bound for fitted sigma in VG units.")
    parser.add_argument("--max-sigma", type=float, default=None, help="Upper bound for fitted sigma in VG units.")
    parser.add_argument("--output-dir", type=Path, default=Path("output") / "extracted_sigmas")
    return parser.parse_args()


def _parse_sheet(value: str | None) -> str | int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _parse_id_columns(value: str | None) -> list[str] | None:
    if value is None:
        return None
    columns = [part.strip() for part in value.split(",") if part.strip()]
    return columns or None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    df = load_transfer_table(args.input, sheet_name=_parse_sheet(args.sheet))
    result, vg_column, id_columns = fit_transfer_table(
        df,
        vg_column=args.vg_column,
        id_columns=_parse_id_columns(args.id_columns),
        use_abs=not args.no_abs_current,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
    )

    print("[extract_sigmas] Input summary")
    print(f"  file: {args.input}")
    print(f"  VG column: {vg_column}")
    print(f"  ID columns fitted: {len(id_columns)}")
    print(f"  ID column names: {', '.join(id_columns)}")
    print(f"  abs current: {not args.no_abs_current}")
    unnamed = [name for name in id_columns if str(name).lower().startswith("unnamed")]
    if unnamed:
        print("[extract_sigmas] WARNING")
        print(f"  Headerless numeric columns were found: {', '.join(unnamed)}")
        print("  They were fitted, but you may want to put names in row 1, e.g. ID_#21 or VD_6V_ID.")

    results_path = args.output_dir / "gaussian_fit_results.xlsx"
    sigma_path = args.output_dir / "sigma_values.xlsx"
    sigma_result = result.loc[
        result["success"] & result["sigma"].notna(),
        ["curve_index", "column", "vd", "sigma", "mu", "amplitude", "baseline", "r2"],
    ]
    result.to_excel(results_path, index=False, sheet_name="gaussian_fit_results")
    sigma_result.to_excel(sigma_path, index=False, sheet_name="sigma_values")

    metadata = {
        "input": str(args.input),
        "sheet": args.sheet,
        "vg_column": vg_column,
        "id_columns": id_columns,
        "use_abs_current": not args.no_abs_current,
        "min_sigma_bound": args.min_sigma,
        "max_sigma_bound": args.max_sigma,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    save_fit_overview(df, result, vg_column, id_columns, fig_dir / "gaussian_fit_overview.png", use_abs=not args.no_abs_current)
    save_sigma_summary(result, fig_dir / "sigma_summary.png")

    success = int(result["success"].sum())
    total = len(result)
    ok = result[result["success"] & result["sigma"].notna()]
    if not ok.empty:
        print("[extract_sigmas] Extracted sigma summary")
        print(f"  sigma_min: {ok['sigma'].min():.6g} V")
        print(f"  sigma_max: {ok['sigma'].max():.6g} V")
        print(f"  sigma_median: {ok['sigma'].median():.6g} V")
        print(f"  fit_R2_min: {ok['r2'].min():.6g}")
        print(f"  fit_R2_median: {ok['r2'].median():.6g}")
        print("[extract_sigmas] Extracted sigma values")
        preview_columns = ["curve_index", "column", "vd", "sigma", "mu", "r2"]
        print(ok[preview_columns].to_string(index=False, max_rows=200))
    print(f"Fitted {success}/{total} curves")
    print(f"Wrote {results_path}")
    print(f"Wrote {sigma_path}")
    print(f"Wrote {fig_dir / 'gaussian_fit_overview.png'}")
    print(f"Wrote {fig_dir / 'sigma_summary.png'}")


def save_fit_overview(
    df: pd.DataFrame,
    result: pd.DataFrame,
    vg_column: str,
    id_columns: list[str],
    path: Path,
    *,
    use_abs: bool,
) -> None:
    vg = pd.to_numeric(df[vg_column], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    colors = plt.cm.Reds(np.linspace(0.25, 0.95, max(len(id_columns), 1)))
    x_fit = np.linspace(np.nanmin(vg), np.nanmax(vg), 500)
    for color, column in zip(colors, id_columns):
        y = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if use_abs:
            y = np.abs(y)
        ax.plot(vg, y, color=color, alpha=0.35, linewidth=0.9)
        row = result[result["column"] == str(column)]
        if not row.empty and bool(row.iloc[0]["success"]):
            fit = row.iloc[0]
            y_fit = gaussian_with_baseline(
                x_fit,
                float(fit["amplitude"]),
                float(fit["mu"]),
                float(fit["sigma"]),
                float(fit["baseline"]),
            )
            ax.plot(x_fit, y_fit, color=color, linestyle="--", linewidth=1.0)
    ax.set_xlabel("VG (V)")
    ax.set_ylabel("|ID| (A)" if use_abs else "ID (A)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_sigma_summary(result: pd.DataFrame, path: Path) -> None:
    ok = result[result["success"] & result["sigma"].notna()].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    if ok.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No successful fits", ha="center", va="center")
            ax.set_axis_off()
    else:
        x = ok["vd"] if ok["vd"].notna().any() else np.arange(len(ok))
        axes[0].plot(x, ok["sigma"], marker="o", linewidth=0.9)
        axes[0].set_xlabel("VD (V)" if ok["vd"].notna().any() else "Curve index")
        axes[0].set_ylabel("sigma from VG fit (V)")
        axes[0].grid(alpha=0.25)
        axes[1].plot(x, ok["r2"], marker="o", linewidth=0.9, color="#b46b38")
        axes[1].set_xlabel("VD (V)" if ok["vd"].notna().any() else "Curve index")
        axes[1].set_ylabel("fit R2")
        axes[1].set_ylim(min(-0.05, float(ok["r2"].min()) - 0.05), 1.02)
        axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
