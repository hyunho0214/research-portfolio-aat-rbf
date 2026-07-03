# Reproducibility Guide

The repository contains several project generations. The commands below are the
recommended checks for a reviewer or future maintainer.

## Environment

Use Python 3.10 or newer. On this Windows machine, `py -3` is the safest Python
entry point.

Common dependencies:

```powershell
py -3 -m pip install numpy pandas scipy scikit-learn matplotlib openpyxl pytest pyyaml
```

For the packaged hardware-aware kernel project:

```powershell
cd new_AAT
py -3 -m pip install -e .[dev]
py -3 -m pytest -q
```

## `20260616Panama`

Run tests:

```powershell
cd 20260616Panama
py -3 -m pytest -q
```

Extract Gaussian sigma values from the transfer-curve template:

```powershell
py -3 extract_sigmas.py templates\transfer_curve_template.xlsx --output-dir output\my_sigmas
```

Run the Panama RBF simulation with extracted sigma values:

```powershell
py -3 run_panama.py --sigma-file output\my_sigmas\sigma_values.xlsx --sigma-scale 0.1 --n-values 1,3,10,80 --selected-n 1,3,10,80 --output-dir output\panama_my_sigmas
```

Run Duffing reconstruction:

```powershell
py -3 run_duffing.py --n-values 3,80 --selected-n 3,80 --output-dir output\duffing_selected
```

Generate the interactive Panama plot:

```powershell
py -3 make_interactive_panama.py --output-dir output\panama_my_sigmas
```

## Expected Artifacts

After the main workflows, inspect:

- `20260616Panama/output/my_sigmas/sigma_values.xlsx`
- `20260616Panama/output/my_sigmas/figures/gaussian_fit_overview.png`
- `20260616Panama/output/panama_my_sigmas/panama_rbf_metrics.csv`
- `20260616Panama/output/panama_my_sigmas/figures/panama_15_day_segments.png`
- `20260616Panama/output/panama_my_sigmas/interactive/panama_interactive.html`
- `20260616Panama/output/duffing_selected/duffing_rbf_metrics.csv`
- `20260616Panama/output/duffing_selected/figures/duffing_phase_space.png`

## `FEDL_Data`

Check that Python files compile:

```powershell
py -3 -m compileall -q FEDL_Data
```

Run the integrated GUI:

```powershell
cd FEDL_Data\final
run_fedl_tool.bat
```

Run standalone preprocessing:

```powershell
cd "FEDL_Data\data preprocessing"
run_raw_data_gui.bat
```

Run standalone plotting:

```powershell
cd FEDL_Data\plotting
run_plotting_gui.bat
```

Representative FEDL output:

- `FEDL_Data/final/outputs/Plot1_V1_Abs_Id_log.png`
- `FEDL_Data/plotting/outputs/Plot1_V1_Abs_Id_log.png`
- `FEDL_Data/data preprocessing/raw_data.csv`

## Notes on Data and Outputs

- Experimental data and generated outputs are intentionally included for
  portfolio evidence.
- Python caches, package build metadata, and virtual environments are excluded
  from Git.
- If old scripts fail because of dependency drift, use `new_AAT` and
  `20260616Panama` as the primary maintained review targets.
