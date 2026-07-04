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

## `memT_BO`

Regenerate the README preview image from the saved prediction grid:

```powershell
py -3 memT_BO\scripts\make_prediction_preview.py
```

Compile the curated Bayesian-optimization scripts:

```powershell
py -3 -m py_compile "memT_BO\TFT\250723 TFT BO_HH\simple TFT BO(ratio-mobility)(Cons&RBF)_csv input.py"
py -3 -m py_compile "memT_BO\memT\csv input\(ratio,thickness)-retention, excel (ConRBF)_file load_NEW_1.py"
py -3 -m py_compile "memT_BO\memT\csv input\(ratio,thickness)-retention, excel (RBF)_file load_NEW_1.py"
py -3 -m py_compile "memT_BO\memT\csv input\새 폴더\(ratio,thickness)-retention, excel (Const+RBF)_file load_NEW_1.py"
py -3 -m py_compile "memT_BO\memT\csv input\새 폴더\(ratio,thickness)-retention, excel (Const+RQ)_file load.py"
py -3 -m py_compile "memT_BO\memT\TEST\TFT test.py"
```

Run the final TFT mobility optimizer:

```powershell
cd "memT_BO\TFT\250723 TFT BO_HH"
py -3 "simple TFT BO(ratio-mobility)(Cons&RBF)_csv input.py"
```

Run the final memtransistor on/off-ratio optimizer:

```powershell
cd "memT_BO\memT\csv input"
py -3 "(ratio,thickness)-retention, excel (ConRBF)_file load_NEW_1.py"
```

Representative memT BO outputs:

- `memT_BO/TFT/250723 TFT BO_HH/mobility_prediction_iter_0.csv`
- `memT_BO/TFT/250723 TFT BO_HH/next_point_iter_0.csv`
- `memT_BO/memT/csv input/onoff_ratio_prediction_iter_0.csv`
- `memT_BO/memT/csv input/next_point_iter_0.csv`
- `memT_BO/assets/memT_bo_rbf_iteration_comparison.png`
- `memT_BO/assets/memT_bo_generated_prediction_preview.png`
- `memT_BO/memT/retention_prediction_iter_final.xlsx`
- `memT_BO/memT/RBF 결과/*.xlsx`

## `SECOM_Defect_Prediction`

Install the required ML dependencies:

```powershell
cd SECOM_Defect_Prediction
py -3 -m pip install -r requirements.txt
```

Run a compile check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_validation.ps1 -Mode compile
```

Run the smoke benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_validation.ps1 -Mode smoke
```

Representative SECOM outputs:

- `SECOM_Defect_Prediction/reports/final/model_metric_comparison.png`
- `SECOM_Defect_Prediction/reports/final/best_model_confusion_matrix.png`
- `SECOM_Defect_Prediction/reports/final/threshold_tradeoff.png`
- `SECOM_Defect_Prediction/reports/final/metrics_summary.csv`
- `SECOM_Defect_Prediction/reports/final/tuning_logistic_best_params.json`

## `MES_SQLD_Practice`

Run the SQLite practice workflow:

```powershell
cd MES_SQLD_Practice
powershell -ExecutionPolicy Bypass -File scripts\run_validation.ps1
```

Representative SQLD/MES outputs:

- `MES_SQLD_Practice/outputs/01_defect_wafer_filter.csv`
- `MES_SQLD_Practice/outputs/02_pm_yield_relation.csv`
- `MES_SQLD_Practice/outputs/03_wafer_map_defect_preprocess.csv`
- `MES_SQLD_Practice/outputs/run_summary.md`

## Notes on Data and Outputs

- Experimental data and generated outputs are intentionally included for
  portfolio evidence.
- Python caches, package build metadata, and virtual environments are excluded
  from Git.
- If old scripts fail because of dependency drift, use `new_AAT` and
  `20260616Panama` as the primary maintained review targets.
