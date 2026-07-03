# AAT Hardware-Aware Kernel

This project builds a hardware-aware kernel library from measured AAT transfer
curves and compares four kernel-regression variants on generic supervised CSV
data:

- `sigma_only`
- `a_sigma`
- `full_gaussian`
- `direct_curve`

The initial implementation follows `guide.md` and keeps measured device
parameters in the gate-voltage domain by using a PCA-to-sigmoid scalar encoder.

## Install

```powershell
py -m pip install -e .[dev]
```

## Calibrate Transfer Curves

Input curves use long CSV format with required columns:

- `curve_id`
- `Vg`
- `response`

Optional metadata columns such as `device_id`, `Vd`, `light_intensity`,
`wavelength`, `pulse_width`, and `state_tag` are preserved when present.

```powershell
aat-calibrate-curves --curves data/raw_curves.csv --out outputs/kernel_library
```

If console scripts are not on `PATH`, use:

```powershell
py -m aat_kernel calibrate-curves --curves data/raw_curves.csv --out outputs/kernel_library
```

Outputs include:

- `kernel_library.csv`
- `curve_fit_metrics.csv`
- `normalized_transfer_curves.csv`
- `curve_fit_examples.png`
- `calibration_config.json`

## Run Hardware-Aware Regression

Edit `configs/hardware_experiment.yaml` with the actual data paths and columns:

```powershell
aat-run-hardware --config configs/hardware_experiment.yaml
```

If console scripts are not on `PATH`, use:

```powershell
py -m aat_kernel run-hardware --config configs/hardware_experiment.yaml
```

Outputs include:

- `hardware_kernel_metrics.csv`
- `hardware_kernel_predictions_<variant>.csv`
- `hardware_kernel_predictions_<variant>.png`
- `hardware_kernel_config.json`
