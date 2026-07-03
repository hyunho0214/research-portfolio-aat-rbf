# memT_BO Curation Notes

This folder arrived as a working lab directory with many exploratory scripts,
versioned drafts, duplicate exports, and result files. I reviewed the scripts by
kernel, input schema, output behavior, and modification date, then kept the
files that best support a portfolio review.

## Selection Criteria

- Keeps a clear experimental story: TFT pre-test, TFT mobility optimizer, then
  memtransistor optimizer.
- Preserves final CSV-input workflows rather than one-off interactive drafts.
- Keeps representative outputs that prove the scripts generated prediction
  grids and next-point recommendations.
- Preserves kernel-comparison evidence for RBF, Constant x RBF, RQ, and Matern.
- Excludes duplicate ZIP archives, `untitled` scratch files, and older root-level
  version chains when a cleaner later script exists.

## Primary Files Kept

| Purpose | Kept files |
| --- | --- |
| Final TFT mobility BO | `TFT/250723 TFT BO_HH/simple TFT BO(ratio-mobility)(Cons&RBF)_csv input.py`, `experiment_data.csv`, `mobility_prediction_iter_0.csv`, `next_point_iter_0.csv` |
| TFT evidence deck | `TFT/250723 TFT BO_HH/IGZO TFT Bayesian Optimization (Mobility).pptx` |
| Memtransistor RBF workflows | `memT/csv input/(ratio,thickness)-retention, excel (ConRBF)_file load_NEW_1.py`, `memT/csv input/(ratio,thickness)-retention, excel (RBF)_file load_NEW_1.py` |
| Memtransistor RQ comparison | `memT/csv input/*/(ratio,thickness)-retention, excel (Const+RQ)_file load.py` |
| Memtransistor seed/output CSV | `memT/csv input/experiment_data.csv`, `onoff_ratio_prediction_iter_0.csv`, `next_point_iter_0.csv` |
| TFT process FOM test | `memT/TEST/TFT test.py`, `memT/TEST/full_data.csv` |
| Earlier prototype | `main/main.py`, `main/file_input.py`, `main/hello.csv` |
| Saved result evidence | `memT/RBF 결과/*.xlsx`, `memT/retention_prediction_iter_*.xlsx` |
| Supporting artifacts | `TFT BO 기초 교육.pptx`, `memT/band_structure_comparison.png`, `memT/memt BO.opj` |
| README visuals | `assets/memT_bo_rbf_rq_prediction_comparison.png`, `assets/memT_bo_generated_prediction_preview.png` |

## Local Files Excluded From Git

The ignored files are still present locally, but they are intentionally left out
of the GitHub portfolio snapshot.

- Root-level `GPR_RBF ver*.py`, `GPR_RQ ver*.py`, and `GPR_Simple_*.py` files:
  useful development history, but superseded by the curated memT CSV-input
  workflows.
- Root-level `simple TFT BO*.py` and `untitled*.py`: early scratch and notebook
  spillover.
- Duplicate ZIP archives: the extracted folders are easier to inspect on
  GitHub and avoid redundant large binaries.
- Older TFT presentations and duplicate TFT root outputs: the `250723 TFT
  BO_HH` package is the cleanest final TFT evidence.
- Older memT root scripts: replaced by later CSV-input versions with explicit
  output saving and next-point logging.

## Interview Framing

This project is strongest as evidence of experimental automation rather than as
a polished software package. The useful discussion points are:

- how a limited process grid was converted into a Bayesian optimization problem,
- why Expected Improvement is useful when each device run is expensive,
- how kernel choice changed the recommendation behavior,
- how TFT experiments were used as a lower-risk pre-test before memtransistor
  optimization,
- how prediction grids and next-point CSVs made the workflow auditable across
  lab iterations.
