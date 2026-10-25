# campaign: result data

This folder stores the results and run records of studies 1, 2 and 4. Study 3 keeps its two score tables in `../study3_pinn_favoured/results/` instead.

Everything here is a record: running the code does not regenerate any of it. Some CSV and JSON files store file paths (for example `campaign\results\datasets_v2_rk4_zero\...`). These paths are relative to `extended_studies/` and were written on the machines that ran the jobs.

## What is in this folder

| Item | What it holds | Study |
|---|---|---|
| `*.csv` in this folder | Summary tables built from all the fits | 1 and 2 |
| `results/` | One folder per fit: its estimate of D, metrics and model checkpoint | 1 and 2 |
| `pilots/` | One folder per version of the dense-sensor study: protocol, validation records and results | 4 |
| `v24_case_results/` | Raw outputs of the final dense-sensor sweep, as returned from Kaggle and Colab | 4 |
| `figures/` | Plots of the grid vs neural-adjoint comparison | 1 |

**Where to start:**

- Study 1 (grid vs neural adjoint): `paired_adjoint_results.csv` and `paired_adjoint_summary.csv`.
- Study 2 (neural adjoint vs neural PINN): `primary_neural_comparison.csv`.
- Study 4 (dense sensors): `pilots/noise_density_v24_1000_7500/SWEEP_REPORT.md`.

## Terms used below

- **Job:** one fit of one method to one dataset. Studies 1 and 2 ran 856 jobs.
- **Collection:** a folder of datasets, such as `datasets_v2_rk4_zero` (see `../../submission/README_datasets.md`).
- **Methods:** `adjoint_nn` is the neural adjoint. `grid` is the grid adjoint run as a campaign job. `grid_historical` is a classical (grid-adjoint) estimate that already existed and was reused instead of re-run. `pinn` is the neural PINN.

## Summary tables (studies 1 and 2)

`summary.csv` is written by `campaign.py summary` from the per-job `metrics.json` files. All the other tables were aggregated from the same per-job results, but no script in this repository writes them.

**Tables over all fits** (every method and collection)

| File | Contents |
|---|---|
| `summary.csv` | One row per fit, 1,332 rows: the 856 campaign jobs (662 `adjoint_nn`, 186 `grid`, 8 `pinn`) and the 476 reused classical estimates (`grid_historical`). Columns: collection, dataset, method, relL2(D), replay, sensor fit, time, flag, path to `metrics.json` |
| `analysis_results.csv` | `summary.csv` plus dataset factors (Fo, protocol, layout, whether the BC and source are known) and replay under the true physics |
| `paired_errors.csv` | relL2(D) per dataset for every method: `grid`, `grid_historical`, `adjoint_nn`, `pinn` |
| `grouped_results.csv` | Median error, replay and time per collection, physics mode and method |
| `effects_Fo.csv`, `effects_eta.csv`, `effects_layout.csv`, `effects_n_sensors.csv`, `effects_n_times.csv`, `effects_protocol.csv`, `effects_tier.csv` | Median error and replay as one dataset factor varies at a time |
| `defect_results.csv`, `defect_details.csv` | Defect detection on the F_defects fields: Dice score, detections, contrast and localisation per defect |

**Study 1: grid vs neural adjoint**

| File | Contents |
|---|---|
| `paired_adjoint_results.csv` | One row per grid / neural-adjoint pair (662 pairs) |
| `paired_adjoint_summary.csv` | The pairs grouped by physics mode and field |

**Study 1 on the extended-density collection** (180 datasets)

| File | Contents |
|---|---|
| `density_results.csv` | The rows of `analysis_results.csv` for this collection: 180 grid and 180 neural-adjoint fits |
| `density_paired_results.csv` | One row per pair |
| `density_paired_summary.csv` | The pairs grouped by field and noise level |
| `density_complete_curves.csv` | Error at 5 vs 3,969 sensors, per field, noise level and method |
| `density_path_diagnostics.csv` | Per fit: the selected alpha, iteration caps, and replay along the alpha path |

**Study 2: neural adjoint vs neural PINN**

| File | Contents |
|---|---|
| `primary_neural_comparison.csv` | The two methods on 8 datasets |

**Dataset inventory**

| File | Contents |
|---|---|
| `dataset_inventory.csv`, `dataset_parameter_axes.csv` | The seven collections, and the values each dataset factor takes |

## `results/`: one folder per fit (studies 1 and 2)

```
results/<collection>/<job id>/
    run_config.json, summary.csv
    <dataset key>/<method>__s42/
        metrics.json, D_hat.npz, model.pt
```

The job id is the first 16 hex digits of the SHA-256 of `"<collection>/<key>/<method>/42"`.

There are 662 `adjoint_nn`, 186 `grid` and 8 `pinn` fits in total, spread over seven collections:

| Collection | Datasets |
|---|---:|
| `datasets_v2_rk4_zero` | 389 |
| `datasets_v2_density_extended` | 180 |
| `datasets_v2_pinnwin` | 57 |
| `datasets_v2_pinnwin2` | 24 |
| `datasets_v2_pinnwin3` | 6 |
| `datasets_v2_density` | 4 |
| `datasets_v2_density_wideic` | 2 |

The two small pilot collections, `datasets_v2_density` and `datasets_v2_density_wideic`, are referenced by the path columns of `analysis_results.csv`.

## `pilots/`: the dense-sensor study (study 4)

There is one folder per version of the study. Each holds its protocol (`PROTOCOL.md`), a pilot record (`pilot.json`, with the SHA-256 of every source file used), an input manifest, validation records and results.

| Folder | Version | Content |
|---|---|---|
| `dense10000_v21_gpu/` | v21: 10,000 sensors | `RESULTS.md`, `primary_matched_neural_comparison.csv`, checkpoints in `artifacts/`, the Kaggle kernel in `kernel/` |
| `dense1000_v22_kaggle/` | v22: 1,000 sensors, 5 % noise | One kernel per arm |
| `noise_density_v23/` | v23: 1,000 sensors at 1 % and 10 % noise | `RESULTS.md`, `noise_comparison_n1000.csv`, `cases/`. `quota_after_v22.json` records the Kaggle GPU quota left after v22 |
| `noise_density_v24_1000_7500/` | v24: the final sweep, and the final study-4 result | `SWEEP_REPORT.md`, `paired_cell_summary.csv`, `primary_matched_neural_comparison.csv`, `density_trend.png`, `noise_trend.png`, `cases/case_01` to `case_09`, and the Colab returns in `colab_returns/` |

## `v24_case_results/`: raw outputs of the v24 sweep

These are the per-case outputs exactly as returned. `r24c0N_v1/` holds the Kaggle runs (environment, job status, checkpoints, `D_hat.npz`, metrics), and `colab_returns/` holds the two Colab runs. 54 of the files here are byte-identical copies of files under `pilots/noise_density_v24_1000_7500/`.

## `figures/` (study 1)

| File | Shows |
|---|---|
| `density_noise_0.01.png`, `density_noise_0.05.png`, `density_noise_0.1.png` | Error vs sensor count, at each noise level |
| `density_defects.png` | Defect recovery (Dice score and fraction of defects found) vs sensor count, on the extended-density collection |
| `paired_coefficients.png`, `paired_fields.png` | Grid vs neural-adjoint estimates |
| `fit_vs_replay.png` | Sensor fit vs replay |

