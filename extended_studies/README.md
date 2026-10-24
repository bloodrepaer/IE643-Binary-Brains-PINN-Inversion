# Extended studies: where can a PINN beat the classical adjoint?

This folder holds the evidence for Deliverable 4 of the project. It builds on the main code in `../submission/` (forward solver, dataset generator, classical adjoint, PINN) and asks *why* the two methods differ.

The main comparison in `submission/` changes two things at once: the **method** (the adjoint enforces the PDE exactly, as a hard constraint; the PINN enforces it only as a soft penalty in the loss) and the **representation of D** (a 16 x 16 control grid for the adjoint, a neural network for the PINN). The studies below separate these two factors, then look for the conditions under which the PINN comes out ahead.

## The four studies

| # | Question | Methods compared | Datasets | Folder | Main results |
|---|---|---|---|---|---|
| 1 | Does the representation of D matter? | grid adjoint vs neural adjoint (same CN adjoint, only D changes) | 662 paired: 389 main benchmark, 180 density, 87 hypothesis suites, 6 small pilots | `study1_representation/` | `campaign/paired_adjoint_results.csv`, `paired_adjoint_summary.csv`, `density_*.csv`, `effects_*.csv`, `analysis_results.csv` |
| 2 | With the same network for D, does hard vs soft physics matter? | neural adjoint vs neural PINN (identical D-network and initial weights) | 8 | `study2_matched_neural/` | `campaign/primary_neural_comparison.csv` |
| 3 | Does the PINN win when the physics is partly unknown, or the data are unusual? | grid adjoint vs PINN | 81 (hypothesis suites H1-H7, H9) | `study3_pinn_favoured/` | `study3_pinn_favoured/results/` |
| 4 | Does the PINN win with very dense sensors? | temporal neural PINN vs backward-Euler neural adjoint (same D-network) | 1,000 to 10,000 sensors, 1/5/10 % noise | `study4_dense_sensors/` | `campaign/pilots/*/RESULTS.md`, `campaign/pilots/noise_density_v24_1000_7500/SWEEP_REPORT.md` |

Short answers:

1. On the main benchmark the representation makes no difference to accuracy (median relL2(D) 0.225 grid vs 0.226 neural), but the grid reaches the noise level more often (324 vs 236 of 389). With extended sensor density the grid is more accurate (0.158 vs 0.187).
2. With the same network, the hard constraint wins all 8 cases (median relL2(D) 0.24 vs 0.61). The PINN fits the sensors to about 0.95 sigma but its D, replayed through the solver, misses them by 3 to 16 sigma.
3. The PINN wins 21 of 22 datasets in which the boundary condition or a heat source is hidden and the adjoint is forced to assume zero (median relL2(D) 0.38 vs 5.0). With the physics known, the adjoint wins 58 of 59 (median 0.12 vs 0.41).
4. At 7,500 sensors and 5 % noise the PINN reaches 8.6 % coefficient error against 25.1 % for the neural adjoint and passes both replay screens. This is 1 of the 12 cells of the sweep.

## Folder map

```
extended_studies/
  README.md                  this file
  run.py                     launcher: run any script with core/ and its own folder importable
  campaign.py                queue runner that executed studies 1 and 2 (frozen record)
  core/                      shared library imported across studies
  study1_representation/     grid adjoint vs neural adjoint
  study2_matched_neural/     neural adjoint vs neural PINN
  study3_pinn_favoured/      hypothesis suites H1-H7, H9 (and H10u)
  study4_dense_sensors/      dense-sensor sweep, versions v12 to v24
  analysis/                  figure and rescoring scripts, figures
  archive/                   code kept for completeness, not used by any study
  campaign/                  all result data: summary CSVs, per-job results, study-4 pilots, figures
```

Every folder has its own README. All results and run records live in `campaign/` (study 3 keeps its two score tables in `study3_pinn_favoured/results/`).

## How to run

Requirements: those in `../requirements.txt` (numpy, scipy, pandas, matplotlib, torch, sympy). Always work from this folder and start scripts through `run.py`, which puts `core/` and the script's own folder on the import path; every script reads and writes paths relative to this folder.

```bash
cd extended_studies

# checks (no data needed)
python run.py core/test_neural.py -v

# main benchmark data (389 datasets, about 5 min, fixed seeds) and adjoint checks
python run.py core/generate_dataset_v2.py
python run.py core/test_adjoint.py

# study 1/2: one dataset, all three methods (short budgets shown; see core/README.md)
python run.py core/compare_neural.py --keys B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05 --out results_neural_B

# study 3: hypothesis-suite data
python run.py study3_pinn_favoured/generate_pinn_favoured.py     # H1-H5, 5 to 10 min
python run.py study3_pinn_favoured/generate_pinn_favoured2.py    # H6, H7, H9

# study 1: density data
python run.py study1_representation/make_density_datasets.py
python run.py study1_representation/extend_sensor_density.py
```

Generated datasets (`datasets_v2_*`), new results (`results_*`) and test figures (`figures/`) are written to this folder and are git-ignored.

## What can be re-run and what is a record

| Part | Re-runnable from this repository? |
|---|---|
| `core/` tests, dataset generators, single inversions (`compare_neural.py`, `adjoint_inversion.py`, PINN scripts) | Yes |
| Study 1 and 3 dataset generators, `validate_pinn_favoured.py`, `score_pinnwin.py` | Yes (H10u excepted, below) |
| `campaign.py init / run / summary` (the queue that ran studies 1 and 2) | No: its queue state (`campaign/jobs.json`, `settings.json`, `code/` snapshots) is not committed. The per-job results it produced are in `campaign/results/`. |
| H10u datasets (`datasets_v2_pinnwin3`) | No: they are unpacked from `kaggle_thakur/dataset/data.zip`, which is not committed. Their results are in `campaign/results/datasets_v2_pinnwin3/`. |
| Study 4 `prepare_*` scripts | No: their inputs (`supplied_data`, `datasets_v2_dense10000`) and the v24 Colab runner (`colab_sweep_runner.py`) are not committed. All outputs, checkpoints and validation records are in `campaign/pilots/` and `campaign/v24_case_results/`. |
| `analysis/` scripts | No: they ran in a separate workspace with absolute paths. Their outputs are in `analysis/`. |

## Provenance

The run records (`campaign/pilots/*/pilot.json`, `campaign/**/source_manifest.json`) store the SHA-256 of 18 scripts. The 17 of them that are in this folder match those hashes byte for byte, so the committed code is the code that produced the recorded results. These files were moved into this layout without any edit. The eighteenth, `colab_sweep_runner.py`, is not in the repository.
