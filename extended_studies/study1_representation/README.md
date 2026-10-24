# Study 1: grid adjoint vs neural adjoint

**Question.** In the main comparison the adjoint represents D on a 16 x 16 control grid and the PINN with a neural network. Does the representation itself change the result?

**Design.** Both arms are the same classical method, the exact discrete adjoint of Crank-Nicolson (`core/adjoint_inversion.py`). Everything is shared except how log D is parameterised:

- same data, CN time grid, Tikhonov regulariser and discrepancy-principle choice of alpha;
- same optimiser (L-BFGS-B, 150 iterations per alpha) and the same best-constant starting point;
- **grid arm:** 16 x 16 control values, bilinear map to the 65 x 65 grid;
- **neural arm:** the D-network of `core/neural_diffusivity.py` (3 x 32 tanh, D in [0.001, 10]), with the adjoint gradient passed back through it (`core/adjoint_nn.py`).

Every neural-adjoint fit was run by `core/compare_neural.py` (method `adjoint_nn`, seed 42) through the queue in `campaign.py`. For the grid arm the queue reused an existing classical estimate where one existed and otherwise ran `compare_neural.py` with method `grid`:

| Collection | Grid arm |
|---|---|
| Main benchmark (389) | The Tikhonov estimates of `core/adjoint_inversion.py` from the main code (`../../submission/models/classical/tikhonov_all389/`; identical values in `../analysis/tikhonov_summary_all_389.csv`) |
| Hypothesis suites H1-H7, H9 (81), small density pilots (6) | The adjoint results of study 3 and of the density pilots (column `grid_historical` in `campaign/paired_errors.csv`) |
| Extended density (180), H10u (6) | Run in the campaign (`campaign/results/<collection>/<job>/<key>/grid__s42/`) |

## Files

| File | What it does |
|---|---|
| `make_density_datasets.py` | Builds `datasets_v2_density/` (A_smooth, noise-free, 40 times, nested 25 / 100 / 400 / 3,969 sensors, after the thinning in Thakur et al. 2024) and, with `--wide-ic`, `datasets_v2_density_wideic/` (400 and 3,969 sensors, wider initial bumps) |
| `extend_sensor_density.py` | Builds `datasets_v2_density_extended/` (180 datasets: 6 fields x 10 sensor counts from 5 to 3,969 x noise 1 / 5 / 10 %) as nested subsets of the existing full-field readings, without re-simulating, then validates them |

Run from `extended_studies/`, after `python run.py core/generate_dataset_v2.py`:

```bash
python run.py study1_representation/make_density_datasets.py
python run.py study1_representation/make_density_datasets.py --wide-ic
python run.py study1_representation/extend_sensor_density.py
```

`extend_sensor_density.py` ends with a step that adds the new datasets to the campaign queue. That step needs the queue state (`campaign/jobs.json`), which is not committed, so it is skipped when the file is absent; the datasets are still written and validated.

## Results

662 paired datasets. Source: `campaign/paired_adjoint_results.csv` (one row per pair), summarised by field in `campaign/paired_adjoint_summary.csv`. "Reached noise level" means the discrepancy principle stopped with replay RMS at sigma (flag `reached_sigma`).

| Collection | Pairs | Median relL2(D), grid / neural | Lower error, grid / neural | Reached noise level, grid / neural |
|---|---:|---|---|---|
| Main benchmark (`datasets_v2_rk4_zero`) | 389 | 0.225 / 0.226 | 192 / 197 | 324 / 236 (30 noise-free) |
| Extended density (`datasets_v2_density_extended`) | 180 | 0.158 / 0.187 | 112 / 68 | 157 / 84 |
| Hypothesis suites H1-H7, H9 (`datasets_v2_pinnwin`, `_pinnwin2`) | 81 | 0.198 / 0.190 | 38 / 43 | 49 / 31 |
| H10u, hidden BC (`datasets_v2_pinnwin3`) | 6 | 5.08 / 5.88 | 3 / 3 | 0 / 0 (2 noise-free) |
| Small density pilots (`datasets_v2_density`, `_wideic`) | 6 | 0.021 / 0.087 | 6 / 0 | noise-free |

On the main benchmark the two representations are equally accurate: the medians differ by 0.001 and each arm is better on about half the datasets. The difference is in the fit: the grid reaches the noise level on 324 datasets against 236 for the network, which more often stops at a data-fit floor above sigma (`floor_above_sigma`). With more sensors (extended density) the grid is clearly better. By field, the network is lower more often on B_gaussian_bump (76 of 93) and D_layered (79 of 93), and less often on F_defects (7 of 93), E_grf (28 of 101) and A_smooth (44 of 123) (`paired_adjoint_summary.csv`, all collections with known physics).

In the hypothesis suites, 22 of the 81 pairs have a hidden boundary condition or source and both arms must assume zero (`physics_mode = assumed_zero`); there the grid is lower in 15 of 22. On the 59 with known physics the network is lower in 36 of 59 (median 0.129 vs 0.123 for the grid).

Other tables built from the same runs:

| File | Contents |
|---|---|
| `campaign/density_paired_results.csv`, `density_paired_summary.csv` | The 180 extended-density pairs, and per field and noise level |
| `campaign/density_complete_curves.csv` | Error at 5 vs 3,969 sensors per field, noise and method |
| `campaign/density_path_diagnostics.csv` | Selected alpha, iteration caps and replay along the alpha path |
| `campaign/effects_*.csv` | Median error and replay by one factor at a time (Fo, noise, layout, sensors, times, protocol, field) |
| `campaign/defect_results.csv`, `defect_details.csv` | Defect detection scores on F_defects fields |
| `campaign/figures/` | Density curves, paired coefficient maps, data fit vs replay |

**Limitations.** One seed (42) per pair. The optimiser budget is capped (150 iterations per alpha), and this study does not separate a data-fit floor caused by the representation from one caused by the cap; `density_path_diagnostics.csv` records where the cap was hit on the density collection.
