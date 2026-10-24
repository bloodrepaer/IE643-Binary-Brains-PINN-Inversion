# Study 4: very dense sensors

**Question.** The PINN fits a continuous function to the readings, so it may gain more from very dense data than the adjoint does. Does it beat the adjoint when there are thousands of sensors?

**Design.** One smooth field (A_smooth, Fo 0.1, 8 uniform times), sensors from 1,000 to 10,000 distinct sites at 1, 5 and 10 % noise. Two methods share the same D-network (3 x 32 tanh, float64, D in [0.001, 10], seed 42, the same serialized initial weights) and the same optimiser budget (1,000 Adam steps, then 10 rounds x 400 SSBroyden quasi-Newton iterations):

- **temporal PINN:** a 3 x 32 state network with the known IC and BC built in (hard constraints), a backward-Euler consistency residual between random pairs of times (step dt/T = 0.01, 4,096 pairs per batch), spatial derivatives by automatic differentiation;
- **backward-Euler (BE) neural adjoint:** the exact adjoint of a backward-Euler solve with 100 full steps (dt/T = 0.01), finite-difference space.

Note that this neural adjoint uses backward Euler with dt/T = 0.01, unlike the Crank-Nicolson neural adjoint of studies 1 and 2 (`core/adjoint_nn.py`).

Each fit is validated before the true D is opened: independent checks of the supplied physics (IC, BC, continuous PDE residual), of the checkpoint and protocol, and a replay of D_hat through two common solvers (CN and BE). A **replay screen** requires replay RMS <= 1.1 sigma under both solvers. A PINN advantage counts as *qualified* only if the fit is complete, its relL2(D) is lower than the adjoint's, and it passes both replay screens.

## Versions and scripts

The folder is flat on purpose: `prepare_dense1000.py` builds the v22 scripts from the v21 ones by text substitution and packages modules by bare file name, and the validate and evaluate scripts import the pair modules as siblings.

| Version | Content | Scripts |
|---|---|---|
| v12 | Matched neural inverse objectives with hard IC/BC | `calibrated_neural_pair.py` |
| v18 | Temporal-consistency PINN on dense cases | `temporal_density_pair.py` |
| v20 | BE neural adjoint matched to the v18 time step | `matched_temporal_adjoint.py` |
| v21 | 10,000 sensors, 5 % noise | `prepare_dense10000.py`, `dense10000_pair.py`, `validate_dense10000.py`, `evaluate_dense10000.py`, `dense10000_kernel.py` |
| v22 | 1,000-sensor subset of v21, 5 % noise | `prepare_dense1000.py`, `dense1000_pair.py`, `validate_dense1000.py`, `evaluate_dense1000.py`, `dense1000_kernel.py` |
| v23 | 1,000 sensors at 1 % and 10 % noise | `prepare_noise_density.py`, `noise_density_pair.py`, `validate_noise_density.py`, `evaluate_noise_density.py`, `noise_density_kernel.py` |
| v24 | Final sweep: 3,000 / 5,000 / 7,500 sensors x 1 / 5 / 10 % noise; its report also includes the v22 and v23 cells as the 1,000-sensor row | `prepare_noise_density_1000_7500.py`, `noise_density_1000_7500_pair.py`, `validate_noise_density_1000_7500.py`, `evaluate_noise_density_1000_7500.py`, `noise_density_1000_7500_kernel.py`, `build_noise_density_sweep_report.py` |
| all | Independent PINN check used by v21, v22, v24 | `evaluate_ic_pilot.py` |

v12, v18 and v20 are building blocks: every v21-v24 pair script imports `temporal_density_pair.py` and `matched_temporal_adjoint.py`, which import `calibrated_neural_pair.py`, which imports `core/pinn_zhang_transfer.py` and `core/pinn_papers.py` (the SSBroyden routines).

Role of each script type:

| Type | Role |
|---|---|
| `prepare_*` | Build the dataset (nested sensor subsets of one fine-grid RK4 simulation, fixed noise draw), the input package and hash manifest, and the pilot record |
| `*_pair` | The two matched fits for one case |
| `validate_*` | Full-count gradient and pipeline checks on the supplied data, without the true D |
| `evaluate_ic_pilot.py` | First independent check of a finished PINN fit, before the true D is opened: known IC and BC, continuous PDE residual by time bin, and CN replay (`--pilot-dir <case folder> --data <collection>`). Used by v21, v22 and v24 |
| `evaluate_*` (others) | Checks of both arms of a finished pair, before the true D is opened: checkpoints, bounds, shared initialisation, protocol, and replay under the common BE and CN solvers |
| `*_kernel` | Entry point for the cloud run (Kaggle); runs inside a flat package with the modules beside it |
| `build_noise_density_sweep_report.py` | Aggregates the validated v24 scores into `SWEEP_REPORT.md`, the CSV tables and the two trend plots; no training or solver evaluation |

## Results

Records: `campaign/pilots/<version>/` (protocol, validation, results, checkpoints), raw v24 case outputs in `campaign/v24_case_results/`.

**v24 sweep** (`campaign/pilots/noise_density_v24_1000_7500/SWEEP_REPORT.md`, `paired_cell_summary.csv`, `density_trend.png`, `noise_trend.png`). relL2(D) in %; replay RMS / sigma under CN / BE.

| Sensors | Noise | PINN | BE adjoint | PINN replay CN / BE | Adjoint replay CN / BE | Qualified PINN advantage |
|---:|---:|---:|---:|---|---|---|
| 1,000 | 1 % | 9.40 | 32.04 | 2.97 / 1.71 | 2.61 / 1.00 | No |
| 3,000 | 1 % | 6.12 | 38.76 | 2.76 / 1.75 | 2.58 / 1.06 | No |
| 5,000 | 1 % | 5.66 | 42.75 | 2.83 / 1.80 | 2.61 / 1.06 | No |
| 7,500 | 1 % | 5.61 | 45.64 | 2.97 / 1.87 | 2.60 / 1.07 | No |
| 1,000 | 5 % | 35.37 | 21.45 | 1.92 / 2.01 | 1.08 / 0.97 | No |
| 3,000 | 5 % | 16.48 | 23.01 | 1.14 / 1.15 | 1.09 / 0.99 | No |
| 5,000 | 5 % | 13.03 | 23.79 | 1.10 / 1.07 | 1.10 / 0.99 | No (CN 1.102) |
| 7,500 | 5 % | **8.62** | 25.09 | 1.10 / 1.03 | 1.10 / 0.99 | **Yes** |
| 1,000 | 10 % | 69.10 | 21.58 | 5.86 / 5.98 | 1.00 / 0.97 | No |
| 3,000 | 10 % | 42.30 | 22.27 | 1.86 / 1.93 | 1.02 / 0.99 | No |
| 5,000 | 10 % | 28.75 | 22.67 | 1.28 / 1.32 | 1.02 / 0.99 | No |
| 7,500 | 10 % | 22.64 | 23.26 | 1.10 / 1.12 | 1.02 / 0.99 | No |

**v21, 10,000 sensors, 5 % noise** (`campaign/pilots/dense10000_v21_gpu/RESULTS.md`): PINN 6.77 % vs BE adjoint 26.36 %. The PINN passes both replay screens (CN 1.083, BE 1.030); the adjoint passes BE (0.996) and narrowly fails CN (1.101).

Findings:

- The PINN's error falls steadily with sensor density at every noise level (5 %: 35.4 % at 1,000 sensors to 8.6 % at 7,500).
- The BE adjoint's error does not fall with density; at 1 % noise it rises (32 % to 46 %), and its CN replay fails there while its BE replay passes. The cause is not established by this study; the BE time discretisation (dt/T = 0.01) is one candidate.
- The PINN has a qualified advantage in 1 of 12 cells (7,500 sensors, 5 % noise), plus the 10,000-sensor run. The adjoint has lower error with both replay screens passed in 4 cells.

## Running

These scripts are a frozen record and cannot be re-run from the repository alone: their inputs (`campaign/pilots/*/supplied_data`, `datasets_v2_dense10000`) and the v24 Colab runner (`colab_sweep_runner.py`) are not committed, Each `prepare_*` script also refuses to run when its output folder exists (it preserves the recorded version). Do not run `prepare_dense1000.py`: it regenerates the v22 scripts in this folder.

All of them import from `extended_studies/`, and the pair and evaluate scripts print their command-line help, for example `python run.py study4_dense_sensors/evaluate_noise_density_1000_7500.py --help`. The `validate_*` scripts take no arguments and stop at once without their `supplied_data` folder.

## Limitations

One smooth field, one observation and noise realisation, one optimiser seed; every outer optimisation round hit its iteration cap, so convergence is not established. The regularisation weight gamma changes with noise level and sensor count at fixed alpha. All cases share an initial network fitted to the parent 5 % data. The data were generated by RK4 on a 129 x 129 grid and inverted on 65 x 65. The two methods differ in more than hard vs soft physics (automatic vs finite-difference derivatives, random time pairs vs a fixed time grid). Cells ran on different hosts (Kaggle CPU, Colab T4), so times are descriptive only.
