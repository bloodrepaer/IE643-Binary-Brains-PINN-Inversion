# Study 5: does the time integrator change the inversion?

**Question.** Study 4's dense-sensor comparison used a backward-Euler (BE) adjoint with 100 time steps, while every other classical result in the project uses Crank-Nicolson (CN). Was the PINN's dense-sensor win reported at the intensive assessment caused by the dense sensors, or by that solver? This study holds everything else fixed and changes only the time integrator of the classical adjoint.

**Design.**

- **Data (standard, unchanged):** RK4 on the 129 x 129 grid, field A_smooth, Fo 0.1, 8 uniform reading times, zero walls. Nested node subsets from `study1_representation/extend_sensor_density.py` (the same 180-dataset collection as Study 1): **sparse 50** sensors (400 readings), **medium 400** (3,200), **dense 3,969** = every interior node (31,752). Noise 1, 5 and 10 %, one fixed noise draw shared across densities.
- **Method (standard, unchanged):** the project's classical adjoint (`core/adjoint_inversion.py`): 65 x 65 finite-volume operator with harmonic-mean links, 16 x 16 control grid on log D, Tikhonov, best-constant start, L-BFGS-B 150 iterations per alpha, alpha from N_d down by 10^-6 in 13 steps, stop at the first alpha whose replay reaches sigma (discrepancy principle). The true D is read only after the inversion.
- **Only change: the time integrator used inside the inversion**, each with its own exact discrete adjoint (`solvers.py`):
  - **CN**: Crank-Nicolson with Rannacher start, the project default, unchanged;
  - **BE**: backward Euler at every step;
  - **RK4**: classical explicit Runge-Kutta 4.
- **Same step count for every solver within a block:**
  - **Block A**, 4,800 steps: RK4, CN, BE. 4,800 is about the coarsest step at which explicit RK4 is stable for every candidate D <= 1.
  - **Block B**, 100 and 200 steps: CN and BE. 100 is study 4's BE step; 200 is the CN adjoint's usual step for these reading times. RK4 is unstable at these steps.
- **Study-wide D bound [0.001, 1]** for every solver (project default [0.001, 10]; RK4 at D = 10 would need about 43,000 steps per solve). The true D spans 0.05-0.5; no estimate reached the bound (largest value 0.64).
- **Scores:** relL2(D) on the 65 x 65 grid; replay RMS / sigma under the run's own solver; and replay under a **common accurate solver** (CN, 4,800 steps), so every estimate is judged by the same solver.

**Verification** (`test_solvers.py`, all pass): forward errors against the matrix exponential fall at the expected orders (BE 1.01, CN 2.00, RK4 below 1e-12); the RK4 forward matches `integrators.integrate(..., "rk4")` exactly; Taylor tests give slope 2.000 for all three adjoints and finite differences agree to within 2e-6. CN at 200 steps reproduces Study 1's grid-adjoint results (for example 0.0924, 0.0517, 0.0588 vs 0.0924, 0.0517, 0.0589 in `campaign/density_paired_results.csv`).

## Results

63 runs, all complete. Full tables in `RESULTS.md`, one row per run in `results_table.csv`, per-run `metrics.json` and `D_hat.npz` in `results/<block>/<solver>_<steps>/<key>/`.

**relL2(D)** (lower is better):

| Noise / sensors | RK4 4800 | CN 4800 | BE 4800 | CN 200 | BE 200 | CN 100 | BE 100 |
|---|---|---|---|---|---|---|---|
| 1 % / 50 | 0.0924 | 0.0924 | 0.0925 | 0.0924 | 0.1003 | 0.0928 | 0.1328 |
| 1 % / 400 | 0.0517 | 0.0517 | 0.0518 | 0.0516 | 0.0663 | 0.0515 | 0.1015 |
| 1 % / 3,969 | 0.0460 | 0.0460 | 0.0460 | 0.0459 | 0.0546 | 0.0459 | 0.0745 |
| 5 % / 50 | 0.1334 | 0.1334 | 0.1333 | 0.1333 | 0.1343 | 0.1333 | 0.1405 |
| 5 % / 400 | 0.0841 | 0.0841 | 0.0842 | 0.0841 | 0.0890 | 0.0841 | 0.1029 |
| 5 % / 3,969 | 0.0589 | 0.0589 | 0.0589 | 0.0588 | 0.0642 | 0.0588 | 0.0907 |
| 10 % / 50 | 0.1755 | 0.1755 | 0.1755 | 0.1755 | 0.1589 | 0.1753 | 0.1602 |
| 10 % / 400 | 0.1169 | 0.1169 | 0.1170 | 0.1169 | 0.1225 | 0.1170 | 0.1334 |
| 10 % / 3,969 | 0.0709 | 0.0709 | 0.0710 | 0.0709 | 0.0767 | 0.0708 | 0.1002 |

**Replay under the common accurate solver (CN, 4,800 steps), RMS / sigma** (about 1 = data explained to the noise level):

| Noise / sensors | All 4,800-step runs and all CN runs | BE 200 | BE 100 |
|---|---|---|---|
| 1 % / 50, 400, 3,969 | 0.93 - 1.02 | 1.51 - 1.56 | 2.55 - 2.58 |
| 5 % / 50, 400, 3,969 | 0.98 - 1.00 | 1.01 - 1.02 | 1.08 - 1.11 |
| 10 % / 50, 400, 3,969 | 0.99 - 1.00 | 0.95 - 1.01 | 0.97 - 1.02 |

Figures: `fig_error_vs_density.png` (relL2 vs sensors) and `fig_replay_vs_density.png` (common-solver replay vs sensors). In both, the five accurate arms (all 4,800-step runs and both CN step counts) lie on one line.

## Findings

1. **With an accurate time step the integrator does not matter.** At 4,800 steps RK4, CN and BE agree to within 0.00016 in relL2(D) in all 9 cells, all reach the noise level the same way, and all improve with density (at 5 % noise: 0.133, 0.084, 0.059).
2. **CN is already accurate at 100 steps.** CN at 100 and 200 steps matches the 4,800-step result in every cell (largest difference 0.0004).
3. **BE at 100 steps (study 4's setting) is less accurate than CN in 8 of 9 cells, and the gap grows with density.** Ratio BE 100 / CN 100: 1 % noise 1.43, 1.97, 1.62; 5 % noise 1.05, 1.22, 1.54; 10 % noise 0.91, 1.14, 1.41 (sparse, medium, dense). The one exception, 10 % noise with 50 sensors, is a single dataset where BE's estimate happened to be closer (0.160 vs 0.175).
4. **BE's error is a time-discretisation error.** BE estimates fit the data under BE itself (own replay 0.95-1.2 sigma) but replayed through the accurate solver they miss by about 2.6 sigma at 1 % noise at every density. Halving BE's step (100 to 200) cuts the extra error by roughly half or more (BE 200 / CN 200: 1.09, 1.28, 1.19 at 1 % noise) and replay to about 1.5 sigma, as a first-order error should; at 4,800 steps it disappears.
5. **Noise hides the solver error.** At 10 % noise all BE replays pass; at 1 % noise none of the BE 100 / 200 replays do. Whether BE's own error matters depends on its size relative to the noise, not on density alone.

## What this says about study 4 (the intensive-assessment dense-sensor result)

- **The solver contributed.** A coarse BE step makes the adjoint clearly worse, more so with more data, and leaves exactly the signature seen in study 4: replay passes under BE but fails under CN at 1 % noise.
- **The solver does not explain all of study 4.** With the grid representation, the BE 100 adjoint still improves with density (1 % noise: 0.133, 0.102, 0.075) and reaches 9.1 % on the full field at 5 % noise. Study 4's neural BE adjoint was at 21-25 % at 5 % noise and got worse with density at 1 % noise (32 % to 46 %). The remaining gap belongs to study 4's other differences: the neural D-network, the fixed regularisation weight, the capped optimiser budget and the off-node sensor sites.
- **With a proper time step the classical adjoint is the more accurate method on dense data.** CN on the full field (3,969 sensors) reaches 4.6 / 5.9 / 7.1 % at 1 / 5 / 10 % noise, below the PINN's 5.6 / 8.6 / 22.6 % at 7,500 sensors in study 4. Study 4's PINN advantage on dense data therefore holds only against the coarse-step neural BE adjoint, not against the standard classical method.

## Limitations

One smooth field (A_smooth), one noise realisation, nested node sites up to the full 63 x 63 interior; no run beyond 3,969 sensors or with off-node sites like study 4's. RK4 and BE support zero boundary values only. The D bound [0.001, 1] differs from the project default but was never active. This study tests the grid adjoint; it does not re-run study 4's neural adjoint with CN.

## Files and running

| File | What it does |
|---|---|
| `solvers.py` | CN (project default), BE and RK4 adjoint solvers with exact discrete adjoints, sharing everything else with `core/adjoint_inversion.py` |
| `test_solvers.py` | Forward-order, forward-consistency, Taylor and finite-difference checks; prints ALL PASSED |
| `run_study.py` | Runs a block (`--block A` or `B`, optional `--solvers`, `--keys`); finished runs are skipped |
| `summarise.py` | Builds `results_table.csv`, `RESULTS.md` and the two figures from the per-run `metrics.json` |
| `log_*.txt` | Run logs as written |

Run from `extended_studies/`, after generating the data:

```bash
python run.py core/generate_dataset_v2.py                     # datasets_v2_rk4_zero/ (about 4 min)
python run.py study1_representation/extend_sensor_density.py  # datasets_v2_density_extended/
python run.py study5_time_integrators/test_solvers.py         # about 40 s
python run.py study5_time_integrators/run_study.py --block B  # 36 runs, about 25 min
python run.py study5_time_integrators/run_study.py --block A  # 27 runs, about 8 h on one core
python run.py study5_time_integrators/summarise.py
```

Block A run time per run on one CPU core: 7-22 min at 50 or 400 sensors, 16-49 min at 3,969 (the 1 % noise runs take longest because the alpha path goes further). Block B runs take 0.2-1.7 min.

## Why backward Euler biases the inversion (mechanism and causal test)

`be_analysis.py` → `BE_MECHANISM.md`, `be_analysis.json`, `fig_be_mechanism.png`.

- One BE step damps a mode by 1/(1+x) instead of e^-x, so BE decays too slowly. It behaves like the true equation with D replaced by D·ln(1+x)/x, i.e. a smaller D. The inversion compensates by **inflating D**: BE's mean D̂ is above CN's in 9 of 9 cells.
- The temperature error is small in absolute terms (2.9 % of the signal at 100 steps) but large against the noise: 2.5σ at 1 % noise, 0.5σ at 5 %. A 1 % change in D moves u by only 0.8 %, so the error is magnified when it is translated back into D. Because the error depends on the mode, a uniform rescaling cannot absorb it (1.8σ remains), and the inversion invents spatial structure.
- The error also misleads the discrepancy principle: either the misfit floor stays above σ, or α stops weaker.
- **Causal test:** on data generated by BE itself, BE inverts as well as CN (0.093/0.093, 0.053/0.051, 0.047/0.046, 0.060/0.059). The damage comes entirely from the mismatch between the data and the solver used to explain them.
