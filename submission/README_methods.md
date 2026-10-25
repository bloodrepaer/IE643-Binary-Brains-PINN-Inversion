# Inversion methods: classical adjoint and PINN

Both methods recover D(x, y) from the same inputs: the sensor readings in `sensors/` and the initial and boundary conditions in `problem/` (see `README_datasets.md`). Neither reads `ground_truth/`; it is opened only after the estimate is made, for scoring. The difference is how the physics enters: the adjoint enforces the PDE exactly (hard constraint), the PINN adds it to the loss (soft penalty).

## 1. Classical adjoint inversion (`adjoint_inversion.py`)

$$\min_\theta\; J(\theta)=\frac{1}{2\sigma^2}\lVert Q\,u(m)-\tilde d\rVert^2+\alpha R(m),\qquad m=P\theta,\quad D=e^{m},\quad\text{s.t. } u \text{ solves the discrete heat equation}$$

- **Unknown:** theta = log D on a K x K control grid (default 16 x 16 = 256 parameters), bilinearly interpolated to the 65^2 grid by P. D = exp(m) keeps D positive; bounds 1e-3 <= D <= 10.
- **Hard constraint:** u is never a free variable. Every evaluation runs the Crank-Nicolson solver with a Rannacher start (two backward-Euler half steps first), with dt = T/200, or T/400 or T/800 when the observation times require it.
- **Observation operator Q:** bilinear in space, linear in time, so sensors and reading times need not lie on the grid.
- **Regularisers:** Tikhonov, 1/2 of the integral of |grad m|^2 (smooth fields), or total variation, the integral of sqrt(|grad m|^2 + eps) (sharp interfaces).
- **Gradient:** the exact discrete adjoint of the scheme, Rannacher steps included. One forward and one adjoint solve give all 256 derivatives. The CN matrix is factorised once per D and reused for both sweeps.
- **Choice of alpha (discrepancy principle):** start at alpha = N_d (the number of readings), divide by sqrt(10) at each step with warm starts (13 values at most), and stop at the first alpha whose replay RMS is at or below sigma. If the misfit stops improving above sigma (two steps with less than 1 % improvement), the run is flagged `floor_above_sigma`. Noise-free data have no sigma to stop at, so they run until the misfit stops improving (flag `noise_free`); this is why noise-free runs are slower.
- **Optimiser:** L-BFGS-B, up to 150 iterations per alpha.
- **Initial guess:** the best single constant D, found by a 1-D search. Its error is reported as a baseline.
- **No trained model.** Each dataset is a separate optimisation with D as the variable: an estimate, not a prediction.

### Usage

```bash
python test_adjoint.py                                # about 10 s; run first, all checks must pass
python adjoint_inversion.py                           # the 6 baseline datasets (one per tier)
python adjoint_inversion.py --keys <key> [<key> ...]
python adjoint_inversion.py --filter tier=C_inclusion protocol=uniform8 eta=0.01
python adjoint_inversion.py --reg tv --out results_adjoint_tv --keys <key>
python adjoint_inversion.py --all                     # all 389 datasets, 4-5 hours (run_parallel.py splits it)
```

Options: `--control K` (control grid), `--maxiter` (L-BFGS iterations per alpha), `--n-steps` (CN steps), `--data` (collection), `--out` (output folder). Output: `<out>/<key>/` with `D_hat.npz`, `metrics.json` and `summary.png`, plus `<out>/summary.csv`.

### Verification (`test_adjoint.py`)

| Test | Result |
|---|---|
| Inversion forward solve = `integrators.ForwardModel` (CN + Rannacher) | max difference 8e-16 |
| <Qx, y> = <x, Q^T y> | 6e-16 |
| Taylor test, benchmark data, Tikhonov / TV | slope 1.999 / 2.000 |
| Taylor test, time-varying BCs and off-grid times, Tikhonov / TV | slope 1.996 / 1.999 |
| Adjoint vs central finite differences | worst relative difference 1e-8 |

A Taylor slope of 2 means the gradient is exact; a slope of 1 would mean it is wrong.

## 2. PINN (`pinn_inversion.py`)

All settings are in the `PINNConfig` dataclass and can be changed from the command line.

**Networks.** Two fully connected networks with random Fourier features of the inputs (scale 0.35) followed by tanh layers. tanh is used because the loss needs second derivatives of u.

| Network | Input | Output | Size |
|---|---|---|---|
| u-network | (x, y, t) | temperature u | 4 layers x 80 units |
| D-network | (x, y) | correction to log D, added to a trainable background value | 4 layers x 64 units |

D = exp(log D), clamped to [e^-5, e^2], so it is always positive. With zero Dirichlet walls the known physics is built into the u-network: its output is a heat-kernel solution of the known initial bumps plus a learned correction multiplied by a window that is zero on all four walls and a gate that is zero at t = 0. The boundary and initial conditions therefore hold by construction.

**Loss.**

$$L = 40\,L_{data} + 20\,L_{PDE} + 10\,L_{BC} + 20\,L_{IC} + \lambda\,R(\log D)$$

- **L_PDE:** residual du/dt - div(D grad u), computed by automatic differentiation at 768 collocation points per epoch (60 % in the interior and at earlier times, the rest uniform) plus the sensor locations. 90 % of its gradient trains the D-network with u held fixed and 10 % trains the u-network with D held fixed, so the u-network cannot lower the residual by moving away from the data. The residual is weighted by a factor that grows with time (from 0.25 to 8), because the signal decays as heat spreads and late times would otherwise count for little.
- **L_data:** mean squared misfit on a batch of up to 400 readings, normalised by the mean square of the data. It is damped once the misfit is within 0.95 sigma^2 (discrepancy-style), so the network is not pushed to fit the noise.
- **L_BC, L_IC:** zero-wall misfit on 128 boundary points and initial-condition misfit on 256 points.
- **R:** on log D, Tikhonov (mean |grad log D|^2) or total variation, each plus a small penalty on deviation from the background value; lambda = 2e-4 by default.

The weights (40, 20, 10, 20, 2e-4) come from a sweep of 10 settings (`--run-lambda-sweep`, presets `C01_equal_unreg` to `C10_noise_overfit_weakreg`).

**Training.** AdamW (weight decay 1e-5) with two parameter groups: learning rate 1.5e-3 for the u-network and 3.5e-3 for the D-network. Both decay by a cosine schedule to 1e-5 over 1,500 epochs. The D-network's rate is held at 2 % for the first 50 epochs, so u fits the data first, then ramps up linearly over the next 15 % of training. Gradient norm clipped at 1.0. The checkpoint with the lowest total loss in the last 35 % of training is reloaded for evaluation.

**Settings used in the comparisons:** Tikhonov with lambda = 2e-4 for the smooth families (A, B, E), total variation with lambda = 3e-4 for the sharp ones (C, D, F). The classical method uses Tikhonov for A, B, D, E, F and TV for C.

### Usage

```bash
python pinn_inversion.py --single-run --data-root . --output-dir results_pinn \
    --exp-id B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05 --epochs 1500   # one dataset, 1-4 min on CPU
python pinn_inversion.py --data-root . --output-dir results_pinn                    # 10-case benchmark
python pinn_inversion.py --data-root . --output-dir results_pinn --run-lambda-sweep # the 10 loss-weight settings
```

Useful options: `--reg-type tikhonov|tv|none`, `--lambda-reg`, `--lambda-data`, `--lambda-pde`, `--lambda-bc`, `--lambda-ic`, `--epochs`, `--lr-u`, `--lr-m`, `--n-collocation`, `--sweep-preset`. `--data-root` is the folder that contains `datasets_v2_rk4_zero/`. Output per dataset: `pinn_best.pt` and `pinn_final.pt` (checkpoints), `D_hat.npz`, `loss_history.csv`, `metrics.json` and a summary figure.

## 3. Scoring (same for both methods)

| Metric | Definition | Needs the true D? |
|---|---|---|
| relL2(D) | norm(D_hat - D) / norm(D) on the 65^2 grid; the main ranking metric, lower is better | yes |
| Replay RMS / sigma | D_hat run through the CN solver and compared with the readings; about 1 means the data are explained to the noise level | no |
| Best-constant baseline | relL2 of the best single constant D; a useful method must beat it | yes |

`adjoint_inversion.evaluate` also reports relL2 on the halves of the domain where the sensors are most and least sensitive (sensitivity map S = integral of |grad u|^2 dt) and the RMS error of log D. `app/core.py` (`score`) applies relL2 and replay to any D, from either method or an uploaded file. The `chi2_red_solver` value inside a PINN `metrics.json` comes from the PINN script's own replay; the comparisons in this repository re-score every estimate with the common CN solver.

**Extra metrics for the defect fields (tier F)**, in `metrics.json` (the background D0 is estimated as the median of D_hat):

| Metric | Meaning |
|---|---|
| Detected | the mean recovered log-contrast inside the true defect has the right sign and is at least 25 % of the true contrast |
| Contrast recovery | recovered / true log-contrast (1 = perfect) |
| Localisation error | distance from the anomaly centroid to the true centre, in defect radii |
| Sensitivity | mean of the integral of |grad u|^2 dt inside the defect, divided by its median over the plate |
| Dice | overlap of the estimated anomaly map (abs(log(D_hat / D0)) above half the true contrast) with the true defects |
| Background error | relL2 of D away from the defects |

## 4. Where the results are

| Folder | Contents |
|---|---|
| `models/classical/tikhonov_all389/` | Classical estimate, metrics and figure for all 389 datasets; `summary_all.csv` lists them |
| `models/classical/results_*`, `models/classical/sweep/` | Classical runs with TV, on the defect family, and on the 44-dataset sweep |
| `models/pinn/benchmark/` | PINN checkpoints, estimates, loss histories and metrics for the 10-case benchmark and the loss-weight sweep |
| `models/pinn/sweep/` | PINN runs of the 44-dataset sensor / noise / layout sweep |
| `experiments/zhang_replication_study20/` | The 20-dataset head-to-head study |
| `experiments/sensor_noise_layout_sweep/` | The 44-dataset sweep comparison (`sweep_comparison.csv`, `sweep_tables.md`) |

The main results are summarised in `README.md` (section 5) and in the report.
