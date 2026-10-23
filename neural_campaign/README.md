# PINN-based Inverse Estimation of Spatially-Varying Diffusion Coefficients

**Neural-coefficient extension:** see [repository review](REPOSITORY_MAP.md)
for the current experiment inventory and [neural comparison guide](README_neural.md)
for the shared D network, adjoint-NN, PINN, verification and run commands.
The historical status tables below describe an earlier project stage.

An [extended sensor-density study](datasets_v2_density_extended/README.md)
adds 180 paired configurations up to 3,969 sensors. See [CAMPAIGN.md](CAMPAIGN.md)
for the current queue, saved results, and continuation protocol.
### IE 643 Course Project · Team **Binary_Brains** · Jash Taparia (24b2451) · Mihir Gune (24b2443)

**This package: Subtask S2 / Deliverable 1 (synthetic dataset and hand-built forward solver) and Subtask S3 / Deliverable 3 (classical adjoint inversion, first version).**

A plate of heterogeneous material, a handful of noisy thermometers, and an unknown conductivity map to recover. We reconstruct a hidden material property from sparse observations of the process it governs, with two methods that embody opposite views of physics: enforced exactly (classical adjoint, hard constraint) or encouraged through the loss (PINN, soft penalty). This package produces the data both methods are tested on, with the ground truth needed to score them.

---

## 1. The problem

$$\frac{\partial u}{\partial t} = \nabla\cdot\big(D(x,y)\,\nabla u\big) + f(x,y,t)$$

| Symbol | Meaning | In this package |
|---|---|---|
| u(x,y,t) | temperature (or concentration) | simulated, then read at sparse noisy sensors |
| D(x,y) | spatially varying diffusivity, **the unknown** | six ground-truth tiers + variants |
| f(x,y,t) | internal heat source / sink | 0 |

The divergence form is used everywhere. Expanding it gives D∇²u + ∇D·∇u; dropping the second term would remove most of the information about *where* D changes.

| | Forward problem | Inverse problem (ours) |
|---|---|---|
| Given | D, initial condition, boundary conditions, source | sparse noisy samples of u, plus IC and BC |
| Find | the full field u(x,y,t) | the coefficient field D(x,y) |
| Well-posed? | yes: unique and stable | no: non-unique and unstable to noise |
| How solved | one time march | iterative optimisation, hundreds of forward solves |

This package solves the **forward** problem to manufacture data for the **inverse** problem.

## 2. Where this fits in the project

| # | Subtask | Status |
|---|---|---|
| S1 | Literature review and forward solver | done |
| S2 | Synthetic dataset generator (**this package, v2**) | done |
| S3 | Classical adjoint inversion (Taylor test, discrepancy-principle α) | **in progress**: implemented and verified, first results below |
| S4 | PINN inverse model with staged training | planned |
| S5 | Experiment grid; where PINN inversion beats classical | planned |
| S6 | Interface (Streamlit), report, packaged deliverables | planned |

### Changes since the prep presentation
| Prep deck | Now | Why |
|---|---|---|
| Hybrid (PINN warm start → adjoint) in the plan | **Dropped** | Scope decision; frees PINN compute for more seeds |
| Stencil unspecified (9-point was implemented) | **5-point** flux-form stencil | 1.3–1.8× more accurate than 9-point on every tier tested; simpler adjoint |
| Single Crank–Nicolson solver | **RK4, Crank–Nicolson, forward Euler**, one interface | RK4 generates data, CN is the inversion/replay solver, FE is a benchmark |
| Data on the inversion grid | Data on a **129² grid**, inversion on 65² | Avoids the inverse crime |
| Absolute noise σ = 0.01 / 0.05 | **Relative** noise η ∈ {0, 1, 5, 10%} | Absolute σ was 30–280% of the signal |
| 8 random observation times per file | **Fixed** time grids + a temporal-density sweep | Reproducibility; time is now a controlled axis |
| `hash()`-based seeds | `zlib.crc32` seeds + `manifest.json` | `hash()` changes every Python run |
| IC/BC implicit in the solver | Written to `problem/` | Both inversions need them as inputs |

## 3. Contents

| File | Purpose |
|---|---|
| `integrators.py` | 5-point flux-form operator (harmonic-mean links, boundary nodes eliminated so L is symmetric), per-edge Dirichlet BCs (constant or time-varying), and the time integrators `rk4`, `cn`, `fe`, plus `expm` (exact in time) for testing. |
| `domain.py` | `Domain2D` grid class and the D(x,y) tiers (unchanged from v1). |
| `generate_dataset_v2.py` | Benchmark generator: all sweeps, IC/BC problem files, manifests, figures. |
| `compare_integrators.py` | Benchmarks RK4 / CN / FE: order of accuracy, cost, stability limit, Rannacher demo. |
| `check_problem_spec.py` | Rebuilds every problem from `problem/` alone, replays with the true D on the 65² grid with CN, reports error / σ. |
| `adjoint_inversion.py` | **S3:** classical PDE-constrained inversion (discrete adjoint of CN + Rannacher, L-BFGS-B, Tikhonov or TV, discrepancy-principle α), evaluation and figures. |
| `test_adjoint.py` | **S3 verification:** forward consistency, transpose identity, Taylor tests, finite-difference check. |
| `datasets_v2_rk4_zero/` | The generated benchmark (RK4 data, zero boundary values). |
| `results_adjoint/`, `results_adjoint_tv/`, `results_defects_tikhonov/`, `results_defects_tv/` | First inversion results (Tikhonov, TV; tiers A–E and tier F): per-dataset `D_hat.npz`, `metrics.json`, `summary.png`, plus `summary.csv`. |
| `figures/`, `integrator_report.json` | Output of `compare_integrators.py`. |

## 4. Quick start
```bash
pip install numpy scipy matplotlib sympy
python compare_integrators.py                    # ~10 s: verify the time integrators
python generate_dataset_v2.py                    # ~2.5 min: full benchmark (RK4, zero BC)
python check_problem_spec.py datasets_v2_rk4_zero
# options
python generate_dataset_v2.py --integrator cn    # or fe
python generate_dataset_v2.py --bc ramp_left     # zero | heated_left | ramp_left | pulsed_top
python generate_dataset_v2.py --quick            # tier A only, coarse grids (smoke test)
```

## 5. Forward solver

**Space.** Node-based finite volumes on [0,1]². Each link between neighbouring nodes carries the **harmonic mean** 2DᵢDⱼ/(Dᵢ+Dⱼ), because thermal resistances across a face act in series. Boundary nodes are eliminated (their values are known), which makes the operator L symmetric negative definite, so the semi-discrete system is du/dt = Lu + b(t).

**Time.** Measured on tier C at 65², against the exact-in-time `expm` solution on the same grid:

| Integrator | Role | Order (measured) | Step | Note |
|---|---|---|---|---|
| RK4 | **data generation** (default) | 4.03 | stability-limited, ≤ 2.785/ρ(L) | error ~1e-10 at its largest stable step |
| Crank–Nicolson | **inversion and forward replay** | 2.01 | free (default T/400) | best accuracy per second; Rannacher start |
| Forward Euler | **benchmark only** | 1.00 | stability-limited, ≤ 2/ρ(L) | the naive baseline |

- **Stability limit.** At 1.05× the limit, RK4 and FE blow up to ~1e200. Explicit schemes need ~30× more steps than CN here, which is why the inversion solver is implicit.
- **Rannacher start.** The first CN step is replaced by two half-size backward-Euler steps. CN does not damp the stiffest modes; it flips their sign every step. From a point-like initial condition with a large step, plain CN gives u = −0.96 (impossible), and with the Rannacher start u stays ≥ 0. This matters for the classical **adjoint**, which is driven by point sources at the sensors; the adjoint must mirror the same first step.
- **One factorisation.** The CN matrix I − (Δt/2)L is constant in time, so it is factorised once and reused for every step and for the adjoint sweep. (LU via SciPy; the matrix is symmetric positive definite, so a Cholesky factorisation via scikit-sparse would also work.)
- **Does the data integrator matter?** At 129², CN and FE data differ from RK4 by 0.2% and 0.5% of the smallest noise σ. The fine grid, not the integrator, is what avoids the inverse crime.

## 6. Dataset design

### Ground-truth D(x,y)
| Tier | Field | Variants (baseline point only) |
|---|---|---|
| A | smooth sinusoid (sanity check) | |
| B | Gaussian bump (localised feature) | |
| C | circular inclusion, tanh-smoothed edge | contrast 2, 7.5 (default), 15 |
| D | layered / checkerboard (multiple scales) | checkerboard 2×2, 6×6 |
| E | Gaussian random field (spectral filter); correlation length is a continuous difficulty dial | correlation length 0.03, 0.1 (default), 0.3 |
| **F** | **uniform plate with localised defects:** D₀ = 0.25 everywhere except circular spots where D is multiplied by a factor (tanh-smoothed rim). Default: 3 defects, radius 0.07, mixed (two conduct 3× worse, one 3× better) | `single` (1 defect), `many` (6, radius 0.05), `small` (radius 0.035), `low_contrast` (×1.5 / ×0.67), `voids` (all 5× worse) |

Tier F defect positions are seeded and **nested**: every variant shares the same first centres, so variants differ only in the property being varied. The defect list (centre, radius, factor) is stored in each ground-truth file as `defects` (JSON).

D is generated on the 129² data grid. The inversion-grid ground truth is the exact restriction `D_inv = D_data[::2, ::2]`, so tier E is a single random-field realisation on both grids.

### Initial and boundary conditions
- **IC:** two Gaussian bumps (peaks 1.0 and 0.7, widths 0.05 and 0.06), the same for every tier.
- **BC:** Dirichlet on each edge, constant or time-varying. Scenarios: `zero` (default), `heated_left` (left = 1 from t = 0), `ramp_left` (left = t/T), `pulsed_top` (top = ½(1 − cos 6πt/T)). A heated or pulsed edge keeps injecting heat, so late times still carry information.

### Experiment axes (deck slide 15)
| Deck axis | Implementation |
|---|---|
| **Sensor density** (5 → 20 → 50 → 100 → full field) | sensor counts {5, 20, 50, 100, full}; "full" = all 3,969 interior nodes of the 65² grid |
| Spatial layout | `random`, `grid`, `clustered` (3 hot-spots), `boundary` (within 0.1 of an edge) |
| **Temporal sampling** (dense → few snapshots → single final snapshot) | fixed-interval `uniform<n>`, n ∈ {4, 8, 16, 40, 80, 200}; plus `log8` (early-heavy), `early8` (t ≤ T/4), `late8` (t ≥ T/2), `final1` (t = T only) |
| Time horizon | Fourier number Fo = D̄T/L² ∈ {0.02, 0.05, 0.1, 0.2}; T is computed per tier |
| **Noise level** (0 → 1 → 5 → 10%) | η ∈ {0, 1, 5, 10%}, σ = η·RMS of a fixed reference signal |
| **Defects** (inclusions, layers, random patches) | tiers C, D, E, and **tier F** (uniform plate with localised defects: count, size, contrast, type) |
| Outliers | not implemented yet |

### Sweeps (per canonical tier; overlaps written once)
| Sweep | Varies | Fixed |
|---|---|---|
| S1 spatial density × noise | sensors {5, 20, 50, 100, full} × η {0, 1, 5, 10%} | Fo 0.1, `uniform8`, random |
| S2 horizon × time protocol | Fo {0.02, 0.05, 0.1, 0.2} × {`uniform40`, `uniform8`, `log8`, `early8`, `late8`, `final1`} | 50 random sensors, η 5% |
| S3 temporal density × noise | `uniform<n>`, n {4, 8, 16, 40, 80, 200} × η {1, 5, 10%} | Fo 0.1, 50 random sensors |
| S4 spatial layout | {random, grid, clustered, boundary} × {20, 50} sensors | Fo 0.1, `uniform8`, η 5% |

**Totals:** 63 datasets per canonical tier × 6 tiers (A–F), plus one baseline dataset for each of the 11 variants = **389 datasets, 35 ground-truth files.**

### Controls that keep comparisons clean
- **Nested sensors.** For random, clustered and boundary layouts, the n-sensor set is the first n points of one fixed 100-point pool, so 5 → 20 → 50 → 100 only *adds* sensors.
- **One thermometer.** σ is fixed per (tier, horizon, η) from a reference (100 random sensors, `uniform40` times). Changing the count, layout or time protocol never changes the noise level.
- **Paired data.** Sensor positions depend only on (tier, layout), so datasets that differ in noise or time sampling share the same sensors.
- **Deterministic seeds.** `seed_of(...)` uses `zlib.crc32`. Two runs give byte-identical files.

## 7. Output format (`datasets_v2_<integrator>_<bc>/`)
```
problem/<tier>__Fo<Fo>.json      PDE, domain, T, IC expression, per-edge BC expressions in t (sympy-parseable)
problem/<tier>__Fo<Fo>.npz       u0 on both grids; g(t) at every inversion-grid boundary node, all 401 output times
problem/<tier>__Fo<Fo>__ic.csv   x,y,u0 on the inversion grid
sensors/<key>.csv                x,y,t,u,sigma   (one row per reading: the interface input format)
sensors/<key>.npz                same + u_clean, sensor_id, time_index
ground_truth/<tier>__Fo<Fo>.npz  D_data, D_inv, 41 snapshots of u, T, integrator, dt   (ANSWER KEY)
manifest.json                    every dataset: config, seeds, explicit observation times, file paths
manifest.csv                     same, one row per dataset (without time lists)
config.json                      all generator constants
figures/                         per-tier overview, time protocols, sensor layouts
```
Key format: `<tier>__Fo<Fo>__<protocol>__n<count>__<layout>__eta<η>`, e.g. `C_inclusion__Fo0.1__uniform8__n50__random__eta0.05`.

**`problem/` + `sensors/` is everything an inversion may use. `ground_truth/` is for evaluation only.**

## 8. Verification
| Check | Result |
|---|---|
| Time order of accuracy (vs exact `expm`) | RK4 4.03, CN 2.01, FE 1.00 |
| Heated edge vs analytic Laplace steady state | agrees to 4 digits |
| Time-varying boundary values | CN stays second order |
| Problem spec is sufficient (replay from `problem/` with the true D, CN on 65²) | RMS / σ = **1.01** median (IQR 0.98–1.06), n = 296 noisy datasets (tiers A–E) |
| Determinism | two runs → byte-identical CSVs |
| Nested sensor sets | verified for random, clustered, boundary |
| Fixed time grids | every (tier, horizon, protocol) group shares identical times |

The replay result is also the **floor for the displayed metric** (deck slide 16). A perfect D̂ gives RMS_replay/σ ≈ 1, so that is the target for both methods.

## 9. Notes for the inversion code (S3, S4)
- **Inputs.** Read the IC and BC from `problem/`, never from `ground_truth/`. σ is in the `sigma` column of each sensor file.
- **Data term as a sum, not a mean.** The number of readings ranges from 40 to 31,752 across the sweeps. Use (1/σ²)·Σ(u_θ − ũ)² in the PINN (the prep deck writes a 1/N_d mean). Otherwise 50× more data carries the same weight as 1×, and the sensor and time sweeps measure nothing. The classical misfit is already a sum.
- **Forward replay.** Use `integrators.ForwardModel(..., method="cn")` on the 65² grid, the same solver as `check_problem_spec.py`.
- **"full"** datasets have sensors exactly on inversion-grid nodes (grid-aligned). All other layouts are off-grid.

## 10. Known limitations
- **Layered and checkerboard interfaces** are hard thresholds sampled at nodes, so their discrete position depends on the grid. Even the **true** D replays the layered data at up to 4.6σ at η = 1%: discretisation error exceeds the noise there. Smooth these interfaces or exclude η = 1% for these tiers.
- **No manufactured-solution test with variable D yet.** Spatial order for variable D is verified only indirectly (grid convergence against a fine reference). Needs a source term f in `integrators.py`.
- **Outliers, sensor dropout, bias drift** (deck "plus outliers") are not generated yet.
- **Dirichlet boundaries only**; Neumann or Robin boundaries need a different operator.
- **Exact IC.** Knowing u₀ exactly everywhere is an idealisation (a perfect full-field measurement at t = 0).

## 11. Classical adjoint inversion (S3)

### Formulation
$$\min_\theta\; J(\theta)=\frac{1}{2\sigma^2}\lVert Qu(m)-\tilde d\rVert^2+\alpha R(m),\qquad m=P\theta,\; D=e^{m},\quad\text{s.t. } u \text{ solves the discrete heat equation}$$

- **Unknown:** θ = log D on a K×K control grid (default 16×16 = 256 parameters), bilinearly interpolated to the 65² grid (P). D = exp(m) keeps D positive by construction; bounds 1e-3 ≤ D ≤ 10.
- **Hard constraint:** u is never a free variable. Every evaluation runs the CN solver (Rannacher start, Δt = T/200, or T/400 when the observation times require it).
- **Q:** bilinear in space, linear in time; it handles off-grid sensors and off-grid times.
- **Regularisers:** Tikhonov ½∫|∇m|² (smooth fields) or TV ∫√(|∇m|²+ε) (sharp interfaces).
- **Gradient:** discrete adjoint of the exact scheme, Rannacher steps included. One forward + one adjoint solve gives all 256 derivatives. The CN matrix is factorised once per D and reused for both sweeps (Mᵀ = M).
- **α:** discrepancy principle, the same rule the PINN will use. Start at α₀ = N_d, divide by √10 each step with warm starts, and stop at the first α with RMS_data ≤ σ. If the misfit plateaus above σ (model error) the run is flagged `floor_above_sigma`. Noise-free data run until the misfit stops improving (`noise_free`).
- **Initial guess:** the best single constant D, found by a 1-D search. Its error is reported as a baseline.
- **No trained model.** Each dataset is a separate optimisation, and D is the optimisation variable. "Estimate", not "train" or "predict".

### Usage
```bash
python test_adjoint.py                               # ~10 s, run first: all checks must PASS
python adjoint_inversion.py                          # 6 baseline datasets (one per tier), ~4 min
python adjoint_inversion.py --keys <key> [<key> ...]
python adjoint_inversion.py --filter tier=C_inclusion protocol=uniform8 eta=0.01
python adjoint_inversion.py --reg tv --out results_adjoint_tv --keys <key>
python adjoint_inversion.py --filter tier=F_defects   # every tier-F dataset
python adjoint_inversion.py --all                    # all 389 datasets, roughly 4–5 hours
```
Options: `--control K` (parameter grid), `--maxiter` (L-BFGS iterations per α), `--n-steps` (CN steps).
The inversion reads only `sensors/` and `problem/`; `ground_truth/` is opened after the inversion, for scoring.

### Verification (`test_adjoint.py`)
| Test | Result |
|---|---|
| Inversion forward solve = `integrators.ForwardModel` (CN + Rannacher) | max diff 8e-16 |
| ⟨Qx, y⟩ = ⟨x, Qᵀy⟩ | 6e-16 |
| Taylor test, benchmark data, Tikhonov / TV | slope 1.999 / 2.000 |
| Taylor test, time-varying BCs + off-grid times, Tikhonov / TV | slope 1.996 / 1.999 |
| Adjoint vs central finite differences | worst relative diff 1e-8 |

### First results
Metrics: RMS_replay/σ (the displayed metric, data only), relL2(D) = ‖D̂ − D‖/‖D‖, and relL2 split at the median of the sensitivity map S = ∫|∇u|²dt.

**Baselines** (Fo 0.1, `uniform8`, 50 random sensors, η = 5%, Tikhonov):

| Tier | RMS_replay/σ | relL2(D) | constant-D guess | high / low sensitivity | time |
|---|---|---|---|---|---|
| A smooth | 0.92 | 0.10 | 0.38 | 0.07 / 0.11 | 24 s |
| B bump | 1.00 | 0.23 | 0.58 | 0.15 / 0.28 | 20 s |
| C inclusion | 0.98 | 0.66 | 0.75 | 0.65 / 0.68 | 20 s |
| D layered | 0.98 | 0.22 | 0.67 | 0.25 / 0.19 | 48 s |
| E random field | 0.95 | 0.21 | 0.34 | 0.16 / 0.23 | 23 s |

**Sensor density** (C inclusion, η = 1%, Tikhonov): relL2(D) = 0.65 / 0.61 / 0.44 / 0.20 for 5 / 20 / 50 / 100 sensors.

**Tikhonov vs TV:**
| Dataset | Tikhonov | TV |
|---|---|---|
| C inclusion, 50 sensors, η 5% | 0.66 | 0.65 |
| C inclusion, 100 sensors, η 1% | 0.20 | **0.19**, sharp edges, contrast 0.67 vs true 0.60 |
| D layered, 50 sensors, η 5% | **0.22** | 0.34 (L-BFGS hit the iteration cap at every α) |

### Tier F: plate with localised defects
Extra metrics for tier F (`metrics.json`, printed per run):
- **Detected:** the mean recovered log-contrast inside the true spot has the right sign and is at least 25% of the true contrast.
- **Contrast recovery:** recovered / true log-contrast (1 = perfect).
- **Localisation error:** distance from the anomaly centroid to the true centre, in units of the defect radius.
- **Sensitivity:** mean of ∫|∇u|²dt inside the spot ÷ median over the plate.
- **Dice:** overlap of the estimated anomaly map (|log D̂/D̂₀| above half the true contrast) with the true defects.
- **Background error:** relL2 of D away from the defects.

The background D̂₀ is estimated as the median of D̂.

Baseline data (50 random sensors, 8 times, η = 5%):

| Variant | Tikhonov: found / Dice | TV: found / Dice | Note |
|---|---|---|---|
| default (3 defects, r 0.07, ×3) | 3/3 / 0.46 | 3/3 / **0.62** | contrast recovered 30–50% |
| single | 0/1 / 0.00 | 1/1 / 0.20 | |
| many (6, r 0.05) | 3/6 / 0.00 | 3/6 / 0.14 | the 3 missed are in low-flux regions (sensitivity 0.15–0.2× median) |
| small (r 0.035) | 0/3 | 0/3 | below the resolution limit at this noise |
| low contrast (×1.5) | 0/3 | 0/3 | below the noise floor |
| voids (all ×0.2) | 3/3 / 0.55 | 3/3 / **0.86** | |

Default plate with better data (TV):

| Data | found | Dice | contrast recovered | localisation error | background err |
|---|---|---|---|---|---|
| 50 sensors, η 5% | 3/3 | 0.62 | 0.3–0.5 | ~0.2–0.5 r | 0.058 |
| 100 sensors, η 1% | 3/3 | 0.88 | 0.66–0.87 | 0.05–0.09 r | 0.063 |
| full field, η 1% | 3/3 | 0.92 | 0.72–0.80 | 0.03–0.04 r | 0.018 |

- **TV beats Tikhonov on every detectable case.** The field is a flat background with spots, which is exactly the piecewise-constant structure TV favours.
- **Whether a defect is seen depends on heat flow past it.** The worst-recovered defects sit where the sensitivity is 0.15–0.2× the median.
- **Small and low-contrast defects are undetectable at 5% noise** with either regulariser: a resolution and noise limit, not a solver failure.
- **The full-field run stops slightly above σ** (RMS/σ 1.06, flag `floor_above_sigma`): with 31,752 readings at 1% noise, the small 129² → 65² discretisation difference becomes visible.

### Observations (tiers A–E)
- **Every run fits to the noise level** (RMS_replay/σ between 0.92 and 1.00). Against the clean data the replay error is 0.25–0.56σ, so the estimates explain the signal, not the noise.
- **The inclusion at 50 sensors / 5% noise is information-limited, not a solver failure.** The error falls steadily with more sensors and with less noise (noise-free, 100 sensors: 0.32 with Tikhonov).
- **D is invisible where there is no flux.** The sensitivity map has dark spots at the two initial-bump centres (∇u ≈ 0). For the smooth tiers the error is lower in the high-sensitivity half.
- **TV gives sharp interfaces when the data support them,** but the 16×16 bilinear control grid makes the inclusion look square: a representation limit. That is the motivation for the multiscale control grid and the representation ablation.
- **Layered tier:** Tikhonov puts a small spurious spike next to a sensor cluster, and TV under-converges within 150 iterations. Both need follow-up.

### To do (S3)
- Multiscale control grid (8 → 16 → 32) and a convergence check with a higher `--maxiter` for TV.
- Run the full grid (`--all`) for both regularisers to produce the classical reference results.
- Representation ablation: the classical method with a neural-network representation of D (deck slide 8).
- Gauss–Newton Hessian for uncertainty bars on D.

## 12. References
- M. Raissi, P. Perdikaris, G. E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 2019. https://doi.org/10.1016/j.jcp.2018.10.045
- S. Thakur, E. Esmaili, S. Libring, L. Solorio, A. M. Ardekani. Inverse resolution of spatially varying diffusion coefficient using Physics-Informed neural networks. *Physics of Fluids*, 2024. https://doi.org/10.1063/5.0207453
- Z. Zhang, A. Alla, G. E. Karniadakis. Adjoint Method versus Physics-Informed Neural Networks in PDE-Constrained Inverse Problems. *Computer Methods in Applied Mechanics and Engineering*, 2026. https://arxiv.org/abs/2606.12337
