# IE 643 project report: PINN vs classical inversion of spatially varying diffusivity

Oct 22, 2026 · @Arun

Team Binary\_Brains (Jash Taparia, Mihir Gune). We recover an unknown diffusivity map D(x, y) in the 2D heat equation from sparse, noisy sensors with a physics-informed neural network and with classical PDE-constrained (adjoint) optimisation, on 389 synthetic datasets from our own finite-volume solver.

## Summary

The classical adjoint inversion recovered D more accurately than the PINN on 61 of 64 head-to-head datasets, typically about 2× more accurately. The three exceptions are cases where neither method beats a constant guess. Our re-runs of the two reference papers show this is the expected outcome for a fully specified problem with a grid-based unknown.

- **Data (deliverable 1):** a hand-built finite-volume solver (5-point, harmonic-mean conductances; RK4, Crank–Nicolson and forward Euler, verified at orders 4.03 / 2.01 / 1.00) generated 389 datasets from 6 families of true D, varying sensor count, layout, timing, horizon and noise.
- **PINN (deliverable 2):** two Fourier-feature MLPs for temperature and log D with a physics-informed loss (data, divergence-form PDE residual, boundary, initial condition, Tikhonov or TV), trained with AdamW.
- **Classical baseline (deliverable 3):** reduced-space PDE-constrained optimisation with an exact discrete adjoint of Crank–Nicolson (Taylor-test slopes 1.996–2.000), L-BFGS-B, regularisation weight by the discrepancy principle. Run on all 389 datasets; it fits 324 of them to the noise level (median replay RMS/σ = 0.98).
- **Comparison (deliverables 4–6):** the adjoint had the lower D error on 20/20 representative datasets (median ratio 2.0×) and on 41/44 in a sensor × noise × layout sweep (median relL2 0.36 vs 0.61). The PINN's 3 wins are on the sharp inclusion with too few or boundary-only sensors: it stays at the best-constant level while the adjoint overshoots. Tuning, 4× longer training and changing the data-generating solver did not change the ranking.
- **Literature:** we re-ran Zhang, Alla & Karniadakis (2026) and Thakur et al. (2024). Both support our result where they apply, and both locate the PINN's advantage outside our setting: neural-network unknowns, high-dimensional or long-horizon problems, gridless image data, unknown initial or boundary conditions.
- **Interface:** a Streamlit app to generate, pick or upload data, run both methods, see the displayed metric (replay RMS/σ, plus relL2(D) when the truth is known) and maps, and download or reload trained models.

## Stage 1 — Problem definition and literature

We posed the problem as recovering D(x, y) in u\_t = ∇·(D∇u) on the unit square, with known initial and Dirichlet boundary conditions, from sparse noisy readings of u.

**Choices and the reasons for them.**

| Choice | Reason |
| --- | --- |
| Divergence form ∇·(D∇u), not DΔu | The ∇D·∇u term carries the information about *where* D changes; it is also the form used by Thakur et al. (2024, eq. 3). |
| Unknown is log D | Positivity for free; relative errors treated symmetrically. |
| Dirichlet boundaries, two-Gaussian initial heat | Simple, well-posed forward problem; boundary scenarios with heated or pulsed edges keep late times informative. |
| Six families of true D (smooth → defects) | A difficulty ladder: smooth, localised, sharp interface, multiscale layers, random, and an engineering case (plate with defects). |
| Two inversion paradigms with the same physics and data | The course brief; also the structure of Zhang et al. (2026), who argue comparisons must hold everything but the paradigm fixed. |
| Displayed metric = replay RMS/σ | Works without the true D (as in real use); ≈ 1 means the data are explained to the noise level. |

**Literature used, and for what.**

- *Raissi, Perdikaris & Karniadakis (2019):* the PINN framework — networks trained on data plus PDE residuals computed by automatic differentiation.
- *Thakur et al. (2024):* the closest published work — a two-network PINN for spatially varying D from concentration data. It motivated our PINN design and is the claim we test.
- *Zhang, Alla & Karniadakis (2026):* a controlled adjoint-vs-PINN comparison. It provides the explanation for our results and the guidance on when each method should win.

**Changes since the prep presentation** (from the project README): the planned hybrid PINN→adjoint method was dropped for scope; the 9-point stencil was replaced by a 5-point flux form (1.3–1.8× more accurate); data moved to a 129² grid with inversion on 65² to avoid the inverse crime; absolute noise (30–280% of the signal) became relative noise η ∈ {0, 1, 5, 10%}; random observation times became fixed time grids; Python `hash()` seeds became `zlib.crc32` seeds for reproducibility.

## Stage 2 — Forward solver

The solver discretises space with node-based finite volumes and offers three time integrators behind one interface (`integrators.py`).

**Space.** 5-point stencil; each link between neighbouring nodes carries the harmonic mean 2D\_pD\_q/(D\_p + D\_q), because the two half-cells act as thermal resistances in series. Boundary nodes are eliminated, giving du/dt = Lu + b(t) with L sparse, symmetric and negative definite.

**Time.**

| Integrator | Role | Measured order | Step limit |
| --- | --- | --- | --- |
| RK4 | data generation | 4.03 | Δt ≤ 2.785/ρ(L) |
| Crank–Nicolson + Rannacher start | inversion and replay metric | 2.01 | none (unconditionally stable) |
| Forward Euler | benchmark only | 1.00 | Δt ≤ 2/ρ(L) |

Orders were measured on tier C at 65² against the exact-in-time matrix-exponential solution. At 1.05× their stability limits, RK4 and forward Euler blow up (\~10²⁰⁰), which is why the inversion uses an implicit scheme. The Rannacher start (two backward-Euler half-steps) removes Crank–Nicolson's sign-flipping of stiff modes (without it a point-like start gives u = −0.96); the classical adjoint mirrors these steps exactly. The Crank–Nicolson matrix is factorised once per D and reused for every forward and adjoint step.

**Verification** (`compare_integrators.py`, `check_problem_spec.py`).

- Orders of accuracy as above; Crank–Nicolson stays second order with time-varying boundary values.
- A heated edge matches the analytic Laplace steady state to 4 digits.
- Problem files are sufficient: rebuilding every problem from `problem/` alone and replaying the true D with Crank–Nicolson on 65² gives RMS/σ = 1.01 median (IQR 0.98–1.06, 296 noisy datasets).
- Data generated with Crank–Nicolson or forward Euler instead of RK4 differ by 0.2% and 0.5% of the smallest noise σ: the fine grid, not the integrator, is what keeps the inversion honest.

&#91;image: Measured convergence of RK4, Crank–Nicolson and forward Euler against the exact-in-time solution\]

## Stage 3 — Dataset generation and extraction

One command builds the whole benchmark: `python generate_dataset_v2.py` (about 3 minutes on a laptop), writing 389 datasets and 35 ground-truth simulations to `datasets_v2_rk4_zero/`.

**Steps the generator follows.**

1. Build the true D on the 129² data grid for each family and variant; the inversion-grid truth is every second node.
2. For each time horizon Fo ∈ {0.02, 0.05, 0.1, 0.2}, set T = Fo/D̄ and the boundary scenario.
3. Simulate u(x, y, t) with RK4 on 129² at its largest stable step; store 401 output times and 41 snapshots.
4. Fix the noise level per (family, Fo, η): σ = η · RMS of a reference signal (100 random sensors × 40 times).
5. Choose sensor positions per (family, layout) from a fixed pool of 100, so sets are nested (5 ⊂ 20 ⊂ 50 ⊂ 100).
6. Choose observation times from the protocol (`uniform4` … `uniform200`, `log8`, `early8`, `late8`, `final1`).
7. Interpolate the clean temperature at every (sensor, time), add Gaussian noise with seed `crc32(name)`.
8. Write files and a manifest entry.

&#91;image: Inclusion family: true D with 50 random sensors (left) and how fast the signal decays for each time horizon (right)\]

&#91;image: The four sensor layouts: random, grid, clustered, boundary\]

&#91;image: Observation-time protocols\]

**Sweeps.** S1 sensors {5, 20, 50, 100, full} × noise {0, 1, 5, 10%}; S2 horizon × time protocol; S3 time density × noise; S4 layout × {20, 50} sensors. 63 datasets per main family × 6 + 11 variant baselines = 389.

**Output format.**

```
problem/<family>__Fo<Fo>.json      PDE, domain, T, initial condition, boundary expressions
problem/<family>__Fo<Fo>.npz       initial condition on both grids, boundary values at every output time
problem/<family>__Fo<Fo>__ic.csv   x, y, u0 on the inversion grid
sensors/<key>.csv                  x, y, t, u, sigma (one row per reading; also the interface's upload format)
sensors/<key>.npz                  the same + clean values, sensor ids, time indices
ground_truth/<family>__Fo<Fo>.npz  true D on both grids, 41 snapshots of u (answer key, scoring only)
manifest.json / manifest.csv       every dataset's configuration, seeds and file paths
```

Dataset key: `<family>__Fo<Fo>__<protocol>__n<count>__<layout>__eta<η>`, e.g. `C_inclusion__Fo0.1__uniform8__n50__random__eta0.05`.

**Controls that keep comparisons clean:** nested sensors; one noise level per family and horizon ("one thermometer"); paired data (same sensors across noise levels and time protocols); deterministic seeds (byte-identical reruns); data grid finer than the inversion grid and a different integrator (no inverse crime).

## Stage 4 — Classical adjoint inversion

The classical baseline was run on all 389 datasets; it reaches the noise level on 324, and its accuracy improves with more sensors, less noise and denser time sampling, as theory predicts.

**Method** (`adjoint_inversion.py`).

- **Objective:** J(θ) = (1/2σ²)‖Qu(m) − d‖² + αR(m), m = Pθ (16 × 16 bilinear control grid → 65 × 65), D = eᵐ, bounds 10⁻³ ≤ D ≤ 10.
- **Hard constraint:** u(m) is computed by Crank–Nicolson (Rannacher start) at every evaluation; Q interpolates bilinearly in space and linearly in time, so sensors and times may be off-grid.
- **Regularisers:** Tikhonov ½∫|∇m|² or total variation ∫√(|∇m|² + ε).
- **Gradient:** exact discrete adjoint of the time-stepping scheme — one forward and one backward solve with the same factorised matrix, for all 256 parameters.
- **Optimiser:** L-BFGS-B (≤ 150 iterations per α), continuation α = N\_d · 10^(−k/2), warm-started, from the best constant D.
- **α selection:** discrepancy principle — the first α with RMS misfit ≤ σ; a misfit floor above σ is flagged.

**Verification** (`test_adjoint.py`): forward solve equals the reference integrator to 8e-16; ⟨Qx, y⟩ = ⟨x, Qᵀy⟩ to 6e-16; Taylor-test slopes 1.999 / 2.000 (Tikhonov / TV, benchmark data) and 1.996 / 1.999 (time-varying boundaries, off-grid times); finite-difference agreement 1e-8.

&#91;image: Taylor test: the remainder falls with slope 2, so the adjoint gradient is exact\]

**Results across the sweeps** (relL2(D), Tikhonov, Fo 0.1, 8 uniform times, random sensors unless stated; lower is better).

*Sensor density at 5% noise:*

| Family | 5 | 20 | 50 | 100 | full |
| --- | --- | --- | --- | --- | --- |
| A smooth | 0.319 | 0.128 | 0.103 | 0.110 | 0.077 |
| B bump | 0.394 | 0.289 | 0.229 | 0.189 | 0.075 |
| C inclusion | 0.690 | 0.631 | 0.661 | 0.462 | 0.215 |
| D layered | 0.211 | 0.319 | 0.216 | 0.240 | 0.149 |
| E random field | 0.327 | 0.186 | 0.207 | 0.175 | 0.069 |
| F defects | 0.236 | 0.193 | 0.182 | 0.160 | 0.112 |

*Noise level with 50 sensors:*

| Family | 0% | 1% | 5% | 10% |
| --- | --- | --- | --- | --- |
| A smooth | 0.060 | 0.075 | 0.103 | 0.135 |
| B bump | 0.014 | 0.119 | 0.229 | 0.317 |
| C inclusion | 0.354 | 0.439 | 0.661 | 0.713 |
| D layered | 1.404 † | 0.975 † | 0.216 | 0.300 |
| E random field | 0.056 | 0.116 | 0.207 | 0.253 |
| F defects | 0.113 | 0.118 | 0.182 | 0.201 |

† Failure mode: with little or no noise the discrepancy principle drives α very low, and the layered field's hard interfaces carry a model error above the noise; the inversion fits that error and overshoots. A noise-aware α floor would fix it.

*Sensor layout at 5% noise (20 sensors / 50 sensors):*

| Family | random | grid | clustered | boundary |
| --- | --- | --- | --- | --- |
| A smooth | 0.128 / 0.103 | 0.135 / 0.120 | 0.320 / 0.269 | 0.232 / 0.210 |
| B bump | 0.289 / 0.229 | 0.163 / 0.116 | 0.201 / 0.234 | 0.382 / 0.379 |
| C inclusion | 0.631 / 0.661 | 0.672 / 0.561 | 0.557 / 0.610 | 2.329 ‡ / 0.749 |
| D layered | 0.319 / 0.216 | 0.212 / 0.175 | 0.409 / 0.483 | 0.236 / 0.193 |
| E random field | 0.186 / 0.207 | 0.206 / 0.150 | 0.275 / 0.304 | 0.263 / 0.170 |
| F defects | 0.193 / 0.182 | 0.466 / 0.166 | 0.223 / 0.227 | 0.225 / 0.217 |

‡ 20 boundary sensors never see the central inclusion; the inversion invents a large spurious feature.

*Time sampling (50 sensors, 5% noise):*

| Family | 4 times | 8 | 16 | 40 | 80 | 200 | log8 | early8 | late8 | final only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A smooth | 0.159 | 0.103 | 0.102 | 0.085 | 0.089 | 0.082 | 0.137 | 0.191 | 0.154 | 0.271 |
| B bump | 0.304 | 0.229 | 0.200 | 0.180 | 0.078 | 0.115 | 0.149 | 0.132 | 0.396 | 0.513 |
| C inclusion | 0.590 | 0.661 | 0.562 | 0.399 | 0.363 | 0.371 | 0.462 | 0.565 | 0.638 | 0.760 |
| E random field | 0.266 | 0.207 | 0.185 | 0.143 | 0.138 | 0.093 | 0.201 | 0.228 | 0.245 | 0.343 |
| F defects | 0.236 | 0.182 | 0.143 | 0.106 | 0.100 | 0.105 | 0.127 | 0.124 | 0.233 | 0.295 |

**What the sweeps show.**

- **More data helps, in every dimension.** Errors fall with more sensors, more snapshots and less noise; a single final snapshot is worst for every smooth family, because by then heat has largely drained away.
- **Grid layouts beat random ones for 50 sensors** on 5 of 6 families; clustered and boundary layouts leave parts of the plate unobserved.
- **The inclusion is hardest:** with 50 sensors it barely beats the best constant D (0.66 vs 0.75) because of its sharp edge and the 16 × 16 control grid; full-field data bring it to 0.22.
- **Defects (family F):** TV found 3/3 defects with Dice overlap 0.62 (Tikhonov 0.46) at 50 sensors and 5% noise, and 0.92 with full-field 1%-noise data; small (radius 0.035) and low-contrast (×1.5) defects are below the resolution and noise limits.

## Stage 5 — The PINN

The PINN learns the temperature and log D with two small networks and a five-term physics-informed loss; it reconstructs the broad shape of D but is less accurate than the adjoint on every case.

**Architecture** (`pinn_inversion.py`, class `InverseDiffusionPINN`).

- **u network:** random Fourier features (scale 0.35) → 3 hidden tanh layers of 80 → 1 output (19,521 parameters). The output is an analytic heat-kernel guess for the two initial bumps plus a correction multiplied by 16x(1−x)y(1−y) (zero on the edges) and tanh(8t/T) (zero at t = 0), so the boundary and initial conditions hold by construction.
- **D network:** Fourier features → 3 hidden tanh layers of 64 → log D correction (12,545 parameters), added to a trainable background and tapered to it near the edges; D = exp(clamp(m, −5, 2)).
- **Why tanh:** the loss needs second derivatives; ReLU networks have zero curvature almost everywhere.

**Loss.** 40·data + 20·PDE + 10·boundary + 20·initial + λ·R(log D):

- *data:* mean squared misfit with "Morozov damping" (only 5% of the pull remains once the fit is within 0.95σ²), normalised by the mean squared reading;
- *PDE:* residual u\_t − (D\_x u\_x + D\_y u\_y + DΔu) by automatic differentiation at 768 collocation points per step, time-weighted, with 90% of its gradient routed to the D network and 10% to the u network;
- *R:* Tikhonov (λ = 2·10⁻⁴, smooth families) or TV (3·10⁻⁴, sharp families).

**Training.** AdamW (weight decay 10⁻⁵), learning rates 1.5·10⁻³ (u) and 3.5·10⁻³ (D) with cosine annealing to 10⁻⁵; the D network's rate is held near zero for \~50 epochs and ramped over 15% of training so u fits the data first; gradient clipping at 1.0; 1,500 epochs; best checkpoint from the last 35% kept. About 1–4 minutes per dataset on CPU.

**Tuning: 10 loss-weight settings** on the Gaussian bump (50 sensors, 5% noise; adjoint 0.229):

| Setting | data / PDE weight | regulariser | relL2(D) | network's own χ² | replay χ² |
| --- | --- | --- | --- | --- | --- |
| C01 equal, unregularised | 1 / 1 | none | **0.398** | 0.95 | 3.2 |
| C08 high PDE weight | 40 / 50 | Tikhonov 8e-4 | 0.422 | 0.95 | 4.3 |
| C02 low data weight | 10 / 5 | Tikhonov 1e-3 | 0.429 | 0.95 | 4.3 |
| C05 "sweet spot" | 50 / 25 | Tikhonov 8e-4 | 0.469 | 0.95 | 13.3 |
| C04 (default used elsewhere) | 40 / 20 | Tikhonov 2e-4 | 0.489 | 0.95 | 12.5 |
| C06 TV | 50 / 25 | TV 4e-4 | 0.653 | 0.95 | 55.9 |
| C10 overfit, weak physics | 250 / 2 | none | 0.873 | 0.41 | 281 |

Two lessons: (1) no setting comes within 1.7× of the adjoint; (2) the network's own data fit is ≈ 0.95 for nine very different D estimates (0.40–0.65), so it cannot be used to judge D — only replaying D̂ through the solver can (that χ² ranges from 3 to 56).

**Benchmark (10 cases, setting C04).**

&#91;image: PINN vs adjoint on 10 benchmark cases (left) and versus sensor count for the inclusion at 1% noise (right)\]

The right panel is the clearest weakness: the adjoint's error falls from 0.64 to 0.20 as sensors go from 5 to 100, but the PINN's stays near 0.70. Its data term is a *mean* over the readings, so more sensors do not add weight to the data — the project README had recommended a sum for exactly this reason.

## Stage 6 — Head-to-head experiments

Across 20 representative datasets, longer training, ablations and four data-generating solvers, the adjoint had the lower D error every time.

**20-dataset study.** Both methods on identical data, both D̂ replayed through the same Crank–Nicolson solver; relL2(D), lower is better.

| Dataset | Adjoint | PINN | Best constant | PINN / adjoint |
| --- | --- | --- | --- | --- |
| C inclusion, Fo 0.02, 50 sensors, 5% | 0.604 | 0.623 | 0.743 | 1.03 |
| C inclusion, 50 sensors, 5% | 0.661 | 0.728 | 0.751 | 1.10 |
| C inclusion, 5 sensors, 1% | 0.645 | 0.760 | 0.746 | 1.18 |
| B bump, late times, 50 sensors, 5% | 0.396 | 0.534 | 0.611 | 1.35 |
| C inclusion, 100 sensors, 10% | 0.499 | 0.747 | 0.751 | 1.50 |
| D layered, 20 clustered, 5% | 0.410 | 0.619 | 0.800 | 1.51 |
| B bump, 50 sensors, 5% | 0.229 | 0.400 | 0.584 | 1.75 |
| E random, early times, 50, 5% | 0.228 | 0.444 | 0.327 | 1.95 |
| F defects, 50 sensors, 5% | 0.182 | 0.355 | 0.241 | 1.95 |
| E random, 50 sensors, 10% | 0.253 | 0.500 | 0.339 | 1.98 |
| F defects, 20 boundary, 5% | 0.225 | 0.465 | 0.239 | 2.07 |
| A smooth, final snapshot only | 0.271 | 0.570 | 0.383 | 2.11 |
| E random, 50 sensors, 5% | 0.207 | 0.440 | 0.337 | 2.13 |
| D layered, 50 sensors, 5% | 0.216 | 0.472 | 0.666 | 2.18 |
| E random, 20 sensors, 1% | 0.170 | 0.476 | 0.336 | 2.81 |
| B bump, 100 sensors, 1% | 0.099 | 0.353 | 0.582 | 3.56 |
| A smooth, 50 sensors, 5% | 0.103 | 0.447 | 0.379 | 4.36 |
| E random, 50 sensors, noise-free | 0.055 | 0.419 | 0.338 | 7.59 |
| F defects, Fo 0.2, 40 times, 5% | 0.112 | 0.951 | 0.240 | 8.53 |
| A smooth, full field, 1% | 0.030 | 0.405 | 0.380 | 13.51 |

- **Adjoint wins 20/20;** median ratio 2.0×. It beats the best constant D on 20/20, the PINN on 8/20.
- **The gap is smallest where the problem is hardest for everyone** (sharp inclusion, few sensors, short horizon) and largest where data are rich (full field, noise-free), because the adjoint converts extra data into accuracy and the PINN does not.
- **Run time:** median 46 s (adjoint) vs 125 s (PINN) on CPU.

**Ablations (4 datasets).** 6,000 epochs instead of 1,500: errors improved (smooth 0.447 → 0.343, random field 0.440 → 0.341) but the PINN still lost 4/4. Removing the PINN's edge taper on D made every case worse (e.g. 0.447 → 0.524).

**Solver-bias test.** Could the adjoint win because the data were generated with a solver like its own? We regenerated three families with RK4 (default), Crank–Nicolson and forward Euler on 129², and with the adjoint's exact solver and grid (Crank–Nicolson on 65², a deliberate inverse crime), then ran both methods on each.

| Dataset | Method | RK4 | CN | Forward Euler | Adjoint's own solver |
| --- | --- | --- | --- | --- | --- |
| B bump, 50 sensors, 5% | adjoint | 0.229 | 0.229 | 0.229 | 0.230 |
|  | PINN | 0.388 | 0.417 | 0.404 | 0.411 |
| C inclusion, 50 sensors, 1% | adjoint | 0.439 | 0.439 | 0.439 | 0.443 |
|  | PINN | 0.695 | 0.683 | 0.685 | 0.687 |
| C inclusion, 5 sensors, 1% | adjoint | 0.645 | 0.645 | 0.645 | 0.646 |
|  | PINN | 0.758 | 0.754 | 0.754 | 0.760 |

The readings change by at most 0.05σ between solvers, the adjoint's error by at most 0.004 (even the inverse crime does not help it), and the PINN's only by its run-to-run variation. A coarser time step inside the adjoint (50 instead of 200) also changed nothing (0.227 / 0.436 / 0.644).

**Sensor × noise × layout sweep, both methods (44 datasets).** Two families: the sharp inclusion (C, the hardest) and the smooth bump (B). Both at Fo 0.1 with 8 uniform snapshots, on a grid of 5 / 20 / 50 / 100 random sensors × 0 / 1 / 5 / 10 % noise, plus grid, clustered and boundary layouts at 20 and 50 sensors with 5 % noise. Settings are the same as everywhere else: adjoint with TV for C and Tikhonov for B; PINN with TV 3e-4 for C and Tikhonov 2e-4 for B, 1,500 epochs. Each run used one CPU thread. Cells show **adjoint / PINN** relL2(D); lower is better. The best constant D scores 0.74–0.76 on C and 0.57–0.62 on B.

*Inclusion (C), random sensors*

| sensors | 0 % noise | 1 % | 5 % | 10 % |
| --- | --- | --- | --- | --- |
| 5 | 0.594 / 0.747 | 0.669 / 0.760 | **0.779 / 0.746** | 0.709 / 0.743 |
| 20 | 0.332 / 0.718 | 0.600 / 0.720 | 0.589 / 0.738 | 0.679 / 0.740 |
| 50 | 0.388 / 0.681 | 0.277 / 0.684 | 0.654 / 0.728 | 0.690 / 0.727 |
| 100 | 0.656 / 0.701 | 0.191 / 0.700 | 0.247 / 0.687 | 0.355 / 0.747 |

*Bump (B), random sensors*

| sensors | 0 % noise | 1 % | 5 % | 10 % |
| --- | --- | --- | --- | --- |
| 5 | 0.227 / 0.542 | 0.297 / 0.539 | 0.394 / 0.525 | 0.413 / 0.512 |
| 20 | 0.080 / 0.483 | 0.260 / 0.490 | 0.289 / 0.495 | 0.358 / 0.505 |
| 50 | 0.014 / 0.339 | 0.119 / 0.341 | 0.229 / 0.400 | 0.317 / 0.433 |
| 100 | 0.019 / 0.328 | 0.099 / 0.353 | 0.189 / 0.355 | 0.204 / 0.417 |

*Layouts at 5 % noise*

| layout | C, 20 sensors | C, 50 sensors | B, 20 sensors | B, 50 sensors |
| --- | --- | --- | --- | --- |
| random | 0.589 / 0.738 | 0.654 / 0.728 | 0.289 / 0.495 | 0.229 / 0.400 |
| grid | 0.659 / 0.968 | 0.510 / 0.690 | 0.163 / 0.416 | 0.116 / 0.358 |
| clustered | 0.527 / 0.686 | 0.654 / 0.688 | 0.201 / 0.482 | 0.234 / 0.470 |
| boundary | **1.529 / 0.766** | **0.747 / 0.740** | 0.382 / 0.510 | 0.379 / 0.509 |

Bold cells are the three where the PINN has the lower error.

&#91;image: relL2(D) versus number of sensors for the adjoint (solid) and PINN (dashed) at four noise levels, inclusion (left) and bump (right)\]

**What the sweep shows.**

- The adjoint has the lower error on **41 of 44** datasets. Median relL2(D) is 0.36 for the adjoint, 0.61 for the PINN and 0.68 for the best constant. Median wall time on one thread is 33 s vs 58 s.
- **Bump:** both methods improve with more sensors and less noise. The adjoint is 1.2× better (5 sensors, 10 % noise) to 24× better (50 sensors, no noise).
- **Inclusion:** the PINN stays between 0.68 and 0.76, within 10 % of the best constant, at every sensor count and noise level, so it does not turn more data into a sharper inclusion. The adjoint reaches 0.19–0.25 with 100 sensors at 1–5 % noise. It is erratic below that (0.28–0.78), and on noise-free data with 100 sensors it is held at 0.66 by model error, as with the layered field in Stage 4.
- **The three PINN wins** are all on the inclusion with too little information: 5 random sensors at 5 % noise (0.746 vs 0.779), 20 boundary-only sensors (0.766 vs 1.529) and 50 boundary-only sensors (0.740 vs 0.747). In each, the PINN sits at the best-constant level (about 0.75) and the adjoint is at or above it. The PINN wins by **degrading gracefully**: its smooth network and soft constraint keep D near a constant when the data cannot locate the inclusion. The adjoint instead fits the noise in an under-determined problem and puts structure in the wrong place. Neither method recovers the inclusion in these cases.
- **Layout:** for both methods a grid beats random sensors on the bump (adjoint 0.116 vs 0.229 at 50 sensors) and boundary-only sensors are worst. On the inclusion, a grid helps the adjoint at 50 sensors (0.51), while 20 grid sensors give the PINN its worst score (0.97).

## Stage 7 — Verification of the reference papers

We re-implemented the key experiments of both reference papers; the numbers mostly reproduce, and both papers' conclusions about *when* each method wins are consistent with our result.

**Zhang, Alla & Karniadakis (2026), adjoint vs PINN.** Independent JAX re-implementation (float64, SSBroyden optimiser with strong-Wolfe line search, as in the paper).

| Test | Paper | Our re-run | Verdict |
| --- | --- | --- | --- |
| Burgers, grid unknown, adjoint | 1.549e-3 | 1.55e-3 | reproduced exactly |
| Burgers, grid unknown, PINN | 1.574e-3 | 1.56e-3 | reproduced |
| Burgers, neural unknown, adjoint (400 it.) | 3.09e-4 | 1.83e-4 | reproduced in range; 2,000 iterations reach 4e-7–1e-6 |
| Burgers, neural unknown, PINN | 1.04e-4 | 1.26e-4 | reproduced; **PINN beats adjoint at this budget** |
| Darcy, 0 / 0.1 / 1 / 5 / 10% noise, adjoint | 0.087 / 0.109 / 0.287 / 0.442 / 0.548 | 0.086 / 0.106 / 0.248 / 0.421 / 0.532 | reproduced (paper caps the adjoint at 50 iterations) |
| Darcy 1%, PINN / EKI | 0.270 / 0.258 | 0.254 / 0.230 | reproduced: three methods tie |
| 3D Allen–Cahn (our 33³ reduced grid) | — | adjoint 1.27e-3, PINN 4.6e-3; adjoint gradient cost 0.31 → 2.4 → 25 s at 17³ → 65³ | cost trend reproduced |

Audit findings: the "under half the cost" PINN warm start relies on a stopping rule that uses the ground truth (with the paper's normal rule: \~103% and \~57% of the cold cost, and 160–175% on our runs once the PINN's own time is counted); a "trapped adjoint" remark is contradicted by their own longer runs; the Darcy text says "the same ordering holds" for state errors when it does not; and the Darcy PINN in the code differs from the text (shared trunk, σ = 3).

**Thakur et al. (2024), PINN for spatially varying D.** Re-implemented in JAX on their four D fields (their initial and boundary conditions are not given, so we chose ones that reproduce their Fig. 2), with their metric eq. (7).

| Case | Their PINN | Classical FD + least squares (10 s) | Our adjoint, 2,500 points |
| --- | --- | --- | --- |
| I | 4.8% | 3.3% | 6.3% |
| II | 4.6% | 0.8% | 2.7% |
| III | 6.31% | 1.3% | 5.2% |
| IV | 4.58% | 1.4% | 3.5% |

Further findings: their backward-Euler loss is biased — with a *perfect* concentration field it gives 31% error at Δt = 0.01, falling to 7% only at Δt = 0.001 (continuous residual: 5%); with the paper's architecture the D network collapsed on Case III; eq. (4) has sign errors; Fig. 3(d) does not match its formula; the reported D in 0.5% agarose (9.0e-10 m²/s) exceeds their own free-water value (4.0e-10), which is physically implausible. Their sparse-point results (Table 2: 6.45% at 250 points, 39% at 25) are better than our adjoint on comparable data (8.7%, 62%) — the one PINN advantage we could not rule out.

Full reports: [Zhang re-run and audit](https://claude.ai/artifact/W3sk4Nf8kfGmKnU3UshcoK); [Thakur verification](https://claude.ai/artifact/RSd9pRuF4Fu8Xv2ctcuy8W).

## Deliverable 4 — Where is the PINN better?

In our heat-equation problem the PINN never recovered a D field that the classical method could not. Over 64 head-to-head datasets it had the lower error only 3 times, all on the sharp inclusion with too little information (5 random or 20–50 boundary-only sensors). There its estimate stays at the best-constant level while the adjoint's goes past it. The regimes where PINNs do win in the literature lie outside a fully specified, low-dimensional problem with a grid-based unknown.

**Our evidence.** 0/20 in the head-to-head study and 3/44 in the sensor × noise × layout sweep (Stage 6), all 3 at the best-constant level. 0/10 in the team's benchmark and 0/10 loss-weight settings. The ranking was unchanged under longer training and under four data-generating solvers. Where the PINN wins or comes close (sharp inclusion, few or boundary-only sensors, short horizon), both methods are near the best-constant baseline, so the data, not the method, set the limit. The useful finding is that the PINN **degrades gracefully** there and does not invent structure, whereas the adjoint can overshoot (relL2 1.53 with 20 boundary sensors). A PINN estimate is therefore a safer default when the sensor layout is known to be poor, but not a better reconstruction.

**Why this is expected.** Both methods use the same physics; the adjoint imposes it exactly and searches 256 parameters, while the PINN imposes it as a penalty and must also learn the full temperature field with \~32,000 weights (Stage 5 and the theory guide). Zhang et al. give the same rule: *"Use the adjoint method when a reliable discrete forward solver is available and the unknown is low-dimensional—a scalar, a coefficient, or a coarse field"* (Sec. 4, p. 29).

**Where a PINN is better, with the strength of the evidence.**

| Regime | Evidence | Status |
| --- | --- | --- |
| Unknown represented by a neural network | Zhang Burgers re-run: PINN 1.26e-4 vs adjoint 1.83e-4 at 400 iterations, and less seed-to-seed spread (best/worst ratio 1.3 vs 3.3) | confirmed (at matched small budget) |
| High-dimensional or long-horizon time problems | Zhang 3D Allen–Cahn: PINN 14× cheaper; our re-run shows adjoint gradient cost growing \~10× per grid doubling | confirmed (cost, not accuracy) |
| Sparse, noisy data | Zhang Darcy at ≥ 1% noise: methods statistically tied | confirmed tie |
| Gridless scattered or image data | Thakur Table 2 at 25–250 points beats our adjoint on comparable data | reported, not reproduced on identical data |
| Unknown or uncertain initial / boundary conditions | the adjoint needs them exactly; a PINN can learn them from data | not tested (our PINN hard-codes the initial condition) |

**Conclusion for the deliverable.** The PINN's value in this project is flexibility rather than accuracy: no adjoint derivation, mesh-free inputs, and the potential to handle uncertain initial or boundary data. When a verified solver and exact problem data exist, as here, classical PDE-constrained optimisation is more accurate and faster.

## Stage 8 — Interface: methodology and testing

The interface is a Streamlit web app (`app/streamlit_app.py`) on a thin backend (`app/core.py`) that calls the project's solvers unchanged.

**What the user can do.**

| Tab | Inputs | Outputs |
| --- | --- | --- |
| 1 · Data | generate a dataset (any of 17 D fields or an uploaded D map; Fo; 5–100 sensors or full field; layout; timing; noise; boundary condition; seed), pick a benchmark dataset, or upload a CSV of readings (x, y, t, u\[, σ\]) with optional initial condition and true D | true D with sensors, initial temperature, reading curves, a slider through the simulated field |
| 2 · Run | classical: control grid 8–32, Tikhonov/TV, iterations; PINN: epochs 100–6,000, regulariser, weight, seed | progress, run times |
| 3 · Results | — | displayed metric (replay RMS/σ) and relL2(D) when the truth is known, best-constant baseline, maps on a shared colour scale, log(D̂/D) error maps, profiles, α-selection curve, sensitivity map, PINN loss curves; downloads (models, D̂ as CSV/NPZ, metrics, figures) |
| 4 · Load model | a saved `classical_model.npz` or `pinn_best.pt`, or one of 40 bundled trained models | the rebuilt D map, its scores, CSV export |

**How it was built, and why.**

1. **One data path for everything.** Every source (generated, benchmark, uploaded) is written to the project's own dataset format, so `adjoint_inversion.Dataset` and the PINN's loader read user data exactly as they read the benchmark — no duplicated logic and no special cases.
2. **Generation reuses the project generator** (same D families, RK4 on 129², sensor sampling and relative noise), so interface datasets are statistically identical to the benchmark.
3. **Uploads are validated** (column names; x, y ∈ \[0, 1\]; t > 0); the average D is estimated from the best constant fit to set the PINN's time scale; a custom initial condition triggers a warning because the PINN's base solution assumes the benchmark's.
4. **Identical scoring for both methods:** each D̂ is replayed through the same Crank–Nicolson solver; the PINN's own network fit is shown for information only, with an explanation.
5. **Models are real artefacts:** the classical model is its control-grid vector θ (plus K, regulariser, α); the PINN is its checkpoint. Tab 4 rebuilds D from either file.
6. **Results stay in session state,** so switching tabs never retrains.

**Testing.** Every path was exercised headlessly with `streamlit.testing.v1.AppTest` (benchmark + both methods; generation + classical; upload + PINN; reload of `.pt` and `.npz`; bundled models), then end to end in a real Chromium browser. The bundled models, rescored in the app, reproduce the stored comparison numbers exactly (inclusion: classical 0.661 / RMS-replay 0.98, PINN 0.728 / 2.33).

Run it: `pip install -r requirements.txt` then `streamlit run app/streamlit_app.py`.

## Deliverables checklist and how to reproduce

Everything below is in the submission archive `ie643_final_submission.zip`; paths are relative to its root. The 389 datasets (about 175 MB) are not in the archive because step 2 regenerates them exactly.

| Course requirement | Where | Status |
| --- | --- | --- |
| 1. Synthetic dataset and forward solver | `generate_dataset_v2.py`, `integrators.py`, `domain.py`; checks `compare_integrators.py`, `check_problem_spec.py` | done |
| 2. PINN inverse model | `pinn_inversion.py`; trained checkpoints in `models/pinn/` | done |
| 3. Classical inversion baseline | `adjoint_inversion.py`, `test_adjoint.py`; estimates for all 389 datasets in `models/classical/tikhonov_all389/` | done |
| 4. Where PINN beats classical | this report, "Deliverable 4"; `experiments/` | done (answer: not in this setting; regimes identified from the literature and re-runs) |
| 5. Sparse/noisy sensor experiments | Stage 4 tables (classical, all sweeps); Stage 6 sweep (both methods) | done |
| 6. Quantitative and qualitative evaluation | Stages 4–7; figures in `models/` and `experiments/` and in the interface | done |
| Trained models, code, documentation, dependencies | `models/`, root `*.py`, `README.md`, `requirements.txt` | done |
| Interface with input/upload | `app/` (Streamlit) | done |
| Displayed metric and qualitative results | interface results tab | done |
| Stage-wise report incl. interface methodology | this document | done |

**Reproduce, in order** (Python 3.10+, `pip install -r requirements.txt`; CPU is enough), from the archive root:

1. `python compare_integrators.py` — verify the time integrators (\~10 s).
2. `python generate_dataset_v2.py` — build the 389-dataset benchmark (3–5 min).
3. `python check_problem_spec.py datasets_v2_rk4_zero` — confirm the problem files are sufficient.
4. `python test_adjoint.py` — adjoint gradient checks; all must pass.
5. `python adjoint_inversion.py --all` — classical inversion of every dataset (4–5 h), or `--keys <dataset>` for one.
6. `python pinn_inversion.py --single-run --data-root . --output-dir results_pinn --exp-id <dataset>` — PINN on one dataset (1–4 min); without `--single-run` it runs the 10-case benchmark; `--run-lambda-sweep` runs the 10 loss-weight settings.
7. `streamlit run app/streamlit_app.py` — the interface.

Companion documents: [TA defence guide](https://claude.ai/artifact/AYCXqwfqKuFH9dRfdWKJUC) (claims, evidence, paper quotations) and [Theory guide](https://claude.ai/artifact/QyZhDHpGkX4WSmYtTvWe5u) (every method from first principles, mapped to the code).

## Limitations and future work

**Limitations.**

- One PINN design, trained with Adam in float32 on CPU for 1,500–6,000 epochs; a second-order optimiser in float64 (as in Zhang et al.) or GPU-scale training could narrow the gap.
- The PINN's data term is a mean, so it does not gain from more sensors; a σ-weighted sum was recommended in the README and not yet adopted.
- The PINN's base solution hard-codes the two-bump initial condition, so uncertain or different initial conditions cannot yet be tested.
- Classical noise-free runs on the layered field overshoot (relL2 up to 1.4) because hard interfaces carry model error above the vanishing noise level.
- 2D, square domain, Dirichlet boundaries, no outliers or sensor dropout in the data.
- The Thakur comparison uses reconstructed data (their initial and boundary conditions are not published).

**Future work, in order of expected value.**

1. **Uncertain initial conditions:** give both methods an initial state with 5–10% error or known only at sensors — the regime where a PINN should beat the adjoint.
2. **PINN with a σ-weighted sum data term and quasi-Newton training.**
3. **Hybrid:** use the PINN's D to warm-start the adjoint, as Zhang et al. recommend, and measure the cost honestly (including PINN time).
4. **Multiscale control grid** (8 → 16 → 32) and TV with more iterations for sharp interfaces.
5. **Uncertainty bars** on D from a Gauss–Newton Hessian.

## References

1. M. Raissi, P. Perdikaris, G. E. Karniadakis. [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://doi.org/10.1016/j.jcp.2018.10.045). Journal of Computational Physics 378, 686–707, 2019.
2. S. Thakur, E. Esmaili, S. Libring, L. Solorio, A. M. Ardekani. [Inverse resolution of spatially varying diffusion coefficient using Physics-Informed neural networks](https://arxiv.org/abs/2403.03970). arXiv 2403.03970, 2024; Physics of Fluids, 2024.
3. Z. Zhang, A. Alla, G. E. Karniadakis. [Adjoint Method versus Physics-Informed Neural Networks in PDE-Constrained Inverse Problems](https://arxiv.org/abs/2606.12337). arXiv 2606.12337v2, 2026.
4. M. Tancik et al. Fourier features let networks learn high frequency functions in low dimensional domains. NeurIPS 2020.
5. N. Rahaman et al. On the spectral bias of neural networks. ICML 2019.
6. S. Wang, Y. Teng, P. Perdikaris. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. SIAM Journal on Scientific Computing, 2021.
