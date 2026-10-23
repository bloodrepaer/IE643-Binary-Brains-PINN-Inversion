# IE643 session record: inverse diffusion, neural adjoints and PINNs

**Snapshot date: 22 October 2026, Asia/Calcutta.** Experiment timestamps and operational logs are recorded in UTC unless explicitly stated otherwise.

This is a consolidated, evidence-linked record of the work represented in this workspace and the current session. It is intended to support a research handoff, project discussion, report writing and future continuation. It summarizes the retained history; it does not replace the original protocols, source snapshots, checkpoints, logs, manifests or score tables. Historical files can describe an earlier stage. The latest dated campaign records and completed scoped reports take precedence over stale status text.

## Contents

1. Current outcome and project status
2. Research question and governing problem
3. Forward solver and synthetic-data provenance
4. Dataset collections and experiment accounting
5. Methods and what “temporal adjoint” means
6. Shared neural representation and objectives
7. Independent validation and definition of a qualified advantage
8. Experiment history: original campaign through v24
9. Complete final density/noise results
10. Scientific findings and interpretation limits
11. Literature review and paper-transfer decisions
12. Numerical verification, diagnostics and failure investigations
13. Kaggle, local execution and Colab transfer
14. Operational failures, repairs and retained provenance
15. Artifact and implementation map
16. Current holds, permissions and continuation rules
17. Remaining scientific questions
18. Communication and résumé summary
19. Machine-readable sources and detailed appendices

## 1. Current outcome and project status

The requested **four-density by three-noise comparison is complete**:

- Sensor densities: **1,000, 3,000, 5,000 and 7,500 distinct locations**.
- Noise levels: **1%, 5% and 10%**.
- One smooth reference field, eight observation times, fixed observation/noise construction and optimization seed 42.
- Two methods per cell: temporal-consistency neural PINN and backward-Euler neural adjoint.
- **12 completed paired cells and 24 completed fits**, with no missing cell.
- Each fit completed **1,000 Adam steps and 4,000 SSB iterations** in ten outer rounds of 400 iterations.
- All outer rounds reached their iteration caps; there were no budget timeouts in these 24 fits. Completion does **not** establish convergence.
- Source/input identities, supplied physics, saved coefficient checkpoints and common forward replays were checked before coefficient truth was scored.
- No experiment job or download remains active for this scoped sweep. The scoped recurring monitor was deleted after the final report was completed.
- No additional experiment was launched because the sweep finished.

The main completed artifacts are:

- [Full sweep report](campaign/pilots/noise_density_v24_1000_7500/SWEEP_REPORT.md)
- [24-fit score table](campaign/pilots/noise_density_v24_1000_7500/primary_matched_neural_comparison.csv)
- [12-cell paired summary](campaign/pilots/noise_density_v24_1000_7500/paired_cell_summary.csv)
- [Coverage record](campaign/pilots/noise_density_v24_1000_7500/report_coverage.json)
- [Completion hashes and scope](campaign/pilots/noise_density_v24_1000_7500/sweep_completion.json)

The strongest consistency-qualified PINN coefficient result **inside this primary matrix** is:

> **7,500 sensors, 5% noise:** temporal PINN coefficient error **8.6164%**, compared with **25.0896%** for matched BE neural adjoint. Both methods pass both declared replay screens. This is approximately **65.66% lower coefficient error** for PINN in this single case.

An earlier **10,000-sensor reference pair**, outside the requested primary matrix, also showed a qualified PINN coefficient advantage: **6.7657% versus 26.3639%**. It is reported separately and is not silently added to the 12-cell win counts.

The broader campaign is **not** declared finished. Its frozen accounting remains **856/1,510 completed new fits**, with **654 broad PINN fits gated**, plus 476 historical grid results counted separately. Later research pilots are outside that frozen ledger. Held v20 controls, other fields and interface work were not resumed.

Sources: [state](campaign/state.json), [dated campaign log](CAMPAIGN.md), [experiment protocol](campaign/EXPERIMENT_PROTOCOL.md).

## 2. Research question and governing problem

The project concerns inverse estimation of a static two-dimensional positive diffusion field from sparse, noisy observations of a transient process.

The main equation is the divergence-form diffusion/heat equation:

$$
\frac{\partial u}{\partial t}
=
\nabla\cdot(D(x,y)\nabla u)
$$

with supplied initial conditions, domain dimensions, observation coordinates/times, noise information and boundary conditions. Some hypothesis suites also contain boundary/source misspecification; those are not pooled with correctly supplied-physics cases.

For a spatially variable scalar coefficient,

$$
\nabla\cdot(D\nabla u)
=
D(u_{xx}+u_{yy})+D_xu_x+D_yu_y.
$$

The coefficient derivative terms matter. Replacing the full operator with only a constant-coefficient Laplacian would change the problem.

The unknown is D(x,y), independent of time. Observations are triples (x,y,t) with measured u. A model can interpolate these observations well and still infer an incorrect coefficient. This motivated reporting coefficient error and independently replayed sensor predictions separately.

The initial comparison was classical PDE-constrained inversion versus PINNs. The important later correction in experimental design was to compare:

> **Neural-network coefficient optimized through an adjoint solver versus neural-network coefficient optimized jointly with a PINN state.**

The grid-parameterized adjoint remains a useful secondary baseline, but a neural-adjoint improvement over a grid adjoint is not a PINN win.

The use case stayed transient heat/diffusion inversion. Literature examples were not substituted for the user's problem merely to obtain a favorable PINN result.

## 3. Forward solver and synthetic-data provenance

### 3.1 Numerical infrastructure

The retained implementation includes:

- Harmonic-mean diffusion links, appropriate to face fluxes through heterogeneous media.
- Elimination of supplied Dirichlet boundary nodes.
- A semi-discrete spatial operator and boundary forcing.
- RK4, Crank–Nicolson with Rannacher startup, forward Euler, and exact-in-time references used for numerical checks.
- Classical and neural-coefficient inverse solvers operating on the inversion grid.
- Verification of forward consistency, discrete adjoint identities, gradients and time integrators.

The repository README distinguishes an earlier implemented 9-point stencil from the later **5-point flux-form stencil**. The current session record should not blindly repeat the old résumé description “9-point stencil” as the current canonical solver. The previously pasted résumé also quoted an 8.2e-4 validation error; that single number is not used here as a replacement for the retained numerical verification reports.

The README reports measured temporal orders approximately **4.03 for RK4**, **2.01 for CN** and **1.00 for forward Euler** on its stated benchmark. These are historical verification measurements, not reconstruction errors.

Sources: [README](README.md), [integrator implementation](integrators.py), [integrator report](integrator_report.json), [adjoint verification](test_adjoint.py), [repository map](REPOSITORY_MAP.md).

### 3.2 Important correction: the data were RK4-generated

Earlier conversation incorrectly described the retained synthetic observations as Crank–Nicolson-generated. This was corrected after tracing the actual generator and metadata.

The verified provenance for the relevant retained observations is:

- Source collection: datasets_v2_rk4_zero.
- Fine forward grid: **129 × 129**.
- Original recorded integrator: **RK4**.
- Recorded reference timestep: **3.787878787878788e-05**.
- Recorded number of steps: **9,600**.
- Subsequent sensor-density collections reuse those saved observations; they are not newly generated with CN or BE.

CN is an earlier inverse/replay solver. BE is the matched temporal-adjoint forward solver. The provenance correction changed **no observations, fitted weights, settings or numerical scores**.

The earlier argument that the tested adjoint benefited from using the exact same CN scheme as the generator was therefore based on an incorrect premise and must not be repeated for these runs. Fine-grid RK4 data versus coarser CN/BE inversion can still have discretization mismatch, but this is different from identical generator/inverse solver reuse.

The correction is explicitly retained in [the provenance audit](campaign/pilots/noise_density_v23/data_generator_provenance_correction.json), including the metadata SHA.

### 3.3 Spatial and temporal resolutions remain distinct

The fine-grid observations do not imply the inversion grid was enlarged. The recent dense comparisons retain a **65 × 65 inversion grid**.

That grid has **3,969 interior nodes**. To reach 10,000 distinct sensor sites, additional locations were selected from the existing fine-grid interior. Thus “10,000 sensors” must not be described as 10,000 different interior nodes on the 65-grid.

Fine concentration snapshots were stored with float32 precision in the relevant extension. The retained comparison recorded parent cast agreement, with a maximum rounding difference about **3.715e-9**, negligible compared with the reference noise scale about **0.0011884**. Clean fields were used for dataset construction only; they were excluded from training packages.

## 4. Dataset collections and experiment accounting

### 4.1 Original and broad collections

The main existing benchmark contains **389 configurations** over six canonical coefficient families:

| Family | Description |
|---|---|
| A_smooth | Smooth reference field |
| B_gaussian_bump | Smooth localized Gaussian bump |
| C_inclusion | Inclusion/discontinuous feature |
| D_layered | Layered/checkerboard-type structure |
| E_grf | Gaussian random field |
| F_defects | Localized defects |

Additional hypothesis suites explored sparsity, noise, horizon, unknown boundary/source physics, finer coefficient features, wide initial conditions and imaging-like settings. Their assumptions and controls differ, so they are not all one homogeneous matched benchmark.

The nested density extension added **180 configurations**:

- Six canonical fields.
- Sensor counts **5, 20, 50, 100, 200, 400, 800, 1,600, 3,200, 3,969**.
- Noise **1%, 5%, 10%**.
- Fo=0.1 and eight uniform observation times.
- Nested retained locations/readings within a field/noise group.
- Grid-node layouts analyzed separately from historical off-grid random layouts.

This extended the broad ledger to **662 configurations / 1,510 planned new fits**, reusing 476 historical grid baselines.

### 4.2 Frozen broad campaign accounting

| Ledger item | Count / status |
|---|---|
| Planned new broad fits | 1510 |
| Completed new broad fits | 856 |
| Neural adjoint | 662 |
| New grid adjoint | 186 |
| Frozen original PINN | 8 |
| Remaining frozen broad PINN | 654 |
| Historical grid rows (separate) | 476 |
| Later pilots including v24 | Outside frozen 1,510-fit ledger |

Later pilots, seed confirmations, continuation studies and diagnostics are not casually added to these numbers. The final 24-fit noise/density matrix is a separate scoped research comparison and includes reused results.

The broad work also retained additional observation-seed 43/44 conditional-initialization confirmations, an extra dense neural-adjoint continuation, matched/assumed-zero physics analyses, and post-fit diagnostics. They are useful provenance but not independent population replications of the final sweep.

### 4.3 Why “all results favor adjoint” is too broad

The original completed frozen neural subset contained **eight matched neural pairs**, all with lower coefficient error for neural adjoint. That was a valid description of that completed subset, not every later experiment.

Broad neural-adjoint-versus-grid comparisons also showed representation-specific strengths and failures. For example, the dated broad report described neural-adjoint advantages on many Gaussian-bump/layered cases and weaknesses on dense inclusions and defects. Those are **adjoint-versus-adjoint** comparisons, not PINN successes.

The broad generated findings file is dated earlier than the final scoped report. Its “partial” status and old coverage are preserved rather than rewritten as a complete current summary.

Source: [dated broad findings](campaign/FINDINGS.md), [broad protocol](campaign/EXPERIMENT_PROTOCOL.md).

## 5. Methods and what “temporal adjoint” means

### 5.1 Grid adjoint

The classical baseline parameterizes log D on a coarse control grid, interpolates it to the PDE grid, and exponentiates or otherwise bounds positivity. It uses a numerical forward solve to enforce the state equation, a manual discrete adjoint for gradients, L-BFGS-B optimization and regularization such as Tikhonov or TV.

A discrepancy-based regularization path appears in the broad baseline. Its selected-alpha flag is not proof that the optimizer converged or that the selected result is a physical/statistical floor.

### 5.2 Neural adjoint

The neural adjoint represents D by a neural network. Given candidate network weights:

1. Evaluate D on the inversion grid.
2. Solve the supplied discrete PDE forward.
3. Compare predicted measurements with the observed data and add the regularizer.
4. Use the discrete adjoint to obtain sensitivity with respect to grid log D.
5. Backpropagate that grid sensitivity through the coefficient network using a vector-Jacobian product.

It learns the coefficient network; it does not need a learned temperature network.

The earlier neural adjoint uses CN/Rannacher. The later matched temporal control uses BE. Both are time-dependent forward/inverse computations.

### 5.3 Continuous-residual PINN

The PINN learns both a state u_theta(x,y,t) and a coefficient D_phi(x,y). Its state predictions are compared directly with observations, and automatic differentiation supplies derivatives for a PDE residual. Initial and boundary constraints were first penalized, then increasingly built into the state construction.

A numerical forward solver is **not called to generate the state during PINN training** in these comparisons. Numerical solvers are used after fitting for independent coefficient replay.

The current hard-IC/BC ansatz satisfies specified constraints by construction to numerical tolerance. It is not an exact solution of the variable-coefficient PDE.

### 5.4 Temporal-consistency PINN

This replaces the unit continuous-time training residual with a backward-Euler-shaped temperature difference. The state remains a neural function. Evaluating the network at two times is not the same as numerically marching a PDE solver through those times.

### 5.5 “Temporal adjoint” in this session

This is shorthand for the **neural adjoint using a BE forward solve with a matched temporal step**. It is not a claim that the earlier CN adjoint lacked time dependence.

The recent comparison matches the time-integration rule/step more closely. It still differs intrinsically in state representation, spatial discretization, collocation measure and hard versus soft state enforcement.

## 6. Shared neural representation and objectives

### 6.1 Coefficient representation

The calibrated neural comparisons use:

- Three hidden tanh layers of width **32**.
- **2,241 coefficient-network parameters**.
- Float64 arithmetic.
- A sigmoid-bounded log-diffusivity representation with **D in [0.001, 10]**.
- A sensor-fitted constant coefficient initialization with reproducible hidden features.
- Exact serialized initialization shared within matched pairs, checked by tensor SHA.

The v11 paper-inspired transfer used a different, unbounded exponential coefficient parameterization and must not be conflated with this later calibrated representation.

The v18+ calibrated PINN state uses a **32-wide, three-layer Gaussian hard-IC/zero-BC construction**, coupled to local D, with the complete spatial derivative path. Earlier original states were larger/differently constrained and remain separate versions.

### 6.2 Observation objective and regularizer

The recent matched training objective uses the complete observation set, not a favorable subset. In schematic notation:

$$
L_{\mathrm{PINN}} =
\operatorname{mean}\left[\left(\frac{u_\theta-y}{s}\right)^2\right]
+L_{\mathrm{physics}}
+\gamma R(D_\phi).
$$

The matched neural adjoint has the corresponding full-data objective and discrete regularizer, but its state is obtained from its numerical forward solve.

The fixed-likelihood alpha=1 conversion is recorded as:

$$
\gamma = \frac{2\sigma_w^2}{N_{\mathrm{obs}}\,s^2}.
$$

The solver's sigma_w is the relevant weighted-noise quantity in the implementation. It should not be silently substituted with an unrelated paper normalization.

Consequently gamma changes with observation count and noise level. The final matrix does **not** isolate density or noise while holding every effective regularization strength fixed.

The common state/data scale in the recent smooth reference cases is **0.99610137**. For the 5%-noise reference, gamma is approximately **3.5586223384e-11 at 10,000 sensors** and **3.5586223384e-10 at 1,000 sensors**.

### 6.3 Temporal loss used here

Let tau=t/T and h=0.01. Uniform 4,096-point space/time clouds use q0 with tau in [0,0.99], and q1=q0+h within the horizon.

The temporal defect is:

$$
e_{\mathrm{BE}} =
u_\theta(q_1)-u_\theta(q_0)
-hT\,\nabla\cdot(D_\phi\nabla u_\theta)(q_1).
$$

The training physics loss is mean((e_BE/s)^2).

**The defect is not divided by h.** Its effective physics weighting carries an h-squared factor relative to a derivative residual. Dividing by h would change the experiment.

The scaled continuous-residual control in v18 uses:

$$
e_{\mathrm{scaled}} =
h\left[u_\tau(q_1)-T\nabla\cdot(D\nabla u)(q_1)\right].
$$

This control helps distinguish temporal discretization from much of the effective weighting change. Earlier mixed-time, unit-residual pilots are not one-factor equivalents of the temporal pilot.

The loss field named pde in generic metrics is the declared **scaled training physics defect**, not an unscaled continuous PDE RMS. Independent continuous-time PDE probes are therefore retained.

### 6.4 Matched BE adjoint

The recent BE forward model uses **100 full steps in normalized time**, h=0.01, without Rannacher startup.

In semi-discrete notation, its state step solves a system of the form:

$$
(I-\Delta t L(D))u^{n+1}=u^n+\Delta t\,b^{n+1}.
$$

The earlier CN solver instead uses the trapezoidal time average, with Rannacher startup where specified. Common CN and BE evaluations are both reported for each recovered coefficient; neither is chosen after observing the outcome.

### 6.5 Optimization and resources

The recent full schedule is 1,000 Adam plus ten rounds of 400 SSB iterations. SSB denotes the retained SSBroyden-style optimizer implementation. Curvature work remains CPU-based even when PINN data/residual evaluations use CUDA.

Iteration caps, evaluation counts, wall times, internal budgets, external timeouts and actual devices are part of the result. “Completed” means the declared schedule completed, not that an optimum was established.

## 7. Independent validation and qualified advantages

### 7.1 Required order

The retained protocol separates these stages:

1. Freeze supplied observations, problem specification, initialization and source identity.
2. Check numerical gradients/manufactured operators and short pipeline/packaging behavior before scientific dispatch.
3. Train without coefficient truth.
4. Recover saved results safely and inspect actual fit status/iteration counts.
5. Independently check supplied IC/BC and held-out continuous PDE probes.
6. Reconstruct D from checkpoints, compare saved arrays, verify bounds and shared initialization.
7. Reproduce objective/physics formulas, gamma, observation counts, time protocol and both common forward replays.
8. Write supplied-check validation evidence.
9. Only then access the coefficient answer key for post-fit relative L2 scoring.

No true D is used to train, tune penalties, pick checkpoints, choose a best seed or select a favorable case.

Clean concentration data used to construct synthetic observations are a distinct generation-time use. They are not coefficient-truth-driven inverse tuning, and they are excluded from private training payloads.

### 7.2 Metrics

Coefficient error:

$$
E_D = \frac{\|\widehat D-D_{\mathrm{true}}\|_2}{\|D_{\mathrm{true}}\|_2}.
$$

The tables display 100 E_D percent.

State-network sensor fit measures direct PINN interpolation at observations. Coefficient replay discards that learned state, solves the supplied numerical model with recovered D, and compares with the same observations.

Both **raw replay RMS** and **RMS divided by observation noise sigma** are retained. Noise changes the denominator, so normalized values alone are not a complete noise trend.

The normalized replay diagnostic is not a fitted-degrees-of-freedom-adjusted chi-squared statistic. Values below one do not independently prove overfitting.

### 7.3 Screening rule

The recent matched BE/CN comparison requires:

- A completed matched pair.
- Lower coefficient error than the matched neural method.
- Independent supplied-physics/protocol/checkpoint checks.
- Common CN replay <= **1.1 sigma**.
- Common BE replay <= **1.1 sigma**.

We call a coefficient advantage satisfying these declared screens “consistency-qualified.” This does not prove exact pointwise PDE satisfaction, convergence, identifiability, population superiority or a robust boundary between regimes.

Earlier pilots often used only their declared CN screen; the final two-solver definition must not be retroactively presented as if every earlier comparison had identical controls.

The IC screen in the retained generic validator is relative L2 <=1e-6, and the boundary criterion uses RMS relative to sensor sigma. Typical later hard-IC relative errors are about **2.4924e-7**, with extremely small boundary RMS, while early-time continuous PDE residuals remain nonzero.

### 7.4 Narrow thresholds and incomplete fits

A CN replay of **1.102375** fails a 1.1 limit, even if rounded to 1.10 in a short table. A replay of **1.097229** passes, but its proximity to the threshold must remain visible.

Missing/timeout/partial fits are not automatically method losses. Full pairs from a partial pilot were selected by completion status, not by coefficient truth. A process returning zero is not enough to classify its scientific fit as complete.

## 8. Experiment history: original campaign through v24

### 8.1 Original broad neural experiment and gate

The original broad PINN used a separate 4x64 tanh state, normalized time, sensor/PDE/IC/BC/coefficient regularization terms, an initial 15,000 Adam +750 L-BFGS schedule, 2,048 collocation points and seed42.

The completed plain-state subset could fit observations near the noise level while failing common coefficient replay. Independent supplied-IC checks found large initial-condition errors, approximately **0.6952, 0.7073 and 0.7597** for A/B/E. Broad migration was held, and the 654 unfinished frozen PINNs were not dispatched blindly.

This led to targeted architecture/constraint/optimization diagnostics instead of labeling incomplete broad work as PINN losses.

### 8.2 v1–v10: hard constraints, state representation and diagnostics

| Pilot | Field | Sensors | Variant | IC relative L2 | State fit / sigma | CN replay / sigma |
|---|---|---|---|---|---|---|
| [v1](campaign/pilots/ic_v1_gpu/validation.json) | A_smooth | 50 | retained variant | 9.19e-07 | 0.8151 | 2.7919 |
| [v1](campaign/pilots/ic_v1_gpu/validation.json) | B_gaussian_bump | 50 | retained variant | 9.19e-07 | 0.9895 | 3.0072 |
| [v1](campaign/pilots/ic_v1_gpu/validation.json) | E_grf | 50 | retained variant | 9.19e-07 | 0.8084 | 1.5867 |
| [v2](campaign/pilots/ic_constraints_v2_gpu/validation.json) | A_smooth | 50 | retained variant | 9.19e-07 | 1.5024 | 2.7476 |
| [v2](campaign/pilots/ic_constraints_v2_gpu/validation.json) | B_gaussian_bump | 50 | retained variant | 9.19e-07 | 1.8414 | 2.8301 |
| [v2](campaign/pilots/ic_constraints_v2_gpu/validation.json) | E_grf | 50 | retained variant | 9.19e-07 | 1.1900 | 1.6440 |
| [v3](campaign/pilots/ic_bc_v3_gpu/validation.json) | A_smooth | 50 | retained variant | 2.49e-07 | 0.8336 | 2.8074 |
| [v4](campaign/pilots/ic_adaptive_v4_gpu/validation.json) | A_smooth | 50 | retained variant | 2.49e-07 | 1.0829 | 2.8764 |
| [v5](campaign/pilots/ic_convergence_v5_gpu/validation.json) | A_smooth | 50 | retained variant | 2.49e-07 | 1.0370 | 2.8962 |
| [v6](campaign/pilots/ic_locald_v6_gpu/validation.json) | A_smooth | 50 | retained variant | 2.49e-07 | 0.7537 | 2.6056 |
| [v9](campaign/pilots/dual_v9_gpu/validation.json) | A_smooth | 50 | augmented | 2.49e-07 | 0.8269 | 2.5920 |
| [v9](campaign/pilots/dual_v9_gpu/validation.json) | A_smooth | 50 | control | 2.49e-07 | 0.6489 | 2.6693 |
| [v10](campaign/pilots/density_v10_gpu/validation.json) | A_smooth | 3969 | gaussian_ic_bc_locald | 2.49e-07 | 1.1556 | 3.4066 |
| [v10](campaign/pilots/density_v10_gpu/validation.json) | A_smooth | 50 | gaussian_ic_bc_locald | 2.49e-07 | 0.7582 | 2.4060 |

Interpretation of these versions:

- **v1:** analytic supplied-Gaussian IC construction substantially fixed initial-condition error, but common replay remained poor.
- **v2:** additional supplied-constraint handling did not establish reliable inverse recovery.
- **v3:** hard zero boundary construction reduced boundary violation but did not solve coefficient/state replay inconsistency.
- **v4:** residual/adaptive sampling tested where physics points were placed. Independent collocation gaps were retained.
- **v5:** additional convergence/optimization effort tested whether simple budget extension fixed the issue. Gradient component/cancellation diagnostics were kept as descriptive measurements, not proofs of a unique cause.
- **v6:** a local-coefficient-coupled Gaussian state base included the full D derivative path. It remained an ansatz, not an exact variable-D solution.
- **v7:** fixed an already fitted neural-adjoint D and trained only the state network. Sensors and CN were evaluation-only; coefficient truth was not used. This was a conditional forward-state diagnostic, not a new inverse win.
- **v8:** refreshed collocation and restarted optimizer histories while retaining total work. It changed more than a single sampling factor and therefore did not isolate one unique explanation.
- **v9:** augmented/dual-style physics treatment versus a control; both retained inverse results failed the declared replay screen.
- **v10:** a 50-versus-3,969 sensor comparison using then-current local-D state construction. More observations did not automatically yield a physically consistent inverse fit.

No further 50-sensor pilot was automatically used after the human prioritized dense data.

### 8.3 v11: Zhang-inspired transfer and matched comparator

The project audited/transferred a paper-inspired implementation using separate coefficient/state networks and SSB-style optimization.

The transfer changed parameterization, objective, regularization and other settings relative to the frozen broad baseline. A historical unmatched comparator was therefore not treated as a controlled method win. Both later transfer arms were retained.

The v11 transfer used an unbounded physical log-D/exponential positivity construction, unlike the calibrated bounded representation. The PINN coefficient collapsed toward a near-zero field and replay was very poor. This did not establish that the paper or PINNs are universally invalid; it showed that this disclosed transfer to the heat use case failed.

### 8.4 v12–v16: calibration, sampling, density, collocation and optimizer controls

| Pilot | Sensors | Arm / method | D error | CN replay / sigma | State fit / sigma | Status |
|---|---|---|---|---|---|---|
| [v11](campaign/pilots/zhang_v11_gpu/primary_matched_neural_comparison.csv) | 50 | neural_adjoint_zhang | 38.1248% | 2.725313 | 2.725313 | complete |
| [v11](campaign/pilots/zhang_v11_gpu/primary_matched_neural_comparison.csv) | 50 | neural_pinn_zhang | 100.0000% | 106.102287 | 1.387451 | complete |
| [v12](campaign/pilots/calibrated_v12_gpu/primary_matched_neural_comparison.csv) | 50 | adjoint_nn | 14.3579% | 0.900287 | 0.900287 | complete |
| [v12](campaign/pilots/calibrated_v12_gpu/primary_matched_neural_comparison.csv) | 50 | pinn | 39.1918% | 1.764278 | 1.649157 | complete |
| [v13](campaign/pilots/sampling_v13_gpu/primary_matched_neural_comparison.csv) | 50 | adjoint_control | 14.3579% | 0.900287 | 0.900287 | complete |
| [v13](campaign/pilots/sampling_v13_gpu/primary_matched_neural_comparison.csv) | 50 | random_control | 39.1918% | 1.764278 | 1.649157 | complete |
| [v13](campaign/pilots/sampling_v13_gpu/primary_matched_neural_comparison.csv) | 50 | sobol | 44.0032% | 2.059836 | 1.979246 | complete |
| [v14](campaign/pilots/dense_calibrated_v14_gpu/primary_matched_neural_comparison.csv) | 3969 | adjoint_nn | 20.0081% | 0.980659 | 0.980659 | complete |
| [v14](campaign/pilots/dense_calibrated_v14_gpu/primary_matched_neural_comparison.csv) | 3969 | pinn | 40.3720% | 2.474627 | 2.259556 | complete |
| [v14](campaign/pilots/dense_calibrated_v14_gpu/primary_matched_neural_comparison.csv) | 400 | adjoint_nn | 17.4056% | 0.944732 | 0.944732 | complete |
| [v14](campaign/pilots/dense_calibrated_v14_gpu/primary_matched_neural_comparison.csv) | 400 | pinn | 38.3100% | 2.521991 | 2.266499 | complete |
| [v15](campaign/pilots/dense_resolution_v15_gpu/primary_matched_neural_comparison.csv) | 3969 | adjoint_control | 20.0081% | 0.980659 | 0.980659 | complete |
| [v15](campaign/pilots/dense_resolution_v15_gpu/primary_matched_neural_comparison.csv) | 3969 | pde16384 | 35.7423% | 2.036014 | 2.024558 | complete |
| [v15](campaign/pilots/dense_resolution_v15_gpu/primary_matched_neural_comparison.csv) | 3969 | pde4096_control | 40.3720% | 2.474627 | 2.259556 | complete |
| [v16](campaign/pilots/optimizer_v16_gpu/primary_matched_neural_comparison.csv) | 3969 | lbfgsb_adjoint | 5.2329% | 0.994934 | 0.994934 | complete |
| [v16](campaign/pilots/optimizer_v16_gpu/primary_matched_neural_comparison.csv) | 3969 | lbfgsb_pinn | 37.8740% | 2.944905 | 2.927266 | complete |
| [v16](campaign/pilots/optimizer_v16_gpu/primary_matched_neural_comparison.csv) | 3969 | ssb_adjoint | 20.0081% | 0.980659 | 0.980659 | complete |
| [v16](campaign/pilots/optimizer_v16_gpu/primary_matched_neural_comparison.csv) | 3969 | ssb_pinn | 35.7423% | 2.036014 | 2.024558 | complete |

These tables are from the retained version-specific score files. Relative coefficient errors have been converted from fractions to percentages. Each version's source/protocol and observation key remain separate.

- **v12:** aligned bounded coefficient representation, full observation objective and regularization scaling. This improved the comparator and made later method comparisons more interpretable.
- **v13:** compared random control versus Sobol sampling, retaining the matched neural-adjoint control. Sobol alone did not fix recovery.
- **v14:** dense 400/3,969-sensor comparisons still favored the matched neural adjoint.
- **v15:** increasing PDE points from 4,096 to 16,384 reduced one PINN coefficient error and replay discrepancy, but did not create a qualified advantage.
- **v16:** compared SSBroyden and L-BFGS-B for both neural methods at 3,969 sensors and 16,384 PINN PDE points. The L-BFGS-B neural adjoint achieved **5.2329%**, versus **37.8740%** for its PINN. The SSB control was **20.0081%**, versus **35.7423%** for its PINN.
- Some control trajectories were bitwise repeats under the same observations/seed. They confirm implementation consistency, not independent population replication.
- All these inverse comparisons retained cap/evaluation diagnostics. Raw gradient size, descent probes or frozen swaps do not prove convergence, state infeasibility or a unique explanation.

### 8.5 v17: deferred conditional-state study

v17 prepared a state-only diagnostic with all four retained estimated v16 coefficient fields held fixed. It used estimated D, not true D. Its fixture checks passed, but it was **deferred and not submitted** after the human prioritized temporal-consistency inverse experiments.

It must not be described as a completed experiment or an inverse success.

Source: [v17 metadata](campaign/pilots/conditional_state_v17_gpu/pilot.json).

### 8.6 v18: five dense fields with temporal consistency

The human authorized about five to ten high-density datasets to look for PINN advantages. Five predeclared 3,969-sensor, 5%-noise fields were selected: **A, B, D, E and F**, with eight times and seed42.

Each field had three intended arms:

1. Temporal-consistency PINN.
2. h-scaled continuous-residual PINN control.
3. Neural-adjoint control.

That is **15 research fits**, outside the frozen broad ledger. Exact serialized sensor-fitted coefficient initialization was supplied to local and remote runs.

All ten remote PINN fits completed their declared full schedules, all rounds capped. Local controls had incomplete coverage:

| Field | Local neural-adjoint state |
|---|---|
| A_smooth | Complete |
| B_gaussian_bump | Budget timeout; five of ten rounds retained |
| D_layered | Budget timeout; five of ten rounds retained |
| E_grf | Complete |
| F_defects | Process exit after eight logged rounds; no saved trained checkpoint |

The F exit code was recorded as 1073807364; the cause was not established. B's observed wall time exceeded the nominal budget window; the reason was not verified, and no unsupported suspend/active-duration explanation was asserted.

The strict evaluator correctly refused a full 15-arm comparison with F missing. An explicit partial-case option retained only status-complete A/E comparisons, while coverage remained false. Fourteen available checkpoints were validated before truth.

| Field | Arm | D error | CN replay / sigma | State fit / sigma |
|---|---|---|---|---|
| A_smooth | adjoint_nn | 20.0013% | 0.981319 | 0.981319 |
| A_smooth | scaled_pde_pinn | 24.0500% | 1.325529 | 0.980208 |
| A_smooth | temporal_pinn | 17.2321% | 1.099161 | 0.984240 |
| E_grf | adjoint_nn | 19.9186% | 0.985573 | 0.985573 |
| E_grf | scaled_pde_pinn | 21.3647% | 1.266746 | 0.987654 |
| E_grf | temporal_pinn | 20.5948% | 1.142029 | 0.988206 |

For A_smooth, temporal PINN **17.2321% versus 20.0013%** was the first preliminary coefficient advantage. Its CN replay **1.09916087** barely passed the 1.1 screen; adjoint replay **0.981319** was better. The scaled continuous control did not show the same advantage.

For E_grf, temporal PINN **20.5948% versus 19.9186%** and replay **1.142029** did not establish an advantage. Other temporal PINN CN replays were approximately **1.11119 (B), 2.74005 (D), 1.21935 (F)**, all above 1.1.

v18 was not a full five-case matched success. Different local/remote resources and incomplete control coverage remain part of the record.

Sources: [v18 protocol](campaign/pilots/temporal_v18_gpu/PROTOCOL.md), [partial complete-case table](campaign/pilots/temporal_v18_gpu/partial_complete_case_comparison.csv), [metadata](campaign/pilots/temporal_v18_gpu/pilot.json).

### 8.7 v19: short ten-dataset screen

The human requested quicker small tests across roughly ten datasets. v19 used five fields at 400 and 3,969 sensors, two methods, **20 fresh fits**, on local CPU2 threads.

The short schedule was 50 Adam + one 50-SSB round, with 120s internal/180s external per-arm caps.

- All ten neural adjoints completed the short schedule.
- All ten PINNs timed out after 50 Adam and **zero completed SSB rounds**.
- Last completed Adam checkpoints were restored.
- All ten retained PINNs failed common replay.
- There were **zero eligible completed pairs** and no verified short-screen PINN win.
- This is **not ten converged PINN losses**.
- The screen was not automatically rerun with an arbitrarily changed short budget.

Source: [v19 result summary](campaign/pilots/quick_temporal_v19_local/RESULTS.md).

### 8.8 v20: matching the time integrator fairly

After discussion of CN versus BE, the user explicitly requested a fairer temporal comparison.

v20 prepared five fresh **BE neural-adjoint** controls against all five frozen temporal PINNs from v18. It matched h=0.01 with 100 full BE steps, shared coefficient network/initialization/full observations/regularizer/gamma/seed and full iteration schedule.

Manufactured BE solutions, observation remapping, discrete residuals, field/network gradients, full pipelines and own evaluation fixtures passed before dispatch.

The coordinator was then **human-paused** when the user prioritized the 10,000-sensor experiment. Only the coordinator was suspended; its current A fit was allowed to finish/save, and no next field could launch.

The pause file is authoritative over stale metadata that still says running. The stored PID is historical, not authorization to operate on a current process with that number. Any future human-authorized resume requires host command and process-creation identity checks and an output inventory.

No v20 resumption occurred during the later sweep.

Source: [human pause record](campaign/pilots/matched_temporal_v20_local/human_pause.json).

### 8.9 v21: 10,000-sensor matched temporal reference

The user requested one case at 10,000 sensors, then prioritized it over the other controls.

Data construction:

- One predeclared smooth field, 5% noise, eight times, Fo=0.1, seed42.
- **10,000 distinct sensors / 80,000 readings**.
- Preserved all original **3,969 sensor observations / 31,752 rows** exactly.
- Added **6,031 sites** from existing 129-grid interior locations.
- Inversion grid remained 65.
- Original parent observation-fitted initialization reused identically for both fresh methods.
- Fixed alpha conversion changed gamma with N; no coefficient-truth tuning.

Both methods ran sequentially on one Kaggle host. PINN used CUDA data/residual evaluation plus CPU curvature; BE adjoint used CPU, with two CPU threads.

| Arm | D error | CN replay / sigma | BE replay / sigma | State fit / sigma | Evals | Wall seconds |
|---|---|---|---|---|---|---|
| temporal_pinn | 6.7657% | 1.083139 | 1.030298 | 0.993065 | 7809 | 3225.82 |
| be_adjoint_nn | 26.3639% | 1.100680 | 0.995894 | 0.995894 | 7993 | 1981.24 |

Both full schedules completed, with all ten rounds capped and no timeout. The coefficient-error reduction was approximately **74.34%**. PINN passed both replay screens; BE adjoint passed BE and narrowly failed CN. Adjoint had the lower BE replay.

Independent PINN IC relative error was about **2.4924e-7**, with boundary RMS about **3.0856e-18**. Continuous PDE RMS across early-to-late bins was approximately **0.611337, 0.601279, 0.199489, 0.006874**. This retained nonzero early-time residual is not the small scaled training defect.

This result is a single-field/single-seed capped coefficient advantage, not uniform superiority over all metrics or all fields.

Source: [v21 results](campaign/pilots/dense10000_v21_gpu/RESULTS.md).

### 8.10 v22: 1,000-sensor matched reference

The user next requested 1,000 sensors.

A fixed seed 20261023 selected **1,000 sites from the existing 10,000**, retaining all **8,000 observation strings/noise values** exactly. No forward regeneration or coefficient-truth access was needed to select the subset.

Both methods kept the original v21 initialization, full observation objective, architecture, regularizer, seed and full schedule. Because GPU quota could cover only one bounded GPU job, PINN and adjoint ran on separate Kaggle GPU/CPU hosts; timing is qualified accordingly.

| Arm | D error | CN replay / sigma | BE replay / sigma | State fit / sigma | Evals | Wall seconds |
|---|---|---|---|---|---|---|
| temporal_pinn | 35.3716% | 1.923599 | 2.005177 | 0.936896 | 8008 | 3302.71 |
| be_adjoint_nn | 21.4547% | 1.075980 | 0.967260 | 0.967260 | 7917 | 1791.26 |

Both completed, with all rounds capped and no timeout. This is a negative completed PINN comparison: coefficient error and both replay screens favor the BE adjoint. The PINN state fits observations slightly better directly, which does not rescue coefficient recovery.

The contrast between 1,000 and 10,000 motivated a density trend study, but two points alone did not prove density as the unique cause.

Source: [v22 results](campaign/pilots/dense1000_v22_kaggle/RESULTS.md).

### 8.11 v23: initial noise/density design and revision

The user authorized varying noise and density after v22. The first plan was **1,000 / 3,969 / 10,000 ×1% /5% /10%**, reusing the completed 5% references.

Only two new v23 CPU pairs were actually dispatched before the human revised the density grid:

- 1,000 sensors /1% noise.
- 1,000 sensors /10% noise.

Both completed and were independently evaluated. The unstarted v23 cases03–07 were **superseded, not dispatched**. They must not be mistaken for missing losses or a queue still awaiting execution.

v23's two completed noise cells are reused in the final v24 matrix. They are not rerun merely to change a version label.

Source: [v23 results](campaign/pilots/noise_density_v23/RESULTS.md).

### 8.12 v24: final user-requested sweep

The user revised the density grid to **1,000 /3,000 /5,000 /7,500**, with noise **1% /5% /10%**.

Initial instruction: finish the density curve at fixed 5% noise, then vary noise. The later explicit instruction “Do vary noise as well to check” released the noise cases while density completion continued.

Construction and reuse:

- 1,000/5% reused frozen v22.
- 1,000/1% and1,000/10% reused the two completed v23 pairs.
- Nine new pairs at3,000/5,000/7,500 across three noise levels.
- Thus **18 fresh v24 fits +6 reused fits =24 fits in the final matrix**.
- Nested sites: fixed v22 1,000 plus prefixes of a permutation of the remaining 10,000-site pool, seed20261024.
- 1,000 is contained in3,000, then5,000, then7,500.
- Noise errors are shared across levels/densities, with exact original5% observation retention.
- Parent v21 coefficient warm start is fixed across cells and disclosed.
- No additional field, architecture, penalty or best-seed sweep was introduced.

The user requested a Colab notebook to use GPU. All unstarted v24 cases were first reserved for user Colab, avoiding duplicate Kaggle dispatch. The user later reassigned 7,500/5% and then the remaining noise cases to the agent, while retaining only5,000/5% personally.

Final execution split:

- User Colab: **3,000/5% and5,000/5%**.
- Agent Kaggle CPU: **3,000/1%,3,000/10%,5,000/1%,5,000/10%,7,500/1%,7,500/5%,7,500/10%**.
- Maximum two active Kaggle CPU case jobs.
- Both arms sequential, CPU2 threads per case; PINN/adjoint on the same CPU host within each Kaggle pair.
- All new CPU arms retained12600s internal/12900s external per-arm bounds and a25800s pair loop.
- All24 final scientific fits completed their full schedules; no timeout was converted into a loss.

## 9. Complete final density/noise results

All percentages below are coefficient relative L2 error, not sensor fit or paper-specific variance-normalized error.

| Noise | Sensors | PINN D error | BE adjoint D error | PINN both replay | Adjoint both replay | Qualified PINN advantage |
|---|---|---|---|---|---|---|
| 1% | 1,000 | 9.40% | 32.04% | Fail | Fail | No |
| 1% | 3,000 | 6.12% | 38.76% | Fail | Fail | No |
| 1% | 5,000 | 5.66% | 42.75% | Fail | Fail | No |
| 1% | 7,500 | 5.61% | 45.64% | Fail | Fail | No |
| 5% | 1,000 | 35.37% | 21.45% | Fail | Pass | No |
| 5% | 3,000 | 16.48% | 23.01% | Fail | Pass | No |
| 5% | 5,000 | 13.03% | 23.79% | Fail | Pass | No |
| 5% | 7,500 | 8.62% | 25.09% | Pass | Pass | Yes |
| 10% | 1,000 | 69.10% | 21.58% | Fail | Pass | No |
| 10% | 3,000 | 42.30% | 22.27% | Fail | Pass | No |
| 10% | 5,000 | 28.75% | 22.67% | Fail | Pass | No |
| 10% | 7,500 | 22.64% | 23.26% | Fail | Pass | No |

The rule is fixed at CN replay <=1.1 sigma **and** BE replay <=1.1 sigma. It is not relaxed after scoring.

### 9.1 Main 5%-noise density comparison

| Sensors | PINN error | BE adjoint error | PINN CN / BE | Adjoint CN / BE |
|---|---|---|---|---|
| 1,000 | 35.3716% | 21.4547% | 1.923599 / 2.005177 | 1.075980 / 0.967260 |
| 3,000 | 16.4812% | 23.0091% | 1.135548 / 1.150085 | 1.094925 / 0.991525 |
| 5,000 | 13.0309% | 23.7878% | 1.102375 / 1.068481 | 1.097919 / 0.991492 |
| 7,500 | 8.6164% | 25.0896% | 1.097229 / 1.025240 | 1.098443 / 0.993579 |

At5% noise, PINN improves with density, while the matched BE adjoint's coefficient error increases modestly. Only7,500/5% passes both PINN replay screens.

The two close threshold cases must remain explicit:

- **5,000/5% PINN:** CN1.102375 fails; BE1.068481 passes.
- **7,500/5% PINN:** CN1.097229 passes; BE1.025240 passes.

This small margin does not establish a robust population transition exactly between5,000 and7,500 sites.

### 9.2 Detailed diagnostics and actual resources

| Noise | Sensors | Arm | Sensor fit / sigma | Raw CN RMS | Raw BE RMS | Evals | Training wall seconds |
|---|---|---|---|---|---|---|---|
| 1% | 1,000 | be_adjoint_nn | 0.996359 | 0.000619533232 | 0.000236821039 | 8146 | 1922.06 |
| 1% | 1,000 | temporal_pinn | 0.954899 | 0.000705541775 | 0.000405810513 | 7942 | 4831.30 |
| 1% | 3,000 | be_adjoint_nn | 1.056811 | 0.000613550398 | 0.000251189793 | 7905 | 2024.98 |
| 1% | 3,000 | temporal_pinn | 0.992562 | 0.000656489204 | 0.0004155029 | 7542 | 5111.80 |
| 1% | 5,000 | be_adjoint_nn | 1.063778 | 0.00062031197 | 0.000252845646 | 8118 | 2312.92 |
| 1% | 5,000 | temporal_pinn | 0.994699 | 0.00067380442 | 0.000427075539 | 8075 | 6057.78 |
| 1% | 7,500 | be_adjoint_nn | 1.067648 | 0.00061680843 | 0.000253765464 | 7820 | 1412.08 |
| 1% | 7,500 | temporal_pinn | 0.995125 | 0.000705642129 | 0.000445160795 | 8085 | 6332.96 |
| 5% | 1,000 | be_adjoint_nn | 0.967260 | 0.00127872894 | 0.00114952296 | 7917 | 1791.26 |
| 5% | 1,000 | temporal_pinn | 0.936896 | 0.00228606789 | 0.00238301691 | 8008 | 3302.71 |
| 5% | 3,000 | be_adjoint_nn | 0.991525 | 0.00130124393 | 0.00117836056 | 7840 | 2090.98 |
| 5% | 3,000 | temporal_pinn | 0.986327 | 0.0013495216 | 0.00136679772 | 7752 | 3189.13 |
| 5% | 5,000 | be_adjoint_nn | 0.991492 | 0.00130480272 | 0.00117832179 | 8184 | 2042.80 |
| 5% | 5,000 | temporal_pinn | 0.989691 | 0.0013100985 | 0.00126981705 | 8046 | 3125.75 |
| 5% | 7,500 | be_adjoint_nn | 0.993579 | 0.00130542532 | 0.00118080091 | 8183 | 2133.50 |
| 5% | 7,500 | temporal_pinn | 0.991313 | 0.00130398261 | 0.00121842806 | 7797 | 6176.06 |
| 10% | 1,000 | be_adjoint_nn | 0.974066 | 0.00237991525 | 0.00231522288 | 7736 | 1848.37 |
| 10% | 1,000 | temporal_pinn | 0.885696 | 0.0139280534 | 0.0142038567 | 7741 | 4822.96 |
| 10% | 3,000 | be_adjoint_nn | 0.993371 | 0.00242320758 | 0.00236110942 | 8139 | 2562.79 |
| 10% | 3,000 | temporal_pinn | 0.978571 | 0.00441597336 | 0.00459030511 | 7971 | 6382.96 |
| 10% | 5,000 | be_adjoint_nn | 0.992225 | 0.00242316588 | 0.00235838504 | 7880 | 2171.94 |
| 10% | 5,000 | temporal_pinn | 0.987008 | 0.00303130574 | 0.00313980515 | 8295 | 6352.48 |
| 10% | 7,500 | be_adjoint_nn | 0.992531 | 0.00242283297 | 0.00235911207 | 8012 | 2380.24 |
| 10% | 7,500 | temporal_pinn | 0.990164 | 0.0026103683 | 0.00266648131 | 8095 | 7284.15 |

Training times are descriptive. Colab GPU residual evaluation versus Kaggle CPU execution, host differences, optimizer curvature on CPU and different historical budgets prevent an unqualified speed ranking.

### 9.3 Plots

![Coefficient error versus density at each noise level](campaign/pilots/noise_density_v24_1000_7500/density_trend.png)

![Noise effects on coefficient error, raw BE replay and normalized BE replay](campaign/pilots/noise_density_v24_1000_7500/noise_trend.png)

Both common solvers' raw and normalized scores remain in the CSV/report even though the second figure illustrates BE replay. The raw RMS row is logarithmic; normalized replay has a dashed1.1 limit.

## 10. Scientific findings and interpretation limits

### 10.1 What is supported by the completed sweep

1. **Sensor density matters to the observed PINN coefficient estimates in this setup.** At each of the three tested noise levels, coefficient error decreases across the four tested densities.
2. **Coefficient accuracy alone is insufficient.** At1% noise, all PINNs recover D more accurately than BE adjoint but fail both replay screens.
3. **At5% noise, a dense temporal PINN can beat the matched neural adjoint on coefficient accuracy with acceptable declared replay.** The7,500-sensor case is the primary qualified result.
4. **At10% noise, the adjoint is generally more reliable under these settings.** It has lower coefficient error at1,000/3,000/5,000 sites and passes both replay screens at all four densities.
5. **The7,500/10% PINN is a near miss, not a verified win.** Its coefficient error22.64% is below23.26%, but BE replay1.121848 exceeds1.1.
6. **The temporal loss was useful in some dense cases but did not universally solve inverse estimation.** Earlier negatives and partials remain essential context.
7. **Full schedules were completed, not demonstrated optima.** All outer rounds remained capped.

### 10.2 Count outcomes carefully

Within the12-cell primary matrix:

- PINN has lower coefficient error in **8 cells**, but a consistency-qualified coefficient advantage in **1**.
- Adjoint has lower coefficient error with both replay screens passing in **4 cells**.
- The remaining **7 cells** do not give either method the declared qualified advantage.
- These are descriptive counts over a fixed exploratory matrix, **not population win rates**.

The10,000 reference is outside these counts. The broad eight frozen neural pairs and the versioned pilots are also separate comparisons.

### 10.3 What is not established

The evidence does not establish:

- Universal PINN or adjoint superiority.
- A proof that density alone causes the trend.
- A proof that BE, CN or RK4 mismatch is the unique failure cause.
- Converged optima or a physical/statistical error floor.
- A general “more noise helps adjoint” law.
- A robust exact sensor threshold for PINN superiority.
- A best architecture, penalty, temporal step or observation seed.
- Equal-hardware speed superiority.
- Exact satisfaction of the continuous PDE from a small temporal training defect.
- A complete reproduction of either cited paper.
- Full completion of the gated broad campaign.

### 10.4 Plausible explanations remain hypotheses

Hard enforcement of the discrete PDE is a reasonable explanation for stronger adjoint state/replay consistency. It was discussed as a likely contributor, not established by a controlled causal ablation.

A PINN has more freedom to fit sensor values through its learned state while its coefficient remains inconsistent with a numerical replay. The observed direct-fit/replay gaps support keeping these metrics separate; they do not uniquely identify the optimizer, representation, residual weighting or discretization as the sole cause.

Low-noise BE-adjoint coefficient errors increasing with density are compatible with a mismatch between the inverse discrete model and more informative fine-grid data, but this was not proved. Gamma, the shared warm start, capped optimization, location coverage and model discretization all remain potential contributors.

## 11. Literature review and paper-transfer decisions

### 11.1 Zhang-inspired comparison

The retained v11 metadata cites [the Zhang-labelled paper](https://arxiv.org/html/2606.12337v2). The practical contribution here was to insist on the same neural coefficient representation and to audit disclosed architecture/objective/optimizer choices rather than compare against an unmatched historical baseline.

The transfer to transient heat inversion involved adaptations, including source settings absent from an exact paper specification. The failed v11 transfer does not invalidate the literature generally.

Source: [v11 metadata](campaign/pilots/zhang_v11_gpu/pilot.json).

### 11.2 Thakur et al.

The use-case audit identified:

Thakur, Esmaili, Libring, Solorio and Ardekani (2024), *Inverse resolution of spatially varying diffusion coefficient using physics-informed neural networks*, Physics of Fluids36,081915, DOI10.1063/5.0207453.

- [Manuscript](https://arxiv.org/html/2403.03970v1)
- [Authors' code](https://github.com/ArezooArdekani/spatially_varying_diffusion)

The audited work uses separate concentration/state and coefficient networks and a BE-shaped temporal consistency loss for a spatially variable divergence-form diffusion problem.

The local audit records500×500 numerical simulation mesh,100 times, smooth/sinusoidal/tanh-interface synthetic fields and experimental imaging context. The reported error metric is variance-normalized:

$$
\sqrt{\frac{\operatorname{mean}((\widehat D-D)^2)}
{\operatorname{mean}((D-\operatorname{mean}D)^2)}}.
$$

That is not our relative L2 coefficient error. Its headline6.31% must not be compared directly with our percentages.

The paper's concentration-standard-deviation loss normalization must not be confused with the measurement-noise standard deviation.

Our implementation adapted the loss format; it did **not** reproduce the full paper architecture or training. The audit records eight hidden layers and Adam in the paper, whereas our calibrated comparison retains its own networks, hard constraints, discrete regularizer and Adam/SSB schedule.

A manuscript rendering appeared to contain a sign inconsistency between displayed equations. The response was to inspect actual code/manufactured divergence behavior, not assume the authors' implementation was wrong.

### 11.3 OpenFOAM discussion

The conversation examined whether a synthetic forward solver in the paper implied BE-generated data. That inference is not justified solely because OpenFOAM was used or because the PINN loss resembles BE.

OpenFOAM's time scheme is configured in its numerical-scheme dictionary. Names such as Euler, backward and CrankNicolson refer to different schemes; “backward” is not simply a synonym for first-order BE in that software context.

The session did not establish the authors' exact time-scheme dictionary from a provided case configuration. Therefore this document does not assert their OpenFOAM data were BE-generated. The locally used dataset provenance is independently known to be RK4.

### 11.4 Other relevant literature

The retained audit also considered:

- Li et al. (2022), *Identifying Varying Thermal Diffusivity of Inhomogeneous Materials Based on a Hybrid Physics-Informed Neural Network*, DOI10.1142/S1758825122500272: direct thermal inversion relevance.
- Jiang et al. (2023), *Practical uncertainty quantification for space-dependent inverse heat conduction problem via ensemble physics-informed neural networks*, DOI10.1016/j.icheatmasstransfer.2023.106940: noisy-data, uncertainty and sampling relevance.

These were research directions, not implemented new benchmark substitutions or evidence that a matched neural adjoint must lose.

Source: [dated use-case literature audit](campaign/diagnostics/usecase_literature_20261020.md).

## 12. Numerical verification and failure investigations

Before scientific dispatch, the relevant versions retained:

- Manufactured variable-D divergence checks.
- Discrete sine-mode BE reference solution checks.
- Backward-Euler truncation/control residual checks.
- Observation-time remapping/interpolation checks.
- Adjoint transpose/gradient verification.
- Two-direction network coefficient gradient checks.
- Full observation-count one-Adam/one-SSB pipeline checks.
- Reloaded checkpoint reconstruction of D and objective parts.
- ZIP-versus-expanded-input branch equivalence.
- Exact source/input/initial-tensor SHA identities.
- Supplied-physics/own-pair evaluation fixtures before truth.

Examples of retained maximum directional-gradient errors:

| Test | Recorded maximum error |
|---|---:|
| v18 full-count preflight | approximately2.732e-6 |
| v20 coefficient-gradient checks | approximately1.545e-6 |
| v20 manufactured field-gradient checks | approximately3.32e-7 |
| v20 discrete residual check | below1e-12 |
| v21 full80,000-reading checks | approximately2.983e-8 |
| v22 full8,000-reading checks | approximately3.76e-8 |
| initial v23 checks | approximately4.4346e-8 |
| revised v24 all18-arm checks | approximately9.804e-8 |

These are numerical verification/smoke results, **not scientific method wins**.

Post-fit diagnostics included independent collocation probes, training-point residual recomputation, loss-gradient components/cancellation, local descent and finite-difference probes, frozen coefficient/state swaps, dense inclusion localization and true-D forward consistency checks used only for evaluation.

Conclusions were intentionally limited:

- Training/held-out residual differences depend on the sampling measures.
- A small raw gradient depends on parameterization/scaling.
- Gradient cancellation or a capped step is not a convergence proof.
- Fixed-D forward success is not inverse coefficient recovery.
- A true-D replay diagnostic can reveal forward mismatch, but true D cannot become a training/tuning input.

## 13. Kaggle, local execution and Colab transfer

### 13.1 Private compute scope

The human explicitly authorized ongoing private project source/synthetic data/models/results uploads under mihirgune123, local workers and available free Kaggle compute. This did not authorize credentials, unrelated files, paid compute, public sharing or deletion of results.

Automatic-review and sandbox restrictions were respected. Earlier upload approval questions and later explicit approvals are retained in the dated campaign log; no rejection is treated as permission to bypass controls.

Kaggle datasets/kernels were private. Payloads were allowlisted supplied observations, problem/IC files, shared initial networks and required project modules/manifests—not coefficient answer keys, clean readings, credentials or unrelated files.

### 13.2 GPU quota limitation

Typed quota objects were read using timedelta.total_seconds(), not a conversion that drops days.

Recorded remaining GPU seconds included:

- v18: about40,786.788 before the repaired submission.
- v21: about9,852.411 before the second submission.
- v22: about4,612.684 before split GPU/CPU submission.
- After v22 and before later CPU dispatch: **1,265.156 seconds**, zero reserved, paid scaling disabled.

That remaining allowance was inadequate for a full bounded GPU fit. The response was full-schedule free CPU execution or user-run Colab—not a misleading truncated GPU “comparison” or paid compute.

### 13.3 Colab notebook

The delivered self-contained notebook is [IE643_GPU_Density_Sweep.ipynb](colab/IE643_GPU_Density_Sweep.ipynb), with [instructions](colab/README.md).

It contains frozen source/supplied inputs, no coefficient answer keys, and no Kaggle credential requirement. The user selects a GPU runtime, authorizes Drive persistence and runs one pair at a time.

Key behavior:

- Dependency/version/device checks.
- Input/source hash checks and safe unpacking.
- A small full-count compatibility precheck clearly marked SMOKE_DO_NOT_SCORE.
- PINN CUDA data/residual work, CPU curvature, CPU BE adjoint.
- Completed-arm checkpoints/logs saved to Drive.
- Completed results reused only after identity checks.
- Partial/interrupted attempts preserved; no silent optimizer-state resume or automatic retry.
- Export ZIP with a file SHA manifest.
- A phase gate originally prevented noise work before density review.
- Later ownership reassignments were recorded externally to avoid notebook/Kaggle duplicates.

A killed runtime can lose unsaved in-progress weights; finished-arm persistence is not a promise of per-step optimizer recovery.

### 13.4 User returns

Two explicit user-supplied Colab exports were imported:

| Returned ZIP | Research content | Local integrity record | Archive SHA256 |
|---|---|---|---|
| IE643_Colab_results_20261021T201715_672fef.zip | case02, 3,000/5% | [r672fef](colab_returns/r672fef/recovery_integrity.json) | 7744813d08ba87f4da34b45d71af3a3e9e37617fe17a069d934d405ba8fb740c |
| IE643_Colab_results_20261021T223644_3e8ddc.zip | case05, 5,000/5%; duplicate case02 reused | [r3e8ddc](colab_returns/r3e8ddc/recovery_integrity.json) | 9193d7eb693d347d614776102e8a22c1a23ae17c0eb25368b0c86b6d6b374204 |

The second archive included the earlier3,000/5% result again. That duplicate was **bitwise identical** and was reused without a second evaluation or a second scientific count.

Returned notebook code cells matched the delivered version. Original Colab metadata/data paths were preserved, and independent local validation used a separate pilot with only the canonical local data path changed.

Recorded Colab environment: TeslaT4, Python3.13.15, Torch2.11.0+cu130, NumPy2.1.3, SciPy1.16.3, CPU2threads.

### 13.5 Recent kernel identities

| Scope | Private canonical kernel identity |
|---|---|
| v18 dense five-field | mihirgune123/ie643-temporal-consistency-five-dense-fields, version2 |
| v21 ten-thousand reference | mihirgune123/ie643-dense10000-matched-pair, version2 |
| v22 one-thousand PINN | mihirgune123/ie643-dense1000-temporal-pinn, version1 |
| v22 one-thousand adjoint | mihirgune123/ie643-dense1000-be-adjoint, version1 |
| old v23 dispatched noise cases | mihirgune123/ie643-noise-density-v23-case-01 and-02, version1 |
| v24 agent cases | mihirgune123/ie643-noise-density-v24-case-01/-03/-04/-06/-07/-08/-09, version1 |

Only the stated agent cases were dispatched. v24case02/case05 were user Colab work, not hidden duplicate Kaggle fits.

## 14. Operational failures and reviewed repairs

All important failures were retained rather than erased to improve the narrative.

| Incident | Handling and interpretation |
|---|---|
| Kaggle auto-expanded input ZIP | Loader repaired to accept ZIP or expanded directories; branch equivalence checked before retry. Pretraining failure is not a method loss. |
| Title-derived kernel slug | Actual returned canonical slug was recorded instead of assuming the initial requested name. |
| v21 Windows manifest separators on Linux | Version1 failed before training; portable path normalization with containment/duplicate checks was validated before version2. Scientific modules/data unchanged. |
| Initial OpenMP import conflict | MKL threading environment set early before NumPy/Torch; no unsafe duplicate-runtime bypass. |
| Relative-path dataset upload cache issue | Failed pre-upload attempt retained; absolute project path used for the unchanged authorized input. |
| Missing packaged dependency in an early transfer smoke | Dependency list repaired with the unchanged required module before scientific dispatch. |
| v19 NumPy-bool JSON serialization | Native-bool serialization repair; initial failure retained, no scientific objective change. |
| Windows long extracted paths | Original archive/failed partial extraction retained; a shorter project path was used and manifests verified. |
| LF-text SHA versus CRLF file-byte SHA | Exact normalized template equality was checked; actual staged byte SHA retained. Not an unreviewed scientific change. |
| v18 local budgets/process exit | Actual statuses/steps/checkpoints preserved; no blind retry or missing-as-loss classification. |
| Asynchronous final download ordering | Initial extract attempt preceded download completion; same session awaited, no duplicate download; initial empty validation preserved and proper validation rerun after recovery. |
| Legacy CSV schema mismatch | Missing v22 eta/key/sigma/raw-RMS fields filled only from frozen protocol/metrics and validated normalized scores; no refit or solver rerun. First aggregation failure evidence retained. |
| Shared-axis plot autoscale clipping | Detected while reviewing final scientific plots; fixed with a global y-range and regenerated. No score changes. |
| Disk pressure |128MiB guard, selective lossless NTFS compression and authorized SHA-verified duplicate-input cleanup; trained results were not deleted. |

A loader or serialization repair is not evidence about inverse-method quality. Conversely, a good scientific result does not justify bypassing a capacity or approval rejection.

## 15. Artifact and implementation map

### 15.1 Core implementation

| Area | Main files |
|---|---|
| Domain/coefficient families | [domain.py](domain.py) |
| Dataset generation | [generate_dataset_v2.py](generate_dataset_v2.py) |
| Numerical integrators | [integrators.py](integrators.py), [compare_integrators.py](compare_integrators.py) |
| Classical inverse/adjoint | [adjoint_inversion.py](adjoint_inversion.py), [test_adjoint.py](test_adjoint.py) |
| Neural coefficient | [neural_diffusivity.py](neural_diffusivity.py) |
| Neural adjoint | [adjoint_nn.py](adjoint_nn.py) |
| Shared neural comparison | [compare_neural.py](compare_neural.py) |
| Paper-inspired PINN implementations | [pinn_papers.py](pinn_papers.py), [pinn_zhang_transfer.py](pinn_zhang_transfer.py) |
| Calibrated comparison | [calibrated_neural_pair.py](calibrated_neural_pair.py) |
| Temporal comparison | [temporal_density_pair.py](temporal_density_pair.py) |
| BE adjoint | [matched_temporal_adjoint.py](matched_temporal_adjoint.py) |
| Dense extensions | [extend_sensor_density.py](extend_sensor_density.py), [prepare_dense1000.py](prepare_dense1000.py) |
| Revised sweep preparation | [prepare_noise_density_1000_7500.py](prepare_noise_density_1000_7500.py) |
| Revised sweep trainer/kernel | [noise_density_1000_7500_pair.py](noise_density_1000_7500_pair.py), [noise_density_1000_7500_kernel.py](noise_density_1000_7500_kernel.py) |
| Independent IC/BC/PDE evaluator | [evaluate_ic_pilot.py](evaluate_ic_pilot.py) |
| Revised sweep own validator/evaluator | [validate_noise_density_1000_7500.py](validate_noise_density_1000_7500.py), [evaluate_noise_density_1000_7500.py](evaluate_noise_density_1000_7500.py) |
| Final aggregation only | [build_noise_density_sweep_report.py](build_noise_density_sweep_report.py) |
| Colab orchestration | [runner](colab/colab_sweep_runner.py), [builder](colab/build_colab_notebook.py) |

Source snapshots inside old payloads are independent retained versions. Root edits do not silently update historical uploads.

### 15.2 Records to read first

- [CAMPAIGN.md](CAMPAIGN.md): dated decisions, approvals, launches, failures and corrections.
- [campaign/state.json](campaign/state.json): current accounting and scoped status.
- [EXPERIMENT_PROTOCOL.md](campaign/EXPERIMENT_PROTOCOL.md): methodology, not a single current results table.
- [SWEEP_REPORT.md](campaign/pilots/noise_density_v24_1000_7500/SWEEP_REPORT.md): latest complete scoped scientific report.
- [FINDINGS.md](campaign/FINDINGS.md): earlier generated broad analysis, dated and explicitly partial.
- [STAGE_REPORT.md](campaign/STAGE_REPORT.md): broader report snapshot; do not substitute it for newer scoped results.
- Per-pilot metadata, validation, score tables and source hashes.

### 15.3 Result directory conventions

Expected trained artifacts are model.pt, D_hat.npz and metrics.json per key/arm/method. Supporting records include validation.json, pair_integrity.json, validation_before_truth.json, pde_time_bins.csv, environment/attempt logs and extraction manifests.

Short extraction roots campaign/r24cXX_v1 avoid Windows path limits while preserving original downloads under the owning case. Colab returns are under colab_returns/r672fef and colab_returns/r3e8ddc.

The original and partial score tables remain retained; final outputs do not silently replace failed attempts or historical protocols.

## 16. Current holds, permissions and continuation rules

### 16.1 State at this snapshot

- Requested12-cell sweep and scientific report: **complete**.
- Active scoped jobs/downloads/queue: **none**.
- Recurring automation continue-ie643-neural-inversion-experiments: **deleted after scoped completion**.
- Broad856/1510 ledger: unchanged.
- Remaining654 frozen broad PINNs: **gated**, not losses.
- v20 coordinator: human-held by stored pause record; no automatic resume.
- v17: deferred/not submitted.
- Interface enhancements, visual UI QA and interface-focused documentation: held.
- Idle localhost server: preserved, not stopped/restarted as part of the scientific sweep.
- No fresh paid/public work or result deletion.

The old PID in a pause file may no longer identify the same process. This document does not assert that the historical process is still alive or authorize resuming it by number.

### 16.2 Allowed scope is not a mandate for new experiments

Standing private/free project authorization does not mean every unstarted hypothesis is automatically approved. Latest human priorities and explicit holds determine the active scope.

No automatic fields/seeds/architecture/penalty/point-count expansion or retry of unfavorable partials is implied by the completed sweep.

If future work is explicitly authorized:

1. Read the latest campaign/state/protocol and relevant pilot/hold files.
2. Inventory real jobs/processes and result coverage before dispatch.
3. Preserve source/input versions and outcomes.
4. Verify free quota and capacity; never use paid scaling by assumption.
5. Freeze case choices, seeds, objective, limits and scoring before fitting.
6. Keep truth out of training/settings/checkpoint selection.
7. Validate supplied physics and common replays before post-fit coefficient scoring.
8. Report continuous metrics, statuses, caps and actual resources.
9. Keep original, modified and diagnostic studies separate.

## 17. Remaining scientific questions

These are discussion directions, **not newly dispatched tasks**:

- Does the dense5%-noise advantage persist across independent observation realizations and optimization seeds?
- Does it generalize beyond A_smooth to bumps, layers, random fields, inclusions and defects?
- How sensitive are narrow1.1-threshold results to forward grid/time refinement and noise calibration?
- Does a reasoned, predeclared longer optimization budget change outcomes when all current runs are capped?
- What is the contribution of likelihood/gamma scaling versus density/noise itself?
- How much does the parent5%-data warm start influence the later low/high-noise cells?
- Can state-only conditional diagnostics separate state optimization from coefficient recovery, while remaining estimated-D diagnostics rather than inverse wins?
- Would a stronger misspecified-physics comparator infer unknown BC/source nuisance parameters instead of assuming zero?
- Can uncertainty/identifiability diagnostics explain coefficient error that remains high despite noise-level replay?

Any new answer requires an authorized, predeclared experiment rather than selecting settings using known coefficient truth.

## 18. Communication and résumé summary

The conversation also produced short progress summaries, explanations of PINN/adjoint training, temporal loss, CN/BE differences, forward-data bias and literature relevance.

A draft-email request asked for at most three paragraphs, a mention of Zhang-inspired neural coefficient representation and Thakur-inspired temporal consistency, and a closing question seeking guidance on next approaches. This was a drafting request, not authorization to send email; no external email send is part of this record.

Résumé bullets were iterated under exact character-count constraints, first111/100 and later111 for a third point. Numerical examples included21.45% adjoint versus35.37% PINN at1,000 sites, and6.77% PINN versus26.36% BE adjoint at10,000 sites.

Scientific wording should now reflect the complete record:

- It is reasonable to describe matched neural PINN/adjoint pipelines, temporal consistency, independent physics/replay validation and completed noise/density comparisons.
- “Adjoint won almost every baseline” must be tied to the earlier completed subset and its criteria; it is not a statement about every final coefficient-only comparison.
- Hard PDE enforcement is a plausible contributor to stronger replay consistency, not a proved unique cause.
- The10,000 result is one case outside the final primary matrix.
- The final primary qualified result is7,500/5% under declared screens; avoid presenting isolated lower coefficient error as a physics-qualified win.
- The old9-point/CN-generated descriptions should not override the corrected implementation/data provenance.

## 19. Machine-readable sources and detailed appendices

The following tables are generated from retained score records for this session document. No experiment is rerun and no coefficient answer key is newly evaluated to produce them.

### 19.1 Versioned-pilot source index

| Retained pilot | Latest interpreted status |
|---|---|
| [calibrated_v12_gpu](campaign/pilots/calibrated_v12_gpu/pilot.json) | complete_independently_evaluated |
| [conditional_state_v17_gpu](campaign/pilots/conditional_state_v17_gpu/pilot.json) | deferred_not_submitted_human_prioritized_temporal_five_case_test |
| [dense10000_v21_gpu](campaign/pilots/dense10000_v21_gpu/pilot.json) | complete_evaluated |
| [dense1000_v22_kaggle](campaign/pilots/dense1000_v22_kaggle/pilot.json) | complete_evaluated |
| [dense_calibrated_v14_gpu](campaign/pilots/dense_calibrated_v14_gpu/pilot.json) | complete_evaluated |
| [dense_resolution_v15_gpu](campaign/pilots/dense_resolution_v15_gpu/pilot.json) | complete_evaluated |
| [density_v10_gpu](campaign/pilots/density_v10_gpu/pilot.json) | complete_recovered_evaluated |
| [dual_v9_gpu](campaign/pilots/dual_v9_gpu/pilot.json) | completed_independently_evaluated |
| [fixed_d_forward_v7_gpu](campaign/pilots/fixed_d_forward_v7_gpu/pilot.json) | completed_independently_evaluated |
| [fixed_d_refresh_v8_gpu](campaign/pilots/fixed_d_refresh_v8_gpu/pilot.json) | completed_independently_evaluated |
| [ic_adaptive_v4_gpu](campaign/pilots/ic_adaptive_v4_gpu/pilot.json) | complete_validated_hold |
| [ic_bc_v3_gpu](campaign/pilots/ic_bc_v3_gpu/pilot.json) | complete_validated_hold |
| [ic_constraints_v2_gpu](campaign/pilots/ic_constraints_v2_gpu/pilot.json) | complete_validated_hold |
| [ic_convergence_v5_gpu](campaign/pilots/ic_convergence_v5_gpu/pilot.json) | complete_validated_hold |
| [ic_locald_v6_gpu](campaign/pilots/ic_locald_v6_gpu/pilot.json) | completed_evaluated |
| [ic_v1_gpu](campaign/pilots/ic_v1_gpu/pilot.json) | complete_validated_hold |
| [matched_temporal_v20_local](campaign/pilots/matched_temporal_v20_local/pilot.json) | HUMAN HELD; pause file overrides stale running metadata |
| [noise_density_v23](campaign/pilots/noise_density_v23/pilot.json) | two_dispatched_pairs_complete_evaluated_remaining_superseded |
| [noise_density_v24_1000_7500](campaign/pilots/noise_density_v24_1000_7500/pilot.json) | all_12_primary_cells_complete_evaluated |
| [optimizer_v16_gpu](campaign/pilots/optimizer_v16_gpu/pilot.json) | complete_evaluated |
| [sampling_v13_gpu](campaign/pilots/sampling_v13_gpu/pilot.json) | complete_independently_evaluated |
| [temporal_v18_gpu](campaign/pilots/temporal_v18_gpu/pilot.json) | complete_evaluated_partial_coverage |
| [zhang_v11_gpu](campaign/pilots/zhang_v11_gpu/pilot.json) | complete_recovered_evaluated_both_arms |

### 19.2 Final-cell continuous scores

The full-precision values remain in the linked CSV; four-decimal display here avoids rounding1.102375 into an apparent1.1 pass.

| Noise | Sensors | PINN D error | Adjoint D error | PINN CN / BE sigma | Adjoint CN / BE sigma |
|---|---|---|---|---|---|
| 1% | 1,000 | 9.398804% | 32.035837% | 2.968372 / 1.707335 | 2.606514 / 0.996359 |
| 1% | 3,000 | 6.117230% | 38.757072% | 2.761996 / 1.748113 | 2.581343 / 1.056811 |
| 1% | 5,000 | 5.661164% | 42.749003% | 2.834845 / 1.796802 | 2.609791 / 1.063778 |
| 1% | 7,500 | 5.610972% | 45.643568% | 2.968794 / 1.872891 | 2.595051 / 1.067648 |
| 5% | 1,000 | 35.371631% | 21.454686% | 1.923599 / 2.005177 | 1.075980 / 0.967260 |
| 5% | 3,000 | 16.481198% | 23.009115% | 1.135548 / 1.150085 | 1.094925 / 0.991525 |
| 5% | 5,000 | 13.030871% | 23.787751% | 1.102375 / 1.068481 | 1.097919 / 0.991492 |
| 5% | 7,500 | 8.616402% | 25.089574% | 1.097229 / 1.025240 | 1.098443 / 0.993579 |
| 10% | 1,000 | 69.098231% | 21.576945% | 5.859843 / 5.975879 | 1.001283 / 0.974066 |
| 10% | 3,000 | 42.300673% | 22.269318% | 1.857898 / 1.931244 | 1.019497 / 0.993371 |
| 10% | 5,000 | 28.745926% | 22.673857% | 1.275338 / 1.320986 | 1.019480 / 0.992225 |
| 10% | 7,500 | 22.636433% | 23.263250% | 1.098240 / 1.121848 | 1.019340 / 0.992531 |

### 19.3 Final-cell provenance/resource rows

| Noise | Sensors | Arm | Gamma | Sigma | Adam / SSB / capped rounds | Saved source |
|---|---|---|---|---|---|---|
| 1% | 1000 | be_adjoint_nn | 1.423448935e-11 | 0.00023768648 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v23\cases\case_01](campaign/pilots/noise_density_v23/cases/case_01/primary_matched_neural_comparison.csv) |
| 1% | 1000 | temporal_pinn | 1.423448935e-11 | 0.00023768648 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v23\cases\case_01](campaign/pilots/noise_density_v23/cases/case_01/primary_matched_neural_comparison.csv) |
| 1% | 3000 | be_adjoint_nn | 4.744829785e-12 | 0.00023768648 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v24_1000_7500\cases\case_01](campaign/pilots/noise_density_v24_1000_7500/cases/case_01/primary_matched_neural_comparison.csv) |
| 1% | 3000 | temporal_pinn | 4.744829785e-12 | 0.00023768648 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v24_1000_7500\cases\case_01](campaign/pilots/noise_density_v24_1000_7500/cases/case_01/primary_matched_neural_comparison.csv) |
| 1% | 5000 | be_adjoint_nn | 2.846897871e-12 | 0.00023768648 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v24_1000_7500\cases\case_04](campaign/pilots/noise_density_v24_1000_7500/cases/case_04/primary_matched_neural_comparison.csv) |
| 1% | 5000 | temporal_pinn | 2.846897871e-12 | 0.00023768648 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v24_1000_7500\cases\case_04](campaign/pilots/noise_density_v24_1000_7500/cases/case_04/primary_matched_neural_comparison.csv) |
| 1% | 7500 | be_adjoint_nn | 1.897931914e-12 | 0.00023768648 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v24_1000_7500\cases\case_07](campaign/pilots/noise_density_v24_1000_7500/cases/case_07/primary_matched_neural_comparison.csv) |
| 1% | 7500 | temporal_pinn | 1.897931914e-12 | 0.00023768648 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v24_1000_7500\cases\case_07](campaign/pilots/noise_density_v24_1000_7500/cases/case_07/primary_matched_neural_comparison.csv) |
| 5% | 1000 | be_adjoint_nn | 3.558622338e-10 | 0.0011884324 | 1000 / 4000 / 10 | [campaign\pilots\dense1000_v22_kaggle](campaign/pilots/dense1000_v22_kaggle/primary_matched_neural_comparison.csv) |
| 5% | 1000 | temporal_pinn | 3.558622338e-10 | 0.0011884324 | 1000 / 4000 / 10 | [campaign\pilots\dense1000_v22_kaggle](campaign/pilots/dense1000_v22_kaggle/primary_matched_neural_comparison.csv) |
| 5% | 3000 | be_adjoint_nn | 1.186207446e-10 | 0.0011884324 | 1000 / 4000 / 10 | [colab_returns\r672fef\check](colab_returns/r672fef/check/primary_matched_neural_comparison.csv) |
| 5% | 3000 | temporal_pinn | 1.186207446e-10 | 0.0011884324 | 1000 / 4000 / 10 | [colab_returns\r672fef\check](colab_returns/r672fef/check/primary_matched_neural_comparison.csv) |
| 5% | 5000 | be_adjoint_nn | 7.117244677e-11 | 0.0011884324 | 1000 / 4000 / 10 | [colab_returns\r3e8ddc\check](colab_returns/r3e8ddc/check/primary_matched_neural_comparison.csv) |
| 5% | 5000 | temporal_pinn | 7.117244677e-11 | 0.0011884324 | 1000 / 4000 / 10 | [colab_returns\r3e8ddc\check](colab_returns/r3e8ddc/check/primary_matched_neural_comparison.csv) |
| 5% | 7500 | be_adjoint_nn | 4.744829785e-11 | 0.0011884324 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v24_1000_7500\cases\case_08](campaign/pilots/noise_density_v24_1000_7500/cases/case_08/primary_matched_neural_comparison.csv) |
| 5% | 7500 | temporal_pinn | 4.744829785e-11 | 0.0011884324 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v24_1000_7500\cases\case_08](campaign/pilots/noise_density_v24_1000_7500/cases/case_08/primary_matched_neural_comparison.csv) |
| 10% | 1000 | be_adjoint_nn | 1.423448935e-09 | 0.0023768648 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v23\cases\case_02](campaign/pilots/noise_density_v23/cases/case_02/primary_matched_neural_comparison.csv) |
| 10% | 1000 | temporal_pinn | 1.423448935e-09 | 0.0023768648 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v23\cases\case_02](campaign/pilots/noise_density_v23/cases/case_02/primary_matched_neural_comparison.csv) |
| 10% | 3000 | be_adjoint_nn | 4.744829785e-10 | 0.0023768648 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v24_1000_7500\cases\case_03](campaign/pilots/noise_density_v24_1000_7500/cases/case_03/primary_matched_neural_comparison.csv) |
| 10% | 3000 | temporal_pinn | 4.744829785e-10 | 0.0023768648 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v24_1000_7500\cases\case_03](campaign/pilots/noise_density_v24_1000_7500/cases/case_03/primary_matched_neural_comparison.csv) |
| 10% | 5000 | be_adjoint_nn | 2.846897871e-10 | 0.0023768648 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v24_1000_7500\cases\case_06](campaign/pilots/noise_density_v24_1000_7500/cases/case_06/primary_matched_neural_comparison.csv) |
| 10% | 5000 | temporal_pinn | 2.846897871e-10 | 0.0023768648 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v24_1000_7500\cases\case_06](campaign/pilots/noise_density_v24_1000_7500/cases/case_06/primary_matched_neural_comparison.csv) |
| 10% | 7500 | be_adjoint_nn | 1.897931914e-10 | 0.0023768648 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v24_1000_7500\cases\case_09](campaign/pilots/noise_density_v24_1000_7500/cases/case_09/primary_matched_neural_comparison.csv) |
| 10% | 7500 | temporal_pinn | 1.897931914e-10 | 0.0023768648 | 1000 / 4000 / 10 | [campaign\pilots\noise_density_v24_1000_7500\cases\case_09](campaign/pilots/noise_density_v24_1000_7500/cases/case_09/primary_matched_neural_comparison.csv) |

### 19.4 Important retained identities

- Shared recent smooth-case initial coefficient tensor SHA: e961924dffcf56c5322f2bda40af3090568a1fc3df4d4684f67b47cd2fff029b.
- Delivered notebook SHA:253bc8d50a9031eb379b3cec6065c785bbb90c9fa75fd0d6b8cc4ee4b777f3cd.
- Embedded payload SHA:9c2f3b998576f4f1e59d782086c11e9fc6beac60d608d2a9c4ba9ae897fbbd69.
- First processed Colab ZIP SHA:7744813d08ba87f4da34b45d71af3a3e9e37617fe17a069d934d405ba8fb740c.
- Second processed Colab ZIP SHA:9193d7eb693d347d614776102e8a22c1a23ae17c0eb25368b0c86b6d6b374204.
- RK4 provenance audit metadata SHA:be9a1b8ed797ca85e4d521554c952699b28ce38ff88fe43978bbea9550924b76.

Returned notebook file bytes can differ because outputs/metadata were saved; code-cell identity was checked independently. These hashes are provenance, not promises of cross-platform identical optimizer trajectories.

### 19.5 Status precedence and document maintenance

This session record captures the state on22October2026. For later work, compare its timestamp with new dated CAMPAIGN/state/pilot records. Do not edit historical outcomes to match a new preferred narrative.

If new authorized experiments occur, add dated sections/new versions, retain previous results, and distinguish completed scoped deliverables from held or gated broader work.

**End state: complete12-cell density/noise comparison, complete scoped scientific report, all24 fits retained, no active scoped compute or monitor, broader held/gated campaign explicitly separate.**

