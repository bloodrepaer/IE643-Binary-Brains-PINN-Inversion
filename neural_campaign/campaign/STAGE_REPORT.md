# IE643 stage-wise project report — working research record

Generated 2026-10-21T09:57:14.507382+00:00. Snapshot counts updated 2026-10-21T09:55:34.541520+00:00.

**Status: partial campaign, not a final submission report.** Completed fits, negative experiments and pending work are distinguished below. The numerical results are conditional on the retained protocols; broad single-seed coverage does not establish population-level superiority.

## Stage 0. Objective and comparison design

Recover a positive spatial diffusivity D(x,y) in u_t = div(D grad u) from sparse noisy temperatures on a two-dimensional unit square. The primary comparison is neural adjoint versus neural PINN: both represent D with a neural network. Grid adjoint is a secondary baseline. Neural adjoint obtains temperature from the finite-difference solver and propagates a reduced gradient through the coefficient network. Neural PINN jointly optimizes temperature and coefficient networks with a PDE penalty.

The project must identify conditions under which either method is effective; a PINN win is an empirical possibility, not a required conclusion. Low sensor fitting error alone is insufficient: recovered coefficients are replayed through the same trusted numerical solver.

## Stage 1. Synthetic inputs and controlled parameter axes

| collection                   |   cases | spatial_sensors                        | noise_eta            | time_samples       |
|:-----------------------------|--------:|:---------------------------------------|:---------------------|:-------------------|
| datasets_v2_rk4_zero         |     389 | 5,20,50,100,3969                       | 0.0,0.01,0.05,0.1    | 1,4,8,16,40,80,200 |
| datasets_v2_pinnwin          |      57 | 3,5,8,50                               | 0.05,0.1,0.2,0.3,0.5 | 16,40,200          |
| datasets_v2_pinnwin2         |      24 | 100,400                                | 0.01,0.05            | 40                 |
| datasets_v2_density          |       4 | 25,100,400,3969                        | 0.0                  | 40                 |
| datasets_v2_density_wideic   |       2 | 400,3969                               | 0.0                  | 40                 |
| datasets_v2_pinnwin3         |       6 | 400,3249                               | 0.0,0.01             | 40                 |
| datasets_v2_density_extended |     180 | 5,20,50,100,200,400,800,1600,3200,3969 | 0.01,0.05,0.1        | 8                  |

Full collection-specific axis values are in dataset_parameter_axes.csv. Collections with similar observations are kept distinct by collection/key, rather than merged solely by sensor count. Sensor count is the number of spatial locations; total readings also depend on the number of measurement times.

The canonical fields include smooth sinusoidal, Gaussian bump, inclusion, layered, Gaussian random and multiple-defect examples, plus retained contrast/checkerboard variants. Sensor layouts, noise, time sampling, Fourier number/horizon and known or hidden physics vary by collection. Answer keys remain evaluation-only.

Input formats: manifest.json/CSV lists cases and paths; problem JSON specifies domain, supplied IC/BC and source; supplied IC CSV gives x,y,u0; sensor CSV gives x,y,t,u,sigma. Synthetic NPZ answer keys contain reference fields/clean data where available. Inversion Dataset loads only sensors and supplied problem/IC. ground_truth() is a separate post-fit evaluation path. New private pilot bundles contain only supplied inputs and project modules, excluding coefficient answer keys and true-field metadata.

The extended density collection has 180 cases: six fields × ten spatial counts (5,20,50,100,200,400,800,1600,3200,3969) × three noise fractions (1%,5%,10%). Fourier number 0.1 and eight uniform times are fixed. Locations and noisy values are nested within each field/noise group, so a sparse subset is an exact subset of the corresponding dense observations. These node-based layouts are analyzed separately from historical off-grid layouts.

Reason for this design: holding the observation realization fixed isolates the change in available observations. It does not quantify uncertainty over newly sampled noise or sensor locations. More sensors do not guarantee better recovered D when the optimizer is capped or the inverse problem is weakly identifiable.

## Stage 2. Forward solver and discrete adjoint

The spatial operator uses flux links with harmonic-mean diffusivity and eliminated Dirichlet boundary nodes. Synthetic canonical data use a finer forward discretization; inversion and common replay use the retained CN/Rannacher configuration, ordinarily 65×65 nodes and 200 steps for the reported canonical cases. RK4 generation versus CN inversion also creates a deliberate finite-discretization distinction.

The manual discrete adjoint differentiates the actual time-stepping/observation operator. It supplies dJ/dm for m=logD on the full grid; PyTorch supplies (dm/dphi)^T(dJ/dm). This avoids a dense neural-network Jacobian and an autograd graph through sparse linear solves. Existing solver/gradient checks and subsequent version-specific finite-difference checks are retained in tests and pilot validation files.

Post-fit layered diagnostics isolated a substantial spatial-grid discrepancy: true-D clean replay on the coarse grid differed more than on the finer grid, while time refinement alone did not remove it. Learned-field transfer diagnostics are separate evaluations and do not replace common coarse-grid campaign scores or choose training settings from truth.

## Stage 3. Replace grid controls by a shared neural coefficient

The frozen broad pair uses a 3×32 tanh MLP, 2,241 parameters, float64, with sigmoid-bounded logD between log(0.001) and log(10). A constant diffusivity is fitted to supplied sensor data; both neural methods receive identical seeded hidden features and the same constant initial coefficient field. The output network has no time input and can be evaluated between solver nodes.

Reason: sharing the coefficient representation separates the inverse method from the grid-versus-network parameterization question. It does not automatically equalize the two state representations, optimizer, alpha selection, compute budget or finite-dimensional objectives.

The broad neural adjoint uses a summed Gaussian observation misfit plus discrete spatial regularization and a discrepancy path, with L-BFGS-B iteration limits. The frozen PINN adds normalized-time continuous PDE, supplied IC and BC penalties; its alpha is fixed. All optimizer caps, data-fit/replay differences and hardware limitations must therefore remain visible in comparisons.

## Stage 4. Evaluation and what counts as a valid comparison

Primary metrics are relative L2 error in physical D, common CN replay RMS divided by sensor sigma, temperature-network sensor fit, independent supplied-IC/BC error, held-out PDE residual by normalized-time bins, optimizer iterations/evaluations/caps, and hardware-qualified elapsed time. Physical D error and logD/log-permeability errors are different metrics and cannot be compared numerically without qualification.

For localized defects, report Dice/false-positive area alongside detection count, contrast and localization; detecting every defect can coexist with a poor recovered mask. For hidden BC/source cases, keep assumed-zero inversion separate from matched-physics inversion and evaluate true physics only after fitting.

A replay near one noise sigma is a useful necessary consistency check, not proof of correct or uniquely identified D. In particular, a network can fit sparse observations yet recover a wrong coefficient in weak-sensitivity regions. A flag named floor_above_sigma does not establish a physical/statistical floor when regularization-path stages hit iteration caps.

Only completed matched observation pairs count toward a descriptive method comparison. Queued, missing, platform-failed or unfinished fits are not method losses. A low scalar training loss is not a convergence certificate or an independent physics check.

## Stage 5. Frozen broad campaign progress

Planned new fits: 1510; completed new fits: 856. Completed by method: {"grid": 186, "adjoint_nn": 662, "pinn": 8}. Historical grid rows reused: 476. Research pilots, extra initialization seeds and convergence continuations are outside these broad counts.

The nominal plan contains 662 neural adjoints, 662 PINNs and 186 new grid fits, plus 476 retained historical grid baselines. Broad PINN expansion is held by a scientific validation gate; the nominal 1510-fit target is not completed while that gate is unresolved. All extended density grid/neural-adjoint curves and both additional seed-confirmation groups have completed. Ongoing workers and exact deadlines are recorded in CAMPAIGN.md and worker state files.

Eight frozen broad neural pairs are retained in primary_neural_comparison.csv; the PINN coefficients failed supplied-constraint/common-replay checks despite reasonable temperature fits. The much larger neural-adjoint-versus-grid density sweep is a secondary representation comparison and must not be reported as PINN wins.

## Stage 6. Versioned PINN investigations and negative evidence

All plain-state v0 fits and subsequent pilots are preserved. Known-Gaussian IC bases repaired initial-condition agreement; image-kernel zero-Dirichlet constructions repaired boundary agreement. Stronger penalties, residual-adaptive sampling, extra quasi-Newton work, coefficient coupling, and a paired multiplier continuation still failed independent coefficient replay. Fixed-D forward-only studies isolated state optimization/generalization difficulties; those were not inverse recovery successes.

The v10 matched density pilot used the same nested smooth-field observations at 50 versus 3969 sensors, 5% noise and eight times. PINN D errors were 30.50% and 32.52%; replay was 2.406 and 3.407 sigma. Retained neural-adjoint errors on the same cases were 16.30% and 9.05%, with replay 0.991 and 1.032 sigma. Both PINNs passed hard supplied IC/BC but reached their quasi-Newton limits. Fixed alpha and data batching imply protocol-dependent density effects, not a universal conclusion that dense sensing harms PINNs.

## Stage 7. Third-reference paper audit and matched neural transfer

The prep slides identify Zhang, Alla and Karniadakis, Adjoint Method versus Physics-Informed Neural Networks in PDE-Constrained Inverse Problems (arXiv2606.12337v2). The primary source covers Burgers forcing, steady Darcy permeability, Allen–Cahn reaction and scalar Navier–Stokes viscosity, not this transient spatial-diffusion problem. Its numerical hyperparameters therefore require a disclosed transfer, not a claim of reproduction. See pilots/zhang_v11_gpu/REFERENCE_AUDIT.md for the source-to-code audit and explicit departures.

The v11 transfer and its matched neural-adjoint arm shared an unbounded exponential neural logD, initial weights, unit mean observation loss, H1 gamma0.1 and 1000Adam+4000SSBroyden schedule. The PINN used sampled soft IC/BC; the adjoint imposed its state equation through CN. Both completed the scheduled iterations.

| method               |   rel_L2_D |   replay_sigma |   sensor_fit_sigma | status   |   quasi_newton_iterations |
|:---------------------|-----------:|---------------:|-------------------:|:---------|--------------------------:|
| neural_adjoint_zhang |   0.381248 |        2.72531 |            2.72531 | complete |                      4000 |
| neural_pinn_zhang    |   1        |      106.102   |            1.38745 | complete |                      4000 |

The PINN coefficient collapsed near2.76e-52. Its fixed-cloud losses were small, but independent initial-field error was66.04% and boundary RMS15.97sensor sigma. Reconstructing the retained training cloud reproduced the reported losses, demonstrating a generalization/constraint failure rather than a successful inverse solution. H1 does not penalize a constant logD shift, so exponential positivity alone did not prevent nearzero D.

The borrowed gamma0.1 also had a radically different objective scale: under the same continuous H1 functional it maps algebraically to roughly1.416e7 alpha in the original sum/noise-weighted objective. Quadrature differs, so this is a scale audit rather than numerical equivalence of both discretizations. The v11 adjoint became nearly constant as well. Preserve the negative result; it does not invalidate SSBroyden or the paper universally.

## Stage 8. Predeclared matched repair v12

The v12 pair restores the existing bounded coefficient prior, uses the validated supplied-Gaussian hard IC/BC state construction, and converts fixedalpha1 from the original likelihood to a normalized mean-data objective. With U=max absolute suppliedIC/observeddata, gamma=2sigma_w²/(N U²); for the pilot it is7.1172446768e-9. The PINN adds a declared unit normalized-time PDE residual penalty. This is a calibrated conversion of the regularizer, not a claim that alpha1 or unit PDE weight is optimal.

Both arms share coefficient initialization, observations, mean-data objective, exact discrete regularizer, float64 and SSBroyden schedule. The state network is32×3 to bound dense-curvature cost. Its local-D heat base is an ansatz; full spatial coefficient derivatives are retained in the PDE. Half early log-time/half uniform collocation intentionally changes the sampling measure. Multiple changes from v11 are bundled into this explicitly new protocol, so eventual improvement cannot isolate one causal ingredient.

The retained v12 paired results have been independently evaluated; see the versioned validation and comparison table below. Fit status and budget caps remain material.

Both arms run sequentially on one Kaggle host with per-arm/internal/external bounds and preserved partial artifacts. CPU/GPU resource use and actual iterations/evaluations still require qualification. A single passing case is only a screen: fields/seeds and an explicit frozen protocol are needed before broad migration.

### Verified v12 results

| method     | key                                                   |   rel_L2_D |   replay_sigma |   sensor_fit_sigma | status   |   wall_s |   adam_completed |   quasi_newton_iterations |   function_evaluations |       gamma |   parameter_count |
|:-----------|:------------------------------------------------------|-----------:|---------------:|-------------------:|:---------|---------:|-----------------:|--------------------------:|-----------------------:|------------:|------------------:|
| adjoint_nn | A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05 |   0.143579 |       0.900287 |           0.900287 | complete |  2618.67 |             1000 |                      4000 |                   7911 | 7.11724e-09 |              2241 |
| pinn       | A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05 |   0.391918 |       1.76428  |           1.64916  | complete |  2957.52 |             1000 |                      4000 |                   8431 | 7.11724e-09 |              4514 |

### Controlled collocation follow-up v13

The retained v12 checkpoint has a12.71-fold held-out/training PDE RMS gap on the same mixed-time measure. A frozen-model random/Sobol quadrature audit was inconclusive; lower replicate variability did not establish integration accuracy or a training repair. No coefficient truth or refitting was used in those diagnostics.

V13 tests only the PINN collocation generator: pseudorandom versus independently scrambled Sobol4096-point clouds, with identical measure, refresh schedule, representation, initialization, loss weights and optimizer. A fresh neural-adjoint control runs sequentially on the same host. All three arms are separate research compute outside1510; record caps, partial status and hardware-qualified cost. Independent constraints and all-arm checkpoint/replay/protocol checks precede answer-key scoring.

| arm             | method     | key                                                   |   rel_L2_D |   replay_sigma |   sensor_fit_sigma | status   |   wall_s |   adam_completed |   quasi_newton_iterations |   capped_rounds |   function_evaluations |       gamma |
|:----------------|:-----------|:------------------------------------------------------|-----------:|---------------:|-------------------:|:---------|---------:|-----------------:|--------------------------:|----------------:|-----------------------:|------------:|
| adjoint_control | adjoint_nn | A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05 |   0.143579 |       0.900287 |           0.900287 | complete |  2401.41 |             1000 |                      4000 |              10 |                   7911 | 7.11724e-09 |
| random_control  | pinn       | A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05 |   0.391918 |       1.76428  |           1.64916  | complete |  2884.46 |             1000 |                      4000 |              10 |                   8431 | 7.11724e-09 |
| sobol           | pinn       | A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05 |   0.440032 |       2.05984  |           1.97925  | complete |  2911.54 |             1000 |                      4000 |              10 |                   9047 | 7.11724e-09 |

### Requested higher-density matched pairs v14

The user requested higher sensor counts before further50-sensor pilots. Fresh3969- and400-sensor neural PINN/adjoint pairs use nested observations at5%noise,eight times,Fo.1,seed42. All readings enter the objective. Architecture, original random mixed-time collocation and optimizer match v12; fixed likelihoodalpha1 gives density-dependent gamma and sensor-fitted constant initialization is recomputed per input, shared within each pair. Density effects are conditional on this protocol.

| key                                                     |   n_sensors | method     |   rel_L2_D |   replay_sigma |   sensor_fit_sigma | status   |   wall_s |   adam_completed |   quasi_newton_iterations |   capped_rounds |   function_evaluations |       gamma |
|:--------------------------------------------------------|------------:|:-----------|-----------:|---------------:|-------------------:|:---------|---------:|-----------------:|--------------------------:|----------------:|-----------------------:|------------:|
| A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 |        3969 | adjoint_nn |   0.200081 |       0.980659 |           0.980659 | complete |  3077.13 |             1000 |                      4000 |              10 |                   8085 | 8.96604e-11 |
| A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 |        3969 | pinn       |   0.40372  |       2.47463  |           2.25956  | complete |  3091.86 |             1000 |                      4000 |              10 |                   8438 | 8.96604e-11 |
| A_smooth__Fo0.1__uniform8__n400__nodes_nested__eta0.05  |         400 | adjoint_nn |   0.174056 |       0.944732 |           0.944732 | complete |  2600.19 |             1000 |                      4000 |              10 |                   7786 | 8.89656e-10 |
| A_smooth__Fo0.1__uniform8__n400__nodes_nested__eta0.05  |         400 | pinn       |   0.3831   |       2.52199  |           2.2665   | complete |  2954.44 |             1000 |                      4000 |              10 |                   8439 | 8.89656e-10 |

All four fits completed1000Adam+4000SSB without timeout; all40SSB rounds reached the iteration cap. Both PINNs pass supplied IC/BC but fail common replay consistency; neither is a verified PINN win. Larger sensor count alone did not repair this protocol. One field/noise/seed and capped optimization do not establish universal sensor-density effects or converged optima. Independent constraints and within-density coefficient/checkpoint/shared-initialization/gamma/common-replay checks preceded answer-key scoring. All four fits are separate research compute outside1510. On the same sequential TeslaT4 host, adjoints use CPU2threads and PINNs use CPU curvature plus CUDA data/residual; times require that resource qualification.

Read-only post-fit diagnostics reproduce the final objective components and unchanged checkpoints. The identical mixed-measure held-out PDE probe gives training/held-out RMS ratios23.20 at3969sensors and24.14 at400; the top1%points carry94.5%/98.1%of held-out residual energy. State/common-CN sensor disagreement is0.829/0.901sigma. Data/PDE gradient cosines are-0.983/-0.683 at the retained parameters, showing an objective tradeoff there. These parameterization-dependent gradients and finite-sample probes do not establish convergence, a unique cause or an optimal penalty. No answer keys, refitting or checkpoint selection enter these diagnostics.

### Controlled higher-density resolution test v15

A fixed-checkpoint audit on the3969-sensor v14 PINN found descriptive four-replicate energy-estimator CV23.2%,9.3%,3.3% at4096,16384,65536 mixed-time points. This motivates a resource-bounded16384-point test, not a proven inverse improvement or optimal count. Three fresh arms use the same3969observed sensors:4096-point PINN control,16384-point PINN and neural-adjoint control. All full-data objectives, loss weights, representations, random sampling measure, refresh and1000Adam+4000SSB schedules remain unchanged. Actual physics compute differs; timing/evaluation/cap and resource qualification is required.

| arm             | method     | key                                                     |   n_sensors |   n_pde |   rel_L2_D |   replay_sigma |   sensor_fit_sigma | status   |   wall_s |   adam_completed |   quasi_newton_iterations |   capped_rounds |   function_evaluations |       gamma |
|:----------------|:-----------|:--------------------------------------------------------|------------:|--------:|-----------:|---------------:|-------------------:|:---------|---------:|-----------------:|--------------------------:|----------------:|-----------------------:|------------:|
| adjoint_control | adjoint_nn | A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 |        3969 |    4096 |   0.200081 |       0.980659 |           0.980659 | complete |  2924.69 |             1000 |                      4000 |              10 |                   8085 | 8.96604e-11 |
| pde16384        | pinn       | A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 |        3969 |   16384 |   0.357423 |       2.03601  |           2.02456  | complete |  3334.07 |             1000 |                      4000 |              10 |                   8663 | 8.96604e-11 |
| pde4096_control | pinn       | A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 |        3969 |    4096 |   0.40372  |       2.47463  |           2.25956  | complete |  2942.8  |             1000 |                      4000 |              10 |                   8438 | 8.96604e-11 |

All three arms completed1000Adam+4000SSB without timeout; all30SSB rounds were capped. At3969sensors,16384physics points improved PINN Derror40.37%to35.74% and replay2.475to2.036sigma relative to4096control, but neural adjoint remained20.01%/0.981sigma. BothPINNs pass IC/BC but fail replay consistency, so no PINN win or broad migration. Fresh controls bitwise reproduce v14 tensors,Darrays and objective histories; this is the same observation realization and seed, not independent confirmation.

On the common independent mixed-time PDE probe, the heldout/training RMS gap fell23.20to1.35 and state/common-CN sensor disagreement fell0.829to0.036sigma. Thus collocation generalization and sampled state consistency improved substantially while inverse/data-fit error remained. Retained data/PDE gradients still oppose each other (cosine-0.953 for16384); their raw norms depend on parameterization and do not prove convergence or a unique failure cause. The same sequentialTeslaT4 host usedCPU2threads for all curvature,CUDAdata/residual forPINNs; actual times/evaluations differ. No coefficient truth, checkpoint selection or refitting enters these postfit diagnoses.

### Controlled dense optimizer comparison v16

Four fresh research arms at3969observed sensors compare SSBroyden and L-BFGS-B for both neural PINN and neural adjoint. The full observations, bounded coefficient network, shared initialization, data objective, regularizer and iteration limits remain fixed; both PINNs use16384 original mixed-time PDE points. The no-truth local derivative and descent profiles motivate an optimizer implementation test, not an assumed repair. Curvature memory, memory reset and stopping conventions differ; the reused trainer retains unused dense identity bookkeeping for L-BFGS-B, so no memory-scaling or unqualified speed claim is supported. All outcomes and actual evaluations/time/resources/caps will be reported.

| arm            | optimizer   | method     | key                                                     |   n_sensors |   n_pde |   rel_L2_D |   replay_sigma |   sensor_fit_sigma | status   |   wall_s |   adam_completed |   quasi_newton_iterations |   capped_rounds |   optimizer_failures |   function_evaluations |       gamma |
|:---------------|:------------|:-----------|:--------------------------------------------------------|------------:|--------:|-----------:|---------------:|-------------------:|:---------|---------:|-----------------:|--------------------------:|----------------:|---------------------:|-----------------------:|------------:|
| lbfgsb_adjoint | lbfgsb      | adjoint_nn | A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 |        3969 |    4096 |  0.0523288 |       0.994934 |           0.994934 | complete | 1745.63  |             1000 |                      4000 |              10 |                    0 |                   5355 | 8.96604e-11 |
| lbfgsb_pinn    | lbfgsb      | pinn       | A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 |        3969 |   16384 |  0.37874   |       2.94491  |           2.92727  | complete |  855.703 |             1000 |                      4000 |              10 |                    0 |                   5328 | 8.96604e-11 |
| ssb_adjoint    | ssb         | adjoint_nn | A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 |        3969 |    4096 |  0.200081  |       0.980659 |           0.980659 | complete | 2988.76  |             1000 |                      4000 |              10 |                    0 |                   8085 | 8.96604e-11 |
| ssb_pinn       | ssb         | pinn       | A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 |        3969 |   16384 |  0.357423  |       2.03601  |           2.02456  | complete | 3341.09  |             1000 |                      4000 |              10 |                    0 |                   8663 | 8.96604e-11 |

All four arms completed1000Adam+4000quasi-Newton iterations with all40rounds capped and no timeout or recorded optimizer failure. L-BFGS-B improved neural-adjoint coefficient error20.01%to5.23%, while PINN error increased35.74%to37.87% and common replay worsened2.036to2.945sigma. Both PINNs pass supplied IC/BC but fail replay; no PINN win. Fresh SSB controls bitwise reproduce v15 network tensors,Darrays and objective histories, so these are not independent observation/seed confirmations. One field/noise/seed and capped fits limit generalization. On the same sequential TeslaT4 host all curvature usesCPU2threads, PINNs additionally useCUDAdata/residual; disclose actual time/evaluations without unqualified speed or memory claims.

The retained-checkpoint audit reproduces both original objectives without refitting or answer keys. L-BFGS-B PINN data/PDE gradients have cosine-0.9999875 and total parameter-gradient L2=3.23e-6; state/common-CN sensor disagreement is0.0446sigma (SSB0.0358sigma). Thus close sampled state consistency and a small parameter gradient can coexist with poor inverse/data fit. These scale-dependent local quantities do not prove convergence, a global optimum, a unique failure cause or conditional state-network feasibility.

### Five high-density temporal-consistency cases v18

The user requested5-10dense cases. Five predeclared fields (smooth,bump,layered,random,defects) use3969nested sensors,5%noise,eighttimesFo.1,seed42. Each case has a temporal-consistency PINN, an h-squared continuous-residual PINN control, and a fresh neural-adjoint control. h=0.01 in normalized time;4096uniform collocation pairs stay inside the horizon. Shared serialized observation-fitted coefficient initialization, bounded networks, full-data objective, regularizer and1000Adam+4000SSB schedule are retained. This adapts the loss format, not all of Thakur et al. architecture/training. The scale-matched control separates time-discretization effects from the changed effective physics weight. All15research fits are outside1510.

PINNs run sequentially on Kaggle CPU2thread curvature plus CUDA data/residual, while matched adjoints run locally withCPU2threads and one worker. Hardware, budgets and actual evaluations/caps must qualify any cost comparison; no unqualified speed claim. Partial/missing arms are not losses. Coefficient truth is used only after independent known-physics and all15arm checkpoint/common-replay/protocol checks.

Overall five-case coverage is INCOMPLETE. All14 available outputs passed checkpoint/protocol/common replay reproduction checks before scoring; only smooth and random cases have all three training schedules completed. Bump/layered adjoints timed out; the defects control has no completed checkpoint. Partial reporting excludes these cases by completion status, not coefficient truth. Smooth temporal PINN coefficient error17.23% versus neural adjoint20.00%, with replay1.09916sigma narrowly passing the predeclared1.1 screen versus adjoint0.98132. Random temporal PINN20.59% versus adjoint19.92%, replay1.14203 fails. One exploratory coefficient advantage, single fixed observation/seed, all rounds capped; no broad migration or unqualified cost claim.

| key                                                     | arm             | method     |   rel_L2_D |   replay_sigma |   sensor_fit_sigma | status   |   adam_completed |   quasi_newton_iterations |   capped_rounds |   function_evaluations |   wall_s |   n_sensors |   time_step_tau |
|:--------------------------------------------------------|:----------------|:-----------|-----------:|---------------:|-------------------:|:---------|-----------------:|--------------------------:|----------------:|-----------------------:|---------:|------------:|----------------:|
| A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 | adjoint_nn      | adjoint_nn |   0.200013 |       0.981319 |           0.981319 | complete |             1000 |                      4000 |              10 |                   7928 |  3680.78 |        3969 |            0.01 |
| A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 | scaled_pde_pinn | pinn       |   0.2405   |       1.32553  |           0.980208 | complete |             1000 |                      4000 |              10 |                   7741 |  3009.19 |        3969 |            0.01 |
| A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 | temporal_pinn   | pinn       |   0.172321 |       1.09916  |           0.98424  | complete |             1000 |                      4000 |              10 |                   8395 |  3171.61 |        3969 |            0.01 |
| E_grf__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    | adjoint_nn      | adjoint_nn |   0.199186 |       0.985573 |           0.985573 | complete |             1000 |                      4000 |              10 |                   8541 |  3797.95 |        3969 |            0.01 |
| E_grf__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    | scaled_pde_pinn | pinn       |   0.213647 |       1.26675  |           0.987654 | complete |             1000 |                      4000 |              10 |                   7773 |  2997.3  |        3969 |            0.01 |
| E_grf__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    | temporal_pinn   | pinn       |   0.205948 |       1.14203  |           0.988206 | complete |             1000 |                      4000 |              10 |                   8258 |  3178.4  |        3969 |            0.01 |

## Stage 9. Reproducibility, artifact integrity and failure retention

Broad numerical source is frozen under campaign/code/48b49b1f17a0, with jobs.json/settings.json and per-job observations hashes, commands, logs, metrics, D_hat.npz and neural checkpoints. Separate pilot folders contain source hashes, source/data protocols, implementation checks and independent validation. Failed attempts and global-budget timeouts are retained before unchanged retries; active workers are checked before restart and protected against duplication.

A disk-full incident was diagnosed and recovered. Canonical data, scientific source versions and trained results were preserved; only hash-verified duplicate input bundles and lossless NTFS input compression were used to reclaim space.293core artifacts passed integrity checks; one absent PNG was regenerated without changing coefficient/metrics. New workers stop dispatch below128MiB free disk. Do not delete trained results or recreate unused large bundles.

Generate current analysis with python analyze_campaign.py --true-physics. Generate this partial report with python build_stage_report.py. Full tables, figures, model checkpoints and validation files remain the underlying evidence; Markdown summaries do not substitute for them.

## Stage 10. Interface and final submission work still required

A local read-only saved-solution interface is functionally verified: neural_interface.py supports primary neural-pair maps/metrics, saved or uploaded checkpoints, entered/uploaded coordinates or sensor CSV, and prediction/field/metric downloads. Its checks reproduce saved coefficient arrays and the supplied coupled-PINN initial field. Inference evaluates each retained inverse solution; uploading different observations does not reconstruct a new D. Live inversion of a newly uploaded problem is not implemented in this prototype. AppTest and local HTTP checks passed; complete browser visual review remains pending because the browser automation runtime was unavailable.

Remaining work includes finishing the bounded neural-adjoint queue, resolving or explicitly reporting the held PINN protocol, validating representative findings across seeds under a valid protocol, extending and visually reviewing the saved-inference interface where required for the final submission, and assembling the final stage-wise report/model/code/dependency package. This draft is a transparent partial record and makes no claim that these submission requirements are complete.
