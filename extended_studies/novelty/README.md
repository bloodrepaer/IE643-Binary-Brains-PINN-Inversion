# Novelty extension: hybrid PINN → adjoint, temperature dependence, data quality

The course's novelty phase asks for new ideas beyond the intensive assessment, with comparative results. This folder is a training pipeline that runs every experiment with one command, at a quick (pilot) or full size, on a CPU or a GPU, and summarises the results into tables and figures.

## The experiments

| ID | Question | Datasets | Arms (methods compared) | A "win" means |
|---|---|---|---|---|
| **E1** | Does combining the PINN and the adjoint beat both? (the novelty plan) | 20-dataset evaluation set (known physics) + 14 unknown-boundary / unknown-source datasets from Study 3 | `adjoint`, `pinn`, `hybrid_path`, `hybrid_balance`, `hybrid_prior`, `reverse` | lower relL2(D) than **both** methods alone, with the PINN's training time included in the cost |
| **E1b** | Can the PINN supply the physics the adjoint is missing? | 8 hidden-wall datasets (Study 3, H2u) | `pinn_bc` (boundary-aware PINN), `hybrid_bc` (PINN's wall temperatures fixed in the adjoint), `hybrid_bc_joint5` (adjoint estimates D and walls, walls start from the PINN's), the fair classical rival `adjoint_joint5` (same joint inversion, walls start at zero), `adjoint_joint1` (walls uniform along each edge, the truth's structure), `adjoint_truebc` (true walls: a bound) | `hybrid_bc*` lower relL2 than both `pinn_bc` and `adjoint_joint5` |
| **E1s** | Does the PINN's hidden-source advantage survive a fair classical rival? | 6 hidden-source datasets (Study 3, H3u) | `adjoint_src` (adjoint estimates D and a source f, free sign), `adjoint_src_pos` (f ≥ 0), `adjoint_truesrc` (true source: a bound) | joint inversion lower relL2 than the boundary-aware PINN |
| **S1** | How much of every PINN number is seed luck? | the 20 known-physics datasets + 6 H3u | `pinn` and `hybrid_balance` at seeds 43–46 (seed 42 = E1) | — |
| **E1c** | Equal budget: is the hybrid better than just training the PINN longer? | the 20 known-physics datasets | `pinn_3x` (3 × the epochs, about the hybrids' cost) | compared with E1's `pinn`, `adjoint` and hybrids |
| **D1** | Does the PINN improve with more sensors? | 6 fields × 20 / 100 / 400 / 1,600 / 3,969 sensors × 1 / 5 / 10 % noise (90, extended density collection) | `pinn` (project PINN); the CN grid and CN neural adjoints of the campaign on the same datasets | error falls with sensor count |
| **D2** | Is the PINN's flat curve a training-budget limit? | A_smooth, E_grf at 100 / 3,969 sensors, 1 % noise | `pinn_long` (4 × the epochs) | — |
| **D3** | Is the bias in the u-network or in the D-step? | as D2; exact-u runs on A_smooth, C_inclusion, E_grf | `pinn_split50` (half the PDE loss trains u instead of 10 %); `oracle_u`, `oracle_u_weaktaper` (the PINN's D-step fed the exact temperature field, re-simulated finely in time) | — |
| **E2a** | Does a third network for D(u) help when the truth has no temperature dependence? | 6 tiers, 50 sensors, 5 % noise | `pinn`, `pinn_T` | expected: no gain, g(u) ≈ 0 (a control) |
| **E2b** | When D really depends on temperature, does modelling it win? | New Arrhenius data: 2 fields × strength S = 0, 0.5, 1, 2 × 2 sensor settings (16) | `adjoint` and `pinn` (assume D(x, y)), `pinn_T` (D(x, y, u)) | `pinn_T` beats both as S grows, and recovers g(u) ≈ s(u) |
| **E3a** | Can hyperparameters be chosen without the true D? (train / validation / test split) | 8 datasets | 10 PINN loss-weight settings on 80 % of sensors; `pinn_selected` vs `pinn_default`; `adjoint_cv` vs `adjoint_dp` | the held-out choice is as good as the default and close to the truth-chosen oracle |
| **E3b** | Do better-placed sensors beat more sensors? | 3 fields × 16 / 36 / 64 sensors × random / grid / designed (27) | `adjoint`, `pinn` | designed < grid < random error at equal budget |
| **E3c** | Do adaptively placed physics points help the PINN? | 8 datasets | `pinn`, `pinn_rar` | lower relL2 at the same number of physics points |
| **E3d** | Does augmenting sparse sensors with interpolated readings help? | 6 sparse datasets (5–20 sensors) | `adjoint`, `adjoint_aug`, `pinn`, `pinn_aug` | expected: no gain (interpolation adds no information about D) |

### E1: the three hybrids and the reverse

All start from the PINN's estimate D_PINN, projected by least squares onto the adjoint's 16 × 16 control grid (`hybrid.project_to_controls`). Everything else is the project's classical adjoint unchanged: CN solver, Tikhonov, L-BFGS-B with 150 iterations per α, and α chosen by the discrepancy principle.

- **`hybrid_path`:** warm start, same α path as the cold adjoint (N_d × 10⁰ … 10⁻⁶). A pure warm start.
- **`hybrid_balance`:** the α path starts where the PINN estimate's data misfit and its regulariser balance (α₀ = misfit / R), then steps down as usual. Keeps more of the PINN's structure.
- **`hybrid_prior`:** the regulariser penalises |∇(log D − log D_PINN)|² instead of |∇ log D|². The adjoint is pulled toward the PINN's map, and the data decide where it moves away. This is the variant that can keep a PINN advantage.
- **`reverse`:** the adjoint's estimate seeds the PINN's D-network (the project's `--warmstart-from-adjoint` procedure), then the PINN trains as usual.

Costs are counted honestly: hybrid total = PINN training + projection + adjoint stage, and reverse total = adjoint + PINN.

The pilot already shows why `hybrid_path` alone is not enough. The large early α values pull every start back toward a smooth field, so it ends on the cold adjoint's answer (relL2 0.1026 for both on A_smooth). The other two variants change where the adjoint ends up, not just where it starts.

### E1b: hidden walls (`hybrid_bc.py`)

`WallSolver` is the project's adjoint with Dirichlet wall values that can be set or optimised. Walls are parameterised per edge as piecewise-linear in time (17 knots) and, for the `*5` arms, piecewise-linear along the edge (5 knots, 340 wall parameters); `*1` arms keep each edge uniform along its length, which is how the hidden truth was built, so it is a generous model for the classical method. The wall gradient is the discrete adjoint of the same CN scheme, checked against finite differences in `test_novelty.py`. In the joint arms the walls get a smoothness penalty (differences between neighbouring knots) with the same alpha as D. Costs are CPU seconds (`job_cpu_s`), so runs that shared a machine compare fairly.

### E2b: the Arrhenius data (`arrhenius.py`)

The truth is D(x, y, u) = D₀(x, y) · exp(s(u)), with s(u) = γθu / (1 + θu): Arrhenius in absolute temperature, T = T_ref(1 + θu), and γ = E_a / (R T_ref) = 20 (about 50 kJ/mol at 300 K). The strength S = s(1) is the log-ratio of D at the IC peak to the reference state:

| S | D ratio at the peak | θ | ΔT at 300 K |
|---|---|---|---|
| 0 | 1 (standard benchmark, control) | 0 | 0 |
| 0.5 | 1.6 | 0.026 | 8 K |
| 1 | 2.7 | 0.053 | 16 K |
| 2 | 7.4 | 0.111 | 33 K |

- **Generator:** the project's discretisation (129², harmonic-mean links, RK4, the same step rule), with D recomputed from u at every RK4 stage. At S = 0 it equals the linear RK4 solver to machine precision (`test_novelty.py`).
- **Sensors and noise:** each strength reuses the same sensors and the same noise draw.
- **Problem files** describe the standard D(x, y) model. The true s(u) is only in the ground truth.
- **Scoring:** against D₀ = D(x, y, u = 0). For `pinn_T`, also g(u) against s(u).

### E3b: designed sensors (`sensor_design.py`)

Greedy D-optimal design on the adjoint's sensitivities J (readings × 256 controls). J is computed at the prior constant D = Fo / T, the diffusivity the experiment was planned for, so no readings and no true D are used. All layouts are row subsets of the 3,969-node dataset, so they share one noise draw.

## Results (full size)

Median relL2(D); replay RMS/σ in brackets (re-simulated with the method's own estimated source or walls). Every method sees the same readings, grid, initial/boundary conditions and source. The win rule: more than 1 % lower error.

| Setting | CN adjoint | PINN | combined PINN→adjoint | joint classical | true-physics bound |
|---|---|---|---|---|---|
| known physics (20) | **0.226** (0.97) | 0.468 (3.56) | **0.225** (0.96) | – | – |
| hidden walls (8) | 4.36 (15.0) | 0.338 (2.48) | 0.296 (0.92) | **0.264** (0.98) | 0.201 |
| hidden source (6) | 3.29 (0.99) | 0.534 (5.63) | 3.57 (0.99) | **0.169** (0.97) | 0.156 |

The hidden-walls "combined" entry is `hybrid_bc_joint5`, the joint inversion started from the PINN's walls.

- **E1, the combined method.** It ends at the same final α as the cold adjoint on 20 of 20 datasets, because the strongly regularised start of the α path washes out any initial field. So it matches the adjoint and does not beat it. It costs about 3× more, because our 2-D adjoint is cheap. `hybrid_balance` beats both parents consistently (at least 4 of 5 seeds) on 4 of 20 sharp or noisy datasets. `hybrid_prior` hurts (0.323). `reverse` is no better than the PINN (0.476).
- **E1b / E1s, unknown physics.** The adjoint fails only because it enforces the wrong PDE. Once it estimates the missing walls or source jointly with D, it beats the PINN on 7 of 8 and 6 of 6 datasets and comes close to the bound. With the hidden source, the adjoint that assumes f = 0 replays at 0.99σ with relL2 3.29: **a good fit at the sensors is not a recovered field.**
- **S1, seeds.** Seed 42 was the PINN's best of five (per-seed medians 0.468–0.554), and its seed-to-seed spread is about 0.06. The PINN beats the adjoint in 3 of 100 dataset–seed pairs, all on one inclusion dataset.
- **D1–D3, density.** From 20 to 3,969 sensors (1 % noise) the PINN improves only from 0.481 to 0.404, while the adjoint goes from 0.156 to 0.070. Training 16× longer takes the PINN from 0.384 to 0.132. Given the exact u, its D-step reaches 0.094. The limit is the PINN's fitting and differentiating of its own û, not the data.
- **E2, temperature dependence.** A learned g(u) is worse on 4 of 6 datasets and invents temperature dependence. With Arrhenius data the adjoint still wins on 14 of 16 datasets; the PINN wins only at S = 2 with dense, clean data.
- **E3, PINN practice.** Held-out sensor validation picks a better model on 7 of 8 datasets. RAR is worse on 8 of 8. Pseudo-data wrecks the adjoint (0.317 → 3.38). Sensor placement matters for the adjoint.
- **Study 5 (`../study5_time_integrators/`).** Backward Euler biases D upward when its time-step error exceeds the noise (2.5σ at 1 % noise). On data generated by backward Euler itself it inverts as well as Crank–Nicolson. This explains the PINN's intensive-phase "dense-sensor win".

Detailed tables: `summary/*.md`. Write-ups: `summary/S1.md`, and `../study5_time_integrators/BE_MECHANISM.md`. Poster figures: `poster_figs/`. Poster source in the official template: `poster/`.

## Fairness rules

- The true D is read only for scoring, never for training, model selection or stopping. E3a's "oracle" column is labelled as such and is not a method.
- PINN arms use the project's PINN (`submission/pinn_inversion.py`) through `pinn_plus.py`. With every option off, `pinn_plus.train` reproduces `train_single_experiment` exactly: zero difference in D after 40 epochs, same seed (`test_novelty.py`). Options only add to the loop: the third network, held-out rows, adaptive points, pseudo-readings or a warm start.
- Same seed (42), epochs (1,500 full, 300 quick), loss weights and optimiser for every PINN arm unless the arm is about changing them (E3a).
- **Scoring:** every estimate goes through the same functions (`common.py`): relL2(D) on the 65² grid, and replay RMS/σ through the CN solver.
- **Unknown-physics datasets (Study 3):** the PINN stage uses the boundary-aware PINN (`core/pinn_inversion_bc.py`), and the adjoint stage runs with the assumed zero walls / no source, exactly as in Study 3.

## Running it

From `extended_studies/`:

```bash
pip install -r ../submission/requirements.txt sympy tabulate

# 1. data (once; about 30 min on a laptop CPU)
python run.py core/generate_dataset_v2.py                      # datasets_v2_rk4_zero/    (E1, E2a, E3)
python run.py study1_representation/extend_sensor_density.py   # datasets_v2_density_extended/ (E3b source)
python run.py study3_pinn_favoured/generate_pinn_favoured.py   # datasets_v2_pinnwin/     (E1 unknown physics)
python run.py novelty/arrhenius.py --size full                 # datasets_novelty_arrhenius/ (E2b)
python run.py novelty/sensor_design.py --size full             # datasets_novelty_oed/   (E3b)

# 2. checks (about 4 min)
python run.py novelty/test_novelty.py

# 3. experiments: a pilot first, then the full size
python run.py novelty/run_novelty.py --exp E1 --size quick
python run.py novelty/run_novelty.py --exp E1 E2a E2b E3a E3b E3c E3d --size full
python run.py novelty/run_novelty.py --exp E1b E1c D1 D2 D3 --size full     # follow-ups

# 4. tables and figures -> novelty/summary/
python run.py novelty/summarise_novelty.py
```

**Kaggle / Colab:** `kaggle_novelty.ipynb` does all of the above for one shard: it clones the repo, installs the packages, builds the data its experiments need, runs the checks and the experiments, and zips the results for download. Set `EXPS`, `SIZE` and `SHARD` in its second cell.

**Parallel workers:** add `--shard i/n` to split by dataset, for example four notebooks with `--shard 0/4` … `--shard 3/4`. All arms of one dataset stay in one worker because the hybrids read the PINN's result. Every finished job writes `novelty/results/<exp>/<arm>/<key>/metrics.json` and is skipped next time, so a stopped session resumes with the same command. Set `NOVELTY_RESULTS=/kaggle/working/results` to write outside the repo. Afterwards, copy all shards' `results/` folders into one before summarising.

**Run time** (one CPU core, measured in the pilots): a full PINN (1,500 epochs) takes about 2 min and an adjoint 0.5–3 min. The boundary-aware PINN on the Study 3 datasets is slower, and `adjoint_cv` walks the full α path twice (about 5 min). Approximate full sizes:

| Experiment | Runs | Approx. CPU time |
|---|---|---|
| E1 | 34 datasets × 5–6 arms | 6–8 h |
| E2a | 6 × 2 PINN | 0.5 h |
| E2b | 16 × (adjoint + 2 PINN) | 1.5 h |
| E3a | 8 × (12 PINN + 2 adjoint) | 4 h |
| E3b | 27 × (adjoint + PINN) | 2 h |
| E3c | 8 × 2 PINN | 0.5 h |
| E3d | 6 × (2 adjoint + 2 PINN) | 0.7 h |

A GPU speeds up the PINN arms; the adjoint arms are CPU code (SciPy) either way.

## Files

| File | What it does |
|---|---|
| `common.py` | dataset collections, scoring (relL2, replay), result files |
| `pinn_plus.py` | the project PINN with switchable extensions (third network, held-out rows, RAR, augmentation, warm start) |
| `hybrid.py` | projection to the adjoint controls, warm-started and prior-regularised adjoint, cold adjoint with timing |
| `hybrid_bc.py` | E1b: adjoint with settable / optimisable wall temperatures, joint (D, walls) inversion, the boundary-aware PINN's walls |
| `hybrid_src.py` | E1s: adjoint with a source term, joint (D, f) inversion with exact discrete-adjoint gradient (checked against finite differences) |
| `summarise_seeds.py` | S1 seed-spread tables (`summary/S1.md`) |
| `make_poster_figures.py` | poster figures F3–F12 (`poster_figs/`); F1, F2 are TikZ in `poster_figs/tikz/` |
| `build_app_bundle.py` | copies slim datasets and every method's estimate into `submission/app/demo/` for the interface |
| `poster/` | the A0/A1 poster in the official IE 643 template (LaTeX source, figures) |
| `arrhenius.py` | temperature-dependent data generator and nonlinear replay |
| `sensor_design.py` | random / grid / D-optimal sensor layouts as datasets |
| `experiments_more.py` | E2 and E3 job definitions |
| `run_novelty.py` | experiment registry, quick/full sizes, shards, resumable jobs |
| `summarise_novelty.py` | tables and figures per experiment |
| `test_novelty.py` | checks: exact port, temperature-dependent residual vs finite differences, projection, Arrhenius at S = 0 |
| `kaggle_novelty.ipynb` | one-shard runner for Kaggle / Colab |
