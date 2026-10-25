# Datasets

All data in this project are synthetic. We choose a ground-truth diffusivity D(x, y), simulate the heat equation with our own forward solver, and read the temperature at sparse sensors with added noise. The inverse methods see only the sensor readings and the problem description; the true D is kept apart and used only for scoring.

$$\frac{\partial u}{\partial t} = \nabla\cdot\big(D(x,y)\,\nabla u\big)\quad\text{on } [0,1]^2,\qquad f = 0$$

The generated files are not stored in the repository (about 550 MB in total). Every collection marked "yes" below is rebuilt exactly by its generator (fixed seeds); a copy is also in the repository's GitHub Release section.

## 1. Collections

| Collection | Datasets | What varies | Generator | Rebuildable |
|---|---:|---|---|---|
| `datasets_v2_rk4_zero` | 389 | **Main benchmark**: 6 D families, sensor count, layout, noise, time horizon, reading schedule (sections 2-5) | `submission/generate_dataset_v2.py` | yes |
| `datasets_v2_pinnwin` | 57 | Hypothesis suites H1-H5: very sparse sensors, unknown boundary condition, unknown heat source, heavy noise, long horizons | `extended_studies/study3_pinn_favoured/generate_pinn_favoured.py` | yes |
| `datasets_v2_pinnwin2` | 24 | Hypothesis suites H6, H7, H9: more readings, unknown BC with more readings, D finer than the adjoint's control grid | `extended_studies/study3_pinn_favoured/generate_pinn_favoured2.py` | yes |
| `datasets_v2_pinnwin3` | 6 | H10u: unknown BC with 400 or 3,249 sensors, 0 or 1 % noise | unpacked from `kaggle_thakur/dataset/data.zip` | no (archive not committed) |
| `datasets_v2_density` | 4 | A_smooth, noise-free, 40 times, nested 25 / 100 / 400 / 3,969 sensors | `extended_studies/study1_representation/make_density_datasets.py` | yes |
| `datasets_v2_density_wideic` | 2 | As above with wider initial bumps, 400 and 3,969 sensors | same script with `--wide-ic` | yes |
| `datasets_v2_density_extended` | 180 | 6 fields x 10 sensor counts (5 to 3,969) x noise 1 / 5 / 10 %, nested subsets of the full-field readings | `extended_studies/study1_representation/extend_sensor_density.py` | yes (needs `datasets_v2_rk4_zero`) |
| `datasets_v2_dense10000` | 1 | A_smooth, 10,000 sensor sites x 8 times (80,000 readings), 5 % noise | `extended_studies/study4_dense_sensors/prepare_dense10000.py` | no (inputs not committed) |
| `datasets_v2_dense1000` | 1 | Fixed 1,000-site subset of the 10,000-site set | `.../prepare_dense1000.py` | no |
| `datasets_v2_noise_density_v23` | 7 | 1,000 / 3,969 / 10,000 sites x 1 / 5 / 10 % noise (5 % at 1,000 and 10,000 sites is in the two collections above); only the two 1,000-site sets were run | `.../prepare_noise_density.py` | no |
| `datasets_v2_noise_density_v24_1000_7500` | 9 | 3,000 / 5,000 / 7,500 sites x 1 / 5 / 10 % noise | `.../prepare_noise_density_1000_7500.py` | no |

The hypothesis suites are described in `extended_studies/study3_pinn_favoured/README.md`, the density collections in `extended_studies/study1_representation/README.md` and the dense-sensor collections in `extended_studies/study4_dense_sensors/README.md`. All collections share the file format of section 6.

To rebuild the main benchmark (about 3-5 minutes):

```bash
cd submission
pip install -r requirements.txt
python generate_dataset_v2.py                         # -> datasets_v2_rk4_zero/
python check_problem_spec.py datasets_v2_rk4_zero     # checks the problem files are sufficient
# options
python generate_dataset_v2.py --integrator cn         # or fe
python generate_dataset_v2.py --bc ramp_left          # zero | heated_left | ramp_left | pulsed_top
python generate_dataset_v2.py --quick                 # tier A only, coarse grids (smoke test)
```

## 2. How the data are made

**Forward solver** (`integrators.py`). Node-based finite volumes on [0,1]^2. Each link between neighbouring nodes carries the harmonic mean 2 D_i D_j / (D_i + D_j), because thermal resistances across a face act in series. Boundary nodes are eliminated (their values are known), so the semi-discrete system is du/dt = L u + b(t) with L symmetric negative definite.

**Time integrators**, measured on tier C at 65^2 against the exact-in-time `expm` solution (`compare_integrators.py`):

| Integrator | Role | Order (measured) | Step |
|---|---|---|---|
| RK4 | data generation (default) | 4.03 | stability-limited, <= 2.785 / rho(L); error about 1e-10 at its largest stable step |
| Crank-Nicolson | used by the inversion and for replay | 2.01 | free (default T/400), Rannacher start |
| Forward Euler | benchmark only | 1.00 | stability-limited, <= 2 / rho(L) |

**Two grids.** D and u are simulated on a 129 x 129 grid; the inversions work on 65 x 65. The inversion-grid truth is the exact restriction `D_inv = D_data[::2, ::2]`. Simulating on a finer grid with a different time scheme from the inversion avoids the inverse crime. At 129^2, CN and FE data differ from RK4 by 0.2 % and 0.5 % of the smallest noise sigma, so the fine grid, not the integrator, is what matters here.

**Noise.** sigma = eta x RMS of a fixed reference signal (100 random sensors, `uniform40` times), with eta in {0, 1, 5, 10 %}. The noise is Gaussian and added to the readings interpolated from the clean field.

## 3. Ground-truth D(x, y)

| Tier | Field | Variants (baseline point only) |
|---|---|---|
| A | smooth sinusoid (sanity check) | |
| B | Gaussian bump (localised feature) | |
| C | circular inclusion, tanh-smoothed edge | contrast 2, 7.5 (default), 15 |
| D | layered / checkerboard (multiple scales) | checkerboard 2x2, 6x6 |
| E | Gaussian random field (spectral filter); the correlation length is a difficulty dial | correlation length 0.03, 0.1 (default), 0.3 |
| F | uniform plate (D0 = 0.25) with circular defects where D is multiplied by a factor (tanh-smoothed rim). Default: 3 defects, radius 0.07, two conducting 3x worse and one 3x better | `single` (1 defect), `many` (6, radius 0.05), `small` (radius 0.035), `low_contrast` (x1.5 / x0.67), `voids` (all 5x worse) |

Tier F defect positions are seeded and nested: every variant shares the same first centres, so variants differ only in the property being varied. The defect list (centre, radius, factor) is stored in each ground-truth file as `defects` (JSON). Tier E is a single random-field realisation on both grids.

**Initial condition:** two Gaussian bumps (peaks 1.0 and 0.7, widths 0.05 and 0.06), the same for every tier.

**Boundary conditions:** Dirichlet on each edge, constant or time-varying. Scenarios: `zero` (default, used by the main benchmark), `heated_left` (left = 1 from t = 0), `ramp_left` (left = t/T), `pulsed_top` (top = (1 - cos 6 pi t/T) / 2).

## 4. Experiment axes of the main benchmark

| Axis | Values |
|---|---|
| Sensor count | 5, 20, 50, 100, full; "full" = all 3,969 interior nodes of the 65^2 grid |
| Sensor layout | `random`, `grid`, `clustered` (3 hot spots), `boundary` (within 0.1 of an edge) |
| Reading schedule | fixed-interval `uniform<n>`, n in {4, 8, 16, 40, 80, 200}; `log8` (early-heavy), `early8` (t <= T/4), `late8` (t >= T/2), `final1` (t = T only) |
| Time horizon | Fourier number Fo = mean(D) T / L^2 in {0.02, 0.05, 0.1, 0.2}; T is computed per tier |
| Noise | eta in {0, 1, 5, 10 %} |
| Field difficulty | tiers A-F and their variants |

**Sweeps** (per canonical tier; datasets shared between sweeps are written once):

| Sweep | Varies | Fixed |
|---|---|---|
| S1 spatial density x noise | sensors {5, 20, 50, 100, full} x eta {0, 1, 5, 10 %} | Fo 0.1, `uniform8`, random |
| S2 horizon x reading schedule | Fo {0.02, 0.05, 0.1, 0.2} x {`uniform40`, `uniform8`, `log8`, `early8`, `late8`, `final1`} | 50 random sensors, eta 5 % |
| S3 temporal density x noise | `uniform<n>`, n {4, 8, 16, 40, 80, 200} x eta {1, 5, 10 %} | Fo 0.1, 50 random sensors |
| S4 spatial layout | {random, grid, clustered, boundary} x {20, 50} sensors | Fo 0.1, `uniform8`, eta 5 % |

**Total:** 63 datasets per canonical tier x 6 tiers (A-F), plus one baseline dataset for each of the 11 variants = **389 datasets, 35 ground-truth files.** The number of readings per dataset ranges from 40 to 31,752.

**Controls that keep comparisons clean:**

- **Nested sensors.** For the random, clustered and boundary layouts, the n-sensor set is the first n points of one fixed 100-point pool, so 5 -> 20 -> 50 -> 100 only adds sensors.
- **Fixed noise level.** sigma is fixed per (tier, horizon, eta) from the reference signal, so changing the sensor count, layout or schedule never changes the noise level.
- **Paired data.** Sensor positions depend only on (tier, layout), so datasets that differ only in noise or schedule share the same sensors.
- **Deterministic seeds.** `seed_of(...)` uses `zlib.crc32`, so two runs give byte-identical files.
- **Grid alignment.** "full" datasets have sensors exactly on inversion-grid nodes; all other layouts are off-grid.

## 5. Verification of the data

| Check | Result |
|---|---|
| Time order of accuracy (vs exact `expm`) | RK4 4.03, CN 2.01, FE 1.00 |
| Heated edge vs analytic Laplace steady state | agrees to 4 digits |
| Time-varying boundary values | CN stays second order |
| Problem files are sufficient (replay from `problem/` with the true D, CN on 65^2) | RMS / sigma = 1.01 median (IQR 0.98-1.06), n = 296 noisy datasets (tiers A-E) |
| Determinism | two runs give byte-identical CSVs |
| Nested sensor sets | verified for random, clustered, boundary |
| Fixed time grids | every (tier, horizon, schedule) group shares identical times |

The replay result is also the floor for the replay metric: even a perfect estimate of D replays at about 1 sigma.

## 6. File format (`datasets_v2_<integrator>_<bc>/`)

```
problem/<tier>__Fo<Fo>.json      PDE, domain, T, IC expression, per-edge BC expressions in t (sympy-parseable)
problem/<tier>__Fo<Fo>.npz       u0 on both grids; g(t) at every inversion-grid boundary node, all 401 output times
problem/<tier>__Fo<Fo>__ic.csv   x,y,u0 on the inversion grid
sensors/<key>.csv                x,y,t,u,sigma   (one row per reading; also the interface's upload format)
sensors/<key>.npz                same + u_clean, sensor_id, time_index
ground_truth/<tier>__Fo<Fo>.npz  D_data, D_inv, 41 snapshots of u, T, integrator, dt   (answer key)
manifest.json                    every dataset: configuration, seeds, observation times, file paths
manifest.csv                     same, one row per dataset (without time lists)
config.json                      all generator constants
figures/                         per-tier overview, reading schedules, sensor layouts
```

Key format: `<tier>__Fo<Fo>__<schedule>__n<count>__<layout>__eta<eta>`, for example `C_inclusion__Fo0.1__uniform8__n50__random__eta0.05` (inclusion field, Fo = 0.1, 8 uniform snapshots, 50 random sensors, 5 % noise).

**`problem/` and `sensors/` are everything an inversion may use; `ground_truth/` is for evaluation only.** The noise level sigma is in the `sigma` column of each sensor file.

## 7. Known limitations

- **Layered and checkerboard interfaces** are hard thresholds sampled at nodes, so their discrete position depends on the grid. Even the true D replays the layered data at up to 4.6 sigma at eta = 1 %: discretisation error exceeds the noise there.
- **No manufactured-solution test with variable D.** Spatial order for variable D is verified only indirectly (grid convergence against a fine reference).
- **Outliers, sensor dropout and bias drift** are not generated.
- **Dirichlet boundaries only**; Neumann or Robin boundaries would need a different operator.
- **Exact initial condition.** Knowing u0 exactly everywhere is an idealisation (a perfect full-field measurement at t = 0).
