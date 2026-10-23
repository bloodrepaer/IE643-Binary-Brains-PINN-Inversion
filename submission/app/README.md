# Diffusivity inversion app (IE 643 · Binary_Brains)

This is an interactive interface for the project's two inverse solvers. It covers deliverable "an interface where the user can input or upload project-specific inputs", the displayed performance metric and the qualitative results.

You give it sensor readings of the heat equation u_t = ∇·(D(x,y)∇u) on the unit square. It recovers the unknown diffusivity map D(x,y) two ways:

- the **classical PDE-constrained (adjoint) inversion** in `adjoint_inversion.py`;
- the **PINN** in `pinn_inversion.py`.

It reports metrics for both, draws the recovered maps and lets you download the trained models.

---

## 1. Install and run

```bash
cd ie643_inversion_app            # the folder that contains app/, domain.py, ...
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt   # CPU is enough; torch installs the CPU or CUDA build for your machine
streamlit run app/streamlit_app.py
```

The app opens in your browser at http://localhost:8501. On Windows, use `.venv\Scripts\activate`.

Tested with:

| Package | Version |
|---|---|
| Python | 3.13 |
| numpy | 2.5 |
| scipy | 1.18 |
| torch | 2.14 |
| pandas | 3.0 |
| matplotlib | 3.11 |
| streamlit | 1.66 |

Any recent version of each should work.

**Hardware.** Everything runs on a laptop CPU. Typical times on 2 CPU cores:

| Step | Time |
|---|---|
| Generate a dataset | 5–40 s |
| Classical inversion (16×16 control grid) | 40–120 s |
| PINN, 1,500 epochs | 2–4 min |

A GPU is used by the PINN automatically if torch finds one.

---

## 2. Folder layout

```
ie643_inversion_app/
├── app/
│   ├── streamlit_app.py      user interface (tabs below)
│   ├── core.py               backend: data I/O, running both solvers, scoring, model save/load
│   ├── sample_inputs/        files to try the upload paths with
│   │   ├── example_sensors.csv              400 readings (50 sensors × 8 times, 5 % noise)
│   │   ├── example_true_D_65x65.csv         the matching true D (optional, for scoring)
│   │   └── example_custom_D_Lshape.npy      a custom true D for the "generate" path
│   └── pretrained/           trained models from the 20-dataset comparison
│       ├── <dataset key>/pinn_best.pt        PINN checkpoint (1,500 epochs)
│       ├── <dataset key>/adjoint_D_hat.npz   classical estimate of D
│       ├── <dataset key>/*_metrics.json
│       └── comparison_20datasets.csv
├── domain.py, integrators.py, generate_dataset_v2.py     project forward solver and data generator
├── adjoint_inversion.py, pinn_inversion.py               project inverse solvers: the team's full files,
│                                                         command-line interfaces included (used unchanged)
├── datasets_v2_rk4_zero/     the 20 benchmark datasets used in the comparison (manifest + files)
├── requirements.txt
└── README.md
```

The solvers also run without the app, exactly as in the project:

```bash
python adjoint_inversion.py --keys B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05
python pinn_inversion.py --single-run --data-root . --output-dir results_pinn \
       --exp-id B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05
```

Their output files (`D_hat.npz` from the classical solver, `pinn_best.pt` / `pinn_final.pt` from the PINN) open in tab 4 of the app.

The full benchmark has 378 datasets and takes about 145 MB. To use it, regenerate it in place (about 3 minutes); the app picks it up automatically:

```bash
python generate_dataset_v2.py
```

---

## 3. Using the interface

### Tab 1 · Data

Choose one of three sources:

1. **Generate a new synthetic dataset.** This is the same pipeline as the project generator:
   - The forward solve is RK4 on a 129×129 grid.
   - Clean readings are interpolated at the sensor positions.
   - Gaussian noise is added with σ = η × RMS(clean signal).

   You choose:
   - the true D, either one of the 17 project fields (tiers A–F and variants) or your own 65×65 or 129×129 array;
   - the time horizon Fo;
   - the number of sensors (5–100, or every grid node) and their layout (random, grid, clustered or boundary);
   - when the sensors record (10 schedules);
   - the noise level, the boundary condition and the noise seed.

2. **Pick a benchmark dataset.** Filter by true D, Fo, times, sensors, layout and noise.

3. **Upload my own readings.** Upload a CSV of readings. Optionally also upload:
   - an initial condition (otherwise the benchmark's two Gaussian bumps are used);
   - the true D, so that field errors can be computed.

The app then shows:
- the true D with the sensor positions;
- the initial temperature;
- the readings over time;
- for generated data, a slider through the simulated temperature field.

### Tab 2 · Run inversion

Tick either method or both.

| Classical adjoint | PINN |
|---|---|
| control grid K×K (8–32) | epochs (100–6,000) |
| Tikhonov or TV regulariser | regulariser (auto, Tikhonov, TV or none) |
| L-BFGS iterations per α | regularisation weight |
| α picked by the discrepancy principle | random seed |

### Tab 3 · Results & downloads

**Performance panel.** For each method it shows:
- **relL2(D)** = ‖D̂ − D‖ / ‖D‖. Lower is better. Shown only if the true D is known.
- **RMS replay / σ.** The estimated map D̂ is run through the same Crank–Nicolson solver and compared with the readings. About 1 means D̂ explains the data to the noise level. This metric is always available, even for real data with no answer key.

A table adds:
- the RMS of the log-D error;
- the range of D̂;
- the wall time;
- the **best constant D** baseline, i.e. the single uniform value that fits the readings best. A useful map should beat it.

**Qualitative results:**
- true vs recovered maps on a shared colour scale, with log(D̂/D) error maps;
- profiles through the centre;
- the α-selection curve and a data-sensitivity map (where the readings carry information about D);
- the PINN loss curves.

**Downloads:**
- one zip with the models, D̂ as CSV and NPZ, metrics.json, all figures and the readings used;
- or each trained model and each D̂ CSV separately.

### Tab 4 · Load a saved model

You can either:
- upload a `classical_model.npz` or `pinn_best.pt` saved from tab 3; or
- pick one of the bundled pre-trained models.

The D map is rebuilt from the model itself:
- for the classical model, from the control-grid coefficients θ;
- for the PINN, from the network weights.

You can then score the map on the current dataset, or on its own benchmark dataset, and export it as CSV. A table lists the scores of all 40 bundled models.

---

## 4. Input formats

| Input | Format |
|---|---|
| Sensor readings | CSV with header `x,y,t,u` and optional `sigma`. One row per reading; x and y in [0, 1]; t > 0. If there is no `sigma` column, the σ entered in the form is used (0 = noise-free). |
| Final time T | Number, at least the latest reading time. |
| Initial condition u₀ | `.npy` of shape 65×65, or `.csv` as a 65-row matrix, or `.csv` with columns `x,y,value`. Index order [i_x, i_y]. |
| True D (scoring, or a custom D for generation) | Same as u₀. For generation, 129×129 is also accepted; a 65×65 array is interpolated onto the data grid. |
| Saved model | `.npz` with `theta`, `control` (classical) or `D_hat` (any array), or a `.pt` PINN checkpoint as written by `pinn_inversion.py`. |

---

## 5. How the interface was built (methodology for the report)

**Design goals:**
1. Anyone can go from inputs to a trained model and a metric without writing code.
2. Both methods are scored in exactly the same way.
3. Nothing in the solvers is re-implemented, so what the interface shows is what the project code does.

**Architecture.**
- `core.py` is a thin backend with no UI code. It:
  - writes every data source (generated, benchmark or uploaded) to the project's own on-disk dataset format: `problem/`, `ground_truth/`, `sensors/` and `manifest.json`;
  - calls `adjoint_inversion.AdjointSolver` and `pinn_inversion.train_single_experiment` unchanged.

  Because every source becomes a standard dataset folder, both solvers' original loaders work on user data.
- `streamlit_app.py` holds only widgets and figures. Results are kept in the Streamlit session state, so switching tabs does not retrain.

**Data generation.** This is the same code path as `generate_dataset_v2.py`:
- `Domain2D`, `get_D_field`, `ForwardModel(integrator="rk4")` on 129² to produce the data;
- `sigma_reference`, `read_clean` and the same noise model.

The inverse solvers work on a 65² grid with Crank–Nicolson. The difference in grid and time integrator avoids an "inverse crime".

**Uploads.** Uploaded readings are validated (columns, ranges, t > 0). The best uniform D is then fitted, and Fo = D̄·T is estimated from it, because the PINN's time scaling needs Fo. If an initial condition is uploaded, it replaces the default in the problem files.

**One metric for both methods.** Each D̂ is pushed through the same Crank–Nicolson forward solver, and the RMS misfit to the readings is divided by σ. This is the displayed performance metric. relL2(D) and the RMS of the log-D error are added when the truth exists. The PINN's own u-network χ² is shown but not used as the score: the network can fit the readings while its D is wrong.

**Models.**
- Classical: saved as its control-grid coefficients θ, plus metadata (K, regulariser, α) and the D̂ array.
- PINN: saved as the best checkpoint, with the config, weights and normalisation.

Tab 4 rebuilds D from either file, which confirms that the saved model alone reproduces the result.

**Testing.**
- Every path was exercised headlessly with `streamlit.testing.v1.AppTest`:
  - benchmark + both methods;
  - generate + classical;
  - upload + PINN;
  - reload of a `.pt` and a `.npz`;
  - the bundled models.
- The bundled models, rescored in the app, reproduce the numbers stored in `comparison_20datasets.csv` exactly. For example, C_inclusion n50, 5 %:

  | Method | relL2 | RMS/σ |
  |---|---|---|
  | Classical | 0.661 | 0.98 |
  | PINN | 0.728 | 2.33 |

- The live app was then driven in a real browser (Chromium/Playwright) end to end.

---

## 6. Limitations

- The project PINN's base solution hard-codes the benchmark's initial condition (two Gaussian bumps). With an uploaded, different initial condition, the PINN will do poorly. The classical method is not affected.
- Uploaded data must be on the unit square, with Dirichlet boundaries from one of the four project scenarios.
- In the 20-dataset comparison, the classical method had the lower D error on all 20 datasets. The median PINN/classical ratio was 2×, and the closest case was C_inclusion at 1.03–1.10×. The app makes it easy to check any new configuration.
