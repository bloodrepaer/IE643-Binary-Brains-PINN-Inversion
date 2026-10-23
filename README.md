# PINN-based inverse estimation of a spatially varying diffusion coefficient

**IE 643 course project · Team Binary_Brains**
Jash Taparia (24b2451) · Mihir Gune (24b2443)

We recover an unknown diffusivity map D(x, y) in the 2D heat equation

    u_t = ∇·( D(x, y) ∇u )      on the unit square, known initial state, Dirichlet boundaries

from sparse, noisy sensor readings of u. We compare a **physics-informed neural network (PINN)** with **classical PDE-constrained optimisation (the adjoint method)** on the same synthetic data. The data come from a hand-built finite-difference forward solver.

---

## 1. What is in this archive

| Folder | Contents | Start with |
|---|---|---|
| `submission/` | **Main deliverable, ready to run.** Forward solver, dataset generator, the full 389-case benchmark dataset, classical adjoint inversion, PINN, Streamlit interface, trained models for both methods, and the follow-up experiments (sensor/noise/layout sweep, 20-dataset study, solver-bias test, paper verification). | `submission/README.md` |
| `neural_campaign/` | **Later extension.** D is represented by the *same* neural network in both methods (neural adjoint vs neural PINN). Includes the "where should a PINN win?" hypothesis suites, sensor-density sweeps up to 10,000 sensors, trained checkpoints, score tables, reports, and a second interface for comparing and testing checkpoints. | `neural_campaign/README_neural.md`, `neural_campaign/REPOSITORY_MAP.md` |
| `docs/` | Written material: stage-wise report draft, theory guide, defence guide (claims with evidence), Thakur et al. verification, 20-dataset table, consolidated research log, interface screenshots. | `docs/project_report.md` |

`submission/` is laid out as a repository root and can be pushed to GitHub as-is. See §7 for size limits.

## 2. Deliverables: where each one is

### Project brief

| # | Deliverable | Where |
|---|---|---|
| 1 | Synthetic dataset and finite-difference forward solver | `submission/integrators.py`: 5-point flux-form operator with harmonic-mean links; RK4, Crank–Nicolson with Rannacher start, forward Euler. `submission/domain.py`: six ground-truth D families. `submission/generate_dataset_v2.py` and `submission/datasets_v2_rk4_zero/`: 389 datasets covering sensors, noise, layout, time protocol and horizon. Integrator check: `compare_integrators.py`, `figures/`. Documentation: `submission/README_dataset_and_classical.md`. Further dataset generators: `neural_campaign/generate_pinn_favoured*.py`, `make_density_datasets.py`, `extend_sensor_density.py`, `prepare_noise_density_1000_7500.py`. |
| 2 | PINN inverse model | `submission/pinn_inversion.py`: temperature network with Fourier features and IC/BC structure, log-D network, five-term loss, AdamW with cosine schedule. Checkpoints in `submission/models/pinn/`. Neural-coefficient variants in `neural_campaign/`: `compare_neural.py`, `pinn_inversion_bc.py` (unknown boundaries), `pinn_inversion_v2.py`, `pinn_papers.py`, `temporal_density_pair.py`, `noise_density_1000_7500_pair.py`. |
| 3 | Classical PDE-constrained inversion | `submission/adjoint_inversion.py`: exact discrete adjoint of CN, log D on a 16×16 control grid, Tikhonov or TV, L-BFGS-B, regularisation weight by the discrepancy principle. `submission/test_adjoint.py`: gradient checks. Estimates for all 389 datasets in `submission/models/classical/tikhonov_all389/`. Neural adjoint: `neural_campaign/adjoint_nn.py`, `matched_temporal_adjoint.py`. |
| 4 | Where PINN inversion beats classical inversion | (a) Unknown boundary condition or source: `neural_campaign/README_pinn_favoured.md`, `results_pinnwin_scores.csv`, `results_pinnwin2_scores.csv`. (b) Very dense sensors: `neural_campaign/campaign/pilots/noise_density_v24_1000_7500/SWEEP_REPORT.md`, plus the 10,000-sensor pair in `campaign/pilots/dense10000_v21_gpu/`. (c) Graceful degradation with too few sensors: `submission/experiments/sensor_noise_layout_sweep/`. Summary in §4. |
| 5 | Sensor density, location and noise experiments | `submission/experiments/sensor_noise_layout_sweep/` (both methods; 5/20/50/100 sensors × 0/1/5/10 % noise × random/grid/clustered/boundary layouts). `submission/models/classical/tikhonov_all389/summary_all.csv` (all 389 configurations). `neural_campaign/campaign/density_*.csv` (180 configurations, 5 to 3,969 sensors). `neural_campaign/campaign/pilots/noise_density_v24_1000_7500/` (1,000 to 7,500 sensors × 1/5/10 % noise). |
| 6 | Quantitative and qualitative evaluation | Metrics: relL2(D), replay RMS/σ, and a best-constant baseline (§5), in every `metrics.json` and summary CSV. Qualitative: true/estimated/error maps (`summary.png` in each `models/classical/...` folder, `submission/experiments/*/figs`, `neural_campaign/campaign/figures/`, `analysis/figures/`). |

### Submission requirements

| Requirement | Where |
|---|---|
| Trained models, with code, documentation and dependencies | `submission/models/`: PINN checkpoints `pinn_best.pt` / `pinn_final.pt`, and classical D̂ for all 389 datasets. `submission/app/pretrained/`: both methods on 20 datasets. `neural_campaign/campaign/results/`: neural-adjoint checkpoints `model.pt`, D̂ and metrics. `neural_campaign/campaign/v24_case_results/` and `campaign/pilots/*/artifacts/`: dense-sensor PINN and adjoint checkpoints. Dependencies: `requirements.txt`. |
| Interface for user inputs | `submission/app/streamlit_app.py`: generate a dataset (choose D, sensors, layout, noise, BC, or upload your own D), pick a benchmark dataset, or upload sensor readings as CSV. Then run either or both methods, view maps and metrics, download models, or reload a saved model. Sample inputs are in `submission/app/sample_inputs/`. Screenshots are in `docs/screenshots/`. Second interface: `neural_campaign/neural_interface.py`. |
| Performance metric displayed | The interface shows relL2(D), RMS of the log-D error, replay RMS/σ, the best-constant baseline, and the field error map. |
| Stage-wise report | `docs/project_report.md` (draft) and `docs/theory_guide.md`. **Still needed: a PDF in the course template** (§6). |

## 3. Install and run

Python 3.10 or newer. A CPU is enough.

```bash
pip install -r requirements.txt          # numpy, scipy, pandas, matplotlib, torch, streamlit, sympy

cd submission
python compare_integrators.py            # integrator orders of accuracy (~15 s)
python test_adjoint.py                   # adjoint gradient checks, prints ALL PASSED (~5 s)
python adjoint_inversion.py --keys C_inclusion__Fo0.1__uniform8__n50__random__eta0.05 --reg tv   # ~1 min
python pinn_inversion.py --single-run --data-root . --output-dir results_pinn \
       --exp-id B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05 --epochs 1500        # 2-4 min
streamlit run app/streamlit_app.py       # interface at http://localhost:8501
```

`python generate_dataset_v2.py` rebuilds `datasets_v2_rk4_zero/` from fixed seeds in about 4 minutes. More commands are in `submission/README.md`.

**Neural campaign:**

```bash
cd neural_campaign
# The code expects the benchmark next to it: copy it (or run python generate_dataset_v2.py here)
#   Windows:      xcopy /E /I ..\submission\datasets_v2_rk4_zero datasets_v2_rk4_zero
#   Linux/macOS:  ln -s ../submission/datasets_v2_rk4_zero datasets_v2_rk4_zero
python -m unittest test_neural -v
python compare_neural.py --keys B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05 --out results_neural_B
python -m streamlit run neural_interface.py
```

The saved-results tab of `neural_interface.py` reads paths stored in `campaign/*.csv`. Those paths were written on Windows with `\` separators, so run that tab on Windows. The checkpoint-testing tab works on any OS.

**Checks run on this package** (fresh copy, Linux, Python 3.13):
- integrator orders 4.03 / 2.01 / 1.00
- `test_adjoint.py`: all passed (Taylor slopes 2.00; adjoint vs finite differences 1e-8)
- the regenerated dataset matches the shipped manifest for all 389 entries, to floating-point round-off
- one classical and one PINN inversion run
- the main Streamlit app loads, and the pretrained, upload and generate paths work
- `test_neural` passes, a short `compare_neural.py` run completes, and all 42 saved dense-sweep checkpoints load and predict

## 4. Main results

| Study | Classical adjoint | PINN |
|---|---|---|
| 20 representative datasets (known IC/BC): lower relL2(D) | 20 / 20 | 0 / 20 |
| … median relL2(D), lower is better | 0.23 | 0.47 (best constant 0.38) |
| 44-dataset sensor × noise × layout sweep: lower relL2(D) | 41 / 44 | 3 / 44 |
| … median relL2(D) | 0.36 | 0.61 (best constant 0.68) |
| Fits data to the noise level (replay RMS/σ ≈ 1), 20-dataset study | yes (median 0.97) | no (median 3.7) |
| **Unknown boundary condition or source** (22 hypothesis-suite cases; adjoint must assume zero) | median relL2 5.0 | **0.38, lower in 21 / 22** |
| Known physics, same hypothesis suites (59 cases) | median 0.12, lower in 58 / 59 | median 0.41 |
| **Very dense sensors**, neural adjoint vs neural PINN, smooth field, 5 % noise, 7,500 sensors | 25.1 % | **8.6 %** (both pass the replay checks) |
| … 10,000 sensors (reference pair) | 26.4 % | **6.8 %** |

**Where the PINN is better (Deliverable 4):**
1. **The boundary condition or heat source is not known.** The standard adjoint must assume a model (zero boundaries, no source), and its estimate breaks down. With an unknown boundary, the PINN leaves the wall values free (`pinn_inversion_bc.py`, mode `free`). With an unknown source, its soft PDE penalty tolerates the mismatch. In both cases it stays below the best-constant error. A stronger classical baseline would add the boundary values or the source as extra unknowns. We did not implement that, so this is a win over the *standard* adjoint formulation.
2. **Very dense sensor data.** At 3,000 or more sensors and 5 % noise, the PINN's coefficient error keeps falling with density (35 % at 1,000 sensors to 8.6 % at 7,500), while the matched neural adjoint stays near 21–25 %. Only the 7,500-sensor case passes both replay-consistency checks, and this is one field, one seed and one noise draw.
3. **Too few sensors to identify D.** With 5 random or boundary-only sensors on the inclusion field, the PINN stays at the constant level while the adjoint overshoots. That is safer behaviour, not a better reconstruction.

With known initial and boundary conditions and up to a few hundred sensors (the standard course setting), the classical adjoint is more accurate, fits the data to the noise level, and is faster.

## 5. Metrics

- **relL2(D) = ‖D̂ − D‖ / ‖D‖** on the 65×65 grid. This is the main ranking metric; lower is better.
- **Replay RMS / σ**: the recovered D is run through the Crank–Nicolson solver and compared with the sensor readings. A value near 1 means the data are fitted to the noise level, well below 1 means over-fitting, and well above 1 means under-fitting.
- **Best-constant baseline**: the relL2 of the best single constant D. A method is useful only if it beats it.

## 6. Still to do for Stage 1 (not in this archive)

The intensive assessment is submitted through a Google Drive folder named `Binary_Brains_IE643_CourseProject_IntensiveAssessment`, shared as "anyone with the link". It must contain:

1. **Project report (PDF, course template)**: from `docs/project_report.md`, with the GitHub link to this code.
2. **Presentation slides**: at most 18, excluding references.
3. **Code-walkthrough video**: at most 10 minutes, faces visible, covering data generation, the networks, the loss, the optimiser and learning-rate settings, and the metrics.
4. **Demo video**: 5 minutes, using `submission/app/streamlit_app.py`.

## 7. Notes

- **Left out to keep the archive small; regenerate with the scripts:**
  - the other dataset collections (`datasets_v2_pinnwin*`, `datasets_v2_density*`, `datasets_v2_noise_density_*`, `datasets_v2_dense*`), via `generate_pinn_favoured.py`, `generate_pinn_favoured2.py`, `generate_pinn_favoured3.py`, `make_density_datasets.py`, `extend_sensor_density.py`, `prepare_noise_density*.py` and `prepare_dense*.py`
  - per-dataset `summary.png` figures of the neural campaign
  - logs, caches, Kaggle/Colab transfer bundles, and smoke-test and packaging runs
- `submission/` and `neural_campaign/` each contain their own copy of the shared modules. In `neural_campaign/`, `adjoint_inversion.py` adds a `field_objective` method used by the neural adjoint. `pinn_inversion.py` differs between the two folders: each folder's scripts use their own copy.
- Some experiment scripts contain the absolute paths of the machines they ran on (`/home/claude/...`). Change them before re-running those scripts.
- **GitHub:** `submission/datasets_v2_rk4_zero/` is about 175 MB. Leave it out of the repository (it is regenerable) or use Git LFS / a release. Every other file is under GitHub's 100 MB limit.
- Reference papers are not bundled. See `submission/README.md` §6 and `docs/`. The main ones are Raissi et al. 2019 (J. Comput. Phys.), Zhang, Alla & Karniadakis 2026 (arXiv 2606.12337) and Thakur et al. 2024 (arXiv 2403.03970).
