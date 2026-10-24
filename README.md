# PINN-based inverse estimation of a spatially varying diffusion coefficient

**IE 643 course project - Team Binary_Brains**
Jash Taparia (24b2451), Mihir Gune (24b2443)

We recover an unknown diffusivity map D(x, y) in the 2D heat equation

    du/dt = div( D(x, y) grad u )    on the unit square, known initial state, Dirichlet boundaries

from sparse, noisy sensor readings of u. We compare a **physics-informed neural network (PINN)** with **classical PDE-constrained optimisation (the adjoint method)** on the same synthetic data, produced by a hand-built finite-difference (finite-volume) forward solver.

---

## 1. Repository layout

| Folder | Contents |
|---|---|
| `submission/` | **Main code, ready to run.** Forward solver, dataset generator, classical adjoint inversion, PINN, Streamlit interface (`submission/app/`, with screenshots in `submission/app/screenshots/`), trained models for both methods, the sensor/noise/layout sweep and the 20-dataset evaluation. See `submission/README.md`. |
| `extended_studies/` | **Deliverable 4 evidence**, in four studies: (1) grid vs neural representation of D in the adjoint; (2) neural adjoint vs neural PINN with the same network for D; (3) the "where should a PINN win?" hypothesis suites; (4) the dense-sensor sweep (1,000 to 10,000 sensors). Trained checkpoints, score tables and run records. See `extended_studies/README.md`. |
| `report/` | LaTeX source of the stage-wise report (course template). |
| `videos/` | Code-walkthrough and demo videos (links in `videos/README.md`). |

## 2. Deliverables and where they are

| # | Deliverable | Where |
|---|---|---|
| 1 | Synthetic dataset and forward solver | `submission/integrators.py` (5-point flux form; RK4, Crank-Nicolson with Rannacher start, forward Euler), `submission/domain.py` (six ground-truth D families), `submission/generate_dataset_v2.py` (389 datasets), `submission/compare_integrators.py`, `submission/README_dataset_and_classical.md`. More dataset generators: `extended_studies/study3_pinn_favoured/generate_pinn_favoured.py`, `generate_pinn_favoured2.py`; `extended_studies/study1_representation/make_density_datasets.py`, `extend_sensor_density.py`; `extended_studies/study4_dense_sensors/prepare_noise_density_1000_7500.py`, `prepare_dense10000.py`. |
| 2 | PINN inverse model | `submission/pinn_inversion.py`; checkpoints in `submission/models/pinn/`. Boundary-aware PINN: `extended_studies/core/pinn_inversion_bc.py`. Neural-coefficient PINN: `extended_studies/core/compare_neural.py`, `extended_studies/study4_dense_sensors/temporal_density_pair.py`. |
| 3 | Classical inversion baseline | `submission/adjoint_inversion.py` (exact discrete adjoint of Crank-Nicolson, L-BFGS-B, discrepancy principle), `submission/test_adjoint.py`; estimates for all 389 datasets in `submission/models/classical/tikhonov_all389/`. Neural adjoint: `extended_studies/core/adjoint_nn.py`, `extended_studies/study4_dense_sensors/matched_temporal_adjoint.py`. |
| 4 | Where PINN inversion beats classical | Unknown boundary condition or source: `extended_studies/study3_pinn_favoured/README.md`, `results/results_pinnwin_scores.csv`, `results/results_pinnwin2_scores.csv`. Very dense sensors: `extended_studies/campaign/pilots/noise_density_v24_1000_7500/SWEEP_REPORT.md` and `campaign/pilots/dense10000_v21_gpu/RESULTS.md`. Too few sensors: `submission/experiments/sensor_noise_layout_sweep/`. |
| 5 | Sensor density, location and noise experiments | `submission/experiments/sensor_noise_layout_sweep/` (both methods; 5 to 100 sensors, 0 to 10 % noise, 4 layouts), `submission/models/classical/tikhonov_all389/summary_all.csv` (389 configurations), `extended_studies/campaign/density_*.csv` (5 to 3,969 sensors). |
| 6 | Quantitative and qualitative evaluation | Metrics in every `metrics.json` and summary CSV; maps in each `summary.png`; `submission/experiments/zhang_replication_study20/` (20-dataset head-to-head); `extended_studies/analysis/figures/`, `extended_studies/campaign/figures/`. |
| - | Trained models | `submission/models/` (PINN checkpoints, classical estimates), `submission/app/pretrained/` (both methods, 20 datasets), `extended_studies/campaign/results/` and `campaign/v24_case_results/` (neural adjoint and neural PINN checkpoints). |
| - | Interface with user input/upload | `submission/app/streamlit_app.py`: generate a dataset, pick a benchmark dataset or upload sensor readings (CSV); run either method; view maps and metrics; download or reload models. Sample inputs: `submission/app/sample_inputs/`. Screenshots: `submission/app/screenshots/`. |
| - | Displayed performance metric | relL2(D), replay RMS/sigma and the best-constant baseline, shown in the interface's results tab. |
| - | Stage-wise report | `report/`. |
| - | Code-walkthrough and demo videos | `videos/`. |

## 3. Install and run

Python 3.10 or newer; a CPU is enough.

```bash
pip install -r requirements.txt          # numpy, scipy, pandas, matplotlib, torch, streamlit, sympy

cd submission
python generate_dataset_v2.py            # builds datasets_v2_rk4_zero/ (389 datasets, about 4 min, fixed seeds)
python compare_integrators.py            # integrator orders of accuracy
python test_adjoint.py                   # adjoint gradient checks, prints ALL PASSED
python adjoint_inversion.py --keys C_inclusion__Fo0.1__uniform8__n50__random__eta0.05 --reg tv
python pinn_inversion.py --single-run --data-root . --output-dir results_pinn --exp-id B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05 --epochs 1500
python -m streamlit run app/streamlit_app.py    # interface at http://localhost:8501
```

Extended studies (run from `extended_studies/`; `run.py` makes the shared `core/` modules importable):

```bash
cd extended_studies
python run.py core/test_neural.py -v
python run.py core/generate_dataset_v2.py      # datasets_v2_rk4_zero/ for this folder
python run.py core/compare_neural.py --keys B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05 --out results_neural_B
python run.py study3_pinn_favoured/generate_pinn_favoured.py     # hypothesis-suite datasets H1-H5
python run.py study3_pinn_favoured/generate_pinn_favoured2.py    # H6, H7, H9
```

The generated datasets are not stored in the repository (about 550 MB in total); the scripts above regenerate every one.

**Run time in the interface (laptop CPU):** a classical inversion on a noisy dataset takes about 1 to 2 minutes with the default settings. On noise-free data (0 % noise) it is much slower, about 10 minutes, because there is no noise level at which to stop. A PINN run takes a few minutes at 1,500 epochs; use 200 to 300 epochs for a quick check.

## 4. Main results

| Study | Classical adjoint | PINN |
|---|---|---|
| 20 representative datasets (known IC/BC): lower relL2(D) | 20 / 20 | 0 / 20 |
| ... median relL2(D), lower is better | 0.23 | 0.47 (best constant 0.38) |
| 44-dataset sensor x noise x layout sweep: lower relL2(D) | 41 / 44 | 3 / 44 |
| Unknown boundary condition or source (22 datasets; adjoint must assume zero) | median 5.0 | **0.38, lower in 21 / 22** |
| Known physics, same hypothesis suites (59 datasets) | median 0.12, lower in 58 / 59 | median 0.41 |
| Very dense sensors (smooth field, 5 % noise, 7,500 sensors; shared D-network) | 25.1 % | **8.6 %** |

**Where the PINN is better:** (1) the boundary condition or a heat source is unknown; (2) very dense sensor data; (3) too few sensors to identify D, where it stays at the best-constant level instead of inventing structure. With exact initial and boundary conditions and up to a few hundred sensors, the classical adjoint is more accurate, fits the data to the noise level and is faster.

## 5. Metrics

- **relL2(D) = ||D_hat - D|| / ||D||** on the 65 x 65 grid: the main ranking metric; lower is better.
- **Replay RMS / sigma**: D_hat is run through the Crank-Nicolson solver and compared with the sensor readings; about 1 means the data are fitted to the noise level.
- **Best-constant baseline**: relL2 of the best single constant D; a useful method must beat it.

## 6. References

1. M. Raissi, P. Perdikaris and G. E. Karniadakis. Physics-informed neural networks, Journal of Computational Physics, 2019. https://doi.org/10.1016/j.jcp.2018.10.045
2. S. Thakur, E. Esmaili, S. Libring, L. Solorio and A. M. Ardekani. Inverse resolution of spatially varying diffusion coefficient using physics-informed neural networks, arXiv, 2024. https://arxiv.org/abs/2403.03970
3. Z. Zhang, A. Alla and G. E. Karniadakis. Adjoint method versus physics-informed neural networks in PDE-constrained inverse problems, arXiv, 2026. https://arxiv.org/abs/2606.12337
