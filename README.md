# PINN-based inverse estimation of a spatially varying diffusion coefficient

**IE 643 course project  Team Binary_Brains**
Jash Taparia (24b2451)  Mihir Gune (24b2443)

We recover an unknown diffusivity map D(x, y) in the 2D heat equation

    u_t = âˆ‡Â·( D(x, y) âˆ‡u )      on the unit square, known initial state, Dirichlet boundaries

from sparse, noisy sensor readings of u. We compare a **physics-informed neural network (PINN)** with **classical PDE-constrained optimisation (the adjoint method)** on the same synthetic data, produced by a hand-built finite-difference (finite-volume) forward solver.

---

## 1. Repository layout

| Folder | Contents |
|---|---|
| `submission/` | **Main code, ready to run.** Forward solver, dataset generator, classical adjoint inversion, PINN, Streamlit interface, trained models for both methods, the sensor/noise/layout sweep and the 20-dataset evaluation. See `submission/README.md`. |
| `neural_campaign/` | **Deliverable 4 evidence.** Both methods with the same neural network for D; the "where should a PINN win?" hypothesis suites; the dense-sensor sweep (1,000â€“10,000 sensors); trained checkpoints and score tables. |
| `report/` | LaTeX source of the stage-wise report (course template). |
| `submission/app/screenshots/` | Screenshots of every interface tab. |

## 2. Deliverables and where they are

| # | Deliverable | Where |
|---|---|---|
| 1 | Synthetic dataset and forward solver | `submission/integrators.py` (5-point flux form; RK4, Crankâ€“Nicolson with Rannacher start, forward Euler), `submission/domain.py` (six ground-truth D families), `submission/generate_dataset_v2.py` (389 datasets), `submission/compare_integrators.py`, `submission/README_dataset_and_classical.md`. Further dataset generators: `neural_campaign/generate_pinn_favoured.py`, `generate_pinn_favoured2.py`, `make_density_datasets.py`, `extend_sensor_density.py`, `prepare_noise_density_1000_7500.py`, `prepare_dense10000.py`. |
| 2 | PINN inverse model | `submission/pinn_inversion.py`; checkpoints in `submission/models/pinn/`. Boundary-aware PINN: `neural_campaign/pinn_inversion_bc.py`. Neural-coefficient PINN: `neural_campaign/compare_neural.py`, `temporal_density_pair.py`. |
| 3 | Classical inversion baseline | `submission/adjoint_inversion.py` (exact discrete adjoint of CN, L-BFGS-B, discrepancy principle), `submission/test_adjoint.py`; estimates for all 389 datasets in `submission/models/classical/tikhonov_all389/`. Neural adjoint: `neural_campaign/adjoint_nn.py`, `matched_temporal_adjoint.py`. |
| 4 | Where PINN inversion beats classical | Unknown boundary condition or source: `neural_campaign/README_pinn_favoured.md`, `results_pinnwin_scores.csv`, `results_pinnwin2_scores.csv`. Very dense sensors: `neural_campaign/campaign/pilots/noise_density_v24_1000_7500/SWEEP_REPORT.md` and `campaign/pilots/dense10000_v21_gpu/RESULTS.md`. Too few sensors: `submission/experiments/sensor_noise_layout_sweep/`. |
| 5 | Sensor density, location and noise experiments | `submission/experiments/sensor_noise_layout_sweep/` (both methods; 5â€“100 sensors Ã— 0â€“10 % noise Ã— 4 layouts), `submission/models/classical/tikhonov_all389/summary_all.csv` (389 configurations), `neural_campaign/campaign/density_*.csv` (5â€“3,969 sensors). |
| 6 | Quantitative and qualitative evaluation | Metrics in every `metrics.json` and summary CSV; maps in each `summary.png`; `submission/experiments/zhang_replication_study20/` (20-dataset head-to-head); `neural_campaign/analysis/figures/`, `neural_campaign/campaign/figures/`. |
| â€” | Trained models | `submission/models/` (PINN checkpoints, classical estimates), `submission/app/pretrained/` (both methods, 20 datasets), `neural_campaign/campaign/results/` and `campaign/v24_case_results/` (neural adjoint and neural PINN checkpoints). |
| â€” | Interface with user input/upload | `submission/app/streamlit_app.py`: generate a dataset, pick a benchmark dataset or upload sensor readings (CSV); run either method; view maps and metrics; download or reload models. Sample inputs: `submission/app/sample_inputs/`. |
| â€” | Displayed performance metric | relL2(D), replay RMS/Ïƒ and the best-constant baseline, shown in the interface's results tab. |
| â€” | Stage-wise report | `report/`. |

## 3. Install and run

Python 3.10 or newer; a CPU is enough.

```bash
pip install -r requirements.txt          # numpy, scipy, pandas, matplotlib, torch, streamlit, sympy

cd submission
python generate_dataset_v2.py            # builds datasets_v2_rk4_zero/ (389 datasets, ~4 min, fixed seeds)
python compare_integrators.py            # integrator orders of accuracy
python test_adjoint.py                   # adjoint gradient checks, prints ALL PASSED
python adjoint_inversion.py --keys C_inclusion__Fo0.1__uniform8__n50__random__eta0.05 --reg tv
python pinn_inversion.py --single-run --data-root . --output-dir results_pinn \
       --exp-id B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05 --epochs 1500
streamlit run app/streamlit_app.py       # interface at http://localhost:8501
```

Neural-coefficient code (run from `neural_campaign/`, with `datasets_v2_rk4_zero` copied or regenerated there):

```bash
python -m unittest test_neural -v
python compare_neural.py --keys B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05 --out results_neural_B
python generate_pinn_favoured.py         # hypothesis-suite datasets H1-H5
python generate_pinn_favoured2.py        # H6, H7, H9
```

The generated datasets are not stored in the repository (about 550 MB in total); every one is regenerated by the scripts above.

## 4. Main results

| Study | Classical adjoint | PINN |
|---|---|---|
| 20 representative datasets (known IC/BC): lower relL2(D) | 20 / 20 | 0 / 20 |
| â€¦ median relL2(D), lower is better | 0.23 | 0.47 (best constant 0.38) |
| 44-dataset sensor Ã— noise Ã— layout sweep: lower relL2(D) | 41 / 44 | 3 / 44 |
| Unknown boundary condition or source (22 datasets; adjoint must assume zero) | median 5.0 | **0.38, lower in 21 / 22** |
| Known physics, same hypothesis suites (59 datasets) | median 0.12, lower in 58 / 59 | median 0.41 |
| Very dense sensors (smooth field, 5 % noise, 7,500 sensors; shared D-network) | 25.1 % | **8.6 %** |

**Where the PINN is better:** (1) the boundary condition or a heat source is unknown; (2) very dense sensor data; (3) too few sensors to identify D, where it stays at the best-constant level instead of inventing structure. With exact initial and boundary conditions and up to a few hundred sensors, the classical adjoint is more accurate, fits the data to the noise level and is faster.

## 5. Metrics

- **relL2(D) = â€–DÌ‚ âˆ’ Dâ€– / â€–Dâ€–** on the 65Ã—65 grid: the main ranking metric; lower is better.
- **Replay RMS / Ïƒ**: DÌ‚ run through the Crankâ€“Nicolson solver and compared with the sensor readings; about 1 means the data are fitted to the noise level.
- **Best-constant baseline**: relL2 of the best single constant D; a useful method must beat it.

## 6. References

1. M. Raissi, P. Perdikaris and G. E. Karniadakis. Physics-informed neural networks, Journal of Computational Physics, 2019. https://doi.org/10.1016/j.jcp.2018.10.045
2. S. Thakur, E. Esmaili, S. Libring, L. Solorio and A. M. Ardekani. Inverse resolution of spatially varying diffusion coefficient using physics-informed neural networks, arXiv, 2024. https://arxiv.org/abs/2403.03970
3. Z. Zhang, A. Alla and G. E. Karniadakis. Adjoint method versus physics-informed neural networks in PDE-constrained inverse problems, arXiv, 2026. https://arxiv.org/abs/2606.12337
