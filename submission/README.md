# PINN vs classical inversion of a spatially varying diffusivity

IE 643 course project, team **Binary_Brains**.

We recover an unknown diffusivity map D(x, y) in the 2D heat equation

    u_t = ∇·( D(x, y) ∇u )   on the unit square, Dirichlet boundaries, known initial state

from sparse, noisy sensor readings of u. Two methods are built and compared on the same data:

- **Classical PDE-constrained inversion (adjoint method)**, in `adjoint_inversion.py`.
- **Physics-informed neural network (PINN)**, in `pinn_inversion.py`.

The full write-up is the stage-wise project report (a Claude Docs document; download it as PDF or Word from the doc). Two companion guides explain the theory and the defence of each claim with paper quotations.

---

## 1. Folder layout

| Path | What it is |
|---|---|
| `domain.py` | Grid class and the ground-truth D families (A smooth, B bump, C inclusion, D layered, E random field, F defects). |
| `integrators.py` | Hand-built forward solver: 5-point finite-volume operator with harmonic-mean link conductances; RK4, Crank–Nicolson (Rannacher start) and forward Euler time integrators; Dirichlet BCs. |
| `generate_dataset_v2.py` | Builds the 389-dataset benchmark (data on 129², RK4) with all sensor/noise/time/layout sweeps, manifests and figures. |
| `compare_integrators.py`, `integrator_report.json`, `figures/` | Convergence check of the integrators (orders 4.03 / 2.01 / 1.00). |
| `check_problem_spec.py` | Checks that each dataset's problem files are sufficient to run an inversion. |
| `adjoint_inversion.py` | Classical inversion: exact discrete adjoint of Crank–Nicolson on 65², log D on a 16×16 bilinear control grid, Tikhonov or TV, L-BFGS-B, α by the discrepancy principle. |
| `test_adjoint.py` | Gradient checks (adjoint vs finite differences, Taylor test slopes ≈ 2). |
| `run_parallel.py` | Runs the classical inversion over all datasets in parallel chunks. |
| `pinn_inversion.py` | The PINN: u-network with Fourier features and hard-coded IC/BC structure, m = log D network, five-term loss, AdamW + cosine schedule. Single-run, 10-case benchmark and λ-sweep modes. |
| `README_dataset_and_classical.md` | Detailed documentation of the dataset and the classical method. |
| `app/` | Streamlit interface (see `app/README.md`). Includes pretrained PINN and adjoint results for 20 datasets. |
| `models/pinn/benchmark/` | Trained PINN checkpoints (`pinn_best.pt`, `pinn_final.pt`), recovered D, loss histories and metrics for the 10-case benchmark and the λ-sweep. |
| `models/pinn/sweep/` | PINN runs (checkpoints, D̂, metrics) of the 44-dataset sensor-density / noise / layout sweep. |
| `models/classical/tikhonov_all389/` | Classical estimate D̂ (`D_hat.npz`), metrics and a summary figure for **all 389 datasets**; `summary_all.csv` lists them. |
| `models/classical/results_*` | Classical runs with TV regularisation and on the defect family. |
| `models/classical/sweep/` | Classical runs of the same 44 datasets. |
| `experiments/zhang_replication_study20/` | Replication of Zhang et al. (2026) and the 20-dataset PINN vs adjoint study. |
| `experiments/solver_bias_test/` | Test of the hypothesis that generating data with CN and inverting with CN biases the result (refuted). |
| `experiments/thakur_verification/` | Re-implementation and audit of Thakur et al. (2024). |
| `experiments/pinn_projected_variants/` | PINN variants (projected D network, longer training, no interior taper). |
| `docs/` | Links to the report, theory guide and defence guide. |

The dataset folder `datasets_v2_rk4_zero/` (about 175 MB) is not included; it is regenerated exactly by step 2 below (fixed crc32 seeds).

## 2. Install

Python 3.10 or newer. CPU is enough; no GPU is needed.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`: numpy, scipy, pandas, matplotlib, torch, streamlit, sympy.

## 3. Reproduce

Run from this folder, in order.

```bash
python compare_integrators.py                       # 1. verify integrators (~10 s)
python generate_dataset_v2.py                       # 2. build the 389-dataset benchmark (~3-5 min)
python check_problem_spec.py datasets_v2_rk4_zero   # 3. check the problem files
python test_adjoint.py                              # 4. adjoint gradient checks, must all pass
python adjoint_inversion.py --keys C_inclusion__Fo0.1__uniform8__n50__random__eta0.05 --reg tv
                                                    # 5. classical inversion of one dataset (1-3 min)
python adjoint_inversion.py --all                   #    ... or all 389 (4-5 h; run_parallel.py splits it)
python pinn_inversion.py --single-run --data-root . --output-dir results_pinn \
    --exp-id B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05 --epochs 1500
                                                    # 6. PINN on one dataset (1-4 min on CPU)
python pinn_inversion.py --data-root . --output-dir results_pinn          # 10-case benchmark
python pinn_inversion.py --data-root . --output-dir results_pinn --run-lambda-sweep
streamlit run app/streamlit_app.py                  # 7. the interface, http://localhost:8501
```

PINN settings used in all comparisons: Tikhonov λ = 2e-4 for smooth families (A, B, E), TV λ = 3e-4 for sharp ones (C, D, F); classical: Tikhonov for A, B, D, E, F and TV for C.

Dataset keys read `<family>__Fo<Fourier number>__<time protocol>__n<sensors>__<layout>__eta<noise>`, for example `C_inclusion__Fo0.1__uniform8__n50__random__eta0.05` = inclusion field, Fo = 0.1, 8 uniform snapshots, 50 random sensors, 5 % noise.

## 4. Metrics

- **relL2(D) = ‖D̂ − D‖ / ‖D‖** on the 65² grid. Main ranking metric. **Lower is better.**
- **Replay RMS / σ**: the recovered D is run through the forward solver and compared with the sensor readings. About 1 means the data are fitted to the noise level; much less than 1 means over-fitting, much more means under-fitting.
- Best-constant baseline: relL2 of the best single constant D; a method is only useful if it beats it.

The interface displays relL2(D), replay RMS/σ and the maps (true, classical, PINN, error).

## 5. Main results

- On the 20-dataset study the classical adjoint has the lower relL2(D) on 20 of 20 datasets: median 0.23 vs 0.47 for the PINN (best constant: 0.38). The adjoint beats the constant baseline on 20 of 20, the PINN on 8 of 20. Median replay RMS/σ is 0.97 (adjoint) vs 3.7 (PINN). Median wall time on CPU: 46 s vs 125 s.
- Sensor × noise × layout sweep (44 datasets, inclusion and bump families; `experiments/sensor_noise_layout_sweep/`): the adjoint has the lower error on 41 of 44, median relL2 0.36 vs 0.61 (best constant 0.68). On the bump both methods improve with more sensors and less noise. On the inclusion the PINN stays at 0.68–0.76 whatever the sensor count, because its data term is an average.
- The PINN's 3 wins are all on the inclusion with 5 random or boundary-only sensors. There it stays at the best-constant level (about 0.75) while the adjoint overshoots, up to 1.53 with 20 boundary sensors. In other words, the PINN degrades gracefully when the data cannot identify D, but it is not a better reconstruction.
- The suspected solver bias (CN data + CN inversion) does not exist: changing the data solver moves the classical error by at most 0.004.
- Where a PINN should win, from the literature and our tests: uncertain or unknown initial/boundary data, very large data sets where a mesh solve is the bottleneck, and amortised use; none of these is in the course set-up, which gives the classical method the exact initial and boundary conditions.

Full tables and figures are in the report.

## 6. References

1. Raissi, Perdikaris, Karniadakis (2019). Physics-informed neural networks. J. Comput. Phys. 378, 686–707.
2. Zhang, Alla, Karniadakis (2026). Adjoint method versus physics-informed neural networks in PDE-constrained inverse problems. arXiv 2606.12337.
3. Thakur et al. (2024). Inverse resolution of spatially varying diffusion coefficient using physics-informed neural networks. arXiv 2403.03970.
4. Tancik et al. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. NeurIPS.
5. Rahaman et al. (2019). On the spectral bias of neural networks. ICML.
6. Wang, Teng, Perdikaris (2021). Understanding and mitigating gradient flow pathologies in PINNs. SIAM J. Sci. Comput.
