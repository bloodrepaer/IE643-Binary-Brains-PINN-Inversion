# Study 2: neural adjoint vs neural PINN

**Question.** If both methods represent D with the *same* neural network, does it still matter whether the PDE is a hard constraint (adjoint) or a soft penalty (PINN)?

**Design.** Both arms use the D-network of `core/neural_diffusivity.py` (3 x 32 tanh, log D squashed into [0.001, 10]) with identical initial weights (seed 42, set at the best constant D). Only the way the physics enters differs:

- **neural adjoint** (`core/adjoint_nn.py`): u is the exact CN solution for the current D; Tikhonov regulariser; alpha from the discrepancy principle; L-BFGS-B, 150 iterations per alpha.
- **neural PINN** (`run_pinn` in `core/compare_neural.py`): a u-network trained jointly with the D-network on data misfit + 1000 x PDE residual + 1000 x (IC + BC) + alpha R(log D), alpha = 1 fixed; Adam (15,000 steps) then L-BFGS (750 steps), 2,048 collocation and 256 boundary points per batch.

Both are scored the same way: relL2(D) on the 65 x 65 grid, and **replay**: D_hat run through the common CN solver and compared with the readings, as RMS / sigma.

This study has no scripts of its own: every case was a job of `core/compare_neural.py` (methods `adjoint_nn` and `pinn`), run through `campaign.py`.

## Results

8 datasets, all from the main benchmark with known physics and 5 % noise. Source: `campaign/primary_neural_comparison.csv`; per-run files in `campaign/results/datasets_v2_rk4_zero/<job>/<key>/<method>__s42/`.

| Dataset | Neural adjoint relL2(D) | replay / sigma | Neural PINN relL2(D) | replay / sigma | PINN sensor fit / sigma |
|---|---:|---:|---:|---:|---:|
| A_smooth, Fo 0.1, uniform8, 50 sensors | 0.105 | 0.94 | 0.506 | 7.30 | 0.93 |
| B_gaussian_bump, Fo 0.1, uniform8, 50 | 0.115 | 1.00 | 0.709 | 5.69 | 1.04 |
| E_grf, Fo 0.1, uniform8, 50 | 0.199 | 0.97 | 0.469 | 6.15 | 0.93 |
| C_inclusion, Fo 0.1, uniform8, 50 | 0.590 | 0.98 | 0.807 | 3.01 | 0.94 |
| D_layered, Fo 0.1, uniform8, 50 | 0.192 | 0.96 | 0.475 | 3.82 | 0.96 |
| F_defects, Fo 0.1, uniform8, 50 | 0.277 | 0.99 | 0.421 | 5.55 | 1.01 |
| A_smooth, Fo 0.02, early8, 50 | 0.328 | 0.95 | 0.862 | 5.92 | 1.00 |
| A_smooth, Fo 0.02, final1, 50 | 0.276 | 0.87 | 0.985 | 16.41 | 0.16 |
| **Median** | **0.237** | | **0.608** | | |

The hard constraint is more accurate on all 8 datasets, and every neural-adjoint estimate reaches the noise level. The PINN's u-network fits the sensors at about the noise level (0.93 to 1.04 sigma on seven datasets), yet its D, replayed through the solver, misses the same readings by 3 to 16 sigma. A network can fit the data with the wrong coefficient; the replay test exposes this, the PINN's own data loss does not. Since the network for D is the same, the difference comes from the soft treatment of the physics.

## Reproduce one case

From `extended_studies/`, after `python run.py core/generate_dataset_v2.py`:

```bash
python run.py core/compare_neural.py --keys A_smooth__Fo0.1__uniform8__n50__random__eta0.05 \
    --methods adjoint_nn pinn --maxiter 150 --adam-steps 15000 --lbfgs-steps 750 \
    --n-collocation 2048 --n-boundary 256 --physics-weight 1000 --condition-weight 1000 \
    --pinn-alpha 1 --seed 42 --threads 1 --out results_study2
```

Add `--device cuda` to train the PINN on a GPU, as in the recorded runs (Tesla T4, about 5.5 minutes each); the neural adjoint ran on one CPU thread (2.5 to 6 minutes each).

**Limitations.** A single seed (42) and 8 datasets; no variation of the PINN's loss weights or alpha within this study (the loss-weight sweep of the main PINN is in `../../submission/`).
