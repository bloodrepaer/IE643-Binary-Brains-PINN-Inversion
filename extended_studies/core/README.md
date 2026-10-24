# core: shared library

Modules imported by two or more studies. Run anything here from the `extended_studies/` folder through the launcher, for example `python run.py core/test_neural.py -v`.

## Files

| File | What it does | Used by |
|---|---|---|
| `integrators.py` | Finite-volume forward model of du/dt = div(D grad u): 5-point flux operator, RK4, Crank-Nicolson (Rannacher start), forward Euler | dataset generators, `adjoint_inversion.py`, tests |
| `domain.py` | Grid and the ground-truth D families | dataset generators, tests |
| `generate_dataset_v2.py` | Main benchmark generator: 389 datasets in `datasets_v2_rk4_zero/` (RK4 on 129 x 129, inversion grid 65 x 65, fixed seeds) | studies 1 and 3 generators, `pinn_papers.py` |
| `adjoint_inversion.py` | Classical adjoint: exact discrete adjoint of CN, 16 x 16 control grid, Tikhonov or TV, L-BFGS-B, discrepancy-principle alpha; also `Dataset` (data loader) and `evaluate` (scoring) | every study |
| `test_adjoint.py` | Forward consistency, observation-operator transpose and Taylor tests of the adjoint gradient | `test_neural.py`; run before trusting an inversion |
| `neural_diffusivity.py` | The shared D-network: 3 x 32 tanh, output log D squashed into [0.001, 10], seeded so both methods start from identical weights | studies 1, 2, 4 |
| `adjoint_nn.py` | Neural adjoint: the CN adjoint of `adjoint_inversion.py` with D given by the network (vector-Jacobian product, no dense Jacobian) | studies 1, 2, 4 |
| `compare_neural.py` | Runs the grid adjoint, neural adjoint and neural PINN on the same dataset and scores them identically; the program every job of studies 1 and 2 ran | `campaign.py`, study 4 |
| `test_neural.py` | Numerical checks of the network bounds, the discrete chain rule and the PINN flux | |
| `pinn_inversion.py` | The project PINN (u-network and D-network, Fourier features, physics-informed loss) | study 3 (`run_pinn_pinnwin.py`) |
| `pinn_inversion_bc.py` | `pinn_inversion.py` with three boundary-condition modes read from the problem file (`dirichlet`, `dirichlet_soft`, `free`) | study 3 (H2, H5, H7), `pinn_papers.py` |
| `pinn_papers.py` | The inverse PINNs of Thakur et al. (2024) and Zhang et al. (2026), rebuilt as published; also provides the quasi-Newton (SSBroyden) routines | `pinn_zhang_transfer.py` |
| `pinn_zhang_transfer.py` | Zhang et al.'s optimiser transferred to our transient diffusion data | study 4 (`calibrated_neural_pair.py`) |

`pinn_papers.py` and `pinn_zhang_transfer.py` are paper replications and also dependencies of study 4: every study-4 pair script imports `calibrated_neural_pair.py`, which imports `pinn_zhang_transfer.py`, which imports `pinn_papers.py`. That is why they are kept here.

## Copies of files in `../../submission/`

Several files here are copies of the main code, kept so this folder runs and packages on its own and so the recorded provenance hashes keep matching.

| File | Relation to the `submission/` file |
|---|---|
| `integrators.py`, `domain.py`, `generate_dataset_v2.py`, `test_adjoint.py` | Byte-identical |
| `adjoint_inversion.py` | One refactor: the gradient computation is split into `field_objective(m, alpha)`, which returns the gradient with respect to the full-grid log D. `objective(theta)` calls it and applies the control-grid map, so the grid adjoint is unchanged; the neural adjoint applies its own chain rule instead. The plot title also uses `metrics["method"]` when present. |
| `pinn_inversion.py` | Same executable code; differs in comments and docstrings, the spelling of numeric constants (`1e-3` vs `0.001`) and two unused variables in the sweep plot |

`../archive/` holds two more byte-identical copies (`compare_integrators.py`, `check_problem_spec.py`) that no study uses.

## Tests

```bash
python run.py core/test_neural.py -v          # synthetic data, no download, a few seconds
python run.py core/generate_dataset_v2.py     # about 5 min; needed by test_adjoint.py
python run.py core/test_adjoint.py            # prints ALL PASSED; saves figures/taylor_test.png
```

## Running a comparison

```bash
python run.py core/compare_neural.py --keys A_smooth__Fo0.1__uniform8__n50__random__eta0.05 \
    --methods grid adjoint_nn pinn --out results_neural_A
```

The defaults are short. Studies 1 and 2 used `--maxiter 150 --adam-steps 15000 --lbfgs-steps 750 --n-collocation 2048 --n-boundary 256 --data-batch 4096 --physics-weight 1000 --condition-weight 1000 --pinn-alpha 1 --seed 42 --width 32 --depth 3` (Tikhonov regulariser); the same values are stored with every job in `campaign/results/*/*/run_config.json`. Output: `<out>/<key>/<method>__s<seed>/` with `metrics.json`, `D_hat.npz` and the model checkpoint.
