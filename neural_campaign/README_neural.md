# Shared neural representation of diffusivity

## Local comparison and checkpoint interface

Run `python -m streamlit run neural_interface.py --server.address 127.0.0.1 --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false`.
Open the local address printed by Streamlit. The primary view compares saved
neural-adjoint and neural-PINN fields, coefficient errors, common CN replay,
sensor fits and convergence flags. Research pilots remain separately labelled;
queued or missing arms are not shown as completed comparisons.

The checkpoint tab accepts a saved model or uploaded `.pt` plus entered/uploaded
CSV with `x,y` and optional physical time `t`. It exports `D_pred` and, for PINN
state checkpoints, `u_pred`. Optional `D_reference` supplies a post-fit relative
coefficient metric; optional `u` supplies a temperature-network RMSE. That RMSE
is not coefficient replay. Inference is read-only and never retrains or submits
jobs. Supports frozen bounded models, coupled hard-IC/BC models, and the separately
labelled v11 exponential coefficient. Do not use model inference to select
training settings using the synthetic answer key.

Dependencies are the numerical packages below plus `streamlit` (verified locally
with version1.37.1). Interface implementation and numerical checks are recorded
in `campaign/interface_validation.json`. Stage-wise partial documentation is in
`campaign/STAGE_REPORT.md`; current scientific protocols and limitations are in
`campaign/EXPERIMENT_PROTOCOL.md` and the per-pilot protocol files. The description
below of the plain-state PINN refers to the original broad baseline; later
research variants are not silently substituted into that baseline.

`compare_neural.py` runs three per-dataset inverse formulations:

| Method | Coefficient | Temperature | Optimizer |
|---|---|---|---|
| `grid` | Existing interpolated control grid for log D | CN + Rannacher | L-BFGS-B |
| `adjoint_nn` | `NeuralDiffusivity(x,y)` | Same CN + Rannacher | L-BFGS-B, discrete adjoint + network VJP |
| `pinn` | Identical `NeuralDiffusivity(x,y)` | Independent temperature MLP | Adam + optional fixed-batch L-BFGS |

This is a new controlled baseline. Existing PINN implementations, results,
and Kaggle bundles remain available. It does not rerun or replace them.

## Mathematics and shared controls

Both neural methods use the same float64 tanh coefficient MLP (default three
hidden layers of width 32), random seed, constant initial field fitted from
sensor data, and bounds. With unconstrained output z,

`m_phi = log(D_min) + log(D_max/D_min) sigmoid(z_phi(2[x,y]-1))`

`D_phi = exp(m_phi)`, with `D_min=0.001`, `D_max=10`.

The final layer starts at zero weights with a bias chosen to reproduce the
fitted constant exactly. Hidden-layer features are identical across methods;
they begin receiving gradients after the output weights move. The coefficient
has no wall taper and takes no time input. It can be evaluated off-grid.

The adjoint evaluates the network on the solver grid and computes
`dJ/dphi = (dm/dphi)^T (dJ/dm)`. The existing backward CN adjoint computes
`dJ/dm`; PyTorch computes only the network vector-Jacobian product. There is
no autograd graph through sparse solves and no dense network Jacobian.

Both methods use the same summed Gaussian sensor misfit
`0.5 sum((u-data)/sigma_w)^2` and the exact same discrete spatial regularizer
on log D, evaluated over the entire inversion grid. PINN uses

`J = J_data + w_pde*(N_data/2)*mean(((u_tau - T div(D grad u))/u_scale)^2)`
`    + w_cond*(N_data/2)*(MSE_IC + MSE_BC)/u_scale^2 + alpha*R(m)`.

Time is normalized as `tau=t/T`; coefficient values remain in physical units.
The divergence differentiates both D and the temperature gradient, retaining
the `grad D dot grad u` term. The state network is a plain 4x64 tanh MLP with
soft IC/BC penalties. It has neither the historical heat-kernel prior nor
its fixed Gaussian-IC assumptions. Narrow initial peaks can require much more
training or a better state representation.

Adjoints select alpha along the existing discrepancy path. PINN currently
uses a **fixed CLI alpha**, not an implemented discrepancy search. Matching
architecture alone does not make optimizers, convergence or alpha selection
identical. Record these differences when reporting results. All penalties
and budgets are recorded in the run configuration; tune them without using
ground-truth coefficient error for model selection.

## Run locally

Dependencies: `numpy scipy sympy matplotlib torch` (Python 3.10+).
The existing Anaconda Python has these packages installed. The runner sets
`MKL_THREADING_LAYER=SEQUENTIAL` before numerical imports to avoid conflicting
Intel OpenMP runtimes in mixed conda/PyTorch installations. Torch thread count
is configurable. CPU is the default; `--device cuda` trains the PINN on a
CUDA GPU, while the sparse adjoint and diagnostic replay remain on CPU.
Use `--device auto` to select CUDA when available. Explicit CUDA requests
fail if a GPU is unavailable. Precision remains float64 on either device.

```powershell
python -m unittest test_neural -v
python test_adjoint.py

# Default: all three methods, A_smooth, existing inversion time resolution.
python compare_neural.py

# Neural pair on another benchmark, with the identical coefficient architecture.
python compare_neural.py --methods adjoint_nn pinn --width 32 --depth 3 --seed 42 --keys B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05 --out results_neural_B

# Explicit common alpha for an optimization study (no discrepancy search).
python compare_neural.py --alphas 1 --pinn-alpha 1 --adam-steps 5000 --lbfgs-steps 500 --out results_neural_fixedalpha

# Execution smoke test only; deliberately insufficient convergence budgets.
python compare_neural.py --out results_neural_smoke --width 12 --depth 2 --maxiter 15 --alphas 400 40 4 --adam-steps 100 --lbfgs-steps 20 --n-collocation 128 --n-boundary 64
```

Use a distinct output directory per study/seed. `summary.csv` describes the
current invocation; reusing an output directory does not merge older rows.
Only known Dirichlet BCs, zero source and the existing unit-square format are
supported. Unknown-BC/source cases are rejected so assumptions cannot silently
differ. Nonzero and time-varying known Dirichlet boundaries are supported.
All fits read `problem/` and `sensors/`; `ground_truth()` is called only after
fitting and parameter selection.

## Saved outputs and inference

Each `<out>/<key>/<method>__s<seed>/` contains `D_hat.npz`, `metrics.json`
and `summary.png`. Neural methods also save `model.pt`. PINN checkpoints
include temperature weights/config and the physical time horizon.

```python
import torch
from neural_diffusivity import NeuralDiffusivity
from compare_neural import TemperatureNetwork, temperature_value

ckpt = torch.load("path/to/model.pt", map_location="cpu", weights_only=True)
D = NeuralDiffusivity.from_checkpoint(ckpt["coefficient"])
xy = torch.tensor([[0.23, 0.51], [0.8, 0.2]], dtype=torch.float64)
with torch.no_grad():
    print(D.diffusivity(xy))

if "temperature_state_dict" in ckpt:
    u = TemperatureNetwork(**ckpt["temperature_config"])
    u.load_state_dict(ckpt["temperature_state_dict"])
    xyt = torch.tensor([[0.23, 0.51, 0.1 / ckpt["T"]]], dtype=torch.float64)
    with torch.no_grad():
        print(temperature_value(u, D, xyt))
```

Metrics include full-grid relative L2(D), log-D RMSE, replay RMS/sigma,
the squared replay discrepancy, absolute replay RMS, relative-to-signal replay
RMS, wall time and coefficient parameter count. PINN also records temperature
network sensor RMS/sigma and loss components. These are training-data replay
diagnostics, not held-out generalization scores. With zero noise, the solver
uses its existing numerical weighting floor for optimization; replay RMS/sigma
and its square are null in the new metrics and absolute/relative RMS should
be used. Some inherited diagnostic columns/plots still use that numerical
floor. Report timings together with shared initialization time and solver
resolution. The PINN training timer excludes its final diagnostic CN replay;
adjoint optimization time includes its PDE solves.

## Verification

`test_neural.py` checks exact paired initialization, positive bounded outputs,
safe checkpoint reload, equality of neural and grid forward objectives for
the same field, central finite differences and second-order Taylor remainders
through the full neural adjoint (both regularizers, time-dependent BC and
off-grid readings), equality of PINN/grid regularization values and gradients,
and an analytic variable-D PDE residual. `test_adjoint.py` retains the existing
forward, transpose and Taylor regression checks.

Short smoke runs establish executable code, gradient correctness and artifact
serialization. They do not establish that either new model has converged or
that a method wins. Full multi-seed sensor/noise studies and PINN penalty
selection are still required before a scientific conclusion.
An adjoint flag of `floor_above_sigma` only says the attempted path did not
reach the discrepancy target; iteration limits can also cause it. Check
`alpha_path` optimizer statuses before interpreting it as a model-error floor.

## Initial local runs

A separate known-IC state-model pilot is available through
`--state-model gaussian_ic`. It parses Gaussian peaks from the supplied initial
condition, evolves a constant-D heat-kernel base using the sensor-fitted
constant, and gates the learned correction to zero at t=0. This enforces the
known initial function rather than asking a plain MLP to learn narrow peaks.
The coefficient network is unchanged. The base is infinite-domain: plate
boundaries still require training and independent replay validation. The
implementation rejects unsupported ICs or a mismatch with the supplied IC grid.
This is a distinct pilot, not a silent replacement for frozen campaign models.
See `campaign/pilots/ic_v1_gpu/pilot.json` and `campaign/pinn_review_gate.json`.

All runs below use A_smooth, Fo=0.1, 50 random sensors, eight times and 5% noise,
seed 42. They are execution/optimization diagnostics, not a converged benchmark.

| Smoke method | Relative L2(D) | CN replay RMS/sigma |
|---|---:|---:|
| Grid adjoint | 0.0968 | 0.8758 |
| Adjoint-NN | 0.3092 | 1.3266 |
| PINN | 0.3843 | 2.9425 |

These are saved under `results_neural_smoke/`, with the short command above.
Both neural checkpoints were reloaded from disk and their evaluations were
checked against the saved D grids to numerical precision. The default network
has 2,241 coefficient parameters; this smoke run used 205.

Additional default-network pilots intentionally explored only limited budgets:

- `results_neural_pilot/`: adjoint-NN, alpha path [40, 4], 100 iterations each.
  Relative L2(D) was 1.1986 despite replay RMS/sigma 1.2923; high/low sensitivity
  errors were 0.1847/1.4396. Both optimizations reached their iteration caps.
  This is evidence of poor coefficient recovery in weakly observed regions,
  not evidence that a better sensor fit identifies the coefficient.
- `results_neural_pinn_pilot/`: PINN, 1,000 Adam steps and 100 L-BFGS steps,
  256 collocation and 128 boundary samples, fixed alpha 1. Relative L2(D)
  was 0.5833. Its replay remains poor; longer joint training alone did not
  resolve the coefficient identification problem in this pilot.

Do not compare these pilots as equal-budget or equally regularized methods.
Next, establish stable recovery on the smooth case using the full regularization
path, stronger PINN physics/IC enforcement or a better temperature approximation,
and data-based stopping before spending compute on the full experiment suite.

## Versioned campaign and experimental state models

See CAMPAIGN.md and campaign/FINDINGS.md for current evidence. The legacy
plain-state PINN default is retained for reproducibility; broad new PINN
training is held pending independent IC/BC/PDE and common CN replay checks.
Root optional Gaussian state models are separate pilots, not replacements
for frozen campaign/code/48b49b1f17a0. The latest coefficient-coupled
`gaussian_ic_bc_locald` pilot uses a local-constant Gaussian heat ansatz, not
an exact variable-D solver; all coefficient paths are differentiated in the
PDE residual. It requires supplied known zero boundaries and compatible
Gaussian IC, and remains experimental until replay validation.

Reload temperature checkpoints with the shared helper, which supports
legacy independent states and the new coupled state:

```python
import torch
from compare_neural import TemperatureNetwork, temperature_value
from neural_diffusivity import NeuralDiffusivity

checkpoint = torch.load(path, map_location="cpu", weights_only=True)
coefficient = NeuralDiffusivity.from_checkpoint(checkpoint["coefficient"])
temperature = TemperatureNetwork(**checkpoint["temperature_config"])
temperature.load_state_dict(checkpoint["temperature_state_dict"])
# x has columns x, y, t/T, float64.
u = temperature_value(temperature, coefficient, x)
```

Import TemperatureNetwork and temperature_value from compare_neural, and
NeuralDiffusivity from neural_diffusivity. Keep pilot versions distinct and
do not use true D for training or hyperparameter/model selection.
