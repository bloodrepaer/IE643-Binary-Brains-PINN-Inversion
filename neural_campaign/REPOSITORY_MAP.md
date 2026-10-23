# Repository review and neural coefficient extension

This review describes the local snapshot, including stored experiment outputs.
The original README and prep presentation describe earlier stages; they are
not a reliable inventory of everything now implemented.

## What has already been done

The problem is transient diffusion on the unit square,
`u_t = div(D(x,y) grad u)`, with known initial conditions and edge-wise
Dirichlet boundaries in the main benchmark. The unknown is physical
diffusivity, independent of time.

| Component | Implementation and current state |
|---|---|
| Spatial domain / ground truth | `domain.py`: smooth sinusoid, Gaussian bump, inclusion, layers/checkerboards, random fields, local defects |
| Forward PDE | `integrators.py`: harmonic-mean flux links, eliminated Dirichlet boundary nodes; RK4, CN with Rannacher start, Euler, exact-in-time reference |
| Main synthetic benchmark | `generate_dataset_v2.py`: 129x129 forward data, 65x65 inversion grid, paired/nested sensors, relative noise, temporal/spatial/horizon sweeps; **389 entries** in the existing main manifest |
| Classical inversion | `adjoint_inversion.py`: log D on a bilinearly interpolated control grid, normally 16x16; exact discrete CN adjoint, bounded L-BFGS-B, Tikhonov/TV, constant-D initialization, discrepancy path |
| Classical results | `results_tikhonov`: 389 coefficient files; `results_adjoint`: ten selected cases; additional TV and defect studies |
| Original PINN | `pinn_inversion.py`: temperature and log-D networks, Fourier features, heat-kernel temperature base, loss-weight sweeps, Adam, replay evaluation |
| Boundary-aware PINN | `pinn_inversion_bc.py`: known nonzero boundaries and unknown/free boundary modes |
| PINN improvements | `pinn_inversion_v2.py`, `run_pinn_v2_batch.py`: remove coefficient wall taper, fit constant initialization, disable data damping, add L-BFGS, increase feature frequencies; V0-V4 ablations |
| Simpler PINN | `pinn_literature.py`: float64 tanh networks, dimensionless time, Adam/L-BFGS, optional analytic IC base |
| Paper-inspired implementations | `pinn_papers.py`: Thakur-style backward-Euler consistency loss and a Zhang-labelled continuous-residual implementation with SSBroyden; includes explicitly chosen settings where source details are absent |
| Hypothesis suites | `generate_pinn_favoured.py`: 57 H1-H5 cases (sparsity, unknown BC/source, heavy noise, long horizon), including controls; `generate_pinn_favoured2.py`: 24 H6/H7/H9 cases (more data, unknown BC, features finer than control grid) |
| Density studies | `make_density_datasets.py`: 4 main density cases and 2 wide-IC cases in the existing manifests |
| Further imaging-like studies | `generate_pinn_favoured3.py`: H10 wide-IC / dense-data / known-vs-unknown BC generator; this suite is not present as a top-level dataset folder in this snapshot |
| Verification | `test_adjoint.py`, `compare_integrators.py`, `check_problem_spec.py`, `check_physics.py`, `validate_pinn_favoured.py` |
| Evaluation / figures | `score_pinnwin.py`, `score_papers.py`, `analysis/fig_*.py`, `analysis/rescore.py`, and stored score CSVs |
| Execution / packaging | `run_parallel.py`, `run_queue.py`, PINN batch scripts, `kaggle_pinnwin/`, `kaggle_thakur/` |

`datasets_v2_pinnwin/code/` and `kaggle_thakur/dataset/` contain packaged
copies of source files. Changes to root modules do not automatically update
those copies. `__pycache__`, its large zip, and result folders are artifacts,
not independent implementations.

The two complete hypothesis score tables contain 57+57 and 24+24 paired
adjoint/PINN rows respectively (plus two higher-resolution adjoint controls
in the second table). `results_papers_scores.csv` contains only 19 rows across
several methods; it is not a complete comparison of all datasets.

## What the requested change actually means

The existing PINNs **already represent D by a neural network**. The existing
classical solver represents **log D on a coarse grid**, interpolates it, and
exponentiates it. The missing experiment was adjoint optimization of a neural
coefficient with the same representation as the PINN.

The new files provide that three-way experiment without replacing historical
PINNs or overwriting their results. See `README_neural.md` for commands.

## Important interpretation and remaining work

- A small sensor error does not imply an accurate coefficient. Some regions
  are weakly excited by the initial condition; compare coefficient error and
  CN replay error together.
- A PINN's temperature network can fit sensors while its recovered D fails
  a forward replay. Keep those metrics separate.
- Changing D from a grid to a network does not model an unknown boundary or
  source. The existing unknown-physics suites compare differing assumptions;
  a stronger classical comparator would infer those nuisance parameters too.
- A residual normalized by the number of readings is a noise-discrepancy
  diagnostic, not a statistically degrees-of-freedom-adjusted chi-squared.
  Values slightly below one do not by themselves establish overfitting.
- Do not infer a general winner from single seeds or unequal training budgets.
  The pasted conversation's numerical literature claims have not been
  independently verified in this implementation task.
- Some analysis scripts hard-code `/home/claude/v2` or `/home/claude/work`;
  they need path configuration before use on this Windows machine.
- No top-level Streamlit app or complete stage-wise submission report was
  found. Those project deliverables remain separate work.
- There is no Git metadata in this workspace snapshot. Existing results and
  historical source variants have been retained.

Suggested next study: run the neural comparison on A/B/E (smooth fields), C
(inclusion) and F (defects), then paired sensor/noise sweeps and multiple
seeds. Tune PINN physics/condition penalties using data and replay diagnostics;
use the answer key only for final scoring. Add nuisance-parameter inference
before drawing conclusions about methods under unknown physics.
