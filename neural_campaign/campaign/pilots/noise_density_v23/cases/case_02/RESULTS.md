# v23 case 02: 1,000 sensors, 10% noise

Both methods completed 1,000 Adam + 4,000 SSB iterations; all 10 outer rounds hit iteration caps. No budget timeouts. Independent supplied IC/BC/continuous-PDE checks and both checkpoint/protocol/common-BE/CN checks passed before coefficient truth was accessed. Passing implementation checks does not imply replay consistency.

| Method | D relative L2 error | CN replay / sigma | BE replay / sigma | CN raw RMS | BE raw RMS | Sensor fit / sigma | Evals | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| temporal_pinn | 69.098231% | 5.859843 | 5.975879 | 0.0139280534 | 0.0142038567 | 0.885696 | 7741 | 4822.961 |
| be_adjoint_nn | 21.576945% | 1.001283 | 0.974066 | 0.00237991525 | 0.00231522288 | 0.974066 | 7736 | 1848.368 |

Predeclared consistency screen: each common solver replay <= 1.1 sigma; report both, without choosing a solver after seeing outcomes.

BE adjoint has lower coefficient error and passes BOTH replay screens; PINN fails BOTH. Negative PINN result retained.

PINN initial-condition relative L2: 2.49238522e-07; boundary RMS: 2.61309438e-18. Continuous-PDE RMS in increasing-time bins: 0.48153343, 0.481872299, 0.488465552, 0.0811736094.

Both arms ran sequentially on the same free Kaggle CPU host, two threads. Different cells/reference results use different hosts/devices/budgets; no speed superiority claim. One smooth field, one fixed observation/noise realization and optimization seed, capped optimization: not converged or population-level superiority. Frozen parent warm start, density/noise-dependent gamma, random PINN pairs vs fixed BE grid and soft vs hard state constraints qualify comparisons. Data generated with fine-grid RK4, not CN. All source/data versions, archives, logs, checkpoints, validation and negative outcomes retained.
