# v23 case 01: 1,000 sensors, 1% noise

Both methods completed 1,000 Adam + 4,000 SSB iterations; all 10 outer rounds hit iteration caps. No budget timeouts. Independent supplied IC/BC/continuous-PDE checks and both checkpoint/protocol/common-BE/CN checks passed before coefficient truth was accessed. Passing implementation checks does not imply replay consistency.

| Method | D relative L2 error | CN replay / sigma | BE replay / sigma | CN raw RMS | BE raw RMS | Sensor fit / sigma | Evals | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| temporal_pinn | 9.398804% | 2.968372 | 1.707335 | 0.000705541775 | 0.000405810513 | 0.954899 | 7942 | 4831.297 |
| be_adjoint_nn | 32.035837% | 2.606514 | 0.996359 | 0.000619533232 | 0.000236821039 | 0.996359 | 8146 | 1922.063 |

Predeclared consistency screen: each common solver replay <= 1.1 sigma; report both, without choosing a solver after seeing outcomes.

PINN has lower coefficient error but fails BOTH replay screens; BE adjoint passes BE and fails CN. No verified consistency-qualified PINN win.

PINN initial-condition relative L2: 2.49238522e-07; boundary RMS: 3.15395139e-18. Continuous-PDE RMS in increasing-time bins: 0.751745178, 0.641487961, 0.207620654, 0.00573576836.

Both arms ran sequentially on the same free Kaggle CPU host, two threads. Different cells/reference results use different hosts/devices/budgets; no speed superiority claim. One smooth field, one fixed observation/noise realization and optimization seed, capped optimization: not converged or population-level superiority. Frozen parent warm start, density/noise-dependent gamma, random PINN pairs vs fixed BE grid and soft vs hard state constraints qualify comparisons. Data generated with fine-grid RK4, not CN. All source/data versions, archives, logs, checkpoints, validation and negative outcomes retained.
