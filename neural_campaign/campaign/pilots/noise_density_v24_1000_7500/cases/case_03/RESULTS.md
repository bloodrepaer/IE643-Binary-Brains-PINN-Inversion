# v24 case03: 3,000 sensors, 10% noise

Complete and independently evaluated 2026-10-22T04:46:35.166554+00:00. Expected archive/log recovered once with CRC/path/symlink/casefold/conflict checks and extraction SHA manifest; compact original extraction campaign/r24c03_v1. Supplied ICBC/continuous-PDE then BOTH checkpoint/bounds/shared-initial/gamma/full24,000-observation/temporal-formula/commonBE+CN checks passed BEFORE coefficient truth access.

| Method | D relative L2 error | CN replay / sigma | BE replay / sigma | CN raw RMS | BE raw RMS | Sensor fit / sigma | Evals | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| temporal_pinn | 42.300673% | 1.857898 | 1.931244 | 0.00441597336 | 0.00459030511 | 0.978571 | 7971 | 6382.964 |
| be_adjoint_nn | 22.269318% | 1.019497 | 0.993371 | 0.00242320758 | 0.00236110942 | 0.993371 | 8139 | 2562.785 |

Both fits completed 1,000 Adam + 4,000 SSB; all 10 rounds capped, no timeouts. BE adjoint has lower coefficient error and passes BOTH predeclared 1.1-sigma replay screens. PINN fails both screens; negative PINN result retained. No post-hoc solver selection. Passing implementation checks is distinct from satisfying replay consistency.

PINN IC relative L2 2.49238522e-07; BC RMS 3.30428076e-18; independent continuous PDE RMS early-to-late: 0.986777684, 0.68925645, 0.299413848, 0.0318653463.

Both arms sequential on same free Kaggle CPU host, two threads. Different cells/reference Colab/Kaggle hosts/devices differ; no speed or identical trajectory claim. One smooth field/noise realization/seed, capped optimization, density/noise-dependent gamma and frozen parent warm start qualify trends. Spatial AD/FD soft/hard state constraints differ. Data originate from independent fine-grid RK4, not CN. All outcomes preserved. User case05 (5,000/5%) remains separate and awaited.
