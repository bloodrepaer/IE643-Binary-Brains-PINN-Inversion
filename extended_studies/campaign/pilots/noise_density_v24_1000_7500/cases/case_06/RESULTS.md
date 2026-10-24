# v24 case06: 5,000 sensors, 10% noise

Complete and independently evaluated 2026-10-22T04:46:35.166554+00:00. Expected archive/log recovered once with CRC/path/symlink/casefold/conflict checks and extraction SHA manifest; compact original extraction campaign/r24c06_v1. Supplied ICBC/continuous-PDE then BOTH checkpoint/bounds/shared-initial/gamma/full40,000-observation/temporal-formula/commonBE+CN checks passed BEFORE coefficient truth access.

| Method | D relative L2 error | CN replay / sigma | BE replay / sigma | CN raw RMS | BE raw RMS | Sensor fit / sigma | Evals | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| temporal_pinn | 28.745926% | 1.275338 | 1.320986 | 0.00303130574 | 0.00313980515 | 0.987008 | 8295 | 6352.483 |
| be_adjoint_nn | 22.673857% | 1.019480 | 0.992225 | 0.00242316588 | 0.00235838504 | 0.992225 | 7880 | 2171.944 |

Both fits completed 1,000 Adam + 4,000 SSB; all 10 rounds capped, no timeouts. BE adjoint has lower coefficient error and passes BOTH predeclared 1.1-sigma replay screens. PINN fails both screens; negative PINN result retained. No post-hoc solver selection. Passing implementation checks is distinct from satisfying replay consistency.

PINN IC relative L2 2.49238522e-07; BC RMS 3.20634982e-18; independent continuous PDE RMS early-to-late: 0.642627298, 0.511157875, 0.249834459, 0.0223073438.

Both arms sequential on same free Kaggle CPU host, two threads. Different cells/reference Colab/Kaggle hosts/devices differ; no speed or identical trajectory claim. One smooth field/noise realization/seed, capped optimization, density/noise-dependent gamma and frozen parent warm start qualify trends. Spatial AD/FD soft/hard state constraints differ. Data originate from independent fine-grid RK4, not CN. All outcomes preserved. User case05 (5,000/5%) remains separate and awaited.
