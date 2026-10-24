# v24 case07: 7,500 sensors, 1% noise

Complete and independently evaluated 2026-10-22T02:16:40.462501+00:00. Expected archive/log recovered once with CRC/path/symlink/casefold/conflict checks and extraction SHA manifest; compact original extraction campaign/r24c07_v1. Independent supplied ICBC/continuous-PDE then both checkpoint/bounds/shared-initial/gamma/full60,000-observation/temporal-formula/commonBE+CN checks passed BEFORE coefficient truth access.

| Method | D relative L2 error | CN replay / sigma | BE replay / sigma | CN raw RMS | BE raw RMS | Sensor fit / sigma | Evals | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| temporal_pinn | 5.610972% | 2.968794 | 1.872891 | 0.000705642129 | 0.000445160795 | 0.995125 | 8085 | 6332.960 |
| be_adjoint_nn | 45.643568% | 2.595051 | 1.067648 | 0.00061680843 | 0.000253765464 | 1.067648 | 7820 | 1412.084 |

Both fits completed 1,000 Adam + 4,000 SSB; all 10 rounds capped, no timeouts. PINN has lower coefficient error but fails BOTH predeclared 1.1-sigma replay screens. BE adjoint passes BE and fails CN. No consistency-qualified PINN win; no post-hoc solver selection. Passing implementation checks is distinct from replay consistency.

PINN IC relative L2 2.49238522e-07; BC RMS 3.1207326e-18; independent continuous PDE RMS by early-to-late bins: 0.893402476, 0.880995437, 0.256521013, 0.00438678476.

Both arms sequential on same free Kaggle CPU host, two threads. Cross-cell/reference Colab/Kaggle hosts/devices differ; no speed or identical trajectory claim. One smooth field/noise realization/seed and capped optimization. Density/noise-dependent gamma, frozen parent warm start and AD/FD soft/hard state differences qualify trends; fine-grid RK4 data, not CN. All outputs including negative consistency results retained. User 5,000-sensor/5%-noise Colab pair remains separate and awaited.

Active case03 (3,000/10%) and newly dispatched06 (5,000/10%), max2. Next only09 (7,500/10%) as a slot opens.
