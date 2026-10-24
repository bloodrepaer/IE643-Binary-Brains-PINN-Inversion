# v24 case04: 5,000 sensors, 1% noise

Complete and independently evaluated 2026-10-22T01:46:37.566435+00:00. Expected archive/log recovered once, safe CRC/path/symlink/casefold/conflict checks and SHA manifest retained, compact extraction campaign/r24c04_v1. Independent supplied ICBC/continuous-PDE then both checkpoint/bounds/shared-initial/gamma/full40,000-observation/temporal-formula/commonBE+CN checks passed BEFORE coefficient truth access.

| Method | D relative L2 error | CN replay / sigma | BE replay / sigma | CN raw RMS | BE raw RMS | Sensor fit / sigma | Evals | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| temporal_pinn | 5.661164% | 2.834845 | 1.796802 | 0.00067380442 | 0.000427075539 | 0.994699 | 8075 | 6057.777 |
| be_adjoint_nn | 42.749003% | 2.609791 | 1.063778 | 0.00062031197 | 0.000252845646 | 1.063778 | 8118 | 2312.916 |

Both fits completed 1,000 Adam + 4,000 SSB; all 10 rounds capped, no timeouts. PINN has lower coefficient error but fails BOTH predeclared 1.1-sigma replay screens. BE adjoint passes BE and fails CN. No consistency-qualified PINN win; no post-hoc evaluation solver selection. Passing implementation checks is distinct from satisfying replay consistency.

PINN IC relative L2 2.49238522e-07; BC RMS 3.07129313e-18; independent continuous PDE RMS by early-to-late bins: 0.871827236, 0.633718299, 0.2610581, 0.00475271992.

Both arms sequential on same free Kaggle CPU host, two threads. Cross-cell/reference Colab/Kaggle hosts/devices differ, no unqualified speed or identical trajectory claim. One smooth field/noise realization/seed and capped optimization; density/noise-dependent gamma, frozen parent warm start, spatial AD/FD soft/hard state constraints qualify trends. Independent fine-grid RK4 data, not CN-generated. Negative consistency results retained with all outputs. User 5,000-sensor/5%-noise Colab pair remains separate and awaited, never dispatched by agent.

Active noise pairs: case07 (7,500/1%) and newly dispatched03 (3,000/10%). Next06 (5,000/10%) then09 (7,500/10%) as slots open; max2.
