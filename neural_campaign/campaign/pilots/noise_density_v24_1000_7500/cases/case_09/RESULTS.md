# v24 case09: 7,500 sensors, 10% noise

Complete and independently evaluated 2026-10-22T08:01:47.646906+00:00. Expected archive/log recovered once with CRC/path/symlink/casefold/conflict checks and SHA manifest; compact original extraction campaign/r24c09_v1. Supplied ICBC/continuous-PDE then BOTH checkpoint/bounds/shared-initial/gamma/full60,000-observation/temporal-formula/commonBE+CN checks passed BEFORE coefficient truth access. Initial asynchronous-download ordering check found no archive yet; same download awaited, no redownload; empty-coverage validation preserved before recovered validation.

| Method | D relative L2 error | CN replay / sigma | BE replay / sigma | CN raw RMS | BE raw RMS | Sensor fit / sigma | Evals | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| temporal_pinn | 22.636433% | 1.098240 | 1.121848 | 0.0026103683 | 0.00266648131 | 0.990164 | 8095 | 7284.148 |
| be_adjoint_nn | 23.263250% | 1.019340 | 0.992531 | 0.00242283297 | 0.00235911207 | 0.992531 | 8012 | 2380.245 |

Both fits completed 1,000 Adam + 4,000 SSB; all 10 rounds capped, no timeouts. PINN has slightly lower coefficient error and passes CN replay, but FAILS the predeclared BE replay limit of 1.1 sigma. BE adjoint passes both. Not a consistency-qualified PINN win; no post-hoc solver selection.

PINN IC relative L2 2.49238522e-07; BC RMS 3.13982522e-18; continuous PDE RMS early-to-late: 0.800674246, 0.625985181, 0.27889704, 0.0161224404.

Both arms sequential on same free Kaggle CPU host, two threads. Different cells/reference Colab/Kaggle hosts/devices differ; no speed/identical trajectory claim. One smooth field/noise realization/seed, capped optimization, density/noise-dependent gamma and frozen parent warm start qualify trends. Spatial AD/FD soft/hard state constraints differ. Data originate from independent fine-grid RK4, not CN. All outcomes retained. User case05 (5,000/5%) still awaited, not a method loss. No further agent fits/active remote jobs/pending downloads.
