# User Colab case05: 5,000 sensors, 5% noise

Locally recovered/independently evaluated 2026-10-22T08:12:36.152693+00:00. Original ZIP/notebook/Colab outputs preserved. All export SHA/source/input hashes match. Notebook code cells unchanged; duplicate case02 is bitwise identical and reused without re-evaluation. Research case05 only; all smokes excluded. Only a separate local pilot data path is remapped to canonical data. Known ICBC/continuous-PDE then BOTH checkpoint/bounds/sharedinitial/gamma/full40,000-observation/temporal-formula/commonBE+CN checks passed BEFORE coefficient truth.

| Method | Coefficient relative L2 error | CN replay / sigma | BE replay / sigma | CN raw RMS | BE raw RMS | Sensor fit / sigma | Evals | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| temporal_pinn | 13.030871% | 1.102375277 | 1.068480677 | 0.0013100985 | 0.00126981705 | 0.989691 | 8046 | 3125.749 |
| be_adjoint_nn | 23.787751% | 1.097919177 | 0.991492483 | 0.00130480272 | 0.00117832179 | 0.991492 | 8184 | 2042.804 |

Both fits completed 1,000 Adam + 4,000 SSB, all 10 rounds capped, no budget timeouts. PINN has lower coefficient error but FAILS common CN replay (1.102375 sigma > predeclared 1.1), while passing BE replay. BE adjoint passes both. Not a consistency-qualified PINN win; keep threshold unchanged and do not round a failure into a pass.

PINN IC relative L2 2.49238522e-07; BC RMS 3.16616592e-18; independent continuous-PDE RMS by early-to-late bins: 0.777440256, 0.736432509, 0.287586916, 0.00974459459.

Colab Tesla T4/Python3.13.15/Torch2.11.0+cu130/NumPy2.1.3/SciPy1.16.3, CPU2 threads. PINN observations/residual on CUDA, curvature on CPU; BE adjoint on CPU, sequential within same session. Cross-cell Kaggle/Colab timing is qualified; no speed superiority or identical trajectory claim. One smooth field/noise realization/seed, capped optimization, gamma varying with noise/count and shared parent warm start qualify trends. AD/FD, soft/hard state constraints and random pairs/fixed grid differ. RK4 fine-grid data, not CN-generated.

This fills the last missing primary cell. Full12-cell scientific report is campaign/pilots/noise_density_v24_1000_7500/SWEEP_REPORT.md. No additional fits or held work resumed.
