# analysis: rescoring and figures

Scripts that rescored the PINN runs of the main benchmark with the common CN solver and drew the comparison figures, with their outputs.


## Scripts

| Script | Produces |
|---|---|
| `rescore.py` | `rescored.csv` (and `fields.npz`): every PINN estimate of the main benchmark replayed through the CN solver of `core/adjoint_inversion.py` and scored exactly like the adjoint, including error on the high- and low-sensitivity halves of the domain, a scale-corrected error and the replay bias |
| `fig_adjoint.py` | Classical-adjoint figures A1-A7: replay calibration, sensors x noise, horizon x protocol, temporal density, layout, data fit vs accuracy, cost |
| `fig_pinn.py` | PINN vs adjoint figures P1-P6: accuracy, replay consistency, field gallery, scale bias, density on C_inclusion, profiles |
| `fig_sweep.py` | L1 (PINN loss-weight sweep) and L2 (PINN training curves) |
| `fig_defects.py` | F1 (defect detection on F_defects fields) |
| `style.py` | Shared matplotlib style imported by the figure scripts |

## Tables

| File | Contents |
|---|---|
| `rescored.csv` | Output of `rescore.py`: 47 PINN runs on main-benchmark datasets, common scoring |
| `tikhonov_summary_all_389.csv` | Classical adjoint (Tikhonov) metrics on all 389 main-benchmark datasets |
| `sweep_pinn_vs_adj.csv` | 48 PINN runs (A_smooth, E_grf, C_inclusion x 5 / 20 / 50 / 100 sensors x 0 / 1 / 5 / 10 % noise, fixed loss weights) with the adjoint's and the best constant's relL2(D) and the PINN / adjoint ratio |
| `seed_table.csv` | PINN relL2(D) over three seeds per dataset, with the adjoint's value |

## Figures (`figures/`)

| Figures | Made by |
|---|---|
| `A1`-`A7` | `fig_adjoint.py` |
| `P1`-`P6` | `fig_pinn.py` |
| `L1`, `L2` | `fig_sweep.py` |
| `F1` | `fig_defects.py` |
| `C1_pinnwin_comparison.png`, `W1_pinnwin_families.png`, `W2_misspecification_gap.png`, `W3_pinnwin2.png` (study 3); `S1_pinn_vs_adjoint_sweep.png` (PINN / adjoint error ratio from `sweep_pinn_vs_adj.csv`) | Made in the same workspace from `../study3_pinn_favoured/results/` and `sweep_pinn_vs_adj.csv`; the plotting scripts are not included |

The report uses `C1_pinnwin_comparison.png` and `W2_misspecification_gap.png`.
