# Colab return: 3,000 sensors, 5% noise

Independently recovered and evaluated 2026-10-21T20:20:58.494877+00:00. Only research case_02 is present; prechecks/SMOKE_DO_NOT_SCORE excluded. Original archive, returned notebook and outputs preserved. Notebook code cells unchanged; all export, frozen source and supplied input hashes match. Canonical local data replaces only the Colab data path in a separate local validation pilot.

| Method | Coefficient relative L2 error | Common CN / sigma | Common BE / sigma | CN raw RMS | BE raw RMS | Sensor fit / sigma | Evals | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| temporal_pinn | 16.481198% | 1.135548 | 1.150085 | 0.0013495216 | 0.00136679772 | 0.986327 | 7752 | 3189.125 |
| be_adjoint_nn | 23.009115% | 1.094925 | 0.991525 | 0.00130124393 | 0.00117836056 | 0.991525 | 7840 | 2090.977 |

Both fits completed 1,000 Adam + 4,000 SSB iterations; all 10 rounds capped, no budget timeouts. PINN has lower coefficient error, but fails BOTH predeclared 1.1-sigma common replay screens. Adjoint passes both. This is not a consistency-qualified PINN win. Lower fitted state error does not establish coefficient/state physics consistency.

Known IC relative L2 2.49238522e-07; BC RMS 3.17158575e-18. Independent continuous-PDE RMS by early-to-late bins: 0.626350517, 0.52899668, 0.265259277, 0.0132697841.

Source/input identity, both checkpoint coefficient reconstructions/bounds/shared initial SHA/gamma/24,000 full observations/temporal formula/common BE and CN were checked BEFORE coefficient truth access. Passing implementation checks is distinct from satisfying replay screens. See check/validation_before_truth.json, validation.json and primary_matched_neural_comparison.csv.

User Colab Tesla T4, Python3.13.15, Torch2.11.0+cu130, NumPy2.1.3, SciPy1.16.3. CPU2threads: PINN observation/residual on CUDA and curvature on CPU; BE adjoint on CPU. Same session sequential arms, no unqualified speed claim. Parent warm start, density-dependent gamma, different cells/hosts, spatial AD vs FD and soft vs hard state constraints qualify density comparisons. Data originate from independent fine-grid RK4. One smooth field/noise realization/seed and capped optimization do not establish population or convergence superiority.

Remaining user-owned density phase: 5,000 then 7,500 sensors at 5% noise. No new fits were dispatched by this import. Noise phase remains gated until density report review.

Recovery note: initial long Windows extraction path failed; archive and partial extraction retained in campaign/pilots/noise_density_v24_1000_7500/colab_returns/20261021T201715_672fef. Short-path extraction in colab_returns/r672fef succeeded with full manifest verification; no scientific data or source changes.
