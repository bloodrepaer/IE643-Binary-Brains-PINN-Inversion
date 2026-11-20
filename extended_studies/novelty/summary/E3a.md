# E3a: sensor-split validation

relL2(D). oracle_80 = best of the 10 settings by the TRUE D (trained on 80 %), for reference only.

| dataset                                         |   pinn_default |   pinn_selected |   adjoint_dp |   adjoint_cv | selected_config   |   oracle_80 | oracle_config           |
|:------------------------------------------------|---------------:|----------------:|-------------:|-------------:|:------------------|------------:|:------------------------|
| A_smooth                                        |          0.447 |           0.372 |        0.103 |        0.103 | C08_high_pde_tik  |       0.404 | C08_high_pde_tik        |
| B_gaussian_bump  [uniform8/n100/random/eta0.01] |          0.353 |           0.326 |        0.099 |        0.067 | C08_high_pde_tik  |       0.324 | C06_sweetspot_tv        |
| B_gaussian_bump                                 |          0.400 |           0.398 |        0.229 |        0.302 | C08_high_pde_tik  |       0.397 | C04_balanced_lowreg_tik |
| C_inclusion  [uniform8/n100/random/eta0.1]      |          0.736 |           0.706 |        0.499 |        0.584 | C08_high_pde_tik  |       0.722 | C01_equal_unreg         |
| C_inclusion                                     |          0.735 |           0.726 |        0.661 |        0.661 | C08_high_pde_tik  |       0.729 | C08_high_pde_tik        |
| D_layered                                       |          0.460 |           0.449 |        0.216 |        0.201 | C08_high_pde_tik  |       0.432 | C08_high_pde_tik        |
| E_grf                                           |          0.440 |           0.404 |        0.207 |        0.161 | C08_high_pde_tik  |       0.378 | C08_high_pde_tik        |
| F_defects                                       |          0.350 |           0.313 |        0.182 |        0.179 | C08_high_pde_tik  |       0.321 | C08_high_pde_tik        |
