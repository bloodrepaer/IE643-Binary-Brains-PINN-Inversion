# E3d: interpolation augmentation of sparse sensors

relL2(D).

| key                                         |   adjoint |   adjoint_aug |   pinn |   pinn_aug |
|:--------------------------------------------|----------:|--------------:|-------:|-----------:|
| A_smooth  [uniform8/n20/random/eta0.05]     |     0.128 |         2.570 |  0.529 |      0.444 |
| C_inclusion  [uniform8/n20/random/eta0.05]  |     0.631 |         1.085 |  0.736 |      0.730 |
| C_inclusion  [uniform8/n5/random/eta0.01]   |     0.645 |        11.428 |  0.754 |      0.738 |
| D_layered  [uniform8/n20/clustered/eta0.05] |     0.410 |         0.693 |  0.608 |      0.622 |
| E_grf  [uniform8/n20/random/eta0.01]        |     0.170 |         4.185 |  0.476 |      0.471 |
| F_defects  [uniform8/n20/boundary/eta0.05]  |     0.225 |         8.611 |  0.450 |      0.548 |
