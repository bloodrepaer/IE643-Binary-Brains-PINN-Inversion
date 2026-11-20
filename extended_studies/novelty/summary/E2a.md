# E2a: temperature-dependent diffusivity

relL2 against the reference field D(x, y, u = 0) (the true D for E2a).

| key             |   pinn |   pinn_T |
|:----------------|-------:|---------:|
| A_smooth        |  0.447 |    0.471 |
| B_gaussian_bump |  0.400 |    0.404 |
| C_inclusion     |  0.735 |    0.785 |
| D_layered       |  0.460 |    0.481 |
| E_grf           |  0.440 |    0.435 |
| F_defects       |  0.350 |    0.391 |

## Third network g(u) (true s(1) = strength S)

| key             |   strength_S |   g_at_1 |   g_rmse |   g_max_abs |   replay_nonlinear_over_sigma |
|:----------------|-------------:|---------:|---------:|------------:|------------------------------:|
| A_smooth        |        0.000 |    0.356 |    0.342 |       0.366 |                         4.639 |
| B_gaussian_bump |        0.000 |    0.128 |    0.094 |       0.128 |                         1.805 |
| C_inclusion     |        0.000 |    0.431 |    0.429 |       0.457 |                         2.737 |
| D_layered       |        0.000 |    0.669 |    0.408 |       0.669 |                         3.774 |
| E_grf           |        0.000 |    0.097 |    0.106 |       0.121 |                         3.258 |
| F_defects       |        0.000 |    0.307 |    0.299 |       0.324 |                         3.262 |
