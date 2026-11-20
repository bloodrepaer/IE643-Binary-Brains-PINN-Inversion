# E2b: temperature-dependent diffusivity

relL2 against the reference field D(x, y, u = 0) (the true D for E2a).

| key                                                 |   adjoint |   pinn |   pinn_T |
|:----------------------------------------------------|----------:|-------:|---------:|
| A_smooth-arrS0.5  [uniform8/n100/random/eta0.01]    |     0.064 |  0.411 |    0.459 |
| A_smooth-arrS0.5                                    |     0.104 |  0.449 |    0.476 |
| A_smooth-arrS0  [uniform8/n100/random/eta0.01]      |     0.059 |  0.410 |    0.450 |
| A_smooth-arrS0                                      |     0.103 |  0.447 |    0.471 |
| A_smooth-arrS1  [uniform8/n100/random/eta0.01]      |     0.131 |  0.411 |    0.442 |
| A_smooth-arrS1                                      |     0.111 |  0.434 |    0.468 |
| A_smooth-arrS2  [uniform8/n100/random/eta0.01]      |     0.629 |  0.410 |    0.414 |
| A_smooth-arrS2                                      |     0.139 |  0.427 |    0.464 |
| C_inclusion-arrS0.5  [uniform8/n100/random/eta0.01] |     0.217 |  0.700 |    0.727 |
| C_inclusion-arrS0.5                                 |     0.654 |  0.725 |    0.779 |
| C_inclusion-arrS0  [uniform8/n100/random/eta0.01]   |     0.203 |  0.713 |    0.735 |
| C_inclusion-arrS0                                   |     0.661 |  0.735 |    0.785 |
| C_inclusion-arrS1  [uniform8/n100/random/eta0.01]   |     0.253 |  0.689 |    0.730 |
| C_inclusion-arrS1                                   |     0.646 |  0.721 |    0.766 |
| C_inclusion-arrS2  [uniform8/n100/random/eta0.01]   |     0.783 |  0.664 |    0.709 |
| C_inclusion-arrS2                                   |     0.590 |  0.711 |    0.804 |

## Third network g(u) (true s(1) = strength S)

| key                                                 |   strength_S |   g_at_1 |   g_rmse |   g_max_abs |   replay_nonlinear_over_sigma |
|:----------------------------------------------------|-------------:|---------:|---------:|------------:|------------------------------:|
| A_smooth-arrS0.5  [uniform8/n100/random/eta0.01]    |        0.500 |    0.412 |    0.138 |       0.412 |                        24.983 |
| A_smooth-arrS0.5                                    |        0.500 |    0.304 |    0.137 |       0.330 |                         4.934 |
| A_smooth-arrS0  [uniform8/n100/random/eta0.01]      |        0.000 |    0.461 |    0.413 |       0.462 |                        23.315 |
| A_smooth-arrS0                                      |        0.000 |    0.356 |    0.342 |       0.366 |                         4.639 |
| A_smooth-arrS1  [uniform8/n100/random/eta0.01]      |        1.000 |    0.541 |    0.207 |       0.541 |                        24.300 |
| A_smooth-arrS1                                      |        1.000 |    0.349 |    0.317 |       0.373 |                         4.890 |
| A_smooth-arrS2  [uniform8/n100/random/eta0.01]      |        2.000 |    0.452 |    0.857 |       0.452 |                        24.414 |
| A_smooth-arrS2                                      |        2.000 |    0.324 |    0.909 |       0.349 |                         5.193 |
| C_inclusion-arrS0.5  [uniform8/n100/random/eta0.01] |        0.500 |    0.518 |    0.210 |       0.518 |                        13.572 |
| C_inclusion-arrS0.5                                 |        0.500 |    0.595 |    0.335 |       0.596 |                         2.662 |
| C_inclusion-arrS0  [uniform8/n100/random/eta0.01]   |        0.000 |    0.529 |    0.459 |       0.529 |                        13.153 |
| C_inclusion-arrS0                                   |        0.000 |    0.431 |    0.429 |       0.457 |                         2.737 |
| C_inclusion-arrS1  [uniform8/n100/random/eta0.01]   |        1.000 |    0.588 |    0.240 |       0.588 |                        12.110 |
| C_inclusion-arrS1                                   |        1.000 |    0.535 |    0.256 |       0.536 |                         2.407 |
| C_inclusion-arrS2  [uniform8/n100/random/eta0.01]   |        2.000 |    0.478 |    0.800 |       0.478 |                        12.937 |
| C_inclusion-arrS2                                   |        2.000 |    0.778 |    0.612 |       0.783 |                         3.522 |
