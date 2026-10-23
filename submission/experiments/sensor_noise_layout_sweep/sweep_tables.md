Completed pairs: 44 of 44. relL2(D), lower is better; cells are adjoint / PINN.


### C_inclusion: sensors x noise (random layout)

| sensors | 0% noise | 1% noise | 5% noise | 10% noise |
|---|---|---|---|---|
| 5 | 0.594 / 0.747 | 0.669 / 0.760 | 0.779 / 0.746 | 0.709 / 0.743 |
| 20 | 0.332 / 0.718 | 0.600 / 0.720 | 0.589 / 0.738 | 0.679 / 0.740 |
| 50 | 0.388 / 0.681 | 0.277 / 0.684 | 0.654 / 0.728 | 0.690 / 0.727 |
| 100 | 0.656 / 0.701 | 0.191 / 0.700 | 0.247 / 0.687 | 0.355 / 0.747 |

### C_inclusion: layouts at 5% noise

| layout | 20 sensors | 50 sensors |
|---|---|---|
| random | 0.589 / 0.738 | 0.654 / 0.728 |
| grid | 0.659 / 0.968 | 0.510 / 0.690 |
| clustered | 0.527 / 0.686 | 0.654 / 0.688 |
| boundary | 1.529 / 0.766 | 0.747 / 0.740 |

### B_gaussian_bump: sensors x noise (random layout)

| sensors | 0% noise | 1% noise | 5% noise | 10% noise |
|---|---|---|---|---|
| 5 | 0.227 / 0.542 | 0.297 / 0.539 | 0.394 / 0.525 | 0.413 / 0.512 |
| 20 | 0.080 / 0.483 | 0.260 / 0.490 | 0.289 / 0.495 | 0.358 / 0.505 |
| 50 | 0.014 / 0.339 | 0.119 / 0.341 | 0.229 / 0.400 | 0.317 / 0.433 |
| 100 | 0.019 / 0.328 | 0.099 / 0.353 | 0.189 / 0.355 | 0.204 / 0.417 |

### B_gaussian_bump: layouts at 5% noise

| layout | 20 sensors | 50 sensors |
|---|---|---|
| random | 0.289 / 0.495 | 0.229 / 0.400 |
| grid | 0.163 / 0.416 | 0.116 / 0.358 |
| clustered | 0.201 / 0.482 | 0.234 / 0.470 |
| boundary | 0.382 / 0.510 | 0.379 / 0.509 |

Wins: adjoint 41, PINN 3.
Median relL2: adjoint 0.357, PINN 0.612, best constant 0.680.
Median wall time (1 CPU thread each): adjoint 33 s, PINN 58 s.
PINN wins: C_inclusion__Fo0.1__uniform8__n5__random__eta0.05 (0.779 vs 0.746), C_inclusion__Fo0.1__uniform8__n20__boundary__eta0.05 (1.529 vs 0.766), C_inclusion__Fo0.1__uniform8__n50__boundary__eta0.05 (0.747 vs 0.740)
