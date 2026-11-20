# E1b: hidden walls -- PINN walls handed to the adjoint, against the classical joint inversion

relL2(D); wall error = RMS(estimated - true walls) / RMS(true walls); cost = CPU seconds (hybrids include the PINN). adjoint_truebc uses the hidden true walls: a bound, not a method.

## relL2(D)

| key                         |   adjoint_zero_walls (E1) |   pinn_bc |   hybrid_bc |   adjoint_joint5 |   hybrid_bc_joint5 |   adjoint_joint1 |   adjoint_truebc |
|:----------------------------|--------------------------:|----------:|------------:|-----------------:|-------------------:|-----------------:|-----------------:|
| H2u-A_smooth-heatedleft-a1  |                     3.466 |     0.409 |       6.365 |            0.339 |              0.339 |            0.288 |            0.316 |
| H2u-A_smooth-pulsedtop-a0.1 |                     5.609 |     0.321 |       5.655 |            0.300 |              0.292 |            0.157 |            0.098 |
| H2u-A_smooth-pulsedtop-a0.3 |                     2.992 |     0.343 |       0.406 |            0.241 |              0.244 |            0.161 |            0.097 |
| H2u-A_smooth-pulsedtop-a1   |                     4.401 |     0.332 |       0.358 |            0.315 |              0.316 |            0.221 |            0.136 |
| H2u-A_smooth-rampleft-a1    |                     3.561 |     0.418 |       0.313 |            0.383 |              0.383 |            0.329 |            0.217 |
| H2u-E_grf-heatedleft-a1     |                     5.379 |     0.327 |       4.827 |            0.283 |              0.283 |            0.283 |            0.244 |
| H2u-E_grf-pulsedtop-a0.3    |                     6.423 |     0.284 |       0.270 |            0.301 |              0.300 |            0.267 |            0.198 |
| H2u-E_grf-rampleft-a1       |                     4.326 |     0.376 |       0.344 |            0.286 |              0.286 |            0.262 |            0.203 |

## Medians

| arm                     |   median relL2(D) |
|:------------------------|------------------:|
| adjoint_zero_walls (E1) |             4.364 |
| pinn_bc                 |             0.338 |
| hybrid_bc               |             0.382 |
| adjoint_joint5          |             0.300 |
| hybrid_bc_joint5        |             0.296 |
| adjoint_joint1          |             0.264 |
| adjoint_truebc          |             0.201 |

## Wall error

| key                         |   pinn_bc |   hybrid_bc |   adjoint_joint5 |   hybrid_bc_joint5 |   adjoint_joint1 |   adjoint_truebc |
|:----------------------------|----------:|------------:|-----------------:|-------------------:|-----------------:|-----------------:|
| H2u-A_smooth-heatedleft-a1  |     0.351 |       0.351 |            0.085 |              0.086 |            0.083 |            0.000 |
| H2u-A_smooth-pulsedtop-a0.1 |     0.576 |       0.576 |            0.586 |              0.567 |            0.161 |            0.000 |
| H2u-A_smooth-pulsedtop-a0.3 |     0.598 |       0.598 |            0.339 |              0.334 |            0.097 |            0.000 |
| H2u-A_smooth-pulsedtop-a1   |     0.620 |       0.620 |            0.313 |              0.313 |            0.107 |            0.000 |
| H2u-A_smooth-rampleft-a1    |     0.287 |       0.287 |            0.097 |              0.097 |            0.087 |            0.000 |
| H2u-E_grf-heatedleft-a1     |     0.302 |       0.302 |            0.059 |              0.059 |            0.054 |            0.000 |
| H2u-E_grf-pulsedtop-a0.3    |     0.475 |       0.475 |            0.183 |              0.183 |            0.158 |            0.000 |
| H2u-E_grf-rampleft-a1       |     0.248 |       0.248 |            0.071 |              0.071 |            0.050 |            0.000 |

## Cost (CPU s)

| key                         |   pinn_bc |   hybrid_bc |   adjoint_joint5 |   hybrid_bc_joint5 |   adjoint_joint1 |   adjoint_truebc |
|:----------------------------|----------:|------------:|-----------------:|-------------------:|-----------------:|-----------------:|
| H2u-A_smooth-heatedleft-a1  |        71 |         896 |              306 |                358 |              375 |              381 |
| H2u-A_smooth-pulsedtop-a0.1 |        71 |         483 |              214 |                279 |              291 |               67 |
| H2u-A_smooth-pulsedtop-a0.3 |        68 |         153 |              277 |                344 |              353 |              141 |
| H2u-A_smooth-pulsedtop-a1   |        66 |         164 |              321 |                369 |              285 |              135 |
| H2u-A_smooth-rampleft-a1    |        62 |         198 |              233 |                292 |              311 |              152 |
| H2u-E_grf-heatedleft-a1     |        66 |         626 |              189 |                230 |              231 |               97 |
| H2u-E_grf-pulsedtop-a0.3    |        59 |         169 |              156 |                207 |              218 |               61 |
| H2u-E_grf-rampleft-a1       |        57 |         231 |              191 |                251 |              253 |              178 |

hybrid_bc_joint5 beats both pinn_bc and adjoint_joint5 (by > 1 %) on 1 of 8 datasets; hybrid_bc on 2.
