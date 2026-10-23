# IE643 experiment findings (partial)

Updated 2026-10-21T09:55:54.427911+00:00.

Primary comparison: neural adjoint versus neural PINN. Grid adjoint is a secondary baseline. Both primary methods represent the unknown coefficient by a neural network; the neural adjoint obtains temperature through the discrete forward solver, while the PINN learns a temperature network. Versioned pilot protocols with different coefficient networks or objectives are reported separately until matched adjoint runs exist.

Completed new fits: 856 / 1510.
Paired completed neural comparisons: 8.

## Primary neural comparison

Full paired rows, coefficient error, common replay, sensor fit, convergence flags and qualified timings: primary_neural_comparison.csv. Only identical collection/key/physics rows are paired. Sparse pilot coverage and failed PINN consistency checks limit conclusions; unfinished PINNs are not method losses.

## Current evidence

```text
                                                           cases  median_rel_L2_D  median_replay  median_wall_s
collection                   physics_mode method                                                               
datasets_v2_density          matched      adjoint_nn           4         0.089730            NaN     843.611357
                                          grid_historical      4         0.053388            NaN     398.913970
datasets_v2_density_extended matched      adjoint_nn         180         0.186909       1.013825     202.195687
                                          grid               180         0.158228       0.993104      48.474741
datasets_v2_density_wideic   matched      adjoint_nn           2         0.083277            NaN     820.877980
                                          grid_historical      2         0.014365            NaN     351.412504
datasets_v2_pinnwin          assumed_zero adjoint_nn          14         4.371445       5.408862    1235.021689
                                          grid_historical     14         3.943616       5.559927     306.446575
                             matched      adjoint_nn          43         0.129419       0.999093     389.464682
                                          grid_historical     43         0.130343       0.992602      61.994482
datasets_v2_pinnwin2         assumed_zero adjoint_nn           8         8.066540      12.000949     660.757851
                                          grid_historical      8         6.514041      13.401128     137.945560
                             matched      adjoint_nn          16         0.126009       1.018126     526.492528
                                          grid_historical     16         0.108760       0.998163      92.507319
datasets_v2_pinnwin3         assumed_zero adjoint_nn           6         5.878182      28.840392     467.790560
                                          grid                 6         5.083858      32.032895     117.448155
datasets_v2_rk4_zero         matched      adjoint_nn         389         0.225510       0.995079     347.835636
                                          grid_historical    389         0.224634       0.984142     134.593296
                                          pinn                 8         0.607507       5.806012     340.805615
```

The initial GPU PINNs fit the sensor temperatures near the noise level, yet their coefficients fail common CN replay (approximately 5.7-7.3 sigma on A/B/E). Sensor interpolation alone is not reliable coefficient recovery.

## Interpretation limits

Only completed pairs count toward a method comparison. Pending or failed runs are not losses. Broad single-seed coverage is exploratory; representative conclusions need additional seeds.

Grid and neural adjoints use discrepancy paths; the PINN uses fixed alpha 1. The two neural models share coefficient architecture and initial field, but differ in state representation, optimizer and physics enforcement. Historical and current hardware timings are not directly comparable.

Matched-physics and assumed-zero hidden-physics rows must be discussed separately. Told-model replay can look good for the wrong coefficient under misspecification. Optional true-physics replay reads hidden BC/source only after fitting.

Convergence caps are recorded. A discrepancy flag alone does not prove optimizer convergence or a physical/statistical noise floor.

## Added sensor-density experiment

180 additional configurations cover six canonical fields, 5/20/50/100/200/400/800/1600/3200/3969 sensors, and 1/5/10% noise. Fo=0.1 and eight uniform observation times are fixed. Sensor locations, clean values and noise draws are nested within each field/noise group; original problem and answer-key files are unchanged.
This uses grid-node sensors and is analyzed separately from the historical off-grid layouts. Compare full density curves at fixed field/noise before pooling. Larger datasets can require more optimization, so budget caps and data-minibatch sampling must be reported. Low-noise layered fields may have a discretization-error floor even with the true coefficient.

## Paired descriptive results

matched: 8 pairs; lower coefficient error for neural adjoint in 8, PINN in 0; equal in 0. This is descriptive of the completed subset.

## Paired neural and grid adjoints

```text
                                        pairs  neural_lower_error  grid_lower_error  median_neural_error  median_grid_error
physics_mode tier                                                                                                          
assumed_zero A_smooth                      15                   4                11             5.808715           4.791310
             E_grf                         10                   3                 7             7.205963           5.629625
             E_grf_corrlen_long             3                   3                 0             2.322007           3.097502
matched      A_smooth                     123                  44                79             0.123819           0.112490
             B_gaussian_bump               93                  76                17             0.129172           0.179741
             C_inclusion                   93                  45                48             0.643630           0.510728
             C_inclusion_contrast_high      1                   1                 0             0.535220           0.639816
             C_inclusion_contrast_low       1                   1                 0             0.152858           0.156257
             D_checkerboard_coarse          1                   0                 1             0.555636           0.497429
             D_checkerboard_fine            2                   0                 2             1.329262           0.772914
             D_layered                     93                  79                14             0.185682           0.240362
             E_grf                        101                  28                73             0.208304           0.177281
             E_grf_corrlen_long            12                   9                 3             0.180837           0.199744
             E_grf_corrlen_short            2                   0                 2             0.358760           0.267040
             F_defects                     93                   7                86             0.233510           0.145104
             F_defects_low_contrast         1                   0                 1             0.085631           0.077889
             F_defects_many                 1                   0                 1             0.306180           0.227319
             F_defects_single               1                   0                 1             0.101164           0.082580
             F_defects_small                1                   0                 1             0.116338           0.109182
             F_defects_voids                1                   0                 1             0.395827           0.153172
             G_lowfreq                     14                  11                 3             0.062204           0.066145
```

Only identical observation cases and physics assumptions are paired. These are descriptive single-seed comparisons under the retained budgets and discrepancy paths, not a controlled claim that the representation alone causes each difference. Historical grid and current neural timings use different execution conditions. Full rows are in paired_adjoint_results.csv.

## Paired nested-density comparisons

```text
           tier  eta  paired_densities  complete_curve  neural_lower_error  grid_lower_error  median_error_ratio  full_density_neural_error  full_density_grid_error
       A_smooth 0.01                10            True                   0                10            1.714410                   0.076642                 0.045939
       A_smooth 0.05                10            True                   0                10            1.361451                   0.090481                 0.058893
       A_smooth 0.10                10            True                   2                 8            1.121251                   0.113693                 0.070948
B_gaussian_bump 0.01                10            True                   7                 3            0.766201                   0.110900                 0.065095
B_gaussian_bump 0.05                10            True                  10                 0            0.733297                   0.066105                 0.089927
B_gaussian_bump 0.10                10            True                  10                 0            0.849876                   0.156261                 0.187882
    C_inclusion 0.01                10            True                   0                10            8.741282                   2.172386                 0.233878
    C_inclusion 0.05                10            True                   4                 6            2.029123                   0.824791                 0.217458
    C_inclusion 0.10                10            True                   4                 6            1.129169                   0.471026                 0.478661
      D_layered 0.01                10            True                   8                 2            0.594239                   0.086648                 0.152954
      D_layered 0.05                10            True                   8                 2            0.935660                   0.134939                 0.151464
      D_layered 0.10                10            True                   9                 1            0.976519                   0.156986                 0.161009
          E_grf 0.01                10            True                   0                10            3.164799                   0.171004                 0.031553
          E_grf 0.05                10            True                   3                 7            1.374352                   0.171970                 0.050880
          E_grf 0.10                10            True                   2                 8            1.150094                   0.158488                 0.121565
      F_defects 0.01                10            True                   1                 9            2.401963                   0.224383                 0.075009
      F_defects 0.05                10            True                   0                10            1.772981                   0.206251                 0.111910
      F_defects 0.10                10            True                   0                10            1.379689                   0.202370                 0.127413
```

Each row fixes field and noise; only identical nested observations are paired. Complete curves require all ten prespecified densities. Partial curve counts are coverage, not complete-curve method rankings. An error ratio below one favors the neural adjoint. These single-seed, retained-budget comparisons do not establish converged optima.

## Dense inclusion failure localization (post-fit evaluation)

```text
 eta     method  rel_L2_D  replay_sigma    D_max  low_sensitivity_error_energy_fraction  boundary_error_energy_fraction  worst_one_percent_error_energy_fraction
0.01       grid  0.233878      1.026969 0.896466                               0.480075                        0.000130                                 0.295377
0.01 adjoint_nn  2.172386      1.148949 4.493559                               0.834126                        0.000009                                 0.727472
0.05       grid  0.217458      0.997586 0.731295                               0.343370                        0.000867                                 0.362541
0.05 adjoint_nn  0.824791      1.014038 2.190069                               0.820732                        0.000526                                 0.731173
0.10       grid  0.478661      1.012971 0.349788                               0.465134                        0.000073                                 0.197016
0.10 adjoint_nn  0.471026      1.022159 0.462705                               0.400586                        0.000348                                 0.219478
```

Both methods are evaluated against the answer key only after fitting, using the same fitted-constant forward sensitivity proxy and disjoint interior-high/interior-low/boundary masks. At 1% noise, the neural estimate concentrates about 73% of squared coefficient error in its worst 1% of grid nodes, and about 83% in the common low-sensitivity interior mask; boundary nodes contribute less than 0.001%. The maps show a sharp central overshoot. This is not a boundary-only artifact. The sensitivity proxy does not establish nonidentifiability, and iteration caps prevent separating optimization from representation limitations. Original per-fit high/low sensitivity metrics use different masks and should not be treated as a common-mask comparison. No checkpoint is refitted or selected here.

## Local defect recovery at full sensor density

```text
    method  eta  rel_L2_D  defect_dice  defects_detected  n_defects  background_rel_err
adjoint_nn 0.01  0.224383     0.421687                 3          3            0.176399
      grid 0.01  0.075009     0.944751                 3          3            0.026851
adjoint_nn 0.05  0.206251     0.144000                 3          3            0.133986
      grid 0.05  0.111910     0.896142                 3          3            0.040180
adjoint_nn 0.10  0.202370     0.020513                 2          3            0.107537
      grid 0.10  0.127413     0.880952                 3          3            0.059245
```

Global coefficient error can hide missed small defects. These metrics use the retained evaluation definitions: detected means recovered signed log-contrast at least 25% of the true contrast, and Dice uses the original anomaly threshold. Neither threshold is tuned to these results. Full density curves and per-defect contrast/localization scores are saved in defect_results.csv, defect_details.csv and figures/density_defects.png. These are synthetic post-fit evaluations, not real-data detection guarantees.

## Completed density curves

```text
           tier  eta     method  cases  error_5  error_3969  relative_error_reduction_percent  error_increases_between_adjacent_densities
       A_smooth 0.01 adjoint_nn     10 0.256743    0.076642                         70.148526                                           1
       A_smooth 0.01       grid     10 0.208719    0.045939                         77.989869                                           1
       A_smooth 0.05 adjoint_nn     10 1.193404    0.090481                         92.418229                                           2
       A_smooth 0.05       grid     10 0.908095    0.058893                         93.514671                                           0
       A_smooth 0.10 adjoint_nn     10 0.332062    0.113693                         65.761427                                           2
       A_smooth 0.10       grid     10 0.332582    0.070948                         78.667447                                           2
B_gaussian_bump 0.01 adjoint_nn     10 0.188135    0.110900                         41.053231                                           4
B_gaussian_bump 0.01       grid     10 0.386426    0.065095                         83.154592                                           2
B_gaussian_bump 0.05 adjoint_nn     10 0.384738    0.066105                         82.818176                                           2
B_gaussian_bump 0.05       grid     10 0.472450    0.089927                         80.965751                                           3
B_gaussian_bump 0.10 adjoint_nn     10 0.500489    0.156261                         68.778341                                           3
B_gaussian_bump 0.10       grid     10 0.506303    0.187882                         62.891438                                           4
    C_inclusion 0.01 adjoint_nn     10 0.723490    2.172386                       -200.264957                                           5
    C_inclusion 0.01       grid     10 0.693253    0.233878                         66.263740                                           3
    C_inclusion 0.05 adjoint_nn     10 0.720717    0.824791                        -14.440371                                           3
    C_inclusion 0.05       grid     10 0.742053    0.217458                         70.695095                                           1
    C_inclusion 0.10 adjoint_nn     10 0.744692    0.471026                         36.748889                                           4
    C_inclusion 0.10       grid     10 0.754944    0.478661                         36.596430                                           3
      D_layered 0.01 adjoint_nn     10 0.249139    0.086648                         65.221255                                           4
      D_layered 0.01       grid     10 0.232398    0.152954                         34.184363                                           4
      D_layered 0.05 adjoint_nn     10 0.223457    0.134939                         39.613033                                           2
      D_layered 0.05       grid     10 0.219431    0.151464                         30.974228                                           2
      D_layered 0.10 adjoint_nn     10 0.260144    0.156986                         39.654410                                           2
      D_layered 0.10       grid     10 0.265990    0.161009                         39.468085                                           2
          E_grf 0.01 adjoint_nn     10 0.332196    0.171004                         48.523249                                           4
          E_grf 0.01       grid     10 0.317850    0.031553                         90.073139                                           1
          E_grf 0.05 adjoint_nn     10 0.356211    0.171970                         51.722360                                           3
          E_grf 0.05       grid     10 0.354465    0.050880                         85.646077                                           0
          E_grf 0.10 adjoint_nn     10 0.344555    0.158488                         54.002176                                           3
          E_grf 0.10       grid     10 0.342748    0.121565                         64.532137                                           3
      F_defects 0.01 adjoint_nn     10 0.707893    0.224383                         68.302641                                           4
      F_defects 0.01       grid     10 1.374266    0.075009                         94.541889                                           1
      F_defects 0.05 adjoint_nn     10 0.276036    0.206251                         25.281279                                           2
      F_defects 0.05       grid     10 0.235411    0.111910                         52.462003                                           1
      F_defects 0.10 adjoint_nn     10 0.475897    0.202370                         57.475967                                           3
      F_defects 0.10       grid     10 0.290003    0.127413                         56.065068                                           2
```

These are complete ten-density curves for the listed method/field/noise groups only. Method comparisons require matched completed curves. Sensor noise and locations are nested; there is one observation realization per field/noise group.
Nonmonotonic coefficient error can reflect regularization selection as well as optimization and noise. The discrepancy rule stops at the first alpha reaching RMS/sigma <=1. If none reaches it before the path plateau rule, it selects the strongest alpha whose RMS is within 2% of the best path value. A small crossing of one can therefore change the selected regularization substantially. The floor_above_sigma flag does not establish a physical error floor. Inspect density_path_diagnostics.csv for selected alpha, path minimum and iteration caps; grid outputs do not record optimizer success, so an iteration count alone cannot prove convergence.

## Post-fit full-density forward consistency

Evaluation only: replay the true supplied coefficient through the same 200-step CN solver. This does not select inverse models or training settings. The clean-field discrepancy isolates forward discretization and observation mismatch; noisy replay also includes the actual noise realization.

```text
           tier  noise  true_D_CN_replay_sigma  true_D_CN_vs_clean_sigma  selected_grid_replay_sigma  selected_grid_coefficient_error
       A_smooth   0.01                1.005059                  0.086771                    1.016693                         0.045939
       A_smooth   0.05                0.996631                  0.017354                    0.997029                         0.058893
       A_smooth   0.10                1.000118                  0.008677                    0.998791                         0.070948
B_gaussian_bump   0.01                0.997578                  0.043808                    0.997317                         0.065095
B_gaussian_bump   0.05                1.000522                  0.008762                    0.998738                         0.089927
B_gaussian_bump   0.10                0.994372                  0.004381                    0.998393                         0.187882
    C_inclusion   0.01                1.006959                  0.080256                    1.026969                         0.233878
    C_inclusion   0.05                0.994827                  0.016051                    0.997586                         0.217458
    C_inclusion   0.10                1.001230                  0.008026                    1.012971                         0.478661
      D_layered   0.01                2.097942                  1.848848                    1.881853                         0.152954
      D_layered   0.05                1.063369                  0.369770                    1.058220                         0.151464
      D_layered   0.10                1.012816                  0.184885                    1.020128                         0.161009
          E_grf   0.01                0.998093                  0.056419                    0.997776                         0.031553
          E_grf   0.05                1.000896                  0.011284                    0.998613                         0.050880
          E_grf   0.10                0.996453                  0.005642                    0.999722                         0.121565
      F_defects   0.01                1.006142                  0.102423                    1.056138                         0.075009
      F_defects   0.05                0.992572                  0.020485                    0.999822                         0.111910
      F_defects   0.10                0.995000                  0.010242                    0.995938                         0.127413
```

Coefficient error and temperature discrepancy quantify different things. A noise-level temperature replay can coexist with substantial coefficient error in weakly observed regions or restricted coefficient representations. A path flag alone cannot attribute the coefficient error to forward discretization.

### Forward time-step refinement (evaluation only)

```text
     tier  n_steps  noise  true_D_CN_replay_sigma  true_D_CN_vs_clean_sigma
 A_smooth      200   0.01                1.005059                  0.086771
 A_smooth      400   0.01                1.004768                  0.082798
 A_smooth      800   0.01                1.004703                  0.081871
D_layered      200   0.01                2.097942                  1.848848
D_layered      400   0.01                2.095567                  1.846138
D_layered      800   0.01                2.094975                  1.845463
```

The layered-field mismatch barely changes from 200 to 800 CN steps. Temporal refinement alone does not resolve it; spatial/operator and data-generation differences remain to be separated. This is a post-fit diagnostic, not truth-based selection of inverse training settings.

### Forward spatial-grid check (evaluation only)

```text
     tier  grid_N  n_steps  noise  true_D_CN_replay_sigma  true_D_CN_vs_clean_sigma
 A_smooth      65      200   0.01                1.005059                  0.086771
 A_smooth     129      200   0.01                1.002377                  0.009479
 A_smooth      65      800   0.01                1.004703                  0.081871
 A_smooth     129      800   0.01                1.002386                  0.000592
D_layered      65      200   0.01                2.097942                  1.848848
D_layered     129      200   0.01                1.006092                  0.006986
D_layered      65      800   0.01                2.094975                  1.845463
D_layered     129      800   0.01                1.006098                  0.000437
```

For the layered field at 1% noise, switching from the inversion grid (65x65) to the original data-generation grid (129x129) reduces true-D clean-data discrepancy from about 1.849 to 0.007 sensor sigma at 200 CN steps. At 800 steps it falls from 1.845 to 0.000437 sigma. This isolates the coarser spatial discretization as the dominant mismatch in these forward checks; simply increasing CN time steps on the coarse grid is insufficient. Original observations, inverse estimates and training settings are preserved.

### Retained layered estimates on a finer replay grid

```text
 eta     method  grid_N               deployment  replay_sigma
0.01       grid      65                 retained      1.881853
0.01       grid     129 common_logD_prolongation      1.888405
0.01 adjoint_nn      65                 retained      1.071962
0.01 adjoint_nn     129 common_logD_prolongation      1.171216
0.01 adjoint_nn     129   native_neural_function      1.105776
0.05       grid      65                 retained      1.058220
0.05       grid     129 common_logD_prolongation      1.058646
0.05 adjoint_nn      65                 retained      1.014589
0.05 adjoint_nn     129 common_logD_prolongation      1.017342
0.05 adjoint_nn     129   native_neural_function      1.015152
0.10       grid      65                 retained      1.020128
0.10       grid     129 common_logD_prolongation      1.020346
0.10 adjoint_nn      65                 retained      1.013612
0.10 adjoint_nn     129 common_logD_prolongation      1.014159
0.10 adjoint_nn     129   native_neural_function      1.013565
```

These are post-fit replays, not refits, and use no coefficient truth or answer-key snapshots. The supplied analytic IC is evaluated on129-grid after checking agreement with the original65-grid IC. Common transfer bilinearly interpolates logD for both methods; native neural evaluation is a separate deployment. All transferred fields preserve original65-grid node values. At1% noise and3969 sensors, fine common-transfer replay is1.171 neural versus1.888 grid sigma; native neural replay is1.106. This does not replace the common65-grid campaign comparison or establish convergence.

### PINN state consistency on a finer replay grid

```text
                 pilot  grid_N  n_steps  state_vs_CN_sigma  CN_vs_observations_sigma  state_vs_observations_sigma
      ic_locald_v6_gpu      65      200           2.342708                  2.605599                     0.753668
      ic_locald_v6_gpu     129      200           2.345521                  2.608234                     0.753668
fixed_d_forward_v7_gpu      65      200           0.722944                  0.941967                     1.254996
fixed_d_forward_v7_gpu     129      200           0.723975                  0.940545                     1.254996
fixed_d_refresh_v8_gpu      65      200           0.773713                  0.941967                     1.259339
fixed_d_refresh_v8_gpu     129      200           0.775830                  0.940545                     1.259339
```

This post-fit check uses only the supplied analytic IC and each retained coefficient function, without coefficient truth or refitting. Refining65 to129 nodes changes v6 state-versus-CN mismatch from2.343 to2.346 noise sigma, v7 from.723 to.724 and v8 from.774 to.776. The observed mismatches are therefore not removed by this spatial refinement. The analytic IC check introduces only the recorded small CSV-rounding difference relative to original replay.

## Known-initial-condition diagnosis

The completed plain-state PINNs were checked against the supplied initial field, without reading the true coefficient. These v0 models substantially underfit the initial condition despite matching later sensor measurements.
```text
                                                   key  ic_relative_L2  ic_true_peak  ic_predicted_peak
       A_smooth__Fo0.1__uniform8__n50__random__eta0.05        0.695190      0.996101           0.285661
B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05        0.707289      0.996101           0.256941
          E_grf__Fo0.1__uniform8__n50__random__eta0.05        0.759652      0.996101           0.171914
    C_inclusion__Fo0.1__uniform8__n50__random__eta0.05        0.623025      0.996101           0.286874
      D_layered__Fo0.1__uniform8__n50__random__eta0.05        0.435063      0.996101           0.869398
      F_defects__Fo0.1__uniform8__n50__random__eta0.05        0.720855      0.996101           0.208970
        A_smooth__Fo0.02__early8__n50__random__eta0.05        0.133335      0.996101           0.962398
        A_smooth__Fo0.02__final1__n50__random__eta0.05        0.137352      0.996101           0.988340
```

Further plain-state batch preparation is held by campaign/pinn_review_gate.json. Separate versioned pilots retain the same coefficient network and observations and are evaluated using known constraints and replay, not coefficient truth.

Bulk gate HELD. Smooth temporal PINN17.23% vsNN20.00%,replay1.09916vs0.98132; narrow predeclaredscreen pass,onecase/seed,capped. Random no advantage/replayfail. B/Dpartial controls,Fmissing;no migration.

## Separate PINN pilot: calibrated_v12_gpu

These pilot fits are retained separately and are not counted as frozen v0 campaign completions. Validation uses supplied physics and observations, without true coefficient selection.

```text
                    variant                                                   key  ic_relative_L2  replay_sigma  bc_rms_over_sensor_sigma  replay_improvement_over_constant_percent  consistency_screen_passed
calibrated_hard_physics_v12 A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05    2.492385e-07      1.764278              1.951347e-15                                 35.270182                      False
```

Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.

Post-fit coefficient scores (answer-key evaluation only; all retained cases/arms scored, no selection):

```text
                                                  key                     variant                        artifact  n_sensors  eta  rel_L2_D  rmse_logD  replay_sigma  sensor_fit_sigma
A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05 calibrated_hard_physics_v12 results\adjoint_nn\metrics.json         50 0.05  0.143579   0.175454      0.900287          0.900287
A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05 calibrated_hard_physics_v12       results\pinn\metrics.json         50 0.05  0.391918   0.477673      1.764278          1.649157
```

## Separate PINN pilot: dense_calibrated_v14_gpu

These pilot fits are retained separately and are not counted as frozen v0 campaign completions. Validation uses supplied physics and observations, without true coefficient selection.

```text
             variant                                                     key  ic_relative_L2  replay_sigma  bc_rms_over_sensor_sigma  replay_improvement_over_constant_percent  consistency_screen_passed
dense_calibrated_v14 A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    2.492385e-07      2.474627              1.925546e-15                                 16.239628                      False
dense_calibrated_v14  A_smooth__Fo0.1__uniform8__n400__nodes_nested__eta0.05    2.492385e-07      2.521991              1.870441e-15                                 12.064835                      False
```

Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.

Post-fit coefficient scores (answer-key evaluation only; all retained cases/arms scored, no selection):

```text
                                                    key              variant                              artifact  n_sensors  eta  rel_L2_D  rmse_logD  replay_sigma  sensor_fit_sigma
A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 dense_calibrated_v14 results\n3969\adjoint_nn\metrics.json       3969 0.05  0.200081   0.193694      0.980659          0.980659
A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 dense_calibrated_v14       results\n3969\pinn\metrics.json       3969 0.05  0.403720   0.488613      2.474627          2.259556
 A_smooth__Fo0.1__uniform8__n400__nodes_nested__eta0.05 dense_calibrated_v14  results\n400\adjoint_nn\metrics.json        400 0.05  0.174056   0.170742      0.944732          0.944732
 A_smooth__Fo0.1__uniform8__n400__nodes_nested__eta0.05 dense_calibrated_v14        results\n400\pinn\metrics.json        400 0.05  0.383100   0.495214      2.521991          2.266499
```

## Separate PINN pilot: dense_resolution_v15_gpu

These pilot fits are retained separately and are not counted as frozen v0 campaign completions. Validation uses supplied physics and observations, without true coefficient selection.

```text
             variant                                                     key  ic_relative_L2  replay_sigma  bc_rms_over_sensor_sigma  replay_improvement_over_constant_percent  consistency_screen_passed
dense_resolution_v15 A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    2.492385e-07      2.036014              2.382916e-15                                 31.085632                      False
dense_resolution_v15 A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    2.492385e-07      2.474627              1.925546e-15                                 16.239628                      False
```

Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.

Post-fit coefficient scores (answer-key evaluation only; all retained cases/arms scored, no selection):

```text
                                                    key              variant                                        artifact  n_sensors  eta  rel_L2_D  rmse_logD  replay_sigma  sensor_fit_sigma
A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 dense_resolution_v15 results\adjoint_control\adjoint_nn\metrics.json       3969 0.05  0.200081   0.193694      0.980659          0.980659
A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 dense_resolution_v15              results\pde16384\pinn\metrics.json       3969 0.05  0.357423   0.413832      2.036014          2.024558
A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 dense_resolution_v15       results\pde4096_control\pinn\metrics.json       3969 0.05  0.403720   0.488613      2.474627          2.259556
```

## Separate PINN pilot: density_v10_gpu

These pilot fits are retained separately and are not counted as frozen v0 campaign completions. Validation uses supplied physics and observations, without true coefficient selection.

```text
              variant                                                     key  ic_relative_L2  replay_sigma  bc_rms_over_sensor_sigma  replay_improvement_over_constant_percent  consistency_screen_passed
gaussian_ic_bc_locald A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    2.492385e-07      3.406555              1.987155e-15                                -15.303977                      False
gaussian_ic_bc_locald   A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05    2.492385e-07      2.406042              2.106367e-15                                 11.724418                      False
```

Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.

Post-fit coefficient scores (answer-key evaluation only; all retained cases/arms scored, no selection):

```text
                                                    key               variant                                                                                                                                       artifact  n_sensors  eta  rel_L2_D  rmse_logD  replay_sigma  sensor_fit_sigma
A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 gaussian_ic_bc_locald results\A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05\A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05\pinn__s42\metrics.json       3969 0.05  0.325202   0.414999      3.406555          1.155646
  A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05 gaussian_ic_bc_locald     results\A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05\A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05\pinn__s42\metrics.json         50 0.05  0.305009   0.406850      2.406042          0.758226
```

## Separate PINN pilot: dual_v9_gpu

These pilot fits are retained separately and are not counted as frozen v0 campaign completions. Validation uses supplied physics and observations, without true coefficient selection.

```text
  variant                                             key  ic_relative_L2  replay_sigma  bc_rms_over_sensor_sigma  replay_improvement_over_constant_percent  consistency_screen_passed
augmented A_smooth__Fo0.1__uniform8__n50__random__eta0.05    2.492385e-07      2.592037              1.927765e-15                                  7.649655                      False
  control A_smooth__Fo0.1__uniform8__n50__random__eta0.05    2.492385e-07      2.669320              1.998583e-15                                  4.896198                      False
```

Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.

This continuation starts from a prior pilot checkpoint: cumulative Adam budget 15000, cumulative L-BFGS budget 3000. It is not an equal-budget fresh fit.

Post-fit coefficient scores (answer-key evaluation only; all retained cases/arms scored, no selection):

```text
                                            key   variant               artifact  n_sensors  eta  rel_L2_D  rmse_logD  replay_sigma  sensor_fit_sigma
A_smooth__Fo0.1__uniform8__n50__random__eta0.05 augmented augmented\metrics.json         50 0.05  0.324239   0.436686      2.592037          0.826867
A_smooth__Fo0.1__uniform8__n50__random__eta0.05   control   control\metrics.json         50 0.05  0.321229   0.423734      2.669320          0.648879
```

## Separate PINN pilot: ic_adaptive_v4_gpu

These pilot fits are retained separately and are not counted as frozen v0 campaign completions. Validation uses supplied physics and observations, without true coefficient selection.

```text
                                            key  ic_relative_L2  replay_sigma  bc_rms_over_sensor_sigma  replay_improvement_over_constant_percent
A_smooth__Fo0.1__uniform8__n50__random__eta0.05    2.492385e-07      2.876352              1.863489e-15                                 -2.480054
```

Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.

## Separate PINN pilot: ic_bc_v3_gpu

These pilot fits are retained separately and are not counted as frozen v0 campaign completions. Validation uses supplied physics and observations, without true coefficient selection.

```text
                                            key  ic_relative_L2  replay_sigma  bc_rms_over_sensor_sigma  replay_improvement_over_constant_percent
A_smooth__Fo0.1__uniform8__n50__random__eta0.05    2.492385e-07      2.807432              1.863489e-15                                 -0.024545
```

Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.

## Separate PINN pilot: ic_constraints_v2_gpu

These pilot fits are retained separately and are not counted as frozen v0 campaign completions. Validation uses supplied physics and observations, without true coefficient selection.

```text
                                                   key  ic_relative_L2  replay_sigma  bc_rms_over_sensor_sigma  replay_improvement_over_constant_percent
       A_smooth__Fo0.1__uniform8__n50__random__eta0.05    9.188814e-07      2.747573                  2.824993                                  2.108135
B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05    9.188814e-07      2.830128                  2.020984                                  5.962789
          E_grf__Fo0.1__uniform8__n50__random__eta0.05    9.188814e-07      1.643996                  1.703804                                  2.448595
```

Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.

## Separate PINN pilot: ic_convergence_v5_gpu

These pilot fits are retained separately and are not counted as frozen v0 campaign completions. Validation uses supplied physics and observations, without true coefficient selection.

```text
                                            key  ic_relative_L2  replay_sigma  bc_rms_over_sensor_sigma  replay_improvement_over_constant_percent
A_smooth__Fo0.1__uniform8__n50__random__eta0.05    2.492385e-07        2.8962              1.863489e-15                                 -3.187216
```

Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.

This continuation starts from a prior pilot checkpoint: cumulative Adam budget 15000, cumulative L-BFGS budget 4500. It is not an equal-budget fresh fit.

## Separate PINN pilot: ic_locald_v6_gpu

These pilot fits are retained separately and are not counted as frozen v0 campaign completions. Validation uses supplied physics and observations, without true coefficient selection.

```text
                                            key  ic_relative_L2  replay_sigma  bc_rms_over_sensor_sigma  replay_improvement_over_constant_percent
A_smooth__Fo0.1__uniform8__n50__random__eta0.05    2.492385e-07      2.605599              1.987394e-15                                  7.166463
```

Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.

## Separate PINN pilot: ic_v1_gpu

These pilot fits are retained separately and are not counted as frozen v0 campaign completions. Validation uses supplied physics and observations, without true coefficient selection.

```text
                                                   key  ic_relative_L2  replay_sigma  bc_rms_over_sensor_sigma  replay_improvement_over_constant_percent
       A_smooth__Fo0.1__uniform8__n50__random__eta0.05    9.188814e-07      2.791922                  2.909967                                  0.528067
B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05    9.188814e-07      3.007244                  2.026962                                  0.077726
          E_grf__Fo0.1__uniform8__n50__random__eta0.05    9.188814e-07      1.586714                  2.311187                                  5.847621
```

Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.

## Separate PINN pilot: optimizer_v16_gpu

These pilot fits are retained separately and are not counted as frozen v0 campaign completions. Validation uses supplied physics and observations, without true coefficient selection.

```text
            variant                                                     key  ic_relative_L2  replay_sigma  bc_rms_over_sensor_sigma  replay_improvement_over_constant_percent  consistency_screen_passed
dense_optimizer_v16 A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    2.492385e-07      2.944905              1.878998e-15                                  0.321794                      False
dense_optimizer_v16 A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    2.492385e-07      2.036014              2.382916e-15                                 31.085632                      False
```

Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.

Post-fit coefficient scores (answer-key evaluation only; all retained cases/arms scored, no selection):

```text
                                                    key             variant                                       artifact  n_sensors  eta  rel_L2_D  rmse_logD  replay_sigma  sensor_fit_sigma
A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 dense_optimizer_v16 results\lbfgsb_adjoint\adjoint_nn\metrics.json       3969 0.05  0.052329   0.053351      0.994934          0.994934
A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 dense_optimizer_v16          results\lbfgsb_pinn\pinn\metrics.json       3969 0.05  0.378740   0.510402      2.944905          2.927266
A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 dense_optimizer_v16    results\ssb_adjoint\adjoint_nn\metrics.json       3969 0.05  0.200081   0.193694      0.980659          0.980659
A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 dense_optimizer_v16             results\ssb_pinn\pinn\metrics.json       3969 0.05  0.357423   0.413832      2.036014          2.024558
```

## Separate PINN pilot: sampling_v13_gpu

These pilot fits are retained separately and are not counted as frozen v0 campaign completions. Validation uses supplied physics and observations, without true coefficient selection.

```text
            variant                                                   key  ic_relative_L2  replay_sigma  bc_rms_over_sensor_sigma  replay_improvement_over_constant_percent  consistency_screen_passed
sampling_random_v13 A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05    2.492385e-07      1.764278              1.951347e-15                                 35.270182                      False
 sampling_sobol_v13 A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05    2.492385e-07      2.059836              1.837895e-15                                 24.426391                      False
```

Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.

Post-fit coefficient scores (answer-key evaluation only; all retained cases/arms scored, no selection):

```text
                                                  key             variant                                        artifact  n_sensors  eta  rel_L2_D  rmse_logD  replay_sigma  sensor_fit_sigma
A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05 sampling_random_v13 results\adjoint_control\adjoint_nn\metrics.json         50 0.05  0.143579   0.175454      0.900287          0.900287
A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05 sampling_random_v13        results\random_control\pinn\metrics.json         50 0.05  0.391918   0.477673      1.764278          1.649157
A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05  sampling_sobol_v13                 results\sobol\pinn\metrics.json         50 0.05  0.440032   0.522415      2.059836          1.979246
```

## Separate PINN pilot: temporal_v18_gpu

These pilot fits are retained separately and are not counted as frozen v0 campaign completions. Validation uses supplied physics and observations, without true coefficient selection.

```text
             variant                                                            key  ic_relative_L2  replay_sigma  bc_rms_over_sensor_sigma  replay_improvement_over_constant_percent  consistency_screen_passed
temporal_density_v18        A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    2.492385e-07      1.325529              2.644715e-15                                 55.133915                      False
temporal_density_v18        A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    2.492385e-07      1.099161              2.647046e-15                                 62.795954                       True
temporal_density_v18 B_gaussian_bump__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    2.492385e-07      1.465603              1.893976e-15                                 53.316385                      False
temporal_density_v18 B_gaussian_bump__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    2.492385e-07      1.111190              1.850894e-15                                 64.605447                      False
temporal_density_v18       D_layered__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    2.492385e-07      2.923441              1.049701e-15                                 58.499784                      False
temporal_density_v18       D_layered__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    2.492385e-07      2.740053              1.058313e-15                                 61.103103                      False
temporal_density_v18           E_grf__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    2.492385e-07      1.266746              1.984396e-15                                 24.535164                      False
temporal_density_v18           E_grf__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    2.492385e-07      1.142029              1.936934e-15                                 31.965018                      False
temporal_density_v18       F_defects__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    2.492385e-07      1.401187              1.730296e-15                                  5.123174                      False
temporal_density_v18       F_defects__Fo0.1__uniform8__n3969__nodes_nested__eta0.05    2.492385e-07      1.219347              1.657399e-15                                 17.435878                      False
```

Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.

## Separate PINN pilot: zhang_v11_gpu

These pilot fits are retained separately and are not counted as frozen v0 campaign completions. Validation uses supplied physics and observations, without true coefficient selection.

```text
               variant                                                   key  ic_relative_L2  replay_sigma  bc_rms_over_sensor_sigma  replay_improvement_over_constant_percent  consistency_screen_passed
zhang_v2_heat_transfer A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05        0.660376    106.102287                 15.969952                              -3792.800629                      False
```

Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.

Post-fit coefficient scores (answer-key evaluation only; all retained cases/arms scored, no selection):

```text
                                                  key                variant                                                                             artifact  n_sensors  eta  rel_L2_D  rmse_logD  replay_sigma  sensor_fit_sigma
A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05 zhang_v2_heat_transfer results\A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05\pinn__s42\metrics.json         50 0.05       1.0 117.322433    106.102287          1.387451
```

## PINN optimization diagnostic

The v3 temperature network was frozen and only its coefficient network optimized against PDE residual plus the same spatial regularizer, without reading true D. This is a separate block diagnostic, not a completed campaign fit.
Common replay changed from 2.807 to 2.563 sigma. Independent PDE RMS changed from 0.0402 to 0.0394; neither resolves the physical-consistency failure.

An independent 16,384-point probe using the original mixed sampling distribution found the largest 1% of residuals carry 64.0% of residual energy. This motivates investigating collocation coverage; it does not prove adaptive sampling will fix recovery. No true D is used in this diagnosis.

## Infrastructure failure and recovery

The C: drive filled during an edit and grid visualization. The root source was restored from its retained validated copy; frozen campaign source was unaffected. Integrity checks passed for 293 completed core artifacts (JSON, array finiteness and NPZ/model CRCs). One missing grid PNG was rebuilt from the saved estimate without refitting or changing coefficient/metrics hashes. Its original failure log and state remain under failed_attempts.
Only SHA256-verified duplicate inputs and a reproducible input-only upload container were reclaimed; canonical data, all scientific source versions and trained results were retained. Input folders also received lossless NTFS compression. The grid coordinator was restarted after confirming no surviving grid process; new coordinators stop dispatch below 128 MiB free disk space. This is an infrastructure incident, not a scientific method failure. See diagnostics/disk_incident_recovery.json and the input-deduplication manifests.

## Dense adjoint convergence continuation (separate budget)

A_smooth at 3969 sensors and 1% noise was continued from its retained neural-adjoint checkpoint at unchanged alpha 0.100409. Replay changed from 1.2371 to 1.1241 sensor sigma over 750 additional iterations; optimizer status: STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
This diagnostic was preselected from sensor density, noise and iteration caps, without coefficient truth. It warm-starts the selected-alpha checkpoint rather than repeating the discrepancy path. Additional compute is separate from the frozen campaign; the original fit is preserved and no coefficient-error result is used for selection.

## Fixed-coefficient forward-state diagnostic: fixed_d_forward_v7_gpu

The coefficient is the retained A neural-adjoint fit, held fixed bitwise. Only the fresh temperature network is optimized against the PDE; sensors and CN are evaluation only, and no answer key is supplied.
State-versus-CN sensor RMS is 0.7229 noise sigma; state-versus-observations 1.2550, fixed-coefficient CN-versus-observations 0.9420. Checkpoint metrics independently reproduced: True; coefficient unchanged: True.
This forward-only diagnostic uses 15000 Adam and 750 requested L-BFGS steps in 1 round(s) on Tesla T4. The initial coefficient is spatially variable, unlike fresh v6; it is not an equal-initial-function inverse comparison. Refreshed rounds change both collocation and optimizer history. Success or failure does not by itself reopen the inverse gate or prove state nonexpressivity.

## Fixed-coefficient forward-state diagnostic: fixed_d_refresh_v8_gpu

The coefficient is the retained A neural-adjoint fit, held fixed bitwise. Only the fresh temperature network is optimized against the PDE; sensors and CN are evaluation only, and no answer key is supplied.
State-versus-CN sensor RMS is 0.7737 noise sigma; state-versus-observations 1.2593, fixed-coefficient CN-versus-observations 0.9420. Checkpoint metrics independently reproduced: True; coefficient unchanged: True.
This forward-only diagnostic uses 15000 Adam and 750 requested L-BFGS steps in 5 round(s) on Tesla T4. The initial coefficient is spatially variable, unlike fresh v6; it is not an equal-initial-function inverse comparison. Refreshed rounds change both collocation and optimizer history. Success or failure does not by itself reopen the inverse gate or prove state nonexpressivity.

### Controlled forward collocation-refresh comparison

Refreshing collocation and optimizer history reduced mixed held-out PDE RMS by 34.8%, while state-versus-CN sensor RMS changed from 0.7229 to 0.7737 sigma. Neither temperature fit is an inverse coefficient recovery result. The two retained GPU Adam histories are identical; each used750 L-BFGS iterations and819 function evaluations. Times were 1549.2 and 1586.8 seconds on Tesla T4. Lower held-out PDE RMS alone does not guarantee lower state error at sensor locations; the inverse gate remains held.

## Retained-checkpoint loss and gradient reconstruction

```text
                pilot                                             key term     raw_loss  weight  weighted_loss  temperature_gradient_norm  coefficient_gradient_norm  gradient_norm
ic_convergence_v5_gpu A_smooth__Fo0.1__uniform8__n50__random__eta0.05 data 2.150806e+02     1.0   2.150806e+02               28823.832542               0.000000e+00   2.882383e+04
ic_convergence_v5_gpu A_smooth__Fo0.1__uniform8__n50__random__eta0.05  pde 1.396468e-01  1000.0   1.396468e+02               28489.663411               2.310853e+02   2.849060e+04
ic_convergence_v5_gpu A_smooth__Fo0.1__uniform8__n50__random__eta0.05   ic 1.626124e-13  1000.0   1.626124e-10                   0.000000               0.000000e+00   0.000000e+00
ic_convergence_v5_gpu A_smooth__Fo0.1__uniform8__n50__random__eta0.05   bc 6.164964e-34  1000.0   6.164964e-31                   0.000000               0.000000e+00   0.000000e+00
ic_convergence_v5_gpu A_smooth__Fo0.1__uniform8__n50__random__eta0.05  reg 4.297481e-02     1.0   4.297481e-02                   0.000000               1.627125e+00   1.627125e+00
     ic_locald_v6_gpu A_smooth__Fo0.1__uniform8__n50__random__eta0.05 data 1.136030e+02     1.0   1.136030e+02                6562.402780               1.181981e+03   6.667999e+03
     ic_locald_v6_gpu A_smooth__Fo0.1__uniform8__n50__random__eta0.05  pde 8.601797e-02  1000.0   8.601797e+01                6452.371498               1.166072e+03   6.556891e+03
     ic_locald_v6_gpu A_smooth__Fo0.1__uniform8__n50__random__eta0.05   ic 1.626124e-13  1000.0   1.626124e-10                   0.000000               0.000000e+00   0.000000e+00
     ic_locald_v6_gpu A_smooth__Fo0.1__uniform8__n50__random__eta0.05   bc 1.018410e-33  1000.0   1.018410e-30                   0.000000               1.086148e-29   1.086148e-29
     ic_locald_v6_gpu A_smooth__Fo0.1__uniform8__n50__random__eta0.05  reg 1.169693e+00     1.0   1.169693e+00                   0.000000               1.078573e+01   1.078573e+01
```

The v5/v6 fixed-batch objectives reproduce saved losses and gradients without refitting. V6 data/PDE full-gradient cosine is about -0.991, with comparable component norms. Such opposition is normal at a penalized compromise and depends on parameterization. It does not establish a correct penalty or prove the cause of inverse failure. Component and geometry files are retained per pilot; no true coefficient is used.

## Independent collocation and optimization checks

```text
                   pilot                                                     key  heldout_points  heldout_rms  heldout_max  top_one_percent_energy  replay_sigma  sensor_fit_sigma  training_points  training_rms  heldout_to_training_rms                              artifact             arm              variant
            ic_bc_v3_gpu         A_smooth__Fo0.1__uniform8__n50__random__eta0.05           16384     0.035080     0.991078                0.528074      2.807432          0.833626              NaN           NaN                      NaN                                   NaN             NaN                  NaN
      ic_adaptive_v4_gpu         A_smooth__Fo0.1__uniform8__n50__random__eta0.05           16384     0.032646     0.795176                0.419778      2.876352          1.082933              NaN           NaN                      NaN                                   NaN             NaN                  NaN
   ic_convergence_v5_gpu         A_smooth__Fo0.1__uniform8__n50__random__eta0.05           16384     0.034756     0.741118                0.464767      2.896200          1.037016           2048.0      0.026321                 1.320476                                   NaN             NaN                  NaN
        ic_locald_v6_gpu         A_smooth__Fo0.1__uniform8__n50__random__eta0.05           16384     0.031277     0.755305                0.463322      2.605599          0.753668           2048.0      0.020658                 1.514052                                   NaN             NaN                  NaN
      calibrated_v12_gpu   A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05           16384     0.009356     0.433078                0.892500      1.764278          1.649157           4096.0      0.000736                12.710444                                   NaN             NaN                  NaN
        sampling_v13_gpu   A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05           16384     0.009356     0.433078                0.892500      1.764278          1.649157           4096.0      0.000736                12.710444  results\random_control\pinn\model.pt  random_control  sampling_random_v13
        sampling_v13_gpu   A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05           16384     0.006262     0.235894                0.919242      2.059836          1.979246           4096.0      0.000614                10.198246           results\sobol\pinn\model.pt           sobol   sampling_sobol_v13
dense_calibrated_v14_gpu A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05           16384     0.016115     0.548212                0.945305      2.474627          2.259556           4096.0      0.000695                23.196502           results\n3969\pinn\model.pt             NaN dense_calibrated_v14
dense_calibrated_v14_gpu  A_smooth__Fo0.1__uniform8__n400__nodes_nested__eta0.05           16384     0.016296     0.868753                0.981397      2.521991          2.266499           4096.0      0.000675                24.137999            results\n400\pinn\model.pt             NaN dense_calibrated_v14
dense_resolution_v15_gpu A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05           16384     0.001243     0.020731                0.318013      2.036014          2.024558          16384.0      0.000923                 1.346086        results\pde16384\pinn\model.pt        pde16384 dense_resolution_v15
dense_resolution_v15_gpu A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05           16384     0.016115     0.548212                0.945305      2.474627          2.259556           4096.0      0.000695                23.196502 results\pde4096_control\pinn\model.pt pde4096_control dense_resolution_v15
       optimizer_v16_gpu A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05           16384     0.000348     0.003063                0.197863      2.944905          2.927266          16384.0      0.000340                 1.022026     results\lbfgsb_pinn\pinn\model.pt     lbfgsb_pinn  dense_optimizer_v16
       optimizer_v16_gpu A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05           16384     0.001243     0.020731                0.318013      2.036014          2.024558          16384.0      0.000923                 1.346086        results\ssb_pinn\pinn\model.pt        ssb_pinn  dense_optimizer_v16
```

The listed checkpoints use the same independent 16384-point half-log/half-uniform-time probe; retained training-cloud statistics appear where available. Adaptive v5 training changes the sampling measure, whereas v12 uses the same mixed measure as the probe. V6 held-out RMS is about1.51 times training RMS and v12 about12.71 times. Both retained optimization limits and generalization must be considered; these checks do not establish a sole cause of replay failure. No coefficient truth is read and no checkpoint is refitted or selected by this diagnostic.

## Fixed-checkpoint residual integration audit

```text
sampler  replicates  mean_residual_squared  std_of_mean_residual_squared  mean_absolute_relative_difference_from_MC_reference
 random           4               0.000137                      0.000034                                             0.360349
  sobol           4               0.000073                      0.000004                                             0.331154
```

Four predeclared4096-point replicates per sampler estimate the same mixed-time residual integral on the unchanged v12 checkpoint. The65536-point random reference is also noisy, not an exact integral. Lower replicate variability alone does not establish greater accuracy or a better training method; no model is retrained, selected or scored against coefficient truth. All replicates and reference time-bin energies are retained in the pilot quadrature_audit folder.

## Dense PINN fixed-checkpoint quadrature resolution

```text
 points  mean_energy  replicate_std  replicate_cv
   4096     0.000359       0.000083      0.232305
  16384     0.000338       0.000031      0.092973
  65536     0.000321       0.000011      0.033097
```

The retained3969-sensor PINN is evaluated on four predeclared independent random replicates, each with nested4096/16384/65536 prefixes and exactly half uniform/half log-time points. Replicate coefficients of variation are23.23%,9.30%,3.31%; all replicates are retained. These descriptive four-replicate estimates have no exact-integral reference. Reduced integration variability is not a verified training or inverse-accuracy improvement, nor an optimal point-count claim. The checkpoint is unchanged; no coefficient truth, optimizer or checkpoint selection is used.

## Fixed-observation adjoint seed confirmation: adjoint_seed_confirmation (complete)

```text
                                                        completed_seeds  mean_coefficient_error  std_coefficient_error  min_coefficient_error  max_coefficient_error  mean_replay_sigma
key                                                                                                                                                                                    
A_smooth__Fo0.1__uniform8__n50__random__eta0.05                       3                0.107719               0.004793               0.104620               0.113240           0.943219
B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05                3                0.122140               0.006713               0.114905               0.128167           0.995647
F_defects__Fo0.1__uniform8__n50__random__eta0.05                      3                0.271638               0.020780               0.248639               0.289061           0.989346
```

These initialization seeds share exactly the same observations and frozen neural-adjoint settings. They quantify optimization/representation variability conditional on one observation realization, not uncertainty across new sensor locations or noise draws. A single completed seed has no estimated standard deviation. These additional fits are retained separately from the 1510-fit broad plan.
A_smooth__Fo0.1__uniform8__n50__random__eta0.05: neural coefficient error 0.1046--0.1132, fixed grid error 0.1026; neural lower in 0/3 seeds. Replay range 0.942--0.944 sigma. All seeds are retained; none is selected as the preferred fit. Timing differences also reflect worker contention, so these seed timings are not a speed comparison.
B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05: neural coefficient error 0.1149--0.1282, fixed grid error 0.2294; neural lower in 3/3 seeds. Replay range 0.994--0.997 sigma. All seeds are retained; none is selected as the preferred fit. Timing differences also reflect worker contention, so these seed timings are not a speed comparison.
F_defects__Fo0.1__uniform8__n50__random__eta0.05: neural coefficient error 0.2486--0.2891, fixed grid error 0.1821; neural lower in 0/3 seeds. Replay range 0.985--0.995 sigma. All seeds are retained; none is selected as the preferred fit. Timing differences also reflect worker contention, so these seed timings are not a speed comparison.

## Fixed-observation adjoint seed confirmation: adjoint_density_seed_confirmation (complete)

```text
                                                                completed_seeds  mean_coefficient_error  std_coefficient_error  min_coefficient_error  max_coefficient_error  mean_replay_sigma
key                                                                                                                                                                                            
B_gaussian_bump__Fo0.1__uniform8__n3969__nodes_nested__eta0.05                3                0.074122               0.039492               0.039254               0.117008           1.012336
D_layered__Fo0.1__uniform8__n3969__nodes_nested__eta0.01                      3                0.092141               0.006480               0.086648               0.099287           1.075235
E_grf__Fo0.1__uniform8__n3969__nodes_nested__eta0.05                          3                0.145463               0.023153               0.129199               0.171970           1.026008
```

These initialization seeds share exactly the same observations and frozen neural-adjoint settings. They quantify optimization/representation variability conditional on one observation realization, not uncertainty across new sensor locations or noise draws. A single completed seed has no estimated standard deviation. These additional fits are retained separately from the 1510-fit broad plan.
B_gaussian_bump__Fo0.1__uniform8__n3969__nodes_nested__eta0.05: neural coefficient error 0.0393--0.1170, fixed grid error 0.0899; neural lower in 2/3 seeds. Replay range 1.000--1.021 sigma. All seeds are retained; none is selected as the preferred fit. Timing differences also reflect worker contention, so these seed timings are not a speed comparison.
D_layered__Fo0.1__uniform8__n3969__nodes_nested__eta0.01: neural coefficient error 0.0866--0.0993, fixed grid error 0.1530; neural lower in 3/3 seeds. Replay range 1.071--1.082 sigma. All seeds are retained; none is selected as the preferred fit. Timing differences also reflect worker contention, so these seed timings are not a speed comparison.
E_grf__Fo0.1__uniform8__n3969__nodes_nested__eta0.05: neural coefficient error 0.1292--0.1720, fixed grid error 0.0509; neural lower in 0/3 seeds. Replay range 1.021--1.030 sigma. All seeds are retained; none is selected as the preferred fit. Timing differences also reflect worker contention, so these seed timings are not a speed comparison.

## Matched paper-transfer neural adjoint versus neural PINN

```text
              method                                                   key  rel_L2_D  replay_sigma  sensor_fit_sigma   status     wall_s  adam_completed  quasi_newton_iterations  function_evaluations                                       hardware
neural_adjoint_zhang A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05  0.381248      2.725313          2.725313 complete 3061.35900            1000                     4000                  9437 Local CPU1thread, concurrent broad CPU workers
   neural_pinn_zhang A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05  1.000000    106.102287          1.387451 complete 2403.97741            1000                     4000                  6117       Kaggle CUDAresidual/CPUcurvature2threads
```

Separate predeclared research protocol. Each comparison shares coefficient network, initial weights, mean data objective and regularization as declared in its versioned protocol; v16 crosses both methods with two optimizer implementations under common iteration limits. See the versioned protocol for cases, coefficient bounds and state construction. Different state enforcement and finite-dimensional objectives remain intrinsic. Check coverage, fit status, optimizer caps, independent constraints and replay before interpreting accuracy. Record actual hardware, resource use and CPU contention before comparing timings. These research fits are outside the frozen broad count.

## Matched calibrated hard-physics neural adjoint versus neural PINN

```text
    method                                                   key  rel_L2_D  replay_sigma  sensor_fit_sigma   status      wall_s  adam_completed  quasi_newton_iterations  function_evaluations        gamma  parameter_count
adjoint_nn A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05  0.143579      0.900287          0.900287 complete 2618.672250            1000                     4000                  7911 7.117245e-09             2241
      pinn A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05  0.391918      1.764278          1.649157 complete 2957.520603            1000                     4000                  8431 7.117245e-09             4514
```

Separate predeclared research protocol. Each comparison shares coefficient network, initial weights, mean data objective and regularization as declared in its versioned protocol; v16 crosses both methods with two optimizer implementations under common iteration limits. See the versioned protocol for cases, coefficient bounds and state construction. Different state enforcement and finite-dimensional objectives remain intrinsic. Check coverage, fit status, optimizer caps, independent constraints and replay before interpreting accuracy. Record actual hardware, resource use and CPU contention before comparing timings. These research fits are outside the frozen broad count.

## Matched controlled collocation sampling neural adjoint versus neural PINN

```text
            arm     method                                                   key  rel_L2_D  replay_sigma  sensor_fit_sigma   status      wall_s  adam_completed  quasi_newton_iterations  capped_rounds  function_evaluations        gamma
adjoint_control adjoint_nn A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05  0.143579      0.900287          0.900287 complete 2401.405901            1000                     4000             10                  7911 7.117245e-09
 random_control       pinn A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05  0.391918      1.764278          1.649157 complete 2884.458707            1000                     4000             10                  8431 7.117245e-09
          sobol       pinn A_smooth__Fo0.1__uniform8__n50__nodes_nested__eta0.05  0.440032      2.059836          1.979246 complete 2911.539514            1000                     4000             10                  9047 7.117245e-09
```

Separate predeclared research protocol. Each comparison shares coefficient network, initial weights, mean data objective and regularization as declared in its versioned protocol; v16 crosses both methods with two optimizer implementations under common iteration limits. See the versioned protocol for cases, coefficient bounds and state construction. Different state enforcement and finite-dimensional objectives remain intrinsic. Check coverage, fit status, optimizer caps, independent constraints and replay before interpreting accuracy. Record actual hardware, resource use and CPU contention before comparing timings. These research fits are outside the frozen broad count.

## Matched higher-density calibrated neural adjoint versus neural PINN

```text
                                                    key  n_sensors     method  rel_L2_D  replay_sigma  sensor_fit_sigma   status      wall_s  adam_completed  quasi_newton_iterations  capped_rounds  function_evaluations        gamma
A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05       3969 adjoint_nn  0.200081      0.980659          0.980659 complete 3077.127232            1000                     4000             10                  8085 8.966043e-11
A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05       3969       pinn  0.403720      2.474627          2.259556 complete 3091.855683            1000                     4000             10                  8438 8.966043e-11
 A_smooth__Fo0.1__uniform8__n400__nodes_nested__eta0.05        400 adjoint_nn  0.174056      0.944732          0.944732 complete 2600.192816            1000                     4000             10                  7786 8.896556e-10
 A_smooth__Fo0.1__uniform8__n400__nodes_nested__eta0.05        400       pinn  0.383100      2.521991          2.266499 complete 2954.444912            1000                     4000             10                  8439 8.896556e-10
```

Separate predeclared research protocol. Each comparison shares coefficient network, initial weights, mean data objective and regularization as declared in its versioned protocol; v16 crosses both methods with two optimizer implementations under common iteration limits. See the versioned protocol for cases, coefficient bounds and state construction. Different state enforcement and finite-dimensional objectives remain intrinsic. Check coverage, fit status, optimizer caps, independent constraints and replay before interpreting accuracy. Record actual hardware, resource use and CPU contention before comparing timings. These research fits are outside the frozen broad count.
All predeclared paired cases covered: True. Missing or unpaired cases are not method losses.

## Matched dense collocation-count control neural adjoint versus neural PINN

```text
            arm     method                                                     key  n_sensors  n_pde  rel_L2_D  replay_sigma  sensor_fit_sigma   status      wall_s  adam_completed  quasi_newton_iterations  capped_rounds  function_evaluations        gamma
adjoint_control adjoint_nn A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05       3969   4096  0.200081      0.980659          0.980659 complete 2924.693599            1000                     4000             10                  8085 8.966043e-11
       pde16384       pinn A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05       3969  16384  0.357423      2.036014          2.024558 complete 3334.071281            1000                     4000             10                  8663 8.966043e-11
pde4096_control       pinn A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05       3969   4096  0.403720      2.474627          2.259556 complete 2942.799100            1000                     4000             10                  8438 8.966043e-11
```

Separate predeclared research protocol. Each comparison shares coefficient network, initial weights, mean data objective and regularization as declared in its versioned protocol; v16 crosses both methods with two optimizer implementations under common iteration limits. See the versioned protocol for cases, coefficient bounds and state construction. Different state enforcement and finite-dimensional objectives remain intrinsic. Check coverage, fit status, optimizer caps, independent constraints and replay before interpreting accuracy. Record actual hardware, resource use and CPU contention before comparing timings. These research fits are outside the frozen broad count.
All predeclared paired cases covered: True. Missing or unpaired cases are not method losses.

## Matched dense method-by-optimizer control neural adjoint versus neural PINN

```text
           arm optimizer     method                                                     key  n_sensors  n_pde  rel_L2_D  replay_sigma  sensor_fit_sigma   status      wall_s  adam_completed  quasi_newton_iterations  capped_rounds  optimizer_failures  function_evaluations        gamma
lbfgsb_adjoint    lbfgsb adjoint_nn A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05       3969   4096  0.052329      0.994934          0.994934 complete 1745.627675            1000                     4000             10                   0                  5355 8.966043e-11
   lbfgsb_pinn    lbfgsb       pinn A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05       3969  16384  0.378740      2.944905          2.927266 complete  855.703140            1000                     4000             10                   0                  5328 8.966043e-11
   ssb_adjoint       ssb adjoint_nn A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05       3969   4096  0.200081      0.980659          0.980659 complete 2988.756262            1000                     4000             10                   0                  8085 8.966043e-11
      ssb_pinn       ssb       pinn A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05       3969  16384  0.357423      2.036014          2.024558 complete 3341.094305            1000                     4000             10                   0                  8663 8.966043e-11
```

Separate predeclared research protocol. Each comparison shares coefficient network, initial weights, mean data objective and regularization as declared in its versioned protocol; v16 crosses both methods with two optimizer implementations under common iteration limits. See the versioned protocol for cases, coefficient bounds and state construction. Different state enforcement and finite-dimensional objectives remain intrinsic. Check coverage, fit status, optimizer caps, independent constraints and replay before interpreting accuracy. Record actual hardware, resource use and CPU contention before comparing timings. These research fits are outside the frozen broad count.
All predeclared paired cases covered: True. Missing or unpaired cases are not method losses.

## Matched five dense temporal-consistency cases neural adjoint versus neural PINN

```text
                                                    key             arm     method  rel_L2_D  replay_sigma  sensor_fit_sigma   status  adam_completed  quasi_newton_iterations  capped_rounds  function_evaluations      wall_s  n_sensors  time_step_tau
A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05      adjoint_nn adjoint_nn  0.200013      0.981319          0.981319 complete            1000                     4000             10                  7928 3680.781000       3969           0.01
A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 scaled_pde_pinn       pinn  0.240500      1.325529          0.980208 complete            1000                     4000             10                  7741 3009.191103       3969           0.01
A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05   temporal_pinn       pinn  0.172321      1.099161          0.984240 complete            1000                     4000             10                  8395 3171.614778       3969           0.01
   E_grf__Fo0.1__uniform8__n3969__nodes_nested__eta0.05      adjoint_nn adjoint_nn  0.199186      0.985573          0.985573 complete            1000                     4000             10                  8541 3797.953000       3969           0.01
   E_grf__Fo0.1__uniform8__n3969__nodes_nested__eta0.05 scaled_pde_pinn       pinn  0.213647      1.266746          0.987654 complete            1000                     4000             10                  7773 2997.297611       3969           0.01
   E_grf__Fo0.1__uniform8__n3969__nodes_nested__eta0.05   temporal_pinn       pinn  0.205948      1.142029          0.988206 complete            1000                     4000             10                  8258 3178.395484       3969           0.01
```

Separate predeclared research protocol. Each comparison shares coefficient network, initial weights, mean data objective and regularization as declared in its versioned protocol; v16 crosses both methods with two optimizer implementations under common iteration limits. See the versioned protocol for cases, coefficient bounds and state construction. Different state enforcement and finite-dimensional objectives remain intrinsic. Check coverage, fit status, optimizer caps, independent constraints and replay before interpreting accuracy. Record actual hardware, resource use and CPU contention before comparing timings. These research fits are outside the frozen broad count.
All predeclared paired cases covered: False. Missing or unpaired cases are not method losses.