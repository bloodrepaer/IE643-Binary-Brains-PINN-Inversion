# v22: 1,000-sensor completed comparison

Both fits completed1,000Adam+4,000SSB with all10rounds capped and no budget timeout. Independent ICBC/continuousPDE/commonCN then own BOTH-arm checkpoint/bounds/sharedSHA/gamma/count/timeformula/commonBE+CN validation passed before coefficient truth. Complete pair coverage.

| Metric | Temporal PINN | BE neural adjoint |
|---|---:|---:|
| Diffusion relative L2 error |35.3716%|21.4547%|
| Common CN replay / sigma |1.923599|1.075980|
| Common BE replay / sigma |2.005177|0.967260|
| Model sensor fit / sigma |0.936896|0.967260|
| Function evaluations |8,008|7,917|
| Training wall seconds |3,302.714|1,791.256|

Adjoint has lower coefficient error and passes both predeclared1.1sigma replay screens. PINN fails both, despite smaller temperature-network sensorfit error. This is a negative temporal-PINN result at1,000sites, not a missing/partialfit comparison. Both initial/boundary construction and checkpoint/protocol checks pass; these checks do not imply the recovered coefficient has valid replay or zero continuousPDE error.

Hardware: separate freeKaggle GPU PINN (CUDAdata/residual+CPUcurvature) and CPU BEadjoint, bothCPU2threads. Timing descriptive only; no equal-host speed comparison. Both share exactRK4-generated observations/noise, fixedv21initial, coefficientnetwork/bounds/fullmeanobjective/regularizer/alpha1gamma3.5586223384e-10,seed42/iterationlimits/BEh.01. SoftADversusFDhardstate remain different.

Against frozen10,000sensor results (PINN6.7657%,BE26.3639%), this supplies a second density point where the ordering changes. It does not isolate density causally: location coverage and density-dependentgamma also change. Sharedinitial originally estimated from parentobservations is disclosed. Onefield/fixedrealization/seed and cappedoptimization do not establish population robustness or convergence. The user-authorized3x3noise/density test is queued next; all results and failures retained.

Independent PINN supplied-physics record (before coefficient truth):

```json
{
  "updated": "2026-10-21T17:24:07.587673+00:00",
  "validation_seed": 20261019,
  "selection_uses_true_D": false,
  "results": [
    {
      "key": "A_smooth__Fo0.1__uniform8__n1000__fine_nested__eta0.05",
      "artifact": "results\\A_smooth__Fo0.1__uniform8__n1000__fine_nested__eta0.05\\temporal_pinn\\pinn\\metrics.json",
      "variant": "dense1000_matched_temporal_v22",
      "ic_relative_L2": 2.492385217377766e-07,
      "replay_sigma": 1.92359943329362,
      "replay_metric_difference": 6.217248937900877e-15,
      "sensor_fit_sigma": 0.9368962988902192,
      "D_spatial_std": 0.12704526697774385,
      "D_mean": 0.2196514938339337,
      "D_fitted_constant": 0.2658031041258034,
      "constant_replay_sigma": 2.9325327530537377,
      "training_seconds": 3302.713684478,
      "bc_rms": 3.130550696332088e-18,
      "bc_rms_over_sensor_sigma": 2.634184911427935e-15,
      "pinn_v0_replay_sigma": null,
      "adjoint_nn_v0_replay_sigma": null,
      "replay_improvement_over_constant_percent": 34.404844028067004,
      "consistency_screen_passed": false
    }
  ],
  "time_bins": [
    {
      "key": "A_smooth__Fo0.1__uniform8__n1000__fine_nested__eta0.05",
      "artifact": "results\\A_smooth__Fo0.1__uniform8__n1000__fine_nested__eta0.05\\temporal_pinn\\pinn\\metrics.json",
      "variant": "dense1000_matched_temporal_v22",
      "tau_low": 0,
      "tau_high": 0.001,
      "pde_rms": 0.6416416219394853,
      "pde_max_abs": 6.419396696621757
    },
    {
      "key": "A_smooth__Fo0.1__uniform8__n1000__fine_nested__eta0.05",
      "artifact": "results\\A_smooth__Fo0.1__uniform8__n1000__fine_nested__eta0.05\\temporal_pinn\\pinn\\metrics.json",
      "variant": "dense1000_matched_temporal_v22",
      "tau_low": 0.001,
      "tau_high": 0.01,
      "pde_rms": 0.48292536094722766,
      "pde_max_abs": 3.397522606014541
    },
    {
      "key": "A_smooth__Fo0.1__uniform8__n1000__fine_nested__eta0.05",
      "artifact": "results\\A_smooth__Fo0.1__uniform8__n1000__fine_nested__eta0.05\\temporal_pinn\\pinn\\metrics.json",
      "variant": "dense1000_matched_temporal_v22",
      "tau_low": 0.01,
      "tau_high": 0.1,
      "pde_rms": 0.23463469314329918,
      "pde_max_abs": 2.333525211981142
    },
    {
      "key": "A_smooth__Fo0.1__uniform8__n1000__fine_nested__eta0.05",
      "artifact": "results\\A_smooth__Fo0.1__uniform8__n1000__fine_nested__eta0.05\\temporal_pinn\\pinn\\metrics.json",
      "variant": "dense1000_matched_temporal_v22",
      "tau_low": 0.1,
      "tau_high": 1,
      "pde_rms": 0.024559771906161505,
      "pde_max_abs": 0.2691842007346501
    }
  ],
  "coverage_complete": true,
  "consistency_screen_passed": false,
  "decision": "Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.",
  "screening_limits": {
    "ic_relative_L2": 1e-06,
    "replay_sigma": 1.1,
    "bc_rms_over_sensor_sigma": 1
  }
}
```


Provenance correction 2026-10-21T17:26:24.104745+00:00: original simulation is fine-grid RK4, confirmed by original ground-truth metadata and extend_sensor_density.py SOURCE=datasets_v2_rk4_zero. Earlier CN-generation statements were incorrect. CN remains an inversion/replay method. All data, fits and numerical scores are unchanged.
