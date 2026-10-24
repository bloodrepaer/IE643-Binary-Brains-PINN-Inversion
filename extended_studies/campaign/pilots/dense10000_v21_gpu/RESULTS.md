# v21: 10,000-sensor matched temporal comparison

Both fits completed 1,000 Adam steps and all 4,000 SSB iterations, with all 10 outer rounds reaching the iteration cap. Neither timed out. Independent supplied-physics checks and both-arm checkpoint/protocol/common-solver checks passed before coefficient truth was accessed. Full pair coverage is available.

| Metric | Temporal PINN | Backward-Euler neural adjoint |
|---|---:|---:|
| Coefficient relative L2 error | 6.7657% | 26.3639% |
| Common CN replay / sensor noise sigma | 1.083139 | 1.100680 |
| Common BE replay / sensor noise sigma | 1.030298 | 0.995894 |
| Model sensor fit / sigma | 0.993065 | 0.995894 |
| Function evaluations | 7,809 | 7,993 |
| Training wall seconds | 3,225.82 | 1,981.24 |

The temporal PINN has approximately 74.34% lower coefficient error in this case. It passes the predeclared 1.1-sigma replay screen under both solvers. The adjoint passes common BE and narrowly exceeds the CN threshold; its BE replay is better than the PINN's. Therefore this is a coefficient-reconstruction advantage with acceptable declared replay consistency, rather than uniform superiority in every metric.

PINN IC relative L2 = 2.4924e-7; boundary RMS = 3.0856e-18. Independent continuous PDE RMS across normalized-time bins [0,.001], [.001,.01], [.01,.1], [.1,1] is respectively .611337, .601279, .199489, .006874. These retain early-time residual error and are not the small scaled temporal training defect. Passing replay does not establish an exact PDE solution.

Fairness: both fresh fits used identical 80,000 noisy readings at 10,000 distinct sites, coefficient network/bounds, serialized initialization, regularizer/gamma, seed and iteration limits. Both used backward-Euler step dt/T=.01 and ran sequentially on the same Kaggle host. The PINN used CUDA for observations/residuals and CPU curvature; the adjoint used CPU, both with two CPU threads. Spatial AD versus FD, soft versus hard state constraints and random temporal pairs versus a fixed time grid remain different. Timing is descriptive, not an equal-device speed comparison.

Limitations: one smooth field, one fixed observation realization and optimization seed; all rounds capped, so convergence and population robustness are not established. The 6,031 additional sites came from the retained 129-grid field while inversion stayed on grid65; all parent3,969 readings were preserved exactly. Fixed-likelihood alpha implies density-dependent gamma. This does not isolate sensor count alone or show universal PINN superiority. No settings or checkpoints were selected using coefficient truth. All source versions, failed pretraining attempt and trained outputs are retained.

Evidence: primary_matched_neural_comparison.csv, pair_integrity.json (validation before truth), validation.json, evaluation_protocol.log, download_v2/extraction_manifest.json and original metrics/checkpoints. Other experiments remain held by the user's instruction; no automatic confirmation fits or resumed v20 work.
