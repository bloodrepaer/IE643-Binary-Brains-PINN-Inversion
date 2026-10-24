# v23 dispatched cases complete

Both dispatched pairs independently validated and scored. Cases 03–07 remain superseded, not dispatched.

At 1% noise: PINN D error 9.3988% vs BE adjoint 32.0358%; PINN CN/BE replay 2.9684/1.7073 sigma (both fail), adjoint 2.6065/.9964 (CN fails/BE passes). Coefficient advantage is not a consistency-qualified PINN win.

At 10% noise: PINN 69.0982% vs adjoint 21.5769%; PINN replay 5.8598/5.9759 (both fail), adjoint 1.0013/.9741 (both pass).

Frozen 5% result retained: PINN 35.3716% vs adjoint 21.4547%, PINN both replay screens fail/adjoint both pass. See noise_comparison_n1000.csv and per-case RESULTS.md for raw RMS, caps, actual resources, validation and limitations. No verified PINN win across these three 1,000-sensor cells. This is one field/realization/seed and noise-dependent regularization, not a general noise law.

No new compute dispatched. Remaining v24 density-first sweep belongs to the user Colab notebook; all other holds remain.
