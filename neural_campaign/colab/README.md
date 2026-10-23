# Run the IE643 sweep in Colab

Use **IE643_GPU_Density_Sweep.ipynb**. It is self-contained (about 5 MB): you do not need to upload the payload ZIP separately or enter Kaggle credentials.

1. Open [Google Colab](https://colab.research.google.com/) and upload the notebook.
2. Select a GPU through **Runtime → Change runtime type**.
3. Choose **Run all**, then authorize the Google Drive mount. Outputs go into `MyDrive/IE643_v24_density_sweep`.
4. The first research pair is **3,000 sensors at 5% noise**. Run cell 5 again to advance to 5,000, then 7,500. Keep the same Drive folder between sessions.
5. Download the results ZIP using cell 7 and return it for local coefficient-error evaluation. Review the fixed-noise density report before enabling the noise phase.

The 1,000-sensor runs are already completed/running separately and are not duplicated. Pending Kaggle density dispatch is held for this Colab transfer.

The PINN's data/residual calculations use CUDA; its quasi-Newton curvature and the BE adjoint use CPU. Both methods retain the same validated diffusion network, observations, initialization, regularization and iteration limits. Each fit has a 12,600-second internal cap; actual statuses and caps are recorded. GPU/software changes can change trajectories. No speed-superiority claim is made.

Finished-arm checkpoints and validation records are saved to Drive. Completed arms are verified and reused. Interrupted or partial arms are preserved and stop the runner; it does not silently retry or pretend to restore optimizer state. Export partial results for review. Colab allocation/disconnection limits apply; see the [official FAQ](https://research.google.com/colaboratory/faq.html).

The embedded inputs come from fine-grid **RK4** simulations. There are no coefficient answer keys, clean observations or credentials in the payload. The notebook validates checkpoints, supplied IC/BC, continuous PDE probes and both BE/CN replay. Diffusion-field error is evaluated locally only after those checks.

Local verification: notebook format/code compilation; embedded archive/hash checks; full-count one-Adam/one-SSB CUDA PINN plus CPU adjoint; independent validation without truth; ZIP export; completed-arm reuse; noise-phase gate; refusal to automatically retry partial attempts. These are compatibility smokes, not experimental results. Colab's cloud GPU allocation and Google Drive authorization require the user's session and were not exercised locally. See `qa_validation.json` and `bundle_manifest.json`.
