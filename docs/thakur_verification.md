# Verifying Thakur et al. (2024): PINN inversion of spatially varying diffusivity

Oct 21, 2026 · @Arun

The paper's synthetic results are achievable in principle, but not by the method as written: its backward-Euler loss is biased unless Δt is far below the data spacing, and a 10-second classical method beats every Table 1 number on the paper's own test cases.

## Table 1 re-run: classical methods beat the reported PINN

On the paper's four noise-free test cases, plain finite differences + least squares recovers D 1.5–5× more accurately than the paper's reported PINN, in about 10 s per case. Errors use the paper's own metric, eq. (7): RMS error divided by the standard deviation of the true field (lower is better).

| Case | D(x, y) | Paper's PINN (Table 1) | Classical FD + least squares, full grid | Classical adjoint, 2,500 scattered points | Paper's loss with a perfect c, Δt = 0.01 |
| --- | --- | --- | --- | --- | --- |
| I | 0.05 + 0.1\[x(1−x) + y(1−y)\] | 4.8% | 3.3% | 6.3% | 28.5% |
| II | 0.25 + 0.1\[sin 2πx + sin 2πy\] | 4.6% | 0.8% | 2.7% | 10.1% |
| III | 0.25 + 0.1\[sin 4πx + sin 4πy\] | 6.31% | 1.3% | 5.2% | failed (D → 0) |
| IV | 0.3 + 0.15 tanh(20x−10) + 0.1 tanh(20y−7) | 4.58% | 1.4% | 3.5% | 9.1% |

- **Same physics, different data.** The paper does not give its initial or boundary conditions. We used zero initial concentration and Gaussian boundary sources (c = 1 at the middle of the top and bottom edges, 0.5 at the sides), which reproduces the pattern of Fig. 2. Data: finite-volume Crank–Nicolson on a 501 × 501 mesh, 100 snapshots to t = 1, as in the paper.
- **FD + least squares** is the classical method the paper cites as ref. \[7\]. It needs the full gridded field and no noise. No tuning: the smallest regularisation weight tried was used.
- **Adjoint** is the project's own `adjoint_inversion.py` (16 × 16 control grid, 65 × 65 Crank–Nicolson) on 2,500 randomly placed points × 100 times, 5–9 minutes per case.
- **Paper's loss with a perfect c** is the best case for the paper's method: the concentration network is replaced by the exact field, and only the D network is trained. Section 2 explains the large errors.

&#91;image: Recovered D maps: true field, FD + least squares, adjoint, and the paper's loss with a perfect concentration field, for Cases I–IV\]

Case I looks right in every panel but scores 28.5% in the last column. Its D varies only between 0.05 and 0.10, so a uniform 3.6% overestimate is large compared with the field's own variation. Eq. (7) is therefore a strict metric, not a lenient one.

## The backward-Euler loss is biased unless Δt is tiny

Even with a perfect concentration field, the paper's loss (eq. 5) recovers D with 31% error at Δt = 0.01, the data's own time spacing. It needs Δt ≤ 0.002 to fall below 10%. The paper never states its Δt.

&#91;image: Case I error vs Δt in eq. (5), with the exact concentration field and only D trained\]

| Δt in eq. (5) | 0.001 | 0.002 | 0.005 | 0.01 | 0.02 | 0.05 | continuous residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Case I error L(D) | 7.2% | 8.2% | 18.4% | 31.2% | 45.5% | 73.8% | 5.2% |
| Mean D̂ / mean D | 1.007 | 1.007 | 1.023 | 1.040 | 1.057 | 1.093 | 1.002 |

- **Why.** A backward-Euler step is exact only to first order: c(t) = c(t+Δt) − Δt c\_t(t+Δt) + (Δt²/2) c\_tt. The loss asks D to absorb that leftover term, so it is minimised by a D that is too large. Direct evaluation confirms it: the loss is lower at 1.04 × D\_true than at D\_true.
- **Where.** 87–91% of the leftover comes from t < 0.1, when the boundary sources switch on and c changes fastest. The size of the bias therefore depends on the initial and boundary conditions, which the paper does not report.
- **The same with Cases II and IV** at Δt = 0.01: 10.1% and 9.1%, against 0.6% for Case IV with the continuous residual.
- **What the scheme buys.** With c̃ a network, eq. (5) only adds a Δt-dependent bias to a standard continuous PINN residual (c\_t − f)². The paper's accuracy is reachable only in the limit where the two coincide.

**Case III failed with both losses.** The 8-layer tanh D network collapsed to D ≈ 0.01 in all three attempts (backward Euler with two seeds, continuous residual with 10,000 iterations at a lower learning rate). The loss itself is fine: it is 200× lower at the true D than at D = 0. So reproducing Case III needs some unstated detail, such as a different activation, Fourier features or much longer training.

## Full PINN training: D comes out 3–4× too low until c is fitted almost perfectly

Trained as described (two 8 × 128 tanh networks, eq. 5–6, Adam with a decaying rate, all 25 million readings), the PINN recovered a nearly flat D ≈ 0.02 for Case I against a true mean of 0.083. We could afford 20,000 iterations on 2 CPU cores.

| Run (Case I, noise-free) | Iterations | Concentration error L(c) | D error L(D) | Mean D̂ / mean D | Wall time |
| --- | --- | --- | --- | --- | --- |
| Paper (Table 1) | not stated | 0.018% | 4.8% | ≈ 1 | not stated |
| Ours, Δt = 0.01, batch 2,000 | 6,000 | 6.6% | 727% | 0.07 | 21 min |
| Ours, Δt = 0.001, batch 1,000 | 20,000 | 1.4% | 553% | 0.31 | 38 min |

- **Why D collapses.** D enters the loss multiplied by the network's second derivatives of c. While c̃ is only roughly fitted, those derivatives carry large errors, and a least-squares fit with noisy regressors is pulled towards zero: the classical errors-in-variables (attenuation) bias. The FD + least-squares method collapses the same way once 0.1% noise is added to the data (section 4).
- **What reproduction needs.** The concentration fit must get about 100× better (from 1.4% to the paper's 0.018%) before D can be trusted. At the paper's network size that is a GPU job, probably 10⁵–10⁶ iterations. `pinn.py` runs unchanged on a GPU with `jax[cuda]`.
- **Practical consequence.** The method's D accuracy depends entirely on how well the concentration network is trained. The paper gives no iteration count or stopping rule, so readers cannot tell how much training their own data would need.

## Sparse points and noise: where a PINN-type method could still win

For sparse, noise-free points (Table 2), the paper's reported PINN errors are somewhat lower than our adjoint's. This is the one regime here where the paper's claims hold up against a classical method. The paper never tests noise on data with a known answer.

| Case I data | Paper's PINN (Table 2) | Classical adjoint (ours) | FD + least squares (ours) |
| --- | --- | --- | --- |
| every mesh point, noise-free | 4.89% | — | 3.3% |
| 2,500 points × 100 times, noise-free | 6.0% | 6.3% | cannot run (needs a grid) |
| 250 points × 100 times, noise-free | 6.45% | 8.7% | cannot run |
| 25 points × 100 times, noise-free | 39% | 62% | cannot run |
| full grid, 0.1% noise | not tested | — | 777% (D → 0) |
| 2,500 points, 1% noise | not tested | 48% (relL2 6.2%) | cannot run |

Case IV with 2,500 points: adjoint 3.5% noise-free, 20% (relL2 9.1%) at 1% noise; FD + least squares collapses at 0.1% noise.

- **Caveat.** The PINN column is the paper's own numbers on its own data; the other columns use our reconstruction. Our adjoint runs on a 65 × 65 grid, which leaves a small model-error floor on noise-free data.
- **Noise breaks naive finite differences immediately.** Differentiating noisy data twice amplifies the noise by about 1/h². An output-least-squares method (adjoint) or a smooth surrogate (a PINN) avoids that. This is where the paper's approach is genuinely motivated, but it is the case the paper did not test.
- **Scattered points are the PINN's natural input**, while FD + least squares needs a grid. The adjoint handles scattered points too, at 5–9 minutes per case.

## Text and figure audit

The Table 1 and Table 2 numbers are internally consistent, but the equations, one figure and one experimental result do not check out.

| Where | Problem | Severity |
| --- | --- | --- |
| Eq. (4) | f = c\_x D\_x − c\_y D\_y − D(c\_xx + c\_yy): the x and y flux terms have opposite signs. For eq. (5) to be a backward-Euler step, f must equal c\_t = D\_x c\_x + D\_y c\_y + D Δc. As printed, eqs. (4)–(5) do not discretise eq. (3). | Typo, but the method is defined by it |
| Sec. 3.2.2 | D = 9.0e-10 m²/s for Rhodamine 6G in 0.5% agarose is 2.2× the paper's own free-water value (4.0–4.05e-10 at 23 °C). A gel can only slow diffusion, so this suggests an artefact (interface, moving front or scaling), not a gel property. | Major: physically implausible |
| Fig. 3(d) | The Case IV formula gives a step 0.25 → 0.55 at x = 0.5 (along y = 0.5), or 0.20 → 0.40 at y = 0.35 (along x = 0.5). The plot shows about 0.17 → 0.35 near 0.3. | Figure does not match its formula |
| Fig. 2(a), 2(c) | The Case I colourbar tops at 0.090, but D reaches 0.100 at the centre (Fig. 3a shows 0.1). The Case III colourbar's lowest label reads "0.50" instead of 0.05. | Minor |
| Fig. 3, 4(b) | The axes of the dimensionless cases say m²/s. The text gives 4.0e-10 for water, above the 3.9e-10 colourbar maximum. | Minor |
| Eq. (7) | Justified as unchanged by "multiplication or addition of a constant". That holds only if both fields change together (a change of units). It is variance-normalised, so stricter than relative L2 for fields with a large mean. | Clarity |
| Reproducibility | Missing: initial and boundary conditions; D-network width; activation; how D > 0 is enforced; Δt in eq. (5); iterations and learning-rate schedule; how equation points are chosen. No code or data release. A "penalization parameter" is mentioned but the loss has none. | Major for verification |
| Claims | "Robust" to noise, yet every synthetic test is noise-free. The experiments have no ground truth and are compared only with literature ranges. | Overstated |

**Checks that pass.**

- The Stokes–Einstein ratio for the liquid experiment, D₂ = D₁·μ₁/μ₂ = 4.05e-10 × 0.935/1.60 = 2.37e-10, matches the stated 2.36e-10.
- Table 1 Case I (4.8e-2) agrees with Table 2 at 250,000 points (4.89e-2).
- The Case I–III cross-sections in Fig. 3 match their formulas.

## What this means for the project's deliverable 4

This paper does not show a regime where a PINN beats classical inversion, because it never runs one. Its experiments do point to the two places worth testing in your own sweep.

- **Sparse scattered points, noise-free or low noise.** The paper's reported errors at 25–2,500 points are 5–37% lower (relative) than our adjoint on comparable data. That is the only comparison in either paper verified so far where a PINN number is ahead.
- **Dense noisy data without a solver for the forward problem.** Naive finite differences fail at 0.1% noise; a PINN is a smooth surrogate that can differentiate noisy data. An adjoint also copes, so this needs a direct head-to-head on identical data.
- **Avoid copying eq. (5).** Use the continuous residual (c\_t − f)², as your `pinn_inversion.py` already does, or keep Δt well below the data spacing. Otherwise the D estimate carries a Δt bias.
- **Report D accuracy with relL2 as well as eq. (7)** when you cite this paper, since the two differ by up to 8× for fields with a large mean (Case I: 3.3% vs 0.42%).

## How it was run, and sources

Everything was re-implemented from the paper's text; the paper releases no code. The scripts and every run's numbers are in `thakur_verification.zip`, sent in the chat.

| Part | What we did |
| --- | --- |
| Data | Finite-volume, 5-point, harmonic-mean conductances; Crank–Nicolson with two backward-Euler start steps; Δt = 10⁻³, 501 × 501 mesh, snapshots every 0.01 to t = 1; c scaled to \[0, 1\]. Initial and boundary conditions assumed (see section 1). |
| PINN | JAX. c̃(x, y, t) and D̃(x, y) each 8 hidden layers × 128, tanh, D = exp(output); eq. (5) with the sign of eq. (3); eq. (6) loss; Adam, rate 10⁻³ → 10⁻⁵; derivatives propagated exactly through the network (checked against autodiff). |
| Perfect-c test | Same D network; c and its derivatives from finite differences of the 501 × 501 data; the first 0.03 time units skipped. |
| FD + least squares | Field subsampled to 126 × 126; central differences in time; D on a 48 × 48 bilinear grid; normal equations with a small smoothness penalty. |
| Adjoint | Project's `adjoint_inversion.py`, 16 × 16 control grid, Tikhonov, α by the discrepancy principle (noise-free: until the fit stops improving). |
| Compute | 2 CPU cores, no GPU. That limited the full PINN to 20,000 iterations. |

**Sources**

- S. Thakur, E. Esmaili, S. Libring, L. Solorio, A. M. Ardekani. [Inverse resolution of spatially varying diffusion coefficient using physics-informed neural networks](https://arxiv.org/abs/2403.03970). arXiv 2403.03970v1, 2024 (published in Physics of Fluids, 2024, doi 10.1063/5.0207453).
- S. Thakur, M. Raissi, H. Mitra, A. Ardekani. [Temporal consistency loss for physics-informed neural networks](https://arxiv.org/abs/2301.13262). arXiv 2301.13262, 2023 (the paper's ref. \[30\], source of eq. 6).
