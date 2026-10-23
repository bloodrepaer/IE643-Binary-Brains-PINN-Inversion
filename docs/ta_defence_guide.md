# TA defence guide: PINN vs adjoint for spatially varying diffusivity

Oct 22, 2026 · @Arun

Our classical adjoint inversion recovers D(x, y) more accurately than our PINN on 61 of 64 heat-equation datasets tested, typically by about 2×. The three exceptions are cases where neither method beats a constant guess. Both published papers we checked explain why: when the forward solver is reliable and D is a grid field, the adjoint is the expected winner.

## The story in one minute

We recover an unknown diffusivity map D(x, y) in the 2D heat equation from a few noisy temperature sensors, with two methods, and the classical one wins.

1. **Data.** We hand-built a finite-volume solver and generated 389 synthetic datasets from 6 families of true D fields. Each dataset varies sensor count (5 to every node), layout, sampling times, time horizon and noise (0–10%). The data is simulated with RK4 on a 129 × 129 grid; both inversions work on a coarser 65 × 65 grid, so neither method sees the model that made the data (no "inverse crime").
2. **Classical method.** PDE-constrained optimisation with a discrete adjoint: D is 256 numbers on a 16 × 16 grid, every guess is checked by solving the heat equation exactly (Crank–Nicolson), the gradient comes from one extra backward solve, and the smoothing weight is set by the discrepancy principle.
3. **PINN.** Two neural networks, one for temperature u(x, y, t) and one for log D(x, y), trained on the sensor data plus the heat-equation residual, boundary and initial conditions.
4. **Result.** On 20 representative datasets the adjoint had the lower D error on all 20, typically by 2×. The closest case was a circular inclusion, where the PINN was 3–10% worse. On the 44-dataset sensor × noise × layout sweep it won 41; the PINN's 3 wins are all at the best-constant level.
5. **Why this is expected, not a bug.** Both methods use the same physics, but the adjoint enforces it exactly and searches only 256 numbers; the PINN enforces it approximately and must also learn the whole temperature field. Zhang et al. (2026) reach the same conclusion in a controlled study: grid-type unknowns with a reliable solver favour the adjoint.
6. **We checked the obvious objections.** The result does not change if the data is generated with a different solver (RK4, Crank–Nicolson, forward Euler, or the adjoint's own solver). We also re-ran the key experiments of both reference papers and found where their claims hold and where they don't.

**One-line answer for deliverable 4:** in our fully specified heat problem the PINN is not better; the literature and our re-runs locate its advantage where the unknown is a neural field, the problem is high-dimensional or long in time, the data is scattered images with no grid, or the initial and boundary conditions are not known exactly.

## Claims, evidence and the paper behind each

Every claim below has a number from our own runs; the last column names the paper that supports or motivated it. Quotations are in the next section.

| # | Claim | Our evidence | Paper support |
| --- | --- | --- | --- |
| 1 | The classical adjoint beats our PINN on a fully specified heat problem. | 20/20 datasets, median D-error ratio PINN/adjoint 2.0× (range 1.03–13.5×). Adjoint beat a single best-constant D on 20/20; PINN on 8/20. | Zhang et al. 2026, Conclusions: adjoint preferred "when a reliable discrete forward solver is available and the unknown is low-dimensional" (Q1, Q2). |
| 2 | The difference comes from how the physics is imposed, not from a coding mistake. | Adjoint: exact discrete PDE, 256 unknowns. PINN: penalty, \~30,000 weights for u and log D together. Longer training (6,000 epochs) narrowed the gap but still lost 4/4. | Zhang et al. 2026, Sec. 1: hard vs soft constraint, "the governing equation holds only approximately at the optimum" (Q3). |
| 3 | The comparison is fair: no solver bias. | Data from RK4 on 129², inversion on Crank–Nicolson 65². Regenerating with Crank–Nicolson, forward Euler, or even the adjoint's own solver changed the adjoint's error by ≤ 0.004 and never reversed the ranking. The true D replays at RMS/σ = 0.98 (median, 19 datasets). | Zhang et al. 2026 control every shared factor for the same reason (Q4). |
| 4 | The representation of D decides the winner more than the algorithm. | Re-run of Zhang's Burgers test: with a neural-network D and 400 iterations, PINN 1.26e-4 vs adjoint 1.83e-4 (PINN better); with a grid D both 1.55e-3 (tie). | Zhang et al. 2026, Highlights and Table 3 (Q5, Q6). |
| 5 | With sparse, noisy data the method matters little; the data sets the limit. | Re-run of Zhang's Darcy test at 1% noise: adjoint 0.248, PINN 0.254, EKI 0.230. Our 44-dataset heat sweep (sensors × noise × layout): adjoint lower on 41/44. The PINN's 3 wins are all at the best-constant level (about 0.75 on the inclusion). | Zhang et al. 2026, Sec. 3.2: "the problem, not the method" (Q7). |
| 6 | Our PINN design follows the published PINN formulation for spatially varying D. | Two networks (u and log D), divergence-form residual D\_x u\_x + D\_y u\_y + DΔu, D = exp(network) so D > 0. | Thakur et al. 2024, eq. (3) and Sec. 2 (Q8). |
| 7 | Thakur's headline accuracy needs conditions the paper does not state, and a classical method matches it. | On their own four test fields, finite differences + least squares beat their PINN numbers (3.3 / 0.8 / 1.3 / 1.4% vs 4.8 / 4.6 / 6.31 / 4.58%). Their backward-Euler loss is biased: 31% error at Δt = 0.01 even with a perfect concentration field. | Thakur et al. 2024, Table 1 and eq. (5) (Q9, Q10). |
| 8 | The PINN's only clear published edge in our setting is scattered, sparse points. | Thakur Table 2 (25–250 points) is 6–37% better than our adjoint on comparable data; unconfirmed on identical data. | Thakur et al. 2024, Sec. 3.1 and 2 (Q11, Q12). |

## Exact quotations

All quotations are copied from the arXiv PDFs (Zhang et al. arXiv 2606.12337v2; Thakur et al. arXiv 2403.03970v1); line-break hyphens removed. Page numbers are the PDF's printed page numbers.

### Zhang, Alla & Karniadakis (2026), "Adjoint Method versus Physics-Informed Neural Networks in PDE-Constrained Inverse Problems"

Use this paper to explain **why the adjoint wins on our problem** and **where a PINN would win instead**.

- **Q1 (when to use the adjoint), Sec. 4 Conclusions, p. 29:**

  > "Use the adjoint method when a reliable discrete forward solver is available and the unknown is low-dimensional—a scalar, a coefficient, or a coarse field."

  *Say:* "Our D is a 16 × 16 coarse field and we have a verified solver, so this is the adjoint's home ground."
- **Q2 (when to use a PINN), Sec. 4, p. 29:**

  > "Use the PINN for neural-field and state-dependent unknowns and for high-dimensional or long-horizon time-dependent problems, where the adjoint’s trajectory storage becomes prohibitive."
- **Q3 (hard vs soft constraint), Sec. 1, pp. 4–5:**

  > "The adjoint imposes the discrete forward map as a hard constraint, optimizing on the manifold where the discrete residual vanishes exactly; the PINN imposes it softly, as a penalty in a loss minimized over the state and parameter networks jointly, so the governing equation holds only approximately at the optimum."

  *Say:* "Both use the same physics; the adjoint obeys it exactly, the PINN only approximately."
- **Q4 (fair comparison), Sec. 1, p. 5:**

  > "we control the comparison in the sense of matching every factor that does admit a common setting so that the differences that remain are attributable to the two paradigms rather than to incidental implementation choices."
- **Q5 (headline), Highlights:**

  > "Grid representations favor the adjoint; neural networks favor PINNs"
- **Q6 (representation dominates), Sec. 3.1, p. 15:**

  > "The representation is therefore the dominant factor and the algorithm the secondary one."
- **Q7 (noise sets the limit), Sec. 3.2, p. 20:**

  > "What limits the recovered permeability here is therefore the problem, not the method."

  and, same paragraph:

  > "once the noise sets the achievable accuracy, all three estimators reach it, and the choice among them reduces to cost."
- **Q7b (PINN weaknesses), Sec. 1, p. 3:**

  > "These advantages come with weaker accuracy and convergence guarantees, and with training difficulties associated with ill-conditioned PINN losses."
- **Q7c (the open question deliverable 4 asks), Sec. 1, p. 3:**

  > "the regimes in which PINNs provide a genuine practical advantage over adjoint optimization remain incompletely characterized."
- **Q7d (the adjoint's real cost), Sec. 1, pp. 2–3:**

  > "this efficiency comes with an implementation cost: for each forward discretization, the corresponding discrete adjoint must be derived, implemented, and verified."

  *Say:* "That's why we wrote test\_adjoint.py: Taylor-test slopes 1.996–2.000."

### Thakur, Esmaili, Libring, Solorio & Ardekani (2024), "Inverse resolution of spatially varying diffusion coefficient using Physics-Informed neural networks"

Use this paper to justify **our PINN design** and as the **published claim we tested**.

- **Q8 (the governing equation and two networks), Sec. 2, p. 3:**

  > "c\_t − c\_x D\_x − c\_y D\_y − D(c\_xx + c\_yy) = 0." (eq. 3)

  > "This is a parabolic equation, analogous to the equations pertinent to problems in heat transfer and the same framework can be used to determine thermal properties using temperature measurements. We approximate the functions c̃(x, y, t) and D̃(x, y) using two deep fully connected neural networks."

  *Say:* "This is exactly our setup: their concentration is our temperature, and our PINN uses the same divergence-form residual with two networks."
- **Q9 (their headline claim), Abstract, p. 1:**

  > "an error of less than 6.31% was obtained against all numerical benchmarks"
- **Q10 (their loss), Sec. 2, p. 4:**

  > "We now create a physics-informed neural network using backward Euler discretization."

  *Say:* "We showed this backward-Euler loss is biased unless Δt is very small; our PINN uses the continuous-time residual instead."
- **Q11 (sparse data), Sec. 3.1, p. 7:**

  > "The error in the diffusion coefficient only marginally increases till the number of points are reduced to 250. The solver finally breaks down when only 25 spatial points or “pixels” are considered."
- **Q12 (the real PINN advantage: no grid), Sec. 2, p. 4:**

  > "We are working with point clouds and just one global function and there is no discretization involved. This is significant as a lattice like distribution of points is not a requirement and we can work with any distribution of points in the spatio-temporal domain."
- **Q13 (no flux, no information), Sec. 2, p. 4:**

  > "regions where the concentration field does not vary, or the gradient is zero, gives no information for us to learn and determine the diffusion coefficient."

  *Say:* "This is why our interface shows a sensitivity map ∫|∇u|²dt: D cannot be recovered where heat doesn't flow, for any method."

### Raissi, Perdikaris & Karniadakis (2019), the original PINN paper

Cite as the foundation of the PINN idea: neural networks "trained to solve supervised learning tasks" while respecting physical laws given as PDEs; the PDE residual is computed by automatic differentiation and added to the loss.

## Deliverable 4: where is the PINN better?

On our fully specified heat problem the PINN is not better anywhere we tested; it is better, or equal, in four regimes, two of which we confirmed by re-running published experiments.

| Regime | Evidence | Strength | Applies to our project? |
| --- | --- | --- | --- |
| **Neural-network representation of the unknown** | Zhang Burgers, our re-run: PINN 1.26e-4 vs adjoint 1.83e-4 at 400 iterations; the adjoint in weight space varied 3.3× across seeds vs 1.3× for the PINN. With 2,000 iterations the adjoint reached 4e-7–1e-6. | Confirmed, but budget-dependent | Not directly: our classical D is a grid by design |
| **Sparse, noisy data** | Zhang Darcy at 1% noise, our re-run: PINN 0.254 vs adjoint 0.248 vs EKI 0.230 (a tie). Paper Table 4 at 1%: PINN 0.270 vs adjoint 0.287. | Confirmed tie, not a win | Tested on our heat data (44-dataset sweep): PINN wins 3/44, only on the inclusion with 5 random or boundary-only sensors. There it stays near a constant while the adjoint overshoots (1.53 with 20 boundary sensors). The PINN fails more gracefully, but it is not a better reconstruction. |
| **Scattered points, no grid** | Thakur Table 2: 6.45% at 250 points and 39% at 25 points vs our adjoint's 8.7% and 62% on reconstructed data. | Reported, not reproduced on identical data | Partly: our sensors are scattered, but our adjoint handles that too |
| **High-dimensional or long-horizon time problems** | Zhang Table 2: PINN 14× faster on 3D Allen–Cahn. Our re-run on a reduced 33³ grid: adjoint gradient cost grows 0.31 → 2.4 → 25 s from 17³ to 65³ grids. | Confirmed cost trend | No: our problem is 2D and short |
| **Unknown initial or boundary conditions** | Theory: the adjoint needs them exactly; a PINN can learn them from data. | Not yet tested | Yes — the most promising follow-up |

How to say it: "We did not find a regime in our problem where the PINN is more accurate. That agrees with Zhang et al.'s guidance (Q1) and their statement that PINN advantages 'remain incompletely characterized' (Q7c). Re-running their experiments, we confirmed the regimes where a PINN can win, and none of them is present in a 2D heat problem with known initial and boundary conditions and a grid-based D."

## Questions the TA is likely to ask

**"Your PINN loses everywhere. Isn't it just badly trained?"** We checked four ways. A 10-setting loss-weight sweep on the Gaussian bump: best PINN 0.398 vs adjoint 0.229. Four times longer training (6,000 epochs): errors fell (e.g. 0.447 → 0.343) but still lost 4/4. Removing the edge taper on D: worse. The paper's own controlled study reaches the same ranking on grid-type unknowns (Q1, Q5).

**"So does the PINN win anywhere at all?"** Three times out of 44 in our sweep, all on the sharp inclusion with too little information: 5 random sensors at 5 % noise (PINN 0.746 vs adjoint 0.779), 20 boundary-only sensors (0.766 vs 1.529) and 50 boundary-only sensors (0.740 vs 0.747). The best constant D scores about 0.75, so in all three the PINN is simply staying near a constant while the adjoint fits noise and puts structure in the wrong place. The honest claim is that **the PINN degrades gracefully when the data cannot identify D**; it is not a better reconstruction. Where the PINN genuinely wins is the set of regimes in the Deliverable 4 table above, which our problem does not contain.

**"Isn't the comparison biased because the data and the adjoint use the same solver?"** No. Data: RK4 on 129 × 129. Adjoint: Crank–Nicolson on 65 × 65. We regenerated the data with Crank–Nicolson, forward Euler and even the adjoint's exact solver and grid: the adjoint's error moved by at most 0.004 and the ranking never changed. The verdict uses relL2(D) against the true D, which involves no solver at all.

**"Why does the physics-informed method lose if it uses the physics?"** Both use the same heat equation. The adjoint makes every candidate an exact solution and searches 256 numbers; the PINN treats the equation as a penalty and must also learn u(x, y, t) with \~30,000 weights from a few sensors (Q3). Physics in the loss helps a PINN against a pure neural network, not against a method that already enforces the physics exactly.

**"Which metric do you rank on, and why?"** relL2(D) = ‖D̂ − D‖/‖D‖ against the true field: it measures what we actually want. We also report RMS replay/σ: push D̂ through the solver and compare with the readings; ≈ 1 means the map explains the data to the noise level. It works without the true D, which is why the interface displays it.

**"How do you choose the regularisation weight α?"** The discrepancy principle: start large, decrease by √10, stop at the first α whose data misfit reaches the noise level σ. It uses only σ, never the true D, so it is not tuned on the answer.

**"How do you know the adjoint gradient is right?"** `test_adjoint.py`: Taylor test slopes 1.996–2.000 (2 is exact), agreement with finite differences to 1e-8, forward solve matching the reference integrator to 8e-16.

**"Why log D and D = exp(m)?"** D must be positive, and a log keeps it positive without constraints; it also treats "twice as large" and "half as large" symmetrically. Both our methods do this; Thakur et al. do not say how they keep D positive.

**"Why a harmonic mean between grid points?"** Heat flowing across an interface passes two resistances in series, and the effective conductance of resistances in series is their harmonic mean. An arithmetic mean overestimates flux across a sharp jump.

**"Why is the inclusion the hardest case for both?"** Sharp edges are high-frequency content: the 16 × 16 bilinear grid blurs them and neural networks learn smooth shapes first (spectral bias). With 50 sensors at 5% noise both methods are near the best-constant baseline (0.66 and 0.73 vs 0.75). It is also where the PINN came closest (within 3–10%).

**"Did you check the papers' claims or just cite them?"** We re-ran both. Zhang: Burgers and Darcy numbers reproduce; we found the "under half the cost" warm-start figure depends on a stopping rule that uses the ground truth. Thakur: their accuracy needs an unstated small Δt; a classical method beats their Table 1; their agarose value (9.0e-10 m²/s) exceeds their own free-water value (4.0e-10), which is physically implausible.

**"What would you do with more time?"** Give both methods an imperfect initial condition (the adjoint needs it exactly; a PINN can learn it), and test a hybrid where the PINN's D warm-starts the adjoint, as Zhang et al. recommend.

## Numbers to remember

D error is relL2(D) unless marked; lower is better.

| What | Number |
| --- | --- |
| Datasets in the benchmark | 389 (6 D families + 11 variants, 35 ground-truth files) |
| PINN wins in our 20-dataset study | 0 / 20; median PINN/adjoint error ratio 2.0× |
| Closest case | Circular inclusion, Fo 0.02: adjoint 0.604, PINN 0.623 |
| Typical case (Gaussian bump, 50 sensors, 5%) | adjoint 0.229, PINN 0.400, best constant D 0.584 |
| Beats the best constant D | adjoint 20/20, PINN 8/20 |
| Median run time | adjoint 46 s, PINN 125 s (CPU) |
| Classical baselines (50 sensors, 5% noise) | A 0.10 · B 0.23 · C 0.66 · D 0.22 · E 0.21 |
| Sensor density, inclusion, 1% noise (adjoint) | 5 / 20 / 50 / 100 sensors → 0.65 / 0.61 / 0.44 / 0.20 |
| Solver-bias test | adjoint error change ≤ 0.004 across 4 data solvers |
| True D replayed through the evaluation solver | RMS/σ median 0.98 |
| Adjoint gradient check | Taylor slopes 1.996–2.000 |
| Zhang Burgers, neural D, 400 it. (our re-run) | PINN 1.26e-4 vs adjoint 1.83e-4 |
| Zhang Darcy, 1% noise (our re-run) | adjoint 0.248 · PINN 0.254 · EKI 0.230 |
| Thakur Table 1 vs classical FD + least squares (their metric) | 4.8 / 4.6 / 6.31 / 4.58% vs 3.3 / 0.8 / 1.3 / 1.4% |
| Thakur backward-Euler bias, perfect c, Case I | 31% at Δt = 0.01, 7% at Δt = 0.001, 5% continuous |

## Limitations to state before the TA does

Saying these first shows the conclusions are bounded, not oversold.

- **One PINN design.** We tuned loss weights, training length and the D taper, but did not try second-order optimisers (Zhang et al. use SSBroyden) or very long GPU training. A better-trained PINN could narrow the gap; nothing we saw suggests it would reverse it.
- **The PINN hard-codes the initial condition.** Its built-in base solution assumes the benchmark's two Gaussian bumps, so it cannot yet be tested with a different or uncertain initial condition — the regime most likely to favour it.
- **CPU only.** Runs were on 2 CPU cores; this limited the PINN to 1,500–6,000 epochs per dataset.
- **Thakur comparison is on reconstructed data.** The paper does not give its initial or boundary conditions, so its PINN numbers and our classical numbers are on similar, not identical, data.
- **Model-error floors.** Layered fields with sharp interfaces replay at up to 1.5σ even with the true D, because hard interfaces discretise differently on 129² and 65² grids.
- **The sensor/noise/layout sweep covers only two families** (inclusion and bump, 44 datasets). Result: adjoint lower on 41/44, median relL2 0.36 vs 0.61 (best constant 0.68); full tables in report Stage 6.

## Sources

- Z. Zhang, A. Alla, G. E. Karniadakis. [Adjoint Method versus Physics-Informed Neural Networks in PDE-Constrained Inverse Problems](https://arxiv.org/abs/2606.12337). arXiv 2606.12337v2, 2026 (submitted to Computer Methods in Applied Mechanics and Engineering).
- S. Thakur, E. Esmaili, S. Libring, L. Solorio, A. M. Ardekani. [Inverse resolution of spatially varying diffusion coefficient using Physics-Informed neural networks](https://arxiv.org/abs/2403.03970). arXiv 2403.03970v1, 2024; Physics of Fluids, 2024.
- M. Raissi, P. Perdikaris, G. E. Karniadakis. [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://doi.org/10.1016/j.jcp.2018.10.045). Journal of Computational Physics 378, 2019.
- Our verification reports: [Verifying Thakur et al. (2024)](https://claude.ai/artifact/RSd9pRuF4Fu8Xv2ctcuy8W); [Zhang et al. re-run, audit and 20-dataset study](https://claude.ai/artifact/W3sk4Nf8kfGmKnU3UshcoK); solver-bias test in the project notes (`claude/solver_bias_test.md`).
