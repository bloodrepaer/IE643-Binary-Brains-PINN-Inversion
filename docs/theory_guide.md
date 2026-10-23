# Theory guide: recovering D(x, y) with a classical adjoint and a PINN

Oct 22, 2026 · @Arun

This guide explains every method in the project from first principles, then shows where each idea lives in the code, so nothing in the submission is a black box.

## 1. The big picture

The project solves an **inverse problem**: we see the effect (temperatures at a few sensors) and must infer the cause (the material property D(x, y) that controls how heat spreads).

**Forward vs inverse.**

|  | Forward problem | Inverse problem (ours) |
| --- | --- | --- |
| Given | D(x, y), initial temperature, boundary temperatures | a few noisy temperature readings, plus the initial and boundary conditions |
| Find | the temperature u(x, y, t) everywhere | the diffusivity map D(x, y) |
| Well-posed? | yes: one answer, stable to small changes | no: many D fit the data almost equally well, and noise is amplified |
| How solved | one time march of a solver | an optimisation that runs the forward solver hundreds of times |

A problem is **well-posed** (Hadamard, 1902) if a solution exists, is unique, and depends continuously on the data. Inverse problems usually fail the last two: heat diffusion smooths everything, so very different D maps can produce nearly the same smooth temperatures, and tiny noise in the readings can mean a huge change in D. Handling that instability ("regularisation", section 6) is half of the project.

**The pipeline of the project.**

1. **Choose a true D** (six families: smooth, bump, inclusion, layers, random field, plate with defects).
2. **Simulate the forward problem** with our own finite-volume solver to get u(x, y, t).
3. **Sample sparse, noisy sensors** from u at chosen positions and times. This is the only thing the inversions see, together with the initial and boundary conditions.
4. **Invert two ways:**
   - *Classical:* PDE-constrained optimisation with a discrete adjoint (sections 6–8).
   - *PINN:* two neural networks trained with a physics-informed loss (sections 9–13).
5. **Score** the recovered D̂ against the true D and by how well it explains the data (section 15).
6. **Compare** across sensor density, layout, noise, time sampling and D family (section 16), and expose it all in an interface (section 17).

The key idea to keep in mind throughout: **both methods use the same physics**. The difference is *how*: the classical method makes every candidate obey the heat equation exactly; the PINN only penalises violations of it. Most of what you see in the results follows from that difference.

## 2. The physics: the heat equation in divergence form

The equation comes from two facts: heat is conserved, and heat flows from hot to cold at a rate proportional to the temperature gradient.

**Step 1, conservation.** Take any small region. The rate of change of heat inside equals the net heat flowing in through its boundary. If **q** is the heat flux (heat per unit area per unit time), this gives, in differential form:

```latex
\frac{\partial u}{\partial t} = -\nabla\cdot\mathbf{q}
```

**Step 2, Fourier's law.** Flux points down the gradient, scaled by the local diffusivity D(x, y):

```latex
\mathbf{q} = -D(x,y)\,\nabla u
```

**Together** (the equation we solve, no heat source):

```latex
\frac{\partial u}{\partial t} = \nabla\cdot\big(D(x,y)\,\nabla u\big)
```

**Why the divergence form matters.** Expanding with the product rule:

```latex
\nabla\cdot(D\nabla u) = D\,\Delta u + \nabla D\cdot\nabla u = D\,(u_{xx}+u_{yy}) + D_x u_x + D_y u_y
```

The second term, ∇D·∇u, is where D *changes*. Dropping it (writing u\_t = DΔu, which is only correct for constant D) would throw away most of the information about where the material changes. Both our methods and Thakur et al.'s eq. (3) keep it. The PINN computes exactly the expanded right-hand side above.

**Initial and boundary conditions.**

- *Initial condition:* two Gaussian bumps of heat (peaks 1.0 at (0.3, 0.3) and 0.7 at (0.7, 0.65), widths 0.05 and 0.06). Same for every dataset.
- *Boundary condition:* Dirichlet (temperature prescribed) on all four edges. Default is u = 0 (edges held cold); other scenarios hold the left edge hot, ramp it, or pulse the top edge, so heat keeps entering and late times stay informative.

**Dimensionless time: the Fourier number.** How far heat has spread after time T in a domain of size L depends on the combination

```latex
Fo = \frac{\bar D\,T}{L^2}
```

Small Fo (0.02): heat has barely moved from the bumps, so most of the plate is never probed. Large Fo (0.2): heat has largely drained to the cold edges, so late readings are tiny and carry little information. We fix Fo and compute T = Fo · L² / D̄ per D family, so all families are compared at the same "amount of diffusion". The datasets use Fo ∈ {0.02, 0.05, 0.1, 0.2}; 0.1 is the default.

**What a sensor sees.** A reading at (x, y, t) depends on D everywhere heat has travelled to reach that point. It is most sensitive to D where the temperature gradient is large (heat is flowing). Where ∇u ≈ 0 — at the centre of a heat bump, or in regions heat never reaches — D has almost no effect on any reading, so no method can recover it there. The quantity

```latex
S(x,y) = \int_0^T |\nabla u|^2\,dt
```

measures this (the "sensitivity map" shown in the interface). Thakur et al. state the same: regions where "the gradient is zero, gives no information".

## 3. Forward solver, part 1: space (finite volumes)

The solver turns the PDE into a large system of ordinary differential equations, one per grid node: du/dt = L u + b(t).

**Grid.** The unit square is covered by N × N nodes with spacing h = 1/(N − 1). Data are generated on N = 129 (h ≈ 0.0078); both inversions work on N = 65 (h ≈ 0.0156). Every second data node is an inversion node, so the grids are nested.

**Finite volumes.** Put a small square "cell" around each node. Conservation (section 2) says: rate of change of heat in the cell = sum of the fluxes through its four faces. Each face sits between two neighbouring nodes p and q, and the flux through it is approximated by

```latex
\text{flux}_{q\to p} = c_{pq}\,(u_q - u_p),\qquad c_{pq} = \frac{1}{h^2}\,\frac{2D_pD_q}{D_p + D_q}
```

Adding the four faces gives the **5-point stencil**: node p talks only to its left, right, up and down neighbours.

**Why the harmonic mean.** Between nodes p and q, heat crosses half a cell of material D\_p and then half a cell of material D\_q. These act like two resistors in series: total resistance (h/2)/D\_p + (h/2)/D\_q, so the effective diffusivity is 1/(average of 1/D) = 2D\_pD\_q/(D\_p + D\_q). If one side is an insulator (D\_q → 0) the harmonic mean correctly goes to 0; the arithmetic mean would still let heat through at half rate. This matters at the sharp interfaces of the inclusion, layers and defects.

**Boundary nodes are eliminated.** Edge temperatures are known (Dirichlet), so they are not unknowns. Their effect on interior nodes moves to the right-hand side as a known vector b(t). What remains is

```latex
\frac{d\mathbf{u}}{dt} = L\,\mathbf{u} + \mathbf{b}(t)
```

for the interior unknowns (63 × 63 = 3,969 on the inversion grid). L is sparse (at most 5 non-zeros per row), **symmetric** (c\_pq = c\_qp), and **negative definite** (all eigenvalues negative: every temperature pattern decays). Symmetry is what later lets one matrix factorisation serve both the forward solve and the adjoint solve.

**Design choice recorded in the README.** A 9-point stencil was tried first; the 5-point flux form was 1.3–1.8× more accurate on every tier and gives a simpler adjoint, so it was kept.

**Accuracy.** The scheme is second order in space: halving h cuts the spatial error by about 4. Generating data on a finer grid (129²) than the inversion grid (65²) means the inversion's own model has a small built-in error — just as a real experiment never matches a simulator exactly (section 5, "inverse crime").

## 4. Forward solver, part 2: time stepping

We march du/dt = Lu + b(t) forward in steps of Δt. RK4 generates the data; Crank–Nicolson (CN) is used inside the inversion; forward Euler is kept only as a benchmark.

**Stiffness.** The eigenvalues λ of L range from about −2π²D (smooth patterns, decay slowly) to about −8D\_max/h² (grid-scale zig-zags, decay almost instantly). On 65² with D\_max ≈ 0.6 the largest magnitude ρ(L) is \~10⁴ times the smallest. A system with such a spread of time scales is called **stiff**.

**Explicit methods (forward Euler, RK4)** compute the new state from the old one only:

```latex
\text{Forward Euler: } \mathbf{u}^{n+1} = \mathbf{u}^n + \Delta t\,(L\mathbf{u}^n + \mathbf{b}^n)
```

Applied to one eigenmode, each step multiplies it by (1 + Δtλ). If |1 + Δtλ| > 1 for any mode, that mode grows every step and the solution explodes. So forward Euler needs Δt ≤ 2/ρ(L), and RK4 (four stages per step, error ∝ Δt⁴) needs Δt ≤ 2.785/ρ(L). Since ρ(L) ∝ 1/h², halving h forces 4× more steps. The README records the blow-up: at 1.05× the limit, both explode to \~10²⁰⁰.

**Implicit method: Crank–Nicolson** averages the right-hand side over the old and new times:

```latex
\Big(I - \tfrac{\Delta t}{2}L\Big)\mathbf{u}^{n+1} = \Big(I + \tfrac{\Delta t}{2}L\Big)\mathbf{u}^{n} + \tfrac{\Delta t}{2}\big(\mathbf{b}^{n} + \mathbf{b}^{n+1}\big)
```

Each mode is multiplied by (1 + Δtλ/2)/(1 − Δtλ/2), whose size is below 1 for every negative λ and every Δt: **unconditionally stable**, and second-order accurate. The price is a linear solve per step — but the matrix M = I − (Δt/2)L is the same at every step, so it is factorised once (sparse LU) and each step is two cheap triangular solves.

**The Rannacher start.** For very stiff modes (Δtλ → −∞) the CN factor tends to −1: those modes are not damped, they flip sign every step. From a sharp initial condition this produces visible oscillations (the README shows u = −0.96 from a point-like start). The fix: replace the first CN step by two backward-Euler half-steps, which damp stiff modes strongly, then continue with CN. Our solver and its adjoint both include these two half-steps; with h·θ = Δt/2 in both cases, one matrix M serves all steps.

**Measured orders of accuracy** (tier C, 65², against the exact-in-time matrix exponential): RK4 4.03, CN 2.01, forward Euler 1.00 — exactly the theoretical 4, 2 and 1.

**Who uses what.**

| Use | Integrator | Grid | Step |
| --- | --- | --- | --- |
| Generating the data | RK4 | 129 × 129 | largest stable step (\~10⁻¹⁰ error) |
| Classical inversion and the replay metric | CN + Rannacher | 65 × 65 | T/200 (T/400 if needed to land on the sensor times) |
| PINN's own internal check (χ²\_solver) | RK4, arithmetic-mean faces | 65 × 65 | CFL-limited |

We tested whether this choice could bias the comparison: regenerating the data with CN, forward Euler, or the inversion's own CN on 65² changed the readings by at most 0.05 of the noise level and the adjoint's D error by at most 0.004 (section 16).

## 5. The synthetic dataset

The benchmark has 389 datasets built from 35 ground-truth simulations; each dataset is one way of observing one simulation.

**The true D fields** (`domain.py`), a ladder of difficulty:

| Family | Formula / description | What it tests |
| --- | --- | --- |
| A smooth | 0.05 + 0.45 · ½(1 + sin 2πx · cos 2πy) | sanity check: low frequency, no edges |
| B Gaussian bump | 0.08 + 0.42 · exp(−r² / (2·0.15²)) around (0.5, 0.5) | a localised feature, no sharp edge |
| C inclusion | disc of radius 0.15, D = 0.6 inside, 0.08 outside (contrast 7.5), tanh edge | sharp interface; variants with contrast 2 and 15 |
| D layered | stripes along x: 0.05 / 0.3 / 0.6 with jumps at x = 0.33, 0.66 | hard jumps, several scales; checkerboard variants |
| E random field | white noise filtered by a Gaussian in Fourier space (correlation length 0.1), squashed into \[0.05, 0.5\] | rough, natural-looking fields; correlation length 0.03 / 0.3 variants |
| F plate with defects | D₀ = 0.25 with 3 spots of radius 0.07 where D is ×3 or ÷3 | non-destructive testing; variants with 1 or 6 defects, smaller, lower contrast, voids |

D is built on the 129² data grid; the inversion-grid truth is every second node, so tier E is the same random field on both grids.

**Sensors.**

- *Count:* 5, 20, 50, 100, or "full" (all 3,969 interior nodes of the 65² grid).
- *Layout:* random, regular grid, clustered (3 hot-spots), boundary (within 0.1 of an edge).
- *Nested:* the 5-sensor set is the first 5 of the 20-sensor set, and so on, so adding sensors only adds information.
- *Times:* fixed grids `uniform4` … `uniform200`, `log8` (dense early), `early8` (t ≤ T/4), `late8` (t ≥ T/2), `final1` (only t = T).
- *Readings:* the clean temperature is interpolated at each sensor position and time from the simulation.

**Noise.** Gaussian noise with standard deviation σ = η · RMS(reference signal), η ∈ {0, 1, 5, 10%}. The reference is fixed per (family, horizon, η) — 100 random sensors at 40 times — so changing sensor count, layout or timing never changes the noise level ("one thermometer"). Earlier versions used an absolute σ that ranged from 30% to 280% of the signal; relative noise fixed that.

**Experiment axes (the sweeps).**

| Sweep | Varies | Held fixed |
| --- | --- | --- |
| S1 density × noise | sensors {5, 20, 50, 100, full} × η {0, 1, 5, 10%} | Fo 0.1, 8 uniform times, random layout |
| S2 horizon × timing | Fo {0.02, 0.05, 0.1, 0.2} × 6 time protocols | 50 random sensors, 5% noise |
| S3 time density × noise | uniform4 … uniform200 × η {1, 5, 10%} | Fo 0.1, 50 random sensors |
| S4 layout | 4 layouts × {20, 50} sensors | Fo 0.1, 8 times, 5% noise |

63 datasets per main family × 6 families + 11 variant baselines = **389**. Seeds come from `zlib.crc32` of the dataset name, so two runs give byte-identical files.

**The inverse crime, and how we avoid it.** If the data are made with exactly the same model the inversion uses, the inversion can fit them perfectly, and results look better than they ever would on real measurements. We avoid this on two counts: data on a finer grid (129² vs 65²) and with a different time integrator (RK4 vs CN). Check: replaying the *true* D through the inversion's solver reproduces the noisy data at RMS/σ ≈ 1.01 (median over 296 datasets) — the model error is far below the noise.

**What each method may read.** `problem/` (domain, T, initial condition, boundary values) and `sensors/` (x, y, t, u, σ). `ground_truth/` is the answer key, opened only after an inversion has finished, for scoring.

## 6. Why the inverse problem is hard, and regularisation

Fitting the data alone is not enough: infinitely many D maps fit sparse noisy readings, and most of them are wild. Regularisation adds a preference for plausible maps.

**The least-squares fit.** Let d be the vector of readings, Q the "observation operator" that reads the simulated temperature at the sensor positions and times, and u(m) the simulated temperature for log-diffusivity m = log D. The data misfit is

```latex
\Phi(m) = \frac{1}{2\sigma^2}\,\big\| Q\,u(m) - d \big\|^2
```

Dividing by σ² measures the misfit in units of the noise: a perfect model with noisy data still has Φ ≈ N\_d/2, i.e. RMS error ≈ σ.

**Three reasons minimising Φ alone fails.**

1. **Diffusion smooths.** Fine detail in D produces only tiny, smooth changes in u. Mathematically, the map D → readings has singular values that decay quickly, so inverting it divides small noise by small numbers.
2. **Not enough information.** 50 sensors × 8 times = 400 numbers; D on the 65² grid has 4,225 values.
3. **Blind spots.** Where ∇u ≈ 0 (section 2) D does not affect any reading, so the data cannot pin it down there at all.

**Regularisation** adds a penalty R(m) weighted by α:

```latex
J(m) = \frac{1}{2\sigma^2}\,\big\| Q\,u(m) - d \big\|^2 + \alpha\,R(m)
```

Large α: smooth, safe, but biased (detail is lost). Small α: fits the data closely, but noise turns into fake structure. Choosing α is section 8.

**The two penalties we use** (on m = log D):

|  | Tikhonov (H¹) | Total variation (TV) |
| --- | --- | --- |
| Formula | ½ ∫ \|∇m\|² dx | ∫ √(\|∇m\|² + ε) dx |
| Penalises | gradients *squared*: one big jump costs far more than many small slopes | gradient *size*: a jump costs the same whether sharp or spread out |
| Prefers | smooth fields | piecewise-constant fields with sharp edges |
| Best for | families A, B, E | families C, D, F |
| In our data | layered field: Tikhonov 0.22 vs TV 0.34 | defects: TV found 3/3 with overlap (Dice) 0.62 vs Tikhonov 0.46 |

The small ε (10⁻² in the adjoint, 10⁻⁶ in the PINN) keeps TV differentiable where ∇m = 0.

**Parameterisation is also regularisation.** The classical method does not optimise all 4,225 grid values. It optimises θ, 16 × 16 = 256 values on a coarse "control grid", and spreads them to the fine grid by bilinear interpolation P:

```latex
m = P\,\theta,\qquad D = e^{m},\qquad \theta \in \mathbb{R}^{16\times 16}
```

This cannot represent features smaller than about 1/15 of the domain — which is also roughly what sparse sensors can resolve. The cost: a circular inclusion comes out slightly square, and sharp edges are blurred over one control cell.

**Why log D.** D must be positive. Optimising m = log D makes positivity automatic, and treats "twice too large" and "half too small" as equally wrong. Bounds 10⁻³ ≤ D ≤ 10 keep the solver well-behaved.

## 7. The classical method: the discrete adjoint gradient

The adjoint method gives the exact gradient of J with respect to all 256 unknowns for the cost of about two forward simulations, instead of 257.

**Why we need it.** An optimiser needs ∂J/∂θ. The naive way — nudge each of the 256 values and re-run the simulation — costs 257 solves per gradient and is inexact. The adjoint costs one forward solve plus one backward solve, whatever the number of unknowns.

**"Reduced space".** The temperature u is never a free variable: for any θ we compute u(θ) by solving the heat equation exactly. So the heat equation is a *hard constraint*, and J depends on θ only. This is what Zhang et al. call optimising "on the manifold where the discrete residual vanishes exactly".

**Derivation, in four steps** (discretise-then-optimise: we differentiate the exact time-stepping code, Rannacher steps included, so the gradient is exact for the computed J).

**Step 1 — write each time step as a constraint.** Step j takes state u\_j to u\_{j+1}. F applies the operator (interior part of L times the full-grid state w, which includes boundary values), h\_j is the step size, and θ\_j = ½ for CN or 1 for the two backward-Euler start steps:

```latex
r_j \equiv u_{j+1} - u_j - h_j\big[\theta_j\,F(w_{j+1}) + (1-\theta_j)\,F(w_j)\big] = 0
```

**Step 2 — form the Lagrangian** by adding each constraint times a multiplier vector λ\_{j+1} (the "adjoint state"). For any λ, 𝓛 equals J whenever the constraints hold:

```latex
\mathcal{L} = J(u, m) + \sum_j \lambda_{j+1}^{\top}\, r_j(u_{j+1}, u_j, m)
```

**Step 3 — choose λ to cancel every ∂u/∂θ term.** Setting ∂𝓛/∂u\_s = 0 for every state gives a recursion that runs **backward in time**, starting from λ\_S = 0:

```latex
M\,\lambda_s = \lambda_{s+1} + h_s(1-\theta_s)\,L\,\lambda_{s+1} - \frac{\partial J}{\partial u_s},\qquad M = I - \tfrac{\Delta t}{2}L
```

The source term ∂J/∂u\_s = Qᵀ(Qu − d)/σ² injects the data misfit at each sensor and time, like little point sources of "error" that diffuse backward. Because M is symmetric, the same factorisation as the forward solve is reused.

**Step 4 — read off the gradient.** What remains is the explicit dependence on each link conductance c\_l (link l joins nodes p and q):

```latex
\frac{\partial J}{\partial c_l} = \sum_j h_j\,(\lambda_{j+1,p}-\lambda_{j+1,q})\big[\theta_j\,\delta w_{j+1} + (1-\theta_j)\,\delta w_j\big],\qquad \delta w = w_p - w_q
```

The chain rule through c = 2D\_pD\_q/((D\_p + D\_q)h²), D = eᵐ and m = Pθ then gives ∂J/∂θ = Pᵀ(D ⊙ ∂J/∂D) + α∇R.

**Intuition.** The adjoint state λ\_s answers: "how much would J change if I added a little heat at node i at time s?" A link's conductance matters when heat actually flows through it (δw ≠ 0) at times when extra flow there would change the misfit (δλ ≠ 0).

**Cost and memory.** One forward solve storing every state, then one backward solve. Memory grows with grid size × number of steps: 202 states × 4,225 nodes here, which is trivial. For 3D or long time horizons this storage dominates — the regime where Zhang et al. find PINNs cheaper.

**How we know it is correct** (`test_adjoint.py`, run before trusting any inversion):

| Check | Idea | Result |
| --- | --- | --- |
| Forward consistency | inversion solver vs reference integrator | max difference 8e-16 |
| Transpose identity | ⟨Qx, y⟩ = ⟨x, Qᵀy⟩ on random vectors | 6e-16 |
| Taylor test | \|J(θ + εv) − J(θ) − ε g·v\| must shrink like ε²; log–log slope 2 means the gradient g is exact | slopes 1.996–2.000 (Tikhonov and TV, real data and time-varying boundaries) |
| Finite differences | central differences on a few components | worst relative difference 1e-8 |

A slope of 1 in the Taylor test would mean the gradient is wrong, and L-BFGS would still converge — to the wrong answer. That is why the test matters.

## 8. The classical method: optimiser and choosing α

The inversion runs L-BFGS-B at a sequence of decreasing α values and stops at the first α whose fit matches the noise level.

**L-BFGS-B.** Newton's method would use the matrix of second derivatives (the Hessian) to take ideal steps, but that is a 256 × 256 matrix of expensive quantities. Quasi-Newton methods build an approximation of the inverse Hessian from the history of gradients: each step's change in gradient (y) per change in parameters (s) reveals curvature along s. **L**imited-memory BFGS keeps only the last \~10 pairs (s, y), so memory stays small; **-B** adds simple box bounds, here log 10⁻³ ≤ θ ≤ log 10. Each iteration costs one forward + one adjoint solve. Settings: at most 150 iterations per α, stop when the relative change in J < 10⁻⁸ or the gradient < 10⁻⁶.

**Stage 0: the best constant D.** Before anything else, a one-dimensional search finds the single uniform D that best fits the data. It is the starting guess for θ and also the **baseline** every map must beat: if a 256-parameter map is no better than one number, it has learned nothing. (Adjoint beat it on 20/20 datasets; the PINN on 8/20.)

**Continuation in α.** Start with a large α = N\_d (number of readings), so the first solution is very smooth and easy to find. Then divide α by √10 each round (13 values down to N\_d · 10⁻⁶), starting each round from the previous solution ("warm start"). This walks the solution gradually from smooth to detailed and avoids bad local minima.

**The discrepancy principle (Morozov).** If the model were perfect, the best we can hope for is to fit the readings to within the noise: RMS(data − prediction) ≈ σ. Fitting any closer means the map is reproducing the noise itself, i.e. inventing structure. So we stop at the **first (largest) α whose RMS misfit ≤ σ**:

```latex
\alpha^{*} = \max\Big\{\alpha : \sqrt{\tfrac{1}{N_d}\textstyle\sum_i \big(Qu(m_\alpha) - d\big)_i^2} \le \sigma \Big\}
```

It uses only the noise level σ (stored in each sensor file), never the true D, so the choice is honest. The PINN code uses the same idea in its "Morozov damping" (section 12).

**Run flags.**

- `reached_sigma`: the normal case — an α met the noise level.
- `floor_above_sigma`: the misfit stopped improving (two rounds with < 1% gain) above σ, because of model error (e.g. the layered field's hard interfaces). Then the largest α within 2% of the best misfit is kept.
- `noise_free`: no σ to aim for; α decreases until the misfit stops improving. These runs take longest.

**Typical cost.** 20–120 s per dataset on a laptop CPU, a few hundred PDE solves in total.

## 9. Neural networks: the building blocks of the PINN

Both PINN networks are small multilayer perceptrons (MLPs) with tanh activations and a random Fourier-feature input layer; no convolutions or attention are involved.

**The multilayer perceptron.** An MLP is a chain of layers. Each hidden layer applies an affine map, then a fixed nonlinear function σ elementwise:

```latex
z_k = \sigma\big(W_k\,z_{k-1} + b_k\big),\qquad k = 1,\dots,H;\qquad \text{output} = W_{H+1}\,z_H + b_{H+1}
```

W\_k (weights) and b\_k (biases) are the trainable parameters. Without σ the whole chain would collapse into one linear map; with it, the **universal approximation theorem** says a wide enough network can approximate any continuous function on a bounded domain to any accuracy. It says nothing about how hard that approximation is to *find* by training — which is where PINNs struggle.

**Why tanh, not ReLU.** The physics loss needs second derivatives u\_xx and u\_yy of the network output. ReLU(z) = max(0, z) is piecewise linear, so a ReLU network's second derivative is zero almost everywhere — the Laplacian would vanish and the PDE residual would be meaningless. tanh is infinitely smooth, so all derivatives exist and are informative. Its derivatives are simple: tanh′ = 1 − tanh², tanh″ = −2 tanh (1 − tanh²).

**Initialisation.** Weights start random, drawn so that signals neither explode nor vanish as they pass through the layers. **Xavier (Glorot) normal** initialisation uses variance 2/(fan\_in + fan\_out), designed for tanh. Our output layers use a small gain (0.02 for u, 0.01 for log D), so at the start the networks output almost zero corrections: the PINN begins from its built-in base solution and a uniform D (section 11), then learns deviations.

**Spectral bias.** Neural networks trained by gradient descent learn smooth, low-frequency parts of a function first and sharp, high-frequency parts much later (Rahaman et al., 2019). For us that means smooth D families come out reasonably, but sharp edges — the inclusion, layers and defects — are blurred, and the Thakur verification saw a high-frequency field collapse entirely.

**Random Fourier features** (Tancik et al., 2020) counter spectral bias by feeding the network sines and cosines of the inputs instead of the raw coordinates:

```latex
\gamma(\mathbf{x}) = \big[\sin(2\pi\,\mathbf{x}B),\ \cos(2\pi\,\mathbf{x}B)\big],\qquad B_{ij}\sim\mathcal{N}(0, s^2)
```

B is drawn once and frozen (a buffer, not trained). The scale s sets which frequencies are easy to learn: large s → fine detail but noisy fits; small s → smooth. Our code uses s = 0.35 for both networks, a deliberately smooth choice that suits noisy sparse data but limits sharp edges.

**Our two networks** (`FourierMLP` in `pinn_inversion.py`):

|  | u-network (temperature) | m-network (log D) |
| --- | --- | --- |
| Inputs | (x/L, y/L, t/T) | (x/L, y/L) |
| Fourier features | 3 → 80 (40 sin + 40 cos), s = 0.35 | 2 → 64 (32 sin + 32 cos), s = 0.35 |
| Hidden layers | 3 × 80 units, tanh | 3 × 64 units, tanh |
| Output | 1 value, gain 0.02 | 1 value, gain 0.01 |
| Trainable parameters | 19,521 | 12,545 |

Total ≈ 32,000 parameters, against 256 for the classical method's control grid — one reason the PINN's search is so much harder.

## 10. Automatic differentiation: how the PINN gets u\_t and u\_xx

A neural network is a long chain of simple operations, so its exact derivatives with respect to its inputs can be computed by applying the chain rule mechanically — no finite differences, no grid.

**Two kinds of derivative in a PINN.**

1. **Derivatives of the output with respect to the inputs** (x, y, t): u\_t, u\_x, u\_y, u\_xx, u\_yy, and m\_x, m\_y for the D network. These build the PDE residual.
2. **Derivatives of the loss with respect to the weights**, used by the optimiser. Since the loss already contains (1), this is a derivative of a derivative.

**Reverse mode (backpropagation).** PyTorch records every operation in the forward pass as a graph, then walks it backward multiplying local derivatives. Reverse mode gives the gradient of one scalar with respect to *all* inputs in one backward pass, which is why training networks with millions of weights is cheap. `torch.autograd.grad(u, [x, y, t], create_graph=True)` returns u\_x, u\_y, u\_t; `create_graph=True` records *that* computation too, so we can differentiate again to get u\_xx and u\_yy, and later back-propagate the whole residual to the weights.

**Why the PINN residual is exact but still approximate.** Autodiff gives the derivatives of the *network* exactly (to machine precision — Thakur et al. make this point). But the network is only an approximation of the true temperature, so its second derivatives can be noticeably wrong even when its values fit the data well. Section 14 shows why that matters so much for recovering D.

**Cost.** Each extra order of input derivative roughly doubles to triples the work, and the weight gradient then runs through all of it. With 768 collocation points, second derivatives of a 3-input network and first derivatives of the D network, one PINN epoch costs \~0.1 s on CPU; 1,500 epochs ≈ 2–4 minutes.

**Contrast with the adjoint.** The adjoint also computes an exact derivative — of the *discrete solver's* misfit with respect to θ — but derived by hand (section 7). Autodiff saves that derivation effort, which is the main practical appeal of PINNs that Zhang et al. note ("requires only the PDE residual, rather than a discretization-specific adjoint").

## 11. Our PINN's architecture, piece by piece

The PINN does not learn u and D from scratch: physics-based structure is built into the outputs so that the networks only learn corrections (class `InverseDiffusionPINN`).

**The temperature output.**

```latex
u_\theta(x,y,t) = \underbrace{u_{\text{base}}(x,y,t)\,\tau_{25}(x,y)}_{\text{analytic guess}} + \underbrace{\tanh\!\big(8t/T\big)}_{\text{time gate}}\ \underbrace{16\,x(1-x)\,y(1-y)}_{\text{boundary window}}\ \mathrm{NN}_u(x,y,t)
```

Each factor has a job:

- **Base solution u\_base.** The initial condition is two Gaussian bumps. In an infinite uniform medium, a Gaussian of width s spreads into a wider Gaussian: variance s² + 2Dt, with the peak lowered by s²/(s² + 2Dt) so total heat is conserved. The code uses exactly this "heat kernel" for both bumps, with one trainable effective diffusivity D\_eff = exp(clamp(β, −4.5, 0)). So before any training, u\_θ is already a decent approximation of the true temperature.
- **Edge taper τ₂₅ = tanh(25x)·tanh(25(1−x))·tanh(25y)·tanh(25(1−y))** pulls the base solution to zero at the edges, matching the zero boundary temperature.
- **Boundary window 16x(1−x)y(1−y)** is 0 on every edge and 1 at the centre, so the network's correction can never violate the zero Dirichlet condition. This is a **hard constraint** built into the PINN.
- **Time gate tanh(8t/T)** is 0 at t = 0, so at the initial time u\_θ equals the (tapered) initial bumps exactly, whatever the network does.

**The diffusivity output.**

```latex
m_\phi(x,y) = m_{\text{bg}} + \tau_6(x,y)\,\mathrm{NN}_m(x,y),\qquad D_\phi = \exp\!\big(\mathrm{clamp}(m_\phi,\,-5,\,2)\big)
```

- **Background m\_bg** is a trainable number initialised at log(0.85 · Fo · L²/T), i.e. close to the average D implied by the time horizon. The network then learns only the spatial variation.
- **Interior taper τ₆ = tanh(6x)·tanh(6(1−x))·tanh(6y)·tanh(6(1−y))** forces D back to the background within \~0.15 of the edges, where readings carry little information. Our ablation removing it made results worse (e.g. smooth field 0.447 → 0.524).
- **exp and clamp** keep D positive and within \[0.0067, 7.4\].

**What this design buys and costs.**

| Design choice | Benefit | Cost |
| --- | --- | --- |
| Heat-kernel base | starts near the right temperature, fast early training | hard-codes the two-bump initial condition: a different initial condition breaks it |
| Boundary window, time gate | boundary and initial conditions satisfied exactly | assumes zero-temperature edges |
| Background + taper on D | stable, avoids nonsense near edges | cannot represent D that differs near the edges (smooth family A) |
| exp(log D) | positivity | none |

The classical method has the equivalent of all these properties automatically, because it solves the actual PDE with the actual initial and boundary conditions.

## 12. Our PINN's loss, term by term

The PINN minimises a weighted sum of five terms; the weights 40 / 20 / 10 / 20 were chosen by a 10-setting sweep (setting "C04").

```latex
\mathcal{L} = 40\,\mathcal{L}_{\text{data}} + 20\,\mathcal{L}_{\text{PDE}} + 10\,\mathcal{L}_{\text{BC}} + 20\,\mathcal{L}_{\text{IC}} + \lambda_{\text{reg}}\,R(m)
```

**1. Data term (fit the sensors), with "Morozov damping".** The plain mean squared error over a random batch of 400 readings is L\_d = mean (u\_θ − ũ)². The code uses

```latex
\mathcal{L}_{\text{data}} = \frac{\mathrm{ReLU}\big(L_d - 0.95\,\sigma^2\big) + 0.05\,L_d}{\overline{\tilde u^{2}}}
```

Once the fit is within the noise level (L\_d ≤ 0.95σ²), the ReLU switches off and only 5% of the pull remains, so the network stops chasing the noise — the same idea as the discrepancy principle (section 8), applied softly inside training. Dividing by the mean squared reading makes the term scale-free.

**2. PDE residual (obey the physics).** At 768 random "collocation" points plus the current data points:

```latex
r = u_t - \big(D_x u_x + D_y u_y + D\,(u_{xx}+u_{yy})\big),\qquad D_x = D\,m_x,\ D_y = D\,m_y
```

This is the expanded divergence form of section 2. Two refinements in the code:

- **Time weighting** w(t) = clamp((0.02 + t/T)/0.12, 0.25, 8): temperatures shrink as heat drains away, so late-time residuals are tiny; the weight boosts them so late times still inform D.
- **Asymmetric gradient routing.** The residual is computed twice. In r\_m the u-derivatives are *detached* (treated as constants), so its gradient only updates the D network; in r\_u, D is detached, so it only updates the u network. The loss is 0.9·mean((w r\_m)²) + 0.1·mean((w r\_u)²). Reason: without this, the easiest way to reduce the residual is often to bend the u network (distorting its Laplacian) rather than fix D; routing 90% of the physics signal to D keeps u anchored to the data.

**3. Boundary term.** mean(u\_θ²) at random edge points. Mostly redundant because the boundary window already forces zero, but it also polices the base solution.

**4. Initial-condition term.** mean (u\_θ(x, y, 0) − u₀)² at 256 random points from the initial-condition file, divided by the mean of u₀².

**5. Regulariser on m = log D**, same purpose as in section 6:

- Tikhonov (families A, B, E, λ = 2·10⁻⁴): mean(|∇m|² + 0.05 (m − m\_bg)²)
- TV (families C, D, F, λ = 3·10⁻⁴): mean(√(|∇m|² + 10⁻⁶) + 0.02 (m − m\_bg)²)

The small extra term pulls D gently towards its background where data say nothing.

**Compared with the classical objective.** The classical J has two parts (misfit + αR) and the PDE holds exactly. The PINN has five parts with hand-set weights, and the PDE is only one of them, so the result depends on the balance. That is the "ill-conditioned PINN losses" Zhang et al. refer to.

**A known weakness: mean vs sum.** The data term is a *mean* over the readings in each batch (all of them when there are ≤ 400, otherwise a random 400), so 5 sensors and 100 sensors pull with the same total strength. The project README itself recommended a sum (1/σ²)Σ(u\_θ − ũ)² so that more data counts for more; with a mean, sensor-density sweeps partly measure nothing for the PINN.

## 13. Training the PINN: optimiser, schedules, sampling

Training runs 1,500 "epochs"; each epoch is one gradient step of AdamW on a freshly sampled set of points.

**Adam, in one paragraph.** Plain gradient descent takes θ ← θ − η g. Adam keeps two running averages per parameter — of the gradient (momentum, smooths noisy directions) and of the squared gradient (scale) — and steps by their ratio:

```latex
m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2,\quad \theta \leftarrow \theta - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
```

(m̂, v̂ are bias-corrected; β₁ = 0.9, β₂ = 0.999.) Because each parameter's step is normalised by its own gradient scale, Adam copes with loss terms of very different sizes — important when the PDE term is thousands of times smaller than the data term. **AdamW** adds weight decay (10⁻⁵) applied directly to the weights rather than through the gradient.

**Two learning rates.** The u network (and the base solution's D\_eff) uses η = 1.5·10⁻³; the D network (and m\_bg) uses 3.5·10⁻³.

**Schedules.**

- **Cosine annealing:** η(e) = η\_min + (η₀ − η\_min)·½(1 + cos(πe/E)), η\_min = 10⁻⁵: large steps early, fine steps at the end.
- **Delayed start for D:** the D network's rate is multiplied by min(1, max(0.02, (e − 50)/(0.15E))). For the first \~50 epochs D barely moves while u fits the data; then D's rate ramps up over 15% of training. Learning D from a wildly wrong u would be meaningless.
- **Gradient clipping** at norm 1.0 stops single bad batches from throwing the weights far.

**Point sampling each epoch.**

| Points | How many | Where |
| --- | --- | --- |
| Data | all readings if ≤ 400, else random 400 | sensor positions and times |
| Initial condition | 256 | random grid nodes at t = 0 |
| Boundary | 128 | random points on the four edges, random times |
| Collocation (PDE) | 768 | 60% "focus": x, y ∈ \[0.05, 0.95\] and times skewed early, t = T(0.02 + 0.65r¹·⁴); 40% uniform; plus the data points |

Resampling every epoch means the PDE is enforced on a different random cloud each time — the network cannot "cheat" at fixed points, and the noise acts like stochastic gradient descent.

**Checkpointing.** Among the last 35% of epochs, the one with the lowest total loss is saved as `pinn_best.pt` and used for evaluation; `pinn_final.pt` keeps the last epoch. A checkpoint stores the weights of both networks and the full configuration, which is how the interface rebuilds D from a saved model.

**Compared with Zhang et al.** They train PINNs with a quasi-Newton optimiser (SSBroyden) in double precision with residual resampling, which converges much further than Adam in float32. We used Adam on CPU in float32, so our PINN is likely less converged than theirs; longer training (6,000 epochs) improved our PINN's errors but did not change any ranking.

## 14. Why the PINN struggles on this problem

The PINN is solving a strictly harder optimisation problem than the adjoint with the same information, for six concrete reasons.

1. **Soft vs hard physics.** The adjoint only ever considers temperatures that solve the heat equation exactly; the PINN's temperature satisfies it only approximately, at sampled points, weighted against four other loss terms (Zhang et al., Q3 in the defence guide).
2. **It must learn the temperature too.** The adjoint gets u for free from the solver. The PINN must learn u(x, y, t) everywhere — between and far from the sensors — with \~19,500 weights, guided only by the approximate physics there.
3. **D is read off u's second derivatives, and errors there bias D low.** The residual u\_t = DΔu + ∇D·∇u is linear in D, with Δu and ∇u as the "regressors". If those come from a network whose second derivatives are noisy, least squares systematically shrinks the fitted D towards zero — the classic **errors-in-variables (attenuation) bias**. We measured this directly in the Thakur verification: D came out 3–4× too low until the concentration fit was \~100× better than we could train on CPU.
4. **The network can fit the data with the wrong D.** Example, inclusion, 50 sensors, 5% noise: the PINN's own network fits the readings at χ² = 0.95 (looks "optimal"), but pushing its D̂ through the real solver gives χ² = 6.4 and relL2(D) = 0.74. A flexible u can absorb errors that the physics penalty does not fully punish. That is why we always score D by replaying it through the solver, never by the network's own fit.
5. **Spectral bias and smooth features.** Networks learn smooth shapes first, and our Fourier scale s = 0.35 favours smoothness further. Sharp edges (inclusion, layers, defects) are where both methods struggle, the PINN more so.
6. **Balancing and optimising a five-term loss.** The relative weights decide what the network prioritises, and gradients of different terms can differ by orders of magnitude and pull in different directions (Wang, Teng & Perdikaris, 2021, call these "gradient pathologies"). The adjoint's reduced problem is smooth and low-dimensional, solved with exact gradients by L-BFGS; the PINN's is non-convex in \~32,000 weights, solved stochastically by Adam in float32.

**What we tried, and what each did** (Gaussian bump unless noted):

| Change | Effect on PINN D error | Still loses? |
| --- | --- | --- |
| 10 loss-weight settings (C01–C10) | best 0.398 (unregularised, equal weights) vs adjoint 0.229 | yes |
| 4× longer training (6,000 epochs) | smooth field 0.447 → 0.343; random field 0.440 → 0.341 | yes, 4/4 |
| Remove the edge taper on D | worse on 4/4 (e.g. 0.447 → 0.524) | yes |
| Different data-generating solver | changes within run-to-run noise (±0.03) | yes |

**The positive side.** None of these are bugs; they are properties of the soft-constraint approach. They become worth paying when the adjoint is unavailable or expensive: no reliable solver, unknown initial or boundary conditions, high dimensions or long horizons, or a neural-network unknown (Zhang et al., Q2).

## 15. Metrics: what each number means

We rank methods by relL2(D) against the true field, and use the replay misfit RMS/σ as the metric that works without an answer key.

| Metric | Formula | Ideal | Needs true D? | Use |
| --- | --- | --- | --- | --- |
| **relL2(D)** | ‖D̂ − D‖ / ‖D‖ over the 65² grid | 0 | yes | main ranking |
| RMS of log-D error | √mean (log D̂ − log D)² | 0 | yes | treats ×2 and ÷2 errors equally; good for high-contrast fields |
| **RMS replay / σ** | run D̂ through the CN solver, RMS(prediction − readings) / σ | ≈ 1 | no | the interface's displayed metric |
| χ²\_red ("slide 16" metric in the PINN code) | (RMS replay / σ)² | ≈ 1 | no | same information, squared |
| Best-constant D | relL2 of the best single uniform D | — | yes | floor every map must beat |
| High / low sensitivity error | relL2(D) split at the median of ∫\|∇u\|²dt | — | yes | shows D is only recoverable where heat flows |
| Defect metrics (family F) | detected?, contrast recovered, localisation error, Dice overlap | — | yes | non-destructive-testing view |

**Reading RMS replay / σ.**

- **≈ 1:** D̂ explains the readings to the noise level — the best any method can claim from the data.
- **> 1:** D̂ does not explain the data (under-fit, or model error).
- **< 1:** D̂ fits part of the noise (over-fit).

It is a necessary, not sufficient, check: many different D maps replay at ≈ 1 (that is ill-posedness, section 6), so a good replay does not prove a good map. Conversely the *true* D replays at ≈ 1 too (median 0.98), which shows the metric is fair to any correct answer.

**Why not the PINN's own data fit (χ² of the u network)?** Because the network can fit the readings while its D is wrong (section 14, point 4). Every PINN result in our comparisons replays D̂ through the same CN solver as the adjoint's.

**The Thakur metric (their eq. 7)** divides the RMS error by the *standard deviation* of the true field instead of its norm: √(mean(a − â)² / mean(a − ā)²). For fields with a large mean and small variation it is far stricter than relL2: their Case I, 3.3% on their metric, is 0.42% relL2. Never compare numbers across the two metrics.

## 16. Experiments and what each one proves

Nine experiments, each answering one question; together they show the adjoint's advantage is real, robust, and explained.

| # | Experiment | Question | Result |
| --- | --- | --- | --- |
| 1 | Classical baselines (6 families, sweeps S1–S4) | Does the adjoint work, and how does accuracy depend on the data? | Fits every dataset to the noise level (RMS/σ 0.92–1.00). relL2: smooth 0.10, bump 0.23, inclusion 0.66, layers 0.22, random 0.21. Inclusion error falls 0.65 → 0.20 as sensors go 5 → 100 (1% noise). |
| 2 | PINN 10-case benchmark (Kaggle run) | Does the PINN work? | Yes, but 0.35–0.75 relL2 vs the adjoint's 0.10–0.66 on the same 10 cases; lost 10/10. |
| 3 | PINN loss-weight sweep (10 settings, bump) | Is it a tuning problem? | Best 0.398 vs adjoint 0.229; tuning moves results ±0.25 but never below the adjoint. |
| 4 | 20-dataset head-to-head | Is there any dataset where the PINN wins? | 0/20; median ratio 2.0×; closest 1.03× (inclusion, short horizon). Adjoint beats the constant baseline 20/20, PINN 8/20. |
| 5 | Ablations (6,000 epochs; no edge taper) | Undertrained? Taper harmful? | Longer helps but still loses 4/4; removing the taper hurts. |
| 6 | Re-run of Zhang et al. (Burgers, Darcy, Allen–Cahn) | Does the published comparison hold up? | Numbers reproduce. PINN wins with a neural-network unknown at equal small budget (1.26e-4 vs 1.83e-4); tie under sparse noisy data (Darcy 1%: 0.254 vs 0.248). The warm-start "under half the cost" claim depends on a stopping rule that uses the ground truth. |
| 7 | Verification of Thakur et al. | Does the PINN-for-D paper hold up? | A classical method beats their Table 1; their backward-Euler loss is biased (31% at Δt = 0.01 with a perfect field); several settings missing; agarose value implausible. |
| 8 | Solver-bias test | Is the comparison rigged by the data solver? | No: 4 data solvers, adjoint error changes ≤ 0.004, PINN within ±0.03, ranking unchanged. |
| 9 | Sensor density × noise × layout sweep for both methods (inclusion, bump; 44 datasets) | How do both degrade with fewer sensors, more noise, other layouts? | Running; results go into the report. |

**How to read the body of evidence.** Experiments 1–5 establish the result on our problem; 6–7 connect it to the literature and test the published claims; 8 rules out the most likely flaw in the setup; 9 is the systematic sweep deliverable 5 asks for.

## 17. The interface and how it maps to the theory

The Streamlit app (`app/streamlit_app.py` on top of `app/core.py`) runs the project's own solvers unchanged; every tab corresponds to a stage of the pipeline.

| Tab | What the user does | Theory behind it |
| --- | --- | --- |
| 1 · Data | generate a synthetic dataset (any D family or an uploaded D, sensors, layout, times, noise, boundary condition), pick a benchmark dataset, or upload a CSV of readings | sections 2–5: the forward solver (RK4, 129²), sensor sampling, relative noise |
| 2 · Run | run the classical adjoint (control grid K, Tikhonov/TV, iterations) and/or the PINN (epochs, regulariser, weight, seed) | sections 6–8 and 11–13 |
| 3 · Results | metrics, maps, error maps log(D̂/D), profiles, α-selection curve, sensitivity map, PINN loss curves; download models and D̂ | section 15; the sensitivity map is ∫\|∇u\|²dt from section 2 |
| 4 · Load model | open a saved `classical_model.npz` (θ on the control grid) or `pinn_best.pt`, rebuild D, score it | D = exp(Pθ) for the adjoint; the m network for the PINN |

**Design decisions.**

- **One data format for every source.** Generated, benchmark and uploaded data are all written to the project's own folder format (`problem/`, `sensors/`, `ground_truth/`, `manifest.json`), so both original solvers load user data with no special cases.
- **Same scoring for both methods.** Every D̂ is replayed through the same CN solver; relL2(D) is added when the true D is known.
- **Uploads.** Readings must have columns x, y, t, u (optional σ); values are checked (x, y in \[0, 1\], t > 0). The app estimates the average D from the best constant fit to set the PINN's time scale (Fo). An uploaded initial condition replaces the default, with a warning that the PINN's base solution assumes the two-bump start.
- **Trained models.** The classical "model" is its control-grid vector θ plus metadata; the PINN's is its checkpoint. Reloading either reproduces the stored scores exactly (checked: inclusion, 0.661 / 0.98 and 0.728 / 2.33).
- **Testing.** Every path was run headlessly with Streamlit's AppTest and end to end in a real browser.

## 18. Theory → code map

Where each idea in this guide lives in the submitted files.

| Concept (section) | File | Function / class |
| --- | --- | --- |
| D families (5) | `domain.py` | `D_tier_A_smooth` … `D_tier_F_defects`, `get_D_field` |
| 5-point operator, harmonic-mean links (3) | `integrators.py` | build\_operator, harmonic\_mean, `edge_masks`, `BoundaryCondition` |
| RK4 / CN / forward Euler, Rannacher (4) | `integrators.py` | `integrate(..., method="rk4" \| "cn" \| "fe"), ForwardModel, spectral_radius_bound` |
| Integrator accuracy and stability (4) | `compare_integrators.py` | writes `integrator_report.json`, convergence figure |
| Dataset generation, sensors, noise, sweeps (5) | `generate_dataset_v2.py` | `plan`, `s``ensor_locations, sigma_reference, read_clean, sample, write_problem, seed_of`, `main` |
| Problem files are sufficient (5) | `check_problem_spec.py` | replays the true D from `problem/` alone |
| Misfit, Tikhonov/TV, control grid P (6) | `adjoint_inversion.py` | `AdjointSolver.objective`, `.regulariser`, `bilinear_1d` |
| Observation operator Q (6) | `adjoint_inversion.py` | `Discretization.observation_operator` |
| Forward solve, adjoint sweep, gradient (7) | `adjoint_inversion.py` | `AdjointSolver.forward`, `.objective(want_grad=True)` |
| Gradient verification (7) | `test_adjoint.py` | `taylor`, finite-difference check |
| L-BFGS-B, continuation, discrepancy principle (8) | `adjoint_inversion.py` | `AdjointSolver.best_constant`, `.invert` |
| Sensitivity map, defect metrics (2, 15) | `adjoint_inversion.py` | `sensitivity_map`, `defect_metrics`, `evaluate` |
| Fourier-feature MLP (9) | `pinn_inversion.py` | `FourierMLP` |
| Base solution, windows, tapers (11) | `pinn_inversion.py` | `InverseDiffusionPINN.diffusing_heat_kernel`, `.forward_u`, `.forward_m`, `.forward_D` |
| Autodiff derivatives, PDE residual, regulariser (10, 12) | `pinn_inversion.py` | `compute_pde_and_reg_loss`, `d_wrt` |
| Boundary loss (12) | `pinn_inversion.py` | `compute_bc_loss` |
| Data loader (5) | `pinn_inversion.py` | `load_v2_experiment`, `resolve_v2_root` |
| AdamW, schedules, sampling, checkpoints (13) | `pinn_inversion.py` | `train_single_experiment` |
| PINN config and loss-weight sweep (12) | `pinn_inversion.py` | `PINNConfig`, `LAMBDA_SWEEP_GRID` |
| χ² replay metric (15) | `pinn_inversion.py` | `evaluate_slide16_chi2_metric` |
| Interface backend: data I/O, scoring, models (17) | `app/core.py` | `generate_synthetic`, `load_upload`, `score`, `constant_baseline`, `run_classical`, `run_pinn`, `load_model_file` |
| Interface UI (17) | `app/streamlit_app.py` | the four tabs |

**Command-line entry points.**

- `python generate_dataset_v2.py` — build the 389-dataset benchmark (\~3 min).
- `python test_adjoint.py` — verify the adjoint (\~10 s; all checks must pass).
- `python adjoint_inversion.py --keys <dataset> [--reg tv] [--control 16]` — classical inversion.
- `python pinn_inversion.py --single-run --data-root . --exp-id <dataset>` — PINN (also `--run-lambda-sweep`, benchmark suite by default).
- `streamlit run app/streamlit_app.py` — the interface.

## 19. Glossary

| Term | Meaning here |
| --- | --- |
| Adjoint state λ | Multiplier field solved backward in time; tells how a small heat injection at a node and time would change the misfit |
| Collocation points | Random points where the PINN's PDE residual is evaluated |
| Control grid | The 16 × 16 coarse grid of values θ that defines D in the classical method |
| Crank–Nicolson (CN) | Implicit, second-order, unconditionally stable time stepping |
| Discrepancy principle | Choose the regularisation weight so the data are fitted to the noise level, no closer |
| Dirichlet condition | Temperature prescribed on the boundary |
| Divergence form | ∇·(D∇u); keeps the ∇D·∇u term that locates changes in D |
| Fourier number Fo | D̄T/L², dimensionless amount of diffusion |
| Hard / soft constraint | Equation satisfied exactly (adjoint) / only penalised in the loss (PINN) |
| Ill-posed | Solution not unique or not stable to data noise |
| Inverse crime | Generating data with exactly the inversion's model, giving over-optimistic results |
| L-BFGS-B | Limited-memory quasi-Newton optimiser with box bounds |
| Morozov damping | The PINN's soft version of the discrepancy principle inside the data loss |
| Rannacher start | Two backward-Euler half-steps before CN to damp stiff modes |
| relL2(D) | ‖D̂ − D‖/‖D‖, main accuracy metric |
| Replay | Pushing D̂ through the forward solver and comparing with the readings |
| Sensitivity map | ∫\|∇u\|²dt: where the data carry information about D |
| Spectral bias | Networks learn smooth functions before sharp ones |
| Stiff system | ODEs with very different time scales; explicit methods need tiny steps |
| Taylor test | Checks a gradient: the first-order remainder must shrink like ε² |
| Tikhonov / TV | Penalties on \|∇m\|² (smooth) / \|∇m\| (edge-preserving) |

**Further reading.** Raissi, Perdikaris & Karniadakis, J. Comput. Phys. 378 (2019) — PINNs. Zhang, Alla & Karniadakis, arXiv 2606.12337 (2026) — adjoint vs PINN. Thakur et al., arXiv 2403.03970 (2024) — PINNs for spatially varying D. Tancik et al., NeurIPS 2020 — Fourier features. Rahaman et al., ICML 2019 — spectral bias. Wang, Teng & Perdikaris, SIAM J. Sci. Comput. 2021 — gradient pathologies in PINNs.
