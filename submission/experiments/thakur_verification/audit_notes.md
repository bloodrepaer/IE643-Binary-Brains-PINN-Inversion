# Text and figure audit: Thakur et al. 2024 (arXiv 2403.03970v1)

1. **Eq. (4) has sign errors.** f = c_x D_x − c_y D_y − D(c_xx + c_yy) mixes signs: the x and y flux terms have opposite signs. Eq. (3) gives c_t = D_x c_x + D_y c_y + DΔc. For eq. (5) to be a backward-Euler step, c(t) = c(t+Δt) − Δt c_t(t+Δt), f must equal that c_t. As printed, eq. (5) with eq. (4) is not a consistent discretisation of eq. (3). Our re-implementation uses the consistent version.

2. **Missing for reproduction:**
   - initial condition and boundary conditions of the OpenFOAM simulations;
   - width of the D network;
   - activation function and output transform (how D > 0 is enforced);
   - Δt in eq. (5);
   - number of iterations and the learning-rate schedule;
   - how the equation points Nᵉ are chosen;
   - no code or data release is mentioned.

   "To choose the penalization parameter, we use the backward Euler discretization and the standard deviation of the concentration field" does not name any penalisation parameter. The loss has no free weight.

3. **Figures:**
   - Fig. 2(a): the D colourbar tops out at 0.090, but the stated Case I field reaches 0.100 at the centre. Fig. 3(a) itself shows 0.1.
   - Fig. 2(c): the bottom colourbar label reads "0.50"; it should read 0.05.
   - Fig. 3(d): the profile cannot be reproduced from the Case IV formula on either midline.
     - Along y = 0.5, D steps from 0.25 to 0.55 at x = 0.5.
     - Along x = 0.5, D steps from 0.20 to 0.40 at y = 0.35.
     - The figure shows a step from about 0.17 to about 0.35 near 0.3.
   - Fig. 3: the axes are labelled m²/s, but the cases are dimensionless.

4. **Error metric, eq. (7).** It is normalised by the variance of the true field, not by its norm, so it is stricter than relL2 for fields with a large mean. For Case I (D between 0.05 and 0.10), L = 4.8% corresponds to a much smaller relL2 (computed in the results). The paper's justification says the metric is unchanged by multiplying or adding a constant. That holds only if both fields are changed, i.e. a change of units.

5. **Noise robustness is asserted but not tested on data with a known answer.**
   - All numerical cases are noise-free.
   - The noisy cases are experiments without ground truth, checked only against literature ranges.

6. **Agarose result is physically implausible.** The paper reports D ≈ 9.0e-10 m²/s for Rhodamine 6G in 0.5% agarose. That is more than twice its own free-water value of 4.0–4.05e-10 m²/s (Sec. 3.2.1, 23 °C). Diffusion in a hydrogel is hindered, so D_gel ≤ D_water. A value 2.2× the free-solution value suggests an artefact: for example the moving dye front or the interface, or a time or length scaling error, rather than a property of the gel. The gelatin 5% value (3.7e-10) is about 0.9× water, which is plausible.

7. **Fig. 4(b).** The text gives 4.0e-10 for the water region, but the colourbar maximum is 3.9e-10. This is minor.

8. **Internal consistency.**
   - The Stokes–Einstein ratio D₂ = D₁·μ₁/μ₂ = 4.05e-10 × 0.935/1.60 = 2.37e-10 matches the stated 2.36e-10.
   - Table 1 Case I (4.8e-2) and Table 2 at 250,000 points (4.89e-2) agree.
