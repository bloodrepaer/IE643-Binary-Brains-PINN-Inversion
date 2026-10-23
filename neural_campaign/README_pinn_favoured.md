# datasets_v2_pinnwin — where the PINN should win (Deliverable 4)

57 datasets testing the five "PINN should win when …" hypotheses on slide 17 of the prep deck.
Each family has a matched control, so a win (or a loss) can be attributed to the one thing that changed.

Generate: `python generate_pinn_favoured.py` (from `v2/`, about 5 min, deterministic).
Check: `python validate_pinn_favoured.py` (writes `validation.csv`).

Format is identical to `datasets_v2_rk4_zero/`: `sensors/`, `problem/`, `ground_truth/`, `manifest.json`.
`adjoint_inversion.py --data datasets_v2_pinnwin --all` and `pinn_inversion.py --data-root datasets_v2_pinnwin`
read it unchanged. The manifest adds `family`, `hypothesis`, `role` (test / control), `pair`,
`bc_true`, `bc_known`, `source_q`, `source_known`.

## The families

| Family | Slide-17 hypothesis | What changes | Control |
| --- | --- | --- | --- |
| H1 | Sensors very sparse, scattered, off-grid | 3 / 5 / 8 random off-grid sensors, 40 time levels | same field, 50 sensors |
| H2 | Boundary condition unknown | left edge heated (step or ramp) or top edge pulsed; amplitude 0.1 / 0.3 / 1.0 of the IC peak; **BC not given** | identical sensor files, BC given |
| H3 | Source unknown | a Gaussian heater at (0.62, 0.35), r = 0.06, injecting 0.1 / 0.3 / 1.0 × the IC's heat over the horizon; **source not given** | same sensors and noise draw, no heater |
| H4 | Heavy noise, smooth D | η = 10 / 20 / 30 / 50 % | η = 5 % |
| H5 | Long time horizons | Fo = 0.5 / 1.0, 200 time levels, top edge pulsed every Fo 0.05 (known), so longer = more pulses | Fo = 0.2 |

Fields: `A_smooth`, `E_grf` and `E_grf_corrlen_long` from the main benchmark, plus a new tier
`G_lowfreq` (D = 0.10 + 0.25·(0.6x + 0.4·(0.5 + 0.5 sin πy)), range 0.10–0.35). "D smooth and
low-frequency" is covered by these field choices inside H1, H4 and H5.

## Fairness rules

- Same solver, grids, IC and noise convention as the main benchmark (RK4 on 129², inversion on 65²,
  σ = η × RMS of the clean field at 100 random sensors × uniform40).
- "Unknown" means unknown to **both** methods. `problem/*.json` then has `"known": false` and the
  value a method that needs one must assume (zero edges; f = 0). The true BC and source are only in
  `ground_truth/` (`bc_true_edges`, `source_true`), for scoring.
- In an unknown-BC problem file the IC's boundary nodes are set to the assumed value, so g(0) does not leak.
- The adjoint as written cannot leave the BC or source free, so it runs **misspecified** on H2u/H3u.
  A stronger classical baseline would add the boundary values or the source as extra unknowns. The report
  should say this one does not, so any PINN win on H2/H3 is a win over the standard formulation.
- `validation.csv` gives `truth_replay_over_sigma`: the TRUE D run through the model each inversion is told.
  Controls sit near 1. On unknown-BC/source datasets it is well above 1. That gap is the floor the adjoint
  cannot get under with a correct D.

## Boundary conditions in the PINN: `pinn_inversion_bc.py`

`pinn_inversion.py` builds u_θ = 0 on every wall into the network (`bc_win`, `edge_taper`) and adds a zero-BC loss.
That is right for the main benchmark but cannot represent a heated wall. `pinn_inversion_bc.py` is a copy that reads
the BC mode from `problem/*.json`; nothing else changes:

| Mode | When | What the PINN does |
| --- | --- | --- |
| `dirichlet` | all edges 0, known | unchanged: hard zero window + zero-BC loss |
| `dirichlet_soft` | edges non-zero, known (H2k, H5) | no hard window; L_BC = mean (u_θ(wall, t) − g(t))², g from the problem file |
| `free` | `"known": false` (H2u) | no hard window, no BC loss: u is free on the walls, pinned only by the data |

Run: `python run_pinn_pinnwin.py` (H1/H3/H4 with the original file) and
`python run_pinn_pinnwin.py --families H2 H5 --module pinn_inversion_bc`.

Still open for H3: when `source_known` is false, add a small `f_net(x, y)` with an L1 penalty, or a robust (Huber/L1)
residual, so an unexplained local source does not drag D everywhere. Until then the PINN assumes f = 0 like the adjoint.

Scoring: use relL2(D) and RMS_replay/σ through the CN solver **under the true BC** (for H2u, the paired H2k problem
file holds it). Replay under the told model is not a fair score when the told model is wrong; on H3u the adjoint
replays at 0.99σ with relL2 3.1.

## Notes

- Long horizons (H5): the top edge is pulsed with a fixed period (Fo 0.05), so the field stays transient; a steady
  heated wall would reach steady state by Fo ≈ 0.2 and later readings would add nothing. At 65² the adjoint stores
  at most 800 states × 3,969 nodes ≈ 25 MB, so memory never binds here; H5 tests accuracy over long horizons only.
- Sensor positions come from the same nested 100-point pool as the main benchmark (3 ⊂ 5 ⊂ 8 ⊂ 50).
  `min_offgrid_over_h` in the manifest is the closest any sensor comes to a 65² node, in grid spacings.
