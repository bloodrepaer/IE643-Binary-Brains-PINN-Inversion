"""
generate_dataset_v2.py
----------------------
Synthetic benchmark generator for IE 643 (team Binary_Brains): PINN-based
inverse estimation of a spatially varying diffusion coefficient.

    du/dt = div( D(x,y) grad u ) + f,   f = 0,
    u = g_edge(t) on each edge (Dirichlet),   u(x,y,0) = two Gaussian bumps

Design decisions
  * Time integrator (--integrator): rk4 (default), cn, or fe.
      rk4 - explicit, 4th order; its stability limit forces a tiny step, so
            the time error is negligible (~1e-10). It is also a DIFFERENT
            scheme from the Crank-Nicolson solver the classical inversion
            uses, so the data avoid an "inverse crime" in time.
      cn  - the inversion/replay solver itself (Rannacher start). Generating
            data with it reintroduces the time part of the inverse crime;
            useful as a control.
      fe  - forward Euler, 1st order. Benchmark only: shows what a naive
            explicit scheme costs and how inaccurate it is.
  * Space: 5-point flux-form stencil with harmonic-mean link diffusivities.
    Data are generated on a fine grid (--n-data 129) and sensors are read off
    that fine solution; the inversion grid is 65^2. The grids are nested, so the
    ground-truth D on the inversion grid is an exact restriction of the data
    grid D (tier E stays one random-field realisation).
  * Initial and boundary conditions are written to problem/ -- everything an
    inversion is allowed to know (all but D and the full u field).
  * Boundary scenarios (--bc): zero, heated_left, ramp_left, pulsed_top.
  * Time grids are fixed and deterministic (no random draws).
  * Noise: one sigma per (tier, horizon, eta), sigma = eta * RMS of the clean
    field at a fixed reference (100 random sensors, uniform40 times). Changing
    the sensor count, layout or time protocol never changes the thermometer.
  * Seeds are deterministic (zlib.crc32, not Python's per-process hash()).
  * Sensor sets are nested for the random/clustered/boundary layouts: the
    n-sensor set is the first n points of one fixed 100-point pool.

Sweeps per canonical tier (duplicates across sweeps are written once)
  S1 spatial density x noise : sensors {5, 20, 50, 100, full field} x
                               eta {0, 1, 5, 10%}  (Fo 0.1, uniform8, random)
  S2 horizon x time protocol : Fo {0.02, 0.05, 0.1, 0.2} x
                               {uniform40, uniform8, log8, early8, late8, final1}
                               (50 random sensors, eta 5%)
  S3 temporal density x noise: uniform readouts every T/n_t,
                               n_t {4, 8, 16, 40, 80, 200} x eta {1, 5, 10%}
  S4 spatial layout          : {random, grid, clustered, boundary} x {20, 50}
                               sensors (Fo 0.1, uniform8, eta 5%)
  Tier F (uniform plate D0 = 0.25 with localised defects) is canonical too.
  Difficulty variants (contrast, checkerboard scale, correlation length;
  defect count, size, contrast and type) get the baseline point only: Fo 0.1, uniform8, 50 random sensors, eta 5%.

Folder layout (datasets_v2_<integrator>_<bc>/)
  problem/       IC on both grids, g(t) at every boundary node, JSON spec
  sensors/       noisy readings, long format  x,y,t,u,sigma  (+ .npz)
  ground_truth/  the answer key (D, u history) -- evaluation only
  manifest.json  every dataset: full config, seeds, explicit observation times
  manifest.csv   same, one row per dataset (no time lists)

Usage
  python generate_dataset_v2.py                     # full benchmark, rk4, zero BC
  python generate_dataset_v2.py --integrator cn     # same benchmark, CN data
  python generate_dataset_v2.py --bc ramp_left      # left edge heated as t/T
  python generate_dataset_v2.py --quick             # tier A only, small grid
"""
import argparse, csv, json, os, time, zlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from domain import Domain2D, get_D_field, defect_layout
from integrators import ForwardModel, BoundaryCondition, INTEGRATORS, EDGES, edge_masks

SEED_BASE = 20260908

# tag -> (domain.py class name, kwargs). First five are the canonical tiers.
TIERS = {
    "A_smooth":                 ("A_smooth", {}),
    "B_gaussian_bump":          ("B_gaussian_bump", {}),
    "C_inclusion":              ("C_inclusion", {}),
    "D_layered":                ("D_layered_checkerboard", {"pattern": "layered"}),
    "E_grf":                    ("E_gaussian_random_field", {"correlation_length": 0.1, "seed": SEED_BASE}),
    "C_inclusion_contrast_low": ("C_inclusion", {"contrast": 2.0}),
    "C_inclusion_contrast_high":("C_inclusion", {"contrast": 15.0}),
    "D_checkerboard_coarse":    ("D_layered_checkerboard", {"pattern": "checkerboard", "n_blocks": 2}),
    "D_checkerboard_fine":      ("D_layered_checkerboard", {"pattern": "checkerboard", "n_blocks": 6}),
    "E_grf_corrlen_short":      ("E_gaussian_random_field", {"correlation_length": 0.03, "seed": SEED_BASE}),
    "E_grf_corrlen_long":       ("E_gaussian_random_field", {"correlation_length": 0.3, "seed": SEED_BASE}),
    # Tier F: uniform plate (D0 = 0.25) with localised defects. Default: 3 spots
    # of radius 0.07, mixed (two conduct 3x worse, one 3x better).
    "F_defects":                ("F_defects", {}),
    "F_defects_single":         ("F_defects", {"n_defects": 1}),
    "F_defects_many":           ("F_defects", {"n_defects": 6, "radius": 0.05}),
    "F_defects_small":          ("F_defects", {"radius": 0.035}),
    "F_defects_low_contrast":   ("F_defects", {"contrast": 1.5}),
    "F_defects_voids":          ("F_defects", {"kind": "low", "contrast": 5.0}),
}
CANONICAL = ["A_smooth", "B_gaussian_bump", "C_inclusion", "D_layered", "E_grf", "F_defects"]

# Spatial sensor density (deck slide 15: 5 -> 20 -> 50 -> 100 -> full field).
# "full" = every interior node of the inversion grid (63^2 = 3969 points).
SENSOR_COUNTS = (5, 20, 50, 100, "full")
N_MAX_SENSORS = 100
# Spatial layouts. random/clustered/boundary are NESTED: the n-sensor set is
# the first n of one fixed 100-point pool, so going 5 -> 20 -> 50 -> 100 only
# ADDS sensors. grid is a regular lattice (not nested).
LAYOUTS = ("random", "grid", "clustered", "boundary")
LAYOUT_SWEEP_COUNTS = (20, 50)
NOISE_ETAS = (0.0, 0.01, 0.05, 0.10)
FOURIER_NUMBERS = (0.02, 0.05, 0.1, 0.2)
N_OUT = 400                    # solver output grid: t_k = k T / N_OUT
GT_EVERY = 10                  # ground-truth file keeps every 10th output (41 snapshots)
SENSOR_MARGIN = 0.05
BASE = dict(Fo=0.1, protocol="uniform8", n_sensors=50, layout="random", eta=0.05)
TIME_SWEEP_PROTOCOLS = ("uniform40", "uniform8", "log8", "early8", "late8", "final1")
# Noise reference: one "instrument" per (tier, horizon). sigma = eta * RMS of the
# clean field read by the 100-sensor layout at the uniform40 times. Every
# dataset of that (tier, horizon) uses this sigma, whatever its sensor count
# or time protocol, so changing WHEN or HOW OFTEN you measure never changes
# the thermometer.
SIGMA_REF = dict(n_sensors=100, layout="random", protocol="uniform40")


def _idx(fracs):
    """Fractions of T -> unique indices on the output grid (t=0 excluded)."""
    k = np.unique(np.clip(np.round(np.asarray(fracs) * N_OUT).astype(int), 1, N_OUT))
    return k

# Sampling protocols: which output times the sensors record (as fractions of T).
# Fixed, deterministic time grids -- no random draws. Every dataset of a given
# (tier, horizon, protocol) uses exactly the same times.
TIME_COUNTS = (4, 8, 16, 40, 80, 200)     # uniform spacing T/n_t; all divide N_OUT


def _uniform(n_t):
    return lambda: _idx(np.arange(1, n_t + 1) / n_t)          # t = T/n_t, 2T/n_t, ..., T


PROTOCOLS = {
    **{f"uniform{n}": _uniform(n) for n in TIME_COUNTS},      # fixed interval T/n
    "log8":     lambda: _idx(np.geomspace(0.01, 1.0, 8)),           # dense early, sparse late
    "early8":   lambda: _idx(np.linspace(1, 8, 8) / 8 * 0.25),      # only t in (0, T/4]
    "late8":    lambda: _idx(0.5 + np.linspace(0, 1, 8) * 0.5),     # only t in [T/2, T]
    "final1":   lambda: _idx([1.0]),                                 # single snapshot at T
}


def seed_of(*parts):
    """Deterministic across runs and machines (unlike Python's hash())."""
    return SEED_BASE + zlib.crc32("|".join(map(str, parts)).encode()) % 1_000_000


# Initial condition: sum of Gaussian bumps (same field as v1). Kept as data so
# the exact analytic form can be written to the problem spec.
IC_BUMPS = [dict(cx=0.30, cy=0.30, s=0.05, amp=1.0),
            dict(cx=0.70, cy=0.65, s=0.06, amp=0.7)]


def initial_condition(domain, bc=None):
    """IC on the grid. Boundary nodes are set to g(0) so u0 is consistent with
    the boundary condition at t = 0."""
    X, Y = domain.X, domain.Y
    u0 = sum(b["amp"] * np.exp(-((X - b["cx"]) ** 2 + (Y - b["cy"]) ** 2) / (2 * b["s"] ** 2))
             for b in IC_BUMPS)
    bc = bc or BoundaryCondition()
    g0 = bc.field(domain.shape, 0.0)
    u0[0, :], u0[-1, :], u0[:, 0], u0[:, -1] = g0[0, :], g0[-1, :], g0[:, 0], g0[:, -1]
    return u0


def ic_expression():
    return " + ".join(f"{b['amp']}*exp(-((x-{b['cx']})**2+(y-{b['cy']})**2)/(2*{b['s']}**2))"
                      for b in IC_BUMPS)


def make_bc(name, T):
    """Boundary scenarios. Values may depend on T so every horizon sees the
    same shape of forcing. Each edge gets a callable plus a sympy-parseable
    expression in t for the problem spec."""
    if name == "zero":
        return BoundaryCondition()
    if name == "heated_left":            # sudden heating at t=0 (sharp start)
        return BoundaryCondition({"left": 1.0}, {"left": "1"})
    if name == "ramp_left":              # heated smoothly, 0 -> 1 over the horizon
        return BoundaryCondition({"left": lambda t: t / T}, {"left": f"t/{T:.10g}"})
    if name == "pulsed_top":             # three heating pulses on the top edge
        return BoundaryCondition({"top": lambda t: 0.5 * (1 - np.cos(6 * np.pi * t / T))},
                                 {"top": f"0.5*(1-cos(6*pi*t/{T:.10g}))"})
    raise ValueError(name)

BC_SCENARIOS = ("zero", "heated_left", "ramp_left", "pulsed_top")


def write_problem(out, name, dom, dom_inv, r, T, Fo, bc, bc_name, t):
    """Everything the inversion may use: IC, BC history, domain, horizon."""
    u0_data = initial_condition(dom, bc)
    u0_inv = u0_data[::r, ::r]
    masks = edge_masks(dom_inv.shape)
    xy, lab = [], []
    for e in EDGES:
        i, j = np.nonzero(masks[e])
        xy.append(np.c_[dom_inv.x[i], dom_inv.y[j]]); lab += [e] * len(i)
    xy = np.vstack(xy)
    gvals = np.array([[bc.at(tk)[e] for e in lab] for tk in t])
    np.savez_compressed(os.path.join(out, "problem", name + ".npz"),
                        Lx=1.0, Ly=1.0, T=T, Fo=Fo, x_inv=dom_inv.x, y_inv=dom_inv.y, u0_inv=u0_inv,
                        x_data=dom.x, y_data=dom.y, u0_data=u0_data,
                        t_bc=t, bnd_xy=xy, bnd_edge=np.array(lab), bnd_u=gvals)
    X, Y = np.meshgrid(dom_inv.x, dom_inv.y, indexing="ij")
    np.savetxt(os.path.join(out, "problem", name + "__ic.csv"),
               np.c_[X.ravel(), Y.ravel(), u0_inv.ravel()], delimiter=",",
               header="x,y,u0", comments="", fmt="%.8g")
    spec = {
        "pde": "u_t = div(D(x,y) grad u) + f(x,y,t)",
        "unknown": "D(x,y) > 0, time-independent",
        "source_f": "0",
        "domain": {"Lx": 1.0, "Ly": 1.0, "T": T, "Fourier_number": Fo},
        "initial_condition": {"expression": ic_expression(),
                              "note": "boundary nodes of u0 are set to g(0)"},
        "boundary_condition": {"type": "dirichlet", "scenario": bc_name,
                               "edges": {e: bc.describe[e] for e in EDGES},
                               "edge_location": {"left": "x=0", "right": "x=Lx", "bottom": "y=0", "top": "y=Ly"},
                               "corner_rule": "corners take the left/right edge value"},
        "arrays": f"{name}.npz: u0 on both grids; g(t) at every boundary node of the "
                  f"inversion grid at t_bc (the {len(t)} solver output times)",
    }
    with open(os.path.join(out, "problem", name + ".json"), "w") as f:
        json.dump(spec, f, indent=2)


def _pool(tier, layout):
    """Fixed 100-point pool per (tier, layout); n-sensor sets are its first n."""
    rng = np.random.default_rng(seed_of("locations", tier, layout))
    lo, hi, N = SENSOR_MARGIN, 1 - SENSOR_MARGIN, N_MAX_SENSORS
    if layout == "random":
        return rng.uniform(lo, hi, (N, 2))
    if layout == "clustered":              # three hot-spots, assigned round-robin
        centres = rng.uniform(0.2, 0.8, (3, 2))
        return np.clip(centres[np.arange(N) % 3] + 0.07 * rng.standard_normal((N, 2)), lo, hi)
    if layout == "boundary":               # within 0.10 of an edge (inside the margin)
        edge = rng.integers(0, 4, N)
        d, s = rng.uniform(lo, lo + 0.10, N), rng.uniform(lo, hi, N)
        x = np.where(edge == 0, d, np.where(edge == 1, 1 - d, s))
        y = np.where(edge == 2, d, np.where(edge == 3, 1 - d, s))
        return np.c_[x, y]
    raise ValueError(layout)


def _factor(n):
    """n = nx * ny with nx as close to sqrt(n) as possible (20 -> 5x4, 50 -> 10x5)."""
    nx = int(np.ceil(np.sqrt(n)))
    while n % nx:
        nx += 1
    return nx, n // nx


def sensor_locations(tier, n, layout="random", dom_inv=None):
    if n == "full":                        # every interior node of the inversion grid
        return np.c_[dom_inv.X[1:-1, 1:-1].ravel(), dom_inv.Y[1:-1, 1:-1].ravel()]
    if layout == "grid":
        nx, ny = _factor(n)
        lo, hi = SENSOR_MARGIN, 1 - SENSOR_MARGIN
        gx = lo + (np.arange(nx) + 0.5) / nx * (hi - lo)
        gy = lo + (np.arange(ny) + 0.5) / ny * (hi - lo)
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        return np.c_[X.ravel(), Y.ravel()]
    return _pool(tier, layout)[:n]


def read_clean(dom, U, tier, protocol, n, layout, dom_inv):
    """Clean field at the sensor layout and the protocol's times."""
    ks = PROTOCOLS[protocol]()
    xy = sensor_locations(tier, n, layout, dom_inv)
    return ks, xy, np.concatenate([RegularGridInterpolator((dom.x, dom.y), U[k])(xy) for k in ks])


def sigma_reference(dom, U, tier):
    _, _, u = read_clean(dom, U, tier, SIGMA_REF["protocol"], SIGMA_REF["n_sensors"],
                         SIGMA_REF["layout"], None)
    return float(np.sqrt(np.mean(u ** 2)))


def sample(dom, dom_inv, t, U, tier, Fo, protocol, n, layout, eta, sig_ref):
    """Interpolate the clean fine-grid field at sensor locations x protocol times,
    add noise with sigma = eta * sig_ref. Long format (one row per reading)."""
    ks, xy, u_clean = read_clean(dom, U, tier, protocol, n, layout, dom_inv)
    m = len(xy)
    xs = np.tile(xy[:, 0], len(ks)); ys = np.tile(xy[:, 1], len(ks))
    ts = np.repeat(t[ks], m); kidx = np.repeat(ks, m); sid = np.tile(np.arange(m), len(ks))
    sigma = eta * sig_ref
    rng = np.random.default_rng(seed_of("noise", tier, Fo, protocol, n, layout, eta))
    u_noisy = u_clean + sigma * rng.standard_normal(u_clean.shape)
    return dict(x=xs, y=ys, t=ts, u=u_noisy, u_clean=u_clean, sigma=np.full_like(u_clean, sigma),
                sensor_id=sid, time_index=kidx), sigma, m


def plan(tiers):
    """List of (tier, Fo, protocol, n_sensors, layout, eta), duplicates removed."""
    B = BASE
    jobs = []
    for tier in tiers:
        if tier in CANONICAL:
            # S1 spatial density x noise
            for n in SENSOR_COUNTS:
                for eta in NOISE_ETAS:
                    jobs.append((tier, B["Fo"], B["protocol"], n, "nodes" if n == "full" else "random", eta))
            # S2 horizon x time protocol
            for Fo in FOURIER_NUMBERS:
                for p in TIME_SWEEP_PROTOCOLS:
                    jobs.append((tier, Fo, p, B["n_sensors"], B["layout"], B["eta"]))
            # S3 temporal density (fixed interval T/n_t) x noise
            for n_t in TIME_COUNTS:
                for eta in NOISE_ETAS[1:]:
                    jobs.append((tier, B["Fo"], f"uniform{n_t}", B["n_sensors"], B["layout"], eta))
            # S4 spatial layout
            for layout in LAYOUTS:
                for n in LAYOUT_SWEEP_COUNTS:
                    jobs.append((tier, B["Fo"], B["protocol"], n, layout, B["eta"]))
        else:
            jobs.append((tier, B["Fo"], B["protocol"], B["n_sensors"], B["layout"], B["eta"]))
    return list(dict.fromkeys(jobs))


def key_of(tier, Fo, protocol, n, layout, eta):
    return f"{tier}__Fo{Fo:g}__{protocol}__n{n}__{layout}__eta{eta:g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--integrator", default="rk4", choices=INTEGRATORS)
    ap.add_argument("--bc", default="zero", choices=BC_SCENARIOS, help="boundary scenario")
    ap.add_argument("--n-data", type=int, default=129, help="data-generation grid (nodes per side)")
    ap.add_argument("--n-inv", type=int, default=65, help="inversion grid; (n_data-1) must be a multiple of (n_inv-1)")
    ap.add_argument("--tiers", nargs="*", default=list(TIERS))
    ap.add_argument("--out", default=None, help="output folder (default datasets_v2_<integrator>)")
    ap.add_argument("--quick", action="store_true", help="tier A only on a 65^2 data grid")
    a = ap.parse_args()
    if a.quick:
        a.tiers, a.n_data, a.n_inv = ["A_smooth"], 65, 33
    if (a.n_data - 1) % (a.n_inv - 1):
        raise SystemExit("n_data-1 must be a multiple of n_inv-1 so the grids are nested")
    r = (a.n_data - 1) // (a.n_inv - 1)
    out = a.out or f"datasets_v2_{a.integrator}_{a.bc}"
    for sub in ("problem", "ground_truth", "sensors", "figures"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    dom = Domain2D(Nx=a.n_data, Ny=a.n_data)
    dom_inv = Domain2D(Nx=a.n_inv, Ny=a.n_inv)
    jobs = plan(a.tiers)
    manifest, obs_times = [], {}
    t_start = time.time()

    for tier in a.tiers:
        cls, kw = TIERS[tier]
        D = get_D_field(dom, cls, **kw)
        D_mean = float(D.mean())
        decay = {}
        for Fo in sorted({j[1] for j in jobs if j[0] == tier}):
            T = Fo / D_mean                      # L = 1
            bc = make_bc(a.bc, T)
            fm = ForwardModel(dom, D, stencil="5pt", bc=bc)
            u0 = initial_condition(dom, bc)
            t, U, info = fm.solve(u0, T, N_OUT, a.integrator)
            print(f"[{tier:26s}] Fo={Fo:<5g} T={T:.3f}  {a.integrator}: "
                  f"{info['n_steps']} steps, dt={info['dt']:.2e}, {info['wall_s']:.1f}s")
            decay[Fo] = (t / T, np.sqrt((U ** 2).mean(axis=(1, 2))))
            gt_name = f"{tier}__Fo{Fo:g}.npz"
            write_problem(out, f"{tier}__Fo{Fo:g}", dom, dom_inv, r, T, Fo, bc, a.bc, t)
            sig_ref = sigma_reference(dom, U, tier)
            np.savez_compressed(
                os.path.join(out, "ground_truth", gt_name),
                D_data=D, D_inv=D[::r, ::r], x_data=dom.x, y_data=dom.y, x_inv=dom_inv.x, y_inv=dom_inv.y,
                t=t[::GT_EVERY], u=U[::GT_EVERY].astype(np.float32), T=T, Fo=Fo, D_mean=D_mean,
                integrator=a.integrator, stencil="5pt", dt=info["dt"], n_steps=info["n_steps"],
                bc=a.bc, defects=json.dumps(defect_layout(**kw) if cls == "F_defects" else []))

            for (tg, F, p, n, layout, eta) in jobs:
                if tg != tier or F != Fo:
                    continue
                data, sigma, m = sample(dom, dom_inv, t, U, tier, Fo, p, n, layout, eta, sig_ref)
                key = key_of(tier, Fo, p, n, layout, eta)
                np.savez_compressed(os.path.join(out, "sensors", key + ".npz"), **data)
                with open(os.path.join(out, "sensors", key + ".csv"), "w", newline="") as f:
                    w = csv.writer(f, lineterminator="\n"); w.writerow(["x", "y", "t", "u", "sigma"])
                    for row in zip(data["x"], data["y"], data["t"], data["u"], data["sigma"]):
                        w.writerow([f"{v:.8g}" for v in row])
                obs_times[key] = [round(float(v), 8) for v in t[PROTOCOLS[p]()]]
                manifest.append(dict(
                    key=key, tier=tier, d_class=cls, d_kwargs=json.dumps(kw), Fo=Fo, T=round(T, 6),
                    D_mean=round(D_mean, 6), protocol=p, n_times=len(PROTOCOLS[p]()), n_sensors=m,
                    sensor_set=str(n), layout=layout,
                    n_readings=len(data["u"]), eta=eta, sigma=sigma, sigma_ref=sig_ref,
                    readout_interval=(T / len(PROTOCOLS[p]()) if p.startswith("uniform") else ""),
                    snr_db=(np.inf if sigma == 0 else 20 * np.log10(1 / eta)),
                    seed_locations=("" if n == "full" or layout == "grid" else seed_of("locations", tier, layout)),
                    seed_noise=seed_of("noise", tier, Fo, p, n, layout, eta),
                    bc=a.bc, problem=f"{tier}__Fo{Fo:g}", ground_truth=gt_name, integrator=a.integrator, stencil="5pt",
                    n_data=a.n_data, n_inv=a.n_inv))

        # overview figure: D with the 40-sensor layout, and signal decay per horizon
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        im = ax[0].imshow(D.T, origin="lower", extent=[0, 1, 0, 1], cmap="plasma")
        xy = sensor_locations(tier, BASE["n_sensors"], BASE["layout"])
        ax[0].plot(xy[:, 0], xy[:, 1], "wo", ms=4, mec="k")
        ax[0].set_title(f"{tier}: D(x,y) with {BASE['n_sensors']} sensors"); plt.colorbar(im, ax=ax[0])
        if cls == "F_defects":
            for dd in defect_layout(**kw):
                ax[0].add_patch(plt.Circle((dd["cx"], dd["cy"]), dd["r"], fill=False, ec="w", ls="--", lw=1))
        for Fo, (tau, rms) in decay.items():
            ax[1].semilogy(tau, rms, label=f"Fo={Fo:g}")
        ax[1].set(xlabel="t / T", ylabel="RMS of u over the grid", title="Signal decay per horizon")
        ax[1].grid(True, which="both", alpha=0.3); ax[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(out, "figures", f"{tier}_overview.png"), dpi=120)
        plt.close(fig)

    # sensor layout figure (first tier's seeds)
    tier0 = a.tiers[0]
    fig, ax = plt.subplots(1, len(LAYOUTS) + 1, figsize=(4 * (len(LAYOUTS) + 1), 4))
    for i, lay in enumerate(LAYOUTS):
        for n, mk in ((50, "o"), (20, "s")):
            xy = sensor_locations(tier0, n, lay)
            ax[i].plot(xy[:, 0], xy[:, 1], mk, ms=4 if n == 50 else 6, mfc="none" if n == 50 else "C3",
                       mec="C0" if n == 50 else "C3", label=f"n={n}")
        ax[i].set_title(lay)
    for n, c in zip((5, 20, 50, 100), ("C3", "C1", "C2", "C0")):
        xy = sensor_locations(tier0, n, "random")
        ax[-1].plot(xy[:, 0], xy[:, 1], "o", ms=3 + 6 * (n == 5) + 3 * (n == 20), color=c, label=f"n={n}", zorder=5 - n / 50)
    ax[-1].set_title("random, nested counts")
    for x in ax:
        x.set_xlim(0, 1); x.set_ylim(0, 1); x.set_aspect("equal"); x.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "figures", "sensor_layouts.png"), dpi=110)
    plt.close(fig)

    # protocol diagram
    fig, ax = plt.subplots(figsize=(8, 3))
    for i, (p, f) in enumerate(PROTOCOLS.items()):
        ax.plot(f() / N_OUT, np.full(len(f()), i), "|", ms=14, mew=2)
    ax.set_yticks(range(len(PROTOCOLS))); ax.set_yticklabels(list(PROTOCOLS))
    ax.set_xlim(0, 1.02); ax.set_xlabel("t / T"); ax.set_title("Time-sampling protocols")
    fig.tight_layout(); fig.savefig(os.path.join(out, "figures", "time_protocols.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]), lineterminator="\n"); w.writeheader(); w.writerows(manifest)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump([{**{k: v for k, v in m.items()}, "d_kwargs": json.loads(m["d_kwargs"]),
                    "observation_times": obs_times[m["key"]],
                    "files": {"sensors_csv": f"sensors/{m['key']}.csv",
                              "sensors_npz": f"sensors/{m['key']}.npz",
                              "problem": f"problem/{m['problem']}.json",
                              "ground_truth": f"ground_truth/{m['ground_truth']}"}}
                   for m in manifest], f, indent=1, default=float)
    config = dict(integrator=a.integrator, stencil="5pt", sensor_counts=SENSOR_COUNTS, layouts=LAYOUTS, bc=a.bc, ic_bumps=IC_BUMPS, n_data=a.n_data, n_inv=a.n_inv,
                  seed_base=SEED_BASE, noise_etas=NOISE_ETAS,
                  fourier_numbers=FOURIER_NUMBERS, protocols={p: (f() / N_OUT).round(4).tolist()
                                                              for p, f in PROTOCOLS.items()},
                  baseline=BASE, n_out=N_OUT, time_counts=TIME_COUNTS, sigma_reference=SIGMA_REF, tiers={k: TIERS[k] for k in a.tiers})
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n{len(manifest)} datasets, {len({m['ground_truth'] for m in manifest})} ground-truth "
          f"files -> {out}/  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
