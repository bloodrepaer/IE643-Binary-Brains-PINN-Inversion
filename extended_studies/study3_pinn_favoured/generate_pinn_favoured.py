"""
generate_pinn_favoured.py
-------------------------
Benchmark families that test the "PINN should win when ..." hypotheses on
slide 17 of the prep deck (Deliverable 4). One family per hypothesis, each
with a matched control, written in exactly the same format as
datasets_v2_rk4_zero/ so adjoint_inversion.py and pinn_inversion.py read it
unchanged:

    H1  very sparse, scattered, off-grid sensors   (3 / 5 / 8 sensors, dense in time)
    H2  boundary condition unknown to the inversion (time-varying edge heating)
    H3  heat source unknown to the inversion        (a localised heater left out of the model)
    H4  heavy noise on a smooth D                   (eta 10 / 20 / 30 / 50 %)
    H5  long time horizons                          (Fo 0.2 / 0.5 / 1.0, top edge pulsed every Fo 0.05)

Smooth, low-frequency D (the deck's "D is smooth and low-frequency") is not a
family of its own: it is the field choice inside H1, H4 and H5 (tier G_lowfreq,
tier A_smooth, and E_grf_corrlen_long).

Design rules (so a PINN win, if it appears, can be attributed and is not rigged)
  * Same solver as the main benchmark: RK4 on 129^2, harmonic-mean 5-point
    operator, inversion grid 65^2 (no inverse crime), same IC.
  * Same noise convention: sigma = eta * RMS of the clean field at the 100-sensor
    random layout and uniform40 times, one sigma per field.
  * Paired controls. In H2 the "unknown" dataset and its "known" control have
    IDENTICAL sensor files; only problem/*.json differs (what the inversion is
    told). In H3 the source / no-source pair share sensor positions and the
    noise draw. Sensor positions are nested (3 in 5 in 8 in 50) and shared with
    the main benchmark's pool for the same base tier.
  * "Unknown" means unknown to BOTH methods. problem/*.json then carries
    "known": false and the value a method that needs one must ASSUME (zero
    edges, f = 0), so adjoint_inversion.py still runs (and is misspecified).
    The true BC / source is stored only in ground_truth/ for scoring.
    A stronger classical baseline would add the boundary values or the source
    as extra unknowns; the report should say the adjoint here does not.
  * The PINN needs code changes to EXPLOIT H2/H3 (see study3_pinn_favoured/README.md):
    drop the hard Dirichlet window and the BC loss when "known" is false, and
    use a learnable or robust (L1/Huber) residual when the source is unknown.

Run (from extended_studies/, ~5-10 min, deterministic):
    python run.py study3_pinn_favoured/generate_pinn_favoured.py     # -> datasets_v2_pinnwin/
"""
import csv, json, os, time
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import generate_dataset_v2 as G
from domain import Domain2D, get_D_field
from integrators import ForwardModel, BoundaryCondition, integrate

OUT = "datasets_v2_pinnwin"
N_DATA, N_INV = 129, 65
R = (N_DATA - 1) // (N_INV - 1)


# ----------------------------------------------------------------------------
# Fields
# ----------------------------------------------------------------------------
def D_G_lowfreq(dom):
    """Tier G: smooth, low-frequency. A gentle left-to-right gradient plus one
    half-wave in y. Range ~0.10-0.35, no feature smaller than the domain."""
    X, Y = dom.X, dom.Y
    return 0.10 + 0.25 * (0.6 * X + 0.4 * (0.5 + 0.5 * np.sin(np.pi * Y)))


def D_field(dom, tier):
    if tier == "G_lowfreq":
        return D_G_lowfreq(dom)
    cls, kw = G.TIERS[tier]
    return get_D_field(dom, cls, **kw)


# ----------------------------------------------------------------------------
# Forcing: boundary heating (amplitude dial) and an internal heater
# ----------------------------------------------------------------------------
def make_bc(name, T, amp, Fo=0.1):
    if name == "zero":
        return BoundaryCondition()
    if name == "heated_left":                     # left edge held at amp from t = 0
        return BoundaryCondition({"left": amp}, {"left": f"{amp:g}"})
    if name == "ramp_left":                       # left edge warmed 0 -> amp over the horizon
        return BoundaryCondition({"left": lambda t: amp * t / T}, {"left": f"{amp:g}*t/{T:.10g}"})
    if name == "pulsed_top":                      # three heating pulses on the top edge
        return BoundaryCondition({"top": lambda t: amp * 0.5 * (1 - np.cos(6 * np.pi * t / T))},
                                 {"top": f"{amp:g}*0.5*(1-cos(6*pi*t/{T:.10g}))"})
    if name == "periodic_top":                    # top edge pulsed with a FIXED period of Fo 0.05,
        P = 0.05 * T / Fo                         # so a longer horizon means more pulses, not slower ones
        return BoundaryCondition({"top": lambda t: amp * 0.5 * (1 - np.cos(2 * np.pi * t / P))},
                                 {"top": f"{amp:g}*0.5*(1-cos(2*pi*t/{P:.10g}))"})
    raise ValueError(name)


IC_HEAT = sum(b["amp"] * 2 * np.pi * b["s"] ** 2 for b in G.IC_BUMPS)   # integral of u0 (~0.0315)
SRC = dict(cx=0.62, cy=0.35, s=0.06)              # heater between the two IC bumps


def source_amplitude(q, T):
    """Heater strength so that the heat it injects over [0, T] is q x the IC's heat."""
    return q * IC_HEAT / (2 * np.pi * SRC["s"] ** 2 * T)


def source_expression(A):
    return f"{A:.8g}*exp(-((x-{SRC['cx']})**2+(y-{SRC['cy']})**2)/(2*{SRC['s']}**2))"


class ForwardModelSource(ForwardModel):
    """ForwardModel plus a time-constant source f(x, y) on interior nodes."""

    def __init__(self, domain, D, bc=None, A=0.0):
        super().__init__(domain, D, stencil="5pt", bc=bc)
        X, Y = domain.X, domain.Y
        f = A * np.exp(-((X - SRC["cx"]) ** 2 + (Y - SRC["cy"]) ** 2) / (2 * SRC["s"] ** 2))
        self.f_int = f[1:-1, 1:-1].ravel()

    def solve(self, u0_full, T, n_out, method="rk4", **kw):
        b = lambda t: self.b(t) + self.f_int
        t, U_I, info = integrate(self.L, b, u0_full[1:-1, 1:-1].ravel(), T, n_out, method, **kw)
        Nx, Ny = self.domain.shape
        U = np.stack([self.bc.field((Nx, Ny), tk) for tk in t])
        U[:, 1:-1, 1:-1] = U_I.reshape(n_out + 1, Nx - 2, Ny - 2)
        return t, U, info


# ----------------------------------------------------------------------------
# The plan: one row per dataset
# ----------------------------------------------------------------------------
def plan():
    J = []

    def add(family, hyp, case, tier, Fo=0.1, protocol="uniform8", n=50, layout="random", eta=0.05,
            bc=("zero", 0.0), bc_known=True, q=0.0, src_known=True, role="test", pair=None):
        J.append(dict(family=family, hypothesis=hyp, case=case, base_tier=tier, Fo=Fo, protocol=protocol,
                      n=n, layout=layout, eta=eta, bc=bc, bc_known=bc_known, q=q, src_known=src_known,
                      role=role, pair=pair or case))

    # H1 very sparse, scattered, off-grid sensors; dense in time (uniform40)
    h = "Sensors are very sparse, scattered and off-grid"
    for tier in ("A_smooth", "E_grf_corrlen_long", "G_lowfreq"):
        for n in (3, 5, 8):
            add("H1", h, f"H1-{tier}", tier, protocol="uniform40", n=n, pair=f"H1-{tier}")
        add("H1", h, f"H1-{tier}", tier, protocol="uniform40", n=50, role="control", pair=f"H1-{tier}")

    # H2 boundary condition unknown; control = same data, BC given
    h = "The boundary condition is unknown"
    cases = [("A_smooth", "pulsed_top", 0.1), ("A_smooth", "pulsed_top", 0.3), ("A_smooth", "pulsed_top", 1.0),
             ("A_smooth", "heated_left", 1.0), ("A_smooth", "ramp_left", 1.0),
             ("E_grf", "pulsed_top", 0.3), ("E_grf", "heated_left", 1.0), ("E_grf", "ramp_left", 1.0)]
    for tier, name, amp in cases:
        tag = f"{tier}-{name.replace('_', '')}-a{amp:g}"
        add("H2", h, f"H2u-{tag}", tier, protocol="uniform16", bc=(name, amp), bc_known=False, pair=tag)
        add("H2", h, f"H2k-{tag}", tier, protocol="uniform16", bc=(name, amp), bc_known=True,
            role="control", pair=tag)

    # H3 source unknown; control = q = 0 (no source, same sensors and noise draw)
    h = "The source is unknown"
    for tier in ("A_smooth", "E_grf"):
        for q in (0.1, 0.3, 1.0):
            add("H3", h, f"H3u-{tier}-q{q:g}", tier, protocol="uniform16", q=q, src_known=False, pair=f"H3-{tier}")
        add("H3", h, f"H3k-{tier}-q0", tier, protocol="uniform16", q=0.0, role="control", pair=f"H3-{tier}")

    # H4 heavy noise on smooth D (control eta 5%)
    h = "Noise is heavy (spectral bias helps)"
    for tier in ("A_smooth", "E_grf_corrlen_long", "G_lowfreq"):
        for eta in (0.05, 0.10, 0.20, 0.30, 0.50):
            add("H4", h, f"H4-{tier}", tier, protocol="uniform40", eta=eta,
                role="control" if eta == 0.05 else "test", pair=f"H4-{tier}")

    # H5 long horizons under periodic (known) heating that keeps the field transient; control Fo 0.2
    h = "Time horizons are long"
    for tier in ("A_smooth", "G_lowfreq"):
        for Fo in (0.2, 0.5, 1.0):
            add("H5", h, f"H5-{tier}-periodictop", tier, Fo=Fo, protocol="uniform200", bc=("periodic_top", 0.5),
                role="control" if Fo == 0.2 else "test", pair=f"H5-{tier}")
    return J


def key_of(j):
    return f"{j['case']}__Fo{j['Fo']:g}__{j['protocol']}__n{j['n']}__{j['layout']}__eta{j['eta']:g}"


def field_id(j):
    """Rows sharing a field_id share one forward solve (BC-known flag does not change the physics)."""
    return (j["base_tier"], j["Fo"], j["bc"], j["q"])


# ----------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------
def write_problem(out, name, dom, dom_inv, T, Fo, bc_true, bc_name, bc_known, A, src_known, t):
    """problem/ = what the inversion is TOLD. Unknown BC -> zero edges written as an
    assumption (IC boundary nodes too, so g(0) does not leak); unknown source -> f = 0 assumed."""
    told = bc_true if bc_known else BoundaryCondition()
    G.write_problem(out, name, dom, dom_inv, R, T, Fo, told, bc_name if bc_known else "unknown", t)
    p = os.path.join(out, "problem", name + ".json")
    spec = json.load(open(p))
    if not bc_known:
        spec["boundary_condition"].update(
            type="unknown", known=False, scenario="unknown",
            note="The true boundary values are NOT given. 'edges' holds the zero-Dirichlet value a "
                 "method that requires a BC must assume; a method that can leave the BC free should do so.")
    else:
        spec["boundary_condition"]["known"] = True
    if A > 0 and not src_known:
        spec["source_f"] = "unknown"
        spec["source_known"] = False
        spec["source_note"] = "A source is present but not given. Methods that need f must assume f = 0."
    else:
        spec["source_f"] = source_expression(A) if A > 0 else "0"
        spec["source_known"] = True
    json.dump(spec, open(p, "w"), indent=2)


def sample(dom, t, U, j, sig_ref):
    ks = G.PROTOCOLS[j["protocol"]]()
    xy = G.sensor_locations(j["base_tier"], j["n"], j["layout"])        # same pool as the main benchmark
    u_clean = np.concatenate([RegularGridInterpolator((dom.x, dom.y), U[k])(xy) for k in ks])
    m = len(xy)
    sigma = j["eta"] * sig_ref
    # one noise draw per (pair, Fo, protocol, n, layout, eta): H2 unknown/known share it exactly
    rng = np.random.default_rng(G.seed_of("noise-pinnwin", j["pair"], j["Fo"], j["protocol"], j["n"], j["layout"], j["eta"]))
    u = u_clean + sigma * rng.standard_normal(u_clean.shape)
    data = dict(x=np.tile(xy[:, 0], len(ks)), y=np.tile(xy[:, 1], len(ks)), t=np.repeat(t[ks], m), u=u,
                u_clean=u_clean, sigma=np.full_like(u_clean, sigma), sensor_id=np.tile(np.arange(m), len(ks)),
                time_index=np.repeat(ks, m))
    h = 1.0 / (N_INV - 1)
    return data, sigma, m, ks, float(np.min(np.hypot(*(xy / h - np.round(xy / h)).T)))


def main():
    t0 = time.time()
    for sub in ("problem", "ground_truth", "sensors"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    dom, dom_inv = Domain2D(Nx=N_DATA, Ny=N_DATA), Domain2D(Nx=N_INV, Ny=N_INV)
    jobs = plan()
    solved = {}
    manifest = []
    for j in jobs:
        fid = field_id(j)
        tier = j["base_tier"]
        D = D_field(dom, tier)
        D_mean = float(D.mean())
        T = j["Fo"] / D_mean
        bc_name, amp = j["bc"]
        bc = make_bc(bc_name, T, amp, j["Fo"])
        A = source_amplitude(j["q"], T)
        if fid not in solved:
            fm = ForwardModelSource(dom, D, bc=bc, A=A)
            u0 = G.initial_condition(dom, bc)
            t, U, info = fm.solve(u0, T, G.N_OUT, "rk4")
            ks40 = G.PROTOCOLS["uniform40"]()
            xy100 = G.sensor_locations(tier, 100, "random")
            ref = np.concatenate([RegularGridInterpolator((dom.x, dom.y), U[k])(xy100) for k in ks40])
            solved[fid] = (t, U, info, float(np.sqrt(np.mean(ref ** 2))))
            print(f"solve {tier:20s} Fo={j['Fo']:<4g} bc={bc_name}:{amp:g} q={j['q']:g}  "
                  f"{info['n_steps']} RK4 steps, {info['wall_s']:.0f}s", flush=True)
        t, U, info, sig_ref = solved[fid]

        key = key_of(j)
        prob = f"{j['case']}__Fo{j['Fo']:g}"
        if not os.path.exists(os.path.join(OUT, "problem", prob + ".json")):
            write_problem(OUT, prob, dom, dom_inv, T, j["Fo"], bc, bc_name, j["bc_known"], A, j["src_known"], t)
            bnd = np.array([[bc.at(tk)[e] for e in ("left", "right", "bottom", "top")] for tk in t])
            np.savez_compressed(
                os.path.join(OUT, "ground_truth", prob + ".npz"),
                D_data=D, D_inv=D[::R, ::R], x_data=dom.x, y_data=dom.y, x_inv=dom_inv.x, y_inv=dom_inv.y,
                t=t[::G.GT_EVERY], u=U[::G.GT_EVERY].astype(np.float32), T=T, Fo=j["Fo"], D_mean=D_mean,
                integrator="rk4", stencil="5pt", dt=info["dt"], n_steps=info["n_steps"],
                bc=f"{bc_name}:{amp:g}", bc_true_t=t, bc_true_edges=bnd, bc_true_edge_order="left,right,bottom,top",
                source_true=source_expression(A) if A > 0 else "0", source_q=j["q"], defects="[]")

        data, sigma, m, ks, min_off = sample(dom, t, U, j, sig_ref)
        np.savez_compressed(os.path.join(OUT, "sensors", key + ".npz"), **data)
        with open(os.path.join(OUT, "sensors", key + ".csv"), "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n"); w.writerow(["x", "y", "t", "u", "sigma"])
            for row in zip(data["x"], data["y"], data["t"], data["u"], data["sigma"]):
                w.writerow([f"{v:.8g}" for v in row])
        manifest.append(dict(
            key=key, family=j["family"], hypothesis=j["hypothesis"], role=j["role"], pair=j["pair"],
            tier=j["case"], base_tier=tier, Fo=j["Fo"], T=round(T, 6), D_mean=round(D_mean, 6),
            protocol=j["protocol"], n_times=len(ks), n_sensors=m, sensor_set=str(j["n"]), layout=j["layout"],
            n_readings=len(data["u"]), eta=j["eta"], sigma=sigma, sigma_ref=sig_ref,
            bc_true=f"{bc_name}:{amp:g}", bc_known=j["bc_known"], source_q=j["q"], source_known=j["src_known"],
            min_offgrid_over_h=round(min_off, 3),
            problem=prob, ground_truth=prob + ".npz", integrator="rk4", stencil="5pt", n_data=N_DATA, n_inv=N_INV,
            observation_times=[round(float(v), 8) for v in t[ks]],
            files={"sensors_csv": f"sensors/{key}.csv", "sensors_npz": f"sensors/{key}.npz",
                   "problem": f"problem/{prob}.json", "ground_truth": f"ground_truth/{prob}.npz"}))

    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1, default=float)
    cols = [k for k in manifest[0] if k not in ("observation_times", "files")]
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader(); w.writerows(manifest)
    print(f"\n{len(manifest)} datasets, {len(solved)} forward solves -> {OUT}/  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
