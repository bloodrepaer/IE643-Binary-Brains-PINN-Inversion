"""
novelty_core.py -- backend for the novelty-phase methods in the interface.

Everything runs on the SAME dataset folder as the classical adjoint and the PINN in core.py (same readings,
same 65 x 65 grid, same initial and boundary conditions, same source), so every method is comparable:

  combined PINN -> adjoint   hybrid.run_warm(..., "balance"): the PINN's log D, projected onto the adjoint's
                             16 x 16 controls, starts the CN adjoint; the alpha path starts where the PINN
                             estimate's misfit and regulariser balance (E1 "hybrid_balance")
  joint + source             hybrid_src.SourceSolver: the CN adjoint estimates D AND a heat source f >= 0 (E1s)
  joint + walls              hybrid_bc.WallSolver: the CN adjoint estimates D AND the wall temperatures (E1b)
  backward Euler adjoint     time_integrators.BESolver: the same adjoint with backward-Euler steps (Study 5)
  boundary-aware PINN        pinn_inversion_bc (Study 3): the PINN learns the wall values instead of assuming them

Scoring: relL2(D) and RMS of the log-D error against the truth (when known), and replay RMS / sigma: the estimate
is re-simulated with the physics the method itself assumed or estimated (its own source / walls) and compared
with the readings. A method can replay at ~1 sigma and still have a wrong D -- the error map shows it.

The demo bundle (demo/) holds slim copies of the 34 novelty datasets and every method's precomputed estimate
(built by extended_studies/novelty/build_app_bundle.py), so the full comparison can be shown without waiting.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent
METHODS = APP / "methods"
if str(METHODS) not in sys.path:
    sys.path.append(str(METHODS))           # after v2/: adjoint_inversion, integrators come from the project

import core                                  # noqa: E402  (sets up v2/ on the path)
import adjoint_inversion as A                # noqa: E402

DEMO = APP / "demo"
DEMO_GROUPS = {"benchmark": "Known physics (20 benchmark datasets)",
               "hidden_source": "Hidden heat source (6 datasets)",
               "hidden_walls": "Hidden wall temperatures (8 datasets)"}


# =============================================================================
# demo datasets and precomputed results
# =============================================================================
def demo_available():
    return (DEMO / "benchmark" / "datasets_v2_rk4_zero" / "manifest.json").exists()


def _demo_root(group):
    return DEMO / ("benchmark" if group == "benchmark" else "hidden_physics") / "datasets_v2_rk4_zero"


def demo_keys(group):
    man = json.load(open(_demo_root(group) / "manifest.json"))
    keys = [e["key"] for e in man]
    if group == "hidden_source":
        keys = [k for k in keys if k.startswith("H3u")]
    elif group == "hidden_walls":
        keys = [k for k in keys if k.startswith("H2u")]
    return keys


def load_demo(group, key):
    root = _demo_root(group)
    entry = next(e for e in json.load(open(root / "manifest.json")) if e["key"] == key)
    return dict(root=str(root), entry=entry, key=key, source="demo", group=group)


def physics_of(data):
    """'standard', 'hidden_source' or 'hidden_walls' (what the dataset's problem file does NOT tell you)."""
    k = data["key"]
    if k.startswith("H3u"):
        return "hidden_source"
    if k.startswith("H2u"):
        return "hidden_walls"
    return "standard"


EXTRA_ARMS = ("hybrid_path", "hybrid_prior")      # the other couplings of E1, shown on request


def demo_results(data, include_extra=False):
    """Every method's precomputed estimate on a demo dataset, re-scored live with the same scorer."""
    d = DEMO / "results" / data["key"]
    if not (d / "methods.json").exists():
        return {}
    out = {}
    for row in json.load(open(d / "methods.json")):
        if row["arm"] in EXTRA_ARMS and not include_extra:
            continue
        z = np.load(d / row["file"])
        D = z["D_hat"].astype(float)
        f = z["f_hat"].astype(float) if "f_hat" in z.files else None
        G_bd = z["G_bd"].astype(float) if "G_bd" in z.files else None
        if row["arm"] == "adjoint_truesrc":
            f = true_source(data)
        if row["arm"] == "adjoint_truebc":
            G_bd = true_walls_bd(data)
        m = score(data, D, f=f, G_bd=G_bd)
        m.update(method=row["label"], wall_s=row.get("wall_s") or float("nan"), precomputed=True)
        out[row["label"]] = dict(D=D, metrics=m, f=f, G_bd=G_bd, model_file=None, precomputed=True)
    return out


# =============================================================================
# truth (scoring only)
# =============================================================================
def true_source(data):
    import hybrid_src as HS
    gt = data["entry"]["files"].get("ground_truth")
    if not gt or not (Path(data["root"]) / gt).exists():
        return None
    f = HS.true_source_from_file(Path(data["root"]) / gt, core.N_INV)
    return None if f is None else f.reshape(core.N_INV, core.N_INV)


def true_walls_bd(data):
    import hybrid_bc as HB
    gt = data["entry"]["files"].get("ground_truth")
    if not gt:
        return None
    z = np.load(Path(data["root"]) / gt)
    if "bc_true_edges" not in z.files:
        return None
    s = HB.WallSolver(core.dataset(data))
    return HB.true_G(s, data["root"], data["entry"])[:, s.disc.Bd]


# =============================================================================
# scoring with the method's own physics
# =============================================================================
def score(data, D, f=None, G_bd=None):
    """core.score (replay with the problem file's physics) plus, for joint estimates, replay with the
    estimated source or walls (the physics the method itself inferred)."""
    m, _, _ = core.score(data, D)
    ds = core.dataset(data)
    m["rms_replay_over_sigma_problem_physics"] = m.get("rms_replay_over_sigma")
    if f is not None:
        import hybrid_src as HS
        s = HS.SourceSolver(ds)
        s.set_f(np.asarray(f, float).ravel())
        W, *_ = s.forward(np.asarray(D, float).ravel())
        rms = float(np.sqrt(np.mean((s.predict(W) - s.d) ** 2)))
        m.update(rms_replay=rms, rms_replay_over_sigma=rms / s.sigma_w, replay_physics="own source")
    elif G_bd is not None:
        import hybrid_bc as HB
        s = HB.WallSolver(ds)
        G = np.zeros((s.disc.S, ds.N * ds.N))
        G[:, s.disc.Bd] = G_bd
        s.set_G(G)
        W = s.forward(np.asarray(D, float).ravel())[0]
        rms = float(np.sqrt(np.mean((s.predict(W) - s.d) ** 2)))
        m.update(rms_replay=rms, rms_replay_over_sigma=rms / s.sigma_w, replay_physics="own walls")
    else:
        m["replay_physics"] = "problem file"
    tf = true_source(data) if f is not None else None
    if tf is not None and np.linalg.norm(tf) > 0:
        m["source_rel_L2"] = float(np.linalg.norm(np.asarray(f).ravel() - tf.ravel()) / np.linalg.norm(tf))
    return m


def _pack(data, D, f=None, G_bd=None, **extra):
    m = score(data, D, f=f, G_bd=G_bd)
    m.update(extra)
    return m


# =============================================================================
# methods
# =============================================================================
def run_classical_solver(data, solver="CN", n_steps=None, outdir=None):
    """The project's classical adjoint (16 x 16, Tikhonov, discrepancy principle) with CN or backward Euler."""
    ds = core.dataset(data)
    t0 = time.perf_counter()
    if solver == "CN":
        s = A.AdjointSolver(ds, n_steps=n_steps)
    else:
        import time_integrators as TI
        s = TI.BESolver(ds, n_steps or A.AdjointSolver._pick_steps(ds))
    out = s.invert(verbose=False)
    wall = time.perf_counter() - t0
    D = out["D"].reshape(ds.N, ds.N)
    m = _pack(data, D, method=f"adjoint ({solver}, {s.disc.n_steps} steps)", alpha=out["alpha"], flag=out["flag"],
              wall_s=wall, n_steps=s.disc.n_steps)
    model = _save(outdir, f"classical_{solver}_model.npz", kind=f"classical_adjoint_{solver}", theta=out["theta"],
                  control=16, reg="tikhonov", alpha=out["alpha"], D_hat=D, n_steps=s.disc.n_steps)
    return dict(D=D, metrics=m, sensitivity=A.sensitivity_map(s, out["W"]) if solver == "CN" else None,
                path=out["path"], model_file=model)


def run_combined(data, D_pinn, pinn_wall_s=0.0, outdir=None):
    """Combined method: PINN estimate -> projected onto the controls -> CN adjoint (hybrid_balance)."""
    import hybrid as H
    ds = core.dataset(data)
    out = H.run_warm(ds, D_pinn, "balance")
    D = np.asarray(out["D"]).reshape(ds.N, ds.N)
    m = _pack(data, D, method="combined PINN→adjoint", alpha=out["alpha"], flag=out["flag"],
              adjoint_wall_s=out["wall_s"], pinn_wall_s=pinn_wall_s, wall_s=out["wall_s"] + pinn_wall_s,
              alpha_start=out.get("alpha_start"))
    model = _save(outdir, "combined_model.npz", kind="combined_pinn_adjoint", D_hat=D,
                  D_pinn_projected=np.asarray(out["D_projected"]).reshape(ds.N, ds.N), alpha=out["alpha"])
    return dict(D=D, metrics=m, path=out["path"], model_file=model)


def run_joint_source(data, positive=True, outdir=None):
    """Joint classical inversion: D and a heat source f (>= 0) on 16 x 16 controls, exact CN adjoint."""
    import hybrid_src as HS
    ds = core.dataset(data)
    s = HS.SourceSolver(ds)
    r = HS.invert_src(s, optimise_source=True, positive=positive)
    D = r["D"].reshape(ds.N, ds.N)
    f = r["f"].reshape(ds.N, ds.N)
    m = _pack(data, D, f=f, method="joint: adjoint + source", alpha=r["alpha"], flag=r["flag"], wall_s=r["wall_s"])
    model = _save(outdir, "joint_source_model.npz", kind="joint_adjoint_source", D_hat=D, f_hat=f, alpha=r["alpha"])
    return dict(D=D, f=f, metrics=m, path=r["path"], model_file=model)


def run_joint_walls(data, G_start_bd=None, outdir=None):
    """Joint classical inversion: D and the wall temperatures (each wall uniform, piecewise linear in time).
    G_start_bd: optional starting walls (e.g. the boundary-aware PINN's), else zero walls."""
    import hybrid_bc as HB
    ds = core.dataset(data)
    s = HB.WallSolver(ds, n_s=1)
    t0 = time.perf_counter()
    if G_start_bd is None:
        b0 = np.zeros(s.n_b)
    else:
        G = np.zeros((s.disc.S, ds.N * ds.N)); G[:, s.disc.Bd] = G_start_bd
        b0 = s.b_of_G(G)
    s.set_G(s.G_of_b(b0))
    theta0 = np.full(s.K ** 2, s.best_constant())
    r = HB.invert_joint(s, theta0, b0)
    D = r["D"].reshape(ds.N, ds.N)
    G_bd = r["G"][:, s.disc.Bd]
    m = _pack(data, D, G_bd=G_bd, method="joint: adjoint + walls", alpha=r["alpha"], flag=r["flag"],
              wall_s=time.perf_counter() - t0)
    model = _save(outdir, "joint_walls_model.npz", kind="joint_adjoint_walls", D_hat=D, G_bd=G_bd.astype(np.float32),
                  alpha=r["alpha"])
    return dict(D=D, G_bd=G_bd, metrics=m, path=r["path"], model_file=model)


def run_pinn_bc(data, epochs=1500, outdir=None):
    """Study 3's boundary-aware PINN (learns the wall values); also returns its walls for replay."""
    import hybrid_bc as HB
    ds = core.dataset(data)
    s = HB.WallSolver(ds)
    t0 = time.perf_counter()
    D, wall, G, _ = HB.pinn_bc_with_walls(data["root"], data["key"], int(epochs), s)
    D = np.asarray(D, float).reshape(ds.N, ds.N)
    G_bd = G[:, s.disc.Bd]
    m = _pack(data, D, G_bd=G_bd, method="PINN (boundary-aware)", wall_s=time.perf_counter() - t0, epochs=int(epochs))
    model = _save(outdir, "pinn_bc_D_hat.npz", kind="pinn_boundary_aware", D_hat=D, G_bd=G_bd.astype(np.float32))
    return dict(D=D, G_bd=G_bd, metrics=m, model_file=model)


def _save(outdir, name, **arrays):
    outdir = Path(outdir or core.new_workdir("novelty"))
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / name
    np.savez_compressed(p, **{k: np.asarray(v) for k, v in arrays.items()})
    return str(p)


def walls_by_edge(data, G_bd):
    """Mean wall temperature of each edge over the solver's time grid (for plotting)."""
    import hybrid_bc as HB
    ds = core.dataset(data)
    s = HB.WallSolver(ds)
    N = ds.N
    Bd = np.asarray(s.disc.Bd)
    i, j = Bd // N, Bd % N
    masks = {"left": i == 0, "right": i == N - 1, "bottom": j == 0, "top": j == N - 1}
    G_bd = np.asarray(G_bd, float)
    return s.disc.t_state, {k: G_bd[:, m].mean(1) for k, m in masks.items()}
