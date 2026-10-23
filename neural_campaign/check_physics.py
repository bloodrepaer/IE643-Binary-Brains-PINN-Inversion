"""
check_physics.py -- do the stored datasets obey the heat equation they claim to?

    python check_physics.py <dataset folder> [chunk_index n_chunks]

Per ground-truth field (problem):
  resolve_rel_err   re-solve the TRUE D, IC, BC (and source) with a DIFFERENT integrator
                    (Crank-Nicolson, dt = T/2000) on the same 129^2 grid and compare with the
                    stored RK4 snapshots: max over snapshots of ||U_cn - U|| / ||U||
  residual_rel      discrete PDE residual of the STORED snapshots,
                    r = (U[k+1]-U[k-1])/(2 dt) - (L U[k] + b(t_k) + f),  median_k ||r|| / ||u_t||
  residual_rel_cn   the same number for the CN re-solve: the part of the residual that is only
                    the coarse snapshot spacing (T/40), not a physics violation
  min_u, max_u      maximum principle: 0 <= u <= max(u0, max g) when f = 0 (all data here are >= 0)
  heat_monotone     zero walls, no source: total heat sum(u) never increases
  energy_monotone   zero walls, no source: sum(u^2) never increases  (d/dt int u^2 = -2 int D|grad u|^2)
Per sensor file:
  noise z = (u - u_clean)/sigma: mean, std, KS-test p value against N(0,1)
  clean_vs_field    clean sensor values vs the stored field interpolated at the sensor (at stored times), / sigma
Writes <folder>/physics_check_fields[_chunk].csv and physics_check_sensors[_chunk].csv.
"""
import csv, json, os, sys
import numpy as np
import sympy
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import kstest

from domain import Domain2D
from integrators import BoundaryCondition, build_operator, integrate, EDGES

ROOT = sys.argv[1]
ci, cn = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
man = json.load(open(os.path.join(ROOT, "manifest.json")))
problems = {}
for e in man:
    problems.setdefault(e["problem"], e)
probs = sorted(problems)[ci::cn]

x, y = sp_x, sp_y = sympy.symbols("x y")
ts = sympy.Symbol("t")


def true_bc_and_source(e, spec, gt, T):
    """True boundary condition and source. Unknown-BC problem files carry an ASSUMED zero BC, so the true
    one is rebuilt from the generator's definition recorded in the manifest (bc_true = name:amp)."""
    if "bc_true" in e:
        import generate_pinn_favoured as GP
        name, amp = e["bc_true"].split(":")
        bc = GP.make_bc(name, T, float(amp), e["Fo"])
        src = str(gt["source_true"]) if "source_true" in gt.files else "0"
    else:
        edges = spec["boundary_condition"]["edges"]
        bc = BoundaryCondition({k: sympy.lambdify(ts, sympy.sympify(v), "numpy") for k, v in edges.items()},
                               {k: v for k, v in edges.items()})
        src = spec.get("source_f", "0")
    return bc, src


rows_f, rows_s = [], []
for p in probs:
    e = problems[p]
    gt = np.load(os.path.join(ROOT, "ground_truth", e["ground_truth"]), allow_pickle=True)
    spec = json.load(open(os.path.join(ROOT, "problem", p + ".json")))
    D, U, tt, T = gt["D_data"], gt["u"].astype(float), gt["t"], float(gt["T"])
    dom = Domain2D(Nx=D.shape[0], Ny=D.shape[1])
    bc, src = true_bc_and_source(e, spec, gt, T)
    f = sympy.lambdify((sp_x, sp_y), sympy.sympify(src), "numpy")(dom.X, dom.Y) * np.ones_like(D)
    L, B = build_operator(D, dom.dx, dom.dy, "5pt")
    f_int = f[1:-1, 1:-1].ravel()
    b = lambda t: sum(bc.at(t)[k] * B[k] for k in EDGES) + f_int
    # 1. independent re-solve (CN) at the stored times
    n_out = int(round((len(tt) - 1)))
    _, Ucn_int, _ = integrate(L, b, U[0, 1:-1, 1:-1].ravel(), T, n_out, "cn", dt_target=T / 2000)
    Ucn = U.copy()
    Ucn[:, 1:-1, 1:-1] = Ucn_int.reshape(len(tt), D.shape[0] - 2, D.shape[1] - 2)
    rel = [np.linalg.norm(Ucn[k] - U[k]) / max(np.linalg.norm(U[k]), 1e-300) for k in range(1, len(tt))]
    # 2. discrete residual of the stored snapshots (and of the CN snapshots, for reference)
    dt = tt[1] - tt[0]
    def resid(V):
        out = []
        for k in range(1, len(tt) - 1):
            ut = (V[k + 1, 1:-1, 1:-1] - V[k - 1, 1:-1, 1:-1]).ravel() / (2 * dt)
            r = ut - (L @ V[k, 1:-1, 1:-1].ravel() + b(tt[k]))
            out.append(np.linalg.norm(r) / max(np.linalg.norm(ut), 1e-300))
        return float(np.median(out))
    zero_forcing = all(abs(bc.at(t_)[k]) < 1e-14 for t_ in tt for k in EDGES) and not np.any(f)
    gmax = max(max(abs(bc.at(t_)[k]) for k in EDGES) for t_ in tt)
    H = U[:, 1:-1, 1:-1].sum(axis=(1, 2)); E = (U[:, 1:-1, 1:-1] ** 2).sum(axis=(1, 2))
    rows_f.append(dict(problem=p, resolve_rel_err=float(max(rel)), residual_rel=resid(U), residual_rel_cn=resid(Ucn),
                       min_u=float(U.min()), max_u=float(U.max()), bound=float(max(U[0].max(), gmax)) if not np.any(f) else np.nan,
                       max_principle_ok=bool(U.min() >= -1e-9 and (np.any(f) or U.max() <= max(U[0].max(), gmax) + 1e-9)),
                       heat_monotone=bool(np.all(np.diff(H) <= 1e-12 * H[0])) if zero_forcing else None,
                       energy_monotone=bool(np.all(np.diff(E) <= 1e-12 * E[0])) if zero_forcing else None,
                       forcing="none" if zero_forcing else ("source" if np.any(f) else "boundary")))
    r = rows_f[-1]
    print(f"{p:45s} resolve {r['resolve_rel_err']:.1e}  resid {r['residual_rel']:.1e} (cn {r['residual_rel_cn']:.1e})  "
          f"min {r['min_u']:.1e} max {r['max_u']:.3f}  maxprinc {r['max_principle_ok']}  heat {r['heat_monotone']}  energy {r['energy_monotone']}",
          flush=True)
    # 3. sensors of this problem
    interp_k = {k: RegularGridInterpolator((dom.x, dom.y), U[k]) for k in range(len(tt))}
    for s in [m for m in man if m["problem"] == p]:
        z_ = np.load(os.path.join(ROOT, s["files"]["sensors_npz"]))
        sig = float(z_["sigma"][0])
        row = dict(key=s["key"], problem=p, n=len(z_["u"]), sigma=sig)
        if sig > 0:
            z = (z_["u"] - z_["u_clean"]) / sig
            row.update(z_mean=float(z.mean()), z_std=float(z.std()), ks_p=float(kstest(z, "norm").pvalue))
        ti = z_["time_index"]; sel = ti % 10 == 0
        if sel.any():
            vals = np.array([interp_k[int(k // 10)]((xx, yy)) for k, xx, yy in zip(ti[sel], z_["x"][sel], z_["y"][sel])])
            ref = sig if sig > 0 else float(np.sqrt(np.mean(z_["u_clean"] ** 2)))
            row["clean_vs_field_over_sigma"] = float(np.max(np.abs(vals - z_["u_clean"][sel])) / ref)
        rows_s.append(row)

suffix = f"_chunk{ci}" if cn > 1 else ""
for name, rows in (("fields", rows_f), ("sensors", rows_s)):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(ROOT, f"physics_check_{name}{suffix}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(f"{len(rows_f)} fields, {len(rows_s)} sensor files checked")
