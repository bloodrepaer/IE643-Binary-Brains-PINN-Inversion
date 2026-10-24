"""
compare_integrators.py
----------------------
Benchmarks the three time integrators (rk4, cn, fe) on a REAL benchmark case
(variable D, the two-bump initial condition), not just an eigenmode.

Space is held fixed (same grid, same operator L for every method), and
'expm' -- the exact solution of du/dt = L u + b -- is the reference. So every
error printed here is pure TIME error; spatial error cancels out by design.

Outputs
  1. Observed order of accuracy in time for fe, cn, rk4 (expected 1, 2, 4).
  2. Cost to reach a target accuracy (wall time, number of steps).
  3. Two stiffness demos:
       - explicit methods past their stability limit blow up;
       - CN with a large step "rings" (spurious negative u) on rough data
         without the Rannacher start, and does not with it.
  4. figures/integrator_convergence.png and integrator_report.json

Usage:  python compare_integrators.py [--tier C_inclusion] [--N 65] [--T 0.4]
"""
import argparse, json, os
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from domain import Domain2D, get_D_field
from integrators import ForwardModel, spectral_radius_bound, RK4_STABILITY, FE_STABILITY
from generate_dataset_v2 import initial_condition, TIERS, make_bc, BC_SCENARIOS


def rel_err(U, Uref, ks):
    return max(np.linalg.norm(U[k] - Uref[k]) / np.linalg.norm(Uref[k]) for k in ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="C_inclusion")
    ap.add_argument("--N", type=int, default=65)
    ap.add_argument("--T", type=float, default=0.4)
    ap.add_argument("--n-out", type=int, default=40)
    ap.add_argument("--bc", default="zero", choices=BC_SCENARIOS)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    dom = Domain2D(Nx=a.N, Ny=a.N)
    cls, kw = TIERS[a.tier]
    D = get_D_field(dom, cls, **kw)
    bc = make_bc(a.bc, a.T)
    u0 = initial_condition(dom, bc)
    fm = ForwardModel(dom, D, stencil="5pt", bc=bc)
    # Reference: exact-in-time expm for constant BCs; for time-varying BCs,
    # rk4 at 1/10 of its stability step (time error ~1e-14, see report).
    ref = "expm" if bc.is_constant else "rk4"
    t, Uref, info_ref = fm.solve(u0, a.T, a.n_out, ref, **({} if ref == "expm" else {"cfl_safety": 0.09}))
    ks = range(1, a.n_out + 1)        # error = worst over all output times
    rho = spectral_radius_bound(fm.L)
    true_rho = abs(spla.eigsh(fm.L, k=1, which="LM", return_eigenvectors=False)[0])
    print(f"tier={a.tier}  grid={a.N}^2  T={a.T}  bc={a.bc}  reference={ref}  D in [{D.min():.3f},{D.max():.3f}]")
    print(f"rho(L): Gershgorin bound {rho:.4g}, true {true_rho:.4g}")
    print(f"explicit step limits: FE {FE_STABILITY/true_rho:.2e}, RK4 {RK4_STABILITY/true_rho:.2e}\n")

    report = {"tier": a.tier, "N": a.N, "T": a.T, "rho_true": true_rho, "methods": {}}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    print(f"{'method':>6} {'order':>6}   {'dt':>9} {'steps':>7} {'max rel err':>12} {'wall s':>8}")
    for m in ("cn", "fe", "rk4"):
        rows = []
        if m == "rk4" and ref == "rk4":
            continue                     # can't measure rk4 against itself
        if m in ("fe", "rk4"):
            sweeps = [dict(cfl_safety=s) for s in (0.9, 0.45, 0.225, 0.1125)]
        else:
            sweeps = [dict(dt_target=a.T / n) for n in (50, 100, 200, 400, 800, 1600)]
        for kw_ in sweeps:
            _, U, info = fm.solve(u0, a.T, a.n_out, m, **kw_)
            rows.append((info["dt"], info["n_steps"], rel_err(U, Uref, ks), info["wall_s"]))
        dts, steps, errs, walls = map(np.array, zip(*rows))
        ok = errs > 1e-12          # ignore points already at round-off
        order = np.polyfit(np.log(dts[ok]), np.log(errs[ok]), 1)[0] if ok.sum() >= 2 else np.nan
        for r in rows:
            print(f"{m:>6} {order:>6.2f}   {r[0]:>9.2e} {r[1]:>7d} {r[2]:>12.2e} {r[3]:>8.3f}")
        print()
        report["methods"][m] = {"observed_order": order,
                                "runs": [dict(dt=r[0], steps=int(r[1]), err=r[2], wall_s=r[3]) for r in rows]}
        ax[0].loglog(dts, np.maximum(errs, 1e-16), "o-", label=f"{m} (slope {order:.2f})")
        ax[1].loglog(walls, np.maximum(errs, 1e-16), "o-", label=m)

    ax[0].set(xlabel="time step dt", ylabel="max relative error vs expm",
              title="Time convergence (space held fixed)")
    ax[1].set(xlabel="wall time [s]", ylabel="max relative error vs expm",
              title="Accuracy vs cost")
    for x in ax:
        x.grid(True, which="both", alpha=0.3); x.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    fig.savefig(os.path.join(a.out, "figures", "integrator_convergence.png"), dpi=140)

    # --- stiffness demo 1: explicit step slightly above the TRUE stability limit
    print("Explicit stability limit (step forced to 1.05x the true limit):")
    for m, lim in (("fe", FE_STABILITY), ("rk4", RK4_STABILITY)):
        safety = 1.05 * rho / true_rho      # cfl_safety is relative to the Gershgorin bound
        _, U, info = fm.solve(u0, a.T, a.n_out, m, cfl_safety=safety)
        print(f"  {m}: dt={info['dt']:.2e}  max|u| at T = {np.abs(U[-1]).max():.2e}  (true max {np.abs(Uref[-1]).max():.2e})")

    # --- stiffness demo 2: CN ringing from non-smooth data.
    # The benchmark's Gaussian bumps are smooth enough that ringing is
    # negligible. Rough data is where it bites -- including the ADJOINT
    # solve, which is driven by point sources at the sensor locations.
    # A single hot node stands in for that here.
    print("\nCN with a large step (dt = T/20) from a single hot node (point source):")
    fm0 = ForwardModel(dom, D, stencil="5pt")          # zero BC for this demo
    spike = np.zeros(dom.shape); spike[a.N // 2, a.N // 2] = 1.0
    _, Uex, _ = fm0.solve(spike, a.T, 20, "expm")
    for r in (False, True):
        _, U, _ = fm0.solve(spike, a.T, 20, "cn", dt_target=a.T / 20, rannacher=r)
        print(f"  rannacher={r!s:5}: most negative u = {U[1:].min():+.2e},  "
              f"rel err at first output = {np.linalg.norm(U[1]-Uex[1])/np.linalg.norm(Uex[1]):.2e}")
    print("  (exact solution is >= 0 everywhere)")

    with open(os.path.join(a.out, "integrator_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print("\nSaved figures/integrator_convergence.png and integrator_report.json")


if __name__ == "__main__":
    main()
