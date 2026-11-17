"""
test_solvers.py -- checks for the three solvers before any inversion is scored.

  1. Forward solves: each solver's states against the matrix exponential (exact in time on the same 65^2
     operator); errors must fall at the scheme's order (RK4 4, CN 2, BE 1) when the step is halved.
  2. RK4 forward against integrators.ForwardModel(..., "rk4") on the same step count: identical scheme.
  3. Gradients: Taylor test (remainder slope 2) for every solver at a random log D, plus a finite-difference
     check of 5 control values.
Run from extended_studies/:  python run.py study5_time_integrators/test_solvers.py
"""
import json
import sys
import numpy as np
import scipy.sparse.linalg as spla

import adjoint_inversion as A
import solvers as S

DATA = "datasets_v2_density_extended"
KEY = "A_smooth__Fo0.1__uniform8__n400__nodes_nested__eta0.05"
FAILS = []


def report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def load():
    e = next(r for r in json.load(open(f"{DATA}/manifest.json")) if r["key"] == KEY)
    return A.Dataset(DATA, e)


def exact_states(solver, D, times):
    dz = solver.disc
    LII = dz.full_operator(dz.conductance(D))[dz.I][:, dz.I].tocsc()
    u0 = solver.u0_full[dz.I]
    return np.stack([spla.expm_multiply(LII * t, u0) for t in times])


def states_at(solver, W, times):
    dz = solver.disc
    out = []
    for t in times:
        k = int(round(t / dz.dt))
        out.append(W[dz.state_of_step(k), dz.I])
    return np.stack(out)


def main():
    ds = load()
    rng = np.random.default_rng(7)
    # a smooth, non-constant test field inside the cap
    X, Y = np.meshgrid(np.linspace(0, 1, ds.N), np.linspace(0, 1, ds.N), indexing="ij")
    D = (0.15 + 0.25 * (0.5 + 0.5 * np.sin(2 * np.pi * X) * np.cos(np.pi * Y))).ravel()
    times = ds.T * np.array([0.25, 0.5, 1.0])

    print("1. Forward solves against the matrix exponential (same operator, exact in time)")
    for name, n_coarse, order in (("BE", 200, 1), ("CN", 200, 2), ("RK4", 3200, 4)):
        errs = []
        for n in (n_coarse, 2 * n_coarse):
            s = S.make(name, ds, n)
            W = s.forward(D)[0]
            ref = exact_states(s, D, times)
            errs.append(np.abs(states_at(s, W, times) - ref).max() / np.abs(ref).max())
        rate = np.log2(errs[0] / errs[1])
        ok = abs(rate - order) < 0.4 or (order == 4 and errs[1] < 1e-10)
        report(f"{name} order", ok, f"rel err {errs[0]:.2e} -> {errs[1]:.2e}, observed order {rate:.2f} (expected {order})")

    print("2. RK4 forward against integrators.ForwardModel('rk4') at the same step")
    from integrators import integrate
    s = S.make("RK4", ds, 4800)
    W = s.forward(D)[0]
    dz = s.disc
    LII = dz.full_operator(dz.conductance(D))[dz.I][:, dz.I].tocsr()
    t_out, U, info = integrate(LII, np.zeros(dz.nI), s.u0_full[dz.I], ds.T, 8, "rk4",
                               cfl_safety=(ds.T / 4800) * float(np.max(2 * np.abs(LII.diagonal()))) / 2.785)
    mine = np.stack([W[k * 600, dz.I] for k in range(9)])
    diff = np.abs(mine - U).max() / np.abs(U).max()
    report("RK4 matches integrators.integrate", diff < 1e-6, f"max rel diff {diff:.1e} (integrate used {info['n_steps']} steps)")

    print("3. Gradients (Taylor remainder slope 2; finite differences)")
    for name, n in (("CN", 200), ("BE", 200), ("RK4", 4800)):
        s = S.make(name, ds, n)
        K2 = s.K ** 2
        theta = np.log(0.25) + 0.3 * rng.standard_normal(K2)
        theta = np.clip(theta, np.log(0.05), np.log(0.9))
        alpha = 1e-2 * len(ds.u)
        J0, g = s.objective(theta, alpha)
        v = rng.standard_normal(K2)
        eps = np.logspace(-2, -6, 9)
        rem = np.array([abs(s.objective(theta + e * v, alpha, want_grad=False) - J0 - e * g @ v) for e in eps])
        use = rem > 1e-10 * abs(J0)
        slope = np.polyfit(np.log(eps[use]), np.log(rem[use]), 1)[0]
        report(f"{name} Taylor test", 1.8 < slope < 2.2, f"slope {slope:.3f}")
        worst = 0.0
        for i in rng.choice(K2, 5, replace=False):
            e = np.zeros(K2); e[i] = 1e-5
            fd = (s.objective(theta + e, alpha, want_grad=False) - s.objective(theta - e, alpha, want_grad=False)) / 2e-5
            worst = max(worst, abs(fd - g[i]) / max(abs(g[i]), 1e-12 * abs(J0)))
        report(f"{name} finite differences", worst < 1e-5, f"worst relative difference {worst:.1e}")
        if name == "RK4":
            report("RK4 stability margin", s.max_dt_rho < S.RK4_STABILITY, f"max dt*rho {s.max_dt_rho:.2f} (limit {S.RK4_STABILITY})")

    print("ALL PASSED" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
