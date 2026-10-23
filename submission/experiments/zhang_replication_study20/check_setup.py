"""Sanity checks: solver timing, discretisation floor, Taylor test of the
reduced gradient, and reproduction of one project baseline (E_grf, eta 5%)."""
import time, numpy as np, sys
sys.path.insert(0, "replication")
from problem import load, Reduced, truth, solver_floor, metrics
from adjoint_inversion import AdjointSolver

for key in ["E_grf__Fo0.1__uniform8__n50__random__eta0.05", "A_smooth__Fo0.1__uniform8__nfull__nodes__eta0"]:
    ds = load(key); red = Reduced(ds)
    m = np.log(truth(ds)["D"]).ravel()
    t = time.perf_counter(); J, g = red.J(m, 1.0); t1 = time.perf_counter() - t
    t = time.perf_counter(); red.J(m, 1.0, grad=False); t2 = time.perf_counter() - t
    print(f"{key}: n_steps={red.dz.n_steps}  J+grad {t1:.3f}s  J only {t2:.3f}s  floor eps_u={solver_floor(ds, red):.3e}")
    # Taylor test on the full-grid m
    rng = np.random.default_rng(0); dm = rng.standard_normal(m.size) * 0.1
    m0 = m + 0.2 * rng.standard_normal(m.size)
    J0, g0 = red.J(m0, 1.0)
    errs = [abs(red.J(m0 + e * dm, 1.0, grad=False) - J0 - e * g0 @ dm) for e in (1e-2, 5e-3, 2.5e-3)]
    print("   Taylor slopes", np.round(np.log2(np.array(errs[:-1]) / np.array(errs[1:])), 3))

ds = load("E_grf__Fo0.1__uniform8__n50__random__eta0.05")
s = AdjointSolver(ds, control=16, reg="tikhonov")
out = s.invert(verbose=False)
red = Reduced(ds)
print("project pipeline repro E_grf eta5%: alpha", out["alpha"], "metrics", metrics(ds, red, out["D"]), f"{out['wall_s']:.0f}s",
      "(project summary.csv: rel_L2_D 0.2068, alpha 12.649)")
