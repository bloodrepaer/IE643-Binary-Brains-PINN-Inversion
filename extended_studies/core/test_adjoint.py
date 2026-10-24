"""
test_adjoint.py
---------------
Verification of adjoint_inversion.py. Run this BEFORE trusting any inversion.

  1. Forward consistency: the inversion's forward solve (CN + Rannacher on the
     inversion grid) matches integrators.ForwardModel to round-off.
  2. Observation operator: Q is exactly the transpose used in the adjoint
     (<Q x, y> = <x, Q^T y> on random vectors).
  3. Taylor test, for Tikhonov and TV, on (a) a real benchmark dataset and
     (b) a synthetic case with TIME-VARYING boundary values and readings
     OFF the time grid (exercises every code path):
        |J(th + eps v) - J(th) - eps g.v| = O(eps^2)
     The log-log slope must be ~2. Slope ~1 means the gradient is wrong
     (and L-BFGS would still converge -- to the wrong answer).
  4. Gradient vs central finite differences on a few components.

Usage:  python test_adjoint.py [--data datasets_v2_rk4_zero]
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from domain import Domain2D
from integrators import ForwardModel, BoundaryCondition
from adjoint_inversion import Dataset, AdjointSolver

ok_all = True


def report(name, ok, detail):
    global ok_all
    ok_all &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def taylor(solver, theta, alpha, rng, label, ax=None):
    g = solver.objective(theta, alpha)[1]
    J0 = solver.objective(theta, alpha, want_grad=False)
    v = rng.standard_normal(theta.shape)
    eps = np.logspace(-1, -6, 11)
    rem = np.array([abs(solver.objective(theta + e * v, alpha, want_grad=False) - J0 - e * g @ v) for e in eps])
    use = (rem > 1e-9 * abs(J0))                  # drop points lost in round-off
    slope = np.polyfit(np.log(eps[use]), np.log(rem[use]), 1)[0]
    if ax is not None:
        ax.loglog(eps, rem, "o-", label=f"{label} (slope {slope:.2f})")
    report(f"Taylor test {label}", 1.8 < slope < 2.2, f"slope {slope:.3f}")
    return slope


class SyntheticDataset:
    """Small in-memory problem: time-varying BC on two edges, random off-grid times."""
    def __init__(self, N=33, T=0.5, seed=0):
        rng = np.random.default_rng(seed)
        self.key, self.N, self.T = "synthetic_timevarying_bc", N, T
        self.bc = BoundaryCondition({"left": lambda t: t / T, "top": lambda t: 0.5 * np.sin(np.pi * t / T)})
        dom = Domain2D(Nx=N, Ny=N)
        self.u0 = np.exp(-((dom.X - 0.4) ** 2 + (dom.Y - 0.5) ** 2) / 0.02)
        g0 = self.bc.field(dom.shape, 0.0)
        self.u0[0, :], self.u0[-1, :], self.u0[:, 0], self.u0[:, -1] = g0[0, :], g0[-1, :], g0[:, 0], g0[:, -1]
        n = 30
        self.x, self.y = rng.uniform(0.05, 0.95, n), rng.uniform(0.05, 0.95, n)
        self.x, self.y = np.tile(self.x, 4), np.tile(self.y, 4)
        self.t = np.repeat(rng.uniform(0.05, 1.0, 4) * T, n)       # deliberately off-grid
        self.u = rng.normal(0.2, 0.05, len(self.x))
        self.sigma = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets_v2_rk4_zero")
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    manifest = json.load(open(os.path.join(a.data, "manifest.json")))
    e = next(r for r in manifest if r["key"] == "C_inclusion__Fo0.1__uniform8__n50__random__eta0.05")
    ds = Dataset(a.data, e)
    D_true, _ = ds.ground_truth()

    print("1. Forward consistency (inversion solver vs integrators.ForwardModel, CN + Rannacher)")
    solver = AdjointSolver(ds, n_steps=200)
    W = solver.forward(D_true.ravel())[0]
    dom = Domain2D(Nx=ds.N, Ny=ds.N)
    _, U, _ = ForwardModel(dom, D_true, bc=ds.bc).solve(ds.u0, ds.T, 200, "cn", dt_target=ds.T / 200)
    full_steps = [solver.disc.state_of_step(k) for k in range(201)]
    err = np.abs(W[full_steps].reshape(201, ds.N, ds.N) - U).max()
    report("same discrete solution", err < 1e-12, f"max |diff| = {err:.1e}")

    print("2. Observation operator transpose")
    x = rng.standard_normal(solver.Q.shape[1]); y = rng.standard_normal(solver.Q.shape[0])
    lhs, rhs = (solver.Q @ x) @ y, x @ (solver.Q.T @ y)
    report("<Qx,y> = <x,Q^T y>", abs(lhs - rhs) < 1e-10 * abs(lhs), f"rel diff {abs(lhs-rhs)/abs(lhs):.1e}")

    print("3. Taylor tests")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    theta = np.log(0.2) + 0.3 * rng.standard_normal(16 * 16)
    alpha = 10.0
    for reg in ("tikhonov", "tv"):
        s = AdjointSolver(ds, n_steps=200, control=16, reg=reg)
        taylor(s, theta, alpha, rng, f"benchmark, {reg}", ax)
    syn = SyntheticDataset()
    for reg in ("tikhonov", "tv"):
        s = AdjointSolver(syn, n_steps=60, control=8, reg=reg)
        th = np.log(0.3) + 0.3 * rng.standard_normal(64)
        taylor(s, th, 1.0, rng, f"time-varying BC + off-grid times, {reg}", ax)
    e_ = np.logspace(-1, -6, 11)
    ax.loglog(e_, 1e2 * e_**2, "k--", lw=1, label="slope 2")
    ax.loglog(e_, 1e1 * e_, "k:", lw=1, label="slope 1 (wrong gradient)")
    ax.set(xlabel="ε", ylabel="|J(θ+εv) − J(θ) − ε gᵀv|", title="Taylor test")
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/taylor_test.png", dpi=120)

    print("4. Adjoint gradient vs central finite differences (benchmark, Tikhonov)")
    s = AdjointSolver(ds, n_steps=200, control=16)
    g = s.objective(theta, alpha)[1]
    worst = 0.0
    for k in rng.choice(len(theta), 5, replace=False):
        h = 1e-5; dth = np.zeros_like(theta); dth[k] = h
        fd = (s.objective(theta + dth, alpha, False) - s.objective(theta - dth, alpha, False)) / (2 * h)
        worst = max(worst, abs(fd - g[k]) / max(abs(fd), 1e-12))
        print(f"      theta[{k:3d}]  adjoint {g[k]: .6e}   FD {fd: .6e}")
    report("adjoint = FD", worst < 1e-5, f"worst relative difference {worst:.1e}")

    print("\nALL PASSED" if ok_all else "\nSOME TESTS FAILED")
    print("Saved figures/taylor_test.png")


if __name__ == "__main__":
    main()
