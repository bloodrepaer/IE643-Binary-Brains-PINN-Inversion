"""
hybrid.py -- the classical adjoint started from someone else's estimate (the PINN's), with honest costs.

The standard inversion (AdjointSolver.invert) starts from the best constant D and walks a regularisation
path alpha = N_d * 10^0 ... 10^-6 (13 values), stopping at the first alpha whose replay reaches sigma
(discrepancy principle). The hybrid keeps everything identical except the starting point:

  project_to_controls   least-squares fit of log D_PINN onto the 16 x 16 bilinear control grid
  invert_from(..., "path")     same alpha path as the cold run, warm-started at the projection
  invert_from(..., "balance")  path starts at alpha0 = misfit(theta0) / R(theta0), the weight at which
                               the PINN estimate's data misfit and its regulariser balance, then steps
                               down by 10^-0.5 as the standard path does. Uses data only, no truth.
  PriorAdjoint ("prior")       same path, but Tikhonov on grad(log D - log D_PINN): the adjoint is pulled
                               toward the PINN's estimate instead of toward a flat field.

Costs are counted in full: hybrid wall time = PINN training + projection + adjoint stage.
"""
import time

import numpy as np
from scipy.optimize import minimize

import adjoint_inversion as A


def project_to_controls(solver, D):
    """theta minimising || P theta - log D ||^2 on the inversion grid."""
    Pm = solver.P
    PtP = (Pm.T @ Pm).toarray()
    rhs = Pm.T @ np.log(np.clip(np.asarray(D, float).ravel(), A.D_MIN, A.D_MAX))
    return np.linalg.solve(PtP, rhs)


def balance_alphas(solver, theta0):
    n_d = len(solver.d)
    solver.objective(theta0, 0.0, want_grad=False)
    misfit = solver._last["misfit"]
    R = solver.regulariser(solver.m_of(theta0))[0]
    a0 = float(np.clip(misfit / max(R, 1e-12), n_d * 1e-6, n_d))
    k = int(np.floor(2 * np.log10(a0 / (n_d * 1e-6)) + 1e-9))      # steps of 10^-0.5 down to N_d * 1e-6
    return a0 * 10.0 ** (-0.5 * np.arange(k + 1)), a0


def invert_from(solver, theta0, alphas, maxiter=150, tau=1.0):
    """AdjointSolver.invert with a given start (same bounds, optimiser, stopping and floor rules)."""
    tic = time.perf_counter()
    K2 = solver.K * solver.K
    theta = np.asarray(theta0, float).copy()
    bounds = [(np.log(A.D_MIN), np.log(A.D_MAX))] * K2
    theta = np.clip(theta, bounds[0][0], bounds[0][1])
    path, chosen, flag = [], None, "reached_sigma"
    n0 = solver.n_evals
    for a in alphas:
        res = minimize(solver.objective, theta, args=(a,), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-6})
        theta = res.x
        solver.objective(theta, a, want_grad=False)
        rms = solver._last["rms"]
        path.append(dict(alpha=float(a), rms=float(rms), rms_over_sigma=float(rms / solver.sigma_w),
                         iters=int(res.nit), theta=theta.copy()))
        if solver.ds.sigma > 0 and rms <= tau * solver.sigma_w:
            chosen = len(path) - 1
            break
        if len(path) >= 3:
            r0, r1, r2 = path[-3]["rms"], path[-2]["rms"], path[-1]["rms"]
            if r1 > 0.99 * r0 and r2 > 0.99 * r1:
                break
    if chosen is None:
        flag = "floor_above_sigma" if solver.ds.sigma > 0 else "noise_free"
        best = min(p["rms"] for p in path)
        chosen = next(i for i, p in enumerate(path) if p["rms"] <= 1.02 * best)
    sel = path[chosen]
    D = np.exp(solver.m_of(sel["theta"]))
    return dict(D=D, alpha=sel["alpha"], flag=flag, path=[{k: v for k, v in p.items() if k != "theta"} for p in path],
                total_iters=int(sum(p["iters"] for p in path)), n_evals=int(solver.n_evals - n0),
                wall_s=time.perf_counter() - tic)


def run_cold(ds):
    """The standard classical adjoint, timed (best-constant start + full alpha path)."""
    s = A.AdjointSolver(ds)
    tic = time.perf_counter()
    res = s.invert(verbose=False)
    wall = time.perf_counter() - tic
    return dict(D=res["D"], alpha=res["alpha"], flag=res["flag"], D_const=res["D_const"],
                path=[{k: v for k, v in p.items() if k != "theta"} for p in res["path"]],
                total_iters=int(sum(p["iters"] for p in res["path"])), n_evals=int(res["n_evals"]), wall_s=wall)


class PriorAdjoint(A.AdjointSolver):
    """The adjoint regularised toward a reference field instead of toward a flat one:
    R(m) = 1/2 sum over links of ((m - m_ref)_p - (m - m_ref)_q)^2  (Tikhonov on grad(m - m_ref)).
    With m_ref = log D_PINN, the data decide where the adjoint departs from the PINN's estimate."""

    def __init__(self, ds, m_ref, **kw):
        super().__init__(ds, **kw)
        self.m_ref = np.asarray(m_ref, float).ravel()

    def regulariser(self, m):
        return super().regulariser(m - self.m_ref)


def run_warm(ds, D_start, mode):
    if mode == "prior":
        m_ref = np.log(np.clip(np.asarray(D_start, float).ravel(), A.D_MIN, A.D_MAX))
        s = PriorAdjoint(ds, m_ref)
        tic = time.perf_counter()
        theta0 = project_to_controls(s, D_start)
        proj_s = time.perf_counter() - tic
        out = invert_from(s, theta0, len(s.d) * np.logspace(0, -6, 13))
        out.update(projection_s=proj_s, alpha_start=None, D_projected=np.exp(s.m_of(theta0)))
        return out
    s = A.AdjointSolver(ds)
    tic = time.perf_counter()
    theta0 = project_to_controls(s, D_start)
    proj_s = time.perf_counter() - tic
    if mode == "path":
        alphas, a0 = len(s.d) * np.logspace(0, -6, 13), None
    elif mode == "balance":
        alphas, a0 = balance_alphas(s, theta0)
    else:
        raise ValueError(mode)
    out = invert_from(s, theta0, alphas)
    out.update(projection_s=proj_s, alpha_start=None if a0 is None else float(a0),
               D_projected=np.exp(s.m_of(theta0)))
    return out
