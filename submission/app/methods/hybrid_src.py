"""App copy of extended_studies/novelty/hybrid_src.py (the project-only helpers import common lazily).

hybrid_src.py -- E1s: the hidden-source datasets (Study 3, H3u), and the fair classical rival.

Why: in E1 the PINN beat the adjoint on the H3u datasets (an unmodelled interior heater), because the
adjoint assumes f = 0 and bends D to explain the injected heat. E1b showed that the PINN's advantage on
hidden WALLS disappears once the adjoint estimates the walls itself. This module asks the same question
for a hidden SOURCE: give the classical method a source field to estimate jointly with D.

Arms (all on the 6 H3u datasets: a time-constant Gaussian heater, absent from the problem file):
  adjoint_src        joint inversion of log D and a source f(x, y), both on 16 x 16 bilinear controls;
                     f starts at 0 and is free in sign
  adjoint_src_pos    the same with f >= 0 (the source is known to be a heater, not a sink)
  adjoint_truesrc    the adjoint given the TRUE source (a bound, not a method: uses hidden information)

Everything else is the project's adjoint unchanged (CN, 16 x 16 controls on log D, Tikhonov on grad log D,
L-BFGS-B 150 iterations per alpha, alpha path N_d 10^0 ... 10^-6, discrepancy principle). The source gets
the same kind of smoothness penalty with the same alpha: alpha * 1/2 sum (phi_i - phi_j)^2 over neighbouring
controls.

The source enters each theta-scheme substep j (state j -> j+1) of the CN integrator as + h_j f on the
right-hand side, so with the adjoint states Lam of field_objective_src the gradient is
    dJ/df_I = - sum_j h_j Lam[j+1, I].
test_source_gradient() checks it against finite differences.
"""
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sympy
from scipy.optimize import minimize

import adjoint_inversion as A

K_SRC = 16
E1S_ARMS = ("adjoint_truesrc", "adjoint_src", "adjoint_src_pos")


class SourceSolver(A.AdjointSolver):
    """The project's AdjointSolver with a time-constant interior source f = Ps @ phi."""

    def __init__(self, ds, k_src=K_SRC, **kw):
        super().__init__(ds, **kw)
        P1 = A.bilinear_1d(ds.N, k_src)
        self.Ps = sp.kron(P1, P1).tocsr()                  # (N^2, k_src^2)
        self.k_src = k_src
        self.f_full = np.zeros(ds.N * ds.N)                # current source on the full grid
        # neighbour pairs on the k_src x k_src control grid (for the smoothness penalty)
        idx = np.arange(k_src * k_src).reshape(k_src, k_src)
        self.sp_ = np.r_[idx[:-1, :].ravel(), idx[:, :-1].ravel()]
        self.sq_ = np.r_[idx[1:, :].ravel(), idx[:, 1:].ravel()]

    # source ------------------------------------------------------------------------------------
    def set_f(self, f_full):
        self.f_full = np.asarray(f_full, float).copy()

    def f_of_phi(self, phi):
        return self.Ps @ phi

    def src_regulariser(self, phi):
        diff = phi[self.sp_] - phi[self.sq_]
        g = np.zeros_like(phi)
        np.add.at(g, self.sp_, diff)
        np.add.at(g, self.sq_, -diff)
        return 0.5 * np.sum(diff ** 2), g

    # forward with source -----------------------------------------------------------------------
    def forward(self, D):
        dz = self.disc
        c = dz.conductance(D)
        Lf = self.full_operator = dz.full_operator(c)
        LII = Lf[dz.I][:, dz.I]
        LIB = Lf[dz.I][:, dz.Bd]
        M = (sp.identity(dz.nI, format="csc") - (dz.dt / 2) * LII).tocsc()
        solve = spla.splu(M, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0,
                          options=dict(SymmetricMode=True)).solve
        W = dz.G.copy()
        W[0, dz.I] = self.u0_full[dz.I]
        u = W[0, dz.I]
        has_bc = np.any(dz.G[:, dz.Bd])
        bterm = (LIB @ dz.G[:, dz.Bd].T).T if has_bc else None
        fI = self.f_full[dz.I]
        has_f = np.any(fI)
        for j in range(dz.S - 1):
            h, th = dz.h_sub[j], dz.th_sub[j]
            rhs = u + h * (1 - th) * (LII @ u) if th < 1 else u.copy()
            if has_bc:
                rhs += h * (1 - th) * bterm[j] + h * th * bterm[j + 1]
            if has_f:
                rhs += h * fI
            u = solve(rhs)
            W[j + 1, dz.I] = u
        return W, c, LII, solve

    # objective in log D (full grid) with gradients for D and f ---------------------------------
    def field_objective_src(self, m, alpha, want_grad=True):
        self.n_evals += 1
        dz = self.disc
        D = np.exp(m)
        W, c, LII, solve = self.forward(D)
        r = self.predict(W) - self.d
        s2 = self.sigma_w ** 2
        misfit = 0.5 * np.dot(r, r) / s2
        R, gR = self.regulariser(m)
        J = misfit + alpha * R
        self._last = dict(misfit=misfit, R=R, rms=np.sqrt(np.mean(r ** 2)), W=W, D=D)
        if not want_grad:
            return J
        dJdW = (self.Q.T @ (r / s2)).reshape(dz.S, -1)
        Lam = np.zeros_like(W)
        lam_next = np.zeros(dz.nI)
        for s in range(dz.S - 1, 0, -1):
            rhs = -dJdW[s, dz.I]
            if s < dz.S - 1:
                hs, ths = dz.h_sub[s], dz.th_sub[s]
                rhs = rhs + lam_next + hs * (1 - ths) * (LII @ lam_next)
            lam = solve(rhs)
            Lam[s, dz.I] = lam
            lam_next = lam
        dW = W[:, dz.p] - W[:, dz.q]
        dL = Lam[:, dz.p] - Lam[:, dz.q]
        h, th = dz.h_sub[:, None], dz.th_sub[:, None]
        g_c = np.sum(h * dL[1:] * (th * dW[1:] + (1 - th) * dW[:-1]), axis=0)
        Dp, Dq = D[dz.p], D[dz.q]
        den = (Dp + Dq) ** 2
        g_D = np.zeros_like(D)
        np.add.at(g_D, dz.p, g_c * dz.w_link * 2 * Dq ** 2 / den)
        np.add.at(g_D, dz.q, g_c * dz.w_link * 2 * Dp ** 2 / den)
        g_m = D * g_D + alpha * gR
        g_f = np.zeros_like(D)
        g_f[dz.I] = -(dz.h_sub[:, None] * Lam[1:, dz.I]).sum(0)
        return J, g_m, g_f

    def joint_objective(self, z, alpha, want_grad=True):
        K2 = self.K ** 2
        theta, phi = z[:K2], z[K2:]
        self.set_f(self.f_of_phi(phi))
        Rf, gRf = self.src_regulariser(phi)
        out = self.field_objective_src(self.m_of(theta), alpha, want_grad)
        if not want_grad:
            return out + alpha * Rf
        J, g_m, g_f = out
        self._last["Rf"] = Rf
        return J + alpha * Rf, np.r_[self.P.T @ g_m, self.Ps.T @ g_f + alpha * gRf]

    def objective_fixed_f(self, theta, alpha, want_grad=True):
        out = self.field_objective_src(self.m_of(theta), alpha, want_grad)
        if not want_grad:
            return out
        J, g_m, _ = out
        return J, self.P.T @ g_m


# ---------------------------------------------------------------------------------------------
# inversion driver (same rules as AdjointSolver.invert)
# ---------------------------------------------------------------------------------------------
def invert_src(s, optimise_source=True, positive=False, maxiter=150, tau=1.0):
    tic = time.perf_counter()
    K2, F2 = s.K ** 2, s.k_src ** 2
    alphas = len(s.d) * np.logspace(0, -6, 13)
    lo, hi = np.log(A.D_MIN), np.log(A.D_MAX)
    # start as the project's adjoint does: the best-fitting constant D (with the current source, i.e. f = 0
    # for the joint arms and the true f for adjoint_truesrc)
    theta0 = np.full(K2, s.best_constant())
    if optimise_source:
        z = np.r_[theta0, np.zeros(F2)]
        bounds = [(lo, hi)] * K2 + [((0.0, None) if positive else (None, None))] * F2
        fun = s.joint_objective
    else:
        z = theta0
        bounds = [(lo, hi)] * K2
        fun = s.objective_fixed_f
    path, chosen, flag = [], None, "reached_sigma"
    for a in alphas:
        res = minimize(fun, z, args=(a,), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-6})
        z = res.x
        fun(z, a, want_grad=False)
        rms = s._last["rms"]
        path.append(dict(alpha=float(a), rms=float(rms), rms_over_sigma=float(rms / s.sigma_w), iters=int(res.nit),
                         z=z.copy()))
        if s.ds.sigma > 0 and rms <= tau * s.sigma_w:
            chosen = len(path) - 1
            break
        if len(path) >= 3:
            r0, r1, r2 = path[-3]["rms"], path[-2]["rms"], path[-1]["rms"]
            if r1 > 0.99 * r0 and r2 > 0.99 * r1:
                break
    if chosen is None:
        flag = "floor_above_sigma" if s.ds.sigma > 0 else "noise_free"
        best = min(p["rms"] for p in path)
        chosen = next(i for i, p in enumerate(path) if p["rms"] <= 1.02 * best)
    z = path[chosen]["z"]
    theta = z[:K2]
    phi = z[K2:] if optimise_source else None
    D = np.exp(np.clip(s.m_of(theta), lo, hi))
    f = s.f_of_phi(phi) if optimise_source else s.f_full.copy()
    return dict(D=D, f=f, alpha=path[chosen]["alpha"], flag=flag, wall_s=time.perf_counter() - tic,
                total_iters=int(sum(p["iters"] for p in path)), n_evals=s.n_evals,
                path=[{k: v for k, v in p.items() if k != "z"} for p in path])


# ---------------------------------------------------------------------------------------------
# truth (scoring only) and jobs
# ---------------------------------------------------------------------------------------------
def true_source(collection, key, N):
    import common as C
    e = C.entry(collection, key)
    return true_source_from_file(C.EXT / C.COLLECTIONS[collection] / e["files"]["ground_truth"], N)


def true_source_from_file(gt_path, N):
    """The hidden heater f(x, y) of a hidden-source dataset on the N x N inversion grid (scoring only)."""
    g = np.load(gt_path)
    if "source_true" not in g.files:
        return None
    expr = str(g["source_true"])
    xs = np.linspace(0, 1, N)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    if expr.strip() == "0":
        return np.zeros(N * N)
    x, y = sympy.symbols("x y")
    f = sympy.lambdify((x, y), sympy.sympify(expr), "numpy")(X, Y)
    f = np.asarray(f, float) * np.ones_like(X)
    f[[0, -1], :] = 0.0
    f[:, [0, -1]] = 0.0                                # the solver only applies f on interior nodes
    return f.ravel()


def _grid_check(ds):
    """The inversion grid is node-aligned with x = i h, y = j h and full-grid index i * N + j."""
    return ds.N


def run_job(arm, collection, key):
    import common as C
    ds = C.dataset(collection, key)
    s = SourceSolver(ds)
    f_true = true_source(collection, key, ds.N)
    if arm == "adjoint_truesrc":
        s.set_f(f_true)
        r = invert_src(s, optimise_source=False)
    elif arm in ("adjoint_src", "adjoint_src_pos"):
        r = invert_src(s, optimise_source=True, positive=(arm == "adjoint_src_pos"))
    else:
        raise ValueError(arm)
    D_true, _ = ds.ground_truth()
    m = C.field_scores(r["D"], D_true)
    # replay with the method's own source (the physics it inferred), and with no source
    s.set_f(r["f"])
    W, *_ = s.forward(r["D"])
    m["replay_over_sigma"] = float(np.sqrt(np.mean((s.predict(W) - s.d) ** 2)) / s.sigma_w)   # own physics
    m["replay_no_source_over_sigma"] = C.replay_rms_over_sigma(ds, r["D"])
    nf = np.linalg.norm(f_true)
    m["source_rel_L2"] = float(np.linalg.norm(r["f"] - f_true) / nf) if nf > 0 else float("nan")
    h2 = (1.0 / (ds.N - 1)) ** 2
    m["source_heat_true"] = float(f_true.sum() * h2)
    m["source_heat_est"] = float(r["f"].sum() * h2)
    m.update(alpha=r["alpha"], flag=r["flag"], total_wall_s=r["wall_s"], total_iters=r["total_iters"],
             n_evals=r["n_evals"], path=r["path"])
    return m, dict(D_hat=r["D"].reshape(ds.N, ds.N), f_hat=r["f"].reshape(ds.N, ds.N))


# ---------------------------------------------------------------------------------------------
# gradient check
# ---------------------------------------------------------------------------------------------
def test_source_gradient(seed=0, eps=1e-6):
    import common as C
    key = "H3u-A_smooth-q0.3__Fo0.1__uniform16__n50__random__eta0.05"
    ds = C.dataset("pinnwin", key)
    s = SourceSolver(ds, n_steps=40, k_src=6)
    s.K = 6
    P1 = A.bilinear_1d(ds.N, 6)
    s.P = sp.kron(P1, P1).tocsr()
    rng = np.random.default_rng(seed)
    z = np.r_[np.log(0.2) + 0.2 * rng.standard_normal(36), 0.5 * rng.standard_normal(36)]
    J, g = s.joint_objective(z, 1.0)
    errs = []
    for i in list(rng.choice(36, 4, replace=False)) + list(36 + rng.choice(36, 4, replace=False)):
        e = np.zeros_like(z); e[i] = eps
        fd = (s.joint_objective(z + e, 1.0, False) - s.joint_objective(z - e, 1.0, False)) / (2 * eps)
        errs.append(abs(fd - g[i]) / max(abs(fd), 1e-12))
    rel_D, rel_f = max(errs[:4]), max(errs[4:])
    return rel_D, rel_f
