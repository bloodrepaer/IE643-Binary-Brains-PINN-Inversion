"""
solvers.py -- the classical adjoint inversion with three interchangeable time integrators.

Everything except the time-stepping is the project's standard method (core/adjoint_inversion.py):
same 65 x 65 finite-volume operator with harmonic-mean links, 16 x 16 bilinear control grid on log D,
Tikhonov regulariser, L-BFGS-B (150 iterations per alpha), best-constant start and discrepancy-principle
alpha path. Each integrator has its own EXACT discrete adjoint, so every gradient is the true gradient of
that integrator's discrete objective (checked by test_solvers.py).

  CN   Crank-Nicolson with the Rannacher start (two backward-Euler half steps): the project default,
       taken unchanged from AdjointSolver; only n_steps is fixed by the study.
  BE   backward Euler, every step: M = I - dt L, one factorisation per D.
  RK4  classical explicit Runge-Kutta 4. Explicit, so dt must satisfy dt * rho(L) <= 2.78; the study
       caps D at D_CAP so one fixed step is stable for every candidate D (checked on every solve).

Study-wide change for all three: D is bounded to [1e-3, D_CAP] (default 1.0; the true field spans
0.05-0.5). The project default is [1e-3, 10]; RK4 at D = 10 would need about 43,000 steps per solve.
RK4 and BE support zero boundary values only (the A_smooth datasets used here).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize_scalar

import adjoint_inversion as A

D_CAP = 1.0
A.D_MAX = D_CAP                       # bounds used by AdjointSolver.invert for every solver below
RK4_STABILITY = 2.785                 # |real-axis stability limit| of classical RK4


def plain_observation_operator(dz, x, y, t):
    """Same bilinear-space / linear-time operator as Discretization.observation_operator, for time grids
    whose state index equals the step index (BE and RK4 have no Rannacher half step)."""
    N, h = dz.N, dz.h
    i = np.clip(np.floor(x / h).astype(int), 0, N - 2); a = x / h - i
    j = np.clip(np.floor(y / h).astype(int), 0, N - 2); b = y / h - j
    nodes = [(i, j, (1 - a) * (1 - b)), (i + 1, j, a * (1 - b)), (i, j + 1, (1 - a) * b), (i + 1, j + 1, a * b)]
    tau = np.clip(t / dz.dt, 0, dz.n_steps)
    k = np.minimum(np.floor(tau + 1e-9).astype(int), dz.n_steps)
    wt = np.clip(tau - k, 0, 1)
    rows, cols, vals = [], [], []
    r = np.arange(len(x))
    for kk, ww in ((k, 1 - wt), (np.minimum(k + 1, dz.n_steps), wt)):
        for ii, jj, sw in nodes:
            rows.append(r); cols.append(kk * N * N + ii * N + jj); vals.append(ww * sw)
    rows, cols, vals = map(np.concatenate, (rows, cols, vals))
    keep = vals != 0
    return sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(len(x), dz.S * N * N))


class PlainDiscretization(A.Discretization):
    """Whole steps only: states at t = k dt, k = 0..n_steps (used by BE and RK4)."""

    def __init__(self, N, T, n_steps, bc):
        super().__init__(N, T, n_steps, bc)
        self.h_sub = np.full(n_steps, self.dt)
        self.th_sub = np.ones(n_steps)               # backward Euler in the generic CN/BE formulas
        self.t_state = np.arange(n_steps + 1) * self.dt
        self.S = n_steps + 1
        self.G = np.stack([self.g(t) for t in self.t_state])

    def state_of_step(self, k):
        return k

    def observation_operator(self, x, y, t):
        return plain_observation_operator(self, x, y, t)


class _Capped:
    """Best-constant start bounded by D_CAP (the project searches up to D = 5)."""
    name = "?"

    def best_constant(self):
        K2 = self.K * self.K
        f = lambda lm: self.objective(np.full(K2, lm), 0.0, want_grad=False)
        res = minimize_scalar(f, bounds=(np.log(A.D_MIN), np.log(D_CAP)), method="bounded",
                              options={"xatol": 1e-4})
        return float(res.x)


class CNSolver(_Capped, A.AdjointSolver):
    """Project default (CN + Rannacher), unchanged apart from the fixed step count and the D cap."""
    name = "CN"

    def __init__(self, ds, n_steps, **kw):
        super().__init__(ds, n_steps=n_steps, **kw)


class BESolver(_Capped, A.AdjointSolver):
    """Backward Euler at every step, with its exact discrete adjoint.

    The generic adjoint sweep and conductance gradient of AdjointSolver.field_objective are written for
    substeps u_{j+1} - u_j = h_j [th_j F(w_{j+1}) + (1 - th_j) F(w_j)] with h_j th_j shared by all steps;
    with h = dt and th = 1 they are exactly backward Euler's, so only the forward solve (M = I - dt L)
    and the time grid change."""
    name = "BE"

    def __init__(self, ds, n_steps, **kw):
        super().__init__(ds, n_steps=n_steps, **kw)
        self.disc = PlainDiscretization(ds.N, ds.T, n_steps, ds.bc)
        self.Q = self.disc.observation_operator(ds.x, ds.y, ds.t)

    def forward(self, D):
        dz = self.disc
        c = dz.conductance(D)
        Lf = self.full_operator = dz.full_operator(c)
        LII = Lf[dz.I][:, dz.I]
        LIB = Lf[dz.I][:, dz.Bd]
        M = (sp.identity(dz.nI, format="csc") - dz.dt * LII).tocsc()
        solve = spla.splu(M, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0,
                          options=dict(SymmetricMode=True)).solve
        W = dz.G.copy()
        W[0, dz.I] = self.u0_full[dz.I]
        u = W[0, dz.I]
        has_bc = np.any(dz.G[:, dz.Bd])
        bterm = (LIB @ dz.G[:, dz.Bd].T).T if has_bc else None
        for j in range(dz.S - 1):
            rhs = u.copy()
            if has_bc:
                rhs += dz.dt * bterm[j + 1]
            u = solve(rhs)
            W[j + 1, dz.I] = u
        return W, c, LII, solve


class RK4Solver(_Capped, A.AdjointSolver):
    """Classical explicit RK4 for du/dt = L u (zero walls), with its exact discrete adjoint.

    For a linear autonomous system one RK4 step is u_{n+1} = P(dt L) u_n with
    P(z) = 1 + z + z^2/2 + z^3/6 + z^4/24 (computed here by the four RK4 stages). L is symmetric, so
      adjoint:   lam_S = -dJ/du_S,   lam_s = P(dt L) lam_{s+1} - dJ/du_s
      gradient:  dJ/dc_l = sum_n sum_{k=1..4} dt^k/k! sum_{i+j=k-1} [L^j lam_{n+1}]_pq [L^i u_n]_pq
    where [v]_pq = v_p - v_q over link l = (p, q), with zero boundary values."""
    name = "RK4"

    def __init__(self, ds, n_steps, chunk=200, **kw):
        super().__init__(ds, n_steps=n_steps, **kw)
        self.disc = PlainDiscretization(ds.N, ds.T, n_steps, ds.bc)
        self.Q = self.disc.observation_operator(ds.x, ds.y, ds.t)
        if np.any(self.disc.G[:, self.disc.Bd]):
            raise ValueError("RK4Solver supports zero boundary values only")
        self.chunk = chunk
        self.max_dt_rho = 0.0

    def _operator(self, D):
        dz = self.disc
        c = dz.conductance(D)
        LII = dz.full_operator(c)[dz.I][:, dz.I].tocsr()
        rho = float(np.max(2 * np.abs(LII.diagonal())))        # Gershgorin bound on the spectral radius
        dt_rho = dz.dt * rho
        self.max_dt_rho = max(self.max_dt_rho, dt_rho)
        if dt_rho > RK4_STABILITY:
            raise FloatingPointError(f"RK4 unstable: dt*rho = {dt_rho:.3f} > {RK4_STABILITY} (D max {D.max():.3g})")
        return c, LII

    def forward(self, D):
        dz = self.disc
        c, LII = self._operator(D)
        dt = dz.dt
        U = np.empty((dz.S, dz.nI))
        u = self.u0_full[dz.I].copy()
        U[0] = u
        for n in range(dz.S - 1):
            k1 = LII @ u
            k2 = LII @ (u + 0.5 * dt * k1)
            k3 = LII @ (u + 0.5 * dt * k2)
            k4 = LII @ (u + dt * k3)
            u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            U[n + 1] = u
        W = np.zeros((dz.S, dz.N * dz.N))
        W[:, dz.I] = U
        return W, c, LII, None

    def _step_adjoint(self, LII, v):
        dt = self.disc.dt
        k1 = LII @ v
        k2 = LII @ (v + 0.5 * dt * k1)
        k3 = LII @ (v + 0.5 * dt * k2)
        k4 = LII @ (v + dt * k3)
        return v + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)   # P(dt L)^T v = P(dt L) v (L symmetric)

    def field_objective(self, m, alpha, want_grad=True):
        self.n_evals += 1
        dz = self.disc
        D = np.exp(m)
        W, c, LII, _ = self.forward(D)
        r = self.predict(W) - self.d
        s2 = self.sigma_w ** 2
        misfit = 0.5 * np.dot(r, r) / s2
        R, gR = self.regulariser(m)
        J = misfit + alpha * R
        self._last = dict(misfit=misfit, R=R, rms=np.sqrt(np.mean(r ** 2)), W=W, D=D)
        if not want_grad:
            return J
        dJdU = (self.Q.T @ (r / s2)).reshape(dz.S, -1)[:, dz.I]
        Lam = np.zeros((dz.S, dz.nI))
        lam = -dJdU[dz.S - 1]
        Lam[dz.S - 1] = lam
        for s in range(dz.S - 2, 0, -1):
            lam = self._step_adjoint(LII, lam) - dJdU[s]
            Lam[s] = lam
        # conductance gradient, in chunks of steps: sum over n of lam_{n+1} and u_n terms
        U = W[:, dz.I]
        dt = dz.dt
        coef = {1: dt, 2: dt ** 2 / 2, 3: dt ** 3 / 6, 4: dt ** 4 / 24}
        Nn = dz.N * dz.N
        p, q = dz.p, dz.q
        g_c = np.zeros(len(p))

        def link_diff(V):                              # V: (nI, m) interior values -> (links, m)
            full = np.zeros((Nn, V.shape[1]))
            full[dz.I] = V
            return full[p] - full[q]

        for a in range(0, dz.S - 1, self.chunk):
            b = min(a + self.chunk, dz.S - 1)
            Uc = U[a:b].T.copy()                        # u_n, n = a..b-1
            Lc = Lam[a + 1:b + 1].T.copy()              # lam_{n+1}
            up, lp = [Uc], [Lc]
            for _ in range(3):
                up.append(LII @ up[-1]); lp.append(LII @ lp[-1])
            du = [link_diff(V) for V in up]
            dl = [link_diff(V) for V in lp]
            for i in range(4):
                for j in range(4 - i):
                    g_c += coef[i + j + 1] * np.sum(dl[j] * du[i], axis=1)
        Dp, Dq = D[p], D[q]
        den = (Dp + Dq) ** 2
        g_D = np.zeros_like(D)
        np.add.at(g_D, p, g_c * dz.w_link * 2 * Dq ** 2 / den)
        np.add.at(g_D, q, g_c * dz.w_link * 2 * Dp ** 2 / den)
        return J, D * g_D + alpha * gR


SOLVERS = {"CN": CNSolver, "BE": BESolver, "RK4": RK4Solver}


def make(name, ds, n_steps, **kw):
    return SOLVERS[name](ds, n_steps, **kw)
