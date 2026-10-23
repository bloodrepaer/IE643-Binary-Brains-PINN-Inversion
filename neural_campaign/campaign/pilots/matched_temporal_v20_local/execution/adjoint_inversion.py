"""
adjoint_inversion.py
--------------------
Classical PDE-constrained inversion (Subtask S3 / Deliverable 3) for

    du/dt = div( D(x,y) grad u ),   D = exp(m),   Dirichlet boundaries, known u0.

    min_theta  J(theta) = 1/(2 sigma^2) || Q u(m) - d ||^2  +  alpha R(m),
               m = P theta                  (bilinear: K x K control grid -> 65 x 65)
    s.t.       u solves the discrete heat equation EXACTLY (hard constraint)

Only the coefficient is optimised; u is eliminated by solving the PDE at every
evaluation ("reduced space"). Nothing is "trained": this is a per-dataset
optimisation, D is the optimisation variable, and there is no train/test split.

Discretisation (identical to integrators.py, 5-point, harmonic-mean links)
  States s = 0, 1, ..., S-1 live at t = 0, dt/2, dt, 2dt, ..., n_steps*dt.
  Every substep j (state j -> j+1) has the form

     u_{j+1} - u_j = h_j [ th_j F(w_{j+1}) + (1 - th_j) F(w_j) ]

  where w is the full-grid vector (interior u + boundary g(t)) and F(w) is the
  operator applied to w, restricted to interior nodes.
     substeps 0, 1 : backward Euler, h = dt/2, th = 1   (Rannacher start)
     substeps 2..  : Crank-Nicolson, h = dt,   th = 1/2
  In both cases h*th = dt/2, so ONE matrix  M = I - (dt/2) L  is factorised per
  D and reused for every forward step and every adjoint step.

Discrete adjoint (discretise-then-optimise; derived from the scheme above,
Rannacher steps included, so the gradient is exact for THIS discrete J)
  Lagrangian  J + sum_j lam_{j+1}^T [ u_{j+1} - u_j - h_j( th_j F(w_{j+1}) + (1-th_j) F(w_j) ) ]
  Backward sweep, lam_S = 0:
     M lam_s = lam_{s+1} + h_s (1 - th_s) L lam_{s+1} - dJ/du_s
  Gradient w.r.t. the conductance c_l of link l = (p, q):
     dJ/dc_l = sum_j h_j (lam_{j+1,p} - lam_{j+1,q}) [ th_j dw_{j+1} + (1-th_j) dw_j ],
     dw = w_p - w_q
  then chain rule  c_l = 2 D_p D_q / (D_p + D_q) / h^2,  D = exp(m),  m = P theta.
  One forward + one adjoint solve gives the gradient for ALL parameters.
  Verify with test_adjoint.py (Taylor test must show slope 2).

Regularisation weight alpha: discrepancy principle (same rule the PINN will
use). alpha is decreased geometrically with warm starts; the first alpha whose
data RMS reaches sigma is kept. If a noise/model-error floor stops RMS from
reaching sigma, the largest alpha within 2% of the best RMS is kept and the
run is flagged. Noise-free data have no sigma to aim for: alpha is decreased
until the misfit stops improving (flag "noise_free").

Usage
  python adjoint_inversion.py                         # 5 baseline datasets (one per tier)
  python adjoint_inversion.py --keys C_inclusion__Fo0.1__uniform8__n50__random__eta0.05
  python adjoint_inversion.py --filter tier=C_inclusion protocol=uniform8
  python adjoint_inversion.py --reg tv --control 16
  python adjoint_inversion.py --all                   # every dataset (hours)
"""
import argparse, csv, json, os, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize, minimize_scalar
import sympy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from integrators import BoundaryCondition, EDGES, edge_masks

D_MIN, D_MAX = 1e-3, 10.0


# =============================================================================
# 1. Loading a dataset (inversion inputs only: sensors/ + problem/)
# =============================================================================
class Dataset:
    """Everything the inversion is allowed to see for one manifest entry."""

    def __init__(self, root, entry):
        self.root, self.entry, self.key = root, entry, entry["key"]
        d = np.loadtxt(os.path.join(root, entry["files"]["sensors_csv"]), delimiter=",", skiprows=1, ndmin=2)
        self.x, self.y, self.t, self.u, sig = d.T
        self.sigma = float(sig[0])
        spec = json.load(open(os.path.join(root, entry["files"]["problem"])))
        self.T = float(spec["domain"]["T"])
        ts = sympy.Symbol("t")
        self.bc = BoundaryCondition({e: sympy.lambdify(ts, sympy.sympify(ex), "numpy")
                                     for e, ex in spec["boundary_condition"]["edges"].items()})
        ic = np.loadtxt(os.path.join(root, "problem", entry["problem"] + "__ic.csv"), delimiter=",", skiprows=1)
        self.N = int(round(np.sqrt(len(ic))))
        self.u0 = ic[:, 2].reshape(self.N, self.N)

    def ground_truth(self):
        """Answer key: ONLY for scoring after the inversion has finished."""
        g = np.load(os.path.join(self.root, self.entry["files"]["ground_truth"]))
        s = np.load(os.path.join(self.root, self.entry["files"]["sensors_npz"]))
        self.defects = json.loads(str(g["defects"])) if "defects" in g.files else []
        return g["D_inv"], s["u_clean"]


# =============================================================================
# 2. Discretisation: links, operator, time stepping, observation operator
# =============================================================================
def bilinear_1d(n_fine, n_coarse):
    """(n_fine x n_coarse) linear interpolation on [0,1] node grids."""
    xf, xc = np.linspace(0, 1, n_fine), np.linspace(0, 1, n_coarse)
    k = np.clip(np.searchsorted(xc, xf, side="right") - 1, 0, n_coarse - 2)
    w = (xf - xc[k]) / (xc[k + 1] - xc[k])
    rows = np.r_[np.arange(n_fine), np.arange(n_fine)]
    return sp.csr_matrix((np.r_[1 - w, w], (rows, np.r_[k, k + 1])), shape=(n_fine, n_coarse))


class Discretization:
    def __init__(self, N, T, n_steps, bc):
        self.N, self.T, self.n_steps, self.bc = N, T, n_steps, bc
        self.h = 1.0 / (N - 1)
        self.dt = T / n_steps
        idx = np.arange(N * N).reshape(N, N)
        inner = np.zeros((N, N), bool); inner[1:-1, 1:-1] = True
        self.I = idx[inner]                       # full-grid indices of interior nodes
        self.Bd = idx[~inner]
        self.nI = len(self.I)
        # all 5-point links (p, q); "active" ones touch at least one interior node
        p = np.r_[idx[:-1, :].ravel(), idx[:, :-1].ravel()]
        q = np.r_[idx[1:, :].ravel(), idx[:, 1:].ravel()]
        self.p_all, self.q_all = p, q
        act = inner.ravel()[p] | inner.ravel()[q]
        self.p, self.q = p[act], q[act]
        self.w_link = 1.0 / self.h**2
        # boundary values: full-grid vector g(t) = sum_e g_e(t) * mask_e
        masks = edge_masks((N, N))
        self.edge_mask = {e: masks[e].ravel().astype(float) for e in EDGES}
        # substeps: two backward-Euler half steps, then Crank-Nicolson
        dt = self.dt
        self.h_sub = np.r_[dt / 2, dt / 2, np.full(n_steps - 1, dt)]
        self.th_sub = np.r_[1.0, 1.0, np.full(n_steps - 1, 0.5)]
        self.t_state = np.r_[0.0, dt / 2, dt * np.arange(1, n_steps + 1)]
        self.S = len(self.t_state)
        self.G = np.stack([self.g(t) for t in self.t_state])          # (S, N^2)

    def g(self, t):
        vals = self.bc.at(t)
        return sum(vals[e] * self.edge_mask[e] for e in EDGES)

    def state_of_step(self, k):
        """State index holding u at t = k*dt."""
        return 0 if k == 0 else k + 1

    # --- operator ----------------------------------------------------------
    def conductance(self, D):
        Dp, Dq = D[self.p], D[self.q]
        return self.w_link * 2 * Dp * Dq / (Dp + Dq)

    def full_operator(self, c):
        n = self.N * self.N
        rows = np.r_[self.p, self.q, self.p, self.q]
        cols = np.r_[self.q, self.p, self.p, self.q]
        vals = np.r_[c, c, -c, -c]
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    # --- observation operator over all states --------------------------------
    def observation_operator(self, x, y, t):
        """Sparse (N_d x S*N^2): bilinear in space, linear in time between
        whole steps. Readings exactly on the time grid touch one state."""
        N, h = self.N, self.h
        i = np.clip(np.floor(x / h).astype(int), 0, N - 2); a = x / h - i
        j = np.clip(np.floor(y / h).astype(int), 0, N - 2); b = y / h - j
        sp_nodes = [(i, j, (1 - a) * (1 - b)), (i + 1, j, a * (1 - b)),
                    (i, j + 1, (1 - a) * b), (i + 1, j + 1, a * b)]
        tau = np.clip(t / self.dt, 0, self.n_steps)
        k = np.minimum(np.floor(tau + 1e-9).astype(int), self.n_steps)
        wt = np.clip(tau - k, 0, 1)
        rows, cols, vals = [], [], []
        r = np.arange(len(x))
        for (kk, ww) in ((k, 1 - wt), (np.minimum(k + 1, self.n_steps), wt)):
            s = np.where(kk == 0, 0, kk + 1)
            for ii, jj, sw in sp_nodes:
                rows.append(r); cols.append(s * N * N + ii * N + jj); vals.append(ww * sw)
        rows, cols, vals = map(np.concatenate, (rows, cols, vals))
        keep = vals != 0
        return sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(len(x), self.S * N * N))


# =============================================================================
# 3. Forward solve, misfit, adjoint gradient
# =============================================================================
class AdjointSolver:
    def __init__(self, ds, n_steps=None, control=16, reg="tikhonov", tv_eps=1e-2):
        self.ds = ds
        n_steps = n_steps or self._pick_steps(ds)
        self.disc = Discretization(ds.N, ds.T, n_steps, ds.bc)
        dz = self.disc
        self.Q = dz.observation_operator(ds.x, ds.y, ds.t)
        self.d = ds.u
        self.K = control
        P1 = bilinear_1d(ds.N, control)
        self.P = sp.kron(P1, P1).tocsr()                               # (N^2, K^2)
        self.reg, self.tv_eps = reg, tv_eps
        # noise weight: sigma from the data file; noise-free data get a tiny floor
        self.sigma_w = ds.sigma if ds.sigma > 0 else 1e-3 * np.sqrt(np.mean(ds.u**2))
        self.n_evals = 0
        self.u0_full = ds.u0.ravel().copy()

    @staticmethod
    def _pick_steps(ds):
        """Smallest step count >= 200 that puts every reading on the time grid."""
        for n in (200, 400, 800):
            tau = ds.t / ds.T * n
            if np.all(np.abs(tau - np.round(tau)) < 1e-4):
                return n
        return 400                          # otherwise: linear-in-time interpolation

    # --- parameters -----------------------------------------------------------
    def m_of(self, theta):
        return self.P @ theta

    # --- regulariser ------------------------------------------------------------
    def regulariser(self, m):
        dz = self.disc
        if self.reg == "tikhonov":          # 1/2 * integral |grad m|^2  (h^2 * 1/h^2 cancel)
            diff = m[dz.p_all] - m[dz.q_all]
            R = 0.5 * np.sum(diff**2)
            g = np.zeros_like(m)
            np.add.at(g, dz.p_all, diff); np.add.at(g, dz.q_all, -diff)
            return R, g
        # total variation, cell-wise forward differences
        N, h, eps = dz.N, dz.h, self.tv_eps
        M2 = m.reshape(N, N)
        gx = (M2[1:, :-1] - M2[:-1, :-1]) / h
        gy = (M2[:-1, 1:] - M2[:-1, :-1]) / h
        s = np.sqrt(gx**2 + gy**2 + eps)
        R = h**2 * np.sum(s)
        G = np.zeros((N, N))
        ax, ay = h * gx / s, h * gy / s       # h^2 * (g/s) * (1/h)
        G[1:, :-1] += ax; G[:-1, :-1] -= ax
        G[:-1, 1:] += ay; G[:-1, :-1] -= ay
        return R, G.ravel()

    # --- forward ------------------------------------------------------------------
    def forward(self, D):
        """Returns full-grid states W (S, N^2), the operator and its factorisation."""
        dz = self.disc
        c = dz.conductance(D)
        Lf = self.full_operator = dz.full_operator(c)
        LII = Lf[dz.I][:, dz.I]
        LIB = Lf[dz.I][:, dz.Bd]
        M = (sp.identity(dz.nI, format="csc") - (dz.dt / 2) * LII).tocsc()
        # M is symmetric positive definite: symmetric ordering, no pivoting
        # (~1.7x faster solves than the default LU settings)
        solve = spla.splu(M, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0,
                          options=dict(SymmetricMode=True)).solve
        W = dz.G.copy()
        W[0, dz.I] = self.u0_full[dz.I]
        u = W[0, dz.I]
        has_bc = np.any(dz.G[:, dz.Bd])
        bterm = (LIB @ dz.G[:, dz.Bd].T).T if has_bc else None      # (S, nI): L_IB g(t_s)
        for j in range(dz.S - 1):
            h, th = dz.h_sub[j], dz.th_sub[j]
            rhs = u + h * (1 - th) * (LII @ u) if th < 1 else u.copy()
            if has_bc:
                rhs += h * (1 - th) * bterm[j] + h * th * bterm[j + 1]
            u = solve(rhs)
            W[j + 1, dz.I] = u
        return W, c, LII, solve

    def predict(self, W):
        return self.Q @ W.ravel()

    # --- objective + gradient ----------------------------------------------------
    def objective(self, theta, alpha, want_grad=True):
        value = self.field_objective(self.m_of(theta), alpha, want_grad)
        if not want_grad:
            return value
        J, g_m = value
        return J, self.P.T @ g_m

    def field_objective(self, m, alpha, want_grad=True):
        """Reduced objective and gradient with respect to full-grid log D.

        Keeping the adjoint independent of the parameterisation lets a neural
        coefficient apply its own vector-Jacobian product after this sweep.
        """
        self.n_evals += 1
        dz = self.disc
        D = np.exp(m)
        W, c, LII, solve = self.forward(D)
        r = self.predict(W) - self.d
        s2 = self.sigma_w**2
        misfit = 0.5 * np.dot(r, r) / s2
        R, gR = self.regulariser(m)
        J = misfit + alpha * R
        self._last = dict(misfit=misfit, R=R, rms=np.sqrt(np.mean(r**2)), W=W, D=D)
        if not want_grad:
            return J
        # adjoint sweep
        dJdW = (self.Q.T @ (r / s2)).reshape(dz.S, -1)
        Lam = np.zeros_like(W)              # full-grid adjoint, zero on boundary nodes
        lam_next = np.zeros(dz.nI)
        for s in range(dz.S - 1, 0, -1):
            rhs = -dJdW[s, dz.I]
            if s < dz.S - 1:
                hs, ths = dz.h_sub[s], dz.th_sub[s]
                rhs = rhs + lam_next + hs * (1 - ths) * (LII @ lam_next)
            lam = solve(rhs)                # M is symmetric: M^T = M
            Lam[s, dz.I] = lam
            lam_next = lam
        # gradient w.r.t. link conductances
        dW = W[:, dz.p] - W[:, dz.q]                       # (S, links)
        dL = Lam[:, dz.p] - Lam[:, dz.q]
        h, th = dz.h_sub[:, None], dz.th_sub[:, None]
        g_c = np.sum(h * dL[1:] * (th * dW[1:] + (1 - th) * dW[:-1]), axis=0)
        # chain: c = w * 2 Dp Dq / (Dp + Dq)  ->  D  ->  m  ->  theta
        Dp, Dq = D[dz.p], D[dz.q]
        den = (Dp + Dq) ** 2
        g_D = np.zeros_like(D)
        np.add.at(g_D, dz.p, g_c * dz.w_link * 2 * Dq**2 / den)
        np.add.at(g_D, dz.q, g_c * dz.w_link * 2 * Dp**2 / den)
        g_m = D * g_D + alpha * gR
        return J, g_m

    # --- inversion driver -------------------------------------------------------
    def best_constant(self):
        """Stage 0: the single D that best fits the data (initial guess)."""
        K2 = self.K * self.K
        f = lambda lm: self.objective(np.full(K2, lm), 0.0, want_grad=False)
        res = minimize_scalar(f, bounds=(np.log(1e-3), np.log(5.0)), method="bounded",
                              options={"xatol": 1e-4})
        return float(res.x)

    def invert(self, alphas=None, maxiter=150, tau=1.0, verbose=True):
        tic = time.perf_counter()
        K2 = self.K * self.K
        m_c = self.best_constant()
        theta = np.full(K2, m_c)
        J0 = self.objective(theta, 0.0, want_grad=False)
        rms_const = self._last["rms"]
        if alphas is None:                  # scale: alpha_0 ~ number of readings
            alphas = len(self.d) * np.logspace(0, -6, 13)
        bounds = [(np.log(D_MIN), np.log(D_MAX))] * K2
        path, history = [], []
        chosen, flag = None, "reached_sigma"
        for a in alphas:
            hist = []
            res = minimize(self.objective, theta, args=(a,), jac=True, method="L-BFGS-B", bounds=bounds,
                           callback=lambda th: hist.append(self._last["misfit"]),
                           options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-6})
            theta = res.x
            self.objective(theta, a, want_grad=False)
            rms = self._last["rms"]
            path.append(dict(alpha=float(a), rms=float(rms), rms_over_sigma=float(rms / self.sigma_w),
                             iters=int(res.nit), theta=theta.copy()))
            history.append(hist)
            if verbose:
                print(f"    alpha={a:9.3e}  RMS/sigma={rms / self.sigma_w:7.3f}  iters={res.nit}")
            if self.ds.sigma > 0 and rms <= tau * self.sigma_w:
                chosen = len(path) - 1
                break
            if len(path) >= 3:              # floor: two steps with <1% improvement
                r0, r1, r2 = (path[-3]["rms"], path[-2]["rms"], path[-1]["rms"])
                if r1 > 0.99 * r0 and r2 > 0.99 * r1:
                    break
        if chosen is None:
            flag = "floor_above_sigma" if self.ds.sigma > 0 else "noise_free"
            best = min(p["rms"] for p in path)
            chosen = next(i for i, p in enumerate(path) if p["rms"] <= 1.02 * best)
        sel = path[chosen]
        theta = sel["theta"]
        self.objective(theta, sel["alpha"], want_grad=False)
        out = dict(theta=theta, m=self.m_of(theta), D=np.exp(self.m_of(theta)), W=self._last["W"],
                   alpha=sel["alpha"], rms=self._last["rms"], flag=flag,
                   path=[{k: v for k, v in p.items() if k != "theta"} for p in path],
                   history=history, D_const=float(np.exp(m_c)), rms_const=float(rms_const),
                   n_evals=self.n_evals, wall_s=time.perf_counter() - tic, n_steps=self.disc.n_steps)
        return out


# =============================================================================
# 4. Evaluation
# =============================================================================
def sensitivity_map(solver, W):
    """S(x,y) = integral_0^T |grad u|^2 dt on the inversion grid (no ground truth needed)."""
    dz = solver.disc
    N, h = dz.N, dz.h
    U = W.reshape(dz.S, N, N)
    gx, gy = np.gradient(U, h, axis=(1, 2))
    g2 = gx**2 + gy**2
    t = dz.t_state
    return np.sum(0.5 * (g2[1:] + g2[:-1]) * np.diff(t)[:, None, None], axis=0)


def defect_metrics(Dh, Dt, S, defects, N):
    """Tier F: per-defect detection, contrast recovery, localisation; Dice of
    the anomaly map. Background D0 is estimated as the median of D_hat (the
    true D0 is used only for the log-contrast of the truth)."""
    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N), indexing="ij")
    inner = np.zeros((N, N), bool); inner[1:-1, 1:-1] = True
    D0_hat = float(np.median(Dh[inner]))
    D0_true = float(np.median(Dt[inner]))
    a_hat = np.log(Dh / D0_hat)                       # estimated anomaly
    S_med = np.median(S[inner])
    true_mask = np.zeros((N, N), bool)
    rows = []
    for d in defects:
        dist = np.hypot(X - d["cx"], Y - d["cy"])
        inside = dist < d["r"]
        true_mask |= inside
        ell = np.log(d["factor"])
        rec = float(np.mean(a_hat[inside]) / ell) if inside.any() else np.nan
        win = dist < 3 * d["r"]
        w = np.clip(np.sign(ell) * a_hat, 0, None) * win
        if w.sum() > 0:
            cx, cy = (w * X).sum() / w.sum(), (w * Y).sum() / w.sum()
            loc = float(np.hypot(cx - d["cx"], cy - d["cy"]) / d["r"])
        else:
            loc = np.nan
        rows.append(dict(cx=d["cx"], cy=d["cy"], r=d["r"], factor=d["factor"],
                         type="low" if d["factor"] < 1 else "high",
                         contrast_recovery=rec, detected=bool(rec >= 0.25),
                         localisation_error_in_radii=loc,
                         sensitivity_rel=float(np.mean(S[inside]) / S_med) if inside.any() else np.nan))
    thr = 0.5 * min(abs(np.log(d["factor"])) for d in defects)
    pred_mask = (np.abs(a_hat) > thr) & inner
    dice = float(2 * (pred_mask & true_mask).sum() / max(pred_mask.sum() + true_mask.sum(), 1))
    bg = inner.copy()
    for d in defects:
        bg &= np.hypot(X - d["cx"], Y - d["cy"]) > 1.5 * d["r"]
    return dict(defect_dice=dice, defects_detected=int(sum(r["detected"] for r in rows)),
                n_defects=len(rows), D0_hat=D0_hat, D0_true=D0_true,
                background_rel_err=float(np.linalg.norm((Dh - Dt)[bg]) / np.linalg.norm(Dt[bg])),
                defects=rows)


def evaluate(ds, solver, out):
    """Metrics. replay_* use only the data; *_D use the answer key."""
    pred = solver.predict(out["W"])
    sig = solver.sigma_w
    m = dict(key=ds.key, sigma=ds.sigma, n_readings=len(ds.u),
             rms_replay_over_sigma=float(np.sqrt(np.mean((pred - ds.u) ** 2)) / sig),
             alpha=out["alpha"], flag=out["flag"], D_const=out["D_const"],
             rms_const_over_sigma=out["rms_const"] / sig, n_evals=out["n_evals"],
             wall_s=out["wall_s"], n_steps=out["n_steps"], control=solver.K, reg=solver.reg)
    S = sensitivity_map(solver, out["W"])
    try:
        D_true, u_clean = ds.ground_truth()
    except (FileNotFoundError, KeyError):
        return m, S, None
    Dh, Dt = out["D"].reshape(ds.N, ds.N), D_true
    inner = np.zeros_like(Dt, bool); inner[1:-1, 1:-1] = True
    hi = inner & (S >= np.median(S[inner]))
    lo = inner & ~hi
    rel = lambda mask: float(np.linalg.norm((Dh - Dt)[mask]) / np.linalg.norm(Dt[mask]))
    m.update(rel_L2_D=float(np.linalg.norm(Dh - Dt) / np.linalg.norm(Dt)),
             rmse_logD=float(np.sqrt(np.mean((np.log(Dh) - np.log(Dt)) ** 2))),
             rel_L2_D_high_sensitivity=rel(hi), rel_L2_D_low_sensitivity=rel(lo),
             rel_L2_D_constant_guess=float(np.linalg.norm(out["D_const"] - Dt) / np.linalg.norm(Dt)),
             rms_vs_clean_over_sigma=(float(np.sqrt(np.mean((pred - u_clean) ** 2)) / ds.sigma)
                                      if ds.sigma > 0 else None))
    if getattr(ds, "defects", None):
        m.update(defect_metrics(Dh, Dt, S, ds.defects, ds.N))
    return m, S, Dt


def plot(ds, solver, out, metrics, S, D_true, path_png):
    N = ds.N
    Dh = out["D"].reshape(N, N)
    ext = [0, 1, 0, 1]
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    vmin = min(Dh.min(), D_true.min()) if D_true is not None else Dh.min()
    vmax = max(Dh.max(), D_true.max()) if D_true is not None else Dh.max()
    if D_true is not None:
        im = ax[0, 0].imshow(D_true.T, origin="lower", extent=ext, cmap="plasma", vmin=vmin, vmax=vmax)
        ax[0, 0].set_title("true D (answer key)"); plt.colorbar(im, ax=ax[0, 0])
    im = ax[0, 1].imshow(Dh.T, origin="lower", extent=ext, cmap="plasma", vmin=vmin, vmax=vmax)
    ax[0, 1].plot(ds.x, ds.y, "w.", ms=3)
    method_label = metrics.get("method", f"adjoint, {solver.reg}, {solver.K}x{solver.K}")
    ax[0, 1].set_title(f"estimated D ({method_label})"); plt.colorbar(im, ax=ax[0, 1])
    if D_true is not None:
        err = np.log(Dh / D_true)
        lim = np.abs(err).max()
        im = ax[0, 2].imshow(err.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax[0, 2].set_title(f"log(D_est / D_true)   relL2 = {metrics['rel_L2_D']:.3f}"); plt.colorbar(im, ax=ax[0, 2])
    lS = np.log10(S.T + 1e-300)
    im = ax[1, 0].imshow(lS, origin="lower", extent=ext, cmap="viridis", vmin=lS.max() - 4, vmax=lS.max())
    ax[1, 0].plot(ds.x, ds.y, "r.", ms=3)
    ax[1, 0].set_title("sensitivity  log10 ∫|∇u|² dt"); plt.colorbar(im, ax=ax[1, 0])
    if D_true is not None:
        mid = N // 2
        xs = np.linspace(0, 1, N)
        ax[1, 1].plot(xs, D_true[:, mid], "k-", label="true, y=0.5")
        ax[1, 1].plot(xs, Dh[:, mid], "C3--", label="estimate, y=0.5")
        ax[1, 1].plot(xs, D_true[mid, :], "k:", label="true, x=0.5")
        ax[1, 1].plot(xs, Dh[mid, :], "C0--", label="estimate, x=0.5")
        ax[1, 1].set_title("1D profiles"); ax[1, 1].legend(fontsize=8)
    for d in getattr(ds, "defects", None) or []:
        for a_ in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0]):
            a_.add_patch(plt.Circle((d["cx"], d["cy"]), d["r"], fill=False, ec="k", ls="--", lw=1))
    if "defect_dice" in metrics:
        ax[0, 1].set_title(f"estimated D: {metrics['defects_detected']}/{metrics['n_defects']} defects found, "
                           f"Dice {metrics['defect_dice']:.2f}", fontsize=10)
    al = [p["alpha"] for p in out["path"]]; rs = [p["rms_over_sigma"] for p in out["path"]]
    ax[1, 2].loglog(al, rs, "o-"); ax[1, 2].axhline(1.0, color="k", ls="--", lw=1, label="RMS = σ")
    ax[1, 2].axvline(out["alpha"], color="C3", lw=1, label=f"chosen α ({out['flag']})")
    ax[1, 2].invert_xaxis(); ax[1, 2].set(xlabel="α", ylabel="RMS_data / σ", title="discrepancy principle")
    ax[1, 2].legend(fontsize=8)
    for a in ax.ravel()[:2]:
        a.set_xlabel("x"); a.set_ylabel("y")
    fig.suptitle(f"{ds.key}\nRMS_replay/σ = {metrics['rms_replay_over_sigma']:.3f}   "
                 f"({metrics['n_readings']} readings, {metrics['wall_s']:.0f} s)", fontsize=11)
    fig.tight_layout(); fig.savefig(path_png, dpi=110); plt.close(fig)


# =============================================================================
# 5. CLI
# =============================================================================
BASELINE_KEYS = [f"{t}__Fo0.1__uniform8__n50__random__eta0.05"
                 for t in ("A_smooth", "B_gaussian_bump", "C_inclusion", "D_layered", "E_grf", "F_defects")]


def select(manifest, a):
    if a.keys:
        return [e for e in manifest if e["key"] in set(a.keys)]
    if a.filter:
        conds = dict(f.split("=", 1) for f in a.filter)
        return [e for e in manifest if all(str(e.get(k)) == v for k, v in conds.items())]
    if a.all:
        return manifest
    return [e for e in manifest if e["key"] in set(BASELINE_KEYS)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets_v2_rk4_zero")
    ap.add_argument("--out", default="results_adjoint")
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--filter", nargs="*", help="manifest field=value pairs, e.g. tier=C_inclusion eta=0.05")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--reg", default="tikhonov", choices=("tikhonov", "tv"))
    ap.add_argument("--control", type=int, default=16, help="control grid K (D has K x K parameters)")
    ap.add_argument("--maxiter", type=int, default=150, help="L-BFGS iterations per alpha")
    ap.add_argument("--n-steps", type=int, default=None, help="CN steps (default: 200, or 400 if needed)")
    ap.add_argument("--chunk", nargs=2, type=int, metavar=("I", "N"),
                    help="run only every N-th selected dataset, starting at I (0-based)")
    a = ap.parse_args()

    manifest = json.load(open(os.path.join(a.data, "manifest.json")))
    jobs = select(manifest, a)
    if a.chunk:
        i, n = a.chunk
        jobs = jobs[i::n]
    if not jobs:
        raise SystemExit("no datasets matched")
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for k, e in enumerate(jobs, 1):
        print(f"[{k}/{len(jobs)}] {e['key']}  ({e['n_readings']} readings, sigma={e['sigma']:.2e})")
        ds = Dataset(a.data, e)
        solver = AdjointSolver(ds, n_steps=a.n_steps, control=a.control, reg=a.reg)
        out = solver.invert(maxiter=a.maxiter)
        metrics, S, D_true = evaluate(ds, solver, out)
        d = os.path.join(a.out, e["key"]); os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "D_hat.npz"), D_hat=out["D"].reshape(ds.N, ds.N),
                            m_hat=out["m"].reshape(ds.N, ds.N), theta=out["theta"], control=a.control,
                            x=np.linspace(0, 1, ds.N), y=np.linspace(0, 1, ds.N), sensitivity=S,
                            alpha=out["alpha"])
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump({**metrics, "alpha_path": out["path"]}, f, indent=2, default=float)
        plot(ds, solver, out, metrics, S, D_true, os.path.join(d, "summary.png"))
        line = (f"    -> RMS_replay/σ={metrics['rms_replay_over_sigma']:.3f}  flag={out['flag']}  "
                f"{out['wall_s']:.0f}s, {out['n_evals']} PDE solves")
        if "rel_L2_D" in metrics:
            line += (f"\n       relL2(D)={metrics['rel_L2_D']:.3f}  (constant guess {metrics['rel_L2_D_constant_guess']:.3f})"
                     f"  high/low sensitivity {metrics['rel_L2_D_high_sensitivity']:.3f}/"
                     f"{metrics['rel_L2_D_low_sensitivity']:.3f}")
        if "defect_dice" in metrics:
            line += (f"\n       defects found {metrics['defects_detected']}/{metrics['n_defects']}, "
                     f"Dice {metrics['defect_dice']:.2f}, background err {metrics['background_rel_err']:.3f}")
            for r in metrics["defects"]:
                line += (f"\n         {r['type']:4s} x{r['factor']:.2f} at ({r['cx']:.2f},{r['cy']:.2f}): "
                         f"contrast recovered {r['contrast_recovery']:+.2f}, "
                         f"loc err {r['localisation_error_in_radii']:.2f} r, sensitivity {r['sensitivity_rel']:.2f}x median")
        print(line)
        rows.append({k2: v for k2, v in metrics.items() if k2 != "defects"})
    path = os.path.join(a.out, "summary.csv")
    old = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    done = {r["key"] for r in rows}
    allrows = [r for r in old if r["key"] not in done] + rows
    fields = list(dict.fromkeys(k for r in allrows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(allrows)
    print(f"\nResults in {a.out}/  (summary.csv, one folder per dataset)")


if __name__ == "__main__":
    main()
