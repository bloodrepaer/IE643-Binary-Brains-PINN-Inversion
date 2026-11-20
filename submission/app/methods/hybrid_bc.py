"""App copy of extended_studies/novelty/hybrid_bc.py.


hybrid_bc.py -- hybrid E1b: the PINN supplies the physics the adjoint is missing (the wall temperatures),
the adjoint supplies D. Plus the fair classical competitor (the adjoint estimating the walls itself) and
the bound (the adjoint given the true walls).

Why: in E1 every hybrid handed the final step to an adjoint that assumes zero walls, so on the hidden-wall
datasets (Study 3, H2u) they all inherited its model error (relL2 ~ 4) while the PINN alone got ~ 0.35.
A warm start cannot fix a wrong model; giving the adjoint better walls can.

Arms (all on the 8 H2u datasets: wall temperature hidden, assumed zero in the problem file):
  adjoint_truebc     adjoint with the TRUE wall temperatures (a bound, not a method: uses hidden information)
  adjoint_joint      classical joint inversion: D and the walls together, walls start at the assumed zero.
                     Walls: each edge uniform along its length, piecewise linear in time (17 knots); this
                     matches the structure of the hidden truth, so it is a generous model for the adjoint.
  hybrid_bc          the boundary-aware PINN's u on the walls (every boundary node, every solver step) is
                     handed to the adjoint as fixed Dirichlet data; the adjoint then inverts D as usual
  hybrid_bc_joint    joint inversion as adjoint_joint, but the walls start from the PINN's walls (projected
                     onto the knots). The joint problem is not convex, so here the start can matter.
Everything else is the project's adjoint unchanged (CN, 16 x 16 controls, Tikhonov on grad log D,
L-BFGS-B 150 iterations per alpha, alpha path N_d 10^0 ... 10^-6, discrepancy principle). In the joint arms
the walls get a time-smoothness penalty with the same alpha: alpha * 1/2 sum (b_{k+1} - b_k)^2.
Costs: hybrid totals include the PINN's training time.

The wall gradient is the discrete adjoint of the same CN scheme (see wall_grad); test_hybrid_bc() checks it
against finite differences.
"""
import time

import numpy as np
from scipy.optimize import minimize

import adjoint_inversion as A
from integrators import EDGES, edge_masks

N_T_KNOTS = 17
EDGE_ORDER = ("left", "right", "bottom", "top")


# ---------------------------------------------------------------------------------------------
# wall parameterisation
# ---------------------------------------------------------------------------------------------
def hat_matrix(t_knots, t):
    """(len(t) x len(t_knots)) piecewise-linear interpolation."""
    k = np.clip(np.searchsorted(t_knots, t, side="right") - 1, 0, len(t_knots) - 2)
    w = (t - t_knots[k]) / (t_knots[k + 1] - t_knots[k])
    H = np.zeros((len(t), len(t_knots)))
    H[np.arange(len(t)), k] = 1 - w
    H[np.arange(len(t)), k + 1] = w
    return H


class WallSolver(A.AdjointSolver):
    """The project's AdjointSolver with Dirichlet wall values that can be set (G) or optimised (b)."""

    def __init__(self, ds, n_t=N_T_KNOTS, n_s=1, **kw):
        super().__init__(ds, **kw)
        dz = self.disc
        N = ds.N
        masks = edge_masks((N, N))
        self.E = np.stack([masks[e].ravel().astype(float) for e in EDGE_ORDER], 1)        # (N^2, 4) edge masks
        # wall basis: per edge, n_s hat functions along the edge (n_s = 1: uniform along the edge)
        ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        pos = {"left": jj, "right": jj, "bottom": ii, "top": ii}                          # coordinate along the edge
        cols = []
        for e in EDGE_ORDER:
            if n_s == 1:
                cols.append(self.E[:, EDGE_ORDER.index(e)])
                continue
            Hs = hat_matrix(np.linspace(0, 1, n_s), pos[e].ravel() / (N - 1))           # (N^2, n_s)
            cols += [self.E[:, EDGE_ORDER.index(e)] * Hs[:, k] for k in range(n_s)]
        self.Bmat = np.stack(cols, 1)                                                      # (N^2, 4 n_s)
        self.n_s = n_s
        self.t_knots = np.linspace(0, ds.T, n_t)
        self.Ht = hat_matrix(self.t_knots, dz.t_state)                                   # (S, n_t)
        self.n_b = 4 * n_s * n_t
        self.G0 = dz.G.copy()

    # walls ------------------------------------------------------------------------------------
    def set_G(self, G):
        self.disc.G = np.asarray(G, float).copy()

    def G_of_b(self, b):
        B = b.reshape(self.Bmat.shape[1], -1).T      # (n_t, 4 n_s)
        return self.Ht @ B @ self.Bmat.T             # (S, N^2); zero inside

    def b_of_G(self, G):
        """Least-squares fit of a full wall field G (S, N^2) onto the wall basis (space, then time)."""
        Bd = self.disc.Bd
        Cs, *_ = np.linalg.lstsq(self.Bmat[Bd], G[:, Bd].T, rcond=None)     # (4 n_s, S)
        B, *_ = np.linalg.lstsq(self.Ht, Cs.T, rcond=None)                 # (n_t, 4 n_s)
        return B.T.ravel()

    # objective in (theta, b) ------------------------------------------------------------------
    def joint_objective(self, z, alpha, want_grad=True):
        K2 = self.K ** 2
        theta, b = z[:K2], z[K2:]
        self.set_G(self.G_of_b(b))
        out = self.field_objective_walls(self.m_of(theta), alpha, want_grad)
        Rb, gR = self.wall_regulariser(b)
        if not want_grad:
            return out + alpha * Rb
        J, g_m, g_G = out
        g_b = (self.Ht.T @ g_G @ self.Bmat).T.ravel() + alpha * gR
        self._last["Rb"] = Rb
        return J + alpha * Rb, np.r_[self.P.T @ g_m, g_b]

    def wall_regulariser(self, b):
        """1/2 sum of squared differences between neighbouring knots, in time and along each edge."""
        Bk = b.reshape(4, self.n_s, -1)
        dt_ = np.diff(Bk, axis=2)
        ds_ = np.diff(Bk, axis=1)
        R = 0.5 * (np.sum(dt_ ** 2) + np.sum(ds_ ** 2))
        g = np.zeros_like(Bk)
        g[:, :, 1:] += dt_; g[:, :, :-1] -= dt_
        g[:, 1:, :] += ds_; g[:, :-1, :] -= ds_
        return R, g.ravel()

    def field_objective_walls(self, m, alpha, want_grad=True):
        """AdjointSolver.field_objective, plus the gradient with respect to the full wall field G."""
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
        # walls: dJ/dg_s = dJ/dW_s|B - L_IB^T [ h_s (1 - th_s) lam_{s+1} + h_{s-1} th_{s-1} lam_s ]
        LIB = self.full_operator[dz.I][:, dz.Bd]
        LamI = Lam[:, dz.I]                                   # (S, nI); row 0 unused (zero)
        hs, ths = dz.h_sub, dz.th_sub                         # length S - 1 (substep j: state j -> j+1)
        coef = np.zeros_like(LamI)
        coef[:-1] += (hs * (1 - ths))[:, None] * LamI[1:]     # from substep j = s
        coef[1:] += (hs * ths)[:, None] * LamI[1:]            # from substep j = s - 1
        g_G = np.zeros_like(W)
        g_G[:, dz.Bd] = dJdW[:, dz.Bd] - (LIB.T @ coef.T).T
        return J, g_m, g_G


# ---------------------------------------------------------------------------------------------
# inversion driver (same rules as AdjointSolver.invert / hybrid.invert_from)
# ---------------------------------------------------------------------------------------------
def invert_joint(s, theta0, b0, alphas=None, maxiter=150, tau=1.0, optimise_walls=True):
    tic = time.perf_counter()
    K2 = s.K ** 2
    if alphas is None:
        alphas = len(s.d) * np.logspace(0, -6, 13)
    lo, hi = np.log(A.D_MIN), np.log(A.D_MAX)
    if optimise_walls:
        z = np.r_[np.clip(theta0, lo, hi), b0]
        bounds = [(lo, hi)] * K2 + [(None, None)] * len(b0)
        fun = s.joint_objective
    else:
        z = np.clip(np.asarray(theta0, float), lo, hi)
        bounds = [(lo, hi)] * K2
        fun = s.objective
    path, chosen, flag, n0 = [], None, "reached_sigma", s.n_evals
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
    zz = path[chosen]["z"]
    G = s.G_of_b(zz[K2:]) if optimise_walls else s.disc.G.copy()
    return dict(D=np.exp(s.m_of(zz[:K2])), G=G, alpha=path[chosen]["alpha"], flag=flag,
                path=[{k: v for k, v in p.items() if k != "z"} for p in path],
                total_iters=int(sum(p["iters"] for p in path)), n_evals=int(s.n_evals - n0),
                wall_s=time.perf_counter() - tic)


# ---------------------------------------------------------------------------------------------
# wall fields: true, PINN, scoring
# ---------------------------------------------------------------------------------------------
def true_G(s, collection_root, entry):
    z = np.load(f"{collection_root}/{entry['files']['ground_truth']}")
    order = str(z["bc_true_edge_order"]).split(",")
    tb, vals = z["bc_true_t"], z["bc_true_edges"]
    E = np.stack([s.E[:, EDGE_ORDER.index(e)] for e in order], 1)
    V = np.stack([np.interp(s.disc.t_state, tb, vals[:, i]) for i in range(len(order))], 1)   # (S, 4)
    return V @ E.T


def wall_error(s, G, G_true):
    """RMS over boundary nodes and solver states, relative to the RMS of the true walls."""
    Bd = s.disc.Bd
    num = np.sqrt(np.mean((G[:, Bd] - G_true[:, Bd]) ** 2))
    den = np.sqrt(np.mean(G_true[:, Bd] ** 2))
    return float(num / den) if den > 0 else float(num)


def pinn_bc_with_walls(collection_root, key, epochs, s):
    """Study 3's boundary-aware PINN (free walls) trained as in E1; also returns its u on every boundary
    node at every solver state time of s (the walls handed to the adjoint)."""
    import tempfile
    import torch
    import pinn_inversion_bc as PB
    grab = {}

    def keep(D_hat, model, bundle, device):
        grab.update(model=model, device=device)
        return dict(chi2_red_solver=float("nan"), chi2_red_u_net=float("nan"), chi2_solver_dist_from_1=float("nan"),
                    chi2_u_net_dist_from_1=float("nan"), slide16_regime="n/a")
    PB.evaluate_slide16_chi2_metric = keep
    PB.plot_summary_figure = lambda *a, **k: None
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PB.PINNConfig()
        cfg.data_root, cfg.output_dir, cfg.exp_id, cfg.epochs = str(collection_root), tmp, key, epochs
        t0 = time.time()
        PB.train_single_experiment(cfg, run_tag="r")
        wall = time.time() - t0
        D = np.load(f"{tmp}/r/D_hat.npz")["D_hat"]
    model, device = grab["model"], grab["device"]
    dz = s.disc
    N = dz.N
    xg = np.linspace(0, 1, N)
    XX, YY = np.meshgrid(xg, xg, indexing="ij")
    xb, yb = XX.ravel()[dz.Bd], YY.ravel()[dz.Bd]
    G = np.zeros((dz.S, N * N))
    tic = time.time()
    model.eval()
    with torch.no_grad():
        for k, tk in enumerate(dz.t_state):
            xyt = torch.tensor(np.c_[xb, yb, np.full_like(xb, tk)], dtype=torch.float32, device=device)
            G[k, dz.Bd] = model.forward_u(xyt).cpu().numpy().ravel()
    return D, wall, G, time.time() - tic


# ---------------------------------------------------------------------------------------------
def test_hybrid_bc(seed=0):
    """Finite-difference check of the joint gradient (theta and b) on a small H2u problem."""
    import common as C
    key = "H2u-A_smooth-pulsedtop-a1__Fo0.1__uniform16__n50__random__eta0.05"
    ds = C.dataset("pinnwin", key)
    s = WallSolver(ds, n_steps=200, n_s=3)
    rng = np.random.default_rng(seed)
    z = np.r_[np.log(0.25) + 0.3 * rng.standard_normal(s.K ** 2), 0.3 * rng.standard_normal(s.n_b)]
    v = rng.standard_normal(z.size)
    J, g = s.joint_objective(z, 1.0)
    out = []
    for eps in (1e-3, 1e-4, 1e-5):
        Jp = s.joint_objective(z + eps * v, 1.0, want_grad=False)
        Jm = s.joint_objective(z - eps * v, 1.0, want_grad=False)
        out.append(abs((Jp - Jm) / (2 * eps) - g @ v) / abs(g @ v))
    vb = np.r_[np.zeros(s.K ** 2), v[s.K ** 2:]]
    Jp = s.joint_objective(z + 1e-4 * vb, 1.0, want_grad=False)
    Jm = s.joint_objective(z - 1e-4 * vb, 1.0, want_grad=False)
    wall_only = abs((Jp - Jm) / 2e-4 - g @ vb) / abs(g @ vb)
    return out, wall_only


if __name__ == "__main__":
    full, walls = test_hybrid_bc()
    print("relative error of the directional derivative (all params) at eps 1e-3, 1e-4, 1e-5:",
          " ".join(f"{e:.1e}" for e in full))
    print(f"walls only, eps 1e-4: {walls:.1e}")


# ---------------------------------------------------------------------------------------------
# E1b jobs (called from run_novelty.py)
# ---------------------------------------------------------------------------------------------
E1B_ARMS = ["pinn_bc", "adjoint_truebc", "adjoint_joint1", "adjoint_joint5", "hybrid_bc", "hybrid_bc_joint5"]


def replay_own_walls(ds, D, G):
    s = WallSolver(ds)
    s.set_G(G)
    W = s.forward(np.asarray(D, float).ravel())[0]
    return float(np.sqrt(np.mean((s.predict(W) - s.d) ** 2)) / s.sigma_w)


def run_job(arm, collection, key, epochs):
    import common as C
    ds = C.dataset(collection, key)
    e = C.entry(collection, key)
    root = C.EXT / C.COLLECTIONS[collection]
    D_true = ds.ground_truth()[0]

    def finish(D, G, s, extra):
        m = C.field_scores(D, D_true)
        Gt = true_G(s, root, e)
        m.update(wall_rel_err=wall_error(s, G, Gt), zero_wall_rel_err=wall_error(s, np.zeros_like(Gt), Gt),
                 replay_own_walls_over_sigma=replay_own_walls(ds, D, G))
        m.update(extra)
        return m

    if arm == "pinn_bc":
        s = WallSolver(ds)
        D, wall, G, g_s = pinn_bc_with_walls(root, key, epochs, s)
        m = finish(D, G, s, dict(stage_wall_s=wall, wall_extract_s=g_s, total_wall_s=wall + g_s, epochs=epochs))
        return m, dict(D_hat=D, G_bd=G[:, s.disc.Bd].astype(np.float32))

    if arm == "adjoint_truebc":
        s = WallSolver(ds)
        s.set_G(true_G(s, root, e))
        r = invert_joint(s, np.full(s.K ** 2, s.best_constant()), None, optimise_walls=False)
        m = finish(r["D"], r["G"], s, dict(stage_wall_s=r["wall_s"], total_wall_s=r["wall_s"], alpha=r["alpha"],
                                           flag=r["flag"], total_iters=r["total_iters"], note="bound: uses the hidden true walls"))
        return m, dict(D_hat=r["D"].reshape(ds.N, ds.N))

    n_s = 5 if arm.endswith("5") else 1
    s = WallSolver(ds, n_s=n_s)
    tic = time.perf_counter()
    if arm.startswith("adjoint_joint"):
        b0, pinn_s = np.zeros(s.n_b), 0.0
    elif arm in ("hybrid_bc", "hybrid_bc_joint5"):
        mp, ap = C.load_run("E1b", "pinn_bc", key)
        G = np.zeros((s.disc.S, ds.N * ds.N))
        G[:, s.disc.Bd] = ap["G_bd"]
        pinn_s = mp["total_wall_s"]
        if arm == "hybrid_bc":
            s.set_G(G)
        else:
            b0 = s.b_of_G(G)
    else:
        raise ValueError(arm)
    theta0 = np.full(s.K ** 2, s.best_constant())
    if arm == "hybrid_bc":
        r = invert_joint(s, theta0, None, optimise_walls=False)
    else:
        # best constant D under the starting walls, then the joint path
        s.set_G(s.G_of_b(b0))
        theta0 = np.full(s.K ** 2, s.best_constant())
        r = invert_joint(s, theta0, b0)
    stage = time.perf_counter() - tic
    m = finish(r["D"], r["G"], s, dict(stage_wall_s=stage, pinn_wall_s=pinn_s, total_wall_s=stage + pinn_s,
                                       alpha=r["alpha"], flag=r["flag"], total_iters=r["total_iters"],
                                       n_wall_params=int(s.n_b) if arm != "hybrid_bc" else 0, n_s=n_s))
    return m, dict(D_hat=r["D"].reshape(ds.N, ds.N), G_bd=r["G"][:, s.disc.Bd].astype(np.float32))
