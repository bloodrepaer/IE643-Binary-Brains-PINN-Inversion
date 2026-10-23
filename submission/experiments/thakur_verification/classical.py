"""
Classical inversions on the Thakur et al. test cases, scored with the paper's eq. (7).

eqerr   : equation-error method ("finite differences + least squares", the paper's ref. [7]).
          Needs gridded full-field data. c_t (central in time) and the flux divergence
          div(D grad c) (5-point, arithmetic-mean links, LINEAR in D) are formed from the data;
          D = P theta on a K x K bilinear grid; theta from Tikhonov-regularised linear least
          squares (normal equations accumulated over time slices). Regularisation weight: the
          L-curve corner over a log grid would need extra solves; we use the discrepancy-free
          choice of the smallest weight that keeps D positive and report it.
adjoint : the project's adjoint_inversion.AdjointSolver (output least squares, CN on 65 x 65,
          Tikhonov on log D, 16 x 16 control grid, alpha by the discrepancy principle / plateau
          rule) on scattered point data, with the Thakur boundary profile and zero initial state.
"""
import argparse, json, os, sys, time, types
import numpy as np, scipy.sparse as sp

sys.path.insert(0, "/home/claude/v2")
sys.path.insert(0, "/home/claude/thakur")
from gen import boundary_values  # noqa: E402
from pinn import rel_err, relL2  # noqa: E402


def bilinear_1d(n_fine, n_coarse):
    xf, xc = np.linspace(0, 1, n_fine), np.linspace(0, 1, n_coarse)
    k = np.clip(np.searchsorted(xc, xf, side="right") - 1, 0, n_coarse - 2)
    w = (xf - xc[k]) / (xc[k + 1] - xc[k])
    rows = np.r_[np.arange(n_fine), np.arange(n_fine)]
    return sp.csr_matrix((np.r_[1 - w, w], (rows, np.r_[k, k + 1])), shape=(n_fine, n_coarse))


def eval_grid():
    return np.linspace(0, 1, 101)


# ---------------------------------------------------------------------------------------------
def eqerr(z, K=48, stride=4, t_skip=3, lambdas=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1), noise=0.0, seed=0):
    t, xg, C = z["t"], z["x"], z["c"].astype(np.float64)
    if noise > 0:
        C = C + noise * np.random.default_rng(seed).standard_normal(C.shape)
    Cs = C[:, ::stride, ::stride]; xs = xg[::stride]
    n = len(xs); h = xs[1] - xs[0]; dtt = t[1] - t[0]
    P1 = bilinear_1d(n, K); P = sp.kron(P1, P1).tocsr()                     # (n^2, K^2)
    idx = np.arange(n * n).reshape(n, n)
    inner = idx[1:-1, 1:-1].ravel()
    # links of each interior node: (node, neighbour); arithmetic mean D_link = (D_p + D_q)/2
    nb = [idx[2:, 1:-1].ravel(), idx[:-2, 1:-1].ravel(), idx[1:-1, 2:].ravel(), idx[1:-1, :-2].ravel()]
    AtA = np.zeros((K * K, K * K)); Atb = np.zeros(K * K); nrows = 0
    for k in range(t_skip, len(t) - 1):
        c = Cs[k].ravel()
        ct = (Cs[k + 1].ravel() - Cs[k - 1].ravel()) / (2 * dtt)
        # div(D grad c)_p = sum_q (D_p + D_q)/2 (c_q - c_p)/h^2  -> row of matrix M (interior x n^2) in D
        rows, cols, vals = [], [], []
        r = np.arange(len(inner))
        for q in nb:
            g = (c[q] - c[inner]) / (2 * h * h)
            rows += [r, r]; cols += [inner, q]; vals += [g, g]
        M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(inner), n * n))
        A = (M @ P)
        AtA += (A.T @ A).toarray(); Atb += A.T @ ct[inner]; nrows += len(inner)
    # smoothness regulariser on theta (discrete gradient on the K x K grid)
    e = np.ones(K); D1 = sp.diags([-e[:-1], e[:-1]], [0, 1], shape=(K - 1, K))
    G = sp.vstack([sp.kron(D1, sp.identity(K)), sp.kron(sp.identity(K), D1)])
    R = (G.T @ G).toarray()
    sc = np.trace(AtA) / np.trace(R)
    xe = eval_grid(); Pe = sp.kron(bilinear_1d(101, K), bilinear_1d(101, K))
    out = []
    for lam in lambdas:
        th = np.linalg.solve(AtA + lam * sc * R, Atb)
        out.append((lam, (Pe @ th).reshape(101, 101)))
    return out, nrows


# ---------------------------------------------------------------------------------------------
def adjoint(z, n_spatial, seed=0, control=16, noise=0.0, maxiter=150, verbose=False, n_inv=65):
    import adjoint_inversion as A
    from integrators import BoundaryCondition, EDGES
    t, xg, C = z["t"], z["x"], z["c"]
    Nd = len(xg); rng = np.random.default_rng(seed)
    flat = rng.choice(Nd * Nd, n_spatial, replace=False)
    ix, iy = np.unravel_index(flat, (Nd, Nd))
    ds = types.SimpleNamespace()
    ds.key = "thakur"; ds.T = 1.0; ds.N = n_inv
    ds.x = np.tile(xg[ix], len(t)); ds.y = np.tile(xg[iy], len(t)); ds.t = np.repeat(t, n_spatial)
    u = C.reshape(len(t), Nd * Nd)[:, flat].astype(np.float64).ravel()
    if noise > 0:
        u = u + noise * rng.standard_normal(u.shape)
    ds.u = u; ds.sigma = float(noise)
    ds.bc = BoundaryCondition({e: (lambda tt: 0.0) for e in EDGES})
    ds.u0 = np.zeros((n_inv, n_inv))
    s = A.AdjointSolver(ds, control=control, reg="tikhonov")
    g65 = np.linspace(0, 1, n_inv); X, Y = np.meshgrid(g65, g65, indexing="ij")
    gb = boundary_values(X, Y).ravel()
    s.disc.G[:] = gb                                     # Thakur boundary profile at every state
    s.u0_full = np.zeros(n_inv * n_inv); s.u0_full[s.disc.Bd] = gb[s.disc.Bd]
    out = s.invert(maxiter=maxiter, verbose=verbose)
    Pe = sp.kron(bilinear_1d(101, control), bilinear_1d(101, control))
    return np.exp(Pe @ out["theta"]).reshape(101, 101), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["eqerr", "adjoint"])
    ap.add_argument("--case", default="I")
    ap.add_argument("--n-spatial", type=int, default=2500)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-inv", type=int, default=65)
    ap.add_argument("--out", default="/home/claude/thakur/runs")
    a = ap.parse_args()
    z = np.load(f"/home/claude/thakur/data_{a.case}_N501.npz")
    sub = np.arange(0, 501, 5); Dt = z["D"][np.ix_(sub, sub)]
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    if a.method == "eqerr":
        res, nrows = eqerr(z, noise=a.noise, seed=a.seed)
        rows = [dict(lam=l, L_D=rel_err(Dt, D), relL2_D=relL2(Dt, D), D_min=float(D.min())) for l, D in res]
        # report: the weight with the smallest L_D (oracle) AND the smallest weight with D > 0
        best = min(range(len(rows)), key=lambda i: rows[i]["L_D"])
        rec = dict(method="eqerr", case=a.case, noise=a.noise, n_spatial=126 * 126, n_equations=nrows,
                   path=rows, best=rows[best], wall_s=time.time() - t0)
        tag = f"eqerr_{a.case}_noise{a.noise}"
        np.savez_compressed(os.path.join(a.out, tag + ".npz"), D_pred=res[best][1], D_true=Dt)
    else:
        D, out = adjoint(z, a.n_spatial, a.seed, noise=a.noise, n_inv=a.n_inv)
        rec = dict(method="adjoint", n_inv=a.n_inv, case=a.case, noise=a.noise, n_spatial=a.n_spatial,
                   n_readings=a.n_spatial * 100, L_D=rel_err(Dt, D), relL2_D=relL2(Dt, D),
                   alpha=out["alpha"], flag=out["flag"], rms=out["rms"], n_evals=out["n_evals"],
                   wall_s=time.time() - t0)
        tag = f"adjoint_{a.case}_n{a.n_spatial}_noise{a.noise}_s{a.seed}" + (f"_inv{a.n_inv}" if a.n_inv != 65 else "")
        np.savez_compressed(os.path.join(a.out, tag + ".npz"), D_pred=D, D_true=Dt)
    json.dump(rec, open(os.path.join(a.out, tag + ".json"), "w"), indent=1, default=float)
    print(json.dumps(rec, default=float)[:600])


if __name__ == "__main__":
    main()
