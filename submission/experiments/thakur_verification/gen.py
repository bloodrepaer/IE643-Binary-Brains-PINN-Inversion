"""Synthetic data for Thakur et al. (2024) Cases I-IV.

The paper gives D(x,y), the domain (unit square), T = 1, a 500x500 mesh and 100 time
steps, but NOT the initial or boundary conditions. From Fig. 2 (c at t = 0.5T: c = 1 at
the middle of the bottom and top edges, weaker at the middle of the side edges) we use
    c(x, y, 0) = 0,
    c = g on the boundary:  bottom/top  g = exp(-(x-0.5)^2 / (2*0.1^2)),
                            left/right  g = 0.5 exp(-(y-0.5)^2 / (2*0.1^2)).
Solver: node-based finite volumes (5-point, harmonic-mean link conductance), Crank-
Nicolson with a Rannacher start (two backward-Euler half steps), dt = 1e-3, output
every 10 steps -> t = 0.01 ... 1.00 (100 snapshots). c is scaled by its maximum.
"""
import sys, time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla

CASES = {
    "I":   lambda x, y: 0.05 + 0.1 * (x * (1 - x) + y * (1 - y)),
    "II":  lambda x, y: 0.25 + 0.1 * (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)),
    "III": lambda x, y: 0.25 + 0.1 * (np.sin(4 * np.pi * x) + np.sin(4 * np.pi * y)),
    "IV":  lambda x, y: 0.3 + 0.15 * np.tanh(-10 + 20 * x) + 0.1 * np.tanh(-7 + 20 * y),
}


def boundary_values(X, Y):
    g = np.zeros_like(X)
    gb = np.exp(-(X - 0.5) ** 2 / (2 * 0.1 ** 2))
    gs = 0.5 * np.exp(-(Y - 0.5) ** 2 / (2 * 0.1 ** 2))
    g[:, 0] = gb[:, 0]; g[:, -1] = gb[:, -1]          # bottom (y=0), top (y=1)
    g[0, :] = gs[0, :]; g[-1, :] = gs[-1, :]          # left (x=0), right (x=1)
    return g


def operator(D, h):
    N = D.shape[0]
    idx = np.arange(N * N).reshape(N, N)
    p = np.r_[idx[:-1, :].ravel(), idx[:, :-1].ravel()]
    q = np.r_[idx[1:, :].ravel(), idx[:, 1:].ravel()]
    Df = D.ravel()
    c = 2 * Df[p] * Df[q] / (Df[p] + Df[q]) / h ** 2
    L = sp.csr_matrix((np.r_[c, c, -c, -c], (np.r_[p, q, p, q], np.r_[q, p, p, q])), shape=(N * N, N * N))
    inner = np.zeros((N, N), bool); inner[1:-1, 1:-1] = True
    I, B = idx[inner], idx[~inner]
    return L[I][:, I].tocsc(), L[I][:, B].tocsc(), I, B


def simulate(case, N=501, dt=1e-3, n_out=100, every=10):
    g1 = np.linspace(0, 1, N); X, Y = np.meshgrid(g1, g1, indexing="ij")
    D = CASES[case](X, Y)
    LII, LIB, I, B = operator(D, 1.0 / (N - 1))
    gfull = boundary_values(X, Y).ravel()
    b = LIB @ gfull[B]
    u = np.zeros(len(I))
    Id = sp.identity(len(I), format="csc")
    be = spla.splu((Id - (dt / 2) * LII).tocsc())            # backward Euler, half step
    cn = be                                                   # same matrix: I - dt/2 L
    out = np.zeros((n_out, N, N)); full = gfull.copy()
    # Rannacher: two BE half-steps, then CN
    u = be.solve(u + (dt / 2) * b); u = be.solve(u + (dt / 2) * b)
    step = 1
    while True:
        if step % every == 0:
            k = step // every - 1
            full[I] = u; out[k] = full.reshape(N, N)
            if k == n_out - 1:
                break
        step += 1
        u = cn.solve(u + (dt / 2) * (LII @ u) + dt * b)
    t = (np.arange(n_out) + 1) * dt * every
    return t, X, Y, D, out


if __name__ == "__main__":
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 501
    for case in sys.argv[1].split(","):
        t0 = time.time()
        t, X, Y, D, C = simulate(case, N)
        cmax = C.max(); C = C / cmax
        np.savez_compressed(f"/home/claude/thakur/data_{case}_N{N}.npz", t=t, x=X[:, 0], D=D,
                            c=C.astype(np.float32), cmax=cmax)
        print(case, N, f"{time.time() - t0:.0f}s", "c range", C.min(), C.max(), "c(t=0.5) center", C[49, N // 2, N // 2])
