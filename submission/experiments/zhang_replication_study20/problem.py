"""
problem.py
----------
The ONE inverse problem that every method in the replication solves, built on
the project's own solver (adjoint_inversion.py). Mirrors Sec. 2 of Zhang, Alla
& Karniadakis (2026):

  J(m) = 1/(2 sigma^2) || Q u(m) - d ||^2 + gamma R(m),   D = exp(m),
  R(m) = 1/2 sum_links (m_p - m_q)^2      (discrete H1 seminorm on the 65^2 grid)

Representations of the unknown m (paper Sec. 2.1):
  grid : m = P theta, theta on a K x K control grid, bilinear prolongation P
  nn   : m = m0 + MLP_w(x, y), evaluated at the 65^2 nodes (2 x 32 tanh)

Metrics (paper Sec. 2.4), all on the 65^2 solver grid:
  eps_D      = ||D_hat - D*|| / ||D*||        (the project's relL2(D))
  rmse_logD  = RMS(log D_hat - log D*)
  eps_u      = relative L2 error of the trajectory obtained by pushing D_hat
               through the SAME discrete CN solver, against the ground-truth u
               (129^2 RK4, restricted to 65^2) at the 40 stored snapshots t > 0
  eps_u_solver = the same with D* itself: the discretisation floor
  rms_replay/sigma = the project's displayed metric
"""
import json, os, sys, zlib
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from adjoint_inversion import Dataset, AdjointSolver   # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "datasets_v2_rk4_zero")
_MANIFEST = None


def manifest():
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = {e["key"]: e for e in json.load(open(os.path.join(DATA, "manifest.json")))}
    return _MANIFEST


def load(key, seed=0):
    """Dataset for `key`. seed 0 = the project's own noise draw; seed > 0 redraws
    the noise from u_clean with the same sigma (the paper's seeds vary the noise
    draw as well as the initialisation)."""
    ds = Dataset(DATA, manifest()[key])
    if seed > 0 and ds.sigma > 0:
        s = np.load(os.path.join(DATA, ds.entry["files"]["sensors_npz"]))
        rng = np.random.default_rng(zlib.crc32(f"replication|{key}|{seed}".encode()))
        ds.u = s["u_clean"] + ds.sigma * rng.standard_normal(len(ds.u))
    ds.seed = seed
    return ds


def truth(ds):
    g = np.load(os.path.join(DATA, ds.entry["files"]["ground_truth"]))
    return dict(D=g["D_inv"], t=g["t"], u=g["u"][:, ::2, ::2].astype(np.float64), T=float(g["T"]))


# -----------------------------------------------------------------------------
# Reduced (hard-constraint) objective in m on the 65^2 grid
# -----------------------------------------------------------------------------
class Reduced:
    """J(m) and dJ/dm via the project's discrete CN adjoint. Wraps AdjointSolver
    with P = identity so 'theta' is the full-grid m."""

    def __init__(self, ds, n_steps=None):
        self.ds = ds
        self.solver = AdjointSolver(ds, n_steps=n_steps, control=2, reg="tikhonov")
        N = ds.N
        self.N = N
        self.solver.P = sp.identity(N * N, format="csr")
        self.sigma_w = self.solver.sigma_w
        self.nd = len(ds.u)
        self.dz = self.solver.disc

    def J(self, m, gamma, grad=True):
        return self.solver.objective(np.asarray(m, float), gamma, want_grad=grad)

    def last(self):
        return self.solver._last

    def forward(self, D):
        W, *_ = self.solver.forward(np.asarray(D, float).ravel())
        return W

    def best_constant(self):
        """Data-only initial guess shared by every method (the project's stage 0)."""
        from scipy.optimize import minimize_scalar
        f = lambda lm: self.J(np.full(self.N * self.N, lm), 0.0, grad=False)
        return float(minimize_scalar(f, bounds=(np.log(1e-3), np.log(5.0)), method="bounded",
                                     options={"xatol": 1e-4}).x)


def bilinear_P(N, K):
    from adjoint_inversion import bilinear_1d
    P1 = bilinear_1d(N, K)
    return sp.kron(P1, P1).tocsr()


def links(N):
    """All 5-point links (p, q) of the N x N grid: the regulariser's stencil."""
    idx = np.arange(N * N).reshape(N, N)
    p = np.r_[idx[:-1, :].ravel(), idx[:, :-1].ravel()]
    q = np.r_[idx[1:, :].ravel(), idx[:, 1:].ravel()]
    return p, q


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def metrics(ds, red, D_hat, tr=None):
    tr = tr or truth(ds)
    N = ds.N
    D_hat = np.asarray(D_hat, float).reshape(N, N)
    Dt = tr["D"]
    W = red.forward(D_hat).reshape(red.dz.S, N, N)
    pred = red.solver.predict(W.reshape(red.dz.S, -1))
    out = dict(eps_D=float(np.linalg.norm(D_hat - Dt) / np.linalg.norm(Dt)),
               rmse_logD=float(np.sqrt(np.mean((np.log(D_hat) - np.log(Dt)) ** 2))),
               rms_replay_over_sigma=float(np.sqrt(np.mean((pred - ds.u) ** 2)) / red.sigma_w))
    out["eps_u"] = _traj_err(red, W, tr)
    return out


def _traj_err(red, W, tr):
    dz = red.dz
    ks = np.round(tr["t"][1:] / dz.dt).astype(int)          # GT snapshots t_k = kT/40
    assert np.allclose(ks * dz.dt, tr["t"][1:], rtol=0, atol=1e-9 * tr["T"]), "GT times off the CN grid"
    U = np.stack([W[dz.state_of_step(k)] for k in ks])
    G = tr["u"][1:]
    return float(np.linalg.norm(U - G) / np.linalg.norm(G))


def solver_floor(ds, red, tr=None):
    tr = tr or truth(ds)
    W = red.forward(tr["D"]).reshape(red.dz.S, ds.N, ds.N)
    return _traj_err(red, W, tr)
