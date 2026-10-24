"""
integrators.py
--------------
Method-of-lines forward model for

    du/dt = div( D(x,y) grad u ),   (x,y) in [0,Lx]x[0,Ly],   u = g_edge(t) on each edge

Space and time are deliberately separated:

  1. `build_operator` discretises space ONCE for a given D, producing the
     semi-discrete linear ODE system on the INTERIOR nodes

         du_I/dt = L u_I + b(t),     b(t) = sum_edges g_edge(t) * B_edge

     Boundary nodes are eliminated (they are known Dirichlet values), so L is
     symmetric negative definite. B_edge is the coupling of the interior to a
     UNIT temperature on that edge, so any boundary temperature history --
     constant or time-varying, different on each edge -- costs nothing extra.
     Corner nodes belong to the left/right edges.

  2. `integrate` advances that ODE with one of three time integrators that
     share the same interface, plus an exact reference used only for testing:

        name    type       order   stability            cost per step
        ------  ---------  ------  -------------------  -----------------------
        rk4     explicit   4       dt <= 2.785 / rho(L)  4 mat-vecs        (data generation)
        cn      implicit   2       unconditional (A)    1 sparse solve    (inversion solver;
                                                        + Rannacher start  forward replay)
        fe      explicit   1       dt <= 2   / rho(L)    1 mat-vec         (benchmark only)
        expm    exact      exact   n/a                  Krylov exp(tL)v   (test reference;
                                                                           constant BCs only)

     rho(L) is the spectral radius (~8 D_max / h^2 for the 5-point stencil).
     Explicit methods choose their own step from the stability limit; CN uses
     dt_target. 'expm' is the exact solution of the semi-discrete system (zero
     time error), which makes it the reference for measuring the time error of
     the other three on the SAME grid.

All of them return snapshots on the same uniform output grid t_k = k*T/n_out, so
datasets produced with different integrators are directly comparable.
"""
import time as _time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

INTEGRATORS = ("rk4", "cn", "fe")          # usable for data generation
REFERENCE = "expm"                         # exact in time; verification only
ALL_METHODS = INTEGRATORS + (REFERENCE,)
RK4_STABILITY = 2.785   # extent of the RK4 stability region on the negative real axis
FE_STABILITY = 2.0


# ----------------------------------------------------------------------------
# Space: flux-form operator on interior nodes
# ----------------------------------------------------------------------------
def harmonic_mean(a, b):
    """Series-resistance average for the diffusivity on a link between two nodes."""
    return 2.0 * a * b / (a + b)


EDGES = ("left", "right", "bottom", "top")


def edge_masks(shape):
    """Boolean full-grid masks for each edge. Corners go to left/right."""
    Nx, Ny = shape
    m = {e: np.zeros(shape, bool) for e in EDGES}
    m["bottom"][1:-1, 0] = True
    m["top"][1:-1, -1] = True
    m["left"][0, :] = True
    m["right"][-1, :] = True
    return m


class BoundaryCondition:
    """Dirichlet temperature on each edge: a number (constant in time) or a
    callable g(t). Missing edges default to 0. `describe` holds a
    human-readable expression per edge for the problem-spec file."""

    def __init__(self, values=None, describe=None):
        self.values = {e: 0.0 for e in EDGES}
        self.values.update(values or {})
        self.describe = {e: str(v) for e, v in self.values.items()}
        self.describe.update(describe or {})

    @property
    def is_constant(self):
        return not any(callable(v) for v in self.values.values())

    def at(self, t):
        return {e: float(v(t)) if callable(v) else float(v) for e, v in self.values.items()}

    def field(self, shape, t):
        """Full-grid array: boundary temperatures at time t, zero inside."""
        g, vals, masks = np.zeros(shape), self.at(t), edge_masks(shape)
        for e in EDGES:
            g[masks[e]] = vals[e]
        return g


def build_operator(D, hx, hy, stencil="5pt"):
    """Assemble the interior operator L (CSR, symmetric) and the per-edge
    coupling vectors B_edge (interior response to a unit edge temperature).

    D is the diffusivity at every node of the full (Nx, Ny) grid, 'ij' indexing
    (first axis = x). Each link between neighbouring nodes gets weight
    w * harmonic_mean(D_a, D_b), where w is the stencil weight:

      5pt : 1/hx^2 (x-links), 1/hy^2 (y-links)
      9pt : 2/(3h^2) cardinal, 1/(6h^2) diagonal (needs hx == hy)

    Each row's diagonal is minus the sum of its link weights, so a constant
    field has zero Laplacian. Links that reach a boundary node go into B_edge.
    """
    Nx, Ny = D.shape
    if stencil == "5pt":
        links = [((1, 0), 1 / hx**2), ((-1, 0), 1 / hx**2),
                 ((0, 1), 1 / hy**2), ((0, -1), 1 / hy**2)]
    elif stencil == "9pt":
        if not np.isclose(hx, hy):
            raise ValueError("9pt stencil requires square cells (hx == hy)")
        c, d = 2 / (3 * hx**2), 1 / (6 * hx**2)
        links = [((1, 0), c), ((-1, 0), c), ((0, 1), c), ((0, -1), c),
                 ((1, 1), d), ((1, -1), d), ((-1, 1), d), ((-1, -1), d)]
    else:
        raise ValueError(f"unknown stencil '{stencil}'")

    nI = (Nx - 2) * (Ny - 2)
    idx = -np.ones((Nx, Ny), dtype=np.int64)
    idx[1:-1, 1:-1] = np.arange(nI).reshape(Nx - 2, Ny - 2)
    masks = edge_masks((Nx, Ny))

    me = idx[1:-1, 1:-1].ravel()
    Dc = D[1:-1, 1:-1]
    diag = np.zeros(nI)
    B = {e: np.zeros(nI) for e in EDGES}
    rows, cols, vals = [me], [me], [None]          # diagonal filled in later
    for (di, dj), w in links:
        sl = (slice(1 + di, Nx - 1 + di), slice(1 + dj, Ny - 1 + dj))
        coef = (w * harmonic_mean(Dc, D[sl])).ravel()
        nb = idx[sl].ravel()
        diag -= coef
        inside = nb >= 0
        rows.append(me[inside]); cols.append(nb[inside]); vals.append(coef[inside])
        for e in EDGES:
            on_e = (~inside) & masks[e][sl].ravel()
            B[e][on_e] += coef[on_e]
    vals[0] = diag
    L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nI, nI))
    return L, B


def spectral_radius_bound(L):
    """Gershgorin bound on rho(L): max row sum of |L_ij|. For these operators
    it is within a few percent of the true value and costs one pass."""
    return float(np.max(np.asarray(abs(L).sum(axis=1)).ravel()))


# ----------------------------------------------------------------------------
# Time: six integrators behind one interface
# ----------------------------------------------------------------------------
def integrate(L, b, u0, T, n_out, method="cn", dt_target=None,
              cfl_safety=0.9, rannacher=True):
    """Advance du/dt = L u + b(t) from u(0)=u0 to T.
    b is either a constant vector or a callable b(t) (time-varying BCs).

    Returns (t_out, U, info):
      t_out : (n_out+1,) output times k*T/n_out
      U     : (n_out+1, n) solution at those times
      info  : dict with the step actually used, step count, wall time, etc.

    CN takes m = ceil((T/n_out)/dt_target) steps per output interval
    (default dt_target = T/400). Explicit methods take the smallest m
    that keeps dt <= cfl_safety * stability_limit / rho(L).
    """
    if method not in ALL_METHODS:
        raise ValueError(f"method must be one of {ALL_METHODS}")
    n = L.shape[0]
    bt = b if callable(b) else (lambda t, _b=b: _b)
    Delta = T / n_out
    t_out = np.linspace(0.0, T, n_out + 1)
    U = np.empty((n_out + 1, n))
    U[0] = u0
    tic = _time.perf_counter()
    info = {"method": method}

    if method == "expm":
        if callable(b):
            raise ValueError("expm needs time-constant boundary values; use rk4 as the reference instead")
        # Exact solution: u(t) = u_s + exp(tL)(u0 - u_s), with steady state u_s = -L^{-1} b
        u_s = spla.spsolve(L.tocsc(), -b) if np.any(b) else np.zeros(n)
        U[:] = spla.expm_multiply(L, u0 - u_s, start=0.0, stop=T,
                                  num=n_out + 1, endpoint=True) + u_s
        info.update(dt=0.0, n_steps=0)
        info["wall_s"] = _time.perf_counter() - tic
        return t_out, U, info

    if method in ("fe", "rk4"):
        rho = spectral_radius_bound(L)
        limit = (RK4_STABILITY if method == "rk4" else FE_STABILITY) / rho
        m = max(1, int(np.ceil(Delta / (cfl_safety * limit))))
        info["stability_limit"] = limit
    else:
        dt_target = dt_target or T / 400
        m = max(1, int(np.ceil(Delta / dt_target - 1e-9)))
    dt = Delta / m
    info.update(dt=dt, n_steps=m * n_out)
    I = sp.identity(n, format="csc")
    Lc = L.tocsc()

    u = u0.copy()
    tn = lambda step: step * dt          # time at the start of step `step`
    step = 0
    if method == "fe":
        for k in range(1, n_out + 1):
            for _ in range(m):
                u = u + dt * (L @ u + bt(tn(step)))
                step += 1
            U[k] = u

    elif method == "rk4":
        for k in range(1, n_out + 1):
            for _ in range(m):
                t0 = tn(step)
                k1 = L @ u + bt(t0)
                k2 = L @ (u + 0.5 * dt * k1) + bt(t0 + 0.5 * dt)
                k3 = L @ (u + 0.5 * dt * k2) + bt(t0 + 0.5 * dt)
                k4 = L @ (u + dt * k3) + bt(t0 + dt)
                u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                step += 1
            U[k] = u

    elif method == "cn":
        # Rannacher start: the first CN step is replaced by two backward-Euler
        # half-steps, which damp the stiffest modes that CN would otherwise
        # flip in sign every step (CN is A-stable but not L-stable). Global
        # accuracy stays second order. NOTE: an adjoint of this scheme must
        # mirror the same first step.
        # One factorisation serves both: a BE half-step (dt/2) has the same
        # matrix I - (dt/2) L as the CN implicit side.
        solve = spla.factorized(I - 0.5 * dt * Lc)
        Bexp = I + 0.5 * dt * Lc
        for k in range(1, n_out + 1):
            for _ in range(m):
                t0 = tn(step)
                if rannacher and step == 0:
                    u = solve(u + 0.5 * dt * bt(t0 + 0.5 * dt))
                    u = solve(u + 0.5 * dt * bt(t0 + dt))
                else:
                    u = solve(Bexp @ u + 0.5 * dt * (bt(t0) + bt(t0 + dt)))
                step += 1
            U[k] = u
        info["rannacher"] = rannacher

    info["wall_s"] = _time.perf_counter() - tic
    return t_out, U, info


# ----------------------------------------------------------------------------
# Convenience wrapper on the full grid
# ----------------------------------------------------------------------------
class ForwardModel:
    """Holds the discretised operator for one D field; solves on the full grid.
    Boundary rows of the returned snapshots hold the prescribed g_edge(t)."""

    def __init__(self, domain, D, stencil="5pt", bc=None):
        self.domain, self.D, self.stencil = domain, D, stencil
        self.bc = bc if isinstance(bc, BoundaryCondition) else BoundaryCondition(bc)
        self.L, self.B = build_operator(D, domain.dx, domain.dy, stencil)

    def b(self, t):
        g = self.bc.at(t)
        return sum(g[e] * self.B[e] for e in EDGES)

    def solve(self, u0_full, T, n_out, method="cn", **kw):
        b = self.b(0.0) if self.bc.is_constant else self.b
        t, U_I, info = integrate(self.L, b, u0_full[1:-1, 1:-1].ravel(),
                                 T, n_out, method, **kw)
        Nx, Ny = self.domain.shape
        U = np.stack([self.bc.field((Nx, Ny), tk) for tk in t])
        U[:, 1:-1, 1:-1] = U_I.reshape(n_out + 1, Nx - 2, Ny - 2)
        return t, U, info
