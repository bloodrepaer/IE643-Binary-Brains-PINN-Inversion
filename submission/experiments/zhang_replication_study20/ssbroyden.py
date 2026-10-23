"""
ssbroyden.py
------------
Self-scaled Broyden (SSBroyden) quasi-Newton optimiser with a strong-Wolfe
line search, in float64 NumPy. ONE implementation is shared by the adjoint
and the PINN, as in Zhang, Alla & Karniadakis (2026), Sec. 2.2-2.3.

Inverse-Hessian update (Urban, Stefanou & Pons 2025; Bioli & Mendibe 2026, Eq. 1):

  H+ = (1/tau) ( H - H y y^T H / (y^T H y) + phi (y^T H y) v v^T ) + s s^T / (y^T s)
  v  = s / (y^T s) - H y / (y^T H y)
  b  = s^T B s / y^T s  (= -alpha s^T g / y^T s, since B s = -alpha g)
  h  = y^T H y / y^T s,  a = b h - 1,  c = sqrt(a / (1 + a))
  rho- = min(1, h (1 - c)),  theta- = (rho- - 1) / a,  theta+ = 1 / rho-
  theta = max(theta-, min(theta+, (1 - b) / b))
  sigma = 1 + theta a,  rho+ = min(1, 1 / b),  sigma^(1-N) = |sigma|^(1/(1-N))
  tau = min(rho+ sigma^(1-N), sigma)          if theta <= 0
        rho+ min(sigma^(1-N), 1 / theta)      otherwise
  phi = (1 - theta) / (1 + a theta)

theta = 0, tau = 1 recovers BFGS (variant="bfgs", used in the ablation).
"""
import time
import numpy as np
from scipy.linalg import blas
from scipy.optimize._linesearch import line_search_wolfe2


class _Cache:
    """Evaluate f and g together once per point (the line search asks separately)."""

    def __init__(self, fg):
        self.fg, self.x, self.n = fg, None, 0

    def __call__(self, x):
        if self.x is None or not np.array_equal(x, self.x):
            self.x = x.copy()
            self.f, self.g = self.fg(x)
            self.f = float(self.f)
            self.g = np.asarray(self.g, dtype=np.float64)
            self.n += 1
        return self.f, self.g

    def f_only(self, x):
        return self(x)[0]

    def g_only(self, x):
        return self(x)[1]


def _rank1(H, a, x):
    """H += a x x^T in place (BLAS dger on the Fortran view; H is symmetric)."""
    blas.dger(a, x, x, a=H.T, overwrite_a=1)


def ssbroyden(fg, x0, maxiter=500, H0=None, gtol=1e-12, ftol_rel=1e-14, variant="ssbroyden",
              callback=None, max_time=None, stop=None):
    """Minimise f with gradient g, fg(x) -> (f, g).

    Returns dict(x, f, g, H, nit, nfev, history, reason). history rows are
    (iteration, wall_s, f, nfev). `stop(x, f)` -> True ends the run early
    (used by the warm-start study to stop once a target accuracy is passed).
    """
    t0 = time.perf_counter()
    c = _Cache(fg)
    x = np.array(x0, dtype=np.float64)
    n = x.size
    f, g = c(x)
    H = np.eye(n) if H0 is None else np.array(H0, dtype=np.float64, order="C")
    first = H0 is None
    hist = [(0, 0.0, f, c.n)]
    f_old_old = None
    reason = "maxiter"
    k = 0
    for k in range(1, maxiter + 1):
        p = -(H @ g)
        if g @ p >= 0:                                # not a descent direction: reset
            H = np.eye(n); p = -g; first = True
        if first:                                      # unscaled first step: keep it short
            p = p * min(1.0, 1.0 / max(np.linalg.norm(g), 1e-300))
        ls = line_search_wolfe2(c.f_only, c.g_only, x, p, g, f, f_old_old,
                                c1=1e-4, c2=0.9, amax=1e10, maxiter=30)
        alpha = ls[0]
        if alpha is None:                              # line search failed: one reset, then stop
            if not first:
                H = np.eye(n); first = True; k -= 1
                continue
            reason = "line_search"
            break
        s = alpha * p
        x_new = x + s
        f_new, g_new = c(x_new)
        y = g_new - g
        ys = float(y @ s)
        if ys > 1e-300:
            Hy = H @ y
            yHy = float(y @ Hy)
            if variant == "bfgs":
                theta, tau = 0.0, 1.0
                a_ = 0.0
            else:
                b = -alpha * float(s @ g) / ys           # s^T B s / y^T s
                hh = yHy / ys
                a_ = b * hh - 1.0
                if abs(a_) < 1e-12:
                    theta = 0.0
                else:
                    cc = np.sqrt(max(a_ / (1.0 + a_), 0.0))
                    rho_m = min(1.0, hh * (1.0 - cc))
                    th_m = (rho_m - 1.0) / a_
                    th_p = 1.0 / rho_m if rho_m > 0 else np.inf
                    theta = max(th_m, min(th_p, (1.0 - b) / b))
                sigma = 1.0 + theta * a_
                sig_pow = abs(sigma) ** (1.0 / (1.0 - n)) if n > 1 else 1.0
                rho_p = min(1.0, 1.0 / b)
                if theta <= 0:
                    tau = min(rho_p * sig_pow, sigma)
                else:
                    tau = rho_p * min(sig_pow, 1.0 / theta)
                if not np.isfinite(tau) or tau <= 0:
                    theta, tau = 0.0, 1.0
            phi = (1.0 - theta) / (1.0 + a_ * theta)
            v = s / ys - Hy / yHy
            # H <- (H - Hy Hy^T / yHy + phi yHy v v^T) / tau + s s^T / ys
            _rank1(H, -1.0 / yHy, Hy)
            if phi != 0.0:
                _rank1(H, phi * yHy, v)
            if tau != 1.0:
                H *= 1.0 / tau
            _rank1(H, 1.0 / ys, s)
            first = False
        f_old_old, f = f, f_new
        x, g = x_new, g_new
        hist.append((k, time.perf_counter() - t0, f, c.n))
        if callback is not None:
            callback(x, f, k)
        if stop is not None and stop(x, f):
            reason = "target"
            break
        if np.max(np.abs(g)) < gtol:
            reason = "gtol"
            break
        if abs(f_old_old - f) <= ftol_rel * max(abs(f), 1e-300):
            reason = "ftol"
            break
        if max_time is not None and time.perf_counter() - t0 > max_time:
            reason = "max_time"
            break
    H = 0.5 * (H + H.T)
    return dict(x=x, f=f, g=g, H=H, nit=k, nfev=c.n, history=hist, reason=reason,
                wall_s=time.perf_counter() - t0)


def is_spd(H):
    try:
        np.linalg.cholesky(H)
        return True
    except np.linalg.LinAlgError:
        return False


if __name__ == "__main__":
    # sanity: Rosenbrock in 50-D, both variants
    def rosen(x):
        f = np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
        g = np.zeros_like(x)
        g[:-1] = -400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1])
        g[1:] += 200 * (x[1:] - x[:-1] ** 2)
        return f, g
    for v in ("ssbroyden", "bfgs"):
        r = ssbroyden(rosen, np.full(50, -1.2), maxiter=5000, variant=v)
        print(v, r["nit"], r["nfev"], f"{r['f']:.3e}", r["reason"], f"{np.abs(r['x'] - 1).max():.2e}")
