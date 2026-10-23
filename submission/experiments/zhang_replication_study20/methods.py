"""
methods.py
----------
The three estimators of the replication, all solving problem.J with the same
data, regulariser R, weight gamma, initial guess m0, optimiser and precision.

  adjoint(rep)  hard constraint: reduced objective J(m(theta)) with the project's
                discrete CN adjoint gradient; rep = "grid" (16x16 bilinear) or
                "nn" (MLP, gradient chained through jax.vjp).
  pinn(rep)     soft constraint: state network u(x,y,t) and the same m(theta);
                loss = l_pde + l_bc + l_ic + J_data,reg / N_d   (unit physics weights;
                the data + regulariser block is the adjoint objective per reading,
                so gamma means the same thing for both methods).
  eki           ensemble Kalman inversion in the 16x16 control space (derivative-free).

Every method logs (iteration, wall_s, objective, eps_D) so convergence curves
and time-to-accuracy can be read off.
"""
import time
import numpy as np
from nets import jax, jnp, NNField, GridField, field_on_grid, init_mlp, mlp, ravel_pytree
from problem import Reduced, bilinear_P, links, truth
from ssbroyden import ssbroyden, is_spd


def _eps(D, Dt):
    return float(np.linalg.norm(D - Dt) / np.linalg.norm(Dt))


# =============================================================================
# Adjoint (hard constraint)
# =============================================================================
def adjoint(ds, red, rep, gamma, m0, seed=0, K=16, maxiter=1000, theta_init=None, H0=None,
            stop_eps=None, max_time=None, variant="ssbroyden", tr=None):
    tr = tr or truth(ds)
    Dt = tr["D"].ravel()
    N = ds.N
    t0 = time.perf_counter()
    if rep == "grid":
        P = bilinear_P(N, K)
        theta0 = np.full(K * K, m0) if theta_init is None else np.asarray(theta_init, float)
        m_of = lambda th: P @ th

        def fg(th):
            J, g = red.J(P @ th, gamma)
            return J, P.T @ g
    else:
        field = NNField(m0, seed)
        theta0 = field.theta0 if theta_init is None else np.asarray(theta_init, float)
        mg = jax.jit(lambda th: field_on_grid(field, th, N))
        vjp_fn = jax.jit(lambda th, ct: jax.vjp(mg, th)[1](ct)[0])
        m_of = lambda th: np.asarray(mg(jnp.asarray(th)))

        def fg(th):
            J, g = red.J(m_of(th), gamma)
            return J, np.asarray(vjp_fn(jnp.asarray(th), jnp.asarray(g)))
    log = []

    def cb(th, f, k):
        log.append((k, time.perf_counter() - t0, f, _eps(np.exp(m_of(th)), Dt)))

    cb(theta0, fg(theta0)[0], 0)
    stop = (lambda th, f: log[-1][3] <= stop_eps) if stop_eps else None
    r = ssbroyden(fg, theta0, maxiter=maxiter, H0=H0, callback=cb, stop=stop, max_time=max_time, variant=variant)
    D = np.exp(m_of(r["x"])).reshape(N, N)
    return dict(D=D, theta=r["x"], H=r["H"], nit=r["nit"], nfev=r["nfev"], reason=r["reason"],
                wall_s=time.perf_counter() - t0, log=log, n_params=int(theta0.size), J=float(r["f"]))


# =============================================================================
# PINN (soft constraint)
# =============================================================================
class PINN:
    def __init__(self, ds, red, rep, gamma, m0, seed=0, K=16, width=32, depth_u=3,
                 n_pde=4000, n_bc=400, n_data_sub=4000):
        self.ds, self.red, self.gamma, self.seed = ds, red, float(gamma), seed
        N = ds.N
        self.N = N
        self.field = GridField(m0, K) if rep == "grid" else NNField(m0, seed)
        pu = init_mlp(jax.random.PRNGKey(10_000 + seed), [3] + [width] * depth_u + [1])
        self.th_u0, self.unravel_u = ravel_pytree(pu)
        self.nu = self.th_u0.size
        self.theta0 = np.r_[np.asarray(self.th_u0), self.field.theta0]
        self.T = ds.T
        self.sig2 = red.sigma_w ** 2
        self.Nd = len(ds.u)
        self.Xd_all = np.c_[ds.x, ds.y, ds.t / ds.T]
        self.d_all = np.asarray(ds.u, float)
        self.n_sub = min(n_data_sub, self.Nd) if n_data_sub else self.Nd
        xs = np.linspace(0, 1, N)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        self.Xic = jnp.asarray(np.c_[X.ravel(), Y.ravel(), np.zeros(N * N)])
        self.u0 = jnp.asarray(ds.u0.ravel())
        p, q = links(N)
        self.lp, self.lq = jnp.asarray(p), jnp.asarray(q)
        self.n_pde, self.n_bc = n_pde, n_bc
        self.rng = np.random.default_rng(20_000 + seed)
        self._build()

    # --- networks -------------------------------------------------------------
    def u_pt(self, th_u, z):
        return mlp(self.unravel_u(th_u), 2.0 * z - 1.0)[0]

    def _build(self):
        nu, field, T = self.nu, self.field, self.T

        def residual(th_u, th_m, z):
            uf = lambda zz: self.u_pt(th_u, zz)
            gu = jax.grad(uf)
            g = gu(z)
            uxx = jax.jvp(gu, (z,), (jnp.array([1.0, 0.0, 0.0]),))[1][0]
            uyy = jax.jvp(gu, (z,), (jnp.array([0.0, 1.0, 0.0]),))[1][1]
            m, (mx, my) = jax.value_and_grad(lambda x, y: field.m_pt(th_m, x, y), argnums=(0, 1))(z[0], z[1])
            D = jnp.exp(m)
            return g[2] - T * D * (uxx + uyy + mx * g[0] + my * g[1])

        def parts(theta, Xp, Xb, Xd, dd, wd):
            th_u, th_m = theta[:nu], theta[nu:]
            res = jax.vmap(lambda z: residual(th_u, th_m, z))(Xp)
            u = jax.vmap(lambda z: self.u_pt(th_u, z))
            l_pde = jnp.mean(res ** 2)
            l_bc = jnp.mean(u(Xb) ** 2)                                   # zero Dirichlet data
            l_ic = jnp.mean((u(self.Xic) - self.u0) ** 2)
            data = wd * 0.5 * jnp.sum((u(Xd) - dd) ** 2) / self.sig2
            mg = field_on_grid(field, th_m, self.N)
            R = 0.5 * jnp.sum((mg[self.lp] - mg[self.lq]) ** 2)
            J_dr = (data + self.gamma * R) / self.Nd
            return l_pde + l_bc + l_ic + J_dr, (l_pde, l_bc, l_ic, J_dr)

        self.parts = jax.jit(parts)
        self.vg = jax.jit(jax.value_and_grad(lambda *a: parts(*a)[0]))
        self.m_grid = jax.jit(lambda th_m: field_on_grid(field, th_m, self.N))

    def batch(self):
        r = self.rng
        Xp = r.uniform(0, 1, (self.n_pde, 3))
        e = r.integers(0, 4, self.n_bc); s = r.uniform(0, 1, self.n_bc)
        xb = np.where(e == 0, 0.0, np.where(e == 1, 1.0, s))
        yb = np.where(e == 2, 0.0, np.where(e == 3, 1.0, s))
        Xb = np.c_[xb, yb, r.uniform(0, 1, self.n_bc)]
        if self.n_sub < self.Nd:
            idx = r.choice(self.Nd, self.n_sub, replace=False)
            wd = self.Nd / self.n_sub
        else:
            idx, wd = np.arange(self.Nd), 1.0
        return tuple(map(jnp.asarray, (Xp, Xb, self.Xd_all[idx], self.d_all[idx]))) + (float(wd),)

    def D_of(self, theta):
        return np.exp(np.asarray(self.m_grid(jnp.asarray(theta[self.nu:]))))

    # --- training ---------------------------------------------------------------
    def train(self, n_adam=1000, lr=1e-3, n_outer=10, n_inner=300, theta_init=None, tr=None,
              variant="ssbroyden", max_time=None):
        tr = tr or truth(self.ds)
        Dt = tr["D"].ravel()
        t0 = time.perf_counter()
        theta = jnp.asarray(self.theta0 if theta_init is None else theta_init)
        log = []
        it = 0
        # Adam warm-up with step-decaying learning rate, fresh points every step
        mom, vel = jnp.zeros_like(theta), jnp.zeros_like(theta)
        b1, b2 = 0.9, 0.999

        @jax.jit
        def adam_step(theta, mom, vel, k, lr, Xp, Xb, Xd, dd, wd):
            f, g = jax.value_and_grad(lambda th: self.parts(th, Xp, Xb, Xd, dd, wd)[0])(theta)
            mom = b1 * mom + (1 - b1) * g
            vel = b2 * vel + (1 - b2) * g ** 2
            mh, vh = mom / (1 - b1 ** k), vel / (1 - b2 ** k)
            return theta - lr * mh / (jnp.sqrt(vh) + 1e-8), mom, vel, f

        for k in range(1, n_adam + 1):
            step_lr = lr * 0.5 ** ((k - 1) // max(1, n_adam // 4))
            theta, mom, vel, f = adam_step(theta, mom, vel, k, step_lr, *self.batch())
            it += 1
            if k % 50 == 0 or k == n_adam:
                log.append((it, time.perf_counter() - t0, float(f), _eps(self.D_of(np.asarray(theta)), Dt), "adam"))
        # two-loop SSBroyden: resample outside, deterministic objective inside
        theta = np.asarray(theta)
        H = None
        outer_info = []
        for o in range(n_outer):
            b = self.batch()
            fg = lambda th, b=b: tuple(np.asarray(v) for v in self.vg(jnp.asarray(th), *b))
            base = it

            def cb(th, f, k, base=base):
                log.append((base + k, time.perf_counter() - t0, f, _eps(self.D_of(th), Dt), "ssb"))

            r = ssbroyden(fg, theta, maxiter=n_inner, H0=H, callback=cb, variant=variant,
                          max_time=None if max_time is None else max(1.0, max_time - (time.perf_counter() - t0)))
            theta = r["x"]
            it += r["nit"]
            H = r["H"] if is_spd(r["H"]) else None
            outer_info.append(dict(outer=o, nit=r["nit"], reason=r["reason"], f=r["f"], spd=H is not None))
            if max_time is not None and time.perf_counter() - t0 > max_time:
                break
        wall = time.perf_counter() - t0
        parts = self.parts(jnp.asarray(theta), *self.batch())[1]
        return dict(D=self.D_of(theta).reshape(self.N, self.N), theta=theta, nit=it, wall_s=wall, log=log,
                    outer=outer_info, n_params=int(theta.size),
                    final_parts=dict(zip(("l_pde", "l_bc", "l_ic", "J_dr"), map(float, parts))))

    def state_error(self, theta, tr):
        """Error of the PINN's own state network vs the ground-truth u (secondary)."""
        N = self.N
        xs = np.linspace(0, 1, N)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        th_u = jnp.asarray(theta[:self.nu])
        f = jax.jit(jax.vmap(lambda z: self.u_pt(th_u, z)))
        U = np.stack([np.asarray(f(jnp.asarray(np.c_[X.ravel(), Y.ravel(), np.full(N * N, t / self.T)])))
                      for t in tr["t"][1:]])
        G = tr["u"][1:].reshape(len(tr["t"]) - 1, -1)
        return float(np.linalg.norm(U - G) / np.linalg.norm(G))


# =============================================================================
# Ensemble Kalman inversion (derivative-free baseline, paper Test 2)
# =============================================================================
def eki(ds, red, m0, seed=0, K=16, n_ens=200, n_iter=10, prior_len=0.15, prior_std=0.5, tr=None):
    """EKI in the K x K control space with perturbed observations (Iglesias,
    Law & Stuart 2013; paper Eqs. A.5-A.8). Prior: m0 + GP with a squared-
    exponential kernel (length prior_len, std prior_std) on the control nodes."""
    tr = tr or truth(ds)
    Dt = tr["D"].ravel()
    t0 = time.perf_counter()
    rng = np.random.default_rng(30_000 + seed)
    xc = np.linspace(0, 1, K)
    Xc, Yc = np.meshgrid(xc, xc, indexing="ij")
    pts = np.c_[Xc.ravel(), Yc.ravel()]
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    C = prior_std ** 2 * np.exp(-d2 / (2 * prior_len ** 2)) + 1e-8 * np.eye(K * K)
    Lc = np.linalg.cholesky(C)
    P = bilinear_P(ds.N, K)
    xi = m0 + (Lc @ rng.standard_normal((K * K, n_ens))).T          # (J, K^2)
    y = np.asarray(ds.u, float)
    s2 = red.sigma_w ** 2
    Gam = s2 * np.eye(len(y))

    def G(th):
        red.J(P @ th, 0.0, grad=False)
        return red.solver.predict(red.last()["W"])

    log = [(0, 0.0, np.nan, _eps(np.exp(P @ xi.mean(0)), Dt))]
    for n in range(1, n_iter + 1):
        g = np.stack([G(x) for x in xi])                            # (J, Nd)
        xm, gm = xi.mean(0), g.mean(0)
        Cxg = (xi - xm).T @ (g - gm) / (n_ens - 1)
        Cgg = (g - gm).T @ (g - gm) / (n_ens - 1)
        yp = y + np.sqrt(s2) * rng.standard_normal((n_ens, len(y)))
        xi = xi + np.linalg.solve(Cgg + Gam, (yp - g).T).T @ Cxg.T
        mis = 0.5 * np.sum((gm - y) ** 2) / s2
        log.append((n, time.perf_counter() - t0, float(mis), _eps(np.exp(P @ xi.mean(0)), Dt)))
    theta = xi.mean(0)
    return dict(D=np.exp(P @ theta).reshape(ds.N, ds.N), theta=theta, nit=n_iter, nfev=n_iter * n_ens,
                wall_s=time.perf_counter() - t0, log=log, n_params=K * K)
