"""
Best case for the paper's loss: the concentration network is replaced by the EXACT simulated
field (finite differences on the 501 x 501 data mesh), so only D~(x,y) (8 x 128, tanh, exp output)
is trained on the backward-Euler consistency residual of eq. (5):

    r = c(t + dt) - dt * f(c(t + dt), D~) - c(t),    f = D_x c_x + D_y c_y + D (c_xx + c_yy)

With a perfect c, r = 0 is not attainable for the true D: backward Euler has a truncation error
-(dt^2/2) c_tt, so the minimiser is biased by O(dt). dt in {0.01, 0.02, 0.05} measures that bias;
"cont" replaces the BE step by the exact time derivative (central difference of the data, dt' = 0.01
on both sides), i.e. a continuous-time residual.
"""
import json, sys, time
import numpy as np, jax, jax.numpy as jnp, optax
sys.path.insert(0, "/home/claude/thakur")
from pinn import init_mlp, taylor_mlp, rel_err, relL2  # noqa: E402


def D_and_derivs(pd, x, y):
    m, d1, _ = taylor_mlp(pd, jnp.stack([2 * x - 1, 2 * y - 1], 1), 0, False)
    D = jnp.exp(m)
    return D, D * 2 * d1[0], D * 2 * d1[1]


def run(case, mode, iters=3000, batch=2000, seed=0, data=None, tag=""):
    z = np.load(data or f"/home/claude/thakur/data_{case}_N501.npz")
    t, xg, C, Dt_full = z["t"], z["x"], z["c"].astype(np.float32), z["D"]
    N = len(xg); h = xg[1] - xg[0]; dtd = t[1] - t[0]
    m = 1 if mode == "cont" else int(round(float(mode) / dtd))
    dt = m * dtd
    sig = float(C.std())
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    pd = init_mlp(key, [2] + [128] * 8 + [1]); W, b = pd[-1]; pd[-1] = (W * 0.1, b + jnp.log(0.2))
    opt = optax.adam(optax.exponential_decay(1e-3, iters, 1e-2)); st = opt.init(pd)

    def loss(pd, x, y, cx, cy, lap, lhs):
        D, Dx, Dy = D_and_derivs(pd, x, y)
        f = Dx * cx + Dy * cy + D * lap
        return jnp.mean(((lhs - dt * f) / sig) ** 2)

    @jax.jit
    def step(pd, st, *b):
        l, g = jax.value_and_grad(loss)(pd, *b)
        u, st = opt.update(g, st, pd)
        return optax.apply_updates(pd, u), st, l

    k_lo = 1 if mode == "cont" else 0
    k_hi = len(t) - 1 - m
    t0 = time.time()
    for it in range(iters):
        i = rng.integers(1, N - 1, batch); j = rng.integers(1, N - 1, batch)
        k = rng.integers(max(k_lo, int(round(0.03 / dtd))), k_hi + (0 if mode == "cont" else 1), batch)   # skip the first 3 snapshots
        kn = k if mode == "cont" else k + m                                         # where f is evaluated
        c0 = C[kn, i, j]
        cx = (C[kn, i + 1, j] - C[kn, i - 1, j]) / (2 * h)
        cy = (C[kn, i, j + 1] - C[kn, i, j - 1]) / (2 * h)
        lap = (C[kn, i + 1, j] + C[kn, i - 1, j] + C[kn, i, j + 1] + C[kn, i, j - 1] - 4 * c0) / h ** 2
        if mode == "cont":
            lhs = (C[k + 1, i, j] - C[k - 1, i, j]) / (2 * dtd) * dt        # dt * c_t (central)
        else:
            lhs = C[k + m, i, j] - C[k, i, j]                                # c(t+dt) - c(t)
        pd, st, l = step(pd, st, jnp.asarray(xg[i]), jnp.asarray(xg[j]), jnp.asarray(cx), jnp.asarray(cy),
                         jnp.asarray(lap), jnp.asarray(lhs))
    sub = np.arange(0, N, 5); X, Y = np.meshgrid(xg[sub], xg[sub], indexing="ij")
    Dp = np.asarray(D_and_derivs(pd, jnp.asarray(X.ravel(), jnp.float32), jnp.asarray(Y.ravel(), jnp.float32))[0]).reshape(X.shape)
    Dt = Dt_full[np.ix_(sub, sub)]
    r = dict(case=case, mode=mode, data_dt=float(dtd), tag=tag, iters=iters, L_D=rel_err(Dt, Dp), relL2_D=relL2(Dt, Dp),
             mean_ratio=float(Dp.mean() / Dt.mean()), wall_s=time.time() - t0, final_loss=float(l))
    np.savez_compressed(f"/home/claude/thakur/runs/perfectc{tag}_{case}_{mode}.npz", D_pred=Dp, D_true=Dt)
    json.dump(r, open(f"/home/claude/thakur/runs/perfectc{tag}_{case}_{mode}.json", "w"), indent=1)
    print(json.dumps(r), flush=True)


if __name__ == "__main__":
    for case in sys.argv[1].split(","):
        for mode in sys.argv[2].split(","):
            run(case, mode)
