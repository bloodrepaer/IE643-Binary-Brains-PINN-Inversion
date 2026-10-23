"""
Thakur, Esmaili, Libring, Solorio & Ardekani (2024), "Inverse resolution of spatially varying
diffusion coefficient using physics-informed neural networks" -- independent re-implementation.

What the paper specifies (Sec. 2, 3.1):
  * two fully connected networks c~(x,y,t) and D~(x,y), 8 hidden layers each; c~ has 128 neurons
  * backward-Euler "physics-informed" concentration, eq. (5):
        c_pi(t) = c~(t+dt) - dt * f(c~(t+dt), D~),   f = c_t of eq. (3) = D_x c_x + D_y c_y + D (c_xx + c_yy)
    (eq. (4) as printed has sign errors; the version that makes eq. (5) a backward-Euler step is used)
  * loss, eq. (6): mean |(c~ - c)/sigma_c|^2 over data points + mean |(c_pi - c~)/sigma_c|^2 over
    equation points (temporal consistency loss)
  * Adam with a decaying learning rate; 5000 random space-time points per iteration
  * error metric, eq. (7): L(a, a_pred) = sqrt( mean (a - a_pred)^2 / mean (a - mean a)^2 )

What it does not specify (our choices, stated in every output):
  * D~ width (128, same as c~), activation (tanh), D positivity (D = exp(output)),
  * dt in eq. (5) (default: the data time step 0.01), number of iterations, learning-rate schedule,
  * equation points (default: the same random points as the data batch).

mode="be"   : the paper's backward-Euler consistency loss
mode="cont" : a standard continuous-time PINN residual (c_t - f)^2, for the dt-bias ablation
"""
import argparse, json, os, time
import numpy as np
import jax, jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", False)


def init_mlp(key, sizes):
    params = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        W = jax.random.normal(k, (a, b)) * jnp.sqrt(2.0 / (a + b))      # Glorot normal
        params.append((W, jnp.zeros(b)))
    return params


def mlp(params, z):
    for W, b in params[:-1]:
        z = jnp.tanh(z @ W + b)
    W, b = params[-1]
    return z @ W + b


def c_fn(pc, x, y, t):
    return mlp(pc, jnp.stack([2 * x - 1, 2 * y - 1, 2 * t - 1]))[0]


def D_fn(pd, x, y):
    return jnp.exp(mlp(pd, jnp.stack([2 * x - 1, 2 * y - 1]))[0])


def taylor_mlp(params, Z, n_space, second):
    """Batched forward propagation of an MLP and its input derivatives (exact, by hand).
    Z: (B, d_in) normalised inputs; derivatives w.r.t. the first n_space inputs (first and,
    if second, pure second) and, for d_in = 3, the first derivative w.r.t. input 3 (time).
    Returns out, [d/dz_k], [d2/dz_k2] for the final linear layer."""
    B, d_in = Z.shape
    n1 = d_in                                  # first derivatives for all inputs
    a = Z
    da = [jnp.broadcast_to(jnp.eye(d_in)[k], (B, d_in)) for k in range(n1)]
    dda = [jnp.zeros((B, d_in)) for _ in range(n_space)] if second else []
    for W, b in params[:-1]:
        h = a @ W + b
        dh = [d @ W for d in da]
        ddh = [d @ W for d in dda]
        a = jnp.tanh(h); s = 1 - a ** 2
        dda = [s * ddh[k] - 2 * a * s * dh[k] ** 2 for k in range(len(ddh))]
        da = [s * d for d in dh]
    W, b = params[-1]
    return (a @ W + b)[:, 0], [(d @ W)[:, 0] for d in da], [(d @ W)[:, 0] for d in dda]


def c_and_derivs(pc, x, y, t):
    """c, c_x, c_y, c_t, c_xx, c_yy in physical coordinates (inputs are 2*(.)-1)."""
    out, d1, d2 = taylor_mlp(pc, jnp.stack([2 * x - 1, 2 * y - 1, 2 * t - 1], 1), 2, True)
    return out, 2 * d1[0], 2 * d1[1], 2 * d1[2], 4 * d2[0], 4 * d2[1]


def D_and_derivs(pd, x, y):
    m, d1, _ = taylor_mlp(pd, jnp.stack([2 * x - 1, 2 * y - 1], 1), 0, False)
    D = jnp.exp(m)
    return D, D * 2 * d1[0], D * 2 * d1[1]


def c_batch(pc, x, y, t):
    return mlp(pc, jnp.stack([2 * x - 1, 2 * y - 1, 2 * t - 1], 1))[:, 0]


def make_loss(sigma_c, dt, mode):
    def f_of(pc, pd, x, y, t):
        c, cx, cy, ct, cxx, cyy = c_and_derivs(pc, x, y, t)
        D, Dx, Dy = D_and_derivs(pd, x, y)
        return c, ct, Dx * cx + Dy * cy + D * (cxx + cyy)

    def loss(params, xb, yb, tb, cb, xe, ye, te):
        pc, pd = params
        l_data = jnp.mean(((c_batch(pc, xb, yb, tb) - cb) / sigma_c) ** 2)
        if mode == "be":
            c_next, _, f_next = f_of(pc, pd, xe, ye, te + dt)
            c_pi = c_next - dt * f_next
            l_eq = jnp.mean(((c_pi - c_batch(pc, xe, ye, te)) / sigma_c) ** 2)
        else:
            _, ct, f = f_of(pc, pd, xe, ye, te)
            l_eq = jnp.mean((dt * (ct - f) / sigma_c) ** 2)    # same scale as the BE residual
        return l_data + l_eq, (l_data, l_eq)
    return loss


def rel_err(a, ap):
    """Paper's eq. (7)."""
    a, ap = np.asarray(a, float), np.asarray(ap, float)
    return float(np.sqrt(np.mean((a - ap) ** 2) / np.mean((a - a.mean()) ** 2)))


def relL2(a, ap):
    return float(np.linalg.norm(ap - a) / np.linalg.norm(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="I")
    ap.add_argument("--data", default=None)
    ap.add_argument("--n-spatial", type=int, default=0, help="0 = every mesh node; else this many random spatial points")
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian noise std as a fraction of max(c) (=1)")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=5000)
    ap.add_argument("--width-d", type=int, default=128)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--mode", default="be", choices=["be", "cont"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-end", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/claude/thakur/runs")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    z = np.load(a.data or f"/home/claude/thakur/data_{a.case}_N501.npz")
    t, xg, D_true, C = z["t"], z["x"], z["D"], z["c"]
    N = len(xg)
    rng = np.random.default_rng(a.seed)
    # spatial points: all nodes, or a random subset (fixed for the whole run, as in Table 2)
    if a.n_spatial and a.n_spatial < N * N:
        flat = rng.choice(N * N, a.n_spatial, replace=False)
    else:
        flat = np.arange(N * N)
    ix, iy = np.unravel_index(flat, (N, N))
    nT = len(t)
    # point cloud: every chosen spatial point at every time
    data_c = C.reshape(nT, N * N)[:, flat]                                # (nT, nS)
    if a.noise > 0:
        data_c = data_c + a.noise * rng.standard_normal(data_c.shape).astype(np.float32)
    xs, ys = xg[ix].astype(np.float32), xg[iy].astype(np.float32)
    sigma_c = float(data_c.std())
    n_pts = data_c.size
    print(f"case {a.case}: {len(flat)} spatial points x {nT} times = {n_pts} readings, sigma_c={sigma_c:.3f}, "
          f"noise={a.noise}, mode={a.mode}, dt={a.dt}", flush=True)

    key = jax.random.PRNGKey(a.seed)
    k1, k2 = jax.random.split(key)
    pc = init_mlp(k1, [3] + [128] * 8 + [1])
    pd = init_mlp(k2, [2] + [a.width_d] * 8 + [1])
    W, b = pd[-1]; pd[-1] = (W * 0.1, b + jnp.log(0.2))                  # start near D = 0.2
    params = (pc, pd)
    sched = optax.exponential_decay(a.lr, a.iters, a.lr_end / a.lr)
    opt = optax.adam(sched)
    state = opt.init(params)
    loss = make_loss(sigma_c, a.dt, a.mode)

    @jax.jit
    def step(params, state, xb, yb, tb, cb, xe, ye, te):
        (l, aux), g = jax.value_and_grad(loss, has_aux=True)(params, xb, yb, tb, cb, xe, ye, te)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, l, aux

    t_max_eq = 1.0 - a.dt if a.mode == "be" else 1.0
    hist, t0 = [], time.time()
    B = min(a.batch, n_pts)
    for it in range(1, a.iters + 1):
        k = rng.integers(0, n_pts, B)
        ti, si = np.unravel_index(k, data_c.shape)
        xb, yb, tb, cb = xs[si], ys[si], t[ti].astype(np.float32), data_c[ti, si]
        # equation points: same spatial points, times drawn in [0, 1 - dt] (BE) so t+dt stays in range
        te = (rng.random(B) * t_max_eq).astype(np.float32)
        params, state, l, (ld, le) = step(params, state, xb, yb, tb, cb, xb, yb, te)
        if it % 500 == 0 or it == 1:
            hist.append((it, time.time() - t0, float(l), float(ld), float(le)))
            if it % 2000 == 0 or it == 1:
                print(f"  it {it:6d}  loss {float(l):.3e}  data {float(ld):.3e}  eq {float(le):.3e}  {time.time()-t0:.0f}s", flush=True)

    # evaluation on a 101 x 101 grid (D) and 101 x 101 x 100 (c)
    pc, pd = params
    sub = np.arange(0, N, (N - 1) // 100)
    Xe, Ye = np.meshgrid(xg[sub], xg[sub], indexing="ij")
    Dp = np.asarray(jax.jit(jax.vmap(D_fn, (None, 0, 0)))(pd, jnp.asarray(Xe.ravel(), jnp.float32),
                                                          jnp.asarray(Ye.ravel(), jnp.float32))).reshape(Xe.shape)
    Dt = D_true[np.ix_(sub, sub)]
    vc = jax.jit(jax.vmap(c_fn, (None, 0, 0, 0)))
    Cp = np.stack([np.asarray(vc(pc, jnp.asarray(Xe.ravel(), jnp.float32), jnp.asarray(Ye.ravel(), jnp.float32),
                                 jnp.full(Xe.size, tt, jnp.float32))).reshape(Xe.shape) for tt in t])
    Ct = C[:, sub][:, :, sub]
    inner = np.s_[1:-1, 1:-1]
    res = dict(case=a.case, n_spatial=int(len(flat)), n_readings=int(n_pts), noise=a.noise, mode=a.mode, dt=a.dt,
               iters=a.iters, batch=B, width_d=a.width_d, seed=a.seed, wall_s=time.time() - t0,
               L_D=rel_err(Dt, Dp), L_c=rel_err(Ct, Cp), relL2_D=relL2(Dt, Dp),
               L_D_interior=rel_err(Dt[inner], Dp[inner]), relL2_D_interior=relL2(Dt[inner], Dp[inner]),
               final_loss=float(l))
    tag = a.tag or f"{a.case}_n{a.n_spatial or 'all'}_noise{a.noise}_{a.mode}_dt{a.dt}_it{a.iters}_s{a.seed}"
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, tag + ".npz"), D_pred=Dp, D_true=Dt, x=xg[sub],
                        c_pred_mid=Cp[49], c_true_mid=Ct[49], hist=np.array(hist))
    json.dump(res, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)


if __name__ == "__main__":
    main()
