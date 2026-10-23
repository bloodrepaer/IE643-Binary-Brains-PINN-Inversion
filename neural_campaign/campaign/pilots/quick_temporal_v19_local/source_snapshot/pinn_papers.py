"""
pinn_papers.py -- the two reference papers' inverse PINNs, rebuilt as published, run on our problem.

  --paper thakur   Thakur, Esmaili, Libring, Solorio & Ardekani (2024), arXiv:2403.03970
                   * two fully connected nets, 8 hidden layers each; concentration net 128 wide
                     (D-net width is not given in the paper: 128 used here)
                   * variables made dimensionless: x/L, t/T, c/max(c), D*T/L^2
                   * NO continuous PDE residual: a backward-Euler "consistency loss"
                         c_pi(x,y,t) = c(x,y,t+dt) - dt * div(D grad c)(x,y,t+dt)   must equal c(x,y,t)
                     with dt = 0.01 (their data step: 100 steps on t in (0,1))
                   * loss = mean((c - c_data)^2)/sigma_c^2 + mean((c_pi - c)^2)/sigma_c^2,
                     sigma_c = std of the concentration data; no IC or BC terms (point clouds only)
                   * 5000 data points and 5000 equation points sampled every iteration
                   * Adam with a decaying learning rate (rates/iterations not given: 1e-3 -> 1e-5 here)
                   * not stated in the paper, chosen here: tanh, Xavier init, D = exp(net output)
  --paper thakur --icbc   same, but the initial field and the wall values of our problem are added to the
                   point cloud as extra known concentration points (they are known in our problem)

  --paper zhang    Zhang, Alla & Karniadakis (2026), arXiv:2606.12337, Appendix A.1 + Test 2
                   * state net and parameter net: 3 hidden layers x 32, tanh, Xavier-uniform, float64
                   * unknown is log D (D = exp(f)), as their log-permeability
                   * loss = l_pde + l_bc + l_ic + l_data + gamma/2 int |grad f|^2, all weights 1, gamma = 1e-3
                   * collocation drawn uniformly, 20,000 PDE + 512 BC resampled every outer loop,
                     512 IC + all data fixed (their time-dependent Burgers budget)
                   * Adam ~1000 steps with stepwise-decaying lr, then two-loop SSBroyden (self-scaled
                     Broyden quasi-Newton, strong-Wolfe line search): outer loop resamples points, inner
                     loop on a fixed batch, inverse Hessian symmetrised and kept if Cholesky passes
                   * time made dimensionless (t/T, D*T) so the horizon is 1 as in their benchmarks

    python pinn_papers.py --paper thakur --data-root datasets_v2_rk4_zero --exp-id <key> --out results_pp
"""
import argparse, csv, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import line_search

torch.set_default_dtype(torch.float64)


def mlp(n_in, width, depth, n_out=1):
    layers, d = [], n_in
    for _ in range(depth):
        lin = nn.Linear(d, width); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]; d = width
    out = nn.Linear(d, n_out); nn.init.xavier_uniform_(out.weight); nn.init.zeros_(out.bias)
    return nn.Sequential(*layers, out)


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def div_D_grad(c, D, xyt):
    """div(D grad c) at xyt (D depends on x, y only). Both nets stay in the graph."""
    dc = grad(c, xyt)
    return grad(D * dc[:, 0], xyt)[:, 0] + grad(D * dc[:, 1], xyt)[:, 1], dc


# --------------------------------------------------------------------------- data
def load(data_root, key):
    import pinn_inversion_bc as PB
    import generate_dataset_v2 as G
    cfg = PB.PINNConfig(data_root=data_root, output_dir="/tmp/_pp", exp_id=key)
    b = PB.load_v2_experiment(cfg)
    T = float(b["T_max"])
    X = b["X_data"].double().numpy()
    d = dict(T=T, bc_type=b["bc_type"], g_funcs=b.get("bc_edge_funcs"),
             xyt=np.c_[X[:, 0], X[:, 1], X[:, 2] / T], u=b["u_data"].double().numpy().ravel(),
             sigma=float(b["sigma_data"].double().numpy().ravel()[0]),
             X_ic=b["X_ic"].double().numpy(), u_ic=b["u_ic"].double().numpy().ravel(),
             xg=b["x_grid"], yg=b["y_grid"], D_true=b["D_true"], bumps=G.IC_BUMPS)
    return d


def ic_exact(x, y, bumps):
    return sum(bb["amp"] * np.exp(-((x - bb["cx"]) ** 2 + (y - bb["cy"]) ** 2) / (2 * bb["s"] ** 2)) for bb in bumps)


def wall_points(rng, n, d, t_hi=1.0):
    """n random wall points (x, y, t^) and their Dirichlet value; None if the BC is not given."""
    if d["bc_type"] not in ("dirichlet", "dirichlet_soft"):
        return None
    s = rng.uniform(0, 1, n); tb = rng.uniform(0, t_hi, n); e = rng.integers(0, 4, n)
    xb = np.select([e == 0, e == 1], [0.0, 1.0], s); yb = np.select([e == 2, e == 3], [0.0, 1.0], s)
    xb = np.where(e >= 2, s, xb); yb = np.where(e < 2, s, yb)
    g = np.zeros(n)
    if d["bc_type"] == "dirichlet_soft":
        for i, name in enumerate(("left", "right", "bottom", "top")):
            sel = e == i
            if sel.any():
                g[sel] = np.broadcast_to(np.asarray(d["g_funcs"][name](tb[sel] * d["T"]), dtype=float), (sel.sum(),))
    return np.c_[xb, yb, tb], g


# --------------------------------------------------------------------------- Thakur et al. 2024
def run_thakur(d, seed, steps, icbc=False, width=128, depth=8, n_pts=5000, dt=0.01, log=print):
    """Single precision (TensorFlow's default, which the paper used)."""
    torch.set_default_dtype(torch.float32)
    TT = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    rng = np.random.default_rng(seed)
    c_scale = float(max(np.abs(d["u_ic"]).max(), np.abs(d["u"]).max()))       # c / max(c)
    pool_x, pool_c = d["xyt"], d["u"] / c_scale
    if icbc:                                                                       # known IC + wall values as data
        Xi = d["X_ic"]; pool_x = np.r_[pool_x, np.c_[Xi[:, 0], Xi[:, 1], np.zeros(len(Xi))]]
        pool_c = np.r_[pool_c, d["u_ic"] / c_scale]
        wp = wall_points(rng, 4000, d)
        if wp is not None:
            pool_x = np.r_[pool_x, wp[0]]; pool_c = np.r_[pool_c, wp[1] / c_scale]
    sig_c = float(np.std(pool_c))                                                  # std of the concentration point cloud
    c_net = mlp(3, width, depth); D_net = mlp(2, width, depth)
    params = list(c_net.parameters()) + list(D_net.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(1e-2) ** (1 / steps))   # 1e-3 -> 1e-5
    Px, Pc = TT(pool_x), TT(pool_c)
    hist = []
    for k in range(1, steps + 1):
        idx = rng.choice(len(Pc), n_pts, replace=False) if len(Pc) > n_pts else slice(None)
        xe = TT(np.c_[rng.uniform(0, 1, (n_pts, 2)), rng.uniform(0, 1 - dt, n_pts)])
        xe1 = (xe + torch.tensor([0.0, 0.0, dt])).requires_grad_(True)
        c1 = c_net(xe1).squeeze(-1); D = torch.exp(D_net(xe1[:, :2]).squeeze(-1))
        dv, _ = div_D_grad(c1, D, xe1)
        c_pi = c1 - dt * dv                                                       # backward Euler step back
        L_data = torch.mean((c_net(Px[idx]).squeeze(-1) - Pc[idx]) ** 2) / sig_c ** 2
        L_cons = torch.mean((c_pi - c_net(xe).squeeze(-1)) ** 2) / sig_c ** 2
        loss = L_data + L_cons
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if k == 1 or k % max(1, steps // 20) == 0:
            hist.append(dict(step=k, data=L_data.item(), cons=L_cons.item()))
            log(f"  thakur {k:6d}  data {L_data.item():.3e}  consistency {L_cons.item():.3e}")
    T = d["T"]
    XX, YY = np.meshgrid(d["xg"], d["yg"], indexing="ij")
    with torch.no_grad():
        D_hat = np.exp(D_net(TT(np.c_[XX.ravel(), YY.ravel()])).squeeze(-1).double().numpy()).reshape(XX.shape) / T
        u_fit = c_net(TT(d["xyt"])).squeeze(-1).double().numpy() * c_scale
    torch.set_default_dtype(torch.float64)
    return D_hat, u_fit, hist


# --------------------------------------------------------------------------- SSBroyden (Kiyani et al. 2025, eq. 8)
def ssbroyden_update(H, s, y, g, alpha):
    ys = float(y @ s)
    if ys <= 1e-300:
        return H
    Hy = H @ y; yHy = float(y @ Hy)
    sBs = -alpha * float(s @ g)                        # B_k s_k = -alpha_k grad f_k
    b = sBs / ys; h = yHy / ys; a = b * h - 1.0
    n = len(s)
    if a <= 1e-12:                                      # b h = 1: Broyden family collapses to BFGS
        theta, tau = 0.0, min(1.0, 1.0 / b)
    else:
        c = math.sqrt(a / (1 + a))
        rho_m = min(1.0, h * (1 - c)); theta_m = (rho_m - 1) / a; theta_p = 1 / rho_m
        theta = max(theta_m, min(theta_p, (1 - b) / b))
        rho_p = min(1.0, 1 / b); sig = 1 + theta * a; sigN = abs(sig) ** (1 / (1 - n))
        tau = min(rho_p * sigN, sig) if theta <= 0 else rho_p * min(sigN, 1 / theta)
    phi = (1 - theta) / (1 + a * theta)
    v = s / ys - Hy / yHy
    H -= np.outer(Hy, Hy) / yHy
    H += phi * yHy * np.outer(v, v)
    H /= tau
    H += np.outer(s, s) / ys
    return H


def ssbroyden(fg, x, H, iters, log=print, tag=""):
    f, g = fg(x)
    for k in range(iters):
        p = -H @ g
        if g @ p >= 0:                                  # not a descent direction: restart curvature
            H = np.eye(len(x)); p = -g
        cache = {}
        def fv(z):
            kz = z.tobytes()
            if kz not in cache: cache[kz] = fg(z)
            return cache[kz][0]
        def gv(z):
            kz = z.tobytes()
            if kz not in cache: cache[kz] = fg(z)
            return cache[kz][1]
        alpha = line_search(fv, gv, x, p, gfk=g, old_fval=f, c1=1e-4, c2=0.9, maxiter=30)[0]
        if alpha is None:
            H = np.eye(len(x)); p = -g
            alpha = line_search(fv, gv, x, p, gfk=g, old_fval=f, c1=1e-4, c2=0.9, maxiter=30)[0]
            if alpha is None:
                log(f"  {tag} line search failed at inner iteration {k}: stopping this loop"); break
        x_new = x + alpha * p
        f_new, g_new = fv(x_new), gv(x_new)
        H = ssbroyden_update(H, alpha * p, g_new - g, g, alpha)
        x, f, g = x_new, f_new, g_new
        if not np.isfinite(f) or np.linalg.norm(g) < 1e-12:
            break
    return x, H, f


# --------------------------------------------------------------------------- Zhang, Alla & Karniadakis 2026
def run_zhang(d, seed, adam_steps=1000, outer=10, inner=500, gamma=1e-3, width=32, depth=3,
              n_pde=20000, n_bc=512, n_ic=512, log=print):
    rng = np.random.default_rng(seed)
    u_net = mlp(3, width, depth); f_net = mlp(2, width, depth)
    params = list(u_net.parameters()) + list(f_net.parameters())
    xd, ud = torch.tensor(d["xyt"]), torch.tensor(d["u"])
    xi = rng.uniform(0, 1, (n_ic, 2)); Xic = torch.tensor(np.c_[xi, np.zeros(n_ic)])
    uic = torch.tensor(ic_exact(xi[:, 0], xi[:, 1], d["bumps"]))

    def batch():
        col = torch.tensor(rng.uniform(0, 1, (n_pde, 3)))
        wp = wall_points(rng, n_bc, d)
        return col, (None if wp is None else (torch.tensor(wp[0]), torch.tensor(wp[1])))

    def losses(col, wb):
        col = col.clone().requires_grad_(True)
        u = u_net(col).squeeze(-1); f = f_net(col[:, :2]).squeeze(-1); D = torch.exp(f)
        dv, du = div_D_grad(u, D, col)
        l_pde = torch.mean((du[:, 2] - dv) ** 2)
        l_bc = torch.mean((u_net(wb[0]).squeeze(-1) - wb[1]) ** 2) if wb is not None else torch.zeros(())
        l_ic = torch.mean((u_net(Xic).squeeze(-1) - uic) ** 2)
        l_data = torch.mean((u_net(xd).squeeze(-1) - ud) ** 2)
        df = grad(f, col); reg = 0.5 * gamma * torch.mean(df[:, 0] ** 2 + df[:, 1] ** 2)   # unit-area integral
        tot = l_pde + l_bc + l_ic + l_data + reg
        return tot, dict(pde=l_pde.item(), bc=l_bc.item(), ic=l_ic.item(), data=l_data.item(), reg=reg.item())

    hist = []
    opt = torch.optim.Adam(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, adam_steps // 4), gamma=0.5)
    col, wb = batch()
    for k in range(1, adam_steps + 1):
        if k % 100 == 1:
            col, wb = batch()
        opt.zero_grad(); tot, parts = losses(col, wb); tot.backward(); opt.step(); sched.step()
        if k == 1 or k % 250 == 0:
            hist.append(dict(stage="adam", step=k, total=tot.item(), **parts))
            log(f"  zhang adam {k:5d} total {tot.item():.3e}  " + "  ".join(f"{a} {b:.2e}" for a, b in parts.items()))
    from torch.nn.utils import parameters_to_vector as p2v, vector_to_parameters as v2p
    x = p2v(params).detach().numpy().copy(); H = np.eye(len(x))
    for o in range(1, outer + 1):
        col, wb = batch()
        def fg(z):
            v2p(torch.from_numpy(z.copy()), params)
            tot, _ = losses(col, wb)
            gs = torch.autograd.grad(tot, params)
            return tot.item(), torch.cat([g_.reshape(-1) for g_ in gs]).numpy()
        x, H, f = ssbroyden(fg, x, H, inner, log=log, tag=f"outer {o}")
        H = 0.5 * (H + H.T)
        try:
            np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            H = np.eye(len(x))
        v2p(torch.from_numpy(x.copy()), params)
        tot, parts = losses(col, wb)
        hist.append(dict(stage="ssbroyden", step=o, total=tot.item(), **parts))
        log(f"  zhang ssbroyden outer {o:2d} total {tot.item():.3e}  " + "  ".join(f"{a} {b:.2e}" for a, b in parts.items()))
    T = d["T"]
    XX, YY = np.meshgrid(d["xg"], d["yg"], indexing="ij")
    with torch.no_grad():
        D_hat = np.exp(f_net(torch.tensor(np.c_[XX.ravel(), YY.ravel()].astype(float))).squeeze(-1).numpy()).reshape(XX.shape) / T
        u_fit = u_net(xd).squeeze(-1).numpy()
    return D_hat, u_fit, hist


# --------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", choices=["thakur", "zhang"], required=True)
    ap.add_argument("--icbc", action="store_true", help="thakur: add the known IC and wall values to the point cloud")
    ap.add_argument("--data-root", required=True); ap.add_argument("--exp-id", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=10000, help="thakur: Adam iterations")
    ap.add_argument("--outer", type=int, default=10); ap.add_argument("--inner", type=int, default=500)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    variant = a.paper + ("_icbc" if a.icbc else "")
    for key in a.exp_id:
        torch.manual_seed(a.seed)
        t0 = time.time()
        d = load(a.data_root, key)
        od = Path(a.out) / f"{key}__{variant}__s{a.seed}"; od.mkdir(parents=True, exist_ok=True)
        logf = open(od / "log.txt", "w")
        def log(msg):
            print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
        log(f"{variant} on {key}: {len(d['u'])} readings, sigma {d['sigma']:.3g}, bc {d['bc_type']}")
        if a.paper == "thakur":
            D_hat, u_fit, hist = run_thakur(d, a.seed, a.steps, icbc=a.icbc, log=log)
        else:
            D_hat, u_fit, hist = run_zhang(d, a.seed, outer=a.outer, inner=a.inner, log=log)
        np.savez_compressed(od / "D_hat.npz", D_hat=D_hat, x=d["xg"], y=d["yg"])
        m = dict(exp_id=key, data_root=a.data_root, variant=variant, seed=a.seed, elapsed_sec=round(time.time() - t0, 1),
                 n_readings=len(d["u"]), D_hat_mean=float(D_hat.mean()), history=hist,
                 rms_unet_over_sigma=float(np.sqrt(np.mean((u_fit - d["u"]) ** 2)) / d["sigma"]) if d["sigma"] > 0 else None,
                 rms_unet_rel=float(np.sqrt(np.mean((u_fit - d["u"]) ** 2)) / np.sqrt(np.mean(d["u"] ** 2))))
        Dt = d["D_true"]
        if Dt is not None and Dt.shape == D_hat.shape:
            m["rel_l2_D"] = float(np.linalg.norm(D_hat - Dt) / np.linalg.norm(Dt))
            m["thakur_err_D"] = float(np.sqrt(np.mean((D_hat - Dt) ** 2) / np.mean((Dt - Dt.mean()) ** 2)))
        json.dump(m, open(od / "metrics.json", "w"), indent=1, default=float)
        log(f"  --> {variant} {key}: relL2(D) {m.get('rel_l2_D', float('nan')):.3f}  Thakur err {m.get('thakur_err_D', float('nan')):.3f}  "
            f"u fit rel {m['rms_unet_rel']:.3f}  {m['elapsed_sec']:.0f}s")
        row = {k: v for k, v in m.items() if k != "history"}
        f = Path(a.out) / "summary.csv"; new = not f.exists()
        with open(f, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row), lineterminator="\n")
            if new: w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    main()
