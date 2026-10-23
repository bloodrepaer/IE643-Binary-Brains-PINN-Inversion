"""
pinn_literature.py -- an inverse PINN built the way the reference papers build theirs.

Follows Zhang, Alla & Karniadakis (2026) and Thakur et al. (2024) wherever they agree:
  * two plain tanh MLPs, Xavier-uniform init: u_net(x, y, t) and m_net(x, y), D = exp(m)
    (no Fourier features, no hand-built heat-kernel base, no hard wall windows or tapers)
  * nondimensional variables (Thakur): t^ = t/T, u^ = u/u_scale, D^ = D*T/L^2 (L = 1 here)
  * standard residual r = u_t - div(D grad u) with gradients flowing into BOTH networks
  * all loss weights = 1; data, IC, BC and residual terms divided by the data's spread (Thakur's sigma_c)
  * soft IC and BC: IC points at t = 0; wall points with u = g(t) when the BC is given, nothing when it is
    unknown ("free"); same detection as pinn_inversion_bc.py
  * 4000 collocation points, resampled every Adam step; Adam with stepwise-decaying lr, then an L-BFGS
    polish on fixed batches resampled between outer loops (Zhang's two-stage schedule; L-BFGS stands in
    for their SSBroyden); float64
  * D starts from the data-fitted constant (the adjoint's stage 0), not from the problem file
  * weak H1 regulariser on m (gamma * mean |grad m|^2)
Only the data loader is shared with pinn_inversion_bc.py.

    python pinn_literature.py --data-root datasets_v2_rk4_zero --exp-id <key> --out results_pinn_lit [--seed 42]
"""
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

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


def run(data_root, key, out, seed=42, adam_steps=3000, lbfgs_outer=2, lbfgs_iters=250, n_col=4000,
        n_bc=800, gamma=1e-4, u_width=64, u_depth=4, m_width=32, m_depth=3, threads=1, ic_base=False):
    import pinn_inversion_bc as PB
    from pinn_inversion_v2 import _best_constant_logD
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(threads)
    t0 = time.time()
    cfg = PB.PINNConfig(data_root=data_root, output_dir=out, exp_id=key)
    b = PB.load_v2_experiment(cfg)
    T = float(b["T_max"]); bc_type = b["bc_type"]; g_funcs = b.get("bc_edge_funcs")
    ds_dir = Path(b["inspection"]["ds_dir"])
    # nondimensionalisation
    X = b["X_data"].double().numpy(); u_obs = b["u_data"].double().numpy().ravel()
    sigma = float(b["sigma_data"].double().numpy().ravel()[0])
    u_scale = float(max(np.abs(b["u_ic"].numpy()).max(), np.abs(u_obs).max(), 1e-12))
    s_u = float(np.std(u_obs / u_scale)) or 1.0                       # Thakur's sigma_c
    Xd = torch.tensor(np.c_[X[:, 0], X[:, 1], X[:, 2] / T]); ud = torch.tensor(u_obs / u_scale)
    Xic = torch.tensor(b["X_ic"].double().numpy()); uic = torch.tensor(b["u_ic"].double().numpy().ravel() / u_scale)
    m0 = _best_constant_logD(ds_dir, key, math.log(0.1 / T))            # log of physical D
    u_net = mlp(3, u_width, u_depth); m_net = mlp(2, m_width, m_depth)
    with torch.no_grad():
        m_net[-1].bias.fill_(m0 + math.log(T))                          # m_net outputs log D^ = log(D T)
    params = list(u_net.parameters()) + list(m_net.parameters())

    import generate_dataset_v2 as G
    D_base = math.exp(m0)                                                # fixed spreading rate for the base

    def base(xyt):
        """Heat-kernel spreading of the known IC bumps (the idea from pinn_inversion.py): carries the narrow
        Gaussians a plain tanh MLP cannot represent; the network only learns the smooth correction."""
        x_, y_, t_ = xyt[:, 0], xyt[:, 1], xyt[:, 2] * T
        out = 0.0
        for bb in G.IC_BUMPS:
            v = bb["s"] ** 2 + 2 * D_base * t_
            out = out + bb["amp"] * (bb["s"] ** 2 / v) * torch.exp(-((x_ - bb["cx"]) ** 2 + (y_ - bb["cy"]) ** 2) / (2 * v))
        return out / u_scale

    def u_of(xyt):
        if ic_base:
            return base(xyt) + xyt[:, 2] * u_net(xyt).squeeze(-1)       # IC exact at t = 0 (Lagaris-style)
        return u_net(xyt).squeeze(-1)

    def batch(rng):
        col = torch.tensor(rng.uniform(0, 1, (n_col, 3)))
        s = torch.tensor(rng.uniform(0, 1, (n_bc,))); tb = rng.uniform(0, 1, (n_bc,))
        e = rng.integers(0, 4, n_bc)
        xb = np.where(e == 0, 0.0, np.where(e == 1, 1.0, s.numpy())); yb = np.where(e == 2, 0.0, np.where(e == 3, 1.0, s.numpy()))
        xb = np.where(e < 2, xb, s.numpy()); yb = np.where(e < 2, s.numpy(), yb)
        g = np.zeros(n_bc)
        if bc_type == "dirichlet_soft":
            for i, name in enumerate(("left", "right", "bottom", "top")):
                sel = e == i
                if sel.any():
                    g[sel] = np.broadcast_to(np.asarray(g_funcs[name](tb[sel] * T), dtype=float), (sel.sum(),))
            g = g / u_scale
        return col, torch.tensor(np.c_[xb, yb, tb]), torch.tensor(g)

    def losses(col, xb, gb):
        col = col.clone().requires_grad_(True)
        u = u_of(col); m = m_net(col[:, :2]).squeeze(-1); D = torch.exp(m)
        du = grad(u, col); ux, uy, ut = du[:, 0], du[:, 1], du[:, 2]
        flux_x = D * ux; flux_y = D * uy
        div = grad(flux_x, col)[:, 0] + grad(flux_y, col)[:, 1]          # div(D grad u), both nets in the graph
        r = ut - div
        L_res = torch.mean(r ** 2) / s_u ** 2
        L_data = torch.mean((u_of(Xd) - ud) ** 2) / s_u ** 2
        L_ic = torch.mean((u_of(Xic) - uic) ** 2) / s_u ** 2
        L_bc = torch.mean((u_of(xb) - gb) ** 2) / s_u ** 2 if bc_type in ("dirichlet", "dirichlet_soft") else torch.zeros(())
        dm = grad(m, col); L_reg = gamma * torch.mean(dm[:, 0] ** 2 + dm[:, 1] ** 2)
        tot = L_data + L_res + L_ic + L_bc + L_reg
        return tot, dict(data=L_data.item(), res=L_res.item(), ic=L_ic.item(), bc=L_bc.item(), reg=L_reg.item())

    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, adam_steps // 3), gamma=0.3)
    hist = []
    for k in range(1, adam_steps + 1):
        col, xb, gb = batch(rng)
        opt.zero_grad(); tot, parts = losses(col, xb, gb); tot.backward(); opt.step(); sched.step()
        if k % 250 == 0 or k == 1:
            hist.append(dict(stage="adam", step=k, total=tot.item(), **parts))
            print(f"  adam {k:5d}  total {tot.item():.3e}  data {parts['data']:.2e}  res {parts['res']:.2e}  ic {parts['ic']:.2e}  bc {parts['bc']:.2e}", flush=True)
    for outer in range(lbfgs_outer):
        col, xb, gb = batch(rng)
        lb = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_iters, history_size=50, line_search_fn="strong_wolfe",
                               tolerance_grad=1e-12, tolerance_change=1e-15)
        def closure():
            lb.zero_grad(); t_, _ = losses(col, xb, gb); t_.backward(); return t_
        lb.step(closure)
        tot, parts = losses(col, xb, gb)
        hist.append(dict(stage="lbfgs", step=outer + 1, total=tot.item(), **parts))
        print(f"  lbfgs {outer + 1}  total {tot.item():.3e}  data {parts['data']:.2e}  res {parts['res']:.2e}", flush=True)

    xg, yg = b["x_grid"], b["y_grid"]
    XX, YY = np.meshgrid(xg, yg, indexing="ij")
    with torch.no_grad():
        D_hat = np.exp(m_net(torch.tensor(np.c_[XX.ravel(), YY.ravel()].astype(np.float64))).squeeze(-1).numpy()).reshape(XX.shape) / T
        u_fit = u_of(Xd).numpy() * u_scale
    od = Path(out) / f"{key}__s{seed}"; od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / "D_hat.npz", D_hat=D_hat, x=xg, y=yg)
    m = dict(exp_id=key, seed=seed, elapsed_sec=round(time.time() - t0, 1), bc_type=bc_type, ic_base=ic_base,
             rms_unet_over_sigma=float(np.sqrt(np.mean((u_fit - u_obs) ** 2)) / sigma) if sigma > 0 else np.nan,
             D_hat_mean=float(D_hat.mean()), D_init=float(math.exp(m0)), history=hist)
    if b["D_true"] is not None and b["D_true"].shape == D_hat.shape:
        Dt = b["D_true"]; m["rel_l2_D"] = float(np.linalg.norm(D_hat - Dt) / np.linalg.norm(Dt))
    json.dump(m, open(od / "metrics.json", "w"), indent=1, default=float)
    print(f"  --> {key} seed {seed}: relL2(D)={m.get('rel_l2_D', float('nan')):.3f}  u_net RMS/sigma={m['rms_unet_over_sigma']:.2f}  "
          f"mean D {D_hat.mean():.3f} (init {math.exp(m0):.3f})  {m['elapsed_sec']:.0f}s", flush=True)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True); ap.add_argument("--exp-id", nargs="+", required=True)
    ap.add_argument("--out", default="results_pinn_lit"); ap.add_argument("--seed", type=int, nargs="+", default=[42])
    ap.add_argument("--adam-steps", type=int, default=3000); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--ic-base", action="store_true", help="heat-kernel IC base (see base())")
    a = ap.parse_args()
    import csv
    for s in a.seed:
        for k in a.exp_id:
            r = run(a.data_root, k, a.out, seed=s, adam_steps=a.adam_steps, threads=a.threads, ic_base=a.ic_base)
            row = {kk: v for kk, v in r.items() if kk != "history"}
            f = Path(a.out) / "summary.csv"; new = not f.exists()
            with open(f, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row), lineterminator="\n")
                if new: w.writeheader()
                w.writerow(row)
