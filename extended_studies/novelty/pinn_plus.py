"""
pinn_plus.py -- the project's PINN (submission/pinn_inversion.py) with switchable extensions.

With every option at its default, `train()` repeats train_single_experiment step for step (same seed,
same random draws in the same order, same schedule, same best-checkpoint rule), and returns the same
D_hat: test_novelty.py checks this. The options only add to that loop:

  init_D        warm start: fit the D-network to a given D field first (the reverse hybrid, adjoint -> PINN),
                exactly as pinn_inversion's --warmstart-from-adjoint does
  coef="xyT"    third network g(u): D(x, y, u) = exp(m(x, y) + g(u) - g(0)); D(x, y, 0) is the reference field
  train_rows    use only these sensor rows in the data loss (sensor-split validation); the rest are scored
                afterwards as validation rows
  rar           residual-based adaptive refinement of the collocation points
  augment       extra pseudo-readings interpolated between sensors (thin-plate RBF per reading time)

The networks, loss weights, optimiser and schedule are those of pinn_inversion.py (PINNConfig defaults).
"""
import copy
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from common import pinn_main, COLLECTIONS, EXT

P = pinn_main()


@dataclass
class PlusOptions:
    coef: str = "xy"                 # "xy" (project PINN) or "xyT" (third network for temperature dependence)
    g_hidden: int = 16
    g_layers: int = 2
    init_D: Optional[np.ndarray] = None
    train_rows: Optional[np.ndarray] = None
    rar: bool = False
    rar_start: float = 0.2           # fraction of epochs before the first refinement
    rar_every: int = 100
    rar_pool: int = 8192
    rar_add: int = 64
    rar_max: int = 384               # adaptive points replace up to this many of the random ones
    augment: float = 0.0             # pseudo-readings per real reading (0 = off)
    augment_sigma_factor: float = 2.0
    record_every: int = 0            # >0: store D_hat snapshots every this many epochs (convergence curves)
    pde_split: float = 0.90          # share of the PDE loss that trains D (u detached); the rest trains u (D detached)
    taper_k: float = 6.0             # steepness of the D-network's wall taper tanh(k x)...; 6 = project PINN,
                                     # larger = D free closer to the walls (density diagnosis, D2)


# ---------------------------------------------------------------------------------------------
# third network: g(u)
# ---------------------------------------------------------------------------------------------
class TempPINN(P.InverseDiffusionPINN):
    """Project PINN plus g(u). Built after the parent, so the u- and D-networks get exactly the
    same initial weights as in the plain PINN with the same seed."""

    def __init__(self, cfg, Lx, Ly, T_max, m_init_bias, bc_type, hidden=16, layers=2):
        super().__init__(cfg, Lx, Ly, T_max, m_init_bias=m_init_bias, bc_type=bc_type)
        mods, d = [], 1
        for _ in range(layers):
            lin = nn.Linear(d, hidden)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            mods += [lin, nn.Tanh()]
            d = hidden
        out = nn.Linear(d, 1)
        nn.init.xavier_normal_(out.weight, gain=0.01)
        nn.init.zeros_(out.bias)
        self.g_net = nn.Sequential(*mods, out)

    def g(self, u):
        return self.g_net(u) - self.g_net(torch.zeros_like(u[:1]))

    def forward_D_xyu(self, xy, u):
        return torch.exp(torch.clamp(self.forward_m(xy) + self.g(u), min=-5.0, max=2.0))


def pde_loss_xyT(model, X_pde, cfg):
    """compute_pde_and_reg_loss for D(x, y, u). The x-derivative of log D is the total one,
    m_x + g'(u) u_x, so the residual is still u_t - (D_x u_x + D_y u_y + D Lap u). The 90/10 split is
    kept: the coefficient branch sees u (and its derivatives) frozen, the u branch sees D frozen."""
    x = X_pde[:, 0:1].requires_grad_(True)
    y = X_pde[:, 1:2].requires_grad_(True)
    t = X_pde[:, 2:3].requires_grad_(True)
    xyt = torch.cat([x, y, t], dim=1)
    xy = torch.cat([x, y], dim=1)
    u = model.forward_u(xyt)
    u_x, u_y, u_t = torch.autograd.grad(u, [x, y, t], torch.ones_like(u), create_graph=True)
    u_xx = P.d_wrt(u_x, x)
    u_yy = P.d_wrt(u_y, y)

    m_s = model.forward_m(xy)
    ms_x, ms_y = torch.autograd.grad(m_s, [x, y], torch.ones_like(m_s), create_graph=True)
    u_d = u.detach().requires_grad_(True)
    g = model.g(u_d)
    g_u = torch.autograd.grad(g, u_d, torch.ones_like(g), create_graph=True)[0]
    m_raw = m_s + g
    inside = ((m_raw > -5.0) & (m_raw < 2.0)).float()
    m = torch.clamp(m_raw, min=-5.0, max=2.0)
    D = torch.exp(m)
    D_x = D * inside * (ms_x + g_u * u_x.detach())
    D_y = D * inside * (ms_y + g_u * u_y.detach())

    res_for_m = u_t.detach() - (D_x * u_x.detach() + D_y * u_y.detach() + D * (u_xx.detach() + u_yy.detach()))
    res_for_u = u_t - (D_x.detach() * u_x + D_y.detach() * u_y + D.detach() * (u_xx + u_yy))
    loss_pde_raw = torch.mean(res_for_m.detach() ** 2)
    t_norm = t / model.T_max
    time_scale = torch.clamp((0.02 + t_norm) / 0.12, min=0.25, max=8.0).detach()
    loss_pde_opt = 0.90 * torch.mean((time_scale * res_for_m) ** 2) + 0.10 * torch.mean((time_scale * res_for_u) ** 2)

    grad_m_sq = ms_x ** 2 + ms_y ** 2                    # same regulariser as the plain PINN, on m(x, y)
    delta_m_sq = (m_s - model.m_bg) ** 2
    if cfg.reg_type == "tikhonov":
        loss_reg = torch.mean(grad_m_sq + 0.05 * delta_m_sq)
    elif cfg.reg_type == "tv":
        loss_reg = torch.mean(torch.sqrt(grad_m_sq + cfg.tv_eps) + 0.02 * delta_m_sq)
    else:
        loss_reg = torch.zeros((), device=X_pde.device)
    return loss_pde_raw, loss_pde_opt, loss_reg


def pde_loss_split(model, X_pde, cfg, split):
    """P.compute_pde_and_reg_loss with the 0.90 / 0.10 split between the D-step and the u-step changed."""
    x = X_pde[:, 0:1].requires_grad_(True)
    y = X_pde[:, 1:2].requires_grad_(True)
    t = X_pde[:, 2:3].requires_grad_(True)
    u = model.forward_u(torch.cat([x, y, t], 1))
    m = torch.clamp(model.forward_m(torch.cat([x, y], 1)), min=-5.0, max=2.0)
    D = torch.exp(m)
    u_x, u_y, u_t = torch.autograd.grad(u, [x, y, t], torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    m_x, m_y = torch.autograd.grad(m, [x, y], torch.ones_like(m), create_graph=True)
    D_x, D_y = D * m_x, D * m_y
    raw = u_t - (D_x * u_x + D_y * u_y + D * (u_xx + u_yy))
    ts = torch.clamp((0.02 + t / model.T_max) / 0.12, min=0.25, max=8.0).detach()
    res_m = u_t.detach() - (D_x * u_x.detach() + D_y * u_y.detach() + D * (u_xx.detach() + u_yy.detach()))
    res_u = u_t - (D_x.detach() * u_x + D_y.detach() * u_y + D.detach() * (u_xx + u_yy))
    opt = split * torch.mean((ts * res_m) ** 2) + (1 - split) * torch.mean((ts * res_u) ** 2)
    reg = torch.mean(m_x ** 2 + m_y ** 2 + 0.05 * (m - model.m_bg) ** 2)
    return torch.mean(raw ** 2), opt, reg


def oracle_u_D(cfg, taper_k=6.0, steps=1500, seed=42, t_min=0.0, fine_time=True):
    """The PINN's D-step alone, fed the EXACT temperature field: the D-network (same architecture, taper,
    background, regulariser and PDE-loss weighting as the PINN) is fitted to the residual
    u_t - div(D grad u) with u and its derivatives taken from the noise-free simulation (129 x 129, 41 times;
    central differences). No sensors, no noise, no u-network: this isolates the D-step from the u-net."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    bundle = P.load_v2_experiment(cfg)
    Lx, Ly, T_max = bundle["Lx"], bundle["Ly"], bundle["T_max"]
    if fine_time:
        # the stored truth has only 41 snapshots, too coarse for u_t where D is large: re-simulate the true D on
        # the 65 x 65 inversion grid with the project's CN solver (1,600 steps) and keep every 4th state
        import adjoint_inversion as A
        import common as C
        key = cfg.exp_id
        coll = next(c for c, root in C.COLLECTIONS.items() if root in cfg.data_root)
        ds = C.dataset(coll, key)
        sol = A.AdjointSolver(ds, n_steps=1600)
        W = sol.forward(ds.ground_truth()[0].ravel())[0]
        keep = [sol.disc.state_of_step(k) for k in range(0, 1601, 4)]
        U = W[keep].reshape(len(keep), ds.N, ds.N)
    else:
        U = bundle["u_true_full"].astype(np.float64)                 # (41, 129, 129), [t, x, y]
    nt, n = U.shape[0], U.shape[1]
    h, dt = Lx / (n - 1), T_max / (nt - 1)
    ut = (U[2:] - U[:-2]) / (2 * dt)
    Uc = U[1:-1]
    ux = (Uc[:, 2:, 1:-1] - Uc[:, :-2, 1:-1]) / (2 * h)
    uy = (Uc[:, 1:-1, 2:] - Uc[:, 1:-1, :-2]) / (2 * h)
    lap = (Uc[:, 2:, 1:-1] + Uc[:, :-2, 1:-1] + Uc[:, 1:-1, 2:] + Uc[:, 1:-1, :-2] - 4 * Uc[:, 1:-1, 1:-1]) / h ** 2
    ut = ut[:, 1:-1, 1:-1]
    tt = np.arange(1, nt - 1) * dt
    xg = np.linspace(0, Lx, n)[1:-1]
    TT, XX, YY = np.meshgrid(tt, xg, xg, indexing="ij")
    cols = [XX, YY, TT, ut, ux, uy, lap]
    A = np.stack([c.ravel() for c in cols], 1)
    A = torch.tensor(A[A[:, 2] >= t_min * T_max], dtype=torch.float32)      # optionally drop early times
    model = P.InverseDiffusionPINN(cfg, Lx, Ly, T_max, m_init_bias=bundle["m_init_bias"], bc_type=bundle["bc_type"])
    if taper_k != 6.0:
        import types
        k = float(taper_k)

        def forward_m(self, xy):
            xn, yn = xy[:, 0:1] / self.Lx, xy[:, 1:2] / self.Ly
            taper = torch.tanh(k * xn) * torch.tanh(k * (1 - xn)) * torch.tanh(k * yn) * torch.tanh(k * (1 - yn))
            return self.m_bg + taper * self.m_net(xy / self.scale_xy)
        model.forward_m = types.MethodType(forward_m, model)
    params = list(model.m_net.parameters()) + [model.m_bg]
    optim = torch.optim.AdamW(params, lr=cfg.lr_m, weight_decay=1e-5)
    batch = cfg.n_collocation + cfg.batch_size_data
    best, best_state = float("inf"), None
    for step in range(1, steps + 1):
        optim.param_groups[0]["lr"] = 1e-5 + (cfg.lr_m - 1e-5) * 0.5 * (1 + math.cos(math.pi * (step - 1) / steps))
        idx = torch.randint(0, A.shape[0], (batch,))
        b = A[idx]
        xy = b[:, :2].clone().requires_grad_(True)
        m = torch.clamp(model.forward_m(xy), min=-5.0, max=2.0)
        D = torch.exp(m)
        m_x, m_y = torch.autograd.grad(m, [xy], torch.ones_like(m), create_graph=True)[0].split(1, 1)
        res = b[:, 3:4] - (D * m_x * b[:, 4:5] + D * m_y * b[:, 5:6] + D * b[:, 6:7])
        ts = torch.clamp((0.02 + b[:, 2:3] / T_max) / 0.12, min=0.25, max=8.0)
        reg = torch.mean(m_x ** 2 + m_y ** 2 + 0.05 * (m - model.m_bg) ** 2)
        loss = cfg.lambda_pde * 0.90 * torch.mean((ts * res) ** 2) + cfg.lambda_reg * reg
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        if step > int(0.65 * steps) and loss.item() < best:
            best, best_state = loss.item(), copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    Xg, Yg = np.meshgrid(bundle["x_grid"], bundle["y_grid"], indexing="ij")
    with torch.no_grad():
        Dg = model.forward_D(torch.tensor(np.c_[Xg.ravel(), Yg.ravel()], dtype=torch.float32)).numpy()
    return Dg.reshape(Xg.shape)


def residual_abs(model, X, coef):
    """|PDE residual| at X, no graph kept (for adaptive refinement)."""
    X = X.clone()
    x = X[:, 0:1].requires_grad_(True)
    y = X[:, 1:2].requires_grad_(True)
    t = X[:, 2:3].requires_grad_(True)
    xyt = torch.cat([x, y, t], 1)
    u = model.forward_u(xyt)
    u_x, u_y, u_t = torch.autograd.grad(u, [x, y, t], torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), retain_graph=True)[0]
    m = model.forward_m(torch.cat([x, y], 1))
    if coef == "xyT":
        m = m + model.g(u)
    m = torch.clamp(m, -5.0, 2.0)
    m_x, m_y = torch.autograd.grad(m, [x, y], torch.ones_like(m), retain_graph=True)
    D = torch.exp(m)
    r = u_t - (D * m_x * u_x + D * m_y * u_y + D * (u_xx + u_yy))
    return r.detach().abs().squeeze(-1)


def augment_readings(X, u, sigma, factor, sigma_factor, Lx, Ly, seed):
    """Pseudo-readings: at each reading time, a smoothed thin-plate RBF through that time's readings,
    evaluated at random points inside the sensors' bounding box. They get sigma_factor x sigma."""
    from scipy.interpolate import RBFInterpolator
    rng = np.random.default_rng(seed)
    Xn, un, sn = [], [], []
    for tv in np.unique(X[:, 2]):
        rows = np.abs(X[:, 2] - tv) < 1e-9
        xy, vals = X[rows, :2], u[rows, 0]
        if len(xy) < 4:
            continue
        lo, hi = xy.min(0), xy.max(0)
        n_new = int(round(factor * len(xy)))
        pts = lo + (hi - lo) * rng.random((n_new, 2))
        f = RBFInterpolator(xy.astype(float), vals.astype(float), kernel="thin_plate_spline",
                            smoothing=float(len(xy) * np.mean(sigma[rows]) ** 2))
        Xn.append(np.c_[pts, np.full(n_new, tv)])
        un.append(f(pts)[:, None])
        sn.append(np.full((n_new, 1), sigma_factor * float(np.mean(sigma[rows]))))
    return (np.vstack(Xn).astype(np.float32), np.vstack(un).astype(np.float32), np.vstack(sn).astype(np.float32))


# ---------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------
def make_cfg(collection, key, epochs=1500, seed=42, out_dir=None, **overrides):
    cfg = P.PINNConfig()
    cfg.data_root = str(EXT / COLLECTIONS[collection])
    cfg.output_dir = str(out_dir or (EXT / "novelty" / "_scratch"))
    cfg.exp_id = key
    cfg.epochs = int(epochs)
    cfg.seed = int(seed)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def train(cfg, opt: Optional[PlusOptions] = None, verbose=True):
    """Train one PINN. Returns dict(D_hat (65x65; for xyT the reference field D(x, y, u=0)), g_curve,
    history, wall_s, val_rows, u_net_val_chi2, model, bundle)."""
    opt = opt or PlusOptions()
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = P.load_v2_experiment(cfg)
    Lx, Ly, T_max = bundle["Lx"], bundle["Ly"], bundle["T_max"]
    bc_type = bundle["bc_type"]
    x_grid, y_grid = bundle["x_grid"], bundle["y_grid"]

    X_all_np = bundle["X_data"].numpy()
    n_rows = len(X_all_np)
    train_rows = np.arange(n_rows) if opt.train_rows is None else np.asarray(opt.train_rows)
    val_rows = np.setdiff1d(np.arange(n_rows), train_rows)
    X_data_all = bundle["X_data"][train_rows].to(device)
    u_data_all = bundle["u_data"][train_rows].to(device)
    if opt.augment > 0:
        Xa, ua, _ = augment_readings(X_all_np[train_rows], bundle["u_data"].numpy()[train_rows],
                                     bundle["sigma_data"].numpy()[train_rows], opt.augment,
                                     opt.augment_sigma_factor, Lx, Ly, cfg.seed)
        X_data_all = torch.cat([X_data_all, torch.tensor(Xa, device=device)], 0)
        u_data_all = torch.cat([u_data_all, torch.tensor(ua, device=device)], 0)
    X_ic_all = bundle["X_ic"].to(device)
    u_ic_all = bundle["u_ic"].to(device)

    if opt.coef == "xyT":
        model = TempPINN(cfg, Lx, Ly, T_max, bundle["m_init_bias"], bc_type, opt.g_hidden, opt.g_layers).to(device)
    else:
        model = P.InverseDiffusionPINN(cfg, Lx, Ly, T_max, m_init_bias=bundle["m_init_bias"], bc_type=bc_type).to(device)
    if opt.taper_k != 6.0:                          # same D parameterisation, only the wall taper's steepness changes
        import types
        k = float(opt.taper_k)

        def forward_m(self, xy):
            xn, yn = xy[:, 0:1] / self.Lx, xy[:, 1:2] / self.Ly
            taper = torch.tanh(k * xn) * torch.tanh(k * (1 - xn)) * torch.tanh(k * yn) * torch.tanh(k * (1 - yn))
            return self.m_bg + taper * self.m_net(xy / self.scale_xy)
        model.forward_m = types.MethodType(forward_m, model)

    if opt.init_D is not None:                      # same procedure as --warmstart-from-adjoint
        XX, YY = np.meshgrid(x_grid, y_grid, indexing="ij")
        xy_adj = torch.tensor(np.stack([XX.ravel(), YY.ravel()], -1), dtype=torch.float32, device=device)
        m_adj = torch.tensor(np.log(np.clip(np.asarray(opt.init_D).ravel(), 1e-4, None))[:, None],
                             dtype=torch.float32, device=device)
        ws_opt = torch.optim.Adam(list(model.m_net.parameters()) + [model.m_bg], lr=2e-3)
        for _ in range(300):
            ws_opt.zero_grad()
            ws_loss = torch.mean((model.forward_m(xy_adj) - m_adj) ** 2)
            ws_loss.backward()
            ws_opt.step()

    groups = [{"params": list(model.u_net.parameters()) + [model.u_base_log_d], "lr": cfg.lr_u},
              {"params": list(model.m_net.parameters()) + [model.m_bg], "lr": cfg.lr_m}]
    if opt.coef == "xyT":
        groups[1]["params"] += list(model.g_net.parameters())
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
    mse = nn.MSELoss()
    u_data_msq = float(torch.mean(u_data_all ** 2).item()) + 1e-6
    u_ic_msq = float(torch.mean(u_ic_all ** 2).item()) + 1e-6
    sigma_sq = float(bundle.get("sigma_sq", 2.36e-6))
    lam = dict(data=cfg.lambda_data, pde=cfg.lambda_pde, bc=cfg.lambda_bc, ic=cfg.lambda_ic, reg=cfg.lambda_reg)

    history, snapshots = [], []
    best_total, best_state = float("inf"), None
    warmup_m_epochs = int(0.15 * cfg.epochs)
    rar_pts = torch.zeros((0, 3), device=device)
    rar_gen = torch.Generator(device=device).manual_seed(cfg.seed + 7) if opt.rar else None
    XX, YY = np.meshgrid(x_grid, y_grid, indexing="ij")
    xy_eval = torch.tensor(np.stack([XX.ravel(), YY.ravel()], -1), dtype=torch.float32, device=device)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(1, cfg.epochs)))
        m_lr_scale = min(1.0, max(0.02, (epoch - 50) / max(1, warmup_m_epochs)))
        optimizer.param_groups[0]["lr"] = 1e-5 + (cfg.lr_u - 1e-5) * cos_decay
        optimizer.param_groups[1]["lr"] = (1e-5 + (cfg.lr_m - 1e-5) * cos_decay) * m_lr_scale

        n_obs = X_data_all.shape[0]
        if n_obs > cfg.batch_size_data:
            idx_d = torch.randint(0, n_obs, (cfg.batch_size_data,), device=device)
            Xd, ud = X_data_all[idx_d], u_data_all[idx_d]
        else:
            Xd, ud = X_data_all, u_data_all
        loss_data = mse(model.forward_u(Xd), ud)

        n_ic_tot = X_ic_all.shape[0]
        if n_ic_tot > cfg.n_ic:
            idx_ic = torch.randint(0, n_ic_tot, (cfg.n_ic,), device=device)
            Xic, uic = X_ic_all[idx_ic], u_ic_all[idx_ic]
        else:
            Xic, uic = X_ic_all, u_ic_all
        loss_ic = mse(model.forward_u(Xic), uic)
        loss_bc = P.compute_bc_loss(model, Lx, Ly, T_max, cfg.n_bc, bc_type, device)

        n_adapt = min(len(rar_pts), opt.rar_max)
        n_random = cfg.n_collocation - n_adapt          # same total number of physics points as the plain PINN
        n_focus = int(0.60 * n_random)
        n_rest = n_random - n_focus
        x_focus = 0.05 * Lx + 0.90 * Lx * torch.rand(n_focus, 1, device=device)
        y_focus = 0.05 * Ly + 0.90 * Ly * torch.rand(n_focus, 1, device=device)
        t_focus = T_max * (0.02 + 0.65 * (torch.rand(n_focus, 1, device=device) ** 1.4))
        x_rest = Lx * torch.rand(n_rest, 1, device=device)
        y_rest = Ly * torch.rand(n_rest, 1, device=device)
        t_rest = T_max * torch.rand(n_rest, 1, device=device)
        parts = [torch.cat([x_focus, y_focus, t_focus], 1), torch.cat([x_rest, y_rest, t_rest], 1)]
        if n_adapt:
            parts.append(rar_pts[-n_adapt:])
        parts.append(Xd.detach())
        X_pde = torch.cat(parts, 0)
        if opt.coef == "xyT":
            loss_pde_raw, loss_pde_opt, loss_reg = pde_loss_xyT(model, X_pde, cfg)
        elif opt.pde_split != 0.90:
            loss_pde_raw, loss_pde_opt, loss_reg = pde_loss_split(model, X_pde, cfg, opt.pde_split)
        else:
            loss_pde_raw, loss_pde_opt, loss_reg = P.compute_pde_and_reg_loss(model, X_pde, cfg)

        if cfg.morozov_damping:
            data_term = (torch.relu(loss_data - 0.95 * sigma_sq) + 0.05 * loss_data) / u_data_msq
        else:
            data_term = loss_data / u_data_msq
        loss_total = (lam["data"] * data_term + lam["pde"] * loss_pde_opt + lam["bc"] * loss_bc
                      + lam["ic"] * (loss_ic / u_ic_msq) + lam["reg"] * loss_reg)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        row = dict(epoch=epoch, loss_total=float(loss_total.item()), loss_data=float(loss_data.item()),
                   loss_pde=float(loss_pde_raw.item()), loss_bc=float(loss_bc.item()),
                   loss_ic=float(loss_ic.item()), loss_reg=float(loss_reg.item()))
        history.append(row)
        if epoch > int(0.65 * cfg.epochs) and row["loss_total"] < best_total:
            best_total = row["loss_total"]
            best_state = copy.deepcopy(model.state_dict())

        if opt.rar and epoch >= int(opt.rar_start * cfg.epochs) and epoch % opt.rar_every == 0:
            pool = torch.rand(opt.rar_pool, 3, device=device, generator=rar_gen) * torch.tensor([Lx, Ly, T_max], device=device)
            r = residual_abs(model, pool, opt.coef)
            rar_pts = torch.cat([rar_pts, pool[torch.topk(r, opt.rar_add).indices].detach()], 0)
        if opt.record_every and epoch % opt.record_every == 0:
            with torch.no_grad():
                snapshots.append((epoch, model.forward_D(xy_eval).cpu().numpy().reshape(XX.shape)))
        if verbose and (epoch == 1 or epoch % max(1, cfg.epochs // 5) == 0):
            print(f"    epoch {epoch:5d}/{cfg.epochs}  L={row['loss_total']:.3e}  L_data={row['loss_data']:.3e}"
                  f"  L_pde={row['loss_pde']:.3e}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        D_hat = model.forward_D(xy_eval).cpu().numpy().reshape(XX.shape)   # for xyT: D(x, y, u = 0)
        g_curve = None
        if opt.coef == "xyT":
            uu = torch.linspace(0, 1, 101, device=device)[:, None]
            g_curve = np.c_[uu.cpu().numpy()[:, 0], model.g(uu).cpu().numpy()[:, 0]]
        u_net_val_chi2 = None
        if len(val_rows):
            Xv = bundle["X_data"][val_rows].to(device)
            uv = bundle["u_data"][val_rows].to(device).squeeze(-1)
            sv = bundle["sigma_data"][val_rows].to(device).squeeze(-1)
            u_net_val_chi2 = float(torch.mean(((model.forward_u(Xv).squeeze(-1) - uv) / sv) ** 2).item())
    return dict(D_hat=D_hat, g_curve=g_curve, history=history, snapshots=snapshots, wall_s=time.time() - t0,
                val_rows=val_rows, train_rows=train_rows, u_net_val_chi2=u_net_val_chi2, n_rar=int(len(rar_pts)),
                n_data_rows=int(X_data_all.shape[0]), model=model, bundle=bundle, device=str(device))
