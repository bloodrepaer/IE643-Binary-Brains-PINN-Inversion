"""Compare grid-adjoint, neural-adjoint and PINN on identical known-physics data.

Example: python compare_neural.py --methods adjoint_nn pinn --adam-steps 3000
Run --help for budgets. Short runs verify execution, not method superiority.
"""
import argparse
import os
# Avoid loading a second Intel OpenMP runtime through conda NumPy/MKL when
# PyTorch already supplies one. Do not enable KMP_DUPLICATE_LIB_OK.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import copy
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import adjoint_inversion as ai
from adjoint_nn import NeuralAdjointSolver, grid_coordinates
from neural_diffusivity import CoefficientConfig, NeuralDiffusivity


def derivative(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


class TemperatureNetwork(nn.Module):
    """Physical temperature; inputs x,y,t/T. IC and BC are soft constraints."""
    def __init__(self, width=64, depth=4, seed=42):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            layers, n = [], 3
            for _ in range(depth):
                lin = nn.Linear(n, width, dtype=torch.float64)
                nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
                layers.extend([lin, nn.Tanh()]); n = width
            layers.append(nn.Linear(n, 1, dtype=torch.float64))
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(2*x-1).ravel()


def residual(temperature, coefficient, x, T):
    u = temperature(x)
    du = derivative(u, x)
    D = coefficient.diffusivity(x[:, :2]).ravel()
    div = derivative(D * du[:, 0], x)[:, 0] + derivative(D * du[:, 1], x)[:, 1]
    return du[:, 2] - T * div


def field_regularizer(m, solver):
    """Same discrete R(log D), including boundary links, as the adjoint."""
    dz = solver.disc
    if solver.reg == "tikhonov":
        return .5 * ((m[dz.p_all] - m[dz.q_all])**2).sum()
    M, h = m.reshape(dz.N, dz.N), dz.h
    gx, gy = (M[1:, :-1]-M[:-1, :-1])/h, (M[:-1, 1:]-M[:-1, :-1])/h
    return h*h * torch.sqrt(gx*gx+gy*gy+solver.tv_eps).sum()


def run_pinn(ds, solver, coefficient, args, alpha, D_const):
    start = time.perf_counter()
    requested = getattr(args, "device", "cpu")
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; enable the Kaggle GPU accelerator")
    coefficient = coefficient.to(device)
    temperature = TemperatureNetwork(seed=args.seed).to(device)
    params = list(temperature.parameters()) + list(coefficient.parameters())
    rng = np.random.default_rng(args.seed)
    tensor = lambda a: torch.tensor(a, dtype=torch.float64, device=device)
    xd = tensor(np.c_[ds.x, ds.y, ds.t/ds.T]); yd = tensor(ds.u)
    xy = grid_coordinates(ds.N).to(device)
    xi = torch.cat([xy, torch.zeros((len(xy), 1), dtype=torch.float64, device=device)], dim=1)
    yi = tensor(ds.u0.ravel())
    scale = max(float(np.max(np.abs(ds.u0))), float(np.max(np.abs(ds.u))), 1e-12)
    n = len(ds.u)

    def batch():
        # Half early-time log-spaced samples resolve the sharp IC transient.
        col = rng.uniform(0, 1, (args.n_collocation, 3))
        col[:len(col)//2, 2] = 10 ** rng.uniform(-4, 0, len(col)//2)
        wall = rng.uniform(0, 1, (args.n_boundary, 3))
        edges = rng.integers(0, 4, len(wall))
        target = np.zeros(len(wall))
        for i, edge in enumerate(("left", "right", "bottom", "top")):
            sel = edges == i
            wall[sel, 0 if i < 2 else 1] = i % 2
            target[sel] = [ds.bc.at(t*ds.T)[edge] for t in wall[sel, 2]]
        limit = getattr(args, "data_batch", 4096)
        indices = rng.choice(len(ds.u), min(limit, len(ds.u)), replace=False)
        return tensor(col), tensor(wall), tensor(target), torch.tensor(indices, dtype=torch.long, device=device)

    def losses(points):
        col, xb, yb, indices = points
        col = col.detach().requires_grad_(True)
        # Unbiased estimator of the summed data misfit, with a fixed sample
        # during L-BFGS line searches. Avoid full-field GPU memory blowups.
        data = .5 * n * ((temperature(xd[indices])-yd[indices])/solver.sigma_w).square().mean()
        pde = .5 * n * (residual(temperature, coefficient, col, ds.T)/scale).square().mean()
        ic = .5 * n * ((temperature(xi)-yi)/scale).square().mean()
        bc = .5 * n * ((temperature(xb)-yb)/scale).square().mean()
        reg = field_regularizer(coefficient(xy).ravel(), solver)
        total = data + args.physics_weight*pde + args.condition_weight*(ic+bc) + alpha*reg
        return total, dict(data=float(data.detach()), pde=float(pde.detach()),
                           ic=float(ic.detach()), bc=float(bc.detach()), reg=float(reg.detach()))

    history = []
    opt = torch.optim.Adam(params, lr=args.lr)
    for step in range(1, args.adam_steps+1):
        opt.zero_grad(); loss, parts = losses(batch()); loss.backward(); opt.step()
        if step == 1 or step % max(1, args.adam_steps//10) == 0:
            history.append(dict(stage="adam", step=step, total=float(loss.detach()), **parts))
            print(f"  PINN step={step} loss={loss.item():.4g} data={parts['data']:.4g}", flush=True)
    if args.lbfgs_steps:
        points = batch()  # fixed throughout the line search
        opt = torch.optim.LBFGS(params, max_iter=args.lbfgs_steps, history_size=30,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-8,
                                tolerance_change=1e-12)
        def closure():
            opt.zero_grad(); loss, _ = losses(points); loss.backward(); return loss
        opt.step(closure)
        loss, parts = losses(points)
        history.append(dict(stage="lbfgs", total=float(loss.detach()), **parts))
    with torch.no_grad():
        D = coefficient.diffusivity(xy).cpu().numpy().ravel()
        fit = torch.cat([temperature(part).cpu() for part in xd.split(getattr(args, "data_batch", 4096))]).numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_s = time.perf_counter()-start
    W = solver.forward(D)[0]
    rms = float(np.sqrt(np.mean((solver.predict(W)-ds.u)**2)))
    out = dict(D=D, m=np.log(D), W=W, alpha=alpha, flag="fixed_alpha_budget",
               D_const=D_const, rms_const=solver._constant_rms, n_evals=1,
               wall_s=training_s, n_steps=solver.disc.n_steps, rms=rms,
               history=history, path=[dict(alpha=alpha, rms=rms, rms_over_sigma=rms/solver.sigma_w)])
    checkpoint = dict(coefficient=coefficient.checkpoint(), temperature_state_dict=temperature.state_dict(),
                      temperature_config=dict(width=64, depth=4, seed=args.seed), T=ds.T)
    extra = dict(rms_unet_over_sigma=float(np.sqrt(np.mean((fit-ds.u)**2))/solver.sigma_w),
                 loss_history=history, physics_weight=args.physics_weight,
                 condition_weight=args.condition_weight, alpha_selection="fixed_cli",
                 device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                 adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps)
    return out, checkpoint, extra


def validate_problem(root, entry, physics_mode="matched"):
    with open(Path(root)/entry["files"]["problem"]) as f:
        spec = json.load(f)
    bc = spec["boundary_condition"]
    unknown_bc = bc.get("known", True) is False or entry.get("bc_known", True) is False
    unknown_source = spec.get("source_known", True) is False or entry.get("source_known", True) is False
    if unknown_bc and physics_mode == "matched":
        raise ValueError("Matched comparison requires known BC; unknown-BC inference needs its own model")
    if bc["type"] not in (("dirichlet", "unknown") if physics_mode == "assumed_zero" else ("dirichlet",)):
        raise ValueError("Unsupported boundary type")
    if str(spec.get("source_f", "0")) not in ("0", "0.0") and not (unknown_source and physics_mode == "assumed_zero"):
        raise ValueError("This comparison supports known Dirichlet BC and zero source only")
    if unknown_source and physics_mode == "matched":
        raise ValueError("Hidden-source cases must be compared separately")
    if any(float(spec["domain"][v]) != 1.0 for v in ("Lx", "Ly")):
        raise ValueError("Existing adjoint assumes the unit square")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="datasets_v2_rk4_zero")
    ap.add_argument("--keys", nargs="+", default=[ai.BASELINE_KEYS[0]])
    ap.add_argument("--out", default="results_neural")
    ap.add_argument("--methods", nargs="+", choices=["grid", "adjoint_nn", "pinn"], default=["grid", "adjoint_nn", "pinn"])
    ap.add_argument("--width", type=int, default=32); ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu", help="PINN training device; sparse adjoint and replay remain on CPU")
    ap.add_argument("--physics-mode", choices=["matched", "assumed_zero"], default="matched", help="Explicit diagnostic mode for hidden physics; both methods impose the supplied zero assumptions")
    ap.add_argument("--data-batch", type=int, default=4096, help="PINN sensor minibatch; loss remains an estimator of the sum")
    ap.add_argument("--control", type=int, default=16); ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--reg", choices=["tikhonov", "tv"], default="tikhonov")
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--alphas", nargs="+", type=float, default=None, help="Absolute alpha values; default discrepancy path")
    ap.add_argument("--pinn-alpha", type=float, default=1., help="Fixed alpha; does not use ground truth or adjoint result")
    ap.add_argument("--adam-steps", type=int, default=3000); ap.add_argument("--lbfgs-steps", type=int, default=300)
    ap.add_argument("--n-collocation", type=int, default=1024); ap.add_argument("--n-boundary", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--physics-weight", type=float, default=100.)
    ap.add_argument("--condition-weight", type=float, default=100.)
    args = ap.parse_args()
    if min(args.threads, args.n_collocation, args.n_boundary, args.maxiter, args.data_batch) < 1 or min(args.adam_steps, args.lbfgs_steps, args.pinn_alpha) < 0:
        ap.error("Invalid negative budget/alpha or nonpositive sample count")
    if args.alphas is not None and any(a < 0 for a in args.alphas):
        ap.error("Alphas must be nonnegative")
    torch.set_num_threads(args.threads)
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    (root/"run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    manifest = {e["key"]: e for e in json.loads((Path(args.data)/"manifest.json").read_text())}
    missing = set(args.keys)-manifest.keys()
    if missing:
        ap.error(f"Unknown keys: {sorted(missing)}")
    rows = []
    for key in args.keys:
        entry = manifest[key]; validate_problem(args.data, entry, args.physics_mode)
        ds = ai.Dataset(args.data, entry)
        base = ai.AdjointSolver(ds, n_steps=args.n_steps, control=args.control, reg=args.reg)
        tic = time.perf_counter(); m0 = base.best_constant()
        base.objective(np.full(args.control**2, m0), 0, False)
        base._constant_rms = float(base._last["rms"])
        setup_s = time.perf_counter()-tic
        shared = NeuralDiffusivity(CoefficientConfig(args.width, args.depth), np.exp(m0), args.seed)
        for method in args.methods:
            print(f"\n{key}: {method}", flush=True)
            coefficient = copy.deepcopy(shared)
            solver = NeuralAdjointSolver(ds, coefficient, n_steps=base.disc.n_steps, reg=args.reg) if method == "adjoint_nn" else base
            extra, checkpoint = {}, None
            if method == "pinn":
                out, checkpoint, extra = run_pinn(ds, base, coefficient, args, args.pinn_alpha, float(np.exp(m0)))
            else:
                out = solver.invert(alphas=args.alphas, maxiter=args.maxiter)
                if method == "adjoint_nn":
                    checkpoint = {"coefficient": coefficient.checkpoint()}
                extra["alpha_selection"] = "fixed_cli" if args.alphas is not None and len(args.alphas) == 1 else "discrepancy_path"
            # Ground truth first opened here, after fitting and model selection.
            metrics, S, truth = ai.evaluate(ds, solver, out)
            metrics.update(extra, method=method, seed=args.seed, shared_setup_s=setup_s,
                           physics_mode=args.physics_mode,
                           alpha_path=out["path"],
                           parameter_count=sum(p.numel() for p in coefficient.parameters()) if method != "grid" else args.control**2,
                           dtype="float64", coefficient_config=vars(shared.config) if method != "grid" else None)
            metrics["chi2_red_solver"] = metrics["rms_replay_over_sigma"]**2 if ds.sigma > 0 else None
            if ds.sigma == 0:
                metrics["rms_replay_over_sigma"] = None
            metrics["rms_replay"] = float(out["rms"])
            metrics["rms_replay_relative_signal"] = float(out["rms"]/max(np.sqrt(np.mean(ds.u**2)), 1e-12))
            dest = root/key/f"{method}__s{args.seed}"; dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest/"D_hat.npz", D_hat=out["D"].reshape(ds.N, ds.N),
                                m_hat=out["m"].reshape(ds.N, ds.N), x=np.linspace(0, 1, ds.N), y=np.linspace(0, 1, ds.N))
            if checkpoint is not None:
                torch.save(dict(**checkpoint, key=key, method=method, seed=args.seed), dest/"model.pt")
            (dest/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            ai.plot(ds, solver, out, metrics | {"rms_replay_over_sigma": out["rms"]/solver.sigma_w}, S, truth, dest/"summary.png")
            rows.append({k: v for k, v in metrics.items() if not isinstance(v, (dict, list))})
            columns = sorted(set().union(*(r.keys() for r in rows)))
            with (root/"summary.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=columns); w.writeheader(); w.writerows(rows)
            print(f"  relL2(D)={metrics.get('rel_L2_D', float('nan')):.4f}, replay RMS={out['rms']:.5g}", flush=True)
    (root/"run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
