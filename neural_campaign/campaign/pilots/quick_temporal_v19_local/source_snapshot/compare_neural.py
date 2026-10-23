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
import hashlib
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
    """Physical temperature on x,y,t/T; optional supplied-IC base, soft BC."""
    def __init__(self, width=64, depth=4, seed=42, initial_gaussians=None, T=1., D_base=.25, boundary_model="soft", local_coefficient=False, image_D_max=10.):
        super().__init__()
        self.T, self.D_base = float(T), float(D_base)
        self.has_ic_base = initial_gaussians is not None
        self.max_initial_variance=max(max(g[3],g[4]) for g in initial_gaussians) if self.has_ic_base else 0.
        self.local_coefficient=local_coefficient
        if local_coefficient and (not self.has_ic_base or boundary_model!='zero_images'):
            raise ValueError('Local-coefficient base requires Gaussian IC and zero image boundaries')
        if boundary_model not in ("soft","zero_images"):raise ValueError("Unsupported boundary model")
        if boundary_model=="zero_images" and not self.has_ic_base:raise ValueError("Image base requires Gaussian IC")
        self.boundary_model=boundary_model
        if self.has_ic_base:
            self.register_buffer("initial_gaussians",torch.tensor(initial_gaussians,dtype=torch.float64))
            if boundary_model=="zero_images":
                # Each image solves the heat equation. Bound neglected tails by
                # exp(-40) or smaller over the supplied horizon and unit square.
                largest_variance=max(max(g[3],g[4]) for g in initial_gaussians)+2*(image_D_max if local_coefficient else self.D_base)*self.T
                count=max(3,int(np.ceil(1+np.sqrt(20*largest_variance))))
                self.register_buffer("image_shifts",2*torch.arange(-count,count+1,dtype=torch.float64))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            layers, n = [], 3
            for _ in range(depth):
                lin = nn.Linear(n, width, dtype=torch.float64)
                nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
                layers.extend([lin, nn.Tanh()]); n = width
            layers.append(nn.Linear(n, 1, dtype=torch.float64))
            if self.has_ic_base:
                nn.init.zeros_(layers[-1].weight);nn.init.zeros_(layers[-1].bias)
            self.net = nn.Sequential(*layers)

    def forward(self, x, coefficient=None):
        correction=self.net(2*x-1).ravel()
        if not self.has_ic_base:return correction
        # Known Gaussian IC evolved with the data-fitted constant diffusivity.
        # This is an infinite-domain base, not a claim that it solves the plate
        # boundaries: BC and variable-D corrections are still learned.
        t=x[:,2]*self.T
        local_D=self.D_base
        shifts=getattr(self,'image_shifts',None)
        if self.local_coefficient:
            if coefficient is None:raise ValueError('This state requires its coefficient network')
            local_D=coefficient.diffusivity(x[:,:2]).ravel()
            # This local-constant heat kernel is an ansatz, not an exact
            # variable-D solution. The PDE residual differentiates through D.
            largest=self.max_initial_variance+2*float((local_D*t).detach().max())
            needed=max(3,int(np.ceil(1+np.sqrt(20*largest))))
            middle=len(self.image_shifts)//2
            if needed>middle:raise ValueError('State evaluation exceeds configured diffusion/time image bound')
            shifts=self.image_shifts[middle-needed:middle+needed+1]
        base=torch.zeros_like(t)
        for amp,cx,cy,sx2,sy2 in self.initial_gaussians:
            vx=sx2+2*local_D*t;vy=sy2+2*local_D*t
            if self.boundary_model=="zero_images":
                def image_sum(position,center,variance):
                    a=position[:,None]-center-shifts
                    b=position[:,None]+center-shifts
                    return (torch.exp(-a.square()/(2*variance[:,None]))-
                            torch.exp(-b.square()/(2*variance[:,None]))).sum(1)
                base=base+amp*torch.sqrt(sx2*sy2/(vx*vy))*image_sum(x[:,0],cx,vx)*image_sum(x[:,1],cy,vy)
            else:
                base=base+amp*torch.sqrt(sx2*sy2/(vx*vy))*torch.exp(
                    -(x[:,0]-cx).square()/(2*vx)-(x[:,1]-cy).square()/(2*vy))
        if self.boundary_model=="zero_images":
            correction=16*x[:,0]*(1-x[:,0])*x[:,1]*(1-x[:,1])*correction
        return base+torch.tanh(8*x[:,2])*correction


def gaussian_initial_components(expression):
    """Parse only positive axis-aligned Gaussian sums from the supplied IC."""
    import sympy as sp
    x,y=sp.symbols("x y")
    expr=sp.sympify(expression,locals={"x":x,"y":y})
    components=[]
    for term in sp.Add.make_args(expr):
        atoms=list(term.atoms(sp.exp))
        if len(atoms)!=1:raise ValueError("gaussian_ic requires a sum of Gaussian initial peaks")
        exponential=atoms[0];power=exponential.args[0]
        ix=-sp.diff(power,x,2);iy=-sp.diff(power,y,2)
        if ix.free_symbols or iy.free_symbols or sp.simplify(sp.diff(power,x,y))!=0:
            raise ValueError("Only axis-aligned quadratic Gaussian exponents are supported")
        ix,iy=float(ix),float(iy)
        if min(ix,iy)<=0:raise ValueError("Initial Gaussian variances must be positive")
        cx=float(sp.diff(power,x).subs({x:0,y:0}))/ix
        cy=float(sp.diff(power,y).subs({x:0,y:0}))/iy
        coefficient=sp.simplify(term/exponential)
        if coefficient.free_symbols:raise ValueError("Initial Gaussian amplitudes must be constant")
        amp=float(coefficient)*float(sp.exp(power.subs({x:cx,y:cy})))
        if amp<=0:raise ValueError("Initial Gaussian amplitudes must be positive")
        components.append([amp,cx,cy,1/ix,1/iy])
    # Verify that the parsed expression is the supplied function, rather than
    # silently interpreting a higher-order exponent as a Gaussian.
    xy=np.random.default_rng(643).uniform(0,1,(100,2))
    original=np.asarray(sp.lambdify((x,y),expr,"numpy")(xy[:,0],xy[:,1]),float)
    rebuilt=sum(a*np.exp(-(xy[:,0]-cx)**2/(2*sx2)-(xy[:,1]-cy)**2/(2*sy2)) for a,cx,cy,sx2,sy2 in components)
    if not np.allclose(original,rebuilt,rtol=1e-10,atol=1e-12):
        raise ValueError("Parsed Gaussian base does not match the supplied initial expression")
    return components


def temperature_value(temperature,coefficient,x):
    return temperature(x,coefficient=coefficient) if getattr(temperature,'local_coefficient',False) else temperature(x)


def residual(temperature, coefficient, x, T):
    u = temperature_value(temperature,coefficient,x)
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
    temperature_config=dict(width=64,depth=4,seed=args.seed)
    if getattr(args,"state_model","plain") in ("gaussian_ic","gaussian_ic_bc","gaussian_ic_bc_locald"):
        expression=getattr(ds,"initial_expression",None)
        if expression is None:raise ValueError("Supplied IC expression is needed for gaussian_ic")
        temperature_config.update(initial_gaussians=gaussian_initial_components(expression),T=ds.T,D_base=D_const)
        if args.state_model in ("gaussian_ic_bc","gaussian_ic_bc_locald"):
            import sympy as sp
            spec=json.loads((Path(ds.root)/ds.entry['files']['problem']).read_text())
            bc=spec['boundary_condition']
            if bc.get('known',True) is False or any(sp.sympify(v).equals(0) is not True for v in bc['edges'].values()):
                raise ValueError('gaussian_ic_bc requires supplied known zero Dirichlet boundaries')
            temperature_config['boundary_model']='zero_images'
            if args.state_model=='gaussian_ic_bc_locald':temperature_config.update(local_coefficient=True,image_D_max=coefficient.config.d_max)
    temperature = TemperatureNetwork(**temperature_config).to(device)
    warm_start=getattr(args,'warm_start',None)
    if warm_start:
        saved=torch.load(warm_start,map_location=device,weights_only=True)
        if saved.get('key')!=ds.key or saved.get('method')!='pinn':raise ValueError('Warm start requires this identical PINN observation case')
        restored=NeuralDiffusivity.from_checkpoint(saved['coefficient'])
        if restored.config!=coefficient.config:raise ValueError('Warm-start coefficient architecture/bounds differ')
        previous=saved['temperature_config']
        if set(previous)!=set(temperature_config):raise ValueError('Warm-start state representation differs')
        for name,value in temperature_config.items():
            equal=previous[name]==value if isinstance(value,str) else np.allclose(previous[name],value,rtol=1e-10,atol=1e-12)
            if not equal:raise ValueError('Warm-start supplied state configuration differs: '+name)
        coefficient.load_state_dict(restored.state_dict())
        temperature_config=previous
        temperature=TemperatureNetwork(**temperature_config).to(device)
        temperature.load_state_dict(saved['temperature_state_dict'])
    params = list(temperature.parameters()) + list(coefficient.parameters())
    state_value=lambda points:temperature_value(temperature,coefficient,points)
    rng = np.random.default_rng(args.seed)
    tensor = lambda a: torch.tensor(a, dtype=torch.float64, device=device)
    xd = tensor(np.c_[ds.x, ds.y, ds.t/ds.T]); yd = tensor(ds.u)
    xy = grid_coordinates(ds.N).to(device)
    xi = torch.cat([xy, torch.zeros((len(xy), 1), dtype=torch.float64, device=device)], dim=1)
    yi = tensor(ds.u0.ravel())
    if getattr(args,"state_model","plain") in ("gaussian_ic","gaussian_ic_bc","gaussian_ic_bc_locald"):
        with torch.no_grad():initial_error=float(torch.linalg.vector_norm(state_value(xi)-yi)/torch.linalg.vector_norm(yi))
        if initial_error>1e-6:
            raise ValueError(f"Gaussian IC base disagrees with supplied initial grid (relative L2={initial_error:.3g}); use another state model")
    scale = max(float(np.max(np.abs(ds.u0))), float(np.max(np.abs(ds.u))), 1e-12)
    n = len(ds.u)
    hard_points=None
    sampling_history=[]

    def refresh_hard_points(step):
        nonlocal hard_points
        pool=rng.uniform(0,1,(args.adaptive_pool,3))
        pool[:len(pool)//2,2]=10**rng.uniform(-4,0,len(pool)//2)
        scores=[]
        for offset in range(0,len(pool),512):
            candidates=tensor(pool[offset:offset+512]).requires_grad_(True)
            scores.append(residual(temperature,coefficient,candidates,ds.T).detach().abs().cpu().numpy())
        scores=np.concatenate(scores)
        count=max(1,args.n_collocation//4)
        if count>len(pool):raise ValueError('Adaptive pool must contain the focused collocation subset')
        chosen=np.argsort(scores)[-count:]
        hard_points=pool[chosen].copy()
        sampling_history.append(dict(step=step,pool=len(pool),focused_points=count,
                                     max_residual=float(scores.max()),focused_threshold=float(scores[chosen].min())))
        print(f'  PDE adaptive refresh step={step} max residual={scores.max():.4g}',flush=True)

    def batch():
        # Half early-time log-spaced samples resolve the sharp IC transient.
        col = rng.uniform(0, 1, (args.n_collocation, 3))
        col[:len(col)//2, 2] = 10 ** rng.uniform(-4, 0, len(col)//2)
        if hard_points is not None:col[-len(hard_points):]=hard_points
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
        data = .5 * n * ((state_value(xd[indices])-yd[indices])/solver.sigma_w).square().mean()
        pde = .5 * n * (residual(temperature, coefficient, col, ds.T)/scale).square().mean()
        ic = .5 * n * ((state_value(xi)-yi)/scale).square().mean()
        bc = .5 * n * ((state_value(xb)-yb)/scale).square().mean()
        reg = field_regularizer(coefficient(xy).ravel(), solver)
        total = data + args.physics_weight*pde + args.condition_weight*(ic+bc) + alpha*reg
        return total, dict(data=float(data.detach()), pde=float(pde.detach()),
                           ic=float(ic.detach()), bc=float(bc.detach()), reg=float(reg.detach()))

    history = []
    opt = torch.optim.Adam(params, lr=args.lr)
    for step in range(1, args.adam_steps+1):
        if getattr(args,'pde_sampler','mixed')=='residual_adaptive' and step>1 and (step-1)%1500==0:
            refresh_hard_points(step)
        opt.zero_grad(); loss, parts = losses(batch()); loss.backward(); opt.step()
        if step == 1 or step % max(1, args.adam_steps//10) == 0:
            history.append(dict(stage="adam", step=step, total=float(loss.detach()), **parts))
            print(f"  PINN step={step} loss={loss.item():.4g} data={parts['data']:.4g}", flush=True)
    optimizer_diagnostics=[];last_points=None
    rounds=getattr(args,'lbfgs_rounds',1)
    if rounds<1 or (args.lbfgs_steps and rounds>args.lbfgs_steps):raise ValueError('Invalid LBFGS round count')
    for round_index in range(rounds if args.lbfgs_steps else 0):
        if getattr(args,'pde_sampler','mixed')=='residual_adaptive':refresh_hard_points(args.adam_steps+1+round_index)
        points = batch()  # fixed throughout the line search
        iterations=args.lbfgs_steps//rounds+(round_index<args.lbfgs_steps%rounds)
        opt = torch.optim.LBFGS(params, max_iter=iterations, history_size=30,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-8,
                                tolerance_change=1e-12)
        def closure():
            opt.zero_grad(); loss, _ = losses(points); loss.backward(); return loss
        opt.step(closure)
        opt.zero_grad()
        loss, parts = losses(points)
        loss.backward()
        gradient=lambda group:float(torch.linalg.vector_norm(torch.cat([p.grad.detach().ravel() for p in group if p.grad is not None])))
        state=opt.state[params[0]]
        diagnostics=dict(round=round_index+1,requested_iterations=iterations,iterations=int(state.get('n_iter',0)),
            function_evaluations=int(state.get('func_evals',0)),
            gradient_max_abs=max(float(p.grad.detach().abs().max()) for p in params if p.grad is not None),
            temperature_gradient_norm=gradient(temperature.parameters()),coefficient_gradient_norm=gradient(coefficient.parameters()))
        optimizer_diagnostics.append(diagnostics)
        history.append(dict(stage="lbfgs",total=float(loss.detach()),**diagnostics,**parts))
        last_points={name:value.detach().cpu() for name,value in zip(('collocation','boundary','boundary_target','data_indices'),points)}
        print(f'  LBFGS round={round_index+1} loss={loss.item():.4g} iterations={diagnostics["iterations"]} gradient_max={diagnostics["gradient_max_abs"]:.4g}',flush=True)
    with torch.no_grad():
        D = coefficient.diffusivity(xy).cpu().numpy().ravel()
        fit = torch.cat([state_value(part).cpu() for part in xd.split(getattr(args, "data_batch", 4096))]).numpy()
        initial_difference=state_value(xi)-yi
        initial_relative=float(torch.linalg.vector_norm(initial_difference)/torch.linalg.vector_norm(yi))
        initial_rms=float(initial_difference.square().mean().sqrt())
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
                      temperature_config=temperature_config, T=ds.T,final_training_points=last_points)
    extra = dict(rms_unet_over_sigma=float(np.sqrt(np.mean((fit-ds.u)**2))/solver.sigma_w),
                 loss_history=history, physics_weight=args.physics_weight,
                 condition_weight=args.condition_weight, alpha_selection="fixed_cli",
                 device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                 state_model=getattr(args,"state_model","plain"),
                 pde_sampler=getattr(args,'pde_sampler','mixed'),sampling_history=sampling_history,
                 lbfgs_rounds=rounds,optimizer_diagnostics=optimizer_diagnostics,
                 warm_start_sha256=hashlib.sha256(Path(warm_start).read_bytes()).hexdigest() if warm_start else None,
                 initial_condition_relative_L2=initial_relative,initial_condition_rms=initial_rms,
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
    ap.add_argument("--state-model",choices=["plain","gaussian_ic","gaussian_ic_bc","gaussian_ic_bc_locald"],default="plain",help="Versioned state representation; locald couples the learned coefficient into the known-IC/BC heat base")
    ap.add_argument('--pde-sampler',choices=['mixed','residual_adaptive'],default='mixed',help='Optional separate pilot: focus one quarter of collocation points on current PDE violations')
    ap.add_argument('--adaptive-pool',type=int,default=16384,help='Known-residual candidate pool for residual_adaptive only')
    ap.add_argument('--warm-start',help='Separate continuation: restore a PINN checkpoint for the identical observation case')
    ap.add_argument('--lbfgs-rounds',type=int,default=1,help='Split total LBFGS iterations over refreshed fixed batches')
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
        with open(Path(args.data)/entry["files"]["problem"]) as f:
            ds.initial_expression=json.load(f).get("initial_condition",{}).get("expression")
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
