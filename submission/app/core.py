"""
core.py -- backend of the inversion interface (IE 643, subtask S6).

Every data source (benchmark dataset, newly generated synthetic data, uploaded sensor
readings) is turned into ONE dataset folder in the project's v2 format
(problem/ + sensors/ + optional ground_truth/), so the project's own solvers run on it
unchanged:

  classical : adjoint_inversion.AdjointSolver   (discrete CN adjoint, L-BFGS-B, discrepancy alpha)
  PINN      : pinn_inversion.train_single_experiment

Both estimates are scored the same way: D_hat is pushed through the same Crank-Nicolson
solver and compared with the sensor readings (RMS_replay / sigma, the displayed metric);
if the true D is known, relL2(D), RMS of log D, the constant-D baseline and a field error map
are added.
"""
from __future__ import annotations

import csv, io, json, os, shutil, sys, tempfile, time, zipfile
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parent.parent          # the project's v2/ folder
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import generate_dataset_v2 as G                       # noqa: E402
from domain import Domain2D, get_D_field, defect_layout  # noqa: E402
from integrators import ForwardModel, BoundaryCondition, EDGES  # noqa: E402
import adjoint_inversion as A                         # noqa: E402

BENCH_DIR = V2 / "datasets_v2_rk4_zero"
N_DATA, N_INV = 129, 65
TIERS = list(G.TIERS)
PROTOCOLS = list(G.PROTOCOLS)
LAYOUTS = list(G.LAYOUTS)
BC_SCENARIOS = list(G.BC_SCENARIOS)


# =============================================================================
# 1. Data sources -> a dataset folder in the v2 format
# =============================================================================
def new_workdir(prefix="run"):
    d = Path(tempfile.mkdtemp(prefix=f"ie643_{prefix}_"))
    return d


def benchmark_available():
    return (BENCH_DIR / "manifest.json").exists()


def benchmark_manifest():
    return json.load(open(BENCH_DIR / "manifest.json"))


def load_benchmark(key):
    """A benchmark dataset: the manifest entry and its folder (read in place)."""
    entry = next(e for e in benchmark_manifest() if e["key"] == key)
    return dict(root=str(BENCH_DIR), entry=entry, key=key, source="benchmark")


def _write_sensors(out, key, x, y, t, u, sigma, u_clean=None):
    os.makedirs(out / "sensors", exist_ok=True)
    sig = np.full(len(u), float(sigma))
    data = dict(x=x, y=y, t=t, u=u, sigma=sig)
    if u_clean is not None:
        data["u_clean"] = u_clean
    np.savez_compressed(out / "sensors" / f"{key}.npz", **data)
    with open(out / "sensors" / f"{key}.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["x", "y", "t", "u", "sigma"])
        for row in zip(x, y, t, u, sig):
            w.writerow([f"{v:.8g}" for v in row])


def _entry(key, stem, n_readings, sigma, T, Fo, bc_name, has_truth, extra=None):
    e = dict(key=key, problem=stem, ground_truth=f"{stem}.npz", n_readings=int(n_readings), sigma=float(sigma),
             T=float(T), Fo=float(Fo), bc=bc_name,
             files={"sensors_csv": f"sensors/{key}.csv", "sensors_npz": f"sensors/{key}.npz",
                    "problem": f"problem/{stem}.json",
                    "ground_truth": f"ground_truth/{stem}.npz" if has_truth else None})
    e.update(extra or {})
    return e


def generate_synthetic(tier="C_inclusion", Fo=0.1, protocol="uniform8", n_sensors=50, layout="random",
                       eta=0.05, bc_name="zero", noise_seed=None, custom_D=None, progress=None):
    """Simulate one synthetic dataset exactly as generate_dataset_v2.py does (RK4 on 129^2,
    sensors read off the fine grid, sigma = eta x reference RMS), then write it to a new folder.
    custom_D: optional (65,65) or (129,129) array replacing the tier's field."""
    dom, dom_inv = Domain2D(Nx=N_DATA, Ny=N_DATA), Domain2D(Nx=N_INV, Ny=N_INV)
    r = (N_DATA - 1) // (N_INV - 1)
    if custom_D is not None:
        D = _to_data_grid(np.asarray(custom_D, float), dom)
        cls, kw, tier = "custom", {}, "custom"
    else:
        cls, kw = G.TIERS[tier]
        D = get_D_field(dom, cls, **kw)
    if np.any(D <= 0):
        raise ValueError("D must be positive everywhere")
    T = Fo / float(D.mean())
    bc = G.make_bc(bc_name, T)
    if progress: progress(0.1, f"Simulating the forward problem (RK4, {N_DATA}x{N_DATA}, T = {T:.3f}) ...")
    t, U, info = ForwardModel(dom, D, stencil="5pt", bc=bc).solve(G.initial_condition(dom, bc), T, G.N_OUT, "rk4")
    if progress: progress(0.7, "Sampling sensors and adding noise ...")
    sig_ref = G.sigma_reference(dom, U, tier)
    n = "full" if n_sensors == "full" else int(n_sensors)
    if n == "full":
        layout = "nodes"
    ks, xy, u_clean = G.read_clean(dom, U, tier, protocol, n, layout if n != "full" else "random", dom_inv)
    m = len(xy)
    xs, ys = np.tile(xy[:, 0], len(ks)), np.tile(xy[:, 1], len(ks))
    ts = np.repeat(t[ks], m)
    sigma = eta * sig_ref
    seed = G.seed_of("noise", tier, Fo, protocol, n, layout, eta) if noise_seed is None else int(noise_seed)
    u = u_clean + sigma * np.random.default_rng(seed).standard_normal(u_clean.shape)

    stamp = time.strftime("%H%M%S")
    stem = f"{tier}__Fo{Fo:g}x{stamp}"
    key = f"{stem}__{protocol}__n{n}__{layout}__eta{eta:g}"
    out = new_workdir("synthetic") / "datasets_v2_rk4_zero"
    for sub in ("problem", "ground_truth", "sensors"):
        os.makedirs(out / sub, exist_ok=True)
    G.write_problem(str(out), stem, dom, dom_inv, r, T, Fo, bc, bc_name, t)
    defects = defect_layout(**kw) if cls == "F_defects" else []
    np.savez_compressed(out / "ground_truth" / f"{stem}.npz", D_data=D, D_inv=D[::r, ::r], x_inv=dom_inv.x,
                        y_inv=dom_inv.y, t=t[::G.GT_EVERY], u=U[::G.GT_EVERY].astype(np.float32), T=T, Fo=Fo,
                        integrator="rk4", bc=bc_name, defects=json.dumps(defects))
    _write_sensors(out, key, xs, ys, ts, u, sigma, u_clean)
    entry = _entry(key, stem, len(u), sigma, T, Fo, bc_name, True,
                   dict(tier=tier, protocol=protocol, n_sensors=m, layout=layout, eta=eta))
    json.dump([entry], open(out / "manifest.json", "w"), indent=1)
    if progress: progress(1.0, "Dataset ready")
    return dict(root=str(out), entry=entry, key=key, source="synthetic",
                snapshots=dict(t=t[::G.GT_EVERY], u=U[::G.GT_EVERY][:, ::r, ::r]))


def _to_data_grid(D, dom):
    if D.shape == (N_DATA, N_DATA):
        return D
    if D.shape != (N_INV, N_INV):
        raise ValueError(f"custom D must be {N_INV}x{N_INV} or {N_DATA}x{N_DATA}, got {D.shape}")
    from scipy.interpolate import RegularGridInterpolator
    g = np.linspace(0, 1, N_INV)
    return RegularGridInterpolator((g, g), D)(np.c_[dom.X.ravel(), dom.Y.ravel()]).reshape(dom.shape)


def parse_sensor_csv(raw: bytes):
    """Uploaded readings: CSV with header x,y,t,u and optionally sigma (long format, one row per reading)."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("this file is not a text CSV; save it as CSV (comma-separated) with a header x,y,t,u[,sigma]")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError("the file is empty")
    head = [h.strip().lower() for h in rows[0]]
    need = ["x", "y", "t", "u"]
    miss = [c for c in need if c not in head]
    if miss:
        raise ValueError(f"CSV is missing column(s) {miss}; expected a header with x,y,t,u[,sigma]")
    idx = {c: head.index(c) for c in head}
    try:
        arr = np.array([[float(r[idx[c]]) for c in need + (["sigma"] if "sigma" in idx else [])]
                        for r in rows[1:] if r and any(s.strip() for s in r)], dtype=float)
    except (ValueError, IndexError) as e:
        raise ValueError(f"every row needs numbers in x, y, t, u (problem: {e})")
    if arr.ndim != 2 or len(arr) == 0:
        raise ValueError("the CSV has a header but no data rows")
    if not np.all(np.isfinite(arr)):
        bad = int(np.argwhere(~np.isfinite(arr))[0][0]) + 2
        raise ValueError(f"empty, NaN or infinite value in the data (first at line {bad})")
    out = dict(x=arr[:, 0], y=arr[:, 1], t=arr[:, 2], u=arr[:, 3])
    if arr.shape[1] == 5:
        out["sigma"] = float(np.median(arr[:, 4]))
    if np.any((out["x"] < 0) | (out["x"] > 1) | (out["y"] < 0) | (out["y"] > 1)):
        raise ValueError("x and y must lie in [0, 1] (the unit square)")
    if np.any(out["t"] <= 0):
        raise ValueError("t must be > 0 (t = 0 is the known initial condition)")
    return out


def parse_grid(raw: bytes, name: str):
    """A 65x65 (or 129x129) field from .npy, or .csv as a matrix or as x,y,value columns."""
    if name.lower().endswith(".npy"):
        return np.load(io.BytesIO(raw))
    text = raw.decode("utf-8-sig").strip().splitlines()
    first = [s.strip().lower() for s in text[0].split(",")]
    if len(first) == 3 and first[0] in ("x",) and first[1] == "y":
        a = np.loadtxt(text[1:], delimiter=",")
        n = int(round(np.sqrt(len(a))))
        order = np.lexsort((a[:, 1], a[:, 0]))               # x-major, 'ij' indexing
        return a[order, 2].reshape(n, n)
    return np.loadtxt(text, delimiter=",")


def load_upload(sensors: dict, T: float, sigma: float | None = None, bc_name="zero",
                ic_grid=None, D_true=None):
    """Uploaded readings -> dataset folder. IC defaults to the benchmark's two Gaussian bumps;
    BC is one of the benchmark scenarios; D_true (optional) enables the field metrics."""
    sig = sensors.get("sigma") if sigma is None else sigma
    if sig is None:
        raise ValueError("No sigma column and no noise level given")
    t_max = float(sensors["t"].max())
    if T < t_max * (1 - 1e-3):
        raise ValueError(f"T = {T} is smaller than the latest reading time {t_max:.4g}")
    T = max(float(T), t_max)          # tolerate rounding of t in the CSV
    dom, dom_inv = Domain2D(Nx=N_DATA, Ny=N_DATA), Domain2D(Nx=N_INV, Ny=N_INV)
    bc = G.make_bc(bc_name, T)
    stamp = time.strftime("%H%M%S")
    stem = f"upload__Fo0x{stamp}"
    n = len(np.unique(np.c_[sensors["x"], sensors["y"]], axis=0))
    key = f"{stem}__custom__n{n}__custom__eta0"
    out = new_workdir("upload") / "datasets_v2_rk4_zero"
    for sub in ("problem", "sensors", "ground_truth"):
        os.makedirs(out / sub, exist_ok=True)
    t_out = np.linspace(0, T, G.N_OUT + 1)
    G.write_problem(str(out), stem, dom, dom_inv, 2, T, 0.0, bc, bc_name, t_out)
    if ic_grid is not None:                                   # overwrite the IC with the uploaded one
        ic = np.asarray(ic_grid, float)
        if ic.shape != (N_INV, N_INV):
            raise ValueError(f"initial condition must be {N_INV}x{N_INV}")
        X, Y = np.meshgrid(dom_inv.x, dom_inv.y, indexing="ij")
        np.savetxt(out / "problem" / f"{stem}__ic.csv", np.c_[X.ravel(), Y.ravel(), ic.ravel()],
                   delimiter=",", header="x,y,u0", comments="", fmt="%.8g")
        p = dict(np.load(out / "problem" / f"{stem}.npz"))
        p["u0_inv"] = ic
        np.savez_compressed(out / "problem" / f"{stem}.npz", **p)
    has_truth = D_true is not None
    if has_truth:
        Dt = np.asarray(D_true, float)
        if Dt.shape == (N_DATA, N_DATA):
            Dt = Dt[::2, ::2]
        if Dt.shape != (N_INV, N_INV):
            raise ValueError(f"true D must be {N_INV}x{N_INV}")
        np.savez_compressed(out / "ground_truth" / f"{stem}.npz", D_inv=Dt, D_data=Dt, T=T, defects="[]")
    _write_sensors(out, key, sensors["x"], sensors["y"], sensors["t"], sensors["u"], sig)
    entry = _entry(key, stem, len(sensors["u"]), sig, T, 0.0, bc_name, has_truth,
                   dict(tier="upload", custom_ic=ic_grid is not None))
    json.dump([entry], open(out / "manifest.json", "w"), indent=1)
    data = dict(root=str(out), entry=entry, key=key, source="upload")
    # The PINN initialises D from the Fourier number in the problem spec (D_char = Fo L^2 / T).
    # An upload has no known Fo, so estimate it from the best constant D (data only, no truth).
    c = constant_baseline(data)
    Fo = c["D_const"] * T
    spec_path = out / "problem" / f"{stem}.json"
    spec = json.load(open(spec_path)); spec["domain"]["Fourier_number"] = Fo
    json.dump(spec, open(spec_path, "w"), indent=2)
    p = dict(np.load(out / "problem" / f"{stem}.npz")); p["Fo"] = Fo
    np.savez_compressed(out / "problem" / f"{stem}.npz", **p)
    entry["Fo"] = Fo; entry["D_const"] = c["D_const"]
    json.dump([entry], open(out / "manifest.json", "w"), indent=1)
    return data


def sensor_csv_bytes(data) -> bytes:
    """The dataset's sensor readings as CSV (x,y,t,u,sigma), the same format the upload box accepts."""
    return (Path(data["root"]) / data["entry"]["files"]["sensors_csv"]).read_bytes()


def dataset_zip(data) -> bytes:
    """One dataset as a zip in the project's format (datasets_v2_rk4_zero/ with manifest, problem,
    sensors and, if known, ground truth), readable by the app's benchmark loader and by the solvers."""
    root, e = Path(data["root"]), data["entry"]
    stem = e["problem"]
    files = [e["files"]["sensors_csv"], e["files"]["sensors_npz"],
             f"problem/{stem}.json", f"problem/{stem}.npz", f"problem/{stem}__ic.csv"]
    if e["files"].get("ground_truth"):
        files.append(e["files"]["ground_truth"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            if (root / rel).exists():
                z.write(root / rel, f"datasets_v2_rk4_zero/{rel}")
        z.writestr("datasets_v2_rk4_zero/manifest.json", json.dumps([e], indent=1, default=str))
        z.writestr("datasets_v2_rk4_zero/README.txt",
                   f"Dataset {e['key']}\n"
                   "sensors/<key>.csv   readings: x, y, t, u, sigma (one row per reading)\n"
                   "sensors/<key>.npz   the same plus clean values, sensor ids, time indices\n"
                   "problem/            PDE, domain, final time T, initial condition, boundary values\n"
                   "ground_truth/       true D (answer key, scoring only), if known\n"
                   "manifest.json       this dataset's configuration and seeds\n")
    return buf.getvalue()


# =============================================================================
# 2. Shared scoring
# =============================================================================
def dataset(data):
    return A.Dataset(data["root"], data["entry"])


def truth(data):
    gt = data["entry"]["files"].get("ground_truth")
    path = Path(data["root"]) / gt if gt else None
    if not path or not path.exists():
        return None
    g = np.load(path)
    defects = json.loads(str(g["defects"])) if "defects" in g.files else []
    return dict(D=g["D_inv"], defects=defects)


def score(data, D_hat, ds=None):
    """Same metrics for every method: D_hat replayed through the CN solver on 65^2."""
    ds = ds or dataset(data)
    s = A.AdjointSolver(ds, control=2)
    D_hat = np.asarray(D_hat, float).reshape(ds.N, ds.N)
    W, *_ = s.forward(D_hat.ravel())
    pred = s.predict(W)
    sig = ds.sigma if ds.sigma > 0 else None
    m = dict(rms_replay=float(np.sqrt(np.mean((pred - ds.u) ** 2))),
             rms_replay_over_sigma=float(np.sqrt(np.mean((pred - ds.u) ** 2)) / sig) if sig else None,
             D_min=float(D_hat.min()), D_max=float(D_hat.max()), D_mean=float(D_hat.mean()))
    tr = truth(data)
    if tr is not None:
        Dt = tr["D"]
        m.update(relL2_D=float(np.linalg.norm(D_hat - Dt) / np.linalg.norm(Dt)),
                 rmse_logD=float(np.sqrt(np.mean((np.log(D_hat) - np.log(Dt)) ** 2))))
    return m, W, s


def constant_baseline(data):
    """Best single uniform D for the data -- the floor any useful map should beat."""
    ds = dataset(data)
    s = A.AdjointSolver(ds, control=2)
    from scipy.optimize import minimize_scalar
    f = lambda lm: s.objective(np.full(4, lm), 0.0, want_grad=False)
    lm = float(minimize_scalar(f, bounds=(np.log(1e-3), np.log(5.0)), method="bounded", options={"xatol": 1e-4}).x)
    D = np.full((ds.N, ds.N), np.exp(lm))
    m, _, _ = score(data, D, ds)
    return dict(D_const=float(np.exp(lm)), **m)


# =============================================================================
# 3. Methods
# =============================================================================
def run_classical(data, control=16, reg="tikhonov", maxiter=150, outdir=None):
    ds = dataset(data)
    t0 = time.perf_counter()
    solver = A.AdjointSolver(ds, control=control, reg=reg)
    out = solver.invert(maxiter=maxiter, verbose=False)
    wall = time.perf_counter() - t0
    D = out["D"].reshape(ds.N, ds.N)
    m, W, _ = score(data, D, ds)
    S = A.sensitivity_map(solver, out["W"])
    m.update(method="classical adjoint", alpha=out["alpha"], flag=out["flag"], wall_s=wall,
             pde_solves=out["n_evals"], control=f"{control}x{control}", reg=reg)
    outdir = Path(outdir or new_workdir("classical"))
    os.makedirs(outdir, exist_ok=True)
    model = outdir / "classical_model.npz"
    np.savez_compressed(model, kind="classical_adjoint", theta=out["theta"], control=control, reg=reg,
                        alpha=out["alpha"], D_hat=D, x=np.linspace(0, 1, ds.N), y=np.linspace(0, 1, ds.N))
    return dict(D=D, metrics=m, sensitivity=S, path=[{k: v for k, v in p.items()} for p in out["path"]],
                model_file=str(model), outdir=str(outdir))


def run_pinn(data, epochs=1500, reg_type=None, lambda_reg=None, seed=42, outdir=None, log=None):
    import torch
    import pinn_inversion as P
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    key = data["key"]
    tier = data["entry"].get("tier", "")
    if reg_type is None:                     # the project's benchmark-suite defaults
        reg_type = "tikhonov" if str(tier).startswith(("A_", "B_", "E_")) else "tv"
    if lambda_reg is None:
        lambda_reg = 2e-4 if reg_type == "tikhonov" else 3e-4
    outdir = Path(outdir or new_workdir("pinn"))
    root = str(Path(data["root"]).parent)    # loader searches for datasets_v2_rk4_zero below this
    cfg = P.PINNConfig(data_root=root, output_dir=str(outdir), exp_id=key, epochs=int(epochs),
                       reg_type=reg_type, lambda_reg=float(lambda_reg), seed=int(seed))
    t0 = time.perf_counter()
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pm = P.train_single_experiment(cfg, run_tag="pinn")
    wall = time.perf_counter() - t0
    run_dir = outdir / "pinn"
    D = np.load(run_dir / "D_hat.npz")["D_hat"].astype(float)
    m, W, _ = score(data, D)
    m.update(method="PINN", wall_s=wall, epochs=int(epochs), reg=reg_type, lambda_reg=float(lambda_reg),
             chi2_u_net=pm.get("chi2_red_u_net"))
    hist = None
    if (run_dir / "loss_history.csv").exists():
        import pandas as pd
        hist = pd.read_csv(run_dir / "loss_history.csv")
    return dict(D=D, metrics=m, history=hist, model_file=str(run_dir / "pinn_best.pt"), outdir=str(outdir),
                log=buf.getvalue())


# =============================================================================
# 4. Saved models
# =============================================================================
def load_model_file(raw: bytes, name: str):
    """Return (D on 65x65, info) from a saved classical .npz or PINN .pt checkpoint."""
    if name.endswith(".npz"):
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        if "theta" in z.files and "control" in z.files:
            K = int(z["control"])
            P1 = A.bilinear_1d(N_INV, K)
            import scipy.sparse as sp
            D = np.exp(sp.kron(P1, P1) @ z["theta"]).reshape(N_INV, N_INV)
            return D, dict(kind="classical adjoint", control=f"{K}x{K}", reg=str(z["reg"]) if "reg" in z.files else "?",
                           alpha=float(z["alpha"]) if "alpha" in z.files else None)
        if "D_hat" in z.files:
            return np.asarray(z["D_hat"], float), dict(kind="D_hat array")
        raise ValueError("npz has neither theta/control nor D_hat")
    if name.endswith(".pt"):
        import torch
        import pinn_inversion as P
        ck = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        cfg = P.PINNConfig(**{k: v for k, v in ck["config"].items() if k in P.PINNConfig.__dataclass_fields__})
        model = P.InverseDiffusionPINN(cfg, 1.0, 1.0, 1.0)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        g = np.linspace(0, 1, N_INV, dtype=np.float32)
        X, Y = np.meshgrid(g, g, indexing="ij")
        with torch.no_grad():
            D = model.forward_D(torch.tensor(np.c_[X.ravel(), Y.ravel()])).numpy().reshape(N_INV, N_INV)
        return D.astype(float), dict(kind="PINN", exp_id=cfg.exp_id, epochs=cfg.epochs, reg=cfg.reg_type,
                                     lambda_reg=cfg.lambda_reg)
    raise ValueError("expected a .npz (classical) or .pt (PINN) file")


def results_zip(data, results: dict, figures: dict) -> bytes:
    """Everything from one session in one zip: models, D_hat as csv/npz, metrics, figures, inputs."""
    buf = io.BytesIO()
    g = np.linspace(0, 1, N_INV)
    X, Y = np.meshgrid(g, g, indexing="ij")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("dataset/manifest_entry.json", json.dumps(data["entry"], indent=1, default=str))
        z.write(Path(data["root"]) / data["entry"]["files"]["sensors_csv"], "dataset/sensors.csv")
        metrics = {}
        for name, r in results.items():
            tag = "classical" if name.startswith("classical") else "pinn"
            metrics[name] = r["metrics"]
            z.write(r["model_file"], f"models/{os.path.basename(r['model_file'])}")
            s = io.StringIO()
            np.savetxt(s, np.c_[X.ravel(), Y.ravel(), r["D"].ravel()], delimiter=",", header="x,y,D", comments="", fmt="%.8g")
            z.writestr(f"D_hat_{tag}.csv", s.getvalue())
            nb = io.BytesIO(); np.savez_compressed(nb, D_hat=r["D"], x=g, y=g); z.writestr(f"D_hat_{tag}.npz", nb.getvalue())
        z.writestr("metrics.json", json.dumps(metrics, indent=1, default=float))
        for fname, png in figures.items():
            z.writestr(f"figures/{fname}.png", png)
    return buf.getvalue()
