"""
run_study.py -- solver (RK4 / CN / BE) x sensor density x noise, with the standard classical adjoint.

Data: the standard RK4 129^2 benchmark, thinned to nested node sets (datasets_v2_density_extended,
built by study1_representation/extend_sensor_density.py). Field A_smooth, Fo 0.1, 8 uniform times.
  sparse 50 sensors (400 readings), medium 400 (3,200), dense 3,969 = every interior node (31,752)
  noise 1, 5, 10 %

Blocks (same step count for every solver inside a block):
  A  RK4, CN, BE at 4,800 steps  (the coarsest step at which RK4 is stable for every D <= 1)
  B  CN and BE at 100 steps (the backward-Euler step of study 4 / slide 15) and 200 steps
     (the CN adjoint's usual step for these reading times); RK4 is unstable at these steps.

Each run: best-constant start, Tikhonov, L-BFGS-B 150 iterations per alpha, alpha = N_d * 10^(0..-6)
in 13 steps, stop at the first alpha whose replay RMS <= sigma (discrepancy principle) -- exactly
AdjointSolver.invert. The true D is read only after the inversion, for scoring.

Scores: rel_L2_D (65^2 grid), rmse_logD, replay RMS / sigma under the run's own solver, and replay under
a common reference (CN, 4,800 steps) so every estimate is judged by the same accurate solver.

Usage (from extended_studies/):
  python run.py study5_time_integrators/run_study.py --block B
  python run.py study5_time_integrators/run_study.py --block A --solvers RK4 --keys <key> ...
Results: study5_time_integrators/results/<block>/<solver>_<steps>/<key>/{metrics.json, D_hat.npz}
Finished runs are skipped, so the command can be restarted.
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

import adjoint_inversion as A
import solvers as S

HERE = Path(__file__).resolve().parent
DATA = "datasets_v2_density_extended"
COUNTS = (50, 400, 3969)
ETAS = (0.01, 0.05, 0.1)
KEYS = [f"A_smooth__Fo0.1__uniform8__n{n}__nodes_nested__eta{e:g}" for n in COUNTS for e in ETAS]
BLOCKS = {"A": [("RK4", 4800), ("CN", 4800), ("BE", 4800)],
          "B": [("CN", 100), ("BE", 100), ("CN", 200), ("BE", 200)]}
REF = ("CN", 4800)


def dataset(key):
    e = next(r for r in json.load(open(f"{DATA}/manifest.json")) if r["key"] == key)
    return A.Dataset(DATA, e)


def run_one(block, name, n_steps, key, ref_cache):
    out_dir = HERE / "results" / block / f"{name}_{n_steps}" / key
    if (out_dir / "metrics.json").exists():
        return "skip"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = dataset(key)
    solver = S.make(name, ds, n_steps)
    t0 = time.perf_counter()
    res = solver.invert(verbose=False)
    wall = time.perf_counter() - t0
    D = res["D"]
    own = float(np.sqrt(np.mean((solver.predict(res["W"]) - ds.u) ** 2)) / solver.sigma_w)
    if key not in ref_cache:
        ref_cache[key] = S.make(REF[0], ds, REF[1])
    ref = ref_cache[key]
    Wr = ref.forward(D)[0]
    common = float(np.sqrt(np.mean((ref.predict(Wr) - ds.u) ** 2)) / ref.sigma_w)
    D_true, _ = ds.ground_truth()                     # answer key: only now
    Dt = D_true.ravel()
    m = dict(block=block, solver=name, n_steps=n_steps, key=key, n_sensors=int(len(ds.u) // 8),
             n_readings=int(len(ds.u)), eta=float(key.rsplit("eta", 1)[1]), sigma=ds.sigma,
             rel_L2_D=float(np.linalg.norm(D - Dt) / np.linalg.norm(Dt)),
             rmse_logD=float(np.sqrt(np.mean((np.log(D) - np.log(Dt)) ** 2))),
             replay_own_over_sigma=own, replay_common_CN4800_over_sigma=common,
             const_rel_L2_D=float(np.linalg.norm(res["D_const"] - Dt) / np.linalg.norm(Dt)),
             D_const=res["D_const"], alpha=res["alpha"], flag=res["flag"], n_alphas=len(res["path"]),
             total_iters=int(sum(p["iters"] for p in res["path"])), n_evals=res["n_evals"], wall_s=wall,
             D_min=float(D.min()), D_max=float(D.max()), D_cap=S.D_CAP,
             hit_cap=bool(D.max() >= 0.999 * S.D_CAP), path=res["path"])
    if name == "RK4":
        m["max_dt_rho"] = solver.max_dt_rho
    np.savez_compressed(out_dir / "D_hat.npz", D_hat=D.reshape(ds.N, ds.N))
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=1))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=sorted(BLOCKS), required=True)
    ap.add_argument("--solvers", nargs="*", default=None)
    ap.add_argument("--keys", nargs="*", default=KEYS)
    a = ap.parse_args()
    arms = [(s, n) for s, n in BLOCKS[a.block] if a.solvers is None or s in a.solvers]
    ref_cache = {}
    for key in a.keys:                                 # sparse -> dense, so partial runs are useful
        for name, n in arms:
            t = time.strftime("%H:%M:%S")
            try:
                m = run_one(a.block, name, n, key, ref_cache)
            except Exception:
                print(f"{t} FAILED {a.block} {name}_{n} {key}\n{traceback.format_exc()}", flush=True)
                continue
            if m == "skip":
                print(f"{t} skip {name}_{n} {key}", flush=True)
            else:
                print(f"{t} {name}_{n:<5d} {key}: relL2 {m['rel_L2_D']:.4f}  replay own {m['replay_own_over_sigma']:.3f}"
                      f"  common {m['replay_common_CN4800_over_sigma']:.3f}  {m['flag']}  {m['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
