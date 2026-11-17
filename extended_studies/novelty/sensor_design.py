"""
sensor_design.py -- datasets for E3b: the same sensor budget placed three ways.

Every layout is a set of rows taken from the 3,969-node dataset of the density collection
(datasets_v2_density_extended, A_smooth / C_inclusion / E_grf, 8 uniform times, 5 % noise), so all
layouts share one noise draw and differ only in WHERE the sensors are:

  random    n nodes drawn at random (seeded per tier)
  grid      the nodes nearest a regular sqrt(n) x sqrt(n) lattice on [0.05, 0.95]^2
  designed  greedy D-optimal design: maximise log det(A0 + sum_s J_s^T J_s / sigma^2), where J_s (8 x 256)
            is the sensitivity of sensor s's 8 readings to the adjoint's 16 x 16 log-D controls. J is
            computed at a PRIOR constant D = Fo / T (the diffusivity the experiment was designed for), by
            forward differences through the project's CN solver. No readings and no true D are used.
            A0 = lambda I with lambda = the mean per-sensor information trace / 256 (one average sensor's
            worth, spread evenly), which keeps the first picks well defined.

  python run.py novelty/sensor_design.py --size full      # writes datasets_novelty_oed/
"""
import argparse
import csv
import json
import os
import shutil
import time

import numpy as np

import adjoint_inversion as A

SRC = "datasets_v2_density_extended"
OUT = "datasets_novelty_oed"
TIERS = ("A_smooth", "C_inclusion", "E_grf")
BUDGETS = (16, 36, 64)
LAYOUTS = ("random", "grid", "designed")
ETA = 0.05


def configs(size):
    if size == "quick":
        return [("A_smooth", 16, lay) for lay in LAYOUTS]
    return [(t, n, lay) for t in TIERS for n in BUDGETS for lay in LAYOUTS]


def key_of(tier, n, layout):
    return f"{tier}__Fo0.1__uniform8__n{n}__{layout}__eta{ETA:g}"


def source(tier):
    m = {e["key"]: e for e in json.load(open(os.path.join(SRC, "manifest.json")))}
    e = m[f"{tier}__Fo0.1__uniform8__n3969__nodes_nested__eta{ETA:g}"]
    return e, A.Dataset(SRC, e)


def sensitivities(ds, eps=1e-3):
    """J (n_readings x K^2): d readings / d theta at theta = log(Fo / T) everywhere."""
    s = A.AdjointSolver(ds)
    theta0 = np.full(s.K ** 2, np.log(0.1 / ds.T))
    base = s.predict(s.forward(np.exp(s.m_of(theta0)))[0])
    J = np.empty((len(base), s.K ** 2))
    for j in range(s.K ** 2):
        th = theta0.copy(); th[j] += eps
        J[:, j] = (s.predict(s.forward(np.exp(s.m_of(th)))[0]) - base) / eps
    return J


def greedy_d_optimal(J, loc_of_row, n_pick, sigma):
    locs = np.unique(loc_of_row)
    rows_of = [np.where(loc_of_row == l)[0] for l in locs]
    Jl = np.stack([J[r] for r in rows_of]) / sigma            # (n_loc, n_t, p)
    p = J.shape[1]
    lam = float(np.mean(np.einsum("lij,lij->l", Jl, Jl))) / p
    Ainv = np.eye(p) / lam
    chosen, gains = [], []
    avail = np.ones(len(locs), bool)
    for _ in range(n_pick):
        B = np.einsum("lij,jk->lik", Jl, Ainv)                 # J_l A^-1
        M = np.einsum("lik,ljk->lij", B, Jl)                   # J_l A^-1 J_l^T
        sign, ld = np.linalg.slogdet(np.eye(Jl.shape[1])[None] + M)
        ld[~avail] = -np.inf
        b = int(np.argmax(ld))
        chosen.append(b); gains.append(float(ld[b])); avail[b] = False
        Jb = Jl[b]
        S = np.eye(Jb.shape[0]) + Jb @ Ainv @ Jb.T
        Ainv = Ainv - Ainv @ Jb.T @ np.linalg.solve(S, Jb @ Ainv)    # Woodbury
    return locs[chosen], gains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=("quick", "full"), default="full")
    a = ap.parse_args()
    for sub in ("problem", "ground_truth", "sensors"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    man_path = os.path.join(OUT, "manifest.json")
    manifest = {e["key"]: e for e in json.load(open(man_path))} if os.path.exists(man_path) else {}
    cfgs = configs(a.size)
    for tier in dict.fromkeys(t for t, _, _ in cfgs):
        todo = [(n, lay) for t, n, lay in cfgs if t == tier and key_of(t, n, lay) not in manifest]
        if not todo:
            print(f"skip {tier}"); continue
        e, ds = source(tier)
        for f in (f"problem/{e['problem']}.json", f"problem/{e['problem']}.npz", f"problem/{e['problem']}__ic.csv",
                  f"ground_truth/{e['ground_truth']}"):
            if os.path.exists(os.path.join(SRC, f)):
                shutil.copy(os.path.join(SRC, f), os.path.join(OUT, f))
        npz = dict(np.load(os.path.join(SRC, e["files"]["sensors_npz"])))
        xy = np.round(np.c_[npz["x"], npz["y"]], 9)
        node_xy, loc = np.unique(xy, axis=0, return_inverse=True)
        loc = loc.ravel()
        J = None
        for (n, lay) in todo:
            if lay == "random":
                pick = np.random.default_rng(sum(map(ord, tier)) + n).choice(len(node_xy), n, replace=False)
                info = {}
            elif lay == "grid":
                k = int(round(np.sqrt(n)))
                g = 0.05 + (np.arange(k) + 0.5) / k * 0.90
                targets = np.array([(gx, gy) for gx in g for gy in g])
                pick = np.array([int(np.argmin(((node_xy - tg) ** 2).sum(1))) for tg in targets])
                info = {}
            else:
                if J is None:
                    tic = time.perf_counter()
                    J = sensitivities(ds)
                    print(f"[{tier}] sensitivities {J.shape} in {time.perf_counter() - tic:.0f}s", flush=True)
                pick, gains = greedy_d_optimal(J, loc, n, ds.sigma)
                info = dict(design_gains=gains, design_prior_D=0.1 / ds.T)
            rows = np.where(np.isin(loc, pick))[0]
            key = key_of(tier, n, lay)
            sub = {k: v[rows] for k, v in npz.items()}
            np.savez_compressed(os.path.join(OUT, "sensors", key + ".npz"), **sub)
            with open(os.path.join(OUT, "sensors", key + ".csv"), "w", newline="") as f:
                w = csv.writer(f, lineterminator="\n"); w.writerow(["x", "y", "t", "u", "sigma"])
                for r in zip(sub["x"], sub["y"], sub["t"], sub["u"], sub["sigma"]):
                    w.writerow([f"{v:.8g}" for v in r])
            manifest[key] = dict(key=key, tier=tier, base_tier=tier, Fo=0.1, T=e["T"], protocol="uniform8",
                                 n_sensors=int(len(pick)), layout=lay, n_readings=int(len(rows)), eta=ETA,
                                 sigma=e["sigma"], problem=e["problem"], ground_truth=e["ground_truth"],
                                 bc="zero", source_key=e["key"], sensor_xy=node_xy[pick].tolist(), **info,
                                 files={"sensors_csv": f"sensors/{key}.csv", "sensors_npz": f"sensors/{key}.npz",
                                        "problem": f"problem/{e['problem']}.json",
                                        "ground_truth": f"ground_truth/{e['ground_truth']}"})
            print(f"  {key}: {len(pick)} sensors, {len(rows)} readings", flush=True)
        json.dump(list(manifest.values()), open(man_path, "w"), indent=1, default=float)
    print(f"{len(manifest)} datasets in {OUT}/")


if __name__ == "__main__":
    main()
