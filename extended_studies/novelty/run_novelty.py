"""
run_novelty.py -- one command to run any novelty experiment, at a quick or a full size.

  python run.py novelty/run_novelty.py --exp E1 --size quick            # pilot
  python run.py novelty/run_novelty.py --exp E1 --size full --shard 0/4  # 1 of 4 parallel workers
  python run.py novelty/run_novelty.py --exp E1 --list                   # show the jobs, run nothing

Experiments (see novelty/README.md for the design):
  E1    hybrid: PINN -> adjoint (warm start on the same alpha path, balanced start, PINN as the
        regularisation reference) and adjoint -> PINN, against both alone
  E2a   third network D(x, y, u) on the standard data (no temperature dependence in the truth)
  E2b   Arrhenius data, D = D0(x, y) exp(s(u)); adjoint and PINN assuming D(x, y) vs PINN with g(u)
  E3a   sensor-split validation: choose PINN loss weights and adjoint alpha from held-out sensors
  E3b   sensor placement: random vs grid vs sensitivity-designed sensors, same budget
  E3c   adaptive collocation (RAR) vs the default sampler
  E3d   interpolation augmentation of sparse sensors

Every job writes results/<exp>/<arm>/<key>/metrics.json (+ D_hat.npz) and is skipped if that exists,
so a stopped run is resumed by running the same command again. Shards split by dataset, so all arms of
one dataset (which depend on each other) stay in the same worker.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

import common as C

DATASETS20 = (C.REPO / "submission" / "experiments" / "zhang_replication_study20" / "datasets20.txt").read_text().split()
QUICK_BENCH = ["A_smooth__Fo0.1__uniform8__n50__random__eta0.05",
               "C_inclusion__Fo0.1__uniform8__n50__random__eta0.05",
               "E_grf__Fo0.1__uniform8__n50__random__eta0.05"]
UNKNOWN_PHYSICS = [k for k in (
    "H2u-A_smooth-pulsedtop-a0.1", "H2u-A_smooth-pulsedtop-a0.3", "H2u-A_smooth-pulsedtop-a1",
    "H2u-A_smooth-heatedleft-a1", "H2u-A_smooth-rampleft-a1", "H2u-E_grf-pulsedtop-a0.3",
    "H2u-E_grf-heatedleft-a1", "H2u-E_grf-rampleft-a1",
    "H3u-A_smooth-q0.1", "H3u-A_smooth-q0.3", "H3u-A_smooth-q1",
    "H3u-E_grf-q0.1", "H3u-E_grf-q0.3", "H3u-E_grf-q1")]
UNKNOWN_SUFFIX = "__Fo0.1__uniform16__n50__random__eta0.05"
EPOCHS = {"quick": 300, "full": 1500}


# ---------------------------------------------------------------------------------------------
# experiment registry: exp -> list of (collection, key, [arms in dependency order])
# ---------------------------------------------------------------------------------------------
def registry(exp, size):
    if exp == "E1":
        bench = QUICK_BENCH if size == "quick" else DATASETS20
        unk = UNKNOWN_PHYSICS[2:3] if size == "quick" else UNKNOWN_PHYSICS
        rows = [("bench", k, ["adjoint", "pinn", "hybrid_path", "hybrid_balance", "hybrid_prior", "reverse"]) for k in bench]
        rows += [("pinnwin", k + UNKNOWN_SUFFIX, ["adjoint", "pinn", "hybrid_path", "hybrid_balance", "hybrid_prior"]) for k in unk]
        return rows
    if exp == "E1c":                                  # equal budget: the PINN alone given the hybrids' compute (3 x epochs)
        bench = QUICK_BENCH[:1] if size == "quick" else DATASETS20
        return [("bench", k, ["pinn_3x"]) for k in bench]
    if exp == "E1b":                                  # hidden walls: PINN walls -> adjoint, and the fair classical rival
        import hybrid_bc as HB
        keys = [k for k in UNKNOWN_PHYSICS if k.startswith("H2u")]
        keys = keys[2:3] if size == "quick" else keys
        return [("pinnwin", k + UNKNOWN_SUFFIX, list(HB.E1B_ARMS)) for k in keys]
    try:
        import experiments_more as XM                 # E2 / E3, added in their own module
    except ImportError:
        raise SystemExit(f"experiment {exp} not available yet")
    return XM.registry(exp, size)


# ---------------------------------------------------------------------------------------------
# E1 jobs
# ---------------------------------------------------------------------------------------------
def score(ds, D, extra):
    D_true, _ = ds.ground_truth()
    m = C.field_scores(D, D_true)
    m["replay_over_sigma"] = C.replay_rms_over_sigma(ds, D)
    m.update(extra)
    return m


def pinn_known(collection, key, epochs, init_D=None):
    import pinn_plus as PP
    cfg = PP.make_cfg(collection, key, epochs=epochs)
    out = PP.train(cfg, PP.PlusOptions(init_D=init_D), verbose=False)
    return out["D_hat"], out["wall_s"], out["device"]


def pinn_unknown_bc(collection, key, epochs):
    """Study 3's boundary-aware PINN (core/pinn_inversion_bc.py): free walls when the BC is unknown.
    Its built-in replay and plot are switched off so the timing is training only."""
    import tempfile
    import pinn_inversion_bc as PB
    PB.evaluate_slide16_chi2_metric = lambda *a, **k: dict(chi2_red_solver=float("nan"), chi2_red_u_net=float("nan"),
                                                           chi2_solver_dist_from_1=float("nan"),
                                                           chi2_u_net_dist_from_1=float("nan"), slide16_regime="n/a")
    PB.plot_summary_figure = lambda *a, **k: None
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PB.PINNConfig()
        cfg.data_root = str(C.EXT / C.COLLECTIONS[collection])
        cfg.output_dir = tmp
        cfg.exp_id = key
        cfg.epochs = epochs
        t0 = time.time()
        PB.train_single_experiment(cfg, run_tag="r")
        wall = time.time() - t0
        D = np.load(f"{tmp}/r/D_hat.npz")["D_hat"]
    return D, wall, C.device_name()


def run_E1(arm, collection, key, size):
    import hybrid as H
    ds = C.dataset(collection, key)
    epochs = EPOCHS[size]
    if arm == "adjoint":
        r = H.run_cold(ds)
        D_true, _ = ds.ground_truth()
        const = float(np.linalg.norm(r["D_const"] - D_true.ravel()) / np.linalg.norm(D_true.ravel()))
        m = score(ds, r["D"], dict(stage_wall_s=r["wall_s"], total_wall_s=r["wall_s"], alpha=r["alpha"], flag=r["flag"],
                                   total_iters=r["total_iters"], n_evals=r["n_evals"], path=r["path"],
                                   D_const=r["D_const"], const_rel_L2_D=const))
        return m, dict(D_hat=r["D"].reshape(ds.N, ds.N))
    if arm == "pinn":
        fn = pinn_known if collection != "pinnwin" else pinn_unknown_bc
        D, wall, dev = fn(collection, key, epochs)
        return score(ds, D, dict(stage_wall_s=wall, total_wall_s=wall, epochs=epochs, device=dev)), dict(D_hat=D)
    if arm in ("hybrid_path", "hybrid_balance", "hybrid_prior"):
        mp, ap = C.load_run("E1", "pinn", key)
        r = H.run_warm(ds, ap["D_hat"], arm.split("_")[1])
        start = C.field_scores(r["D_projected"], ds.ground_truth()[0])
        m = score(ds, r["D"], dict(stage_wall_s=r["wall_s"] + r["projection_s"], pinn_wall_s=mp["stage_wall_s"],
                                   total_wall_s=mp["stage_wall_s"] + r["wall_s"] + r["projection_s"],
                                   alpha=r["alpha"], alpha_start=r["alpha_start"], flag=r["flag"],
                                   total_iters=r["total_iters"], n_evals=r["n_evals"], path=r["path"],
                                   start_rel_L2_D=start["rel_L2_D"]))
        return m, dict(D_hat=r["D"].reshape(ds.N, ds.N))
    if arm == "reverse":
        ma, aa = C.load_run("E1", "adjoint", key)
        D, wall, dev = pinn_known(collection, key, epochs, init_D=aa["D_hat"])
        return score(ds, D, dict(stage_wall_s=wall, adjoint_wall_s=ma["stage_wall_s"],
                                 total_wall_s=ma["stage_wall_s"] + wall, epochs=epochs, device=dev)), dict(D_hat=D)
    raise ValueError(arm)


def run_job(exp, arm, collection, key, size):
    if exp == "E1":
        return run_E1(arm, collection, key, size)
    if exp == "E1c":
        ds = C.dataset(collection, key)
        D, wall, dev = pinn_known(collection, key, 3 * EPOCHS[size])
        return score(ds, D, dict(stage_wall_s=wall, total_wall_s=wall, epochs=3 * EPOCHS[size], device=dev)), dict(D_hat=D)
    if exp == "E1b":
        import hybrid_bc as HB
        return HB.run_job(arm, collection, key, EPOCHS[size])
    import experiments_more as XM
    return XM.run_job(exp, arm, collection, key, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--size", choices=("quick", "full"), default="quick")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th dataset starting at i")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    threads = int(os.environ.get("NOVELTY_THREADS", "0"))
    if threads:                          # several workers on one machine: keep their timings honest
        import torch
        torch.set_num_threads(threads)
    i, n = map(int, a.shard.split("/"))
    for exp in a.exp:
        rows = registry(exp, a.size)
        mine = [r for j, r in enumerate(rows) if j % n == i]
        print(f"{exp} ({a.size}): {len(rows)} datasets, this shard {len(mine)}, device {C.device_name()}", flush=True)
        for collection, key, arms in mine:
            for arm in arms:
                if a.arms and arm not in a.arms:
                    continue
                tag = f"{exp}/{arm}/{key}"
                if C.is_done(exp, arm, key):
                    print(f"  skip {tag}", flush=True)
                    continue
                if a.list:
                    print(f"  todo {tag}")
                    continue
                t0, c0 = time.time(), time.process_time()
                try:
                    m, arrays = run_job(exp, arm, collection, key, a.size)
                except Exception:
                    print(f"  FAILED {tag}\n{traceback.format_exc()}", flush=True)
                    continue
                m.update(exp=exp, arm=arm, key=key, collection=collection, size=a.size,
                         job_s=time.time() - t0, job_cpu_s=time.process_time() - c0, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
                C.save_run(exp, arm, key, m, **arrays)
                print(f"  {tag}: relL2 {m.get('rel_L2_D', float('nan')):.4f}  replay {m.get('replay_over_sigma', float('nan')):.2f}"
                      f"  total {m.get('total_wall_s', m['job_s']):.0f}s", flush=True)


if __name__ == "__main__":
    main()
