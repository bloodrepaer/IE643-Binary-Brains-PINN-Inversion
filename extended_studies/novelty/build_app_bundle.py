"""
build_app_bundle.py -- copy what the Streamlit app needs for the novelty demo into submission/app/demo/.

  datasets  slim copies (ground truth without the 129^2 x 41 temperature field) of
            - benchmark/datasets_v2_rk4_zero : the 20 known-physics datasets of E1
            - hidden_physics/datasets_v2_rk4_zero : the 14 hidden-wall (H2u) and hidden-source (H3u) datasets
  results   precomputed estimates of every method on those datasets, so the app can show the full comparison
            instantly: results/<key>/<arm>.npz (D_hat, plus f_hat / G_bd for the joint inversions) and
            results/<key>/metrics.json (the experiment's own scores, for reference; the app re-scores live).

Run from extended_studies/:   python run.py novelty/build_app_bundle.py
"""
import json
import shutil
from pathlib import Path

import numpy as np

import common as C
from run_novelty import DATASETS20, UNKNOWN_PHYSICS, UNKNOWN_SUFFIX

OUT = C.REPO / "submission" / "app" / "demo"
RES = C.EXT / "novelty" / "results_full"

# (experiment, arm) -> label shown in the app; order = column order
KNOWN_ARMS = [("E1", "adjoint", "CN adjoint"), ("E1", "pinn", "PINN"),
              ("E1", "hybrid_balance", "combined PINN→adjoint"), ("E1", "hybrid_path", "combined, warm start only"),
              ("E1", "hybrid_prior", "combined, PINN as prior")]
SOURCE_ARMS = [("E1", "adjoint", "CN adjoint (assumes f = 0)"), ("E1", "pinn", "PINN (boundary-aware)"),
               ("E1", "hybrid_balance", "combined PINN→adjoint (f = 0)"),
               ("E1s", "adjoint_src_pos", "joint: CN adjoint + source f ≥ 0"),
               ("E1s", "adjoint_truesrc", "bound: adjoint given the true source")]
WALL_ARMS = [("E1", "adjoint", "CN adjoint (assumes zero walls)"), ("E1b", "pinn_bc", "PINN (boundary-aware)"),
             ("E1b", "hybrid_bc_joint5", "combined: joint inversion from the PINN's walls"),
             ("E1b", "adjoint_joint1", "joint: CN adjoint + wall temperatures"),
             ("E1b", "adjoint_truebc", "bound: adjoint given the true walls")]


def copy_slim(collection, key, dst_root):
    root = C.EXT / C.COLLECTIONS[collection]
    e = dict(C.entry(collection, key))
    for rel in (e["files"]["sensors_csv"], e["files"]["sensors_npz"], e["files"]["problem"],
                e["files"]["problem"].replace(".json", ".npz"), e["files"]["problem"].replace(".json", "__ic.csv")):
        src = root / rel
        if src.exists():
            (dst_root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_root / rel)
    gt = np.load(root / e["files"]["ground_truth"], allow_pickle=True)
    keep = {k: gt[k] for k in gt.files if k != "u"}
    (dst_root / "ground_truth").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_root / e["files"]["ground_truth"], **keep)
    return e


def copy_results(key, arms):
    rows = []
    d = OUT / "results" / key
    d.mkdir(parents=True, exist_ok=True)
    for exp, arm, label in arms:
        src = RES / exp / arm / key
        if not (src / "D_hat.npz").exists():
            print("  missing", exp, arm, key)
            continue
        z = np.load(src / "D_hat.npz")
        arrays = {k: z[k].astype(np.float32) for k in z.files}
        np.savez_compressed(d / f"{exp}_{arm}.npz", **arrays)
        m = json.loads((src / "metrics.json").read_text())
        rows.append(dict(exp=exp, arm=arm, label=label, file=f"{exp}_{arm}.npz",
                         rel_L2_D=m.get("rel_L2_D"),
                         replay_over_sigma=m.get("replay_over_sigma", m.get("replay_own_walls_over_sigma")),
                         wall_s=m.get("total_wall_s", m.get("stage_wall_s"))))
    (d / "methods.json").write_text(json.dumps(rows, indent=1))


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    groups = {"benchmark": ("bench", DATASETS20, KNOWN_ARMS),
              "hidden_physics": ("pinnwin", [k + UNKNOWN_SUFFIX for k in UNKNOWN_PHYSICS], None)}
    for name, (collection, keys, arms) in groups.items():
        dst = OUT / name / "datasets_v2_rk4_zero"
        entries = []
        for key in keys:
            entries.append(copy_slim(collection, key, dst))
            a = arms or (SOURCE_ARMS if key.startswith("H3u") else WALL_ARMS)
            copy_results(key, a)
        (dst / "manifest.json").write_text(json.dumps(entries, indent=1))
        print(name, len(entries), "datasets")
    print("bundle size:", sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6, "MB")


if __name__ == "__main__":
    main()
