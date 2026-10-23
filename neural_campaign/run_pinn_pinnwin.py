"""
run_pinn_pinnwin.py -- run pinn_inversion.py, unchanged, on datasets_v2_pinnwin/.

Same settings the benchmark suite used for smooth tiers (Tikhonov, lambda_reg 2e-4,
1500 epochs, seed 42). Usage:
    python run_pinn_pinnwin.py                 # families H1, H3, H4 (zero BC; code applies as-is)
    python run_pinn_pinnwin.py --families H2 H5 --module pinn_inversion_bc
Results: results_pinnwin_pinn/<key>/ and results_pinnwin_pinn/summary.csv (appended per run).
"""
import argparse, json, os, sys
from dataclasses import asdict
import pandas as pd
import torch
import importlib

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="datasets_v2_pinnwin")
ap.add_argument("--out", default="results_pinnwin_pinn")
ap.add_argument("--families", nargs="*", default=["H1", "H3", "H4"])
ap.add_argument("--keys", nargs="*", help="run only these keys")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--module", default="pinn_inversion", help="pinn_inversion or pinn_inversion_bc (needed for H2/H5)")
a = ap.parse_args()
P = importlib.import_module(a.module)
torch.set_num_threads(a.threads)

man = json.load(open(os.path.join(a.data_root, "manifest.json")))
keys = a.keys or [e["key"] for e in man if e["family"] in a.families]
os.makedirs(a.out, exist_ok=True)
summ = os.path.join(a.out, "summary.csv")
done = set(pd.read_csv(summ).exp_id) if os.path.exists(summ) else set()
for i, k in enumerate(keys, 1):
    if k in done:
        continue
    print(f"[{i}/{len(keys)}] {k}", flush=True)
    cfg = P.PINNConfig(data_root=a.data_root, output_dir=a.out, exp_id=k,
                       reg_type="tikhonov", lambda_reg=2e-4, epochs=1500)
    res = P.train_single_experiment(cfg, run_tag=k)
    row = {kk: v for kk, v in res.items() if kk != "final_losses"}
    pd.DataFrame([row]).to_csv(summ, mode="a", header=not os.path.exists(summ), index=False)
print("done", flush=True)
