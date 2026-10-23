"""
run_pinn_v2_batch.py -- run pinn_inversion_v2.py over a job list and append to one summary CSV.

Variants (each adds one fix to the previous):
  V0  original behaviour (taper on D at walls, Fo-based init, lambdas 40/20/10/20, Morozov data damping)
  V1  + D free at walls + best-constant init (same stage 0 as the adjoint)
  V2  + equal loss weights (1/1/1/1) + plain data misfit
  V3  + L-BFGS polish (300 iterations, fixed batch)
  V4  + higher-frequency Fourier features for m (scale 1.0)
All variants: Tikhonov 2e-4 on m (the adjoint's regulariser family), 1500 Adam epochs.

Usage: python run_pinn_v2_batch.py jobs.json results_dir
jobs.json: [{"data_root": "...", "key": "...", "variant": "V3", "seed": 42}, ...]
"""
import json, os, sys
import pandas as pd
import torch
import pinn_inversion_v2 as P

VARIANTS = {
    "V0": dict(),
    "V1": dict(m_wall_taper=False, m_init="best_constant"),
    "V2": dict(m_wall_taper=False, m_init="best_constant", lambda_data=1.0, lambda_pde=1.0, lambda_bc=1.0,
               lambda_ic=1.0, morozov_damping=False),
    "V3": dict(m_wall_taper=False, m_init="best_constant", lambda_data=1.0, lambda_pde=1.0, lambda_bc=1.0,
               lambda_ic=1.0, morozov_damping=False, lbfgs_iters=300),
    "V4": dict(m_wall_taper=False, m_init="best_constant", lambda_data=1.0, lambda_pde=1.0, lambda_bc=1.0,
               lambda_ic=1.0, morozov_damping=False, lbfgs_iters=300, m_fourier_scale=1.0),
}

if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("PINN_THREADS", "1")))
    summ = os.path.join(out, "summary.csv")
    done = set()
    if os.path.exists(summ):
        d = pd.read_csv(summ)
        done = set(zip(d.exp_id, d.variant, d.seed))
    for i, j in enumerate(jobs, 1):
        tag = (j["key"], j["variant"], j.get("seed", 42))
        if tag in done:
            continue
        print(f"[{i}/{len(jobs)}] {j['variant']} seed {tag[2]} {j['key']}", flush=True)
        cfg = P.PINNConfig(data_root=j["data_root"], output_dir=out, exp_id=j["key"], reg_type="tikhonov",
                           lambda_reg=2e-4, epochs=1500, seed=tag[2], **VARIANTS[j["variant"]])
        run_tag = f"{j['key']}__{j['variant']}__s{tag[2]}"
        res = P.train_single_experiment(cfg, run_tag=run_tag)
        row = {k: v for k, v in res.items() if k != "final_losses"}
        row.update(variant=j["variant"], seed=tag[2], data_root=j["data_root"], run_dir=run_tag)
        pd.DataFrame([row]).to_csv(summ, mode="a", header=not os.path.exists(summ), index=False)
    print("done", flush=True)
