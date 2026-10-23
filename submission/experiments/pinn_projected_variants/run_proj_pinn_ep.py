"""Run the project's own PINN (v2/pinn_inversion.py, unchanged) on one dataset with the
settings of its benchmark suite (C04 weights; Tikhonov 2e-4 for tiers A/B/E, TV 3e-4 otherwise)."""
import sys, os
sys.argv = [sys.argv[0]] + sys.argv[1:]
key, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, "/home/claude/v2")
import torch; torch.set_num_threads(1)
import pinn_inversion as P
cfg = P.PINNConfig(data_root="/home/claude/v2", output_dir=out, exp_id=key)
if key.startswith(("A_", "B_", "E_")):
    cfg.reg_type, cfg.lambda_reg = "tikhonov", 2e-4
else:
    cfg.reg_type, cfg.lambda_reg = "tv", 3e-4
cfg.epochs = int(sys.argv[3]) if len(sys.argv) > 3 else cfg.epochs
P.train_single_experiment(cfg, run_tag=key)
