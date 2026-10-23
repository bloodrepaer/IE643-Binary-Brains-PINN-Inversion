"""Project PINN with ONE change: the interior taper on the D-network is removed, so D can
vary up to the boundary (forward_m = m_bg + m_net). Everything else is unchanged."""
import sys
key, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, "/home/claude/v2")
import torch; torch.set_num_threads(1)
import pinn_inversion as P
def forward_m(self, xy):
    return self.m_bg + self.m_net(xy / self.scale_xy)
P.InverseDiffusionPINN.forward_m = forward_m
cfg = P.PINNConfig(data_root="/home/claude/v2", output_dir=out, exp_id=key)
if key.startswith(("A_", "B_", "E_")):
    cfg.reg_type, cfg.lambda_reg = "tikhonov", 2e-4
else:
    cfg.reg_type, cfg.lambda_reg = "tv", 3e-4
P.train_single_experiment(cfg, run_tag=key)
