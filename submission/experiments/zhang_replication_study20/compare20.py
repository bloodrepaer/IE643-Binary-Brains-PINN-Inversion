"""Score classical adjoint vs project PINN on the 20 datasets with identical metrics
(both D_hat pushed through the same CN solver) and list where the PINN wins."""
import csv, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from problem import load, Reduced, metrics, truth  # noqa: E402

PINN_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/v2/pinn_proj/out"
TAG = sys.argv[2] if len(sys.argv) > 2 else "pinn"
keys = open(os.path.join(HERE, "datasets20.txt")).read().split()
rows = []
for k in keys:
    a = json.load(open(os.path.join(HERE, "results", "S20", f"adjoint_{k}.json")))
    pd_ = os.path.join(PINN_DIR, k)
    if not os.path.exists(os.path.join(pd_, "D_hat.npz")):
        continue
    D = np.load(os.path.join(pd_, "D_hat.npz"))["D_hat"].astype(float)
    pm = json.load(open(os.path.join(pd_, "metrics.json")))
    ds = load(k); red = Reduced(ds)
    p = metrics(ds, red, D)
    tr = truth(ds)
    rows.append(dict(key=k, n_readings=a["n_readings"], alpha=a["alpha"],
                     adj_eps_D=a["eps_D"], pinn_eps_D=p["eps_D"],
                     adj_rmse_logD=a["rmse_logD"], pinn_rmse_logD=p["rmse_logD"],
                     adj_eps_u=a["eps_u"], pinn_eps_u=p["eps_u"], eps_u_solver=a["eps_u_solver"],
                     adj_rms_sig=a["rms_replay_over_sigma"], pinn_rms_sig=p["rms_replay_over_sigma"],
                     pinn_chi2_unet=pm.get("chi2_red_u_net"),
                     const_eps_D=float(np.linalg.norm(np.exp(red.best_constant()) - tr["D"]) / np.linalg.norm(tr["D"])),
                     adj_wall_s=a["wall_s"], pinn_wall_s=pm["elapsed_sec"],
                     winner_D="PINN" if p["eps_D"] < a["eps_D"] else "adjoint",
                     ratio_D=p["eps_D"] / a["eps_D"]))
out = os.path.join(HERE, "results", "S20", f"comparison_{TAG}.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{'dataset':52s} {'adj eps_D':>9s} {'PINN eps_D':>10s} {'const':>6s} {'ratio':>6s} {'adj logD':>8s} {'PINN logD':>9s} {'adj rms/s':>9s} {'PINN rms/s':>10s}")
for r in rows:
    print(f"{r['key']:52s} {r['adj_eps_D']:9.3f} {r['pinn_eps_D']:10.3f} {r['const_eps_D']:6.3f} {r['ratio_D']:6.2f} "
          f"{r['adj_rmse_logD']:8.3f} {r['pinn_rmse_logD']:9.3f} {r['adj_rms_sig']:9.2f} {r['pinn_rms_sig']:10.2f}")
wins = [r["key"] for r in rows if r["winner_D"] == "PINN"]
print(f"\nPINN wins on relL2(D): {len(wins)}/{len(rows)}", wins)
print("PINN wins on log-D RMSE:", sum(r["pinn_rmse_logD"] < r["adj_rmse_logD"] for r in rows))
print("PINN wins on eps_u:", sum(r["pinn_eps_u"] < r["adj_eps_u"] for r in rows))
print("median ratio PINN/adjoint eps_D:", float(np.median([r["ratio_D"] for r in rows])))
print("PINN beats constant-D guess:", sum(r["pinn_eps_D"] < r["const_eps_D"] for r in rows),
      " adjoint beats constant:", sum(r["adj_eps_D"] < r["const_eps_D"] for r in rows))
print("wall s: adjoint median", np.median([r["adj_wall_s"] for r in rows]), " PINN median", np.median([r["pinn_wall_s"] for r in rows]))
print("written", out)
