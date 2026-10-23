"""Classical baseline for the 20-dataset study: the project's own adjoint pipeline
(16x16 control grid, Tikhonov, L-BFGS-B, alpha by the discrepancy principle),
scored with the shared metrics. The chosen alpha is then given to the PINN."""
import json, os, resource, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from problem import load, Reduced, truth, metrics, solver_floor  # noqa: E402
from adjoint_inversion import AdjointSolver  # noqa: E402

key = sys.argv[1]
out = os.path.join(HERE, "results", "S20"); os.makedirs(out, exist_ok=True)
ds = load(key)
s = AdjointSolver(ds, control=16, reg="tikhonov")
r = s.invert(verbose=False)
red = Reduced(ds)
m = metrics(ds, red, r["D"])
rec = dict(key=key, method="adjoint", alpha=r["alpha"], flag=r["flag"], wall_s=r["wall_s"], n_evals=r["n_evals"],
           eps_u_solver=solver_floor(ds, red), sigma=ds.sigma, n_readings=len(ds.u),
           peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, **m)
json.dump(rec, open(os.path.join(out, f"adjoint_{key}.json"), "w"), indent=1)
np.savez_compressed(os.path.join(out, f"adjoint_{key}.npz"), D_hat=r["D"].reshape(ds.N, ds.N))
print(f"adjoint {key}: alpha={r['alpha']:.3g} eps_D={m['eps_D']:.4f} rms/sig={m['rms_replay_over_sigma']:.3f} {r['wall_s']:.0f}s", flush=True)
