"""
make_density_datasets.py -- our A_smooth problem (Fo 0.1, zero walls) observed as densely as in
Thakur et al. 2024, then thinned the way their Table 2 thins it.

Thakur et al. use every node of a 500x500 simulation at 100 time steps, noise-free, and then cut the
spatial points to 25 000, 2500, 250 and 25; their D error stays ~5-6 % down to 250 points and fails at 25.
Here: noise-free readings at the 40 uniform output times (t = T/40 ... T) on nested random subsets of the
3969 interior nodes of the 65^2 inversion grid: 3969 (full), 400, 100, 25 sensors.
Same RK4 129^2 ground truth, IC, walls and file format as datasets_v2_rk4_zero, so the PINN loaders and the
adjoint read them unchanged.  Output: datasets_v2_density/
"""
import csv, json, os, shutil
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import generate_dataset_v2 as G
from domain import Domain2D, get_D_field
from integrators import ForwardModel, BoundaryCondition

import sys
SRC, OUT = "datasets_v2_rk4_zero", "datasets_v2_density"
BASE_KEY = "A_smooth__Fo0.1__uniform8__n50__random__eta0.05"
COUNTS = (3969, 400, 100, 25)
WIDE = "--wide-ic" in sys.argv          # same bumps with width 0.12 instead of 0.05 / 0.06
if WIDE:
    OUT, COUNTS = "datasets_v2_density_wideic", (3969, 400)
    G.IC_BUMPS = [dict(b, s=0.12) for b in G.IC_BUMPS]

for sub in ("problem", "ground_truth", "sensors"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
base = [m for m in json.load(open(os.path.join(SRC, "manifest.json"))) if m["key"] == BASE_KEY][0]
import glob
for f in ([] if WIDE else glob.glob(os.path.join(SRC, "problem", base["problem"] + "*"))) + [os.path.join(SRC, base["files"]["ground_truth"])]:
    shutil.copy(f, f.replace(SRC, OUT, 1))                       # problem json + its IC/BC csv files, ground truth

dom, dom_inv = Domain2D(Nx=129, Ny=129), Domain2D(Nx=65, Ny=65)
cls, kw = G.TIERS["A_smooth"]
D = get_D_field(dom, cls, **kw)
T = 0.1 / float(D.mean())
fm = ForwardModel(dom, D, stencil="5pt", bc=BoundaryCondition())
t, U, info = fm.solve(G.initial_condition(dom), T, G.N_OUT, "rk4")
print(f"RK4 solve: {info['n_steps']} steps, T = {T:.6f}")
if WIDE:                                                          # the IC changed: write the problem files afresh
    G.write_problem(OUT, base["problem"], dom, dom_inv, 2, T, 0.1, BoundaryCondition(), "zero", t)
ks = G.PROTOCOLS["uniform40"]()
nodes = np.c_[dom_inv.X[1:-1, 1:-1].ravel(), dom_inv.Y[1:-1, 1:-1].ravel()]
order = np.random.default_rng(643).permutation(len(nodes))           # nested subsets
manifest = []
for n in COUNTS:
    xy = nodes[np.sort(order[:n])]
    u = np.concatenate([RegularGridInterpolator((dom.x, dom.y), U[k])(xy) for k in ks])
    key = f"A_smooth{'-wideic' if WIDE else ''}__Fo0.1__uniform40__n{n}__densitysweep__eta0"
    data = dict(x=np.tile(xy[:, 0], len(ks)), y=np.tile(xy[:, 1], len(ks)), t=np.repeat(t[ks], n), u=u, u_clean=u,
                sigma=np.zeros_like(u), sensor_id=np.tile(np.arange(n), len(ks)), time_index=np.repeat(ks, n))
    np.savez_compressed(os.path.join(OUT, "sensors", key + ".npz"), **data)
    with open(os.path.join(OUT, "sensors", key + ".csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["x", "y", "t", "u", "sigma"])
        for row in zip(data["x"], data["y"], data["t"], data["u"], data["sigma"]):
            w.writerow([f"{v:.8g}" for v in row])
    e = dict(base); e.update(key=key, protocol="uniform40", n_times=len(ks), n_sensors=n, sensor_set=str(n),
                             layout="densitysweep", n_readings=len(u), eta=0.0, sigma=0.0, snr_db=float("inf"),
                             readout_interval=T / len(ks), observation_times=[round(float(v), 8) for v in t[ks]],
                             files={**base["files"], "sensors_csv": f"sensors/{key}.csv", "sensors_npz": f"sensors/{key}.npz"})
    manifest.append(e)
    print(key, len(u), "readings")
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1, default=float)
