"""
check_problem_spec.py
---------------------
Confirms that problem/ contains everything an inversion needs.

For each sampled dataset it rebuilds the forward problem from the problem/
files ALONE -- IC from <name>__ic.csv, boundary values parsed with sympy from
<name>.json -- then runs Crank-Nicolson on the INVERSION grid with the TRUE D
(a stand-in for a perfect estimate) and compares with the sensor readings.

This is exactly the forward-replay metric with D_hat = D_true, so it also
gives the best replay ratio any method can reach on this benchmark:

    RMS(replay - u_clean) / sigma   should be << 1   (model/grid error floor)
    RMS(replay - u_noisy) / sigma   should be ~= 1   (only noise remains)

Usage:  python check_problem_spec.py datasets_v2_rk4_zero [--n 20]
"""
import argparse, csv, json, os
import numpy as np
import sympy
from scipy.interpolate import RegularGridInterpolator

from domain import Domain2D
from integrators import ForwardModel, BoundaryCondition, EDGES


def load_problem(folder, name):
    spec = json.load(open(os.path.join(folder, "problem", name + ".json")))
    ic = np.loadtxt(os.path.join(folder, "problem", name + "__ic.csv"), delimiter=",", skiprows=1)
    t = sympy.Symbol("t")
    bc = BoundaryCondition({e: sympy.lambdify(t, sympy.sympify(expr), "numpy")
                            for e, expr in spec["boundary_condition"]["edges"].items()})
    return spec, ic, bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--n", type=int, default=20, help="number of datasets to check")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(a.folder, "manifest.csv"))))
    picks = [rows[i] for i in np.linspace(0, len(rows) - 1, min(a.n, len(rows))).astype(int)]
    cache = {}
    print(f"{'dataset':58s} {'clean/sigma':>11} {'noisy/sigma':>11}")
    for r in picks:
        name = r["problem"]
        if name not in cache:
            spec, ic, bc = load_problem(a.folder, name)
            N = int(round(np.sqrt(len(ic))))
            dom = Domain2D(Nx=N, Ny=N)
            u0 = ic[:, 2].reshape(N, N)
            D = np.load(os.path.join(a.folder, "ground_truth", r["ground_truth"]))["D_inv"]
            T = spec["domain"]["T"]
            t, U, _ = ForwardModel(dom, D, bc=bc).solve(u0, T, 400, "cn", dt_target=T / 400)
            cache[name] = RegularGridInterpolator((t, dom.x, dom.y), U)
        d = np.loadtxt(os.path.join(a.folder, "sensors", r["key"] + ".csv"), delimiter=",", skiprows=1)
        clean = np.load(os.path.join(a.folder, "sensors", r["key"] + ".npz"))["u_clean"]
        q = d[:, [2, 0, 1]].copy()
        q[:, 0] = np.clip(q[:, 0], 0, cache[name].grid[0][-1])   # CSV rounding can overshoot T
        replay = cache[name](q)
        sigma = d[0, 4]
        if sigma == 0:
            print(f"{r['key']:58s} {'(noise-free) abs RMS':>11} {np.sqrt(np.mean((replay - clean)**2)):.2e}")
            continue
        print(f"{r['key']:58s} {np.sqrt(np.mean((replay - clean)**2))/sigma:11.3f} "
              f"{np.sqrt(np.mean((replay - d[:, 3])**2))/sigma:11.3f}")


if __name__ == "__main__":
    main()
