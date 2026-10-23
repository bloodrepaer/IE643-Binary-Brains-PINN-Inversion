"""
run_case.py -- run ONE estimator on ONE dataset/seed and save everything.

  python run_case.py --exp E1 --method adjoint --rep grid --key <key> --seed 0 --gamma 0
  python run_case.py --exp E1 --method pinn    --rep nn   --key <key> --seed 3 --gamma 0
  python run_case.py --exp E2 --method eki              --key <key> --seed 1
  python run_case.py --exp E3 --method adjoint --rep nn --key <key> --seed 0 \
        --init-from results/E1/pinn_nn_<key>_s0.npz --stop-eps 0.05

Writes results/<exp>/<tag>.json (metrics, timings, peak RSS, logs) and .npz (D_hat, theta).
"""
import argparse, json, os, resource, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from problem import load, Reduced, truth, metrics, solver_floor, bilinear_P  # noqa: E402
import methods  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--method", required=True, choices=("adjoint", "pinn", "eki"))
    ap.add_argument("--rep", default="grid", choices=("grid", "nn"))
    ap.add_argument("--key", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--n-adam", type=int, default=1000)
    ap.add_argument("--n-outer", type=int, default=10)
    ap.add_argument("--n-inner", type=int, default=300)
    ap.add_argument("--n-pde", type=int, default=4000)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth-u", type=int, default=3)
    ap.add_argument("--variant", default="ssbroyden", choices=("ssbroyden", "bfgs"))
    ap.add_argument("--n-ens", type=int, default=200)
    ap.add_argument("--n-iter", type=int, default=10)
    ap.add_argument("--init-from", default=None, help="npz of a previous run: warm start (E3)")
    ap.add_argument("--stop-eps", type=float, default=None, help="stop once eps_D <= this (E3)")
    ap.add_argument("--max-time", type=float, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()

    tag = a.tag or f"{a.method}_{a.rep if a.method != 'eki' else 'grid'}_{a.key}_g{a.gamma:g}_s{a.seed}"
    od = os.path.join(a.out, a.exp)
    os.makedirs(od, exist_ok=True)
    t_all = time.perf_counter()
    ds = load(a.key, a.seed)
    red = Reduced(ds)
    tr = truth(ds)
    t_setup = time.perf_counter()
    m0 = red.best_constant()                       # shared data-only initial guess
    t_m0 = time.perf_counter() - t_setup

    theta_init = None
    if a.init_from:
        src = np.load(a.init_from)
        if a.method == "adjoint" and a.rep == "nn":
            theta_init = src["theta_m"]            # the PINN's parameter network, as is
        elif a.method == "adjoint" and a.rep == "grid":
            # least-squares projection of the PINN's log D onto the control grid
            P = bilinear_P(ds.N, a.K)
            theta_init = np.linalg.lstsq(P.toarray(), np.log(src["D_hat"]).ravel(), rcond=None)[0]

    if a.method == "adjoint":
        r = methods.adjoint(ds, red, a.rep, a.gamma, m0, a.seed, K=a.K, maxiter=a.maxiter,
                            theta_init=theta_init, stop_eps=a.stop_eps, max_time=a.max_time,
                            variant=a.variant, tr=tr)
        extra = {}
    elif a.method == "pinn":
        p = methods.PINN(ds, red, a.rep, a.gamma, m0, a.seed, K=a.K, width=a.width, depth_u=a.depth_u,
                         n_pde=a.n_pde)
        r = p.train(n_adam=a.n_adam, n_outer=a.n_outer, n_inner=a.n_inner, tr=tr, variant=a.variant,
                    max_time=a.max_time)
        extra = dict(eps_u_state_net=p.state_error(r["theta"], tr), final_parts=r["final_parts"],
                     outer=r["outer"])
        r["theta_m"] = r["theta"][p.nu:]
    else:
        r = methods.eki(ds, red, m0, a.seed, K=a.K, n_ens=a.n_ens, n_iter=a.n_iter, tr=tr)
        extra = {}

    met = metrics(ds, red, r["D"], tr)
    out = dict(tag=tag, exp=a.exp, method=a.method, rep=(a.rep if a.method != "eki" else "grid"), key=a.key,
               seed=a.seed, gamma=a.gamma, sigma=ds.sigma, n_readings=len(ds.u), m0=m0, D0=float(np.exp(m0)),
               wall_s=r["wall_s"], wall_m0_s=t_m0, wall_total_s=time.perf_counter() - t_all,
               nit=r["nit"], nfev=r.get("nfev"), reason=r.get("reason"), n_params=r["n_params"],
               peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
               eps_u_solver=solver_floor(ds, red, tr), warm_start=a.init_from, stop_eps=a.stop_eps,
               args=vars(a), **met, **extra,
               log=[list(map(lambda v: v if isinstance(v, str) else float(v), row)) for row in r["log"]])
    json.dump(out, open(os.path.join(od, tag + ".json"), "w"), indent=1)
    np.savez_compressed(os.path.join(od, tag + ".npz"), D_hat=r["D"], theta=r["theta"],
                        theta_m=r.get("theta_m", r["theta"]))
    print(f"{tag}: eps_D={met['eps_D']:.4f} rmse_logD={met['rmse_logD']:.4f} eps_u={met['eps_u']:.3e} "
          f"rms/sig={met['rms_replay_over_sigma']:.3f} nit={r['nit']} wall={r['wall_s']:.1f}s "
          f"rss={out['peak_rss_gb']:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
