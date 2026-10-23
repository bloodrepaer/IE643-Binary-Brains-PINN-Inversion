"""
score_papers.py -- score the paper-faithful PINNs (pinn_papers.py) next to the adjoint, our original PINN,
the earlier paper-style PINN and the best constant D, all the same way.

Per (dataset, method) with a D_hat:
  relL2_D        ||D_hat - D|| / ||D||                       (our metric)
  thakur_err     sqrt(mean((D_hat - D)^2) / var(D))          (Thakur et al. eq. 7; a constant D scores >= 1)
  replay         RMS(CN replay of D_hat - data) / sigma      (Deliverable 6; true BC for H2u via the H2k pair;
                 for noise-free data: RMS / RMS(data) instead, column replay_kind says which)
Writes results_papers_scores.csv.
"""
import copy, glob, json, os
import numpy as np, pandas as pd
import adjoint_inversion as ai

SETS = {  # data root -> {method: result folder pattern (key substituted)}
    "datasets_v2_rk4_zero": {"Adjoint": "results_adjoint/{k}", "PINN (ours)": "results_pinn/{k}",
                             "Paper-style PINN (earlier)": "results_pinn_litb_main/{k}__s42",
                             "Thakur + IC/BC": "results_pp_main/{k}__thakur_icbc__s42",
                             "Zhang": "results_pp_main/{k}__zhang__s42"},
    "datasets_v2_pinnwin": {"Adjoint": "results_pinnwin_adjoint/{k}", "PINN (ours)": "results_pinnwin_pinn/{k}",
                            "Paper-style PINN (earlier)": "results_pinn_litb_pw/{k}__s42",
                            "Thakur + IC/BC": "results_pp_pw/{k}__thakur_icbc__s42",
                            "Zhang": "results_pp_pw/{k}__zhang__s42"},
    "datasets_v2_density_wideic": {"Adjoint": "results_density_wideic_adjoint/{k}",
                                   "Thakur (as published)": "results_pp_density_wideic/{k}__thakur__s42"},
    "datasets_v2_density": {"Adjoint": "results_density_adjoint/{k}",
                            "Thakur (as published)": "results_pp_density/{k}__thakur__s42",
                            "Zhang": "results_pp_density/{k}__zhang__s42"},
}
PP = {"datasets_v2_rk4_zero": "results_pp_main", "datasets_v2_pinnwin": "results_pp_pw",
      "datasets_v2_density": "results_pp_density", "datasets_v2_density_wideic": "results_pp_density_wideic"}

rows = []
for root, methods in SETS.items():
    man = {e["key"]: e for e in json.load(open(os.path.join(root, "manifest.json")))}
    keys = sorted({os.path.basename(p).split("__thakur")[0].split("__zhang")[0]
                   for p in glob.glob(f"{PP[root]}/*") if os.path.isdir(p)} |
                  {os.path.basename(p) for p in glob.glob(SETS[root]["Adjoint"].format(k="*")) if "density" in root and os.path.isdir(p)})
    for k in keys:
        e = man[k]; te = copy.deepcopy(e)
        if e.get("family") == "H2" and not e.get("bc_known", True):
            p = e["problem"].replace("H2u-", "H2k-", 1); te["problem"] = p; te["files"]["problem"] = f"problem/{p}.json"
        ds = ai.Dataset(root, te); sol = ai.AdjointSolver(ds); Dt, _ = ds.ground_truth()
        noisy = ds.sigma > 0
        def replay(D):
            W, *_ = sol.forward(np.asarray(D, float).ravel())
            r = np.sqrt(np.mean((sol.predict(W) - ds.u) ** 2))
            return float(r / ds.sigma) if noisy else float(r / np.sqrt(np.mean(ds.u ** 2)))
        def score(D, method):
            rows.append(dict(data_root=root, key=k, family=e.get("family", "main" if "rk4" in root else ("density-wideIC" if "wideic" in root else "density")),
                             n_readings=e["n_readings"], eta=e["eta"], method=method,
                             relL2_D=float(np.linalg.norm(D - Dt) / np.linalg.norm(Dt)),
                             thakur_err=float(np.sqrt(np.mean((D - Dt) ** 2) / np.var(Dt))),
                             replay=replay(D), replay_kind="RMS/sigma" if noisy else "RMS/RMS(data)",
                             note="replay without the hidden source" if e.get("family") == "H3" and not e.get("source_known", True) else ""))
        am = os.path.join(SETS[root]["Adjoint"].format(k=k), "metrics.json")
        if os.path.exists(am):
            Dc = json.load(open(am))["D_const"]; score(np.full_like(Dt, Dc), "Best constant D")
        for method, pat in methods.items():
            f = os.path.join(pat.format(k=k), "D_hat.npz")
            if os.path.exists(f):
                score(np.load(f)["D_hat"].astype(float), method)
        print(k, " | ".join(f"{r['method']}: {r['relL2_D']:.3f}/{r['thakur_err']:.2f}/{r['replay']:.3g}"
                            for r in rows if r["key"] == k), flush=True)
pd.DataFrame(rows).to_csv("results_papers_scores.csv", index=False)
print(len(rows), "rows -> results_papers_scores.csv")
