"""
score_pinnwin.py -- score the adjoint and the PINN on datasets_v2_pinnwin/ the same way.

For every dataset and method with a D_hat.npz:
  relL2_D            ||D_hat - D_true|| / ||D_true||          (65^2 inversion grid)
  replay_true        RMS(u_replay - data)/sigma, D_hat through the CN solver under the TRUE boundary condition
                     (for H2u the BC comes from the paired H2k problem file). H3u: the CN solver has no source
                     term, so this replay uses f = 0 and is flagged (replay_note).
  truth_replay       the same number for the TRUE D: the floor a perfect D would reach
  const_relL2        the best constant D (adjoint's stage-0 result), the "learned nothing" line
Writes results_pinnwin_scores.csv.
"""
import copy, json, os, sys
import numpy as np, pandas as pd
import adjoint_inversion as ai

ROOT = sys.argv[1] if len(sys.argv) > 1 else "datasets_v2_pinnwin"
RES = ({"Adjoint": "results_pinnwin_adjoint", "PINN": "results_pinnwin_pinn"} if ROOT == "datasets_v2_pinnwin" else
       {"Adjoint": "results_pinnwin2_adjoint", "Adjoint (32x32 control)": "results_pinnwin2_adjoint_c32", "PINN": "results_pinnwin2_pinn"})
OUTCSV = "results_pinnwin_scores.csv" if ROOT == "datasets_v2_pinnwin" else "results_pinnwin2_scores.csv"
man = json.load(open(os.path.join(ROOT, "manifest.json")))
rows, cache = [], {}
for e in man:
    true_e = copy.deepcopy(e)
    note = ""
    if e["family"] in ("H2", "H7") and not e["bc_known"]:
        prob = e["problem"].replace("H2u-", "H2k-", 1).replace("H7u-", "H7k-", 1)
        true_e["problem"] = prob
        true_e["files"]["problem"] = f"problem/{prob}.json"
    if e["family"] == "H3" and not e["source_known"]:
        note = "source omitted from replay (f = 0)"
    ds = ai.Dataset(ROOT, true_e)
    sol = ai.AdjointSolver(ds)
    D_true, _ = ds.ground_truth()
    def replay(D):
        W, *_ = sol.forward(np.asarray(D, float).ravel())
        return float(np.sqrt(np.mean((sol.predict(W) - ds.u) ** 2)) / ds.sigma)
    truth = replay(D_true)
    const = np.nan
    am = os.path.join(RES["Adjoint"], e["key"], "metrics.json")
    if os.path.exists(am):
        const = json.load(open(am)).get("rel_L2_D_constant_guess", np.nan)
    for method, folder in RES.items():
        f = os.path.join(folder, e["key"], "D_hat.npz")
        if not os.path.exists(f):
            continue
        if method == "PINN" and "PINN" in [r["method"] for r in rows if r["key"] == e["key"]]:
            continue
        D = np.load(f)["D_hat"].astype(float)
        mfile = os.path.join(folder, e["key"], "metrics.json")
        m = json.load(open(mfile)) if os.path.exists(mfile) else {}
        rows.append(dict(key=e["key"], family=e["family"], role=e["role"], pair=e["pair"], base_tier=e["base_tier"],
                         n_sensors=e["n_sensors"], eta=e["eta"], Fo=e["Fo"], bc_true=e["bc_true"],
                         bc_known=e["bc_known"], source_q=e["source_q"], source_known=e["source_known"],
                         method=method, relL2_D=float(np.linalg.norm(D - D_true) / np.linalg.norm(D_true)),
                         const_relL2=const, replay_true=replay(D), truth_replay=truth, replay_note=note,
                         mean_ratio=float(D.mean() / D_true.mean()),
                         wall_s=m.get("wall_s", m.get("elapsed_sec", np.nan)), flag=m.get("flag", "")))
    print(f"{e['key'][:66]:66s} " + "  ".join(f"{r['method']} {r['relL2_D']:.3f}/{r['replay_true']:.2f}"
                                             for r in rows if r["key"] == e["key"]), flush=True)
pd.DataFrame(rows).to_csv(OUTCSV, index=False)
print(f"{len(rows)} rows -> {OUTCSV}")
