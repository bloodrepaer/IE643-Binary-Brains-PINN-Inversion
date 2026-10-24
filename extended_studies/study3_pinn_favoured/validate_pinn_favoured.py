"""
validate_pinn_favoured.py
-------------------------
Checks datasets_v2_pinnwin/ before any inversion is run:
  1. every dataset loads through adjoint_inversion.Dataset (format compatible);
  2. the problem stem the PINN loader derives from the key exists;
  3. "truth replay": the TRUE D pushed through the model each inversion is TOLD
     (problem/*.json: assumed BC / f = 0 when unknown), compared with the noisy
     data as RMS/sigma. Controls should sit near 1; 'unknown BC/source' datasets
     should sit well above 1 -- that gap is the misspecification a method must
     cope with. Written to datasets_v2_pinnwin/validation.csv.
"""
import csv, json, os, sys
import numpy as np
import adjoint_inversion as ai

ROOT = sys.argv[1] if len(sys.argv) > 1 else "datasets_v2_pinnwin"
man = json.load(open(os.path.join(ROOT, "manifest.json")))
rows = []
for e in man:
    ds = ai.Dataset(ROOT, e)
    stem = "__".join(e["key"].split("__")[:2])
    assert os.path.exists(os.path.join(ROOT, "problem", stem + ".json")), stem
    sol = ai.AdjointSolver(ds)
    Dt, u_clean = ds.ground_truth()
    W, *_ = sol.forward(Dt.ravel())
    pred = sol.predict(W)
    rms_told = float(np.sqrt(np.mean((pred - ds.u) ** 2)) / ds.sigma)
    rms_noise = float(np.sqrt(np.mean((u_clean - ds.u) ** 2)) / ds.sigma)
    rows.append(dict(key=e["key"], family=e["family"], role=e["role"], bc_true=e["bc_true"],
                     bc_known=e["bc_known"], source_q=e["source_q"], source_known=e["source_known"],
                     n_readings=e["n_readings"], eta=e["eta"], Fo=e["Fo"], n_steps_cn=sol.disc.n_steps,
                     truth_replay_over_sigma=round(rms_told, 3), noise_only_over_sigma=round(rms_noise, 3)))
    print(f"{e['key'][:70]:70s} {e['role']:8s} truth-under-told-model {rms_told:7.2f}  (noise alone {rms_noise:.2f})", flush=True)
with open(os.path.join(ROOT, "validation.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(f"\n{len(rows)} datasets validated -> {ROOT}/validation.csv")
