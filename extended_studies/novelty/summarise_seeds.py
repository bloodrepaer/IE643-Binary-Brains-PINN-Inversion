"""
summarise_seeds.py -- S1: how much of each PINN-based result is seed noise?

Seed 42 is the E1 run; S1 adds seeds 43-46 for the PINN and for hybrid_balance (the one hybrid that ever beat
both parents) on the 20 known-physics datasets, and for the boundary-aware PINN on the 6 hidden-source datasets.
The adjoint is deterministic, so it has no seed spread.

  python run.py novelty/summarise_seeds.py
"""
import json
import re
import statistics as st
from collections import defaultdict

import numpy as np

import common as C

TIE = 0.01


def load():
    runs = defaultdict(dict)                   # (key) -> {(arm, seed): relL2}
    for exp in ("E1", "S1", "E1s"):
        for p in (C.RESULTS / exp).glob("*/*/metrics.json"):
            m = json.loads(p.read_text())
            arm, seed = m["arm"], 42
            mm = re.match(r"^(pinn|hybrid_balance)_s(\d+)$", arm)
            if mm:
                arm, seed = mm.group(1), int(mm.group(2))
            elif exp == "E1s" or arm in ("adjoint", "hybrid_path", "hybrid_prior", "reverse"):
                seed = None                    # deterministic, or not seeded here
            runs[m["key"]][(arm, seed)] = m["rel_L2_D"]
    return runs


def per_seed(v, arm):
    return {s: x for (a, s), x in v.items() if a == arm and s is not None}


def main():
    runs = load()
    bench = sorted(k for k in runs if not k.startswith("H"))
    h3u = sorted(k for k in runs if k.startswith("H3u"))
    out = ["# S1: seed spread", ""]

    # --- known physics -------------------------------------------------------------------------
    rows, spreads_p, spreads_h, wins_both, pinn_beats_adj = [], [], [], [], []
    seeds_all = set()
    for k in bench:
        v = runs[k]
        P, H = per_seed(v, "pinn"), per_seed(v, "hybrid_balance")
        adj = v.get(("adjoint", None))
        if len(P) < 2 or adj is None:
            continue
        seeds_all |= set(P)
        spreads_p.append(max(P.values()) - min(P.values()))
        if len(H) >= 2:
            spreads_h.append(max(H.values()) - min(H.values()))
        common = sorted(set(P) & set(H))
        nb = sum(H[s] < (1 - TIE) * adj and H[s] < (1 - TIE) * P[s] for s in common)
        wins_both.append((k, nb, len(common)))
        pinn_beats_adj.append(sum(P[s] < (1 - TIE) * adj for s in P))
        rows.append((k, adj, min(P.values()), st.median(P.values()), max(P.values()),
                     min(H.values()) if H else np.nan, st.median(H.values()) if H else np.nan,
                     max(H.values()) if H else np.nan, len(P), nb, len(common)))
    if rows:
        out += [f"## Known physics: {len(rows)} datasets, PINN seeds {sorted(seeds_all)}", "",
                "| dataset | adjoint | PINN min | PINN median | PINN max | hybrid_balance min | median | max | beats both (seeds) |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            out.append(f"| {r[0].split('__')[0]} {'/'.join(r[0].split('__')[1:])} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | "
                       f"{r[5]:.3f} | {r[6]:.3f} | {r[7]:.3f} | {r[9]} of {r[10]} |")
        med_p = [st.median([runs[k][("pinn", s)] for k in bench if ("pinn", s) in runs[k]]) for s in sorted(seeds_all)]
        med_h = [st.median([runs[k][("hybrid_balance", s)] for k in bench if ("hybrid_balance", s) in runs[k]])
                 for s in sorted(seeds_all)]
        adj_med = st.median(runs[k][("adjoint", None)] for k in bench if ("adjoint", None) in runs[k])
        out += ["",
                f"- Median over datasets, per seed: PINN {', '.join(f'{x:.3f}' for x in med_p)}; "
                f"hybrid_balance {', '.join(f'{x:.3f}' for x in med_h)}; adjoint {adj_med:.3f} (deterministic).",
                f"- Per-dataset seed spread (max - min): PINN median {st.median(spreads_p):.3f}, max {max(spreads_p):.3f}; "
                f"hybrid_balance median {st.median(spreads_h):.3f}, max {max(spreads_h):.3f}.",
                f"- PINN beats the adjoint (by > 1 %) in {sum(pinn_beats_adj)} of {sum(len(per_seed(runs[k], 'pinn')) for k, *_ in rows)} "
                f"dataset-seed pairs.",
                f"- hybrid_balance beats both parents on a dataset: "
                + ", ".join(f"{k.split('__')[0]}{'' if k.endswith('Fo0.1__uniform8__n50__random__eta0.05') else ' [' + '/'.join(k.split('__')[1:]) + ']'} "
                            f"{nb}/{n}" for k, nb, n in wins_both if nb) + ".",
                f"- Datasets where hybrid_balance beats both parents for every seed: "
                f"{sum(1 for _, nb, n in wins_both if n and nb == n)}; for at least one seed: {sum(1 for _, nb, n in wins_both if nb)}.",
                ""]

    # --- hidden source ---------------------------------------------------------------------------
    hr = []
    for k in h3u:
        v = runs[k]
        P = per_seed(v, "pinn")
        js = v.get(("adjoint_src_pos", None))
        if len(P) < 2 or js is None:
            continue
        hr.append((k.split("__")[0], min(P.values()), st.median(P.values()), max(P.values()), js, len(P)))
    if hr:
        out += ["## Hidden source: PINN seed range vs the joint inversion (f >= 0)", "",
                "| dataset | PINN min | PINN median | PINN max | joint f >= 0 | seeds |", "|---|---|---|---|---|---|"]
        out += [f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]} |" for r in hr]
        out += ["", f"- The joint inversion beats the PINN's BEST seed on {sum(r[4] < (1 - TIE) * r[1] for r in hr)} of {len(hr)} datasets.", ""]
    text = "\n".join(out)
    (C.EXT / "novelty" / "summary").mkdir(exist_ok=True)
    (C.EXT / "novelty" / "summary" / "S1.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
