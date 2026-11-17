"""
summarise_novelty.py -- tables and figures from novelty/results/ (metrics.json files only).

  python run.py novelty/summarise_novelty.py            # every experiment that has results
  python run.py novelty/summarise_novelty.py --exp E1

Writes novelty/summary/<exp>_table.csv, <exp>.md and figures; prints the Markdown.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

OUT = C.EXT / "novelty" / "summary"
TIE = 0.01          # a win needs relL2 more than 1 % lower; within 1 % counts as a tie
COL = {"adjoint": "#2B6A9E", "pinn": "#C0461C", "hybrid_path": "#7A5195", "hybrid_balance": "#BC5090",
       "hybrid_prior": "#2E8B57", "reverse": "#999999", "pinn_T": "#E6A100"}


def load(exp):
    rows = [json.loads(p.read_text()) for p in sorted((C.RESULTS / exp).glob("*/*/metrics.json"))]
    return pd.DataFrame(rows) if rows else None


def md_table(df, floatfmt=".3f"):
    return df.to_markdown(floatfmt=floatfmt)


def flat(t):
    t = t.copy()
    t.index = [f"{short(k)} ({g.split()[0]})" for g, k in t.index]
    t.index.name = "dataset (physics)"
    return t


def short(key):
    parts = key.split("__")
    std = parts[1:] == ["Fo0.1", "uniform8", "n50", "random", "eta0.05"]
    rest = parts[2:] if parts[1:2] == ["Fo0.1"] else parts[1:]
    return parts[0] + ("" if std else "  [" + "/".join(rest) + "]")


# ---------------------------------------------------------------------------------------------
def summarise_E1(df):
    df = df.copy()
    df["group"] = np.where(df["collection"] == "pinnwin", "unknown physics", "known physics")
    err = df.pivot_table(index=["group", "key"], columns="arm", values="rel_L2_D")
    cost = df.pivot_table(index=["group", "key"], columns="arm", values="total_wall_s")
    arms = [a for a in ["adjoint", "pinn", "hybrid_path", "hybrid_balance", "hybrid_prior", "reverse"] if a in err]
    err, cost = err[arms], cost[arms]
    lines = ["# E1: hybrid PINN -> adjoint", "",
             "relL2(D), lower is better. A win = more than 1 % lower relL2; within 1 % = tie. Hybrid cost = PINN training + adjoint stage; reverse cost = adjoint + PINN.", ""]
    rows = []
    for g, sub in err.groupby(level=0):
        best_alone = sub[["adjoint", "pinn"]].min(axis=1)
        for a in arms:
            if a in ("adjoint", "pinn"):
                continue
            ok = sub[a].notna()
            x = sub[a][ok]
            rows.append(dict(group=g, arm=a, n=int(ok.sum()),
                             beats_adjoint=int((x < (1 - TIE) * sub["adjoint"][ok]).sum()),
                             ties_adjoint=int((abs(x / sub["adjoint"][ok] - 1) <= TIE).sum()),
                             beats_pinn=int((x < (1 - TIE) * sub["pinn"][ok]).sum()),
                             beats_both=int((x < (1 - TIE) * best_alone[ok]).sum()),
                             median_rel_L2=float(sub[a][ok].median()),
                             median_adjoint=float(sub["adjoint"][ok].median()),
                             median_pinn=float(sub["pinn"][ok].median()),
                             median_cost_vs_adjoint=float((cost.loc[sub.index, a] / cost.loc[sub.index, "adjoint"])[ok].median())))
    wins = pd.DataFrame(rows)
    lines += ["## Win counts and medians", "", wins.to_markdown(index=False, floatfmt=".3f"), "",
              "## relL2(D) per dataset", "", md_table(flat(err)), "",
              "## Total wall time (s) per dataset", "", md_table(flat(cost), ".0f"), ""]
    if "hybrid_path" in df["arm"].values:
        it = df.pivot_table(index="key", columns="arm", values="total_iters")
        keep = [a for a in ["adjoint", "hybrid_path", "hybrid_balance", "hybrid_prior"] if a in it]
        lines += ["## L-BFGS-B iterations of the adjoint stage", "", md_table(it[keep].rename(index=short), ".0f"), ""]
    # figure: hybrid vs adjoint and vs PINN
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, ref in zip(axes, ["adjoint", "pinn"]):
        for a in [x for x in arms if x not in ("adjoint", "pinn")]:
            for g, mk in (("known physics", "o"), ("unknown physics", "^")):
                s = err.loc[g] if g in err.index.get_level_values(0) else None
                if s is None or a not in s or s[a].isna().all():
                    continue
                ax.scatter(s[ref], s[a], marker=mk, color=COL[a], s=36, alpha=0.85,
                           label=f"{a} ({g.split()[0]})")
        lim = [0.8 * float(np.nanmin(err.values)), float(np.nanmax(err.values)) * 1.25]
        ax.plot(lim, lim, color="0.5", lw=0.8)
        ax.set(xlabel=f"{ref} relL2(D)", ylabel="combined method relL2(D)", xlim=lim, ylim=lim,
               xscale="log", yscale="log", title=f"below the line: combined beats {ref}")
        ax.grid(alpha=0.3)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(OUT / "E1_scatter.png", dpi=170); plt.close(fig)
    return lines, err


def summarise_E2(df, exp):
    lines = [f"# {exp}: temperature-dependent diffusivity", ""]
    idx = ["key"]
    t = df.pivot_table(index="key", columns="arm", values="rel_L2_D")
    lines += ["relL2 against the reference field D(x, y, u = 0) (the true D for E2a).", "", md_table(t.rename(index=short)), ""]
    if "pinn_T" in df["arm"].values:
        g = df[df.arm == "pinn_T"].set_index("key")[[c for c in ["strength_S", "g_at_1", "g_rmse", "g_max_abs",
                                                                  "replay_nonlinear_over_sigma"] if c in df]]
        lines += ["## Third network g(u) (true s(1) = strength S)", "", md_table(g.rename(index=short)), ""]
    if exp == "E2b":
        df = df.copy()
        df["field"] = df["key"].str.split("-arr").str[0]
        df["setting"] = df["key"].str.split("__").str[3] + "/" + df["key"].str.split("__").str[5]
        cases = sorted(df.groupby(["field", "setting"]).groups)
        fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 3.8), squeeze=False)
        for ax, (f, st) in zip(axes[0], cases):
            s = df[(df.field == f) & (df.setting == st)]
            for a in ["adjoint", "pinn", "pinn_T"]:
                q = s[s.arm == a].sort_values("strength_S")
                if len(q):
                    ax.plot(q["strength_S"], q["rel_L2_D"], "o-", color=COL[a], label=a)
            ax.set(title=f"{f}, {st}", xlabel="strength S = log D-ratio at u = 1", ylabel="relL2 vs D0")
            ax.grid(alpha=0.3)
        axes[0][0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "E2b_error_vs_strength.png", dpi=170); plt.close(fig)
        # g curves
        import arrhenius as AR
        sub = df[df.arm == "pinn_T"]
        if len(sub):
            fig, ax = plt.subplots(figsize=(5.2, 4))
            uu = np.linspace(0, 1, 101)
            for S in sorted(sub["strength_S"].unique()):
                ax.plot(uu, AR.s_of(uu, S) if S else 0 * uu, "k--", lw=1)
                for _, r in sub[sub.strength_S == S].iterrows():
                    _, arr = C.load_run(exp, "pinn_T", r["key"])
                    if "g_curve" in arr:
                        ax.plot(arr["g_curve"][:, 0], arr["g_curve"][:, 1], alpha=0.7)
            ax.set(xlabel="u (scaled temperature rise)", ylabel="log D(u) - log D(0)",
                   title="third network g(u) vs true s(u) (dashed)")
            ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(OUT / "E2b_g_curves.png", dpi=170); plt.close(fig)
    return lines, t


def summarise_E3a(df):
    lines = ["# E3a: sensor-split validation", ""]
    sel = df[df.arm == "pinn_selected"].set_index("key")
    rows = []
    for key in sorted(df.key.unique()):
        d = df[df.key == key].set_index("arm")
        r = dict(dataset=short(key))
        for a in ["pinn_default", "pinn_selected", "adjoint_dp", "adjoint_cv"]:
            if a in d.index:
                r[a] = d.loc[a, "rel_L2_D"]
        if key in sel.index:
            r["selected_config"] = sel.loc[key, "selected"]
            r["oracle_80"] = sel.loc[key, "oracle_rel_L2_D_80"]
            r["oracle_config"] = sel.loc[key, "oracle_config_80"]
        rows.append(r)
    t = pd.DataFrame(rows).set_index("dataset")
    lines += ["relL2(D). oracle_80 = best of the 10 settings by the TRUE D (trained on 80 %), for reference only.", "",
              md_table(t), ""]
    return lines, t


def summarise_E3b(df):
    lines = ["# E3b: sensor placement at equal budget", ""]
    df = df.copy()
    t = df.pivot_table(index=["tier", "budget"], columns=["arm", "layout"], values="rel_L2_D")
    lines += ["relL2(D).", "", md_table(t), ""]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, a in zip(axes, ["adjoint", "pinn"]):
        for lay, ls in (("random", ":"), ("grid", "--"), ("designed", "-")):
            q = df[(df.arm == a) & (df.layout == lay)].groupby("budget")["rel_L2_D"].median()
            if len(q):
                ax.plot(q.index, q.values, ls, marker="o", label=lay)
        ax.set(title=f"{a}: median over fields", xlabel="sensors", ylabel="relL2(D)")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "E3b_layouts.png", dpi=170); plt.close(fig)
    return lines, t


def summarise_simple(df, exp, title):
    t = df.pivot_table(index="key", columns="arm", values="rel_L2_D")
    lines = [f"# {exp}: {title}", "", "relL2(D).", "", md_table(t.rename(index=short)), ""]
    return lines, t


def summarise_D(dfs):
    """Density: D1 (project PINN across sensor counts) against the CN grid adjoint and the CN neural adjoint of
    the campaign (campaign/density_paired_results.csv, same datasets), plus D2 / D3 diagnostics."""
    lines = ["# D: does the PINN improve with more sensors?", ""]
    camp = pd.read_csv(C.EXT / "campaign" / "density_paired_results.csv").set_index("key")
    d1 = dfs.get("D1")
    t = None
    if d1 is not None:
        d1 = d1.copy()
        d1["adjoint_grid"] = d1["key"].map(camp["grid_rel_L2_D"])
        d1["adjoint_neural"] = d1["key"].map(camp["neural_rel_L2_D"])
        t = d1.pivot_table(index=["eta", "n_sensors"], values=["rel_L2_D", "adjoint_grid", "adjoint_neural",
                                                                "replay_over_sigma"], aggfunc="median")
        t = t.rename(columns={"rel_L2_D": "pinn"})[["pinn", "adjoint_grid", "adjoint_neural", "replay_over_sigma"]]
        t = t.rename(columns={"replay_over_sigma": "pinn_replay"})
        lines += ["## D1: median relL2(D) over the 6 fields", "",
                  "PINN = submission/pinn_inversion.py unchanged (1,500 epochs). Adjoints = CN grid (16 x 16) and CN neural "
                  "(3 x 32 network) adjoints from the campaign on the same datasets. pinn_replay = PINN's D replayed "
                  "through the CN solver, RMS / sigma (1 = fits to the noise level).", "", md_table(t), ""]
        by = d1.pivot_table(index=["tier", "eta"], columns="n_sensors", values="rel_L2_D")
        lines += ["## D1: PINN relL2(D) per field", "", md_table(by), ""]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
        for ax, eta in zip(axes, sorted(d1["eta"].unique())):
            q = d1[d1.eta == eta]
            for tier in sorted(q.tier.unique()):
                r = q[q.tier == tier].sort_values("n_sensors")
                ax.plot(r.n_sensors, r.rel_L2_D, color=COL["pinn"], alpha=0.25, lw=0.8)
                ax.plot(r.n_sensors, r.adjoint_grid, color=COL["adjoint"], alpha=0.25, lw=0.8)
            m = q.groupby("n_sensors")[["rel_L2_D", "adjoint_grid"]].median()
            ax.plot(m.index, m.rel_L2_D, "o-", color=COL["pinn"], lw=2, label="PINN (median of 6 fields)")
            ax.plot(m.index, m.adjoint_grid, "s-", color=COL["adjoint"], lw=2, label="CN adjoint (median)")
            ax.set(xscale="log", title=f"noise {eta:.0%}", xlabel="sensors")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("relL2(D)")
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "D1_density.png", dpi=170); plt.close(fig)
    for exp in ("D2", "D3"):
        d = dfs.get(exp)
        if d is None:
            continue
        d = d.copy()
        if d1 is not None:
            base = d1.set_index("key")["rel_L2_D"]
            d = pd.concat([d, pd.DataFrame(dict(key=base.index, arm="pinn (D1)", rel_L2_D=base.values))])
        tt = d.pivot_table(index="key", columns="arm", values="rel_L2_D")
        tt = tt.loc[[k for k in tt.index if tt.loc[k].drop("pinn (D1)", errors="ignore").notna().any()]]
        lines += [f"## {exp}", "", md_table(tt.rename(index=lambda k: k.replace("__Fo0.1__uniform8__", " ").replace("__nodes_nested__", " "))), ""]
    return lines, t


def summarise_E1b(df):
    lines = ["# E1b: hidden walls -- PINN walls handed to the adjoint, against the classical joint inversion", "",
             "relL2(D); wall error = RMS(estimated - true walls) / RMS(true walls); cost = CPU seconds (hybrids include "
             "the PINN). adjoint_truebc uses the hidden true walls: a bound, not a method.", ""]
    e1 = load("E1")
    if e1 is not None:
        add = e1[(e1.collection == "pinnwin") & e1.key.isin(df.key) & (e1.arm == "adjoint")].copy()
        add["arm"] = "adjoint_zero_walls (E1)"
        df = pd.concat([df, add])
    err = df.pivot_table(index="key", columns="arm", values="rel_L2_D")
    order = [a for a in ["adjoint_zero_walls (E1)", "pinn_bc", "hybrid_bc", "adjoint_joint5", "hybrid_bc_joint5",
                         "adjoint_joint1", "adjoint_truebc"] if a in err]
    err = err[order]
    wall = df.pivot_table(index="key", columns="arm", values="wall_rel_err")
    df = df.copy()
    cpu = df.set_index(["key", "arm"])["job_cpu_s"] if "job_cpu_s" in df else None
    if cpu is not None:
        df["cost_cpu_s"] = [cpu.get((k, a), np.nan) + (cpu.get((k, "pinn_bc"), 0.0) if a.startswith("hybrid") else 0.0)
                            for k, a in zip(df.key, df.arm)]
    cost = df.pivot_table(index="key", columns="arm", values="cost_cpu_s") if cpu is not None else None
    rename = lambda k: k.split("__")[0]
    lines += ["## relL2(D)", "", md_table(err.rename(index=rename)), "",
              "## Medians", "", md_table(err.median().to_frame("median relL2(D)")), ""]
    w = wall[[a for a in order if a in wall]]
    lines += ["## Wall error", "", md_table(w.rename(index=rename)), ""]
    if cost is not None:
        lines += ["## Cost (CPU s)", "", md_table(cost[[a for a in order if a in cost]].rename(index=rename), ".0f"), ""]
    if {"pinn_bc", "adjoint_joint5", "hybrid_bc_joint5"} <= set(err):
        best = err[["pinn_bc", "adjoint_joint5"]].min(axis=1)
        lines += [f"hybrid_bc_joint5 beats both pinn_bc and adjoint_joint5 (by > 1 %) on "
                  f"{int((err['hybrid_bc_joint5'] < (1 - TIE) * best).sum())} of {int(err['hybrid_bc_joint5'].notna().sum())} datasets; "
                  f"hybrid_bc on {int((err['hybrid_bc'] < (1 - TIE) * best).sum()) if 'hybrid_bc' in err else 'n/a'}.", ""]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(err))
    wdt = 0.8 / len(order)
    for i, a in enumerate(order):
        ax.bar(x + i * wdt, err[a].values, wdt, label=a)
    ax.set(yscale="log", ylabel="relL2(D)", xticks=x + 0.4 - wdt / 2)
    ax.set_xticklabels([rename(k).replace("H2u-", "") for k in err.index], rotation=20, fontsize=7)
    ax.legend(fontsize=7, frameon=False, ncol=4)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "E1b_walls.png", dpi=170); plt.close(fig)
    return lines, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=["E1", "E1b", "E1c", "E2a", "E2b", "E3a", "E3b", "E3c", "E3d", "D"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for exp in a.exp:
        if exp == "D":
            dfs = {e: load(e) for e in ("D1", "D2", "D3")}
            if not any(v is not None for v in dfs.values()):
                continue
            lines, t = summarise_D({k: v for k, v in dfs.items() if v is not None})
            (OUT / "D.md").write_text("\n".join(lines))
            print("\n".join(lines))
            continue
        if exp == "E1c":
            continue                                   # folded into the E1 summary
        df = load(exp)
        if df is None:
            continue
        if exp == "E1":
            lines, t = summarise_E1(df)
            c = load("E1c")
            if c is not None:
                e = df[df.collection == "bench"].pivot_table(index="key", columns="arm", values="rel_L2_D")
                c3 = c.set_index("key")["rel_L2_D"].reindex(e.index)
                cpu = df[df.collection == "bench"].pivot_table(index="key", columns="arm", values="total_wall_s")
                lines += ["## Equal budget (E1c): the PINN alone with 3 x the epochs (about the hybrids' cost)", "",
                          f"median relL2: pinn {e['pinn'].median():.3f}, pinn_3x {c3.median():.3f}, adjoint {e['adjoint'].median():.3f}, "
                          f"hybrid_balance {e['hybrid_balance'].median():.3f}. pinn_3x lower than pinn on "
                          f"{int((c3 < (1 - TIE) * e['pinn']).sum())} of {int(c3.notna().sum())}; lower than the adjoint on "
                          f"{int((c3 < (1 - TIE) * e['adjoint']).sum())}.", "",
                          md_table(pd.DataFrame(dict(pinn=e["pinn"], pinn_3x=c3, adjoint=e["adjoint"],
                                                     hybrid_balance=e["hybrid_balance"])).rename(index=short)), ""]
        elif exp == "E1b":
            lines, t = summarise_E1b(df)
        elif exp in ("E2a", "E2b"):
            lines, t = summarise_E2(df, exp)
        elif exp == "E3a":
            lines, t = summarise_E3a(df)
        elif exp == "E3b":
            lines, t = summarise_E3b(df)
        elif exp == "E3c":
            lines, t = summarise_simple(df, exp, "adaptive collocation (RAR)")
        else:
            lines, t = summarise_simple(df, exp, "interpolation augmentation of sparse sensors")
        df.drop(columns=[c for c in ("path", "candidates", "cv_curve", "design_gains") if c in df], errors="ignore") \
          .to_csv(OUT / f"{exp}_runs.csv", index=False)
        (OUT / f"{exp}.md").write_text("\n".join(lines))
        print("\n".join(lines))


if __name__ == "__main__":
    main()
