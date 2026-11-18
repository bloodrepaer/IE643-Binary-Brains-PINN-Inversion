"""
make_poster_figures.py -- the data-driven figures for the novelty poster (vector PDF + PNG preview).

  python run.py novelty/make_poster_figures.py            # every figure whose results exist
  python run.py novelty/make_poster_figures.py --only F3 F7

Reads metrics.json / D_hat.npz under $NOVELTY_RESULTS (default novelty/results_full) and the benchmark
datasets; writes novelty/poster_figs/.

Colour: one identity per method across every figure -- classical adjoint blue, PINN orange, hybrid / joint
classical inversion aqua, true-physics bound neutral grey. D maps use one single-hue (blue) sequential ramp
on one shared scale per figure. Palette: the dataviz skill's reference categorical slots 1-3 (validated
all-pairs, light mode); aqua sits below 3:1 contrast on white, so aqua marks always carry a value label.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

os.environ.setdefault("NOVELTY_RESULTS", str(Path(__file__).resolve().parent / "results_full"))
import common as C

OUT = C.EXT / "novelty" / "poster_figs"

# ---------------------------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------------------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = dict(adjoint="#2a78d6", pinn="#eb6834", joint="#1baf7a", bound="#8a8984", other="#6250d6")
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DMAP = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
PINN_RAMP = ["#f6b393", "#eb6834", "#a8401a"]       # one hue, light -> dark: PINN at 1.5k / 6k / 24k epochs

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "lines.markersize": 7, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def load(exp):
    rows = [json.loads(p.read_text()) for p in sorted((C.RESULTS / exp).glob("*/*/metrics.json"))]
    return pd.DataFrame(rows) if rows else None


def D_of(exp, arm, key):
    p = C.RESULTS / exp / arm / key / "D_hat.npz"
    return np.load(p)["D_hat"] if p.exists() else None


def show_D(ax, D, vmin, vmax, title=None, sensors=None):
    im = ax.imshow(np.asarray(D).T, origin="lower", extent=[0, 1, 0, 1], cmap=DMAP, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    if sensors is not None:
        ax.scatter(*sensors, s=10, c="white", edgecolors=INK, linewidths=0.6, zorder=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=13)
    return im


STD = "__Fo0.1__uniform8__n50__random__eta0.05"
UNK = "__Fo0.1__uniform16__n50__random__eta0.05"
FIELDS = ["A_smooth", "B_gaussian_bump", "C_inclusion", "D_layered", "E_grf", "F_defects"]
NICE = dict(A_smooth="A  smooth", B_gaussian_bump="B  Gaussian bump", C_inclusion="C  inclusion",
            D_layered="D  layered", E_grf="E  random field", F_defects="F  defects")


# ---------------------------------------------------------------------------------------------
# F3: dataset gallery
# ---------------------------------------------------------------------------------------------
def F3():
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(2, 7, width_ratios=[1] * 6 + [0.06], height_ratios=[1, 1], wspace=0.08, hspace=0.32)
    vmin, vmax = 0.0, 0.75
    for j, f in enumerate(FIELDS):
        ds = C.dataset("bench", f + STD)
        im = show_D(fig.add_subplot(gs[0, j]), ds.ground_truth()[0], vmin, vmax, NICE[f])
    cax = fig.add_subplot(gs[0, 6])
    cb = fig.colorbar(im, cax=cax); cb.set_label("true D(x, y)"); cb.outline.set_visible(False)
    layouts = [("random, 50", "A_smooth" + STD), ("grid, 20", "C_inclusion__Fo0.1__uniform8__n20__grid__eta0.05"),
               ("clustered, 50", "D_layered__Fo0.1__uniform8__n50__clustered__eta0.05"),
               ("boundary, 50", "F_defects__Fo0.1__uniform8__n50__boundary__eta0.05"),
               ("every node, 3,969", None)]
    for j, (name, key) in enumerate(layouts):
        ax = fig.add_subplot(gs[1, j])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_color(INK2); s.set_linewidth(0.8); s.set_visible(True)
        if key:
            ds = C.dataset("bench", key)
            xy = np.unique(np.c_[ds.x, ds.y].round(6), axis=0)
            ax.scatter(xy[:, 0], xy[:, 1], s=26, c=COL["adjoint"], edgecolors="white", linewidths=0.6)
        else:
            g = np.linspace(0, 1, 65)
            X, Y = np.meshgrid(g, g)
            ax.scatter(X, Y, s=0.6, c=COL["adjoint"], linewidths=0)
        ax.set_title(name, fontsize=13)
    ax = fig.add_subplot(gs[1, 5]); ax.axis("off")
    ax.text(0.0, 0.5, "noise  1 %, 5 %, 10 %\nreadings at 8 times\nFo = 0.02 - 0.2\nRK4 on 129 x 129,\ninverted on 65 x 65",
            va="center", ha="left", fontsize=12, color=INK2, linespacing=1.5)
    fig.text(0.005, 0.93, "True diffusivity fields", fontsize=14, fontweight="bold")
    fig.text(0.005, 0.46, "Sensor layouts", fontsize=14, fontweight="bold")
    save(fig, "F3_datasets")


# ---------------------------------------------------------------------------------------------
# F4: qualitative reconstructions, known physics
# ---------------------------------------------------------------------------------------------
def F4():
    rows = [("A_smooth" + STD, "A  smooth"), ("C_inclusion" + STD, "C  inclusion"), ("E_grf" + STD, "E  random field")]
    cols = [("truth", None), ("classical adjoint", ("E1", "adjoint")), ("PINN", ("E1", "pinn")),
            ("hybrid (PINN -> adjoint)", ("E1", "hybrid_balance"))]
    df = load("E1")
    rel = {} if df is None else {(r.arm, r.key): r.rel_L2_D for r in df.itertuples()}
    fig = plt.figure(figsize=(12.5, 9.6))
    gs = fig.add_gridspec(len(rows), len(cols) + 1, width_ratios=[1] * len(cols) + [0.05], wspace=0.06, hspace=0.18)
    vmin, vmax = 0.0, 0.62
    for i, (key, name) in enumerate(rows):
        ds = C.dataset("bench", key)
        sens = np.unique(np.c_[ds.x, ds.y].round(6), axis=0).T
        for j, (cname, src) in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            if src is None:
                im = show_D(ax, ds.ground_truth()[0], vmin, vmax, sensors=sens)
                ax.text(-0.06, 0.5, name, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=13)
            else:
                D = D_of(src[0], src[1], key)
                show_D(ax, D, vmin, vmax)
                r = rel.get((src[1], key))
                if r is not None:
                    ax.text(0.03, 0.04, f"relL2 {r:.2f}", transform=ax.transAxes, fontsize=12, color=INK,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
            if i == 0:
                ax.set_title(cname, fontsize=13)
    cax = fig.add_subplot(gs[:, len(cols)])
    cb = fig.colorbar(im, cax=cax); cb.set_label("D(x, y)"); cb.outline.set_visible(False)
    save(fig, "F4_reconstructions")


# ---------------------------------------------------------------------------------------------
# F5: headline -- the PINN's wins under unknown physics vanish
# ---------------------------------------------------------------------------------------------
def F5():
    e1, e1b, e1s = load("E1"), load("E1b"), load("E1s")
    if e1 is None or e1b is None or e1s is None:
        print("F5: waiting for E1 / E1b / E1s"); return
    def med(df, arm, prefix):
        q = df[(df.arm == arm) & df.key.str.startswith(prefix)]
        return float(q.rel_L2_D.median()), len(q)
    groups = [("hidden wall temperatures\n(8 datasets)", "H2u",
               [("adjoint,\nwrong\nphysics", med(e1, "adjoint", "H2u"), COL["adjoint"]),
                ("PINN", med(e1, "pinn", "H2u"), COL["pinn"]),
                ("adjoint +\nestimated\nwalls", med(e1b, "adjoint_joint1", "H2u"), COL["joint"]),
                ("true walls\n(bound)", med(e1b, "adjoint_truebc", "H2u"), COL["bound"])]),
              ("hidden heat source\n(6 datasets)", "H3u",
               [("adjoint,\nwrong\nphysics", med(e1, "adjoint", "H3u"), COL["adjoint"]),
                ("PINN", med(e1, "pinn", "H3u"), COL["pinn"]),
                ("adjoint +\nestimated\nsource", med(e1s, "adjoint_src_pos", "H3u"), COL["joint"]),
                ("true source\n(bound)", med(e1s, "adjoint_truesrc", "H3u"), COL["bound"])])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, (title, _, bars) in zip(axes, groups):
        x = np.arange(len(bars))
        vals = [b[1][0] for b in bars]
        ax.bar(x, vals, width=0.62, color=[b[2] for b in bars], edgecolor="white", linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v * 1.12, f"{v:.2f}", ha="center", va="bottom", fontsize=13, color=INK, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=12)
        ax.set_yscale("log"); ax.set_ylim(0.05, 12)
        ax.set_title(title, fontsize=14)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("median relL2(D)  (log scale, lower is better)")
    save(fig, "F5_unknown_physics")


# ---------------------------------------------------------------------------------------------
# F6: hybrids with seed spread
# ---------------------------------------------------------------------------------------------
def F6():
    e1, s1 = load("E1"), load("S1")
    if e1 is None:
        return
    b = e1[e1.collection == "bench"]
    t = b.pivot_table(index="key", columns="arm", values="rel_L2_D")
    arms = [("classical\nadjoint", "adjoint", COL["adjoint"]), ("PINN", "pinn", COL["pinn"]),
            ("hybrid\npath", "hybrid_path", COL["joint"]), ("hybrid\nbalance", "hybrid_balance", COL["joint"]),
            ("hybrid\nprior", "hybrid_prior", COL["joint"]), ("reverse\n(adjoint -> PINN)", "reverse", COL["pinn"])]
    seed_med = {}
    if s1 is not None:
        sb = s1[s1.collection == "bench"]
        for base in ("pinn", "hybrid_balance"):
            q = sb[sb.arm.str.match(rf"^{base}_s\d+$")]
            if len(q):
                per_seed = [t[base].median()] + [g.rel_L2_D.median() for _, g in q.groupby("arm") if g.key.nunique() == t.shape[0]]
                if len(per_seed) > 1:
                    seed_med[base] = per_seed
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    x = np.arange(len(arms))
    for xi, (lab, arm, col) in zip(x, arms):
        v = t[arm].dropna()
        jitter = (np.random.default_rng(0).random(len(v)) - 0.5) * 0.28
        ax.scatter(xi + jitter, v, s=22, color=col, alpha=0.35, linewidths=0, zorder=2)
        m = v.median()
        ax.plot([xi - 0.3, xi + 0.3], [m, m], color=col, lw=3, solid_capstyle="round", zorder=3)
        if arm in seed_med:
            lo, hi = min(seed_med[arm]), max(seed_med[arm])
            ax.plot([xi + 0.36] * 2, [lo, hi], color=INK, lw=1.6, zorder=4)
            ax.plot([xi + 0.32, xi + 0.40], [lo, lo], color=INK, lw=1.6)
            ax.plot([xi + 0.32, xi + 0.40], [hi, hi], color=INK, lw=1.6)
            ax.text(xi + 0.45, np.sqrt(lo * hi), f"{len(seed_med[arm])} seeds\n{lo:.3f}-{hi:.3f}", va="center",
                    ha="left", fontsize=11, color=INK2)
        ax.text(xi, m * 0.88 if arm != "hybrid_prior" else m * 1.08, f"{m:.3f}", ha="center",
                va="top" if arm != "hybrid_prior" else "bottom", fontsize=12, fontweight="bold", color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9), zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([a[0] for a in arms])
    ax.set_yscale("log"); ax.set_ylabel("relL2(D)  (log scale)")
    ax.grid(axis="x", visible=False); ax.tick_params(axis="x", length=0)
    note = "dots: 20 datasets   bar: median"
    if seed_med:
        n = max(len(v) for v in seed_med.values())
        note += f"   bracket: range of the median over {n} seeds"
    ax.set_title(note, fontsize=12, color=INK2, loc="left")
    save(fig, "F6_hybrids")


# ---------------------------------------------------------------------------------------------
# F7: sensor density
# ---------------------------------------------------------------------------------------------
def F7():
    d1, d2 = load("D1"), load("D2")
    camp = pd.read_csv(C.EXT / "campaign" / "density_paired_results.csv").set_index("key")
    if d1 is None:
        return
    d1 = d1[d1.eta == 0.01]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), sharey=True)
    for ax, tier, name in zip(axes, ["A_smooth", "E_grf"], ["A  smooth", "E  random field"]):
        q = d1[d1.tier == tier].sort_values("n_sensors")
        n = q.n_sensors.values
        adj = q.key.map(camp["grid_rel_L2_D"]).values
        ax.plot(n, adj, "s-", color=COL["adjoint"], label="classical adjoint (CN)", ms=8)
        ax.plot(n, q.rel_L2_D.values, "o-", color=PINN_RAMP[0], label="PINN, 1,500 epochs", ms=8)
        ax.text(n[-1] * 1.12, q.rel_L2_D.values[-1], "1,500", va="center", fontsize=11, color=INK2)
        ax.text(n[-1] * 1.12, adj[-1], "adjoint", va="center", fontsize=11, color=INK2)
        if d2 is not None:
            for arm, col, lab in (("pinn_long", PINN_RAMP[1], "PINN, 6,000 epochs"),
                                  ("pinn_xlong", PINN_RAMP[2], "PINN, 24,000 epochs")):
                r = d2[(d2.arm == arm) & (d2.key.str.startswith(tier + "__"))].copy()
                if len(r):
                    r["n"] = r.key.str.extract(r"__n(\d+)__").astype(int)
                    r = r.sort_values("n")
                    ax.plot(r.n, r.rel_L2_D, "o-" if len(r) > 2 else "o--", color=col, label=lab, ms=8)
                    ax.text(r.n.iloc[-1] * 1.12, r.rel_L2_D.iloc[-1], lab.split(", ")[1].replace(" epochs", ""),
                            va="center", fontsize=11, color=INK2)
        ax.set_xscale("log"); ax.set_xticks([20, 100, 400, 1600, 3969]); ax.set_xticklabels(["20", "100", "400", "1,600", "3,969"])
        ax.minorticks_off()
        ax.set_xlabel("number of sensors (1 % noise)"); ax.set_title(name)
        ax.set_ylim(0, 0.55); ax.set_xlim(14, 9000)
    axes[0].set_ylabel("relL2(D)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=12)
    save(fig, "F7_density")


# ---------------------------------------------------------------------------------------------
# F8: model error (Arrhenius strength)
# ---------------------------------------------------------------------------------------------
def F8():
    df = load("E2b")
    if df is None:
        return
    df = df.copy()
    df["field"] = df.key.str.split("-arr").str[0]
    df["S"] = df.key.str.extract(r"arrS([\d.]+)__")[0].astype(float)
    df["dense"] = df.key.str.contains("n100")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, field, name in zip(axes, ["A_smooth", "C_inclusion"], ["A  smooth", "C  inclusion"]):
        for arm, col, lab in (("adjoint", COL["adjoint"], "classical adjoint (assumes linear D)"), ("pinn", COL["pinn"], "PINN")):
            for dense, ls, setting in ((True, "-", "100 sensors, 1 % noise"), (False, "--", "50 sensors, 5 % noise")):
                r = df[(df.field == field) & (df.arm == arm) & (df.dense == dense)].sort_values("S")
                ax.plot(r.S, r.rel_L2_D, ls, marker="o" if dense else "s", color=col, ms=8,
                        label=f"{lab}, {setting}")
        ax.set_xticks([0, 0.5, 1, 2]); ax.set_xlabel("strength of temperature dependence S"); ax.set_title(name)
    axes[0].set_ylabel("relL2(D0)")
    axes[0].legend(fontsize=10.5, loc="upper left")
    save(fig, "F8_model_error")


# ---------------------------------------------------------------------------------------------
# F9: what helps a PINN
# ---------------------------------------------------------------------------------------------
def paired(df_new, arm_new, df_old, arm_old, keys=None):
    a = df_new[df_new.arm == arm_new].set_index("key").rel_L2_D
    b = df_old[df_old.arm == arm_old].set_index("key").rel_L2_D
    k = a.index.intersection(b.index)
    if keys is not None:
        k = k.intersection(keys)
    r = a[k] / b[k] - 1
    return float(np.median(r)), int((r < -0.01).sum()), len(r)


def F9():
    e1, e1c, e2a, e3a, e3b, e3c, e3d = (load(x) for x in ("E1", "E1c", "E2a", "E3a", "E3b", "E3c", "E3d"))
    items = []
    if e1 is not None and e1c is not None:
        items.append(("3 x the epochs", *paired(e1c, "pinn_3x", e1, "pinn")))
    if e3a is not None:
        items.append(("loss weights chosen on\nheld-out sensors", *paired(e3a, "pinn_selected", e3a, "pinn_default")))
    if e3b is not None:
        g = e3b.copy()
        g["base"] = g.key.str.replace(r"__(grid|random|designed)__", "__X__", regex=True)
        a = g[(g.arm == "pinn") & g.key.str.contains("__grid__")].set_index("base").rel_L2_D
        b = g[(g.arm == "pinn") & g.key.str.contains("__random__")].set_index("base").rel_L2_D
        k = a.index.intersection(b.index); r = a[k] / b[k] - 1
        items.append(("sensors on a grid\ninstead of random", float(np.median(r)), int((r < -0.01).sum()), len(r)))
    if e3d is not None:
        items.append(("interpolated\npseudo-data", *paired(e3d, "pinn_aug", e3d, "pinn")))
    if e2a is not None:
        items.append(("learned g(u) on data\nwithout it", *paired(e2a, "pinn_T", e2a, "pinn")))
    if e3c is not None:
        items.append(("adaptive collocation\n(RAR)", *paired(e3c, "pinn_rar", e3c, "pinn")))
    items.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(10.5, 0.75 * len(items) + 1.4))
    y = np.arange(len(items))[::-1]
    ax.axvline(0, color=INK2, lw=1)
    for yi, (lab, med, better, n) in zip(y, items):
        ax.plot([0, med * 100], [yi, yi], color=COL["pinn"], lw=2)
        ax.plot(med * 100, yi, "o", color=COL["pinn"], ms=11)
        ax.text(med * 100 + (1.2 if med > 0 else -1.2), yi, f"{med * 100:+.0f} %   (better on {better} of {n})",
                va="center", ha="left" if med > 0 else "right", fontsize=12, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([t[0] for t in items])
    lim = max(abs(t[1]) for t in items) * 100 + 22
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("median change in the PINN's relL2(D)   (left = better)")
    ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
    save(fig, "F9_pinn_practice")


# ---------------------------------------------------------------------------------------------
# F10: backward Euler vs CN inside the classical adjoint
# ---------------------------------------------------------------------------------------------
def F10():
    t = pd.read_csv(C.EXT / "study5_time_integrators" / "results_table.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, eta in zip(axes, (0.01, 0.05, 0.1)):
        q = t[t.eta == eta]
        for (solver, steps), col, ls, lab in ((("CN", 4800), COL["bound"], ":", "fine time step (4,800)"),
                                              (("CN", 100), COL["adjoint"], "-", "Crank-Nicolson, 100 steps"),
                                              (("BE", 100), COL["other"], "-", "backward Euler, 100 steps")):
            r = q[(q.solver == solver) & (q.n_steps == steps)].sort_values("n_sensors")
            ax.plot(r.n_sensors, r.rel_L2_D, ls, marker="o", color=col, label=lab, ms=8)
        ax.set_xscale("log"); ax.set_xticks([50, 400, 3969]); ax.set_xticklabels(["50", "400", "3,969"]); ax.minorticks_off()
        ax.set_title(f"noise {eta:.0%}"); ax.set_xlabel("sensors")
    axes[0].set_ylabel("classical adjoint relL2(D)")
    axes[0].annotate("Crank-Nicolson at 100 steps\nlies on the fine-step line", xy=(400, 0.0517), xytext=(70, 0.035),
                     fontsize=11, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    axes[0].set_ylim(0.025, None)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=12)
    save(fig, "F10_time_integrator")


# ---------------------------------------------------------------------------------------------
# F11: hidden source -- maps
# ---------------------------------------------------------------------------------------------
def F11():
    e1s = load("E1s")
    if e1s is None:
        print("F11: waiting for E1s"); return
    key = "H3u-A_smooth-q0.3" + UNK
    if D_of("E1s", "adjoint_src_pos", key) is None:
        print("F11: waiting for", key); return
    import hybrid_src as HS
    ds = C.dataset("pinnwin", key)
    rel = {(r.exp if hasattr(r, "exp") else "", r.arm): r.rel_L2_D for r in pd.concat([load("E1"), e1s]).itertuples() if r.key == key}
    f_true = HS.true_source("pinnwin", key, ds.N).reshape(ds.N, ds.N)
    f_hat = np.load(C.RESULTS / "E1s" / "adjoint_src_pos" / key / "D_hat.npz")["f_hat"]
    fig = plt.figure(figsize=(16, 4.1))
    gs = fig.add_gridspec(1, 10, width_ratios=[1, 1, 1, 1, 0.05, 0.32, 1, 1, 0.05, 0.02], wspace=0.08)
    vmin, vmax = 0.0, 0.55
    panels = [("truth", ds.ground_truth()[0], None), ("adjoint, no source", D_of("E1", "adjoint", key), "adjoint"),
              ("PINN", D_of("E1", "pinn", key), "pinn"), ("adjoint + source", D_of("E1s", "adjoint_src_pos", key), "adjoint_src_pos")]
    for j, (name, D, arm) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        im = show_D(ax, np.clip(D, vmin, vmax * 1.5), vmin, vmax, name)
        if arm:
            r = next((v for (e, a), v in rel.items() if a == arm), None)
            if r is not None:
                ax.text(0.03, 0.04, f"relL2 {r:.2f}", transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 4])); cb.set_label("D"); cb.outline.set_visible(False)
    smax = max(f_true.max(), f_hat.max())
    grey = LinearSegmentedColormap.from_list("grey_seq", ["#f1f0ec", "#c3c2b7", "#8a8984", "#52514e", "#1f1f1e"])
    for j, (name, F) in enumerate((("true heater", f_true), ("recovered heater", f_hat))):
        ax = fig.add_subplot(gs[0, 6 + j])
        imf = ax.imshow(np.asarray(F).T, origin="lower", extent=[0, 1, 0, 1], cmap=grey, vmin=0, vmax=smax)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ax.set_title(name, fontsize=13)
    cb2 = fig.colorbar(imf, cax=fig.add_subplot(gs[0, 8])); cb2.set_label("source f(x, y)"); cb2.outline.set_visible(False)
    fig.text(0.13, 0.86, "diffusivity D(x, y)", fontsize=14, fontweight="bold")
    fig.text(0.66, 0.86, "hidden heat source", fontsize=14, fontweight="bold")
    save(fig, "F11_hidden_source")


FIGS = dict(F3=F3, F4=F4, F5=F5, F6=F6, F7=F7, F8=F8, F9=F9, F10=F10, F11=F11)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(FIGS))
    a = ap.parse_args()
    for k in a.only:
        FIGS[k]()
