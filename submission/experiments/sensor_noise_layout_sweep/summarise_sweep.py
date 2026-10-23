"""Summarise the PINN vs adjoint sensor/noise/layout sweep (Fo 0.1, uniform8).

Writes sweep_comparison.csv, sweep_tables.md and sweep_comparison.png next to this script.
"""
import json, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
rows = []
for k in open(os.path.join(H, "keys.txt")).read().split():
    m = re.match(r"(.+?)__Fo0\.1__uniform8__n(\w+)__(\w+)__eta([\d.]+)$", k)
    fam, n, lay, eta = m.group(1), int(m.group(2)), m.group(3), float(m.group(4))
    r = dict(key=k, family=fam, sensors=n, layout=lay, noise_pct=eta * 100)
    a = os.path.join(H, "../../models/classical/sweep", k, "metrics.json"); p = os.path.join(H, "../../models/pinn/sweep", k, "metrics.json")
    if os.path.exists(a):
        j = json.load(open(a))
        r.update(adj_relL2=j["rel_L2_D"], adj_replay=j["rms_replay_over_sigma"], adj_flag=j["flag"],
                 const_relL2=j["rel_L2_D_constant_guess"], adj_wall_s=j["wall_s"])
    if os.path.exists(p):
        j = json.load(open(p))
        r.update(pinn_relL2=j["rel_l2_D"], pinn_wall_s=j["elapsed_sec"])
    rows.append(r)
d = pd.DataFrame(rows)
both = d.dropna(subset=["adj_relL2", "pinn_relL2"]).copy()
both["winner"] = np.where(both.adj_relL2 < both.pinn_relL2, "adjoint", "PINN")
d = d.merge(both[["key", "winner"]], on="key", how="left")
d.to_csv(os.path.join(H, "sweep_comparison.csv"), index=False)

out = [f"Completed pairs: {len(both)} of {len(d)}. relL2(D), lower is better; cells are adjoint / PINN.\n"]
for fam in d.family.unique():
    f = d[(d.family == fam) & (d.layout == "random")]
    if f.empty:
        continue
    out.append(f"\n### {fam}: sensors x noise (random layout)\n")
    noises = sorted(f.noise_pct.unique())
    out.append("| sensors | " + " | ".join(f"{x:g}% noise" for x in noises) + " |")
    out.append("|---" * (len(noises) + 1) + "|")
    for n in sorted(f.sensors.unique()):
        cells = []
        for e in noises:
            r = f[(f.sensors == n) & (f.noise_pct == e)]
            a = r.adj_relL2.iloc[0] if len(r) and pd.notna(r.adj_relL2.iloc[0]) else None
            p = r.pinn_relL2.iloc[0] if len(r) and pd.notna(r.pinn_relL2.iloc[0]) else None
            cells.append(("-" if a is None else f"{a:.3f}") + " / " + ("-" if p is None else f"{p:.3f}"))
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    g = d[(d.family == fam) & (d.noise_pct == 5) & (d.sensors.isin([20, 50]))]
    out.append(f"\n### {fam}: layouts at 5% noise\n")
    out.append("| layout | 20 sensors | 50 sensors |\n|---|---|---|")
    for lay in ["random", "grid", "clustered", "boundary"]:
        cells = []
        for n in (20, 50):
            r = g[(g.layout == lay) & (g.sensors == n)]
            a = r.adj_relL2.iloc[0] if len(r) and pd.notna(r.adj_relL2.iloc[0]) else None
            p = r.pinn_relL2.iloc[0] if len(r) and pd.notna(r.pinn_relL2.iloc[0]) else None
            cells.append(("-" if a is None else f"{a:.3f}") + " / " + ("-" if p is None else f"{p:.3f}"))
        out.append(f"| {lay} | " + " | ".join(cells) + " |")
if len(both):
    out.append(f"\nWins: adjoint {int((both.winner == 'adjoint').sum())}, PINN {int((both.winner == 'PINN').sum())}.")
    out.append(f"Median relL2: adjoint {both.adj_relL2.median():.3f}, PINN {both.pinn_relL2.median():.3f}, "
               f"best constant {both.const_relL2.median():.3f}.")
    out.append(f"Median wall time (1 CPU thread each): adjoint {both.adj_wall_s.median():.0f} s, "
               f"PINN {both.pinn_wall_s.median():.0f} s.")
    w = both[both.winner == "PINN"]
    if len(w):
        out.append("PINN wins: " + ", ".join(f"{r.key} ({r.adj_relL2:.3f} vs {r.pinn_relL2:.3f})" for r in w.itertuples()))
open(os.path.join(H, "sweep_tables.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))

# figure: relL2 vs sensors, one panel per family x method, a line per noise level
fams = [f for f in d.family.unique()]
fig, axs = plt.subplots(1, len(fams), figsize=(6 * len(fams), 4.2), squeeze=False)
for ax, fam in zip(axs[0], fams):
    f = d[(d.family == fam) & (d.layout == "random")]
    for i, e in enumerate(sorted(f.noise_pct.unique())):
        s = f[f.noise_pct == e].sort_values("sensors")
        ax.plot(s.sensors, s.adj_relL2, "o-", color=f"C{i}", label=f"adjoint, {e:g}% noise")
        ax.plot(s.sensors, s.pinn_relL2, "s--", color=f"C{i}", label=f"PINN, {e:g}% noise")
    c = f.const_relL2.dropna()
    if len(c):
        ax.axhline(c.iloc[0], color="k", ls=":", label="best constant D")
    ax.set(xscale="log", xlabel="number of sensors", ylabel="relL2(D)  (lower is better)",
           title=f"{fam}: adjoint (solid) vs PINN (dashed)")
    ax.grid(alpha=0.3, which="both")
axs[0][-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout(); fig.savefig(os.path.join(H, "sweep_comparison.png"), dpi=120)
