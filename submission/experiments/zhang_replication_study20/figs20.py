import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = list(csv.DictReader(open("results/S20/comparison_pinn.csv")))
short = {"A_smooth":"A smooth","B_gaussian_bump":"B bump","C_inclusion":"C inclusion","D_layered":"D layered","E_grf":"E random field","F_defects":"F defects"}
def lab(k):
    t, Fo, p, n, lay, eta = k.split("__")
    bits = [short[t]]
    if Fo != "Fo0.1": bits.append(Fo.replace("Fo","Fo "))
    if p != "uniform8": bits.append(p)
    bits.append(n.replace("nfull","full field").replace("n","") + (" sensors" if n!="nfull" else ""))
    if lay not in ("random","nodes"): bits.append(lay)
    bits.append(f"{float(eta[3:])*100:g}% noise")
    return ", ".join(bits)
for r in rows:
    for k in ("adj_eps_D","pinn_eps_D","const_eps_D","ratio_D"): r[k]=float(r[k])
rows.sort(key=lambda r: r["ratio_D"])
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
C_ADJ, C_PINN = "#2a78d6", "#eb6834"
fig, ax = plt.subplots(figsize=(9.5, 8.2), facecolor=SURF); ax.set_facecolor(SURF)
y = np.arange(len(rows))
for i, r in enumerate(rows):
    ax.plot([r["adj_eps_D"], r["pinn_eps_D"]], [i, i], color=GRID, lw=2, zorder=1, solid_capstyle="round")
ax.scatter([r["const_eps_D"] for r in rows], y, marker="|", s=140, color=MUTED, lw=2, zorder=2, label="best constant D (no spatial detail)")
ax.scatter([r["adj_eps_D"] for r in rows], y, s=64, color=C_ADJ, edgecolor=SURF, lw=1.5, zorder=3, label="classical adjoint")
ax.scatter([r["pinn_eps_D"] for r in rows], y, s=64, color=C_PINN, edgecolor=SURF, lw=1.5, zorder=3, label="PINN (project code)")
for i, r in enumerate(rows):
    ax.text(1.02, i, f"{r['ratio_D']:.2f}×", va="center", fontsize=8.5, color=MUTED, transform=ax.get_yaxis_transform())
ax.set_yticks(y); ax.set_yticklabels([lab(r["key"]) for r in rows], fontsize=8.5, color=INK)
ax.set_xscale("log"); ax.set_xlim(0.02, 1.2)
ax.set_xticks([0.03,0.05,0.1,0.2,0.3,0.5,1.0]); ax.set_xticklabels(["0.03","0.05","0.1","0.2","0.3","0.5","1.0"])
ax.set_xlabel("relL2(D) = ‖D̂ − D‖ / ‖D‖   (lower is better, log scale)", color=MUTED)
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED, length=0)
ax.text(1.02, len(rows)-0.3, "PINN ÷\nadjoint", fontsize=8.5, color=MUTED, transform=ax.get_yaxis_transform(), va="bottom")
ax.set_title("Recovered-D error on 20 datasets: PINN never beats the classical adjoint", loc="left", fontsize=12, color=INK, pad=28)
ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False, fontsize=8.5, labelcolor=INK)
fig.tight_layout(); fig.savefig("figs/s20_relL2.png", dpi=150, facecolor=SURF)

# field panels for three datasets
import sys; sys.path.insert(0, "."); 
from problem import load, truth
keys = ["E_grf__Fo0.1__uniform8__n50__random__eta0.05","C_inclusion__Fo0.1__uniform8__n50__random__eta0.05","A_smooth__Fo0.1__uniform8__nfull__nodes__eta0.01"]
fig, axs = plt.subplots(3, 3, figsize=(10, 9.6), facecolor=SURF)
for i, k in enumerate(keys):
    ds = load(k); Dt = truth(ds)["D"]
    Da = np.load(f"results/S20/adjoint_{k}.npz")["D_hat"]; Dp = np.load(f"/home/claude/v2/pinn_proj/out/{k}/D_hat.npz")["D_hat"]
    r = next(r for r in rows if r["key"]==k)
    vmin, vmax = Dt.min(), Dt.max()
    for j, (D, t) in enumerate([(Dt, "true D"), (Da, f"classical  relL2 {r['adj_eps_D']:.3f}"), (Dp, f"PINN  relL2 {r['pinn_eps_D']:.3f}")]):
        im = axs[i, j].imshow(D.T, origin="lower", extent=[0,1,0,1], cmap="viridis", vmin=vmin, vmax=vmax)
        axs[i, j].set_title(t, fontsize=9.5, color=INK); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
        if j == 0: axs[i, j].set_ylabel(lab(k), fontsize=8.5, color=MUTED)
        if j == 0 and ds.x.size < 2000: axs[i, j].plot(ds.x, ds.y, ".", color="white", ms=2.5)
    fig.colorbar(im, ax=axs[i, :].tolist(), shrink=0.85, label="D")
fig.suptitle("Recovered diffusivity maps (same colour scale per row; dots = sensors)", fontsize=11.5, color=INK, x=0.02, ha="left")
fig.savefig("figs/s20_fields.png", dpi=130, facecolor=SURF)
print("ok")
