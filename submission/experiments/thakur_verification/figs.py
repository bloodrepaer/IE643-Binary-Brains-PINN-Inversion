"""Figures and summary table for the Thakur et al. verification."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/claude/thakur/runs"; F = "/home/claude/thakur/figs"; os.makedirs(F, exist_ok=True)
PAPER = {"I": 4.8e-2, "II": 4.6e-2, "III": 6.31e-2, "IV": 4.58e-2}
PAPER_T2 = {250000: 4.89e-2, 25000: 5.9e-2, 2500: 6.0e-2, 250: 6.45e-2, 25: 3.9e-1}


def J(p):
    return json.load(open(p)) if os.path.exists(p) else None


# ---------- Figure 1: D maps, true vs classical vs PINN (paper cases) ----------
cases = ["I", "II", "III", "IV"]
fig, axs = plt.subplots(4, 4, figsize=(13, 12))
for r, c in enumerate(cases):
    eq = np.load(f"{R}/eqerr_{c}_noise0.0.npz"); ad = np.load(f"{R}/adjoint_{c}_n2500_noise0.0_s0.npz")
    pc = f"{R}/perfectc_{c}_0.01.npz"
    panels = [("true D", eq["D_true"]), ("classical FD + LS (full grid)", eq["D_pred"]),
              ("classical adjoint (2,500 pts)", ad["D_pred"]),
              ("paper's loss, perfect c, Δt = 0.01", np.load(pc)["D_pred"] if os.path.exists(pc) else None)]
    Dt = eq["D_true"]; vmin, vmax = Dt.min(), Dt.max()
    for k, (title, D) in enumerate(panels):
        ax = axs[r, k]
        if D is None:
            ax.axis("off"); continue
        if D.shape != Dt.shape:
            ax.axis("off"); continue
        im = ax.imshow(D.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        if k > 0:
            e = np.sqrt(np.mean((Dt - D) ** 2) / np.mean((Dt - Dt.mean()) ** 2))
            title += f"\nL = {e * 100:.1f}%"
        else:
            title = f"Case {c}: " + title + f"\n(paper's PINN: L = {PAPER[c] * 100:.2f}%)"
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Recovered D on Thakur et al.'s four test cases (L = paper's eq. 7 error; lower is better)", fontsize=12)
fig.tight_layout(); fig.savefig(f"{F}/maps.png", dpi=120); plt.close(fig)

# ---------- Figure 2: dt bias of the backward-Euler loss with a perfect c ----------
fig, ax = plt.subplots(figsize=(6, 4))
dts, Ls = [], []
for m in ["0.001", "0.002", "0.005", "0.01"]:
    r = J(f"{R}/perfectcfine_I_{m}.json")
    if r: dts.append(float(m)); Ls.append(r["L_D"] * 100)
ax.plot(dts, Ls, "o-", label="paper's backward-Euler loss (eq. 5)")
for m, lab in (("0.02", None), ("0.05", None)):
    r = J(f"{R}/perfectc_I_{m}.json")
    if r: ax.plot(float(m), r["L_D"] * 100, "o", color="C0")
rc = J(f"{R}/perfectcfine_I_cont.json")
if rc: ax.axhline(rc["L_D"] * 100, color="C2", ls="--", label="continuous-time residual (no Δt)")
ax.axhline(PAPER["I"] * 100, color="k", ls=":", label="paper's reported PINN, Case I")
ax.set(xscale="log", xlabel="Δt in eq. (5)", ylabel="error L(D) [%]",
       title="Case I, exact concentration field, only D trained")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(f"{F}/dt_bias.png", dpi=130); plt.close(fig)

# ---------- summary table ----------
rows = []
for c in cases:
    rows.append(dict(case=c, paper_PINN=PAPER[c],
                     eqerr=(J(f"{R}/eqerr_{c}_noise0.0.json") or {}).get("path", [{}])[0].get("L_D"),
                     adjoint_2500=(J(f"{R}/adjoint_{c}_n2500_noise0.0_s0.json") or {}).get("L_D"),
                     perfect_c_be001=(J(f"{R}/perfectc_{c}_0.01.json") or {}).get("L_D"),
                     perfect_c_cont=(J(f"{R}/perfectc_{c}_cont.json") or {}).get("L_D")))
json.dump(rows, open(f"{F}/summary.json", "w"), indent=1)
for r in rows:
    print(r)
print("full PINN runs:")
for p in sorted(glob.glob(f"{R}/full_*.json")):
    r = J(p); print(os.path.basename(p), {k: r[k] for k in ("L_D", "L_c", "relL2_D", "noise", "dt", "iters")})
