"""
IE 643 -- interface for recovering a spatially varying diffusivity D(x, y) from sensor readings.

    streamlit run app/streamlit_app.py        (from the v2/ folder)

1 Data      benchmark dataset | generate a new synthetic dataset | upload your own readings
2 Run       classical adjoint and/or PINN (the project's own solvers, unchanged)
3 Results   displayed metric (RMS_replay / sigma), field metrics if the true D is known,
            recovered maps, error maps, convergence; download trained models and D_hat
4 Load      open a saved model (.npz classical, .pt PINN) and view / export its D map
"""
import io, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

st.set_page_config(page_title="Diffusivity inversion", page_icon="🌡️", layout="wide")

TIER_LABELS = {
    "A_smooth": "A · smooth sinusoid", "B_gaussian_bump": "B · Gaussian bump", "C_inclusion": "C · circular inclusion",
    "D_layered": "D · three layers", "E_grf": "E · Gaussian random field", "F_defects": "F · plate with 3 defects",
    "C_inclusion_contrast_low": "C · inclusion, contrast 2", "C_inclusion_contrast_high": "C · inclusion, contrast 15",
    "D_checkerboard_coarse": "D · checkerboard 2×2", "D_checkerboard_fine": "D · checkerboard 6×6",
    "E_grf_corrlen_short": "E · random field, short correlation", "E_grf_corrlen_long": "E · random field, long correlation",
    "F_defects_single": "F · one defect", "F_defects_many": "F · six small defects", "F_defects_small": "F · tiny defects",
    "F_defects_low_contrast": "F · low-contrast defects", "F_defects_voids": "F · voids (5× lower D)",
}
PROTO_LABELS = {"uniform8": "8 evenly spaced times", "uniform4": "4 evenly spaced times", "uniform16": "16 evenly spaced times",
                "uniform40": "40 evenly spaced times", "uniform80": "80 evenly spaced times", "uniform200": "200 evenly spaced times",
                "log8": "8 times, dense early", "early8": "8 times in the first quarter", "late8": "8 times in the second half",
                "final1": "final snapshot only"}
C_CLASSICAL, C_PINN = "#2a78d6", "#eb6834"

ss = st.session_state
ss.setdefault("data", None)
ss.setdefault("results", {})
ss.setdefault("figs", {})
ss.setdefault("baseline", None)


# ---------------------------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------------------------
def fig_png(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return b.getvalue()


def show(fig, key):
    png = fig_png(fig)
    ss.figs[key] = png
    st.image(png, use_column_width=True)


def sensor_xy(ds):
    return np.unique(np.c_[ds.x, ds.y], axis=0)


def map_axes(ax, title):
    ax.set_title(title, fontsize=10); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_aspect("equal")


def data_figure(data, ds, tr):
    xy = sensor_xy(ds)
    fig, axs = plt.subplots(1, 2 if tr is None else 3, figsize=(12 if tr is not None else 8.5, 3.6))
    k = 0
    if tr is not None:
        im = axs[0].imshow(tr["D"].T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
        if len(xy) < 2000:
            axs[0].plot(xy[:, 0], xy[:, 1], "w.", ms=4, mec="k", mew=0.3)
        for d in tr["defects"]:
            axs[0].add_patch(plt.Circle((d["cx"], d["cy"]), d["r"], fill=False, ec="w", ls="--", lw=1))
        map_axes(axs[0], "true D (answer key) + sensors"); fig.colorbar(im, ax=axs[0], fraction=0.046)
        k = 1
    im = axs[k].imshow(ds.u0.T, origin="lower", extent=[0, 1, 0, 1], cmap="magma")
    if len(xy) < 2000:
        axs[k].plot(xy[:, 0], xy[:, 1], "c.", ms=4)
    map_axes(axs[k], "initial temperature u₀ + sensors"); fig.colorbar(im, ax=axs[k], fraction=0.046)
    ax = axs[k + 1]
    sid = np.unique(np.c_[ds.x, ds.y], axis=0, return_inverse=True)[1].ravel()
    for s in np.unique(sid)[:12]:
        mk = sid == s
        o = np.argsort(ds.t[mk])
        ax.plot(ds.t[mk][o], ds.u[mk][o], ".-", lw=0.8, ms=3, alpha=0.8)
    ax.set_xlabel("t"); ax.set_ylabel("reading u"); ax.set_title(f"readings over time (first {min(12, len(np.unique(sid)))} sensors)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
