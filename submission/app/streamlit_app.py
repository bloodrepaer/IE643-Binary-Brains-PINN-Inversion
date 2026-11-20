"""
IE 643 -- interface for recovering a spatially varying diffusivity D(x, y) from sensor readings.

    streamlit run app/streamlit_app.py        (from the submission/ folder)

1 Data      novelty demo datasets (every method precomputed) | benchmark dataset |
            generate a new synthetic dataset | upload your own readings
2 Run       classical adjoint (Crank-Nicolson or backward Euler), PINN (standard or boundary-aware),
            and the novelty methods: combined PINN -> adjoint, joint adjoint + source, joint adjoint + walls
3 Results   all methods side by side on the same data: true vs recovered D, spatial error map,
            relL2(D) and replay RMS / sigma; source / wall estimates; downloads
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
import novelty_core as NC  # noqa: E402

st.set_page_config(page_title="Diffusivity inversion", page_icon="🌡️", layout="wide")


def _img(png):
    """st.image filling the column on any Streamlit version."""
    for kw in ({"width": "stretch"}, {"use_container_width": True}, {"use_column_width": True}):
        try:
            return st.image(png, **kw)
        except Exception:
            continue
    return st.image(png)


def _df(data, **kw):
    """st.dataframe filling the column on any Streamlit version."""
    for extra in ({"width": "stretch"}, {"use_container_width": True}):
        try:
            return st.dataframe(data, **kw, **extra)
        except Exception:
            continue
    return st.dataframe(data, **kw)

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
C_COMBINED, C_JOINT, C_BOUND, C_BE = "#1baf7a", "#6250d6", "#8a8984", "#b05f00"
ERRMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#184f95", "#2a78d6", "#86b6ef", "#f0efec", "#f19a99", "#e34948", "#9e2726"])


def color_of(name):
    """One colour per method family, the same in every figure."""
    import re
    n = name.lower()
    if n.startswith("bound"):
        return C_BOUND
    if "combined" in n:
        return C_COMBINED
    if re.search(r"\bjoint\b", n):          # not the "joint" inside "adjoint"
        return C_JOINT
    if "backward" in n:
        return C_BE
    if "pinn" in n:
        return C_PINN
    return C_CLASSICAL

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
    _img(png)


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


def results_figure(D_true, res, xy, emax=1.2):
    """True D, every method's recovered D (same colour scale) and its spatial error map log(D_hat / D)
    (same diverging scale for every method), with relL2(D) and replay RMS / sigma in each title."""
    names = list(res)
    ncol = len(names) + (1 if D_true is not None else 0)
    nrow = 2 if D_true is not None else 1
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.25 * nrow + 0.4), squeeze=False)
    if D_true is not None:
        vmin, vmax = float(D_true.min()), float(D_true.max()) * 1.05
    else:
        allD = np.concatenate([r["D"].ravel() for r in res.values()])
        vmin, vmax = np.percentile(allD, [2, 98])
    c = 0
    if D_true is not None:
        im = axs[0, 0].imshow(D_true.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", vmin=vmin, vmax=vmax)
        if len(xy) < 2000:
            axs[0, 0].plot(xy[:, 0], xy[:, 1], "w.", ms=3, mec="k", mew=0.3)
        map_axes(axs[0, 0], "true D (answer key)\n+ sensors"); fig.colorbar(im, ax=axs[0, 0], fraction=0.046)
        axs[1, 0].axis("off")
        axs[1, 0].text(0.0, 0.5, "row 2: spatial error\nlog(D̂ / D)\n\nred = too high\nblue = too low\nwhite = correct\n"
                       f"(clipped at ±{emax:g})", fontsize=10, va="center")
        c = 1
    for j, n in enumerate(names):
        r = res[n]; m = r["metrics"]; a = axs[0, c + j]
        im = a.imshow(r["D"].T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", vmin=vmin, vmax=vmax)
        sub = []
        if m.get("relL2_D") is not None:
            sub.append(f"relL2 {m['relL2_D']:.3f}")
        if m.get("rms_replay_over_sigma") is not None:
            sub.append(f"replay {m['rms_replay_over_sigma']:.2f}σ")
        a.set_title(n + ("\n" + " · ".join(sub) if sub else ""), fontsize=9.5, color=color_of(n))
        a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1]); a.set_aspect("equal")
        fig.colorbar(im, ax=a, fraction=0.046, extend="max" if r["D"].max() > vmax else "neither")
        if D_true is not None:
            e = np.log(np.maximum(r["D"], 1e-9) / D_true)
            im = axs[1, c + j].imshow(e.T, origin="lower", extent=[0, 1, 0, 1], cmap=ERRMAP, vmin=-emax, vmax=emax)
            if len(xy) < 2000:
                axs[1, c + j].plot(xy[:, 0], xy[:, 1], "k.", ms=1.5)
            map_axes(axs[1, c + j], "error map")
            fig.colorbar(im, ax=axs[1, c + j], fraction=0.046, extend="both")
    fig.tight_layout()
    return fig


def source_figure(data, res):
    """True heat source (if known) and every estimated source."""
    items = [(n, r["f"]) for n, r in res.items() if r.get("f") is not None]
    if not items:
        return None
    tf = NC.true_source(data)
    if tf is not None:
        items = [("true source (answer key)", tf)] + items
    if not items:
        return None
    vmax = max(float(np.max(f)) for _, f in items) or 1.0
    fig, axs = plt.subplots(1, len(items), figsize=(3.5 * len(items), 3.3), squeeze=False)
    for a, (n, f) in zip(axs[0], items):
        im = a.imshow(np.asarray(f).T, origin="lower", extent=[0, 1, 0, 1], cmap="magma", vmin=min(0, float(np.min(f))), vmax=vmax)
        map_axes(a, n); fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    return fig


def walls_figure(data, res):
    """Mean wall temperature of each edge over time: truth (if known) and every estimate."""
    items = [(n, r["G_bd"]) for n, r in res.items() if r.get("G_bd") is not None and not n.lower().startswith("bound")]
    if not items:
        return None
    tG = NC.true_walls_bd(data)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
    for k, edge in enumerate(("left", "right", "bottom", "top")):
        ax = axs[k]
        if tG is not None:
            t, E = NC.walls_by_edge(data, tG)
            ax.plot(t, E[edge], "k-", lw=2.2, label="true walls")
        for n, G in items:
            t, E = NC.walls_by_edge(data, G)
            ax.plot(t, E[edge], "--", lw=1.6, color=color_of(n), label=n)
        ax.set_title(f"{edge} wall", fontsize=10); ax.set_xlabel("t"); ax.grid(alpha=0.3)
    axs[0].set_ylabel("wall temperature")
    axs[0].legend(fontsize=7.5)
    fig.tight_layout()
    return fig


def profile_figure(D_true, res):
    g = np.linspace(0, 1, core.N_INV); mid = core.N_INV // 2
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.3))
    for ax, sl, lab in ((axs[0], np.s_[:, mid], "along x at y = 0.5"), (axs[1], np.s_[mid, :], "along y at x = 0.5")):
        if D_true is not None:
            ax.plot(g, D_true[sl], "k-", lw=2, label="true D")
        for n, r in res.items():
            ax.plot(g, r["D"][sl], "--", lw=1.8, color=color_of(n), label=n)
        ax.set_title(lab, fontsize=10); ax.set_xlabel("position"); ax.set_ylabel("D"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------------------------
st.title("Recover a diffusivity map D(x, y) from sensor readings")
st.caption("u_t = ∇·(D ∇u) + f on the unit square, Dirichlet walls, known initial temperature. "
           "Compare the classical PDE-constrained (adjoint) inversion, a physics-informed neural network (PINN) "
           "and the novelty methods (combined PINN→adjoint, joint inversion of hidden physics) on the same readings.")
with st.expander("What is known, measured and estimated"):
    st.markdown("""
| | standard setting | unknown-physics setting |
|---|---|---|
| **Known** (given to every method) | initial temperature u₀, wall temperatures g, source f = 0, noise level σ | u₀, σ |
| **Measured** | noisy readings d_k = u(x_k, y_k, t_k) + ε at the sensors | same |
| **Estimated** | the full field D(x, y) | D(x, y) **and** the hidden walls g or heat source f |

Data are simulated with RK4 on a 129×129 grid; every inversion uses a 65×65 grid (no "inverse crime").""")

tab_data, tab_run, tab_res, tab_load, tab_help = st.tabs(
    ["1 · Data", "2 · Run inversion", "3 · Results & downloads", "4 · Load a saved model", "Help"])

# ---------------------------------------------------------------------------------------------
# 1 DATA
# ---------------------------------------------------------------------------------------------
with tab_data:
    src = st.radio("Where do the sensor readings come from?",
                   ["Novelty demo datasets (all methods precomputed)", "Generate a new synthetic dataset",
                    "Pick a benchmark dataset", "Upload my own readings"],
                   horizontal=True, key="src")

    if src.startswith("Novelty demo"):
        if not NC.demo_available():
            st.warning("The demo bundle `app/demo/` is missing. Rebuild it from `extended_studies/` with "
                       "`python run.py novelty/build_app_bundle.py`.")
        else:
            st.write("The 34 datasets of the novelty experiments, with every method's estimate precomputed "
                     "(CN adjoint, PINN, combined PINN→adjoint, joint classical inversion, and the true-physics bound). "
                     "The precomputed estimates are re-scored live with the same scorer; you can also re-run any "
                     "method in **2 · Run inversion**.")
            c1, c2 = st.columns([1, 2])
            grp = c1.selectbox("Setting", list(NC.DEMO_GROUPS), format_func=NC.DEMO_GROUPS.get, key="demo_grp")
            keys = NC.demo_keys(grp)
            default = {"benchmark": "A_smooth__Fo0.1__uniform8__n50__random__eta0.05",
                       "hidden_source": "H3u-A_smooth-q0.3__Fo0.1__uniform16__n50__random__eta0.05",
                       "hidden_walls": "H2u-A_smooth-heatedleft-a1__Fo0.1__uniform16__n50__random__eta0.05"}[grp]
            dkey = c2.selectbox("Dataset", keys, index=keys.index(default) if default in keys else 0, key="demo_key")
            pre = st.checkbox("Load the precomputed results of every method (instant)", value=True, key="demo_pre")
            extra = st.checkbox("…including the other combined variants (warm start only, PINN as prior)", value=False,
                                key="demo_extra", disabled=grp != "benchmark")
            if st.button("Use this dataset", type="primary", key="demo_use"):
                ss.data = NC.load_demo(grp, dkey)
                ss.results, ss.figs, ss.baseline = {}, {}, None
                with st.spinner("Scoring the best constant D and the precomputed estimates …"):
                    t0 = time.perf_counter()
                    ss.baseline = core.constant_baseline(ss.data)
                    ss.baseline["wall_s"] = time.perf_counter() - t0
                    if pre:
                        ss.results = NC.demo_results(ss.data, include_extra=extra)
                st.success("Loaded." + (" Precomputed results are in **3 · Results & downloads**." if pre
                                        else " Go to **2 · Run inversion**."))

    elif src == "Generate a new synthetic dataset":
        st.write("Simulates the heat equation with a chosen true D (RK4 on a 129×129 grid), then samples noisy "
                 "sensors, exactly like the project's dataset generator. Takes 5–40 s.")
        c1, c2, c3 = st.columns(3)
        with c1:
            tier = st.selectbox("True D field", core.TIERS, index=core.TIERS.index("B_gaussian_bump"),
                                format_func=lambda k: TIER_LABELS.get(k, k))
            custom = st.file_uploader("…or upload your own D (65×65 or 129×129; .npy or .csv)", type=["npy", "csv"], key="customD")
            Fo = st.select_slider("Time horizon (Fourier number Fo = D̄·T)", [0.02, 0.05, 0.1, 0.2], value=0.1)
        with c2:
            nsel = st.selectbox("Number of sensors", ["5", "20", "50", "100", "full field (every grid node)", "custom"], index=2)
            n_sensors = "full" if nsel.startswith("full") else (st.number_input("custom count", 1, 100, 30) if nsel == "custom" else int(nsel))
            layout = st.selectbox("Sensor layout", ["random", "grid", "clustered", "boundary"], disabled=n_sensors == "full")
            protocol = st.selectbox("When sensors record", core.PROTOCOLS, index=core.PROTOCOLS.index("uniform8"),
                                    format_func=lambda k: PROTO_LABELS.get(k, k))
        with c3:
            eta = st.slider("Noise level (% of signal RMS)", 0.0, 20.0, 5.0, 0.5) / 100
            bc = st.selectbox("Boundary condition", core.BC_SCENARIOS,
                              format_func=lambda k: {"zero": "all edges held at 0", "heated_left": "left edge held at 1",
                                                     "ramp_left": "left edge ramps 0→1", "pulsed_top": "top edge pulsed"}[k])
            seed = st.number_input("Noise seed (blank = project default)", value=None, step=1, placeholder="default")
        if st.button("Generate dataset", type="primary"):
            try:
                customD = core.parse_grid(custom.getvalue(), custom.name) if custom is not None else None
                bar = st.progress(0.0, "Starting …")
                ss.data = core.generate_synthetic(tier, Fo, protocol, n_sensors, layout, eta, bc,
                                                  None if seed is None else int(seed), customD,
                                                  progress=lambda f, m: bar.progress(f, m))
                ss.results, ss.figs, ss.baseline = {}, {}, None
                st.success(f"Dataset ready: {ss.data['entry']['n_readings']} readings. Go to **2 · Run inversion**.")
            except Exception as e:
                st.error(f"Could not generate the dataset: {e}")

    elif src == "Pick a benchmark dataset":
        if not core.benchmark_available():
            st.warning("The benchmark folder `datasets_v2_rk4_zero/` was not found next to the app. "
                       "Create it with `python generate_dataset_v2.py` (about 3 minutes), or generate a single dataset instead.")
        else:
            man = pd.DataFrame(core.benchmark_manifest())
            c = st.columns(6)
            f_t = c[0].selectbox("True D", sorted(man.tier.unique()), format_func=lambda k: TIER_LABELS.get(k, k))
            m1 = man[man.tier == f_t]
            f_fo = c[1].selectbox("Fo", sorted(m1.Fo.unique()), index=sorted(m1.Fo.unique()).index(0.1) if 0.1 in set(m1.Fo) else 0)
            m2 = m1[m1.Fo == f_fo]
            f_p = c[2].selectbox("Times", sorted(m2.protocol.unique()), format_func=lambda k: PROTO_LABELS.get(k, k),
                                 index=sorted(m2.protocol.unique()).index("uniform8") if "uniform8" in set(m2.protocol) else 0)
            m3 = m2[m2.protocol == f_p]
            ns = sorted(m3.sensor_set.unique(), key=lambda s: (s == "full", int(s) if s != "full" else 0))
            f_n = c[3].selectbox("Sensors", ns, index=ns.index("50") if "50" in ns else 0)
            m4 = m3[m3.sensor_set == f_n]
            f_l = c[4].selectbox("Layout", sorted(m4.layout.unique()))
            m5 = m4[m4.layout == f_l]
            es = sorted(m5.eta.unique())
            f_e = c[5].selectbox("Noise", es, index=es.index(0.05) if 0.05 in es else 0, format_func=lambda v: f"{v * 100:g}%")
            key = m5[m5.eta == f_e].key.iloc[0]
            st.caption(f"Dataset key: `{key}`")
            if st.button("Use this dataset", type="primary"):
                ss.data = core.load_benchmark(key)
                ss.results, ss.figs, ss.baseline = {}, {}, None
                st.success("Loaded. Go to **2 · Run inversion**.")

    else:
        st.write("A CSV with one row per reading and the header `x,y,t,u` (optionally `sigma`). "
                 "x, y in [0, 1]; t > 0. Download a filled-in example to see the format.")
        ex = Path(__file__).resolve().parent / "sample_inputs" / "example_sensors.csv"
        if ex.exists():
            st.download_button("Example sensors.csv", ex.read_bytes(), "example_sensors.csv", "text/csv")
        up = st.file_uploader("Sensor readings (.csv)", type=["csv"], key="upcsv")
        c1, c2, c3 = st.columns(3)
        with c1:
            T = st.number_input("Final time T (≥ the latest reading)", min_value=1e-6, value=0.858, format="%.4f")
            sig_in = st.number_input("Noise std σ (used if the CSV has no sigma column)", min_value=0.0, value=0.001, format="%.6f")
        with c2:
            bc_u = st.selectbox("Boundary condition ", core.BC_SCENARIOS,
                                format_func=lambda k: {"zero": "all edges held at 0", "heated_left": "left edge held at 1",
                                                       "ramp_left": "left edge ramps 0→1", "pulsed_top": "top edge pulsed"}[k])
            ic_up = st.file_uploader("Initial temperature u₀ (65×65; optional — default: the benchmark's two Gaussian bumps)",
                                     type=["npy", "csv"], key="icup")
        with c3:
            Dt_up = st.file_uploader("True D for scoring (65×65; optional)", type=["npy", "csv"], key="dtup")
        if up is not None and st.button("Load readings", type="primary"):
            try:
                sens = core.parse_sensor_csv(up.getvalue())
                ic = core.parse_grid(ic_up.getvalue(), ic_up.name) if ic_up is not None else None
                Dt = core.parse_grid(Dt_up.getvalue(), Dt_up.name) if Dt_up is not None else None
                with st.spinner("Checking the readings and estimating the best constant D …"):
                    ss.data = core.load_upload(sens, T, None if "sigma" in sens else sig_in, bc_u, ic, Dt)
                ss.results, ss.figs, ss.baseline = {}, {}, None
                if ic is not None:
                    st.info("A custom initial condition was given. The project PINN's built-in base solution assumes "
                            "the benchmark's two Gaussian bumps, so expect the PINN to do poorly; the classical method is unaffected.")
                st.success(f"Loaded {len(sens['u'])} readings. Go to **2 · Run inversion**.")
            except Exception as e:
                st.error(f"Could not read the upload: {e}")

    if ss.data is not None:
        st.divider()
        ds = core.dataset(ss.data); tr = core.truth(ss.data); e = ss.data["entry"]
        st.subheader("Current dataset")
        c = st.columns(5)
        c[0].metric("Readings", f"{len(ds.u):,}")
        c[1].metric("Sensors", f"{len(sensor_xy(ds)):,}")
        c[2].metric("Times", len(np.unique(np.round(ds.t, 9))))
        c[3].metric("Noise σ", f"{ds.sigma:.2e}" if ds.sigma > 0 else "0 (noise-free)")
        c[4].metric("True D known", "yes" if tr is not None else "no")
        _img(fig_png(data_figure(ss.data, ds, tr)))
        dl1, dl2 = st.columns(2)
        try:
            dl1.download_button("Download sensor readings (CSV)", core.sensor_csv_bytes(ss.data),
                                f"{e['key']}_sensors.csv", "text/csv", key="dl_sensors")
            dl2.download_button("Download full dataset (.zip)", core.dataset_zip(ss.data),
                                f"{e['key']}_dataset.zip", "application/zip", key="dl_dataset")
        except Exception as err:
            st.warning(f"Dataset download unavailable: {err}")
        snaps = ss.data.get("snapshots")
        if snaps is not None:
            k = st.slider("Simulated temperature field at time step", 0, len(snaps["t"]) - 1, len(snaps["t"]) // 4)
            fig, ax = plt.subplots(figsize=(4, 3.4))
            im = ax.imshow(snaps["u"][k].T, origin="lower", extent=[0, 1, 0, 1], cmap="magma")
            map_axes(ax, f"u(x, y, t = {snaps['t'][k]:.3f})"); fig.colorbar(im, ax=ax, fraction=0.046)
            st.image(fig_png(fig), width=420)

# ---------------------------------------------------------------------------------------------
# 2 RUN
# ---------------------------------------------------------------------------------------------
with tab_run:
    if ss.data is None:
        st.info("Choose or create a dataset in **1 · Data** first.")
    else:
        phys = NC.physics_of(ss.data)
        st.write(f"Dataset: `{ss.data['key']}`")
        if phys != "standard":
            st.info({"hidden_source": "This dataset has a **hidden heat source**: the problem file says f = 0. "
                                      "A method that assumes the file is right must explain the extra heat with D.",
                     "hidden_walls": "This dataset has **hidden wall temperatures**: the problem file says the walls "
                                     "are at 0. A method that assumes the file is right must explain the heat with D."}[phys])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Classical adjoint")
            use_c = st.checkbox("Run the classical inversion", value=True)
            solver = st.radio("Time integrator", ["Crank–Nicolson (project)", "backward Euler (controlled comparison)"],
                              help="Everything else is identical. Backward Euler decays too slowly (1/(1+x) vs e^-x), "
                                   "so it mimics a smaller D and biases the inversion when its error exceeds the noise.")
            steps = st.selectbox("Time steps", ["project default (≥ 200)", 100, 200, 400])
            K = st.select_slider("Control grid (D has K×K unknowns)", [8, 12, 16, 24, 32], value=16,
                                 disabled=not solver.startswith("Crank"))
            reg_c = st.radio("Regulariser", ["tikhonov", "tv"], horizontal=True, disabled=not solver.startswith("Crank"),
                             help="Tikhonov favours smooth fields; total variation (TV) favours sharp edges.")
            maxiter = st.slider("L-BFGS iterations per α", 30, 300, 150, 10)
            st.caption("α is chosen by the discrepancy principle (fit the data to the noise level). Typically 20–120 s.")
        with c2:
            st.subheader("PINN")
            use_p = st.checkbox("Run the PINN", value=phys == "standard")
            variant = st.radio("Variant", ["standard (walls as given)", "boundary-aware (learns the walls)"],
                               index=0 if phys == "standard" else 1,
                               help="The boundary-aware PINN (Study 3) does not trust the problem file's walls.")
            epochs = st.select_slider("Training epochs", [100, 300, 750, 1500, 3000, 6000], value=1500)
            reg_p = st.radio("Regulariser ", ["auto", "tikhonov", "tv", "none"], horizontal=True,
                             disabled=variant.startswith("boundary"),
                             help="auto = the project's defaults: Tikhonov 2e-4 for smooth tiers (A, B, E), TV 3e-4 otherwise.")
            lam = st.number_input("Regularisation weight (blank = default)", value=None, format="%.1e", placeholder="default")
            pseed = st.number_input("Random seed", value=42, step=1)
            st.caption(f"About {epochs / 1500 * 2:.0f}–{epochs / 1500 * 4:.0f} min on a CPU (1,500 epochs ≈ 2–4 min).")
        with c3:
            st.subheader("Novelty methods")
            have_pinn = any("pinn" in n.lower() and "combined" not in n.lower() for n in ss.results)
            use_h = st.checkbox("Combined PINN → adjoint", value=False,
                                help="The PINN's D, projected onto the adjoint's 16×16 controls, starts the CN adjoint; "
                                     "the α path starts where the PINN estimate's misfit and regulariser balance. "
                                     "Uses the PINN from this run, or the last PINN result in the session.")
            if use_h and not (use_p or have_pinn):
                st.warning("Tick **Run the PINN** too: the combined method starts from the PINN's estimate.")
            use_js = st.checkbox("Joint: adjoint + hidden heat source f ≥ 0", value=phys == "hidden_source",
                                 help="The CN adjoint estimates D and a source field together (exact adjoint gradients).")
            use_jw = st.checkbox("Joint: adjoint + hidden wall temperatures", value=phys == "hidden_walls",
                                 help="The CN adjoint estimates D and each wall's temperature over time.")
            jw_from_pinn = st.checkbox("…start the walls from the boundary-aware PINN's walls", value=False,
                                       disabled=not use_jw)
            st.caption("Each joint inversion takes about 3–8 min; the combined method about 1–3 min after the PINN.")
        keep = st.checkbox("Keep the results already in the session (e.g. the precomputed ones) and add these",
                           value=True)
        if st.button("Run", type="primary", disabled=not (use_c or use_p or use_h or use_js or use_jw)):
            if not keep:
                ss.results, ss.figs = {}, {}
            with st.status("Running …", expanded=True) as status:
                try:
                    if core.truth(ss.data) is not None and ss.baseline is None:
                        st.write("Best constant D (baseline) …")
                        t0 = time.perf_counter()
                        ss.baseline = core.constant_baseline(ss.data)
                        ss.baseline["wall_s"] = time.perf_counter() - t0
                    n_steps = None if isinstance(steps, str) else int(steps)
                    if use_c:
                        if solver.startswith("Crank") and n_steps is None:
                            st.write("Classical adjoint (Crank–Nicolson) …")
                            name = "CN adjoint (live)"
                            ss.results[name] = core.run_classical(ss.data, K, reg_c, maxiter)
                        else:
                            sv = "CN" if solver.startswith("Crank") else "BE"
                            st.write(f"Classical adjoint ({'Crank–Nicolson' if sv == 'CN' else 'backward Euler'}) …")
                            name = f"{sv if sv == 'CN' else 'backward Euler'} adjoint (live)"
                            ss.results[name] = NC.run_classical_solver(ss.data, sv, n_steps)
                        st.write(f"✓ done in {ss.results[name]['metrics']['wall_s']:.0f} s")
                    pinn_name = None
                    if use_p:
                        if variant.startswith("boundary"):
                            st.write("Training the boundary-aware PINN …")
                            pinn_name = "PINN, boundary-aware (live)"
                            ss.results[pinn_name] = NC.run_pinn_bc(ss.data, epochs)
                        else:
                            st.write("Training the PINN …")
                            pinn_name = "PINN (live)"
                            ss.results[pinn_name] = core.run_pinn(ss.data, epochs, None if reg_p == "auto" else reg_p,
                                                                  None if (lam is None or reg_p == "auto") else lam, int(pseed))
                        st.write(f"✓ PINN done in {ss.results[pinn_name]['metrics']['wall_s']:.0f} s")
                    if use_h:
                        if pinn_name is None:
                            cands = [n for n in ss.results if "pinn" in n.lower() and "combined" not in n.lower()]
                            pinn_name = cands[-1] if cands else None
                        if pinn_name is None:
                            st.write("✗ combined method skipped: no PINN estimate")
                        else:
                            st.write(f"Combined method, starting from **{pinn_name}** …")
                            pr = ss.results[pinn_name]
                            ss.results["combined PINN→adjoint (live)"] = NC.run_combined(
                                ss.data, pr["D"], float(pr["metrics"].get("wall_s") or 0.0))
                            st.write(f"✓ combined done (adjoint stage "
                                     f"{ss.results['combined PINN→adjoint (live)']['metrics']['adjoint_wall_s']:.0f} s)")
                    if use_js:
                        st.write("Joint inversion: D + heat source …")
                        ss.results["joint: adjoint + source (live)"] = NC.run_joint_source(ss.data)
                        st.write("✓ joint (source) done")
                    if use_jw:
                        start = None
                        if jw_from_pinn:
                            cands = [r for n, r in ss.results.items() if r.get("G_bd") is not None and "pinn" in n.lower()]
                            start = cands[-1]["G_bd"] if cands else None
                            if start is None:
                                st.write("(no boundary-aware PINN walls in the session: starting from zero walls)")
                        st.write("Joint inversion: D + wall temperatures …")
                        nm = "combined: joint from PINN walls (live)" if start is not None else "joint: adjoint + walls (live)"
                        ss.results[nm] = NC.run_joint_walls(ss.data, start)
                        st.write("✓ joint (walls) done")
                    status.update(label="Done — see **3 · Results & downloads**", state="complete")
                except Exception as e:
                    status.update(label=f"Failed: {e}", state="error")
                    st.exception(e)
        if ss.results and st.button("Clear all results"):
            ss.results, ss.figs = {}, {}

# ---------------------------------------------------------------------------------------------
# 3 RESULTS
# ---------------------------------------------------------------------------------------------
with tab_res:
    if not ss.results:
        st.info("Run an inversion in **2 · Run inversion** first, or load a novelty demo dataset with its precomputed results.")
    else:
        ds = core.dataset(ss.data); tr = core.truth(ss.data); D_true = tr["D"] if tr else None
        res = ss.results
        st.subheader("Performance — every method on the same readings, grid, source and boundary settings")
        per_row = 4
        names = list(res)
        for r0 in range(0, len(names), per_row):
            cols = st.columns(per_row)
            for col, n in zip(cols, names[r0:r0 + per_row]):
                m = res[n]["metrics"]
                with col:
                    st.markdown(f"<span style='color:{color_of(n)}'>■</span> **{n}**", unsafe_allow_html=True)
                    if m.get("relL2_D") is not None:
                        st.metric("relL2(D) — lower is better", f"{m['relL2_D']:.3f}",
                                  help="‖D̂ − D‖ / ‖D‖ on the 65×65 grid. 0 is perfect.")
                    if m.get("rms_replay_over_sigma") is not None:
                        st.metric("replay RMS / σ — about 1 is ideal", f"{m['rms_replay_over_sigma']:.2f}",
                                  help="D̂ is re-simulated (with the source or walls this method estimated, if any) and "
                                       "compared with the readings. ≈1: explains the data to the noise level.")
                    else:
                        st.metric("replay RMS (noise-free data)", f"{m['rms_replay']:.2e}")
                    w = m.get("wall_s")
                    st.caption(("precomputed · " if m.get("precomputed") else "") +
                               (f"{w:.0f} s" if w is not None and np.isfinite(w) else ""))
        rows = {}
        for n, r in res.items():
            m = r["metrics"]
            rows[n] = {"relL2(D) ↓": m.get("relL2_D"), "RMS of log D error ↓": m.get("rmse_logD"),
                       "replay RMS / σ (≈1)": m.get("rms_replay_over_sigma"),
                       "replayed with": m.get("replay_physics", "problem file"),
                       "D̂ range": f"{m['D_min']:.3f} – {m['D_max']:.3f}",
                       "wall time (s)": None if m.get("wall_s") is None else round(float(m["wall_s"]), 1)}
        if ss.baseline is not None:
            b = ss.baseline
            rows["best constant D (baseline)"] = {"relL2(D) ↓": b.get("relL2_D"), "RMS of log D error ↓": b.get("rmse_logD"),
                                                  "replay RMS / σ (≈1)": b.get("rms_replay_over_sigma"),
                                                  "replayed with": "problem file",
                                                  "D̂ range": f"{b['D_const']:.3f} (uniform)",
                                                  "wall time (s)": round(b.get("wall_s", float("nan")), 1)}
        _df(pd.DataFrame(rows).T)
        if ss.baseline is not None and "relL2_D" in ss.baseline:
            beat = [n for n, r in res.items() if r["metrics"].get("relL2_D", 9) < ss.baseline["relL2_D"]]
            st.caption("The baseline is the best single uniform D for these readings; a useful map should beat it. "
                       f"Beating it here: {', '.join(beat) if beat else 'none'}.")
        # good fit at the sensors but wrong field: the instructor's point, made explicit
        if D_true is not None:
            ref = ss.baseline["relL2_D"] if ss.baseline is not None and "relL2_D" in ss.baseline else 0.5
            fooled = [n for n, r in res.items() if (r["metrics"].get("rms_replay_over_sigma") or 9) < 1.3
                      and (r["metrics"].get("relL2_D") or 0) > max(ref, 0.5)]
            for n in fooled:
                m = res[n]["metrics"]
                st.warning(f"**{n}** explains the readings to the noise level (replay {m['rms_replay_over_sigma']:.2f}σ) "
                           f"but its D is far from the truth (relL2 {m['relL2_D']:.2f}). A good fit at the sensors does not "
                           "prove the field is recovered: check the error map below.")

        st.subheader("True vs recovered full fields and spatial error maps")
        show(results_figure(D_true, res, sensor_xy(ds)), "maps")
        fsrc = source_figure(ss.data, res)
        if fsrc is not None:
            st.subheader("Heat source: true vs estimated")
            show(fsrc, "sources")
        fw = walls_figure(ss.data, res)
        if fw is not None:
            st.subheader("Wall temperatures: true vs estimated")
            show(fw, "walls")
        st.subheader("Profiles through the centre")
        show(profile_figure(D_true, res), "profiles")
        c1, c2 = st.columns(2)
        cl = [n for n, r in res.items() if r.get("path") and r.get("sensitivity") is not None]
        if cl:
            with c1:
                r = res[cl[-1]]
                fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
                al = [p["alpha"] for p in r["path"]]; rs = [p["rms_over_sigma"] for p in r["path"]]
                axs[0].semilogx(al, rs, "o-", color=C_CLASSICAL); axs[0].axhline(1, color="k", ls="--", lw=1)
                axs[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                axs[0].axvline(r["metrics"]["alpha"], color=C_PINN, lw=1); axs[0].invert_xaxis()
                axs[0].set(xlabel="α (regularisation weight)", ylabel="RMS misfit / σ",
                           title=f"{cl[-1]}: stop when misfit ≈ noise")
                axs[0].grid(alpha=0.3)
                lS = np.log10(r["sensitivity"].T + 1e-300)
                im = axs[1].imshow(lS, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", vmin=lS.max() - 4, vmax=lS.max())
                map_axes(axs[1], "where data inform D: log10 ∫|∇u|² dt"); fig.colorbar(im, ax=axs[1], fraction=0.046)
                fig.tight_layout(); show(fig, "classical_diagnostics")
        pl = [n for n, r in res.items() if r.get("history") is not None]
        if pl:
            with c2:
                h = res[pl[-1]]["history"]
                fig, ax = plt.subplots(figsize=(5, 3.4))
                for col, lab in (("loss_total", "total"), ("loss_data", "data"), ("loss_pde", "PDE residual"), ("loss_ic", "initial cond.")):
                    if col in h:
                        ax.semilogy(h["epoch"], h[col], lw=1, label=lab)
                ax.set(xlabel="epoch", title="PINN training losses"); ax.legend(fontsize=8, ncol=2, loc="upper right"); ax.grid(alpha=0.3, which="both")
                fig.tight_layout(); show(fig, "pinn_losses")

        st.subheader("Downloads")
        z = core.results_zip(ss.data, res, ss.figs)
        st.download_button("Everything (models, D̂, metrics, figures) .zip", z, "inversion_results.zip", "application/zip", type="primary")
        g = np.linspace(0, 1, core.N_INV); X, Y = np.meshgrid(g, g, indexing="ij")
        for r0 in range(0, len(names), per_row):
            cols = st.columns(per_row)
            for col, n in zip(cols, names[r0:r0 + per_row]):
                r = res[n]
                tag = "".join(ch if ch.isalnum() else "_" for ch in n).strip("_").lower()
                with col:
                    if r.get("model_file"):
                        st.download_button(f"Model: {n}", Path(r["model_file"]).read_bytes(), Path(r["model_file"]).name,
                                           key=f"dlm_{tag}")
                    s_ = io.StringIO(); np.savetxt(s_, np.c_[X.ravel(), Y.ravel(), r["D"].ravel()], delimiter=",",
                                                   header="x,y,D", comments="", fmt="%.8g")
                    st.download_button(f"D̂ map (CSV): {n}", s_.getvalue(), f"D_hat_{tag}.csv", "text/csv", key=f"dld_{tag}")

# ---------------------------------------------------------------------------------------------
# 4 LOAD A SAVED MODEL
# ---------------------------------------------------------------------------------------------
with tab_load:
    PRE = Path(__file__).resolve().parent / "pretrained"
    pre_keys = sorted(p.name for p in PRE.iterdir() if p.is_dir()) if PRE.exists() else []
    how = st.radio("Model source", ["Upload a saved model file", "Bundled pre-trained model (20 benchmark datasets)"],
                   horizontal=True, key="load_src", disabled=not pre_keys)
    raw = name = bench_key = None
    if how.startswith("Upload"):
        st.write("Open a model saved from tab 3: `classical_model.npz` (adjoint parameters on the control grid) or "
                 "`pinn_best.pt` (PINN checkpoint). The D map is rebuilt from the model itself.")
        mf = st.file_uploader("Saved model", type=["npz", "pt"], key="model_up")
        if mf is not None:
            raw, name = mf.getvalue(), mf.name
    else:
        st.write("Models trained for the 20-dataset comparison: a PINN checkpoint (1,500 epochs) and the classical "
                 "adjoint estimate for each dataset.")
        c1, c2 = st.columns([3, 1])
        bench_key = c1.selectbox("Dataset", pre_keys, key="pre_key")
        meth = c2.radio("Method", ["PINN (pinn_best.pt)", "classical adjoint (D̂)"], key="pre_meth")
        f = PRE / bench_key / ("pinn_best.pt" if meth.startswith("PINN") else "adjoint_D_hat.npz")
        raw, name = f.read_bytes(), f.name
    if raw is not None:
        try:
            D, info = core.load_model_file(raw, name)
            c1, c2 = st.columns([1, 1])
            with c1:
                fig, ax = plt.subplots(figsize=(4.4, 3.8))
                im = ax.imshow(D.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
                map_axes(ax, f"D from {name}"); fig.colorbar(im, ax=ax, fraction=0.046)
                _img(fig_png(fig))
            with c2:
                st.json(info)
                st.metric("D range", f"{D.min():.3f} – {D.max():.3f}")
                st.metric("D mean", f"{D.mean():.3f}")
                target = None
                if bench_key and core.benchmark_available():
                    if st.button("Score on its benchmark dataset"):
                        target = core.load_benchmark(bench_key)
                elif ss.data is not None and st.button("Score this model on the current dataset"):
                    target = ss.data
                if target is not None:
                    with st.spinner("Replaying D̂ through the solver …"):
                        m, _, _ = core.score(target, D)
                    ss.load_score = (name, bench_key, m, target)
            sc = ss.get("load_score")
            if sc is not None and sc[0] == name and sc[1] == bench_key:
                m, target = sc[2], sc[3]
                cc = st.columns(3)
                if "relL2_D" in m:
                    cc[0].metric("relL2(D) — lower is better", f"{m['relL2_D']:.3f}")
                    cc[1].metric("RMS of log D error", f"{m['rmse_logD']:.3f}")
                if m.get("rms_replay_over_sigma") is not None:
                    cc[2].metric("RMS replay / σ — about 1 is ideal", f"{m['rms_replay_over_sigma']:.2f}")
                tr = core.truth(target)
                if tr is not None:
                    _img(fig_png(results_figure(tr["D"], {name: dict(D=D, metrics=m)}, sensor_xy(core.dataset(target)))))
            g = np.linspace(0, 1, core.N_INV); X, Y = np.meshgrid(g, g, indexing="ij")
            s = io.StringIO(); np.savetxt(s, np.c_[X.ravel(), Y.ravel(), D.ravel()], delimiter=",", header="x,y,D", comments="", fmt="%.8g")
            st.download_button("D map (CSV)", s.getvalue(), "D_from_model.csv", "text/csv")
            if bench_key:
                st.download_button(f"Model file ({name})", raw, f"{bench_key}__{name}")
        except Exception as e:
            st.error(f"Could not open the model: {e}")
    if (PRE / "comparison_20datasets.csv").exists():
        with st.expander("Scores of the bundled models (20-dataset comparison)"):
            cmp = pd.read_csv(PRE / "comparison_20datasets.csv")
            _df(cmp[["key", "adj_eps_D", "pinn_eps_D", "const_eps_D", "adj_rms_sig", "pinn_rms_sig", "winner_D"]]
                         .rename(columns={"adj_eps_D": "classical relL2(D)", "pinn_eps_D": "PINN relL2(D)",
                                          "const_eps_D": "constant-D relL2", "adj_rms_sig": "classical RMS/σ",
                                          "pinn_rms_sig": "PINN RMS/σ", "winner_D": "lower relL2"}),
                         hide_index=True)

# ---------------------------------------------------------------------------------------------
# HELP
# ---------------------------------------------------------------------------------------------
with tab_help:
    st.markdown("""
**Problem.** The temperature u(x, y, t) on the unit square obeys u_t = ∇·(D(x, y)∇u) + f, with the edges held at known
values, a known initial temperature and (in the standard setting) no source. Sensors record u at a few points and times, with noise. The task is to recover D.

**Methods.**
- *Classical adjoint* (`adjoint_inversion.py`): D = exp(m), m on a K×K control grid. Every evaluation solves the heat
  equation exactly (Crank–Nicolson, 65×65 grid) and the gradient comes from the discrete adjoint. α is chosen by the
  discrepancy principle. The saved model is the control-grid vector θ (`classical_model.npz`).
- *PINN* (`pinn_inversion.py`): one network for u(x, y, t) and one for log D(x, y), trained on the readings plus the
  PDE residual, boundary and initial conditions. The saved model is the network checkpoint (`pinn_best.pt`).

**Novelty methods** (`novelty_core.py`, solver code in `methods/`; all on the same dataset folder as above).
- *Combined PINN → adjoint*: the PINN's log D, projected onto the adjoint's 16×16 controls, starts the CN adjoint; the α
  path starts where the PINN estimate's data misfit and regulariser balance ("hybrid_balance" in the novelty study).
- *Joint: adjoint + source*: the CN adjoint estimates D and a heat source f ≥ 0 on 16×16 controls, both with exact
  discrete-adjoint gradients. For datasets whose problem file omits a heater.
- *Joint: adjoint + walls*: the CN adjoint estimates D and each wall's temperature over time (17 knots). For datasets
  whose problem file has the wrong wall temperatures. Can start from the boundary-aware PINN's walls (combined).
- *Backward-Euler adjoint*: identical to the classical adjoint except for the time step. Its error relative to the
  noise biases D (it mimics a smaller D); with Crank–Nicolson the bias disappears.
- *Boundary-aware PINN*: the Study 3 PINN that learns the wall values instead of assuming them.

**Metrics.**
- *RMS replay / σ* (always available): D̂ is run through the same solver (with the source or walls the method itself
  estimated, if any) and compared with the readings. About 1 means the map explains the data to the noise level.
  It does **not** prove that D is right: an adjoint that wrongly assumes no source can replay at 1σ with a meaningless
  D, which is why the full field and its error map are always shown.
- *relL2(D)* and *RMS of log D error* (when the true D is known): distance from the true map. Lower is better.
- *Best constant D*: the single uniform value that fits the readings best — the floor any useful map should beat.

**Input formats.**
- Sensor CSV: header `x,y,t,u[,sigma]`, one row per reading; x, y in [0, 1], t > 0.
- Fields (custom true D, initial condition, true D for scoring): `.npy` array of shape 65×65 (or 129×129 for a true D),
  or `.csv` either as a 65-row matrix or as `x,y,value` columns.

**Notes.** Synthetic data are simulated on a finer 129×129 grid with a different time integrator (RK4) than the
inversion uses, to avoid an "inverse crime". The PINN's built-in base solution assumes the benchmark's initial
condition (two Gaussian bumps); with a different initial condition prefer the classical method.
""")
