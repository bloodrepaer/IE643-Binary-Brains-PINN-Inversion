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
    st.image(png, width="stretch")


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


def results_figure(D_true, res, xy):
    names = list(res)
    ncol = len(names) + (1 if D_true is not None else 0)
    fig, axs = plt.subplots(2 if D_true is not None else 1, ncol, figsize=(4.2 * ncol, 7.6 if D_true is not None else 3.9), squeeze=False)
    allD = [r["D"] for r in res.values()] + ([D_true] if D_true is not None else [])
    vmin, vmax = min(float(d.min()) for d in allD), max(float(d.max()) for d in allD)
    c = 0
    if D_true is not None:
        im = axs[0, 0].imshow(D_true.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", vmin=vmin, vmax=vmax)
        map_axes(axs[0, 0], "true D"); fig.colorbar(im, ax=axs[0, 0], fraction=0.046)
        axs[1, 0].axis("off")
        axs[1, 0].text(0.0, 0.5, "bottom row:\nlog(D̂ / D)\nred = too high\nblue = too low\nwhite = correct", fontsize=10, va="center")
        c = 1
    for j, n in enumerate(names):
        r = res[n]; a = axs[0, c + j]
        im = a.imshow(r["D"].T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", vmin=vmin, vmax=vmax)
        if len(xy) < 2000:
            a.plot(xy[:, 0], xy[:, 1], "w.", ms=2.5)
        t = n
        if "relL2_D" in r["metrics"]:
            t += f"\nrelL2(D) = {r['metrics']['relL2_D']:.3f}"
        map_axes(a, t); fig.colorbar(im, ax=a, fraction=0.046)
        if D_true is not None:
            e = np.log(r["D"] / D_true); lim = max(0.05, float(np.percentile(np.abs(e), 99)))
            im = axs[1, c + j].imshow(e.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r", vmin=-lim, vmax=lim)
            map_axes(axs[1, c + j], f"{n}: log(D̂/D)"); fig.colorbar(im, ax=axs[1, c + j], fraction=0.046)
    fig.tight_layout()
    return fig


def profile_figure(D_true, res):
    g = np.linspace(0, 1, core.N_INV); mid = core.N_INV // 2
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.3))
    for ax, sl, lab in ((axs[0], np.s_[:, mid], "along x at y = 0.5"), (axs[1], np.s_[mid, :], "along y at x = 0.5")):
        if D_true is not None:
            ax.plot(g, D_true[sl], "k-", lw=2, label="true D")
        for n, r in res.items():
            ax.plot(g, r["D"][sl], "--", lw=1.8, color=C_CLASSICAL if n.startswith("classical") else C_PINN, label=n)
        ax.set_title(lab, fontsize=10); ax.set_xlabel("position"); ax.set_ylabel("D"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------------------------
st.title("Recover a diffusivity map D(x, y) from sensor readings")
st.caption("u_t = ∇·(D ∇u) on the unit square, Dirichlet boundaries, known initial temperature. "
           "Compare the classical PDE-constrained (adjoint) inversion with a physics-informed neural network (PINN).")

tab_data, tab_run, tab_res, tab_load, tab_help = st.tabs(
    ["1 · Data", "2 · Run inversion", "3 · Results & downloads", "4 · Load a saved model", "Help"])

# ---------------------------------------------------------------------------------------------
# 1 DATA
# ---------------------------------------------------------------------------------------------
with tab_data:
    src = st.radio("Where do the sensor readings come from?",
                   ["Generate a new synthetic dataset", "Pick a benchmark dataset", "Upload my own readings"],
                   horizontal=True, key="src")

    if src == "Generate a new synthetic dataset":
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
        st.image(fig_png(data_figure(ss.data, ds, tr)), width="stretch")
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
        st.write(f"Dataset: `{ss.data['key']}`")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Classical adjoint")
            use_c = st.checkbox("Run the classical inversion", value=True)
            K = st.select_slider("Control grid (D has K×K unknowns)", [8, 12, 16, 24, 32], value=16)
            reg_c = st.radio("Regulariser", ["tikhonov", "tv"], horizontal=True,
                             help="Tikhonov favours smooth fields; total variation (TV) favours sharp edges.")
            maxiter = st.slider("L-BFGS iterations per α", 30, 300, 150, 10)
            st.caption("α is chosen by the discrepancy principle (fit the data to the noise level). Typically 20–120 s.")
        with c2:
            st.subheader("PINN")
            use_p = st.checkbox("Run the PINN", value=True)
            epochs = st.select_slider("Training epochs", [100, 300, 750, 1500, 3000, 6000], value=1500)
            reg_p = st.radio("Regulariser ", ["auto", "tikhonov", "tv", "none"], horizontal=True,
                             help="auto = the project's defaults: Tikhonov 2e-4 for smooth tiers (A, B, E), TV 3e-4 otherwise.")
            lam = st.number_input("Regularisation weight (blank = default)", value=None, format="%.1e", placeholder="default")
            pseed = st.number_input("Random seed", value=42, step=1)
            st.caption(f"About {epochs / 1500 * 2:.0f}–{epochs / 1500 * 4:.0f} min on a CPU (1,500 epochs ≈ 2–4 min).")
        if st.button("Run", type="primary", disabled=not (use_c or use_p)):
            ss.results, ss.figs = {}, {}
            with st.status("Running …", expanded=True) as status:
                try:
                    if core.truth(ss.data) is not None and ss.baseline is None:
                        st.write("Best constant D (baseline) …")
                        t0 = time.perf_counter()
                        ss.baseline = core.constant_baseline(ss.data)
                        ss.baseline["wall_s"] = time.perf_counter() - t0
                    if use_c:
                        st.write("Classical adjoint inversion …")
                        ss.results["classical adjoint"] = core.run_classical(ss.data, K, reg_c, maxiter)
                        st.write(f"✓ classical done in {ss.results['classical adjoint']['metrics']['wall_s']:.0f} s")
                    if use_p:
                        st.write("Training the PINN …")
                        ss.results["PINN"] = core.run_pinn(ss.data, epochs, None if reg_p == "auto" else reg_p,
                                                           None if (lam is None or reg_p == "auto") else lam, int(pseed))
                        st.write(f"✓ PINN done in {ss.results['PINN']['metrics']['wall_s']:.0f} s")
                    status.update(label="Done — see **3 · Results & downloads**", state="complete")
                except Exception as e:
                    status.update(label=f"Failed: {e}", state="error")
                    st.exception(e)

# ---------------------------------------------------------------------------------------------
# 3 RESULTS
# ---------------------------------------------------------------------------------------------
with tab_res:
    if not ss.results:
        st.info("Run an inversion in **2 · Run inversion** first.")
    else:
        ds = core.dataset(ss.data); tr = core.truth(ss.data); D_true = tr["D"] if tr else None
        res = ss.results
        st.subheader("Performance")
        cols = st.columns(len(res))
        for col, (n, r) in zip(cols, res.items()):
            m = r["metrics"]
            with col:
                st.markdown(f"**{n}**")
                if "relL2_D" in m:
                    st.metric("relL2(D) — lower is better", f"{m['relL2_D']:.3f}",
                              help="‖D̂ − D‖ / ‖D‖ on the 65×65 grid. 0 is perfect.")
                if m.get("rms_replay_over_sigma") is not None:
                    st.metric("RMS replay / σ — about 1 is ideal", f"{m['rms_replay_over_sigma']:.2f}",
                              help="D̂ is run through the Crank–Nicolson solver and compared with the readings. "
                                   "≈1: explains the data to the noise level; >1: does not explain the data; <1: fits the noise.")
                else:
                    st.metric("RMS replay (noise-free data)", f"{m['rms_replay']:.2e}")
                st.caption(f"{m['wall_s']:.0f} s")
        rows = {}
        for n, r in res.items():
            m = r["metrics"]
            rows[n] = {"relL2(D) ↓": m.get("relL2_D"), "RMS of log D error ↓": m.get("rmse_logD"),
                       "RMS replay / σ (≈1)": m.get("rms_replay_over_sigma"), "RMS replay": m["rms_replay"],
                       "D̂ range": f"{m['D_min']:.3f} – {m['D_max']:.3f}", "wall time (s)": round(m["wall_s"], 1)}
        if ss.baseline is not None:
            b = ss.baseline
            rows["best constant D (baseline)"] = {"relL2(D) ↓": b.get("relL2_D"), "RMS of log D error ↓": b.get("rmse_logD"),
                                                  "RMS replay / σ (≈1)": b.get("rms_replay_over_sigma"), "RMS replay": b["rms_replay"],
                                                  "D̂ range": f"{b['D_const']:.3f} (uniform)",
                                                  "wall time (s)": round(b.get("wall_s", float("nan")), 1)}
        st.dataframe(pd.DataFrame(rows).T, width="stretch")
        if ss.baseline is not None and "relL2_D" in ss.baseline:
            beat = [n for n, r in res.items() if r["metrics"].get("relL2_D", 9) < ss.baseline["relL2_D"]]
            st.caption("The baseline is the best single uniform D for these readings; a useful map should beat it. "
                       f"Beating it here: {', '.join(beat) if beat else 'none'}.")
        if "PINN" in res and res["PINN"]["metrics"].get("chi2_u_net") is not None:
            st.caption(f"PINN's own network fit to the readings (χ², not the score): {res['PINN']['metrics']['chi2_u_net']:.2f}. "
                       "The network can fit the readings while its D is wrong, which is why the replay metric above is used.")

        st.subheader("Recovered maps")
        show(results_figure(D_true, res, sensor_xy(ds)), "maps")
        show(profile_figure(D_true, res), "profiles")
        c1, c2 = st.columns(2)
        if "classical adjoint" in res:
            with c1:
                r = res["classical adjoint"]
                fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
                al = [p["alpha"] for p in r["path"]]; rs = [p["rms_over_sigma"] for p in r["path"]]
                axs[0].semilogx(al, rs, "o-", color=C_CLASSICAL); axs[0].axhline(1, color="k", ls="--", lw=1)
                axs[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                axs[0].axvline(r["metrics"]["alpha"], color=C_PINN, lw=1); axs[0].invert_xaxis()
                axs[0].set(xlabel="α (regularisation weight)", ylabel="RMS misfit / σ", title="choosing α: stop when misfit ≈ noise")
                axs[0].grid(alpha=0.3)
                lS = np.log10(r["sensitivity"].T + 1e-300)
                im = axs[1].imshow(lS, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", vmin=lS.max() - 4, vmax=lS.max())
                map_axes(axs[1], "where data inform D: log10 ∫|∇u|² dt"); fig.colorbar(im, ax=axs[1], fraction=0.046)
                fig.tight_layout(); show(fig, "classical_diagnostics")
        if "PINN" in res and res["PINN"]["history"] is not None:
            with c2:
                h = res["PINN"]["history"]
                fig, ax = plt.subplots(figsize=(5, 3.4))
                for col, lab in (("loss_total", "total"), ("loss_data", "data"), ("loss_pde", "PDE residual"), ("loss_ic", "initial cond.")):
                    if col in h:
                        ax.semilogy(h["epoch"], h[col], lw=1, label=lab)
                ax.set(xlabel="epoch", title="PINN training losses"); ax.legend(fontsize=8, ncol=2, loc="upper right"); ax.grid(alpha=0.3, which="both")
                fig.tight_layout(); show(fig, "pinn_losses")

        st.subheader("Downloads")
        z = core.results_zip(ss.data, res, ss.figs)
        st.download_button("Everything (models, D̂, metrics, figures) .zip", z, "inversion_results.zip", "application/zip", type="primary")
        c = st.columns(len(res))
        for col, (n, r) in zip(c, res.items()):
            with col:
                st.download_button(f"Trained model: {n}", Path(r["model_file"]).read_bytes(), Path(r["model_file"]).name)
                g = np.linspace(0, 1, core.N_INV); X, Y = np.meshgrid(g, g, indexing="ij")
                s = io.StringIO(); np.savetxt(s, np.c_[X.ravel(), Y.ravel(), r["D"].ravel()], delimiter=",", header="x,y,D", comments="", fmt="%.8g")
                st.download_button(f"D̂ map (CSV): {n}", s.getvalue(), f"D_hat_{'classical' if n.startswith('classical') else 'pinn'}.csv", "text/csv")

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
                st.image(fig_png(fig), width="stretch")
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
                    st.image(fig_png(results_figure(tr["D"], {name: dict(D=D, metrics=m)}, sensor_xy(core.dataset(target)))), width="stretch")
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
            st.dataframe(cmp[["key", "adj_eps_D", "pinn_eps_D", "const_eps_D", "adj_rms_sig", "pinn_rms_sig", "winner_D"]]
                         .rename(columns={"adj_eps_D": "classical relL2(D)", "pinn_eps_D": "PINN relL2(D)",
                                          "const_eps_D": "constant-D relL2", "adj_rms_sig": "classical RMS/σ",
                                          "pinn_rms_sig": "PINN RMS/σ", "winner_D": "lower relL2"}),
                         width="stretch", hide_index=True)

# ---------------------------------------------------------------------------------------------
# HELP
# ---------------------------------------------------------------------------------------------
with tab_help:
    st.markdown("""
**Problem.** The temperature u(x, y, t) on the unit square obeys u_t = ∇·(D(x, y)∇u), with the edges held at known values
and a known initial temperature. Sensors record u at a few points and times, with noise. The task is to recover D.

**Methods.**
- *Classical adjoint* (`adjoint_inversion.py`): D = exp(m), m on a K×K control grid. Every evaluation solves the heat
  equation exactly (Crank–Nicolson, 65×65 grid) and the gradient comes from the discrete adjoint. α is chosen by the
  discrepancy principle. The saved model is the control-grid vector θ (`classical_model.npz`).
- *PINN* (`pinn_inversion.py`): one network for u(x, y, t) and one for log D(x, y), trained on the readings plus the
  PDE residual, boundary and initial conditions. The saved model is the network checkpoint (`pinn_best.pt`).

**Metrics.**
- *RMS replay / σ* (always available): D̂ is run through the same solver and compared with the readings. About 1 means
  the map explains the data to the noise level.
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
