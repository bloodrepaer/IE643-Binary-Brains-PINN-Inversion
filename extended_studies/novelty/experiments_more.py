"""
experiments_more.py -- E2 (temperature dependence) and E3 (data quality) for run_novelty.py.

E2a  third network on the standard data. Truth has no temperature dependence, so g(u) should come out
     near zero and the error should not improve. Arms: pinn, pinn_T.
E2b  Arrhenius data (novelty/arrhenius.py), strengths S = 0, 0.5, 1, 2. Arms: adjoint and pinn (both
     assume D(x, y)), pinn_T (D(x, y, u) = exp(m(x, y) + g(u))). Scored against the reference field
     D0 = D(x, y, u = 0); pinn_T also on its g(u) against the true s(u).
E3a  sensor-split validation: 20 % of sensor locations held out. PINN: the 10 loss-weight settings of
     LAMBDA_SWEEP_GRID trained on 80 %, the one with the lowest held-out replay refitted on 100 %
     (pinn_selected), against the fixed default (pinn_default). Adjoint: alpha chosen by held-out replay
     (adjoint_cv) against the discrepancy principle (adjoint_dp). The true D is used only for scoring.
E3b  sensor placement at equal budget: random, regular grid, and designed (greedy D-optimal on the
     adjoint's sensitivities at a prior constant D = Fo / T; no data, no truth). Datasets are row subsets
     of the 3,969-node dataset, so every layout shares one noise draw. Arms: adjoint, pinn.
E3c  adaptive collocation: pinn (default sampler) vs pinn_rar.
E3d  augmentation of sparse sensors by thin-plate RBF interpolation: pinn vs pinn_aug, adjoint vs adjoint_aug.
"""
import copy
import json
import os

import numpy as np

import common as C

DATASETS20 = (C.REPO / "submission" / "experiments" / "zhang_replication_study20" / "datasets20.txt").read_text().split()
EPOCHS = {"quick": 300, "full": 1500}
HOLDOUT = 0.2

E3A_FULL = DATASETS20[:6] + ["B_gaussian_bump__Fo0.1__uniform8__n100__random__eta0.01",
                             "C_inclusion__Fo0.1__uniform8__n100__random__eta0.1"]
E3D_FULL = ["A_smooth__Fo0.1__uniform8__n20__random__eta0.05", "C_inclusion__Fo0.1__uniform8__n20__random__eta0.05",
            "E_grf__Fo0.1__uniform8__n20__random__eta0.01", "C_inclusion__Fo0.1__uniform8__n5__random__eta0.01",
            "D_layered__Fo0.1__uniform8__n20__clustered__eta0.05", "F_defects__Fo0.1__uniform8__n20__boundary__eta0.05"]
OED_TIERS = ("A_smooth", "C_inclusion", "E_grf")
OED_BUDGETS = (16, 36, 64)
OED_LAYOUTS = ("random", "grid", "designed")
DENS_TIERS = ("A_smooth", "B_gaussian_bump", "C_inclusion", "D_layered", "E_grf", "F_defects")
DENS_N = (20, 100, 400, 1600, 3969)
DENS_ETA = (0.01, 0.05, 0.1)
D2_CASES = [(t, 0.01) for t in ("A_smooth", "E_grf")]
D2_N = DENS_N                    # pinn_long across every sensor count (A_smooth, E_grf, 1 % noise)
D2_ARMS = ("pinn_long",)            # a weaker wall taper was dropped: with the exact u it changes relL2 by < 0.015 (D3)
LONG = 4                                            # pinn_long: 4 x the epochs (6,000 at full size)


def dens_key(tier, n, eta):
    return f"{tier}__Fo0.1__uniform8__n{n}__nodes_nested__eta{eta:g}"


def sweep_configs(size):
    names = list(C.pinn_main().LAMBDA_SWEEP_GRID)
    return names[:3] if size == "quick" else names


def registry(exp, size):
    q = size == "quick"
    if exp == "E2a":
        keys = DATASETS20[:6:3] if q else DATASETS20[:6]
        return [("bench", k, ["pinn", "pinn_T"]) for k in keys]
    if exp == "E2b":
        import arrhenius as AR
        return [("arrhenius", AR.key_of(*c), ["adjoint", "pinn", "pinn_T"]) for c in AR.configs(size)]
    if exp == "E3a":
        keys = E3A_FULL[:1] if q else E3A_FULL
        arms = [f"cfg80_{c}" for c in sweep_configs(size)] + ["pinn_selected", "pinn_default", "adjoint_dp", "adjoint_cv"]
        return [("bench", k, arms) for k in keys]
    if exp == "E3b":
        import sensor_design as SD
        return [("oed", SD.key_of(t, n, lay), ["adjoint", "pinn"]) for (t, n, lay) in SD.configs(size)]
    if exp == "E3c":
        keys = E3A_FULL[:1] if q else E3A_FULL
        return [("bench", k, ["pinn", "pinn_rar"]) for k in keys]
    if exp == "E3d":
        keys = E3D_FULL[:1] if q else E3D_FULL
        return [("bench", k, ["adjoint", "adjoint_aug", "pinn", "pinn_aug"]) for k in keys]
    if exp == "D1":                                 # sensor density: does the project PINN improve with more sensors?
        cells = [("A_smooth", n, 0.05) for n in (20, 3969)] if q else \
            [(t, n, e) for t in DENS_TIERS for e in DENS_ETA for n in DENS_N]
        return [("density", dens_key(*c), ["pinn"]) for c in cells]
    if exp == "D2":                                 # why not: training budget and wall taper at low / high density
        cells = [("A_smooth", 3969, 0.05)] if q else [(t, n, e) for t, e in D2_CASES for n in D2_N]
        rows = [("density", dens_key(*c), list(D2_ARMS)) for c in cells]
        if not q:                                   # one 16 x run: does more training keep closing the gap?
            rows += [("density", dens_key(t, n, 0.01), ["pinn_xlong"])
                     for t, n in (("A_smooth", 3969), ("A_smooth", 100), ("E_grf", 3969), ("E_grf", 100))]
            rows.append(("density", dens_key("C_inclusion", 3969, 0.01), ["pinn_long"]))   # u-net-limited case (D3)
        return rows
    if exp == "D3":                                 # is the bias in the u-net or the D-step? (split, oracle u)
        rows = [("density", dens_key(t, n, 0.01), ["pinn_split50"]) for t in ("A_smooth", "E_grf") for n in (100, 3969)]
        rows += [("density", dens_key(t, 3969, 0.01), ["oracle_u", "oracle_u_weaktaper", "oracle_u_xlong"])
                 for t in ("A_smooth", "C_inclusion", "E_grf")]
        return rows[:1] if q else rows
    raise SystemExit(f"unknown experiment {exp}")


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def holdout_rows(ds, frac=HOLDOUT, seed=0):
    """Hold out whole sensor locations (all their reading times)."""
    xy = np.round(np.c_[ds.x, ds.y], 9)
    locs, inv = np.unique(xy, axis=0, return_inverse=True)
    inv = inv.ravel()
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(frac * len(locs))))
    val_locs = rng.choice(len(locs), n_val, replace=False)
    val = np.isin(inv, val_locs)
    return np.where(~val)[0], np.where(val)[0]


def subset(ds, rows, extra=None):
    """A copy of a Dataset with only these rows (and optional extra pseudo rows x, y, t, u)."""
    d = copy.copy(ds)
    d.x, d.y, d.t, d.u = ds.x[rows], ds.y[rows], ds.t[rows], ds.u[rows]
    if extra is not None:
        d.x, d.y, d.t, d.u = (np.r_[d.x, extra[0]], np.r_[d.y, extra[1]], np.r_[d.t, extra[2]], np.r_[d.u, extra[3]])
    return d


def score(ds, D, extra, D_ref=None):
    D_true = ds.ground_truth()[0] if D_ref is None else D_ref
    m = C.field_scores(D, D_true)
    m["replay_over_sigma"] = C.replay_rms_over_sigma(ds, D)
    m.update(extra)
    return m


def pinn(collection, key, size, **opts):
    import pinn_plus as PP
    cfg_over = opts.pop("cfg_over", {})
    cfg = PP.make_cfg(collection, key, epochs=EPOCHS[size], **cfg_over)
    return PP.train(cfg, PP.PlusOptions(**opts), verbose=False)


def adjoint(ds):
    import hybrid as H
    return H.run_cold(ds)


# ---------------------------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------------------------
def run_job(exp, arm, collection, key, size):
    ds = C.dataset(collection, key)

    if exp in ("E2a", "E2b"):
        D_ref = ds.ground_truth()[0]                       # D0 = D(x, y, u = 0) for Arrhenius data
        e = C.entry(collection, key)
        S = float(e.get("strength_S", 0.0))
        if arm == "adjoint":
            r = adjoint(ds)
            return score(ds, r["D"], dict(total_wall_s=r["wall_s"], alpha=r["alpha"], flag=r["flag"], strength_S=S),
                         D_ref), dict(D_hat=r["D"].reshape(ds.N, ds.N))
        o = pinn(collection, key, size, coef="xyT" if arm == "pinn_T" else "xy")
        m = score(ds, o["D_hat"], dict(total_wall_s=o["wall_s"], strength_S=S, device=o["device"]), D_ref)
        arrays = dict(D_hat=o["D_hat"])
        if arm == "pinn_T":
            uu, g = o["g_curve"][:, 0], o["g_curve"][:, 1]
            arrays["g_curve"] = o["g_curve"]
            import arrhenius as AR
            s_true = AR.s_of(uu, S) if S else np.zeros_like(uu)
            m.update(g_at_1=float(g[-1]), g_rmse=float(np.sqrt(np.mean((g - s_true) ** 2))), g_max_abs=float(np.abs(g).max()))
            g_fn = lambda v, _u=uu, _g=g: np.interp(np.clip(v, 0, 1), _u, _g)
            m["replay_nonlinear_over_sigma"] = AR.replay_nonlinear(o["D_hat"], g_fn, ds)
        return m, arrays

    if exp == "E3a":
        tr, va = holdout_rows(ds)
        if arm.startswith("cfg80_"):
            name = arm[len("cfg80_"):]
            o = pinn(collection, key, size, train_rows=tr, cfg_over=dict(C.pinn_main().LAMBDA_SWEEP_GRID[name]))
            m = score(ds, o["D_hat"], dict(config=name, total_wall_s=o["wall_s"], u_net_val_chi2=o["u_net_val_chi2"],
                                           val_replay_over_sigma=C.replay_rms_over_sigma(ds, o["D_hat"], rows=va),
                                           n_train_rows=len(tr), n_val_rows=len(va)))
            return m, dict(D_hat=o["D_hat"])
        if arm == "pinn_selected":
            cands = []
            for name in sweep_configs(size):
                mm, _ = C.load_run("E3a", f"cfg80_{name}", key)
                cands.append((mm["val_replay_over_sigma"], name, mm["rel_L2_D"]))
            val, name, _ = min(cands)
            o = pinn(collection, key, size, cfg_over=dict(C.pinn_main().LAMBDA_SWEEP_GRID[name]))
            oracle = min(cands, key=lambda c: c[2])
            m = score(ds, o["D_hat"], dict(selected=name, selected_val=val, oracle_config_80=oracle[1],
                                           oracle_rel_L2_D_80=oracle[2], total_wall_s=o["wall_s"],
                                           candidates=[dict(config=c[1], val=c[0], rel_L2_D_80=c[2]) for c in cands]))
            return m, dict(D_hat=o["D_hat"])
        if arm == "pinn_default":
            o = pinn(collection, key, size)
            return score(ds, o["D_hat"], dict(total_wall_s=o["wall_s"])), dict(D_hat=o["D_hat"])
        if arm == "adjoint_dp":
            r = adjoint(ds)
            return score(ds, r["D"], dict(alpha=r["alpha"], flag=r["flag"], total_wall_s=r["wall_s"],
                                          val_replay_over_sigma=C.replay_rms_over_sigma(ds, r["D"], rows=va))), \
                dict(D_hat=r["D"].reshape(ds.N, ds.N))
        if arm == "adjoint_cv":
            return adjoint_cv(ds, tr, va)

    if exp == "E3b":
        if arm == "adjoint":
            r = adjoint(ds)
            return score(ds, r["D"], dict(alpha=r["alpha"], flag=r["flag"], total_wall_s=r["wall_s"],
                                          **layout_info(collection, key))), dict(D_hat=r["D"].reshape(ds.N, ds.N))
        o = pinn(collection, key, size)
        return score(ds, o["D_hat"], dict(total_wall_s=o["wall_s"], **layout_info(collection, key))), dict(D_hat=o["D_hat"])

    if exp == "E3c":
        o = pinn(collection, key, size, rar=(arm == "pinn_rar"))
        return score(ds, o["D_hat"], dict(total_wall_s=o["wall_s"], n_rar=o["n_rar"])), dict(D_hat=o["D_hat"])

    if exp == "E3d":
        if arm in ("adjoint", "adjoint_aug"):
            dd = ds
            n_extra = 0
            if arm == "adjoint_aug":
                import pinn_plus as PP
                Xa, ua, _ = PP.augment_readings(np.c_[ds.x, ds.y, ds.t], ds.u[:, None], np.full((len(ds.u), 1), ds.sigma),
                                                3.0, 2.0, 1.0, 1.0, 42)
                dd = subset(ds, np.arange(len(ds.u)), extra=(Xa[:, 0], Xa[:, 1], Xa[:, 2], ua[:, 0]))
                n_extra = len(ua)
            r = adjoint(dd)
            m = score(ds, r["D"], dict(alpha=r["alpha"], flag=r["flag"], total_wall_s=r["wall_s"], n_pseudo=n_extra))
            return m, dict(D_hat=r["D"].reshape(ds.N, ds.N))
        o = pinn(collection, key, size, augment=3.0 if arm == "pinn_aug" else 0.0)
        return score(ds, o["D_hat"], dict(total_wall_s=o["wall_s"], n_data_rows=o["n_data_rows"])), dict(D_hat=o["D_hat"])

    if exp in ("D1", "D2", "D3"):
        e = C.entry(collection, key)
        info = dict(tier=e["tier"], n_sensors=e["n_sensors"], eta=e["eta"])
        if arm == "adjoint":
            r = adjoint(ds)
            return score(ds, r["D"], dict(alpha=r["alpha"], flag=r["flag"], total_wall_s=r["wall_s"], **info)), \
                dict(D_hat=r["D"].reshape(ds.N, ds.N))
        if arm.startswith("oracle_u"):
            import pinn_plus as PP
            import time as _t
            t0 = _t.time()
            D = PP.oracle_u_D(PP.make_cfg(collection, key), taper_k=40.0 if "weaktaper" in arm else 6.0,
                              steps=EPOCHS[size] * (16 if "xlong" in arm else 1))
            return score(ds, D, dict(total_wall_s=_t.time() - t0, note="exact u from the simulation, no sensors", **info)), \
                dict(D_hat=D)
        if arm == "pinn_split50":
            import pinn_plus as PP
            o = PP.train(PP.make_cfg(collection, key, epochs=EPOCHS[size]), PP.PlusOptions(pde_split=0.5), verbose=False)
            return score(ds, o["D_hat"], dict(total_wall_s=o["wall_s"], **info)), dict(D_hat=o["D_hat"])
        epochs = EPOCHS[size] * (16 if "xlong" in arm else LONG if "long" in arm else 1)
        import pinn_plus as PP
        cfg = PP.make_cfg(collection, key, epochs=epochs)
        o = PP.train(cfg, PP.PlusOptions(taper_k=40.0 if "weaktaper" in arm else 6.0), verbose=False)
        return score(ds, o["D_hat"], dict(total_wall_s=o["wall_s"], epochs=epochs, **info)), dict(D_hat=o["D_hat"])

    raise ValueError((exp, arm))


def adjoint_cv(ds, tr, va):
    """alpha by held-out replay: walk the full standard alpha path on the training rows (no early stop),
    score every step on the held-out rows, refit on all rows down to the chosen alpha."""
    import hybrid as H
    import time
    t0 = time.perf_counter()
    s_tr = C.A.AdjointSolver(subset(ds, tr))
    s_va = C.A.AdjointSolver(subset(ds, va))
    alphas = len(s_tr.d) * np.logspace(0, -6, 13)
    theta = np.full(s_tr.K ** 2, s_tr.best_constant())
    curve = []
    for a in alphas:
        r = H.invert_from(s_tr, theta, [a])
        theta = H.project_to_controls(s_tr, r["D"])
        W = s_va.forward(np.exp(s_va.m_of(theta)))[0]
        curve.append(float(np.sqrt(np.mean((s_va.predict(W) - s_va.d) ** 2)) / s_va.sigma_w))
    k = int(np.argmin(curve))
    scale = len(ds.u) / len(tr)                          # the misfit sums over rows: keep alpha per reading
    s_all = C.A.AdjointSolver(ds)
    theta = np.full(s_all.K ** 2, s_all.best_constant())
    path = alphas[:k + 1] * scale
    for a in path:
        r = H.invert_from(s_all, theta, [a])
        theta = H.project_to_controls(s_all, r["D"])
    D = np.exp(s_all.m_of(theta))
    m = score(ds, D, dict(alpha=float(path[-1]), cv_curve=curve, cv_index=k, total_wall_s=time.perf_counter() - t0,
                          val_replay_over_sigma=curve[k]))
    return m, dict(D_hat=D.reshape(ds.N, ds.N))


def layout_info(collection, key):
    e = C.entry(collection, key)
    return dict(tier=e["base_tier"], budget=e["n_sensors"], layout=e["layout"])
