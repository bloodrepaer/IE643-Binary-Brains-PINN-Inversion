"""
arrhenius.py -- data where diffusivity depends on temperature, for experiment E2b.

Truth:   u_t = div( D(x, y, u) grad u ),   D(x, y, u) = D0(x, y) * exp(s(u))
         s(u) = gamma * theta * u / (1 + theta * u)          (Arrhenius in absolute temperature)
with T = T_ref (1 + theta u): u = 0 at the walls is the reference temperature, gamma = E_a / (R T_ref).
gamma = 20 (E_a about 50 kJ/mol at 300 K). The strength S = s(1) = gamma theta / (1 + theta) is the
log-ratio of D between the hottest (u = 1, the IC peak) and the reference state:
  S = 0     no temperature dependence (the standard benchmark, as a control)
  S = 0.5   D rises by e^0.5 = 1.6x at the IC peak   (theta 0.026: about 8 K above 300 K)
  S = 1     2.7x  (theta 0.053: 16 K)
  S = 2     7.4x  (theta 0.111: 33 K)

Solver: the project's discretisation unchanged (5-point finite-volume operator, harmonic-mean links,
129 x 129 grid, zero walls, explicit RK4, 400 output intervals, step = 0.9 x RK4 limit). The only
difference: D is recomputed from u at every RK4 stage, and the step is chosen for the largest D the run
can reach, D0 * e^S (u <= 1 by the maximum principle). At S = 0 this is exactly integrators' RK4
(self_check). Sampling, noise convention, seeds, file format: generate_dataset_v2's own functions, with
the BASE tier name for seeds, so every strength has the same sensors and the same noise draw.

The problem files describe the standard D(x, y) model (what the adjoint and plain PINN assume); the true
s(u) is only in ground_truth/ (for scoring).

  python run.py novelty/arrhenius.py --size full       # writes datasets_novelty_arrhenius/
"""
import argparse
import csv
import json
import os
import time

import numpy as np

import generate_dataset_v2 as G
from domain import Domain2D
from integrators import ForwardModel, RK4_STABILITY

OUT = "datasets_novelty_arrhenius"
GAMMA = 20.0
STRENGTHS = (0.0, 0.5, 1.0, 2.0)
FIELDS = ("A_smooth", "C_inclusion")
SETTINGS = (("uniform8", 50, "random", 0.05), ("uniform8", 100, "random", 0.01))


def theta_of(S):
    return S / (GAMMA - S)


def s_of(u, S):
    th = theta_of(S)
    return GAMMA * th * u / (1.0 + th * u)


def rhs(u, D0, S, h):
    """div(D(u) grad u) at interior nodes, zero boundary values; equals build_operator(D(u)) @ u."""
    D = D0 * np.exp(s_of(u, S)) if S else D0
    Dc = D[1:-1, 1:-1]
    uc = u[1:-1, 1:-1]
    out = np.zeros_like(u)
    acc = np.zeros_like(uc)
    for sl in ((slice(2, None), slice(1, -1)), (slice(0, -2), slice(1, -1)),
               (slice(1, -1), slice(2, None)), (slice(1, -1), slice(0, -2))):
        Dn = D[sl]
        acc += (2.0 * Dc * Dn / (Dc + Dn)) * (u[sl] - uc)
    out[1:-1, 1:-1] = acc / h ** 2
    return out


def solve(D0, S, u0, T, n_out, h, cfl_safety=0.9):
    """RK4 with the integrators' step rule, evaluated at the largest reachable D."""
    D_worst = D0 * np.exp(S)
    Dc = D_worst[1:-1, 1:-1]
    rho = 0.0
    hm = lambda a, b: 2 * a * b / (a + b)
    row = (hm(Dc, D_worst[2:, 1:-1]) + hm(Dc, D_worst[:-2, 1:-1]) + hm(Dc, D_worst[1:-1, 2:])
           + hm(Dc, D_worst[1:-1, :-2])) / h ** 2
    rho = float(np.max(2 * row))                      # Gershgorin, as spectral_radius_bound
    Delta = T / n_out
    m = max(1, int(np.ceil(Delta / (cfl_safety * RK4_STABILITY / rho))))
    dt = Delta / m
    U = np.empty((n_out + 1,) + u0.shape)
    u = u0.copy()
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    U[0] = u
    tic = time.perf_counter()
    for k in range(1, n_out + 1):
        for _ in range(m):
            k1 = rhs(u, D0, S, h)
            k2 = rhs(u + 0.5 * dt * k1, D0, S, h)
            k3 = rhs(u + 0.5 * dt * k2, D0, S, h)
            k4 = rhs(u + dt * k3, D0, S, h)
            u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        U[k] = u
    t = np.linspace(0.0, T, n_out + 1)
    return t, U, dict(dt=dt, n_steps=m * n_out, wall_s=time.perf_counter() - tic)


def self_check(n=33):
    """S = 0 on a small grid: matrix-free RK4 vs integrators.ForwardModel(...).solve(..., 'rk4')."""
    dom = Domain2D(Nx=n, Ny=n)
    cls, kw = G.TIERS["C_inclusion"]
    D0 = G.get_D_field(dom, cls, **kw)
    T = 0.02 / float(D0.mean())
    bc = G.make_bc("zero", T)
    u0 = G.initial_condition(dom, bc)
    t1, U1, i1 = ForwardModel(dom, D0, stencil="5pt", bc=bc).solve(u0, T, 20, "rk4")
    t2, U2, i2 = solve(D0, 0.0, u0, T, 20, dom.dx)
    assert i1["n_steps"] == i2["n_steps"], (i1["n_steps"], i2["n_steps"])
    return float(np.abs(U1 - U2).max() / np.abs(U1).max())


def replay_nonlinear(D0_inv, g_fn, ds, n_steps_cfl=0.9):
    """Replay an estimate (D0 on the 65 x 65 grid and a callable g(u)) through the nonlinear model on
    that grid; returns RMS / sigma at the sensor readings (bilinear in space, linear in time)."""
    N = D0_inv.shape[0]
    h = 1.0 / (N - 1)
    u0 = ds.u0.copy()
    S_eff = float(max(0.0, np.max(g_fn(np.linspace(0, 1, 101)))))
    D_worst = D0_inv * np.exp(S_eff)
    rho = float(np.max(8 * D_worst / h ** 2))
    n_out = 400
    Delta = ds.T / n_out
    m = max(1, int(np.ceil(Delta / (n_steps_cfl * RK4_STABILITY / rho))))
    dt = Delta / m
    u = u0.copy()
    U = [u.copy()]
    for _ in range(n_out):
        for _ in range(m):
            def f(v):
                D = D0_inv * np.exp(g_fn(v))
                Dc = D[1:-1, 1:-1]; vc = v[1:-1, 1:-1]; acc = np.zeros_like(vc)
                for sl in ((slice(2, None), slice(1, -1)), (slice(0, -2), slice(1, -1)),
                           (slice(1, -1), slice(2, None)), (slice(1, -1), slice(0, -2))):
                    Dn = D[sl]; acc += (2 * Dc * Dn / (Dc + Dn)) * (v[sl] - vc)
                o = np.zeros_like(v); o[1:-1, 1:-1] = acc / h ** 2
                return o
            k1 = f(u); k2 = f(u + 0.5 * dt * k1); k3 = f(u + 0.5 * dt * k2); k4 = f(u + dt * k3)
            u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        U.append(u.copy())
    U = np.array(U)
    from scipy.interpolate import RegularGridInterpolator
    tg = np.linspace(0, ds.T, n_out + 1)
    xg = np.linspace(0, 1, N)
    pred = RegularGridInterpolator((tg, xg, xg), U)(np.c_[ds.t, ds.x, ds.y])
    sig = ds.sigma if ds.sigma > 0 else 1e-3 * np.sqrt(np.mean(ds.u ** 2))
    return float(np.sqrt(np.mean((pred - ds.u) ** 2)) / sig)


def configs(size):
    if size == "quick":
        return [("A_smooth", S, *SETTINGS[0]) for S in (0.0, 2.0)]
    return [(f, S, *st) for f in FIELDS for S in STRENGTHS for st in SETTINGS]


def key_of(field, S, protocol, n, layout, eta):
    return G.key_of(f"{field}-arrS{S:g}", 0.1, protocol, n, layout, eta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=("quick", "full"), default="full")
    a = ap.parse_args()
    for sub in ("problem", "ground_truth", "sensors"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    dom, dom_inv, r = Domain2D(Nx=129, Ny=129), Domain2D(Nx=65, Ny=65), 2
    man_path = os.path.join(OUT, "manifest.json")
    manifest = {e["key"]: e for e in json.load(open(man_path))} if os.path.exists(man_path) else {}
    by_field_S = {}
    for (field, S, p, n, layout, eta) in configs(a.size):
        by_field_S.setdefault((field, S), []).append((p, n, layout, eta))
    for (field, S), settings in by_field_S.items():
        if all(key_of(field, S, *st) in manifest for st in settings):
            print(f"skip {field} S={S:g}", flush=True)
            continue
        cls, kw = G.TIERS[field]
        D0 = G.get_D_field(dom, cls, **kw)
        D_mean = float(D0.mean())
        Fo = 0.1
        T = Fo / D_mean
        bc = G.make_bc("zero", T)
        u0 = G.initial_condition(dom, bc)
        t, U, info = solve(D0, S, u0, T, G.N_OUT, dom.dx)
        print(f"[{field} S={S:g}] T={T:.3f}  {info['n_steps']} RK4 steps, {info['wall_s']:.0f}s, "
              f"u range [{U.min():.3f}, {U.max():.3f}]", flush=True)
        tier_name = f"{field}-arrS{S:g}"
        prob = f"{tier_name}__Fo{Fo:g}"
        G.write_problem(OUT, prob, dom, dom_inv, r, T, Fo, bc, "zero", t)
        sig_ref = G.sigma_reference(dom, U, field)               # base tier name: same reference sensors
        np.savez_compressed(os.path.join(OUT, "ground_truth", prob + ".npz"),
                            D_data=D0, D_inv=D0[::r, ::r], x_data=dom.x, y_data=dom.y, x_inv=dom_inv.x,
                            y_inv=dom_inv.y, t=t[::G.GT_EVERY], u=U[::G.GT_EVERY].astype(np.float32), T=T, Fo=Fo,
                            D_mean=D_mean, integrator="rk4-nonlinear", stencil="5pt", dt=info["dt"],
                            n_steps=info["n_steps"], bc="zero", defects="[]",
                            arrhenius_gamma=GAMMA, arrhenius_theta=theta_of(S), strength_S=S)
        for (p, n, layout, eta) in settings:
            data, sigma, m = G.sample(dom, dom_inv, t, U, field, Fo, p, n, layout, eta, sig_ref)
            key = key_of(field, S, p, n, layout, eta)
            np.savez_compressed(os.path.join(OUT, "sensors", key + ".npz"), **data)
            with open(os.path.join(OUT, "sensors", key + ".csv"), "w", newline="") as f:
                w = csv.writer(f, lineterminator="\n"); w.writerow(["x", "y", "t", "u", "sigma"])
                for row in zip(data["x"], data["y"], data["t"], data["u"], data["sigma"]):
                    w.writerow([f"{v:.8g}" for v in row])
            manifest[key] = dict(key=key, tier=tier_name, base_tier=field, strength_S=S, gamma=GAMMA,
                                 theta=theta_of(S), Fo=Fo, T=round(T, 6), D_mean=round(D_mean, 6), protocol=p,
                                 n_sensors=m, layout=layout, n_readings=len(data["u"]), eta=eta, sigma=sigma,
                                 sigma_ref=sig_ref, bc="zero", problem=prob, ground_truth=prob + ".npz",
                                 integrator="rk4-nonlinear", n_data=129, n_inv=65,
                                 files={"sensors_csv": f"sensors/{key}.csv", "sensors_npz": f"sensors/{key}.npz",
                                        "problem": f"problem/{prob}.json", "ground_truth": f"ground_truth/{prob}.npz"})
        json.dump(list(manifest.values()), open(man_path, "w"), indent=1, default=float)
    print(f"{len(manifest)} datasets in {OUT}/")


if __name__ == "__main__":
    main()
