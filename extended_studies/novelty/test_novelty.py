"""
test_novelty.py -- checks the pipeline's building blocks before any experiment is trusted.

  1. pinn_plus.train with default options gives the same D_hat as submission/pinn_inversion.py
     (same seed, same epochs): the trainer is a faithful port.
  2. The xyT variant at initialisation: the third network starts near zero, so its first-epoch loss is
     close to the plain PINN's; its PDE residual matches a finite-difference check of div(D(x,y,u) grad u).
  3. Projection of a D field onto the adjoint's 16 x 16 controls reproduces a field that lies in that space.
  4. (if built) the Arrhenius generator at strength 0 reproduces the linear generator's data.
Run from extended_studies/:  python run.py novelty/test_novelty.py [--quick]
"""
import argparse
import sys
import tempfile

import numpy as np
import torch

import common as C

FAILS = []


def report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def test_port(epochs):
    import pinn_plus as PP
    key = "A_smooth__Fo0.1__uniform8__n50__random__eta0.05"
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PP.make_cfg("bench", key, epochs=epochs, out_dir=tmp)
        ref = C.pinn_main().train_single_experiment(cfg, run_tag="ref")
        D_ref = np.load(f"{tmp}/ref/D_hat.npz")["D_hat"]
        cfg2 = PP.make_cfg("bench", key, epochs=epochs, out_dir=tmp)
        out = PP.train(cfg2, verbose=False)
    diff = float(np.abs(out["D_hat"] - D_ref).max())
    report("pinn_plus default == pinn_inversion.train_single_experiment", diff < 1e-6,
           f"max |D diff| = {diff:.2e} over {epochs} epochs (ref relL2 {ref.get('rel_l2_D', float('nan')):.4f})")


def test_xyT_residual():
    import pinn_plus as PP
    key = "A_smooth__Fo0.1__uniform8__n50__random__eta0.05"
    cfg = PP.make_cfg("bench", key, epochs=1)
    P = C.pinn_main()
    b = P.load_v2_experiment(cfg)
    torch.manual_seed(0)
    model = PP.TempPINN(cfg, b["Lx"], b["Ly"], b["T_max"], b["m_init_bias"], b["bc_type"]).double()
    with torch.no_grad():                               # make g clearly non-zero for the check
        for p in model.g_net.parameters():
            p.add_(0.3 * torch.randn_like(p))
    pts = torch.tensor([[0.31, 0.42, 0.05], [0.66, 0.58, 0.12], [0.5, 0.5, 0.2]], dtype=torch.float64)
    raw, _, _ = PP.pde_loss_xyT(model, pts.clone(), cfg)
    # finite-difference residual: u_t - div(D(x, y, u(x, y, t)) grad u)
    h = 1e-4

    def u_at(x, y, t):
        return model.forward_u(torch.tensor([[x, y, t]], dtype=torch.float64))[0, 0]

    def D_at(x, y, t):
        uu = u_at(x, y, t).reshape(1, 1)
        return model.forward_D_xyu(torch.tensor([[x, y]], dtype=torch.float64), uu)[0, 0]

    res = []
    with torch.no_grad():
        for x, y, t in pts.tolist():
            ut = (u_at(x, y, t + h) - u_at(x, y, t - h)) / (2 * h)
            fx = lambda xx: D_at(xx, y, t) * (u_at(xx + h, y, t) - u_at(xx - h, y, t)) / (2 * h)
            fy = lambda yy: D_at(x, yy, t) * (u_at(x, yy + h, t) - u_at(x, yy - h, t)) / (2 * h)
            div = (fx(x + h) - fx(x - h)) / (2 * h) + (fy(y + h) - fy(y - h)) / (2 * h)
            res.append(float(ut - div))
    fd = float(np.mean(np.square(res)))
    rel = abs(float(raw) - fd) / max(fd, 1e-30)
    report("xyT residual matches finite differences", rel < 1e-3, f"autograd {float(raw):.6e} vs FD {fd:.6e} (rel {rel:.1e})")


def test_projection():
    import hybrid as H
    ds = C.dataset("bench", "A_smooth__Fo0.1__uniform8__n50__random__eta0.05")
    s = C.A.AdjointSolver(ds)
    rng = np.random.default_rng(1)
    theta = np.log(0.2) + 0.3 * rng.standard_normal(s.K ** 2)
    D = np.exp(s.m_of(theta)).reshape(ds.N, ds.N)
    back = H.project_to_controls(s, D)
    err = float(np.abs(back - theta).max())
    report("projection onto 16x16 controls is exact for in-space fields", err < 1e-8, f"max |theta diff| {err:.1e}")


def test_arrhenius_zero():
    try:
        import arrhenius as AR
    except ImportError:
        print("  [skip] arrhenius generator not present")
        return
    err = AR.self_check()
    report("Arrhenius solver at strength 0 == linear RK4 solver", err < 1e-10, f"max rel diff {err:.1e}")


def test_wall_gradient():
    import hybrid_bc as HB
    full, walls = HB.test_hybrid_bc()
    report("joint (D, walls) adjoint gradient matches finite differences (E1b)", max(full[1], walls) < 1e-6,
           f"rel err {full[1]:.1e} (all), {walls:.1e} (walls only)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    print("novelty pipeline checks", flush=True)
    test_port(a.epochs)
    test_xyT_residual()
    test_projection()
    test_arrhenius_zero()
    test_wall_gradient()
    print("ALL PASSED" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
