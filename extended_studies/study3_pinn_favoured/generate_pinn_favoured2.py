"""
generate_pinn_favoured2.py -- second round of PINN-favoured benchmarks -> datasets_v2_pinnwin2/

Same solver, IC, noise convention, file format and pairing rules as generate_pinn_favoured.py
(RK4 on 129^2, inversion on 65^2, sigma = eta * RMS of the clean field at the 100-sensor random layout
and uniform40 times). New: many more readings, using a "nodes" layout = random interior nodes of the
65^2 inversion grid inside the 0.05 margin (the 100-point random pool is too small for 400 sensors).

  H6  more readings, physics fully known       A_smooth / E_grf_corrlen_long / G_lowfreq
                                               100 (random pool) and 400 (nodes) sensors x 40 times, eta 5 %
  H7  unknown BC + more readings               A_smooth / E_grf, pulsed_top a0.3 and heated_left a1,
                                               100 and 400 sensors x 40 times, eta 5 %; H7u (BC not given)
                                               and H7k (BC given) share identical sensor files
  H9  D finer than the adjoint's 16x16 control E_grf_corrlen_short (corr. length 0.03) and
      grid, dense data                         D_checkerboard_fine (6x6 blocks): 400 nodes x 40 times, eta 1 %

    python generate_pinn_favoured2.py
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import generate_dataset_v2 as G
import generate_pinn_favoured as GP
from domain import Domain2D

GP.OUT = "datasets_v2_pinnwin2"


def node_layout(n, seed):
    dom_inv = Domain2D(Nx=GP.N_INV, Ny=GP.N_INV)
    xy = np.c_[dom_inv.X.ravel(), dom_inv.Y.ravel()]
    m = G.SENSOR_MARGIN
    xy = xy[(xy[:, 0] >= m) & (xy[:, 0] <= 1 - m) & (xy[:, 1] >= m) & (xy[:, 1] <= 1 - m)]
    return xy[np.sort(np.random.default_rng(seed).permutation(len(xy))[:n])]


def plan():
    J = []

    def add(family, hyp, case, tier, protocol="uniform40", n=100, layout="random", eta=0.05,
            bc=("zero", 0.0), bc_known=True, role="test", pair=None):
        J.append(dict(family=family, hypothesis=hyp, case=case, base_tier=tier, Fo=0.1, protocol=protocol, n=n,
                      layout=layout, eta=eta, bc=bc, bc_known=bc_known, q=0.0, src_known=True, role=role,
                      pair=pair or case))

    h = "More sensor readings, physics fully known"
    for tier in ("A_smooth", "E_grf_corrlen_long", "G_lowfreq"):
        add("H6", h, f"H6-{tier}", tier, n=100, layout="random", pair=f"H6-{tier}")
        add("H6", h, f"H6-{tier}", tier, n=400, layout="nodes", pair=f"H6-{tier}")

    h = "Boundary condition unknown, more sensor readings"
    for tier in ("A_smooth", "E_grf"):
        for name, amp in (("pulsed_top", 0.3), ("heated_left", 1.0)):
            for n, lay in ((100, "random"), (400, "nodes")):
                tag = f"{tier}-{name.replace('_', '')}-a{amp:g}-n{n}"
                add("H7", h, f"H7u-{tier}-{name.replace('_', '')}-a{amp:g}", tier, n=n, layout=lay,
                    bc=(name, amp), bc_known=False, pair=tag)
                add("H7", h, f"H7k-{tier}-{name.replace('_', '')}-a{amp:g}", tier, n=n, layout=lay,
                    bc=(name, amp), bc_known=True, role="control", pair=tag)

    h = "D has features finer than the adjoint's control grid, dense data"
    for tier in ("E_grf_corrlen_short", "D_checkerboard_fine"):
        add("H9", h, f"H9-{tier}", tier, n=400, layout="nodes", eta=0.01)
    return J


_orig_sample = GP.sample


def sample(dom, t, U, j, sig_ref):
    if j["layout"] != "nodes":
        return _orig_sample(dom, t, U, j, sig_ref)
    ks = G.PROTOCOLS[j["protocol"]]()
    xy = node_layout(j["n"], G.seed_of("nodes", j["base_tier"], j["n"]))
    u_clean = np.concatenate([RegularGridInterpolator((dom.x, dom.y), U[k])(xy) for k in ks])
    m = len(xy)
    sigma = j["eta"] * sig_ref
    rng = np.random.default_rng(G.seed_of("noise-pinnwin2", j["pair"], j["protocol"], j["n"], j["layout"], j["eta"]))
    u = u_clean + sigma * rng.standard_normal(u_clean.shape)
    data = dict(x=np.tile(xy[:, 0], len(ks)), y=np.tile(xy[:, 1], len(ks)), t=np.repeat(t[ks], m), u=u,
                u_clean=u_clean, sigma=np.full_like(u_clean, sigma), sensor_id=np.tile(np.arange(m), len(ks)),
                time_index=np.repeat(ks, m))
    return data, sigma, m, ks, 0.0


GP.plan, GP.sample = plan, sample

if __name__ == "__main__":
    GP.main()
