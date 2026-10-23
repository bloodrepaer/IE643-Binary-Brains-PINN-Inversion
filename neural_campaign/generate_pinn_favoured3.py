"""
generate_pinn_favoured3.py -- H10: the setting Thakur et al. (2024) built their PINN for -> datasets_v2_pinnwin3/

Imaging-like data: every interior node of the 65^2 inversion grid inside the 0.05 margin (3249 points) or a
400-node subset, at the 40 uniform times; a smooth initial condition (the same two bumps, width 0.12 instead of
0.05/0.06); one heated edge whose values are NOT given (H10u) or given (H10k control, identical sensor files).
Noise eta 0 and 1 %. Thakur's PINN uses neither the IC nor the BC, so it is unaffected by the missing BC;
the adjoint must assume one (zero walls).  Same solver/format rules as generate_pinn_favoured2.py.
"""
import generate_dataset_v2 as G
G.IC_BUMPS = [dict(b, s=0.12) for b in G.IC_BUMPS]
import generate_pinn_favoured as GP
import generate_pinn_favoured2 as GP2

GP.OUT = "datasets_v2_pinnwin3"


def plan():
    J = []
    h = "Dense imaging-like data, boundary condition unknown (Thakur et al. setting)"
    for tier in ("A_smooth", "E_grf_corrlen_long"):
        for name, amp in (("heated_left", 1.0), ("pulsed_top", 0.3)):
            for n in (3249, 400):          # 3249 = all 57x57 inversion nodes inside the margin
                for eta in (0.0, 0.01):
                    tag = f"{tier}-{name.replace('_', '')}-a{amp:g}-n{n}-eta{eta:g}"
                    for known in (False, True):
                        J.append(dict(family="H10", hypothesis=h, case=f"H10{'k' if known else 'u'}-{tier}-{name.replace('_', '')}-a{amp:g}",
                                      base_tier=tier, Fo=0.1, protocol="uniform40", n=n, layout="nodes", eta=eta, bc=(name, amp),
                                      bc_known=known, q=0.0, src_known=True, role="control" if known else "test", pair=tag))
    return J


GP.plan, GP.sample = plan, GP2.sample

if __name__ == "__main__":
    GP.main()
