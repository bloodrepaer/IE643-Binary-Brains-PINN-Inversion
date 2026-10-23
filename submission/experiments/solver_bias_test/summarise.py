import json, glob, os
rows = {}
for p in glob.glob('/home/claude/solverbias/res_adj_*/*/metrics.json'):
    d = p.split('/')[-3].replace('res_adj_', ''); k = p.split('/')[-2]
    rows.setdefault(k, {})[('adjoint', d)] = json.load(open(p))['rel_L2_D']
for p in glob.glob('/home/claude/solverbias/res_pinn_*/*/metrics.json'):
    d = p.split('/')[-3].replace('res_pinn_', ''); k = p.split('/')[-2]
    rows.setdefault(k, {})[('PINN', d)] = json.load(open(p)).get('rel_l2_D')
cols = ['data_rk4_129', 'data_cn_129', 'data_fe_129', 'data_cn_65_crime']
for k, r in sorted(rows.items()):
    print(k)
    for m in ('adjoint', 'PINN'):
        print(f"   {m:8s}", "  ".join(f"{c.replace('data_',''):>13s}={r.get((m, c), float('nan')):.3f}" for c in cols))
    if ('adjoint', 'rk4_coarseCN') in r: print(f"   adjoint with coarse CN (50 steps) on rk4 data: {r[('adjoint','rk4_coarseCN')]:.3f}")
