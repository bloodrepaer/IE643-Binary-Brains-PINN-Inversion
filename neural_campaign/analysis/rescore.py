import sys, os, json, glob, numpy as np, pandas as pd
V2 = '/home/claude/v2'; sys.path.insert(0, V2); os.chdir(V2)
import adjoint_inversion as ai
man = {e['key']: e for e in json.load(open('datasets_v2_rk4_zero/manifest.json'))}
def base_key(name):
    for tag in ('__C0', '__C1'):
        if tag in name: return name.split(tag)[0], name[name.index(tag)+2:]
    return name, None
def score(D, key):
    ds = ai.Dataset('datasets_v2_rk4_zero', man[key]); sol = ai.AdjointSolver(ds)
    W, *_ = sol.forward(D.ravel())
    out = dict(W=W, D=D.ravel(), alpha=np.nan, flag='external', D_const=np.nan, rms_const=np.nan,
               n_evals=1, wall_s=np.nan, n_steps=sol.disc.n_steps)
    m, S, Dt = ai.evaluate(ds, sol, out)
    # sensitivity with the TRUE D (common mask for both methods)
    Wt, *_ = sol.forward(Dt.ravel()); St = ai.sensitivity_map(sol, Wt)
    inner = np.zeros_like(Dt, bool); inner[1:-1, 1:-1] = True
    hi = inner & (St >= np.median(St[inner])); lo = inner & ~hi
    rel = lambda a, b, mk: float(np.linalg.norm((a-b)[mk]) / np.linalg.norm(b[mk]))
    m['relL2_hiS_trueS'] = rel(D, Dt, hi); m['relL2_loS_trueS'] = rel(D, Dt, lo)
    c = float(np.sum(D*Dt)/np.sum(D*D)); m['scale_c'] = c
    m['relL2_scale_corrected'] = float(np.linalg.norm(c*D-Dt)/np.linalg.norm(Dt))
    m['mean_ratio'] = float(D.mean()/Dt.mean())
    pred = sol.predict(W); m['replay_bias_over_sigma'] = float(np.mean(pred-ds.u)/sol.sigma_w)
    return m, St, Dt, pred, ds
if __name__ == '__main__':
    rows = []; store = {}
    for d in sorted(glob.glob('results_pinn/*__eta*')):
        name = os.path.basename(d); key, preset = base_key(name)
        D = np.load(d + '/D_hat.npz')['D_hat'].astype(float)
        m, St, Dt, pred, ds = score(D, key)
        pm = json.load(open(d + '/metrics.json'))
        m.update(method='PINN', run=name, preset=preset or 'benchmark', reg=pm['reg_type'],
                 pinn_chi2_solver=pm['chi2_red_solver'], pinn_chi2_unet=pm['chi2_red_u_net'], wall=pm['elapsed_sec'])
        m.pop('defects', None); rows.append(m); store[name] = dict(D=D, St=St, Dt=Dt)
        print(name[:70], preset, f"relL2={m['rel_L2_D']:.3f} rms/s(CN)={m['rms_replay_over_sigma']:.2f} sqrt(chi2_own)={np.sqrt(pm['chi2_red_solver']):.2f} unet={np.sqrt(pm['chi2_red_u_net']):.2f} mean_ratio={m['mean_ratio']:.2f} scale-corr={m['relL2_scale_corrected']:.3f}", flush=True)
    # adjoint counterparts (re-scored the same way for the common-mask/scale metrics)
    for folder, reg in [('results_adjoint','tikhonov'),('results_adjoint_tv','tv'),('results_defects_tikhonov','tikhonov'),('results_defects_tv','tv')]:
        for d in sorted(glob.glob(folder + '/*__eta*')):
            key = os.path.basename(d); D = np.load(d + '/D_hat.npz')['D_hat'].astype(float)
            m, St, Dt, pred, ds = score(D, key); am = json.load(open(d + '/metrics.json'))
            m.update(method='Adjoint', run=key, preset=folder, reg=reg, wall=am['wall_s'], flag=am['flag'])
            m.pop('defects', None); rows.append(m); store[folder+'/'+key] = dict(D=D)
            print(folder, key[:60], f"relL2={m['rel_L2_D']:.3f} rms/s={m['rms_replay_over_sigma']:.2f} mean_ratio={m['mean_ratio']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv('/home/claude/work/rescored.csv', index=False)
    np.savez_compressed('/home/claude/work/fields.npz', **{k + '|' + f: v for k, dd in store.items() for f, v in dd.items()})
