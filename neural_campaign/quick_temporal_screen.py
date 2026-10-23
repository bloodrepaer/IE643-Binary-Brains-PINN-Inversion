"""Short, predeclared ten-case temporal PINN versus neural-adjoint screen.

Exploratory fixed budgets, not a convergence experiment. No truth in training.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import argparse, datetime, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import calibrated_neural_pair as base
from temporal_density_pair import TemporalObjective, FIELDS

HOME = Path('campaign/pilots/quick_temporal_v19_local')
ROOT = Path('datasets_v2_density_extended')
KEYS = tuple(f'{f}__Fo0.1__uniform8__n{n}__nodes_nested__eta0.05'
             for f in FIELDS for n in (400, 3969))
ARMS = ('temporal_pinn', 'adjoint_nn')
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p, value): p.write_text(json.dumps(value, indent=2))

def fit(a):
    assert a.key in KEYS and a.arm in ARMS
    method = 'pinn' if a.arm == 'temporal_pinn' else 'adjoint_nn'
    original = base.PairObjective
    base.PairObjective = lambda ds, method, device='cpu', n_pde=4096, seed=42: TemporalObjective(
        ds, method, device, n_pde, seed, arm=a.arm)
    try:
        base.train(argparse.Namespace(data=str(ROOT), key=a.key,
            out=str(HOME/'artifacts'/a.key/a.arm), method=method, device='cpu',
            budget=120., adam=a.adam, outer=1, inner=a.inner, n_pde=4096))
    finally: base.PairObjective = original

def evaluate(score=True):
    from adjoint_inversion import Dataset, AdjointSolver
    from adjoint_nn import grid_coordinates
    from neural_diffusivity import NeuralDiffusivity
    from compare_neural import TemperatureNetwork, temperature_value, residual
    torch.set_num_threads(2)
    entries = {e['key']: e for e in json.loads((ROOT/'manifest.json').read_text())}
    checked = []
    for key in KEYS:
        pair = []
        ds = Dataset(ROOT, entries[key])
        for arm in ARMS:
            method = 'pinn' if arm == 'temporal_pinn' else 'adjoint_nn'
            p = HOME/'artifacts'/key/arm/method
            if not (p/'metrics.json').exists(): continue
            m = json.loads((p/'metrics.json').read_text())
            s = torch.load(p/'model.pt', map_location='cpu', weights_only=True)
            c = NeuralDiffusivity.from_checkpoint(s['coefficient'])
            xy = grid_coordinates(ds.N)
            with torch.no_grad(): D = c.diffusivity(xy).numpy().reshape(ds.N, ds.N)
            with np.load(p/'D_hat.npz') as z: assert np.max(abs(D-z['D_hat'])) < 1e-12
            assert np.isfinite(D).all() and D.min() >= .001-1e-12 and D.max() <= 10+1e-12
            solver = AdjointSolver(ds, n_steps=m['n_steps'])
            replay = np.sqrt(np.mean((solver.predict(solver.forward(D.ravel())[0])-ds.u)**2))/ds.sigma
            assert abs(replay-m['rms_replay_over_sigma']) < 1e-8
            row = dict(key=key, arm=arm, status=m['status'], replay_sigma=float(replay),
                fit_sigma=m['rms_unet_over_sigma'], adam=m['adam_completed'],
                qn_iterations=sum(r['iterations'] for r in m['optimizer_diagnostics']),
                evaluations=m['function_evaluations'], wall_s=m['wall_s'],
                initial_sha=m['initial_coefficient_sha256'], gamma=m['gamma'], scale=m['scale'])
            if method == 'pinn':
                t = TemperatureNetwork(**s['temperature_config']); t.load_state_dict(s['temperature_state_dict'])
                with torch.no_grad():
                    ic = temperature_value(t,c,torch.cat([xy,torch.zeros((len(xy),1),dtype=torch.float64)],1)).numpy()
                    rng = np.random.default_rng(20261021)
                    boundary = rng.uniform(0,1,(1024,3))
                    for edge in range(4): boundary[edge*256:(edge+1)*256,edge//2] = edge%2
                    bc = temperature_value(t,c,torch.tensor(boundary,dtype=torch.float64)).numpy()
                row['ic_relative'] = float(np.linalg.norm(ic-ds.u0.ravel())/np.linalg.norm(ds.u0))
                row['bc_sigma'] = float(np.sqrt(np.mean(bc**2))/ds.sigma)
                q = torch.tensor(rng.uniform(0,1,(1024,3)),dtype=torch.float64,requires_grad=True)
                row['heldout_continuous_pde_rms'] = float(residual(t,c,q,ds.T).square().mean().sqrt().detach())
                row['physics_screen'] = bool(row['ic_relative'] <= 1e-6 and row['bc_sigma'] <= 1 and replay <= 1.1)
            else: row['physics_screen'] = bool(replay <= 1.1)
            pair.append((row,D))
        if len(pair)==2:
            for f in ('initial_sha','gamma','scale'): assert pair[0][0][f]==pair[1][0][f], (key,f)
        checked.extend(pair)
    # This artifact is saved before opening any answer key.
    save(HOME/'validation_before_truth.json', dict(updated=now(), truth_used=False,
         coverage_complete=len(checked)==2*len(KEYS), rows=[r for r,_ in checked]))
    if not score: return
    rows=[]
    for r,D in checked:
        with np.load(ROOT/entries[r['key']]['files']['ground_truth']) as z: truth=z['D_inv']
        r['rel_L2_D']=float(np.linalg.norm(D-truth)/np.linalg.norm(truth)); rows.append(r)
    decisions=[]
    for key in KEYS:
        pair={r['arm']:r for r in rows if r['key']==key}
        eligible=len(pair)==2 and all(r['status']=='complete' for r in pair.values())
        win=eligible and pair['temporal_pinn']['physics_screen'] and pair['temporal_pinn']['rel_L2_D']<pair['adjoint_nn']['rel_L2_D']
        decisions.append(dict(key=key, complete_pair=eligible, preliminary_pinn_advantage=bool(win),
            interpretation='Short-budget exploratory screen only; incomplete pairs are not losses.'))
    save(HOME/'screen_results.json',dict(updated=now(),rows=rows,decisions=decisions))

def run():
    HOME.mkdir(parents=True,exist_ok=False); (HOME/'logs').mkdir()
    source_names=('quick_temporal_screen.py','temporal_density_pair.py','calibrated_neural_pair.py',
        'adjoint_inversion.py','adjoint_nn.py','compare_neural.py','integrators.py',
        'neural_diffusivity.py','pinn_papers.py','pinn_zhang_transfer.py')
    hashes={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in source_names}
    (HOME/'source_snapshot').mkdir()
    for n in source_names: shutil.copy2(n,HOME/'source_snapshot'/n)
    save(HOME/'source_sha256.json',hashes)
    entries={e['key'] for e in json.loads((ROOT/'manifest.json').read_text())}
    assert set(KEYS)<=entries
    save(HOME/'protocol.json',dict(created=now(),cases=KEYS,arms=ARMS,
        adam=50,outer=1,inner=50,n_pde=4096,per_arm_internal_s=120,per_arm_external_s=180,
        temporal_step=.01,device='cpu',threads=2,seed=42,
        scope='20 fresh short-budget fits, outside broad1510 and v18. No truth selection; all outcomes retained.',
        limitation='Early optimization screen, not converged superiority or full Thakur reproduction.'))
    # Avoid competing with the previously authorized local control worker.
    old=Path('campaign/pilots/temporal_v18_gpu/local_worker.json')
    while old.exists() and json.loads(old.read_text()).get('status')=='running':
        save(HOME/'worker.json',dict(status='waiting_for_existing_local_worker',updated=now()))
        time.sleep(30)
    finished=[]
    for key in KEYS:
        for arm in ARMS:
            assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in hashes.items()), 'Source changed during screen'
            if shutil.disk_usage(HOME).free < 128*1024**2: raise RuntimeError('128MiB disk guard')
            save(HOME/'worker.json',dict(status='running',key=key,arm=arm,finished=finished,updated=now()))
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--fit','--key',key,'--arm',arm]
            try:
                with (HOME/'logs'/f'{key}__{arm}.log').open('w') as log:
                    result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                outcome=dict(key=key,arm=arm,returncode=result.returncode)
            except subprocess.TimeoutExpired: outcome=dict(key=key,arm=arm,status='external_timeout')
            finished.append(outcome)
    evaluate()
    save(HOME/'worker.json',dict(status='finished',finished=finished,updated=now()))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--fit',action='store_true');p.add_argument('--evaluate',action='store_true')
    p.add_argument('--validate-only',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--key');p.add_argument('--arm');p.add_argument('--adam',type=int,default=50);p.add_argument('--inner',type=int,default=50)
    a=p.parse_args()
    if a.smoke:
        HOME=Path('campaign/pilots/quick_temporal_v19_smoke')
        KEYS=(KEYS[0],)
    if a.fit: fit(a)
    elif a.evaluate: evaluate(not a.validate_only)
    else: run()
