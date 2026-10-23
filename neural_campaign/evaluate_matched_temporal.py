"""v20 matched-time controls: supplied physics/checkpoints before answer keys."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from adjoint_inversion import AdjointSolver,Dataset
from adjoint_nn import grid_coordinates
from neural_diffusivity import NeuralDiffusivity
from compare_neural import TemperatureNetwork,temperature_value,residual
from matched_temporal_adjoint import BackwardEulerSolver,HOME,SUPPLIED,KEYS,STEP

def evaluate(control_root,score):
    torch.set_num_threads(2)
    protocol=json.loads((HOME/'pilot.json').read_text())
    original=Path('campaign/pilots/temporal_v18_gpu')
    entries={e['key']:e for e in json.loads((SUPPLIED/'manifest.json').read_text())}
    records=[];missing=[]
    for key in KEYS:
        ds=Dataset(SUPPLIED,entries[key]);cn=AdjointSolver(ds);be=BackwardEulerSolver(ds)
        pair=[]
        for arm,p in [('temporal_pinn',original/'artifacts/results'/key/'temporal_pinn/pinn'),
                      ('be_adjoint_nn',control_root/key/'adjoint_nn')]:
            if not (p/'metrics.json').exists():missing.append([key,arm]);continue
            m=json.loads((p/'metrics.json').read_text());s=torch.load(p/'model.pt',map_location='cpu',weights_only=True)
            if arm=='temporal_pinn':
                assert hashlib.sha256((p/'model.pt').read_bytes()).hexdigest()==protocol['retained_pinn_checkpoint_sha256'][key]
            else:
                assert m['time_integrator']=='backward_euler' and m['n_steps']==100 and m['time_step_tau']==STEP
            assert m['initial_coefficient_sha256']==protocol['initial_coefficient_sha256_by_key'][key]
            c=NeuralDiffusivity.from_checkpoint(s['coefficient']);xy=grid_coordinates(ds.N)
            with torch.no_grad():D=c.diffusivity(xy).numpy().reshape(ds.N,ds.N)
            with np.load(p/'D_hat.npz') as z:diff=float(np.max(abs(D-z['D_hat'])))
            assert diff<1e-12 and np.isfinite(D).all() and D.min()>=.001-1e-12 and D.max()<=10+1e-12
            rcn=float(np.sqrt(np.mean((cn.predict(cn.forward(D.ravel())[0])-ds.u)**2))/ds.sigma)
            rbe=float(np.sqrt(np.mean((be.predict(be.forward(D.ravel())[0])-ds.u)**2))/ds.sigma)
            assert abs((rcn if arm=='temporal_pinn' else rbe)-m['rms_replay_over_sigma'])<1e-8
            row=dict(key=key,arm=arm,status=m['status'],initial_sha=m['initial_coefficient_sha256'],
                gamma=m['gamma'],scale=m['scale'],alpha=m['alpha'],seed=m['seed'],
                replay_cn_sigma=rcn,replay_be_sigma=rbe,training_fit_sigma=m['rms_unet_over_sigma'],
                adam=m['adam_completed'],qn_iterations=sum(x['iterations'] for x in m['optimizer_diagnostics']),
                capped_rounds=sum(x['reason']=='iteration_cap' for x in m['optimizer_diagnostics']),
                wall_s=m['wall_s'],function_evaluations=m['function_evaluations'],checkpoint_difference=diff)
            if arm=='temporal_pinn':
                t=TemperatureNetwork(**s['temperature_config']);t.load_state_dict(s['temperature_state_dict'])
                rng=np.random.default_rng(20261021)
                with torch.no_grad():
                    q=torch.cat([xy,torch.zeros((len(xy),1),dtype=torch.float64)],1)
                    ic=temperature_value(t,c,q).numpy()
                    q=rng.uniform(0,1,(2048,3))
                    for k in range(4):q[k*512:(k+1)*512,k//2]=k%2
                    bc=temperature_value(t,c,torch.tensor(q,dtype=torch.float64)).numpy()
                row['ic_relative']=float(np.linalg.norm(ic-ds.u0.ravel())/np.linalg.norm(ds.u0))
                row['bc_sigma']=float(np.sqrt(np.mean(bc**2))/ds.sigma)
                q=torch.tensor(rng.uniform(0,1,(4096,3)),dtype=torch.float64,requires_grad=True)
                row['heldout_continuous_pde_rms']=float(residual(t,c,q,ds.T).square().mean().sqrt().detach())
            row['replay_screen_both_solvers']=rcn<=1.1 and rbe<=1.1
            pair.append(row);records.append((row,D))
        if len(pair)==2:
            for field in ('initial_sha','gamma','scale','alpha','seed'):assert pair[0][field]==pair[1][field],(key,field)
    integrity=dict(coverage_complete=not missing,missing=missing,coefficient_truth_used=False,rows=[r for r,_ in records])
    report='validation_smoke.json' if not score else 'validation_before_truth.json'
    (HOME/report).write_text(json.dumps(integrity,indent=2))
    if not score:return integrity
    if missing:raise ValueError('Missing controls: no losses or full comparison claimed')
    # Answer keys are first opened only after all available protocol checks pass.
    canonical=Path('datasets_v2_density_extended')
    truth_entries={e['key']:e for e in json.loads((canonical/'manifest.json').read_text())}
    rows=[]
    for row,D in records:
        with np.load(canonical/truth_entries[row['key']]['files']['ground_truth']) as z:truth=z['D_inv']
        row['rel_L2_D']=float(np.linalg.norm(D-truth)/np.linalg.norm(truth));rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(HOME/'matched_time_comparison.csv',index=False)
    print(frame.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,default=HOME/'results');p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    evaluate(a.control_root,not a.validate_only)
