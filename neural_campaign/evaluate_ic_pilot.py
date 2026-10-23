"""Evaluate the separately retained IC pilot using known inputs and CN replay.

No coefficient truth is used for the validation decision. Existing reported
coefficient errors are deliberately omitted from the gate evaluation.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import campaign as C
from adjoint_inversion import Dataset, AdjointSolver
from adjoint_nn import grid_coordinates
from compare_neural import TemperatureNetwork, residual, temperature_value
from neural_diffusivity import NeuralDiffusivity


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir',default='campaign/pilots/ic_v1_gpu')
    parser.add_argument('--data',default='datasets_v2_rk4_zero',help='Canonical observation collection for this separate pilot')
    args=parser.parse_args()
    home=Path(args.pilot_dir).resolve()
    data_root=(C.ROOT/args.data).resolve()
    entries={e['key']:e for e in json.loads((data_root/'manifest.json').read_text())}
    old=pd.read_csv(C.HOME/'summary.csv')
    rows=[];bins=[]
    for path in sorted((home/'artifacts').rglob('metrics.json')):
        metrics=json.loads(path.read_text());key=metrics['key']
        if metrics.get('method')=='adjoint_nn':continue  # Separate paired-arm evaluator handles hard-state checkpoints.
        ds=Dataset(data_root,entries[key])
        checkpoint=torch.load(path.parent/'model.pt',map_location='cpu',weights_only=True)
        temp=TemperatureNetwork(**checkpoint['temperature_config'])
        temp.load_state_dict(checkpoint['temperature_state_dict'])
        if checkpoint['coefficient'].get('kind')=='zhang_exp':
            from pinn_zhang_transfer import PaperCoefficient
            coeff=PaperCoefficient.from_checkpoint(checkpoint['coefficient'])
        else:
            coeff=NeuralDiffusivity.from_checkpoint(checkpoint['coefficient'])
        value=lambda points:temperature_value(temp,coeff,points)
        xy=grid_coordinates(ds.N)
        with torch.no_grad():
            initial=value(torch.cat([xy,torch.zeros((len(xy),1),dtype=torch.float64)],1)).numpy()
            D=coeff.diffusivity(xy).numpy().ravel()
            obs=value(torch.tensor(np.c_[ds.x,ds.y,ds.t/ds.T],dtype=torch.float64)).numpy()
        solver=AdjointSolver(ds,n_steps=metrics['n_steps'])
        replay=solver.predict(solver.forward(D)[0])
        row=dict(key=key,artifact=str(path.relative_to(home/'artifacts')),variant=metrics.get('variant',metrics.get('state_model','legacy')),
                 ic_relative_L2=float(np.linalg.norm(initial-ds.u0.ravel())/np.linalg.norm(ds.u0)),
                 replay_sigma=float(np.sqrt(np.mean((replay-ds.u)**2))/ds.sigma),
                 replay_metric_difference=float(np.sqrt(np.mean((replay-ds.u)**2))/ds.sigma-metrics['rms_replay_over_sigma']),
                 sensor_fit_sigma=float(np.sqrt(np.mean((obs-ds.u)**2))/ds.sigma),
                 D_spatial_std=float(np.std(D)),D_mean=float(np.mean(D)),D_fitted_constant=metrics['D_const'],
                 constant_replay_sigma=metrics['rms_const_over_sigma'],
                 training_seconds=metrics['wall_s'])
        rng=np.random.default_rng(20261019)
        wall=rng.uniform(0,1,(2048,3));edges=rng.integers(0,4,len(wall));target=np.zeros(len(wall))
        for i,name in enumerate(('left','right','bottom','top')):
            mask=edges==i;wall[mask,0 if i<2 else 1]=i%2
            target[mask]=[ds.bc.at(t*ds.T)[name] for t in wall[mask,2]]
        with torch.no_grad():err=value(torch.tensor(wall,dtype=torch.float64)).numpy()-target
        row['bc_rms']=float(np.sqrt(np.mean(err**2)));row['bc_rms_over_sensor_sigma']=row['bc_rms']/ds.sigma
        for low,high in [(0,.001),(.001,.01),(.01,.1),(.1,1)]:
            points=rng.uniform(0,1,(2048,3));points[:,2]=rng.uniform(low,high,len(points))
            x=torch.tensor(points,dtype=torch.float64,requires_grad=True)
            r=residual(temp,coeff,x,ds.T).detach().numpy()
            bins.append(dict(key=key,artifact=row['artifact'],variant=row['variant'],tau_low=low,tau_high=high,pde_rms=float(np.sqrt(np.mean(r*r))),
                             pde_max_abs=float(np.max(np.abs(r)))))
        for method in ['pinn','adjoint_nn']:
            candidates=old[(old.key==key)&(old.method==method)]
            if 'collection' in candidates:candidates=candidates[candidates.collection==Path(args.data).name]
            row[method+'_v0_replay_sigma']=float(candidates.iloc[0].replay) if len(candidates) else None
        row['replay_improvement_over_constant_percent']=100*(1-row['replay_sigma']/row['constant_replay_sigma'])
        row['consistency_screen_passed']=bool(row['ic_relative_L2']<=1e-6 and row['replay_sigma']<=1.1 and row['bc_rms_over_sensor_sigma']<=1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(home/'validation.csv',index=False)
    pd.DataFrame(bins).to_csv(home/'pde_time_bins.csv',index=False)
    passes=all(r['ic_relative_L2']<=1e-6 and r['replay_sigma']<=1.1 and r['bc_rms_over_sensor_sigma']<=1 for r in rows)
    protocol=json.loads((home/'pilot.json').read_text()) if (home/'pilot.json').exists() else {}
    expected_cases=protocol.get('cases',[]);expected_variants=protocol.get('variants',[])
    coverage_complete=bool(rows) and set(expected_cases)<=set(r['key'] for r in rows) and set(expected_variants)<=set(r['variant'] for r in rows)
    C.atomic_json(home/'validation.json',dict(updated=C.now(),validation_seed=20261019,
        selection_uses_true_D=False,results=rows,time_bins=bins,
        coverage_complete=coverage_complete,consistency_screen_passed=coverage_complete and passes,
        decision='Hold broad PINN migration: pilot coverage incomplete.' if not coverage_complete else
                 'Requires review of all independent residuals before migration.' if passes else
                 'Hold broad PINN migration: at least one IC, boundary or replay consistency screen fails.',
        screening_limits=dict(ic_relative_L2=1e-6,replay_sigma=1.1,bc_rms_over_sensor_sigma=1)))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(bins).to_string(index=False))


if __name__=='__main__':main()
