"""Independent local checks for the forward-only diagnostic; no true D."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from compare_neural import TemperatureNetwork,temperature_value,residual
from neural_diffusivity import NeuralDiffusivity
from adjoint_nn import grid_coordinates
from adjoint_inversion import Dataset,AdjointSolver


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot-dir',required=True)
    ap.add_argument('--artifacts',help='Explicit smoke-test artifacts; never counted as remote results')
    args=ap.parse_args()
    home=Path(args.pilot_dir);folder=Path(args.artifacts) if args.artifacts else home/'artifacts'
    torch.set_num_threads(1)
    protocol=json.loads((home/'pilot.json').read_text())
    reference=Path(protocol['reference'])
    if hashlib.sha256(reference.read_bytes()).hexdigest()!=protocol['reference_sha256']:raise ValueError('Reference hash changed')
    original=torch.load(reference,map_location='cpu',weights_only=True)
    saved=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    if saved['method']!='forward_diagnostic' or saved['key']!=protocol['key']:raise ValueError('Wrong diagnostic checkpoint')
    a=original['coefficient'];b=saved['coefficient']
    identical=a['config']==b['config'] and set(a['state_dict'])==set(b['state_dict']) and all(torch.equal(v,b['state_dict'][k]) for k,v in a['state_dict'].items())
    if not identical:raise ValueError('Fixed coefficient weights changed')
    metrics=json.loads((folder/'metrics.json').read_text())
    if metrics['reference_sha256']!=protocol['reference_sha256']:raise ValueError('Remote reference mismatch')
    root=Path('datasets_v2_rk4_zero')
    entry=next(e for e in json.loads((root/'manifest.json').read_text()) if e['key']==saved['key'])
    ds=Dataset(root,entry)
    temperature=TemperatureNetwork(**saved['temperature_config']);temperature.load_state_dict(saved['temperature_state_dict'])
    coefficient=NeuralDiffusivity.from_checkpoint(b)
    value=lambda x:temperature_value(temperature,coefficient,torch.tensor(x,dtype=torch.float64))
    xy=grid_coordinates(ds.N).numpy()
    with torch.no_grad():
        initial=value(np.c_[xy,np.zeros(len(xy))]).numpy()
        fitted=value(np.c_[ds.x,ds.y,ds.t/ds.T]).numpy()
        D=coefficient.diffusivity(torch.tensor(xy,dtype=torch.float64)).numpy().ravel()
    solver=AdjointSolver(ds,n_steps=200);cn=solver.predict(solver.forward(D)[0])
    calculated=dict(ic_relative_L2=float(np.linalg.norm(initial-ds.u0.ravel())/np.linalg.norm(ds.u0)),
                    sensor_fit_sigma=float(np.sqrt(np.mean((fitted-ds.u)**2))/ds.sigma),
                    fixed_coefficient_CN_sensor_sigma=float(np.sqrt(np.mean((cn-ds.u)**2))/ds.sigma),
                    state_vs_CN_sensor_sigma=float(np.sqrt(np.mean((fitted-cn)**2))/ds.sigma))
    differences={k:calculated[k]-metrics[k] for k in calculated}
    reproduced=all(abs(v)<1e-8 for v in differences.values())
    if not reproduced:raise ValueError('Retained checkpoint does not reproduce metrics: '+str(differences))
    observed=np.load(folder/'sensor_evaluation.npz')
    array_difference=max(float(np.max(np.abs(fitted-observed['state']))),float(np.max(np.abs(cn-observed['CN']))))
    if array_difference>1e-10:raise ValueError('Retained predictions differ')
    rng=np.random.default_rng(20261025)
    wall=rng.uniform(0,1,(2048,3));edges=rng.integers(0,4,len(wall))
    for i in range(4):wall[edges==i,0 if i<2 else 1]=i%2
    with torch.no_grad():boundary=value(wall).numpy()
    bins=[]
    for low,high in [(0,.001),(.001,.01),(.01,.1),(.1,1)]:
        points=rng.uniform(0,1,(1024,3));points[:,2]=rng.uniform(low,high,len(points));values=[]
        for offset in range(0,len(points),256):
            x=torch.tensor(points[offset:offset+256],dtype=torch.float64,requires_grad=True)
            values.append(residual(temperature,coefficient,x,ds.T).detach().numpy())
        r=np.concatenate(values)
        bins.append(dict(tau_low=low,tau_high=high,pde_rms=float(np.sqrt(np.mean(r*r))),pde_max=float(np.max(np.abs(r)))))
    pd.DataFrame(bins).to_csv(folder/'independent_pde_time_bins.csv',index=False)
    result=dict(uses_true_D=False,validation_seed=20261025,coefficient_bitwise_unchanged=identical,
                checkpoint_metrics_reproduced=reproduced,metric_differences=differences,retained_array_max_difference=array_difference,
                measurements=calculated,bc_rms=float(np.sqrt(np.mean(boundary**2))),pde_time_bins=bins,
                note='Forward-state diagnostic only. Finite-difference replay is a discretized comparator; these checks do not prove coefficient recovery or reopen the inverse gate.')
    (folder/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
