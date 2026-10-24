"""v23 data-driven adapter; unchanged temporal and BE neural objectives."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse,json
from pathlib import Path
import numpy as np
import calibrated_neural_pair as base
from adjoint_inversion import Dataset,AdjointSolver
from temporal_density_pair import TemporalObjective,STEP
from matched_temporal_adjoint import MatchedObjective,BackwardEulerSolver

ARMS=('temporal_pinn','be_adjoint_nn')
def objective(ds,method,device='cpu',n_pde=4096,seed=42):
    if method=='adjoint_nn':return MatchedObjective(ds,method,device,n_pde,seed)
    return TemporalObjective(ds,method,device,n_pde,seed,arm='temporal_pinn')

def train(a):
    root=Path(a.data);entry=next(e for e in json.loads((root/'manifest.json').read_text()) if e['key']==a.key)
    assert entry['n_sensors'] in (1000,3969,10000) and entry['eta'] in (.01,.05,.1)
    assert not (entry['eta']==.05 and entry['n_sensors'] in (1000,10000)), 'Frozen reference, do not rerun'
    method='pinn' if a.arm=='temporal_pinn' else 'adjoint_nn'
    args=argparse.Namespace(data=a.data,key=a.key,out=str(Path(a.out)/a.key/a.arm),method=method,
        device=a.device,budget=a.budget,adam=a.adam,outer=a.outer,inner=a.inner,n_pde=4096)
    original=base.PairObjective
    try:base.PairObjective=objective;base.train(args)
    finally:base.PairObjective=original
    path=Path(args.out)/method;m=json.loads((path/'metrics.json').read_text());ds=Dataset(root,entry)
    with np.load(path/'D_hat.npz') as z:D=z['D_hat'].ravel()
    scores={}
    for name,solver in [('cn',AdjointSolver(ds)),('be',BackwardEulerSolver(ds))]:
        rms=float(np.sqrt(np.mean((solver.predict(solver.forward(D)[0])-ds.u)**2)))
        scores['common_'+name+'_replay_sigma']=rms/ds.sigma;scores['common_'+name+'_replay_rms']=rms
    m.update(variant='noise_density_temporal_v23',arm=a.arm,n_sensors=entry['n_sensors'],n_readings=len(ds.u),eta=entry['eta'],
        n_pde=4096,time_step_tau=STEP,optimizer='ssb',time_integrator='soft_backward_euler' if method=='pinn' else 'backward_euler',
        sensor_fit_rms=m['rms_unet_over_sigma']*ds.sigma,**scores)
    (path/'metrics.json').write_text(json.dumps(m,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--key',required=True);p.add_argument('--out',required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--device',default='cpu');p.add_argument('--budget',type=float,default=12600)
    p.add_argument('--adam',type=int,default=1000);p.add_argument('--outer',type=int,default=10);p.add_argument('--inner',type=int,default=400)
    train(p.parse_args())
