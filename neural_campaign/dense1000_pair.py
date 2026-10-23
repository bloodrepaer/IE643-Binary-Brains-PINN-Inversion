"""v22 single1000-sensor temporal PINN / matched BE neural-adjoint pair."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse,json
from pathlib import Path
import numpy as np
import calibrated_neural_pair as base
from adjoint_inversion import AdjointSolver,Dataset
from temporal_density_pair import TemporalObjective,STEP
from matched_temporal_adjoint import MatchedObjective,BackwardEulerSolver

KEY='A_smooth__Fo0.1__uniform8__n1000__fine_nested__eta0.05'
ARMS=('temporal_pinn','be_adjoint_nn')

def objective(ds,method,device='cpu',n_pde=4096,seed=42):
    if method=='adjoint_nn':return MatchedObjective(ds,method,device,n_pde,seed)
    return TemporalObjective(ds,method,device,n_pde,seed,arm='temporal_pinn')

def train(a):
    if a.key!=KEY or a.arm not in ARMS:raise ValueError('Outside predeclared single case')
    method='pinn' if a.arm=='temporal_pinn' else 'adjoint_nn'
    args=argparse.Namespace(data=a.data,key=a.key,out=str(Path(a.out)/a.key/a.arm),method=method,
        device=a.device,budget=a.budget,adam=a.adam,outer=a.outer,inner=a.inner,n_pde=4096)
    original=base.PairObjective
    try:base.PairObjective=objective;base.train(args)
    finally:base.PairObjective=original
    p=Path(args.out)/method;m=json.loads((p/'metrics.json').read_text());root=Path(a.data)
    entry=next(e for e in json.loads((root/'manifest.json').read_text()) if e['key']==KEY)
    ds=Dataset(root,entry)
    with np.load(p/'D_hat.npz') as z:D=z['D_hat'].ravel()
    scores={}
    for name,solver in [('cn',AdjointSolver(ds)),('be',BackwardEulerSolver(ds))]:
        scores['common_'+name+'_replay_sigma']=float(np.sqrt(np.mean((solver.predict(solver.forward(D)[0])-ds.u)**2))/ds.sigma)
    m.update(variant='dense1000_matched_temporal_v22',arm=a.arm,n_sensors=1000,n_readings=8000,
        n_pde=4096,time_step_tau=STEP,optimizer='ssb',time_integrator='soft_backward_euler' if method=='pinn' else 'backward_euler',
        interpretation='SharedBEtime-step/dataset/coefficient/init/optimizer;softADstateversushardFDintrinsic;allcaps/partialsretained',**scores)
    (p/'metrics.json').write_text(json.dumps(m,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--key',default=KEY);p.add_argument('--out',required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--budget',type=float,default=4200);p.add_argument('--adam',type=int,default=1000)
    p.add_argument('--outer',type=int,default=10);p.add_argument('--inner',type=int,default=400)
    train(p.parse_args())
