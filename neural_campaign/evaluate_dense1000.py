"""Independent v22 supplied-physics/checkpoint/protocol validation before truth."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from adjoint_inversion import Dataset,AdjointSolver
from adjoint_nn import grid_coordinates
from neural_diffusivity import NeuralDiffusivity
from compare_neural import TemperatureNetwork,temperature_value
from temporal_density_pair import STEP,spatial_operator
from matched_temporal_adjoint import BackwardEulerSolver
from dense1000_pair import KEY,ARMS

def main():
    torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('--pilot-dir',type=Path,default=Path('campaign/pilots/dense1000_v22_kaggle'))
    p.add_argument('--validate-only',action='store_true');a=p.parse_args();h=a.pilot_dir
    assert json.loads((h/'validation.json').read_text())['coverage_complete']
    protocol=json.loads((h/'pilot.json').read_text());root=Path(protocol['data_collection'])
    entry=json.loads((root/'manifest.json').read_text())[0];ds=Dataset(root,entry)
    assert len(ds.u)==8000 and len(np.unique(np.c_[ds.x,ds.y],axis=0))==1000
    rows=[];fields=[]
    for arm in ARMS:
        method='pinn' if arm=='temporal_pinn' else 'adjoint_nn';path=h/'artifacts/results'/KEY/arm/method
        m=json.loads((path/'metrics.json').read_text());s=torch.load(path/'model.pt',map_location='cpu',weights_only=True)
        assert m['arm']==arm and m['n_sensors']==1000 and m['n_readings']==8000 and m['time_step_tau']==STEP
        assert m['n_pde']==4096 and m['optimizer']=='ssb' and m['method']==method
        if method=='adjoint_nn':assert m['n_steps']==100 and m['time_integrator']=='backward_euler'
        c=NeuralDiffusivity.from_checkpoint(s['coefficient']);xy=grid_coordinates(ds.N)
        with torch.no_grad():D=c.diffusivity(xy).numpy().reshape(ds.N,ds.N)
        with np.load(path/'D_hat.npz') as z:diff=float(np.max(abs(D-z['D_hat'])))
        assert diff<1e-12 and np.isfinite(D).all() and D.min()>=.001-1e-12 and D.max()<=10+1e-12
        replays={}
        for name,solver in [('cn',AdjointSolver(ds)),('be',BackwardEulerSolver(ds))]:
            value=float(np.sqrt(np.mean((solver.predict(solver.forward(D.ravel())[0])-ds.u)**2))/ds.sigma)
            assert abs(value-m['common_'+name+'_replay_sigma'])<1e-8
            replays['replay_'+name+'_sigma']=value
        if method=='pinn':
            t=TemperatureNetwork(**s['temperature_config']);t.load_state_dict(s['temperature_state_dict'])
            q=s['final_training_points'];assert len(q)==4096 and q[:,2].min()>=0 and q[:,2].max()+STEP<=1+1e-14
            q0=q.clone().requires_grad_(True);q1=(q+torch.tensor([0.,0.,STEP],dtype=torch.float64)).requires_grad_(True)
            defect=temperature_value(t,c,q1)-temperature_value(t,c,q0)-STEP*spatial_operator(t,c,q1,ds.T)
            value=float((defect/m['scale']).square().mean().detach());assert abs(value-m['final_parts']['pde'])<1e-12
        shared=tuple(m[n] for n in ('initial_coefficient_sha256','gamma','scale','alpha','seed'));fields.append(shared)
        assert shared[0]==protocol['initial_coefficient_sha256']
        row=dict(arm=arm,status=m['status'],n_sensors=1000,n_readings=8000,
            initial_coefficient_sha256=shared[0],gamma=m['gamma'],checkpoint_difference=diff,
            sensor_fit_sigma=m['rms_unet_over_sigma'],adam_completed=m['adam_completed'],
            quasi_newton_iterations=sum(r['iterations'] for r in m['optimizer_diagnostics']),
            capped_rounds=sum(r['reason']=='iteration_cap' for r in m['optimizer_diagnostics']),
            function_evaluations=m['function_evaluations'],wall_s=m['wall_s'],**replays)
        rows.append((row,D))
    assert fields[0]==fields[1]
    (h/'pair_integrity.json').write_text(json.dumps(dict(coverage_complete=True,coefficient_truth_used=False,
        shared_initialization_objective_verified=True,results=[r for r,_ in rows]),indent=2))
    if a.validate_only:return
    # Coefficient answer keys are first accessed after all supplied checks.
    with np.load(root/entry['files']['ground_truth']) as z:truth=z['D_inv']
    for row,D in rows:row['rel_L2_D']=float(np.linalg.norm(D-truth)/np.linalg.norm(truth))
    frame=pd.DataFrame([r for r,_ in rows]);frame.to_csv(h/'primary_matched_neural_comparison.csv',index=False)
    print(frame.to_string(index=False))

if __name__=='__main__':main()
