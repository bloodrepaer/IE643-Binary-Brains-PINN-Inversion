"""v18 supplied-physics/protocol integrity before truth scoring, all five cases."""
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
from compare_neural import TemperatureNetwork,temperature_value,residual,field_regularizer
from temporal_density_pair import KEYS,ARMS,STEP,spatial_operator


def main():
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-dir',type=Path,default=Path('campaign/pilots/temporal_v18_gpu'))
    p.add_argument('--allow-partial-case-report',action='store_true',help='Report only cases with all three completed arms; retain incomplete overall coverage.')
    p.add_argument('--validate-only',action='store_true',help='Never open coefficient answer keys.')
    a=p.parse_args();h=a.pilot_dir
    known=json.loads((h/'validation.json').read_text());assert known['coverage_complete']
    protocol=json.loads((h/'pilot.json').read_text());root=Path(protocol['data_collection'])
    entries={e['key']:e for e in json.loads((root/'manifest.json').read_text())};found={};checks=[]
    for path in (h/'artifacts').rglob('metrics.json'):
        m=json.loads(path.read_text());pair=(m['key'],m['arm']);assert pair not in found
        assert m['key'] in KEYS and m['arm'] in ARMS and m['time_step_tau']==STEP
        assert m['n_sensors']==3969 and m['n_pde']==4096 and m['optimizer']=='ssb'
        assert m['method']==('adjoint_nn' if m['arm']=='adjoint_nn' else 'pinn')
        s=torch.load(path.with_name('model.pt'),map_location='cpu',weights_only=True)
        ds=Dataset(root,entries[m['key']]);assert len(ds.u)==31752 and len(np.unique(np.c_[ds.x,ds.y],axis=0))==3969
        c=NeuralDiffusivity.from_checkpoint(s['coefficient'])
        with torch.no_grad():D=c.diffusivity(grid_coordinates(ds.N)).numpy().reshape(ds.N,ds.N)
        with np.load(path.with_name('D_hat.npz')) as z:diff=float(np.max(abs(D-z['D_hat'])))
        assert diff<1e-12 and np.isfinite(D).all() and D.min()>=.001-1e-12 and D.max()<=10+1e-12
        solver=AdjointSolver(ds,n_steps=m['n_steps']);r=float(np.sqrt(np.mean((solver.predict(solver.forward(D.ravel())[0])-ds.u)**2))/ds.sigma)
        assert abs(r-m['rms_replay_over_sigma'])<1e-8
        if m['method']=='pinn':
            q=s['final_training_points'];assert len(q)==4096 and float(q[:,2].min())>=0 and float(q[:,2].max())+STEP<=1+1e-14
            temp=TemperatureNetwork(**s['temperature_config']);temp.load_state_dict(s['temperature_state_dict'])
            q0=q.clone().requires_grad_(True);q1=(q+torch.tensor([0.,0.,STEP],dtype=torch.float64)).requires_grad_(True)
            if m['arm']=='temporal_pinn':
                err=temperature_value(temp,c,q1)-temperature_value(temp,c,q0)-STEP*spatial_operator(temp,c,q1,ds.T)
            else:err=STEP*residual(temp,c,q1,ds.T)
            physics=float((err/m['scale']).square().mean().detach())
            assert abs(physics-m['final_parts']['pde'])<1e-12
        found[pair]=(m,D);checks.append(dict(key=m['key'],arm=m['arm'],checkpoint_max_diff=diff,replay_sigma=r))
    expected={(k,a) for k in KEYS for a in ARMS};missing=sorted(expected-set(found))
    for key in KEYS:
        case=[m for (k,_),(m,_) in found.items() if k==key]
        for field in ['initial_coefficient_sha256','gamma','alpha','scale','seed']:
            assert len({m[field] for m in case})<=1,(key,field)
        expected_initial=protocol.get('initial_coefficient_sha256_by_key',{}).get(key)
        if expected_initial:assert all(m['initial_coefficient_sha256']==expected_initial for m in case)
    eligible=[key for key in KEYS if all((key,arm) in found and found[(key,arm)][0]['status']=='complete' for arm in ARMS)]
    integrity=dict(coverage_complete=not missing,missing=missing,coefficient_truth_used=False,
        fully_completed_cases=eligible,
        shared_initialization_objective_verified=True,results=checks)
    (h/'pair_integrity.json').write_text(json.dumps(integrity,indent=2))
    if missing and not a.allow_partial_case_report:raise ValueError(f'All fifteen outputs required; missing arms not losses: {missing}')
    if a.validate_only:return
    rows=[]
    for (key,arm),(m,D) in found.items():
        if a.allow_partial_case_report and key not in eligible:continue
        with np.load(root/entries[key]['files']['ground_truth']) as z:truth=z['D_inv']
        r=dict(key=key,arm=arm,method=m['method'],rel_L2_D=float(np.linalg.norm(D-truth)/np.linalg.norm(truth)),
            replay_sigma=m['rms_replay_over_sigma'],sensor_fit_sigma=m['rms_unet_over_sigma'],status=m['status'],
            adam_completed=m['adam_completed'],quasi_newton_iterations=sum(x['iterations'] for x in m['optimizer_diagnostics']),
            capped_rounds=sum(x['reason']=='iteration_cap' for x in m['optimizer_diagnostics']),
            function_evaluations=m['function_evaluations'],wall_s=m['wall_s'],n_sensors=3969,time_step_tau=STEP)
        rows.append(r)
    frame=pd.DataFrame(rows)
    filename='partial_complete_case_comparison.csv' if a.allow_partial_case_report else 'primary_matched_neural_comparison.csv'
    frame.to_csv(h/filename,index=False)
    print(frame.to_string(index=False))


if __name__=='__main__':main()
