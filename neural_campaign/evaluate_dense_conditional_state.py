"""Validate fixed estimated-D forward diagnostics; never score coefficient truth."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from adjoint_inversion import Dataset, AdjointSolver
from adjoint_nn import grid_coordinates
from neural_diffusivity import NeuralDiffusivity
from compare_neural import TemperatureNetwork, temperature_value
from dense_conditional_state import ARMS, KEY, tensor_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-dir',type=Path,default=Path('campaign/pilots/conditional_state_v17_gpu'))
    p.add_argument('--data',type=Path,default=Path('datasets_v2_density_extended'))
    p.add_argument('--coefficient-dir',type=Path,default=Path('campaign/pilots/optimizer_v16_gpu/artifacts/results'))
    a=p.parse_args();torch.set_num_threads(1)
    known=json.loads((a.pilot_dir/'validation.json').read_text())
    if not known.get('coverage_complete'):raise ValueError('Run independent known-constraint validator first')
    entry=next(e for e in json.loads((a.data/'manifest.json').read_text()) if e['key']==KEY)
    ds=Dataset(a.data,entry);rows=[];missing=[]
    for arm in ARMS:
        matches=list((a.pilot_dir/'artifacts/results'/arm).rglob('model.pt'))
        if len(matches)!=1:missing.append(arm);continue
        path=matches[0];m=json.loads(path.with_name('metrics.json').read_text())
        saved=torch.load(path,map_location='cpu',weights_only=True)
        src=next((a.coefficient_dir/arm).rglob('model.pt'))
        original=torch.load(src,map_location='cpu',weights_only=True)
        assert m['key']==KEY and m['arm']==arm and m['coefficient_frozen'] and m['inverse_fit'] is False
        assert m['n_sensors']==3969 and m['n_pde']==16384 and m['diagnostic']=='fixed_estimated_coefficient_forward_only'
        assert hashlib.sha256(src.read_bytes()).hexdigest()==m['coefficient_source_sha256']
        for k,v in original['coefficient']['state_dict'].items():assert torch.equal(v,saved['coefficient']['state_dict'][k])
        coef=NeuralDiffusivity.from_checkpoint(saved['coefficient'])
        assert tensor_sha(coef)==m['coefficient_tensor_sha256']
        solver=AdjointSolver(ds,n_steps=m['n_steps']);xy=grid_coordinates(ds.N)
        with torch.no_grad():D=coef.diffusivity(xy).numpy().ravel()
        with np.load(path.with_name('D_hat.npz')) as z:assert np.max(abs(D-z['D_hat'].ravel()))<1e-10
        replay=solver.predict(solver.forward(D)[0])
        temp=TemperatureNetwork(**saved['temperature_config']);temp.load_state_dict(saved['temperature_state_dict'])
        xd=torch.tensor(np.c_[ds.x,ds.y,ds.t/ds.T],dtype=torch.float64)
        with torch.no_grad():obs=temperature_value(temp,coef,xd).numpy()
        replay_sigma=float(np.sqrt(np.mean((replay-ds.u)**2))/ds.sigma)
        fit_sigma=float(np.sqrt(np.mean((obs-ds.u)**2))/ds.sigma)
        assert abs(replay_sigma-m['rms_replay_over_sigma'])<1e-8
        assert abs(fit_sigma-m['rms_unet_over_sigma'])<1e-8
        rows.append(dict(arm=arm,status=m['status'],fixed_estimated_coefficient=True,
            coefficient_unchanged=True,common_CN_sigma=replay_sigma,state_sensor_fit_sigma=fit_sigma,
            state_CN_gap_sigma=float(np.sqrt(np.mean((obs-replay)**2))/ds.sigma),
            adam_completed=m['adam_completed'],quasi_newton_iterations=sum(x['iterations'] for x in m['optimizer_diagnostics']),
            capped_rounds=sum(x['reason']=='iteration_cap' for x in m['optimizer_diagnostics']),
            wall_s=m['wall_s'],function_evaluations=m['function_evaluations']))
    result=dict(uses_true_D=False,inverse_comparison=False,coefficient_truth_scored=False,
        coverage_complete=not missing,missing_arms=missing,results=rows,
        limitations='Fixed estimated coefficient forward diagnostic only. Missing arms are not losses; caps do not prove infeasibility.')
    (a.pilot_dir/'conditional_integrity.json').write_text(json.dumps(result,indent=2))
    if missing:raise ValueError(f'Missing conditional arms: {missing}')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
