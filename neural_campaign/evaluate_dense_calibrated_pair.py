"""Verify each predeclared dense neural pair before any answer-key score."""
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


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot-dir',default='campaign/pilots/dense_calibrated_v14_gpu')
    home=Path(ap.parse_args().pilot_dir);torch.set_num_threads(1)
    protocol=json.loads((home/'pilot.json').read_text())
    validation=json.loads((home/'validation.json').read_text())
    validated_keys={r['key'] for r in validation['results']}
    root=Path(protocol['data_collection'])
    entries={e['key']:e for e in json.loads((root/'manifest.json').read_text())}
    groups={key:{} for key in protocol['cases']}
    for path in (home/'artifacts').rglob('metrics.json'):
        m=json.loads(path.read_text());key=m['key'];method=m['method']
        if key not in groups or method in groups[key]:raise ValueError('Unexpected/duplicate case')
        groups[key][method]=(path,m)
    retained={};checks=[];missing=[]
    for key,methods in groups.items():
        if set(methods)!={'pinn','adjoint_nn'} or key not in validated_keys:
            missing.append(key);continue
        members=[]
        for method,(path,m) in methods.items():
            ds=Dataset(root,entries[key])
            saved=torch.load(path.with_name('model.pt'),map_location='cpu',weights_only=True)
            coef=NeuralDiffusivity.from_checkpoint(saved['coefficient'])
            with torch.no_grad():D=coef.diffusivity(grid_coordinates(ds.N)).numpy().reshape(ds.N,ds.N)
            with np.load(path.with_name('D_hat.npz')) as z:error=float(np.max(abs(D-z['D_hat'])))
            if error>1e-12 or not np.isfinite(D).all() or D.min()<.001-1e-12 or D.max()>10+1e-12:raise ValueError('Invalid bounded coefficient')
            solver=AdjointSolver(ds,n_steps=m['n_steps'])
            replay=float(np.sqrt(np.mean((solver.predict(solver.forward(D.ravel())[0])-ds.u)**2))/ds.sigma)
            if abs(replay-m['rms_replay_over_sigma'])>1e-9:raise ValueError('Replay mismatch')
            count=len(np.unique(np.c_[ds.x,ds.y],axis=0))
            if count!=m['n_sensors']:raise ValueError('Sensor count mismatch')
            checks.append(dict(key=key,method=method,n_sensors=count,checkpoint_difference=error,replay_sigma=replay))
            members.append((m,D))
        for field in ['initial_coefficient_sha256','gamma','seed','alpha','scale']:
            if len({m[field] for m,D in members})!=1:raise ValueError('Within-case shared protocol mismatch: '+field)
        retained[key]=members
    (home/'pair_integrity.json').write_text(json.dumps(dict(uses_true_D=False,
        coverage_complete=not missing,missing_or_unpaired_cases=missing,results=checks),indent=2))
    rows=[]
    # No answer keys are read until each retained pair has passed reconstruction.
    for key,members in retained.items():
        with np.load(root/entries[key]['files']['ground_truth']) as z:truth=z['D_inv'].copy()
        for m,D in members:
            rows.append(dict(key=key,n_sensors=m['n_sensors'],method=m['method'],
                rel_L2_D=float(np.linalg.norm(D-truth)/np.linalg.norm(truth)),
                replay_sigma=m['rms_replay_over_sigma'],sensor_fit_sigma=m['rms_unet_over_sigma'],
                status=m['status'],wall_s=m['wall_s'],adam_completed=m['adam_completed'],
                quasi_newton_iterations=sum(r['iterations'] for r in m['optimizer_diagnostics']),
                capped_rounds=sum(r['reason']=='iteration_cap' for r in m['optimizer_diagnostics']),
                function_evaluations=m['function_evaluations'],gamma=m['gamma']))
    if not rows:raise ValueError('No complete independently validated neural pair')
    table=pd.DataFrame(rows);table.to_csv(home/'primary_matched_neural_comparison.csv',index=False)
    print(table.to_string(index=False));print('All predeclared pairs covered:',not missing)


if __name__=='__main__':main()
