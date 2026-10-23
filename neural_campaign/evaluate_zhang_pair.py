"""Post-fit verification and scoring of the separate neural-versus-neural pair."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from adjoint_inversion import Dataset,AdjointSolver
from adjoint_nn import grid_coordinates
from pinn_zhang_transfer import PaperCoefficient


def main():
    torch.set_num_threads(1);h=Path('campaign/pilots/zhang_v11_gpu')
    validation=json.loads((h/'validation.json').read_text())
    if not validation['coverage_complete']:raise ValueError('PINN validation incomplete')
    m=json.loads((h/'matched_adjoint/metrics.json').read_text());p=json.loads((h/'pilot.json').read_text())
    root=Path(p['data_collection']);entry=next(e for e in json.loads((root/'manifest.json').read_text()) if e['key']==m['key']);ds=Dataset(root,entry)
    c=PaperCoefficient.from_checkpoint(torch.load(h/'matched_adjoint/model.pt',map_location='cpu',weights_only=True))
    D=c.diffusivity(grid_coordinates(ds.N)).detach().numpy().reshape(ds.N,ds.N);saved=np.load(h/'matched_adjoint/D_hat.npz')['D_hat'];difference=float(np.max(abs(D-saved)))
    if not np.isfinite(D).all() or np.any(D<=0) or difference>1e-12:raise ValueError('Invalid adjoint checkpoint/arrays')
    solver=AdjointSolver(ds,n_steps=m['n_steps']);replay=float(np.sqrt(np.mean((solver.predict(solver.forward(D.ravel())[0])-ds.u)**2))/ds.sigma)
    if abs(replay-m['rms_replay_over_sigma'])>1e-9:raise ValueError('Adjoint replay mismatch')
    (h/'matched_adjoint_independent_validation.json').write_text(json.dumps(dict(checkpoint_difference=difference,replay_sigma=replay,replay_metric_difference=replay-m['rms_replay_over_sigma'],coefficient_truth_used=False),indent=2))
    # Answer key only after both retained fits have independently replayed.
    with np.load(root/entry['files']['ground_truth']) as z:truth=z['D_inv'].copy()
    score=pd.read_csv(h/'coefficient_scores.csv');v=next(v for v in validation['results'] if v['key']==m['key'])
    pinn_path=next((h/'artifacts').rglob('metrics.json'));pm=json.loads(pinn_path.read_text())
    rows=[dict(method='neural_adjoint_zhang',key=m['key'],rel_L2_D=float(np.linalg.norm(D-truth)/np.linalg.norm(truth)),replay_sigma=replay,sensor_fit_sigma=replay,status=m['status'],wall_s=m['wall_s'],adam_completed=m['adam_completed'],quasi_newton_iterations=sum(r['iterations'] for r in m['optimizer_diagnostics']),function_evaluations=m['function_evaluations'],hardware='Local CPU1thread, concurrent broad CPU workers'),
          dict(method='neural_pinn_zhang',key=m['key'],rel_L2_D=float(score.iloc[0].rel_L2_D),replay_sigma=v['replay_sigma'],sensor_fit_sigma=v['sensor_fit_sigma'],status=pm['status'],wall_s=pm['wall_s'],adam_completed=pm['settings']['adam'],quasi_newton_iterations=sum(r['iterations'] for r in pm['optimizer_diagnostics']),function_evaluations=pm['quasi_newton_function_evaluations'],hardware='Kaggle CUDAresidual/CPUcurvature2threads')]
    pd.DataFrame(rows).to_csv(h/'primary_matched_neural_comparison.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
