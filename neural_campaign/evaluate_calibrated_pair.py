"""Verify both retained v12 arms, then post-fit score the primary neural pair."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from adjoint_inversion import Dataset,AdjointSolver
from adjoint_nn import grid_coordinates
from neural_diffusivity import NeuralDiffusivity


def main():
    torch.set_num_threads(1);h=Path('campaign/pilots/calibrated_v12_gpu');validation=json.loads((h/'validation.json').read_text());protocol=json.loads((h/'pilot.json').read_text());root=Path(protocol['data_collection'])
    if not validation['coverage_complete']:raise ValueError('PINN validation incomplete')
    paths=list((h/'artifacts').rglob('metrics.json'));lookup={json.loads(p.read_text())['method']:p for p in paths}
    if set(lookup)!=set(['pinn','adjoint_nn']):raise ValueError('Both arm outputs required')
    models={};checks=[];hashes=[];gammas=[]
    for method,path in lookup.items():
        m=json.loads(path.read_text());entry=next(e for e in json.loads((root/'manifest.json').read_text()) if e['key']==m['key']);ds=Dataset(root,entry)
        saved=torch.load(path.with_name('model.pt'),map_location='cpu',weights_only=True);coef=NeuralDiffusivity.from_checkpoint(saved['coefficient']);D=coef.diffusivity(grid_coordinates(ds.N)).detach().numpy().reshape(ds.N,ds.N);z=np.load(path.with_name('D_hat.npz'))['D_hat'];error=float(np.max(abs(D-z)))
        if error>1e-12 or not np.isfinite(D).all() or D.min()<.001-1e-12 or D.max()>10+1e-12:raise ValueError('Invalid bounded coefficient/checkpoint')
        solver=AdjointSolver(ds,n_steps=m['n_steps']);replay=float(np.sqrt(np.mean((solver.predict(solver.forward(D.ravel())[0])-ds.u)**2))/ds.sigma)
        if abs(replay-m['rms_replay_over_sigma'])>1e-9:raise ValueError('Replay mismatch')
        checks.append(dict(method=method,checkpoint_difference=error,replay_sigma=replay,replay_metric_difference=replay-m['rms_replay_over_sigma']));hashes.append(m['initial_coefficient_sha256']);gammas.append(m['gamma']);models[method]=(m,D,entry)
    if len(set(hashes))!=1 or not np.allclose(gammas,gammas[0],rtol=0,atol=1e-20):raise ValueError('Initial coefficient/objective mismatch')
    (h/'pair_integrity.json').write_text(json.dumps(dict(results=checks,initial_coefficient_identical=True,gamma_identical=True,coefficient_truth_used=False),indent=2))
    # Only now access the synthetic answer key, with all cases/arms retained.
    rows=[]
    for method,(m,D,entry) in models.items():
        with np.load(root/entry['files']['ground_truth']) as z:truth=z['D_inv'].copy()
        rows.append(dict(method=method,key=m['key'],rel_L2_D=float(np.linalg.norm(D-truth)/np.linalg.norm(truth)),replay_sigma=m['rms_replay_over_sigma'],sensor_fit_sigma=m['rms_unet_over_sigma'],status=m['status'],wall_s=m['wall_s'],adam_completed=m['adam_completed'],quasi_newton_iterations=sum(r['iterations'] for r in m['optimizer_diagnostics']),function_evaluations=m['function_evaluations'],gamma=m['gamma'],parameter_count=m['parameter_count']))
    pd.DataFrame(rows).to_csv(h/'primary_matched_neural_comparison.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
