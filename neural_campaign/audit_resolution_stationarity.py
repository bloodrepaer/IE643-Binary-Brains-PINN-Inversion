"""Read-only v15 objective-gradient and state/CN diagnosis; no answer keys or fits."""
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from adjoint_inversion import Dataset, AdjointSolver
from adjoint_nn import grid_coordinates
from compare_neural import TemperatureNetwork, temperature_value, residual, field_regularizer
from neural_diffusivity import NeuralDiffusivity


def main():
    torch.set_num_threads(1)
    home=Path('campaign/pilots/dense_resolution_v15_gpu')
    root=Path('datasets_v2_density_extended')
    entries={e['key']:e for e in json.loads((root/'manifest.json').read_text())}
    rows=[]
    for path in sorted((home/'artifacts').rglob('model.pt')):
        m=json.loads(path.with_name('metrics.json').read_text())
        if m['method']!='pinn':continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        saved=torch.load(path,map_location='cpu',weights_only=True)
        ds=Dataset(root,entries[m['key']]);solver=AdjointSolver(ds,n_steps=m['n_steps'])
        coef=NeuralDiffusivity.from_checkpoint(saved['coefficient'])
        temp=TemperatureNetwork(**saved['temperature_config'])
        temp.load_state_dict(saved['temperature_state_dict'])
        state_params=list(temp.parameters());params=state_params+list(coef.parameters())
        split=sum(p.numel() for p in state_params)
        xd=torch.tensor(np.c_[ds.x,ds.y,ds.t/ds.T],dtype=torch.float64)
        xy=grid_coordinates(ds.N)
        obs=temperature_value(temp,coef,xd)
        q=saved['final_training_points'].clone().requires_grad_(True)
        parts=dict(data=((obs-torch.tensor(ds.u))/m['scale']).square().mean(),
                   pde=(residual(temp,coef,q,ds.T)/m['scale']).square().mean(),
                   reg=m['gamma']*field_regularizer(coef(xy).ravel(),solver))
        gradients={}
        for name,value in parts.items():
            gs=torch.autograd.grad(value,params,allow_unused=True)
            gradients[name]=torch.cat([(g if g is not None else torch.zeros_like(p)).ravel()
                                       for g,p in zip(gs,params)]).numpy()
        reproduced={k:float(v.detach()) for k,v in parts.items()}
        differences={k:abs(reproduced[k]-m['final_parts'][k]) for k in parts}
        if max(differences.values())>1e-12:raise ValueError('Stored objective mismatch')
        total=sum(gradients.values())
        def stats(v):
            return dict(l2=float(np.linalg.norm(v)),linf=float(np.max(abs(v))),
                        state_l2=float(np.linalg.norm(v[:split])),
                        coefficient_l2=float(np.linalg.norm(v[split:])))
        a,b=gradients['data'],gradients['pde']
        with torch.no_grad():D=coef.diffusivity(xy).numpy().ravel()
        replay=solver.predict(solver.forward(D)[0])
        state_gap=float(np.sqrt(np.mean((obs.detach().numpy()-replay)**2))/ds.sigma)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Checkpoint changed')
        row=dict(arm=m['arm'],n_pde=m['n_pde'],key=m['key'],n_sensors=m['n_sensors'],checkpoint_sha256=digest,
                 checkpoint_unchanged=True,objective_parts=reproduced,
                 stored_part_absolute_differences=differences,
                 gradients={k:stats(v) for k,v in gradients.items()},
                 total_gradient=stats(total),
                 data_pde_gradient_cosine=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),
                 state_common_CN_sensor_gap_sigma=state_gap)
        rows.append(row);print(json.dumps(row),flush=True)
    (home/'stationarity_diagnostic.json').write_text(json.dumps(dict(
        uses_true_D=False,refitting=False,checkpoint_selection=False,
        scope='Both retained dense PINNs; normalized original objective on final training points.',
        caveat='Raw Euclidean gradients depend on parameterization and objective scale. '
               'They do not prove convergence, a unique failure cause or an optimal penalty.',
        results=rows),indent=2))


if __name__=='__main__':main()
