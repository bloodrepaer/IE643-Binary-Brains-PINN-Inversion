"""Post-fit replay of retained layered estimates on a finer grid; no true D.

Common comparison: bilinearly interpolate logD from65 to129 for BOTH methods.
Also evaluate the neural function natively on129 as a separate deployment.
No estimate/setting is selected or modified using these additional replays.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sympy as sp
import torch
from scipy.interpolate import RegularGridInterpolator
from adjoint_inversion import Dataset,AdjointSolver
from adjoint_nn import grid_coordinates
from neural_diffusivity import NeuralDiffusivity


def main():
    torch.set_num_threads(1)
    root=Path('datasets_v2_density_extended');entries={e['key']:e for e in json.loads((root/'manifest.json').read_text())}
    jobs=json.loads(Path('campaign/jobs.json').read_text());rows=[];sources=[]
    axis=np.linspace(0,1,129);X,Y=np.meshgrid(axis,axis,indexing='ij');coords=np.c_[X.ravel(),Y.ravel()]
    for eta in [.01,.05,.1]:
        key=f'D_layered__Fo0.1__uniform8__n3969__nodes_nested__eta{eta:g}';entry=entries[key];ds=Dataset(root,entry)
        spec=json.loads((root/entry['files']['problem']).read_text());x,y=sp.symbols('x y')
        initial=sp.lambdify((x,y),sp.sympify(spec['initial_condition']['expression']),'numpy')(X,Y)
        if np.linalg.norm(initial[::2,::2]-ds.u0)/np.linalg.norm(ds.u0)>1e-6:raise ValueError('Analytic initial field disagrees with supplied CSV')
        fine=copy.copy(ds);fine.N=129;fine.u0=initial
        solver=AdjointSolver(fine,n_steps=200)
        for method in ['grid','adjoint_nn']:
            j=next(j for j in jobs if j['collection']==root.name and j['key']==key and j['method']==method)
            folder=Path(j['output'])/key/(method+'__s42')
            metrics=json.loads((folder/'metrics.json').read_text())
            with np.load(folder/'D_hat.npz') as z:D=z['D_hat'].copy()
            sources.append(dict(key=key,method=method,D_hat_sha256=hashlib.sha256((folder/'D_hat.npz').read_bytes()).hexdigest()))
            rows.append(dict(key=key,eta=eta,method=method,grid_N=65,deployment='retained',replay_sigma=metrics['rms_replay_over_sigma'],shared_node_max_abs=0.))
            coarse_axis=np.linspace(0,1,65)
            prolonged=np.exp(RegularGridInterpolator((coarse_axis,coarse_axis),np.log(D))(coords)).reshape(129,129)
            candidates=[('common_logD_prolongation',prolonged)]
            if method=='adjoint_nn':
                saved=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
                coefficient=NeuralDiffusivity.from_checkpoint(saved['coefficient'])
                with torch.no_grad():native=coefficient.diffusivity(grid_coordinates(129)).numpy().reshape(129,129)
                candidates.append(('native_neural_function',native))
            for label,field in candidates:
                shared=float(np.max(np.abs(field[::2,::2]-D)))
                if shared>1e-10:raise ValueError('Transferred function changes original65nodes')
                predicted=solver.predict(solver.forward(field.ravel())[0])
                row=dict(key=key,eta=eta,method=method,grid_N=129,deployment=label,
                         replay_sigma=float(np.sqrt(np.mean((predicted-ds.u)**2))/ds.sigma),shared_node_max_abs=shared)
                rows.append(row);print(row,flush=True)
    out=Path('campaign/diagnostics/learned_layered_grid_transfer.csv');pd.DataFrame(rows).to_csv(out,index=False)
    out.with_suffix('.json').write_text(json.dumps(dict(uses_true_D=False,uses_answer_key=False,training_or_selection=False,
        initial_condition='Analytic supplied expression, matching supplied65-grid CSV to1e-6relative',n_steps=200,
        common_transfer='Bilinear interpolation of logD from65-grid for both methods',
        caveat='A post-fit resolution/deployment stress test, not a refit or replacement of common65-grid campaign replay. Native neural evaluation is an additional deployment, not the common transfer.',
        sources=sources,results=rows),indent=2)+'\n')


if __name__=='__main__':main()
