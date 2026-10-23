"""Post-fit state/CN consistency at65 and129 nodes; no truth or refitting."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sympy as sp
import torch
from adjoint_inversion import Dataset,AdjointSolver
from adjoint_nn import grid_coordinates
from neural_diffusivity import NeuralDiffusivity
from compare_neural import TemperatureNetwork,temperature_value


def main():
    torch.set_num_threads(1)
    root=Path('datasets_v2_rk4_zero');entries={e['key']:e for e in json.loads((root/'manifest.json').read_text())};rows=[]
    for label in ['ic_locald_v6_gpu','fixed_d_forward_v7_gpu','fixed_d_refresh_v8_gpu']:
        home=Path('campaign/pilots')/label
        for model in (home/'artifacts').rglob('model.pt'):
            saved=torch.load(model,map_location='cpu',weights_only=True);entry=entries[saved['key']];ds=Dataset(root,entry)
            coefficient=NeuralDiffusivity.from_checkpoint(saved['coefficient'])
            temp=TemperatureNetwork(**saved['temperature_config']);temp.load_state_dict(saved['temperature_state_dict'])
            with torch.no_grad():state=temperature_value(temp,coefficient,torch.tensor(np.c_[ds.x,ds.y,ds.t/ds.T],dtype=torch.float64)).numpy()
            spec=json.loads((root/entry['files']['problem']).read_text());x,y=sp.symbols('x y')
            expr=sp.lambdify((x,y),sp.sympify(spec['initial_condition']['expression']),'numpy')
            for N in [65,129]:
                fine=copy.copy(ds);fine.N=N
                axis=np.linspace(0,1,N);X,Y=np.meshgrid(axis,axis,indexing='ij');fine.u0=expr(X,Y)
                old=fine.u0 if N==65 else fine.u0[::2,::2]
                if np.linalg.norm(old-ds.u0)/np.linalg.norm(ds.u0)>1e-6:raise ValueError('Supplied IC expression mismatch')
                with torch.no_grad():D=coefficient.diffusivity(grid_coordinates(N)).numpy().ravel()
                solver=AdjointSolver(fine,n_steps=200);prediction=solver.predict(solver.forward(D)[0])
                row=dict(pilot=label,grid_N=N,n_steps=200,
                    state_vs_CN_sigma=float(np.sqrt(np.mean((state-prediction)**2))/ds.sigma),
                    CN_vs_observations_sigma=float(np.sqrt(np.mean((prediction-ds.u)**2))/ds.sigma),
                    state_vs_observations_sigma=float(np.sqrt(np.mean((state-ds.u)**2))/ds.sigma))
                rows.append(row);print(row,flush=True)
    out=Path('campaign/diagnostics/pinn_state_grid_transfer.csv');pd.DataFrame(rows).to_csv(out,index=False)
    out.with_suffix('.json').write_text(json.dumps(dict(uses_true_D=False,uses_answer_key=False,training_or_selection=False,
        coefficient_deployment='Native retained coefficient function on each spatial grid',
        initial_condition='Supplied analytic expression on each grid; verifies original IC CSV',
        scope='Post-fit check that state mismatch is not solely the65-grid replay comparator; no inverse gate reopening or new estimate',results=rows),indent=2)+'\n')


if __name__=='__main__':main()
