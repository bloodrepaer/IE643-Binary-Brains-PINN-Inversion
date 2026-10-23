"""Post-fit A/D forward-grid diagnostic; no inverse-model selection.

The original 129-grid D and initial snapshot are used only to evaluate the
data-generation spatial operator against the retained 65-grid CN replay.
Original observations, trained estimates and campaign settings are unchanged.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import campaign as C
from adjoint_inversion import Dataset, AdjointSolver


def main():
    root=Path('datasets_v2_density_extended')
    entries={e['key']:e for e in json.loads((root/'manifest.json').read_text())}
    coarse=pd.read_csv('campaign/diagnostics/density_time_refinement.csv')
    rows=[];provenance=[]
    for tier in ['A_smooth','D_layered']:
        key=tier+'__Fo0.1__uniform8__n3969__nodes_nested__eta0.01'
        entry=entries[key];ds=Dataset(root,entry)
        with np.load(root/entry['files']['ground_truth']) as z:
            D=z['D_data'].copy();initial=z['u'][0].copy()
            if abs(float(z['t'][0]))>1e-12:raise ValueError('First full-field snapshot is not the initial field')
            provenance.append(dict(tier=tier,D_shared_node_max_abs_difference=float(np.abs(D[::2,::2]-z['D_inv']).max()),
                initial_csv_rounding_max_abs=float(np.abs(initial[::2,::2]-ds.u0).max())))
        with np.load(root/entry['files']['sensors_npz']) as z:clean=z['u_clean'].copy()
        fine=copy.copy(ds);fine.N=D.shape[0];fine.u0=initial
        for steps in [200,800]:
            old=coarse[(coarse.tier==tier)&(coarse.n_steps==steps)].iloc[0]
            rows.append(dict(tier=tier,grid_N=65,n_steps=steps,noise=.01,
                true_D_CN_replay_sigma=float(old.true_D_CN_replay_sigma),
                true_D_CN_vs_clean_sigma=float(old.true_D_CN_vs_clean_sigma)))
            solver=AdjointSolver(fine,n_steps=steps)
            prediction=solver.predict(solver.forward(D.ravel())[0])
            row=dict(tier=tier,grid_N=fine.N,n_steps=steps,noise=.01,
                true_D_CN_replay_sigma=float(np.sqrt(np.mean((prediction-fine.u)**2))/fine.sigma),
                true_D_CN_vs_clean_sigma=float(np.sqrt(np.mean((prediction-clean)**2))/fine.sigma))
            rows.append(row);print(row,flush=True)
    dest=Path('campaign/diagnostics/density_spatial_resolution.csv')
    pd.DataFrame(rows).to_csv(dest,index=False)
    C.atomic_json(dest.with_suffix('.json'),dict(updated=C.now(),uses_true_D=True,
        scope='Post-fit forward-grid evaluation only; no inverse or training setting selected',results=rows,provenance=provenance))
    print('Saved',dest,flush=True)


if __name__=='__main__':main()
