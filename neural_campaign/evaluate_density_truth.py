"""Post-fit discretization diagnostic on completed full-density grid cases.

True D is used only to evaluate the common CN solver, never to choose an
inverse coefficient, regularizer, model or training setting.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from adjoint_inversion import Dataset, AdjointSolver


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time-refinement',action='store_true',help='Also evaluate fixed A/D 1%% noise cases at 200/400/800 CN steps')
    args=parser.parse_args()
    root=Path('datasets_v2_density_extended')
    entries={e['key']:e for e in json.loads((root/'manifest.json').read_text())}
    density=pd.read_csv('campaign/density_results.csv')
    rows=[]
    for r in density[(density.n_sensors==3969)&(density.method=='grid')].itertuples():
        ds=Dataset(root,entries[r.key]);metrics=json.loads(Path(r.metrics).read_text())
        with np.load(root/entries[r.key]['files']['ground_truth']) as g:D=g['D_inv'].copy()
        with np.load(root/entries[r.key]['files']['sensors_npz']) as z:clean=z['u_clean'].copy()
        solver=AdjointSolver(ds,n_steps=metrics['n_steps'])
        pred=solver.predict(solver.forward(D.ravel())[0])
        rows.append(dict(key=r.key,tier=r.tier,noise=r.eta,n_sensors=3969,n_steps=metrics['n_steps'],
            true_D_CN_replay_sigma=float(np.sqrt(np.mean((pred-ds.u)**2))/ds.sigma),
            true_D_CN_vs_clean_sigma=float(np.sqrt(np.mean((pred-clean)**2))/ds.sigma),
            selected_grid_replay_sigma=r.replay,selected_grid_coefficient_error=r.rel_L2_D,
            selected_grid_flag=r.flag))
    out=Path('campaign/diagnostics/density_true_D_replay.csv');out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    if args.time_refinement:
        refinement=[]
        for tier in ['A_smooth','D_layered']:
            key=tier+'__Fo0.1__uniform8__n3969__nodes_nested__eta0.01'
            entry=entries[key];ds=Dataset(root,entry)
            with np.load(root/entry['files']['ground_truth']) as z:D=z['D_inv'].copy()
            with np.load(root/entry['files']['sensors_npz']) as z:clean=z['u_clean'].copy()
            for steps in [200,400,800]:
                solver=AdjointSolver(ds,n_steps=steps)
                pred=solver.predict(solver.forward(D.ravel())[0])
                refinement.append(dict(tier=tier,n_steps=steps,noise=.01,
                    true_D_CN_replay_sigma=float(np.sqrt(np.mean((pred-ds.u)**2))/ds.sigma),
                    true_D_CN_vs_clean_sigma=float(np.sqrt(np.mean((pred-clean)**2))/ds.sigma)))
        pd.DataFrame(refinement).to_csv(out.parent/'density_time_refinement.csv',index=False)
        print(pd.DataFrame(refinement).to_string(index=False))


if __name__=='__main__':main()
