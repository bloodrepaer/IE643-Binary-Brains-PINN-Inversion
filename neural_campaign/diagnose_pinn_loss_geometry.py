"""Retained-checkpoint loss/gradient decomposition; no refitting or true D."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from compare_neural import TemperatureNetwork,temperature_value,residual,field_regularizer
from neural_diffusivity import NeuralDiffusivity
from adjoint_nn import grid_coordinates
from adjoint_inversion import Dataset,AdjointSolver


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot-dirs',nargs='+',required=True)
    args=ap.parse_args();torch.set_num_threads(1)
    root=Path('datasets_v2_rk4_zero')
    entries={e['key']:e for e in json.loads((root/'manifest.json').read_text())}
    all_rows=[]
    for name in args.pilot_dirs:
        home=Path(name)
        for checkpoint in sorted((home/'artifacts').rglob('model.pt')):
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            retained=saved.get('final_training_points')
            if not retained:raise ValueError('Final training points required')
            m=json.loads(checkpoint.with_name('metrics.json').read_text())
            ds=Dataset(root,entries[saved['key']]);solver=AdjointSolver(ds,n_steps=m['n_steps'])
            t=TemperatureNetwork(**saved['temperature_config']);t.load_state_dict(saved['temperature_state_dict'])
            c=NeuralDiffusivity.from_checkpoint(saved['coefficient'])
            parameters=list(t.parameters())+list(c.parameters());split=sum(p.numel() for p in t.parameters())
            value=lambda x:temperature_value(t,c,x)
            tensor=lambda x:torch.tensor(x,dtype=torch.float64)
            xd=tensor(np.c_[ds.x,ds.y,ds.t/ds.T]);yd=tensor(ds.u)
            xy=grid_coordinates(ds.N);xi=torch.cat([xy,torch.zeros((len(xy),1),dtype=torch.float64)],dim=1)
            scale=max(float(np.max(np.abs(ds.u0))),float(np.max(np.abs(ds.u))),1e-12)
            n=len(ds.u);indices=retained['data_indices'];col=retained['collocation'].detach().requires_grad_(True)
            raw=dict(data=.5*n*((value(xd[indices])-yd[indices])/ds.sigma).square().mean(),
                     pde=.5*n*(residual(t,c,col,ds.T)/scale).square().mean(),
                     ic=.5*n*((value(xi)-tensor(ds.u0.ravel()))/scale).square().mean(),
                     bc=.5*n*((value(retained['boundary'])-retained['boundary_target'])/scale).square().mean(),
                     reg=field_regularizer(c(xy).ravel(),solver))
            weights=dict(data=1.,pde=m['physics_weight'],ic=m['condition_weight'],bc=m['condition_weight'],reg=m['alpha'])
            vectors={}
            rows=[]
            for term,loss in raw.items():
                g=torch.autograd.grad(loss*weights[term],parameters,allow_unused=True)
                flat=torch.cat([(torch.zeros_like(p) if q is None else q).detach().ravel() for p,q in zip(parameters,g)])
                vectors[term]=flat
                rows.append(dict(pilot=home.name,key=ds.key,term=term,raw_loss=float(loss.detach()),weight=weights[term],weighted_loss=float(loss.detach())*weights[term],
                                 temperature_gradient_norm=float(flat[:split].norm()),coefficient_gradient_norm=float(flat[split:].norm()),gradient_norm=float(flat.norm())))
            total=sum(vectors.values());total_loss=sum(r['weighted_loss'] for r in rows)
            recorded=m['loss_history'][-1]
            if abs(total_loss-recorded['total'])>1e-7*max(1,abs(total_loss)):raise ValueError('Loss reconstruction differs from checkpoint')
            groups=dict(all=slice(None),temperature=slice(0,split),coefficient=slice(split,None))
            geometry=[]
            for group,selection in groups.items():
                data=vectors['data'][selection];pde=vectors['pde'][selection]
                denom=float(data.norm()*pde.norm())
                gradient_sum=total[selection]
                geometry.append(dict(group=group,data_pde_cosine=float(torch.dot(data,pde))/denom if denom else None,
                                     total_gradient_norm=float(gradient_sum.norm()),
                                     cancellation_ratio=float(gradient_sum.norm())/sum(float(v[selection].norm()) for v in vectors.values())))
            pd.DataFrame(rows).to_csv(home/'loss_gradient_components.csv',index=False);all_rows.extend(rows)
            result=dict(selection_uses_true_D=False,checkpoint_refitted=False,training_points_reused=True,
                        total_loss=total_loss,recorded_total_loss=recorded['total'],
                        total_gradient_max_abs=float(total.abs().max()),recorded_gradient_max_abs=recorded['gradient_max_abs'],
                        geometry=geometry,components=rows,
                        caveat='Gradient geometry depends on parameterization and checkpoint. Opposing component gradients are normal in a penalized objective; these values alone do not select a correct loss weight or prove the cause of inverse failure.')
            (home/'loss_geometry.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(pilot=home.name,total_loss=total_loss,total_gradient_max_abs=result['total_gradient_max_abs'],geometry=geometry)),flush=True)
    pd.DataFrame(all_rows).to_csv('campaign/diagnostics/pinn_loss_gradient_components.csv',index=False)


if __name__=='__main__':main()
