"""Independent PDE concentration probe of the retained v3 pilot; no true D."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import campaign as C
from compare_neural import TemperatureNetwork
from neural_diffusivity import NeuralDiffusivity
from diagnose_frozen_state import fixed_derivatives, pde


def main():
    torch.set_num_threads(2)
    home=C.HOME/'pilots/ic_bc_v3_gpu'
    checkpoint=torch.load(next((home/'artifacts').rglob('model.pt')),map_location='cpu',weights_only=True)
    model=TemperatureNetwork(**checkpoint['temperature_config'])
    model.load_state_dict(checkpoint['temperature_state_dict'])
    for parameter in model.parameters():parameter.requires_grad_(False)
    coefficient=NeuralDiffusivity.from_checkpoint(checkpoint['coefficient'])
    values=fixed_derivatives(model,16384,20261022)
    residual=[]
    for start in range(0,16384,1024):
        residual.extend(pde(coefficient,tuple(v[start:start+1024] for v in values),checkpoint['T']).detach().numpy())
    residual=np.asarray(residual);points=values[0].numpy()
    energy=residual**2;order=np.argsort(energy)[::-1];concentration=[]
    for fraction in [.001,.01,.05,.1]:
        count=max(1,int(len(residual)*fraction))
        concentration.append(dict(top_fraction=fraction,energy_fraction=float(energy[order[:count]].sum()/energy.sum())))
    table=pd.DataFrame(dict(x=points[:,0],y=points[:,1],tau=points[:,2],residual=residual))
    table.iloc[order[:64]].to_csv(home/'residual_outliers.csv',index=False)
    result=dict(updated=C.now(),uses_true_D=False,seed=20261022,n_points=len(residual),
        sampling='Half log-time, half uniform-time; uniform space, matching training distribution',
        rms=float(np.sqrt(energy.mean())),max_abs=float(np.abs(residual).max()),concentration=concentration,
        largest_residual=table.iloc[order[0]].to_dict())
    C.atomic_json(home/'residual_concentration.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
