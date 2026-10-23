"""Post-fit checks of PINN consistency with its known IC, BC and PDE.

No ground-truth coefficient is read. This diagnoses convergence before scaling
up a soft-constrained PINN; it does not tune against coefficient error.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER","SEQUENTIAL")
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import campaign as C
from adjoint_inversion import Dataset
from adjoint_nn import grid_coordinates
from compare_neural import TemperatureNetwork, residual
from neural_diffusivity import NeuralDiffusivity


def main():
    torch.set_num_threads(2)
    jobs=json.loads((C.HOME/"jobs.json").read_text())
    manifests={}
    rows=[]
    for job in jobs:
        if job["method"]!="pinn" or not C.completed(job):continue
        if job["collection"] not in manifests:
            manifests[job["collection"]]={e["key"]:e for e in json.loads((C.ROOT/job["collection"]/"manifest.json").read_text())}
        ds=Dataset(C.ROOT/job["collection"],manifests[job["collection"]][job["key"]])
        checkpoint=torch.load(C.metric_path(job).parent/"model.pt",map_location="cpu",weights_only=True)
        model=TemperatureNetwork(**checkpoint["temperature_config"])
        model.load_state_dict(checkpoint["temperature_state_dict"])
        coefficient=NeuralDiffusivity.from_checkpoint(checkpoint["coefficient"])
        xy=grid_coordinates(ds.N)
        initial=torch.cat([xy,torch.zeros((len(xy),1),dtype=torch.float64)],1)
        with torch.no_grad():pred=model(initial).numpy().reshape(ds.N,ds.N)
        error=pred-ds.u0
        inner=(slice(1,-1),slice(1,-1))
        row=dict(collection=job["collection"],key=job["key"],physics_mode=job["physics_mode"],
                 ic_relative_L2=float(np.linalg.norm(error)/np.linalg.norm(ds.u0)),
                 ic_relative_L2_interior=float(np.linalg.norm(error[inner])/np.linalg.norm(ds.u0[inner])),
                 ic_rms=float(np.sqrt(np.mean(error**2))),ic_max_absolute_error=float(np.abs(error).max()),
                 ic_true_peak=float(ds.u0.max()),ic_predicted_peak=float(pred.max()))
        rng=np.random.default_rng(20261019)
        points=torch.tensor(rng.uniform(0,1,(1024,3)),dtype=torch.float64,requires_grad=True)
        r=residual(model,coefficient,points,ds.T)
        row["pde_validation_rms_dimensionless"]=float(r.detach().square().mean().sqrt())
        edges=rng.integers(0,4,1024);wall=rng.uniform(0,1,(1024,3));target=np.zeros(1024)
        for i,name in enumerate(("left","right","bottom","top")):
            mask=edges==i;wall[mask,0 if i<2 else 1]=i%2
            target[mask]=[ds.bc.at(t*ds.T)[name] for t in wall[mask,2]]
        with torch.no_grad():boundary=model(torch.tensor(wall,dtype=torch.float64)).numpy()
        row["bc_validation_rms"]=float(np.sqrt(np.mean((boundary-target)**2)))
        rows.append(row)
    dest=C.HOME/"diagnostics";dest.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/"pinn_constraints.csv",index=False)
    C.atomic_json(dest/"pinn_constraints.json",dict(updated=C.now(),validation_seed=20261019,
                  source="Known problem IC/BC and PDE only; no coefficient answer key",results=rows))
    for row in rows:print(row["key"],"IC rel L2",round(row["ic_relative_L2"],4),"IC peak true/pred",round(row["ic_true_peak"],4),round(row["ic_predicted_peak"],4),flush=True)


if __name__=="__main__":main()
