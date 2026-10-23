"""Central-difference supplement to retained microprofiles, no optimizer/checkpoint selection or answer keys."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector,vector_to_parameters
from adjoint_inversion import Dataset,AdjointSolver
from adjoint_nn import grid_coordinates
from compare_neural import TemperatureNetwork,temperature_value,residual,field_regularizer
from neural_diffusivity import NeuralDiffusivity


def main():
    torch.set_num_threads(1)
    home=Path('campaign/pilots/dense_resolution_v15_gpu')
    path=home/'artifacts/results/pde16384/pinn/model.pt'
    output=home/'central_difference_audit';output.mkdir(exist_ok=False)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    steps=[-1e-8,-1e-7]
    protocol=dict(uses_true_D=False,training=False,checkpoint_selection=False,
        checkpoint_sha256=digest,steps=steps,directions=['joint'],
        measure='Saved final training cloud, original full-data/unit-PDE/gamma objective',
        caveat='All probes retained; no accepted step or fitted model. Local profiles do not prove long-run convergence or an optimal optimizer.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    saved=torch.load(path,map_location='cpu',weights_only=True)
    metrics=json.loads(path.with_name('metrics.json').read_text())
    root=Path('datasets_v2_density_extended')
    entry=next(e for e in json.loads((root/'manifest.json').read_text()) if e['key']==metrics['key'])
    ds=Dataset(root,entry);solver=AdjointSolver(ds,n_steps=metrics['n_steps'])
    coef=NeuralDiffusivity.from_checkpoint(saved['coefficient'])
    temp=TemperatureNetwork(**saved['temperature_config']);temp.load_state_dict(saved['temperature_state_dict'])
    ps=list(temp.parameters());split=sum(p.numel() for p in ps);ps+=list(coef.parameters())
    z=parameters_to_vector(ps).detach().numpy().copy();xy=grid_coordinates(ds.N)
    xd=torch.tensor(np.c_[ds.x,ds.y,ds.t/ds.T],dtype=torch.float64);y=torch.tensor(ds.u)
    def parts(weights):
        vector_to_parameters(torch.tensor(weights.copy(),dtype=torch.float64),ps)
        q=saved['final_training_points'].clone().requires_grad_(True)
        data=((temperature_value(temp,coef,xd)-y)/metrics['scale']).square().mean()
        pde=(residual(temp,coef,q,ds.T)/metrics['scale']).square().mean()
        reg=metrics['gamma']*field_regularizer(coef(xy).ravel(),solver)
        return dict(data=data,pde=pde,reg=reg)
    initial=parts(z);base=sum(initial.values())
    assert max(abs(float(initial[k].detach())-metrics['final_parts'][k]) for k in initial)<1e-12
    grad=torch.cat([g.ravel() for g in torch.autograd.grad(base,ps)]).numpy()
    baseline=float(base.detach());rows=[]
    for label in protocol['directions']:
        direction=-grad.copy()
        if label=='state_only':direction[split:]=0
        if label=='coefficient_only':direction[:split]=0
        norm=np.linalg.norm(direction);assert norm>0
        direction/=norm
        for step in steps:
            p=parts(z+step*direction);values={k:float(v.detach()) for k,v in p.items()}
            total=sum(values.values())
            row=dict(direction=label,step=step,objective=total,relative_objective_change=total/baseline-1,**values)
            rows.append(row);print(json.dumps(row),flush=True)
    vector_to_parameters(torch.tensor(z,dtype=torch.float64),ps)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    (output/'results.json').write_text(json.dumps(dict(protocol=protocol,baseline_objective=baseline,
        baseline_parts={k:float(v.detach()) for k,v in initial.items()},gradient_l2=float(np.linalg.norm(grad)),
        results=rows,checkpoint_unchanged=True,no_probe_selected=True),indent=2))


if __name__=='__main__':main()
