"""Known-physics optimization diagnostic; this is NOT a campaign PINN fit.

Freeze the completed v3 temperature network and minimize the same PDE plus
coefficient regularization terms in its D network alone. Independent residual
and common CN replay determine whether this block update is useful. No true D
is read, including for reporting or selecting this diagnostic.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import json
from pathlib import Path
import numpy as np
import torch
import campaign as C
from adjoint_inversion import Dataset, AdjointSolver
from adjoint_nn import grid_coordinates
from compare_neural import TemperatureNetwork, derivative, field_regularizer
from neural_diffusivity import NeuralDiffusivity


def fixed_derivatives(model,n,seed):
    rng=np.random.default_rng(seed)
    points=rng.uniform(0,1,(n,3));points[:n//2,2]=10**rng.uniform(-4,0,n//2)
    x=torch.tensor(points,dtype=torch.float64,requires_grad=True)
    u=model(x);du=derivative(u,x)
    lap=derivative(du[:,0],x)[:,0]+derivative(du[:,1],x)[:,1]
    return x.detach(),du.detach(),lap.detach()


def pde(coefficient,values,T):
    x,du,lap=values
    xy=x[:,:2].detach().requires_grad_(True)
    D=coefficient.diffusivity(xy).ravel();dD=derivative(D,xy)
    return du[:,2]-T*(D*lap+dD[:,0]*du[:,0]+dD[:,1]*du[:,1])


def main():
    torch.set_num_threads(2)
    home=C.HOME/'pilots/ic_bc_v3_gpu'
    path=next((home/'artifacts').rglob('model.pt'))
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=TemperatureNetwork(**checkpoint['temperature_config']);model.load_state_dict(checkpoint['temperature_state_dict'])
    for p in model.parameters():p.requires_grad_(False)
    coefficient=NeuralDiffusivity.from_checkpoint(checkpoint['coefficient'])
    entries=json.loads((C.ROOT/'datasets_v2_rk4_zero/manifest.json').read_text())
    ds=Dataset(C.ROOT/'datasets_v2_rk4_zero',next(e for e in entries if e['key']==checkpoint['key']))
    solver=AdjointSolver(ds,n_steps=200);xy=grid_coordinates(ds.N)
    train=fixed_derivatives(model,2048,20261020);validation=fixed_derivatives(model,4096,20261021)
    scale=max(float(np.abs(ds.u0).max()),float(np.abs(ds.u).max()))
    def objective():
        r=pde(coefficient,train,ds.T)
        return 1000*.5*len(ds.u)*(r/scale).square().mean()+field_regularizer(coefficient(xy).ravel(),solver)
    def assess():
        r=pde(coefficient,validation,ds.T).detach().numpy()
        with torch.no_grad():D=coefficient.diffusivity(xy).numpy().ravel()
        pred=solver.predict(solver.forward(D)[0])
        return dict(training_objective=float(objective().detach()),validation_pde_rms=float(np.sqrt(np.mean(r*r))),
                    validation_pde_max_abs=float(np.abs(r).max()),replay_sigma=float(np.sqrt(np.mean((pred-ds.u)**2))/ds.sigma),
                    D_mean=float(D.mean()),D_spatial_std=float(D.std()))
    before=assess();history=[]
    optimizer=torch.optim.LBFGS(coefficient.parameters(),max_iter=500,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-12)
    def closure():
        optimizer.zero_grad();loss=objective();loss.backward()
        if len(history)%50==0:print('evaluation',len(history),'objective',float(loss.detach()),flush=True)
        history.append(float(loss.detach()));return loss
    optimizer.step(closure);after=assess()
    dest=home/'frozen_state_diagnostic';dest.mkdir(exist_ok=True)
    with torch.no_grad():D=coefficient.diffusivity(xy).numpy().reshape(ds.N,ds.N)
    np.savez_compressed(dest/'D_hat.npz',D_hat=D)
    torch.save(coefficient.checkpoint(),dest/'coefficient.pt')
    C.atomic_json(dest/'diagnostic.json',dict(updated=C.now(),source_checkpoint=str(path.relative_to(C.ROOT)),
        uses_true_D=False,training_seed=20261020,validation_seed=20261021,training_points=2048,validation_points=4096,
        coefficient_only_lbfgs_budget=500,before=before,after=after,objective_history=history,
        scope='Frozen temperature block optimization diagnostic only; not a full retraining or a campaign completion.'))
    print(json.dumps(dict(before=before,after=after),indent=2))


if __name__=='__main__':main()
