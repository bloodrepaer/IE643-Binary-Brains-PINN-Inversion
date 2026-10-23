"""Paired fixed-penalty and multiplier continuations; no coefficient truth.

Both arms start from exactly the retained v6 checkpoint and its fixed points.
Only the augmented arm updates PDE multipliers between L-BFGS rounds.
Bounded approximate inner solves are diagnostic, not convergence guarantees.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import gc
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import time
import zipfile
import numpy as np
import torch


def main():
    source=Path(os.environ['INPUT_DIR']).resolve() if 'INPUT_DIR' in os.environ else next(Path('/kaggle/input').rglob('reference.pt')).parent
    sys.path.insert(0,str(source))
    from compare_neural import TemperatureNetwork,temperature_value,residual,field_regularizer
    from neural_diffusivity import NeuralDiffusivity
    from adjoint_nn import grid_coordinates
    from adjoint_inversion import Dataset,AdjointSolver
    device=torch.device(os.environ.get('DEVICE','cuda'))
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    out=Path(os.environ.get('WORK','/kaggle/working/dual_diagnostic')).resolve();out.mkdir(parents=True,exist_ok=True)
    data=source/'data'
    if not (data/'manifest.json').exists():
        dest=(out.parent/'diagnostic_inputs').resolve();dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(source/'data.zip') as z:
            for item in z.infolist():
                if not (dest/item.filename).resolve().is_relative_to(dest):raise ValueError('Unsafe input archive')
            z.extractall(dest)
        hits=list(dest.rglob('manifest.json'))
        if len(hits)!=1:raise ValueError('Expected one manifest')
        data=hits[0].parent
    protocol=json.loads((source/'protocol.json').read_text())
    entry=json.loads((data/'manifest.json').read_text())[0];ds=Dataset(data,entry)
    reference=torch.load(source/'reference.pt',map_location='cpu',weights_only=True)
    if reference['key']!=ds.key:raise ValueError('Wrong checkpoint')
    n=len(ds.u);scale=max(float(np.max(np.abs(ds.u0))),float(np.max(np.abs(ds.u))),1e-12)
    tensor=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
    xd=tensor(np.c_[ds.x,ds.y,ds.t/ds.T]);yd=tensor(ds.u)
    xy=grid_coordinates(ds.N).to(device);xi=torch.cat([xy,torch.zeros((len(xy),1),dtype=torch.float64,device=device)],dim=1);yi=tensor(ds.u0.ravel())
    points={k:v.to(device) for k,v in reference['final_training_points'].items()}
    solver=AdjointSolver(ds,n_steps=200)
    rho=1000.;rounds=int(os.environ.get('ROUNDS','3'));iterations=int(os.environ.get('LBFGS','750'))
    started=time.monotonic();outcomes=[]
    def budget():
        if time.monotonic()-started>float(os.environ.get('BUDGET_SECONDS','6500')):raise TimeoutError('Paired diagnostic wall budget')
    def pack():
        (out/'job_status.json').write_text(json.dumps(outcomes,indent=2)+'\n')
        with zipfile.ZipFile(out.parent/'pinn_dual_results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in out.rglob('*'):
                if p.is_file():z.write(p,p.relative_to(out))
    for arm in ['control','augmented']:
        budget();arm_start=time.monotonic();folder=out/arm;folder.mkdir(exist_ok=True)
        coefficient=NeuralDiffusivity.from_checkpoint(reference['coefficient']).to(device)
        temperature=TemperatureNetwork(**reference['temperature_config']).to(device);temperature.load_state_dict(reference['temperature_state_dict'])
        parameters=list(temperature.parameters())+list(coefficient.parameters())
        multipliers=torch.zeros(len(points['collocation']),dtype=torch.float64,device=device)
        value=lambda x:temperature_value(temperature,coefficient,x)
        def losses():
            r=residual(temperature,coefficient,points['collocation'].detach().requires_grad_(True),ds.T)/scale
            indices=points['data_indices']
            fit=.5*n*((value(xd[indices])-yd[indices])/ds.sigma).square().mean()
            pde=.5*n*r.square().mean()
            dual=n*(multipliers*r).mean()
            reg=field_regularizer(coefficient(xy).ravel(),solver)
            conditions=.5*n*(((value(xi)-yi)/scale).square().mean()+((value(points['boundary'])-points['boundary_target'])/scale).square().mean())
            objective=fit+rho*pde+dual+reg+1000*conditions
            return objective,r,dict(data=float(fit.detach()),pde=float(pde.detach()),dual=float(dual.detach()),reg=float(reg.detach()),conditions=float(conditions.detach()))
        initial,_,initial_parts=losses()
        initial_grad=torch.autograd.grad(initial,parameters)
        initial_max=max(float(g.detach().abs().max()) for g in initial_grad)
        if abs(float(initial.detach())-protocol['initial_total_loss'])>1e-7:raise ValueError('Initial objective differs from retained v6')
        if abs(initial_max-protocol['initial_gradient_max_abs'])>1e-5:raise ValueError('Initial gradient differs from retained v6')
        diagnostics=[]
        for round_index in range(rounds):
            budget()
            opt=torch.optim.LBFGS(parameters,max_iter=iterations,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-12)
            def closure():
                budget();opt.zero_grad();objective,_,_=losses();objective.backward();return objective
            opt.step(closure);objective=closure();_,r,parts=losses()
            state=opt.state[parameters[0]]
            info=dict(round=round_index+1,iterations=int(state.get('n_iter',0)),function_evaluations=int(state.get('func_evals',0)),
                      gradient_max_abs=max(float(p.grad.detach().abs().max()) for p in parameters if p.grad is not None),
                      pde_rms_normalized=float(r.detach().square().mean().sqrt()),dual_norm=float(multipliers.norm()),total=float(objective.detach()),**parts)
            diagnostics.append(info);print(arm,info,flush=True)
            if arm=='augmented' and round_index<rounds-1:
                multipliers=(multipliers+rho*r.detach()).detach()
                if not torch.isfinite(multipliers).all():raise ValueError('Nonfinite multipliers')
        with torch.no_grad():
            D=coefficient.diffusivity(xy).cpu().numpy().ravel();state_fit=value(xd).cpu().numpy()
        W=solver.forward(D)[0];prediction=solver.predict(W)
        metrics=dict(key=ds.key,variant=arm,method='pinn_dual_diagnostic',sigma=ds.sigma,n_steps=200,
                     D_const=protocol['D_const'],rms_const_over_sigma=protocol['rms_const_over_sigma'],
                     rms_replay_over_sigma=float(np.sqrt(np.mean((prediction-ds.u)**2))/ds.sigma),
                     rms_unet_over_sigma=float(np.sqrt(np.mean((state_fit-ds.u)**2))/ds.sigma),wall_s=time.monotonic()-arm_start,
                     uses_true_D=False,reference_sha256=hashlib.sha256((source/'reference.pt').read_bytes()).hexdigest(),
                     initial_total_loss=float(initial.detach()),initial_gradient_max_abs=initial_max,initial_parts=initial_parts,
                     additional_adam_steps=0,additional_lbfgs_budget=rounds*iterations,rounds=rounds,rho=rho,
                     optimizer_diagnostics=diagnostics,multiplier_updates=rounds-1 if arm=='augmented' else 0,
                     device=str(device),device_name=torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',
                     caveat='Warm continuation, fixed retained points, capped inner solves. Not an equal-budget fresh inverse comparison or a guarantee of augmented-Lagrangian convergence.')
        (folder/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        torch.save(dict(coefficient=coefficient.checkpoint(),temperature_config=reference['temperature_config'],temperature_state_dict=temperature.state_dict(),
                        T=ds.T,key=ds.key,method='pinn_dual_diagnostic',final_training_points={k:v.cpu() for k,v in points.items()},pde_multipliers=multipliers.cpu()),folder/'model.pt')
        np.savez_compressed(folder/'D_hat.npz',D_hat=D.reshape(ds.N,ds.N))
        outcomes.append(dict(arm=arm,status='complete'));pack()
        del opt,parameters,initial_grad,initial,objective,r,temperature,coefficient
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    print('Saved',out.parent/'pinn_dual_results.zip',flush=True)


if __name__=='__main__':main()
