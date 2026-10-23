"""Separate fixed-coefficient forward PINN diagnostic; never an inverse fit.

Uses only a previously fitted adjoint coefficient, supplied IC/BC and PDE for
training. Sensor readings and finite-difference replay are evaluation only.
INPUT_DIR/WORK/DEVICE/STEPS/LBFGS allow the identical packaging CPU smoke.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
import torch


def main():
    source=Path(os.environ['INPUT_DIR']).resolve() if 'INPUT_DIR' in os.environ else next(Path('/kaggle/input').rglob('reference.pt')).parent
    sys.path.insert(0,str(source))
    from compare_neural import TemperatureNetwork,temperature_value,residual
    from neural_diffusivity import NeuralDiffusivity
    from adjoint_nn import grid_coordinates
    from adjoint_inversion import Dataset,AdjointSolver
    device=torch.device(os.environ.get('DEVICE','cuda'))
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    out=Path(os.environ.get('WORK','/kaggle/working/forward_diagnostic')).resolve()
    out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((source/'protocol.json').read_text())
    data_root=source/'data'
    if not (data_root/'manifest.json').exists():
        extracted=(out.parent/'diagnostic_inputs').resolve();extracted.mkdir(exist_ok=True)
        with zipfile.ZipFile(source/'data.zip') as archive:
            for item in archive.infolist():
                target=(extracted/item.filename).resolve()
                if not target.is_relative_to(extracted):raise ValueError('Unsafe input archive path')
            archive.extractall(extracted)
        hits=list(extracted.rglob('manifest.json'))
        if len(hits)!=1:raise ValueError('Expected one input manifest')
        data_root=hits[0].parent
    entry=json.loads((data_root/'manifest.json').read_text())[0]
    ds=Dataset(data_root,entry)
    saved=torch.load(source/'reference.pt',map_location='cpu',weights_only=True)
    if saved['key']!=ds.key or saved['method']!='adjoint_nn':raise ValueError('Unexpected reference')
    coefficient=NeuralDiffusivity.from_checkpoint(saved['coefficient']).to(device)
    for p in coefficient.parameters():p.requires_grad_(False)
    before={k:v.detach().clone() for k,v in coefficient.state_dict().items()}
    config=protocol['temperature_config']
    temperature=TemperatureNetwork(**config).to(device)
    value=lambda x:temperature_value(temperature,coefficient,x)
    params=list(temperature.parameters())
    tensor=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
    xy=grid_coordinates(ds.N).to(device)
    with torch.no_grad():
        D=coefficient.diffusivity(xy).cpu().numpy().ravel()
        initial=value(torch.cat([xy,torch.zeros((len(xy),1),dtype=torch.float64,device=device)],1)).cpu().numpy()
    ic_error=float(np.linalg.norm(initial-ds.u0.ravel())/np.linalg.norm(ds.u0))
    if ic_error>1e-6:raise ValueError('Supplied IC mismatch')
    # Preserve original PDE-loss scaling. There is deliberately no sensor,
    # replay, coefficient-regularization or true-D term in this objective.
    scale=max(float(np.max(np.abs(ds.u0))),1e-12)
    multiplier=.5*len(ds.u)*1000/scale**2
    rng=np.random.default_rng(42)
    def points(n=2048):
        x=rng.uniform(0,1,(n,3));x[:n//2,2]=10**rng.uniform(-4,0,n//2)
        return tensor(x)
    def loss(x):return multiplier*residual(temperature,coefficient,x.detach().requires_grad_(True),ds.T).square().mean()
    history=[];started=time.perf_counter()
    def check_budget():
        if time.perf_counter()-started>float(os.environ.get('BUDGET_SECONDS','6500')):raise TimeoutError('Diagnostic wall budget exhausted; incomplete run, not a convergence result')
    adam_steps=int(os.environ.get('STEPS','15000'));lbfgs_steps=int(os.environ.get('LBFGS','750'))
    opt=torch.optim.Adam(params,lr=1e-3)
    for step in range(1,adam_steps+1):
        check_budget()
        opt.zero_grad();objective=loss(points());objective.backward();opt.step()
        if step==1 or step%max(1,adam_steps//10)==0:
            history.append(dict(stage='adam',step=step,loss=float(objective.detach())))
            print(history[-1],flush=True)
    rounds=int(os.environ.get('LBFGS_ROUNDS','1'))
    if rounds<1 or (lbfgs_steps and rounds>lbfgs_steps):raise ValueError('Invalid number of LBFGS rounds')
    fixed=None;diagnostics={};optimizer_rounds=[]
    for round_index in range(rounds if lbfgs_steps else 0):
        fixed=points()
        iterations=lbfgs_steps//rounds+(round_index<lbfgs_steps%rounds)
        opt=torch.optim.LBFGS(params,max_iter=iterations,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-12)
        def closure():
            check_budget()
            opt.zero_grad();objective=loss(fixed);objective.backward();return objective
        opt.step(closure);objective=closure()
        state=opt.state[params[0]]
        diagnostics=dict(round=round_index+1,requested_iterations=iterations,iterations=int(state.get('n_iter',0)),function_evaluations=int(state.get('func_evals',0)),
                         gradient_max_abs=max(float(p.grad.abs().max()) for p in params if p.grad is not None),loss=float(objective.detach()))
        optimizer_rounds.append(diagnostics)
        print(dict(stage='lbfgs',**diagnostics),flush=True)
    if fixed is None:fixed=points()
    training_seconds=time.perf_counter()-started
    unchanged=all(torch.equal(before[k],v) for k,v in coefficient.state_dict().items())
    if not unchanged:raise AssertionError('Fixed coefficient changed')
    def rms(x):
        r=residual(temperature,coefficient,x.detach().requires_grad_(True),ds.T).detach()
        return float(r.square().mean().sqrt())
    rng=np.random.default_rng(20261024)
    independent=rng.uniform(0,1,(8192,3));independent[:4096,2]=10**rng.uniform(-4,0,4096)
    residuals=[]
    for offset in range(0,len(independent),256):
        residuals.append(residual(temperature,coefficient,tensor(independent[offset:offset+256]).requires_grad_(True),ds.T).detach().cpu().numpy())
    r=np.concatenate(residuals)
    solver=AdjointSolver(ds,n_steps=200)
    replay=solver.predict(solver.forward(D)[0])
    with torch.no_grad():
        fitted=value(tensor(np.c_[ds.x,ds.y,ds.t/ds.T])).cpu().numpy()
        wall=rng.uniform(0,1,(2048,3));edges=rng.integers(0,4,len(wall))
        for edge in range(4):wall[edges==edge,0 if edge<2 else 1]=edge%2
        boundary=value(tensor(wall)).cpu().numpy()
    metrics=dict(key=ds.key,experiment='fixed-adjoint-D forward state only',uses_true_D=False,
        training_uses_sensors=False,training_uses_CN=False,coefficient_unchanged=unchanged,
        reference_sha256=hashlib.sha256((source/'reference.pt').read_bytes()).hexdigest(),
        ic_relative_L2=ic_error,bc_rms=float(np.sqrt(np.mean(boundary**2))),
        sensor_fit_sigma=float(np.sqrt(np.mean((fitted-ds.u)**2))/ds.sigma),
        fixed_coefficient_CN_sensor_sigma=float(np.sqrt(np.mean((replay-ds.u)**2))/ds.sigma),
        state_vs_CN_sensor_sigma=float(np.sqrt(np.mean((fitted-replay)**2))/ds.sigma),
        training_pde_rms=rms(fixed),heldout_pde_rms=float(np.sqrt(np.mean(r**2))),heldout_pde_max=float(np.max(np.abs(r))),
        adam_steps=adam_steps,lbfgs_steps=lbfgs_steps,lbfgs_rounds=rounds,optimizer_diagnostics=diagnostics,optimizer_rounds=optimizer_rounds,history=history,
        training_seconds=training_seconds,device=str(device),device_name=torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    torch.save(dict(coefficient=coefficient.checkpoint(),temperature_config=config,temperature_state_dict=temperature.state_dict(),T=ds.T,
                    final_training_points=dict(collocation=fixed.detach().cpu()),key=ds.key,method='forward_diagnostic'),out/'model.pt')
    np.savez_compressed(out/'sensor_evaluation.npz',state=fitted,CN=replay,observed=ds.u)
    with zipfile.ZipFile(out.parent/'forward_diagnostic_results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.glob('*')):
            if p.is_file():archive.write(p,p.name)
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':main()
