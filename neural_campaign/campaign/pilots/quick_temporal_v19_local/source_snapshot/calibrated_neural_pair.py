"""v12: matched neural inverse objectives with supplied hard IC/BC.

Fixed alpha1 is converted algebraically to the mean-data objective, not tuned
against coefficient truth. PDE penalty is unit weight in dimensionless u,tau.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector,vector_to_parameters
from adjoint_inversion import Dataset,AdjointSolver
from adjoint_nn import grid_coordinates
from neural_diffusivity import NeuralDiffusivity
from compare_neural import TemperatureNetwork,temperature_value,residual,field_regularizer,gaussian_initial_components
from pinn_zhang_transfer import quasi_newton


class PairObjective:
    def __init__(self,ds,method,device='cpu',n_pde=4096,seed=42):
        self.ds=ds;self.method=method;self.device=torch.device(device);self.solver=AdjointSolver(ds);self.rng=np.random.default_rng(seed);self.n_pde=n_pde
        mc=self.solver.best_constant();self.D_const=float(np.exp(mc).mean());self.const_rms=float(self.solver._last['rms'])
        self.coef=NeuralDiffusivity(initial_D=self.D_const,seed=seed)
        self.initial_weights_sha256=hashlib.sha256(parameters_to_vector(self.coef.parameters()).detach().numpy().tobytes()).hexdigest();self.coef.to(self.device)
        self.scale=max(float(np.max(abs(ds.u0))),float(np.max(abs(ds.u))),1e-12);self.factor=2*self.solver.sigma_w**2/(len(ds.u)*self.scale**2)
        self.gamma=self.factor # fixed alpha=1, identical discrete regularizer
        spec=json.loads((ds.root/ds.entry['files']['problem']).read_text())
        if str(spec['source_f'])!='0' or any(float(v)!=0 for v in spec['boundary_condition']['edges'].values()):raise ValueError('Only matched zero-source/zero-BC cases')
        gaussians=gaussian_initial_components(spec['initial_condition']['expression'])
        self.temp_config=dict(width=32,depth=3,seed=seed,initial_gaussians=gaussians,T=ds.T,D_base=self.D_const,boundary_model='zero_images',local_coefficient=True,image_D_max=self.coef.config.d_max)
        self.temp=TemperatureNetwork(**self.temp_config).to(self.device)
        self.params=(list(self.temp.parameters()) if method=='pinn' else [])+list(self.coef.parameters())
        self.xy=grid_coordinates(ds.N).to(self.device);self.xd=self.tensor(np.c_[ds.x,ds.y,ds.t/ds.T]);self.yd=self.tensor(ds.u);self.parts={};self.batch()
        with torch.no_grad():
            xi=torch.cat([self.xy,torch.zeros((len(self.xy),1),dtype=torch.float64,device=self.device)],1)
            ic=temperature_value(self.temp,self.coef,xi).cpu().numpy();self.ic_initial_relative_error=float(np.linalg.norm(ic-ds.u0.ravel())/np.linalg.norm(ds.u0))
        if self.ic_initial_relative_error>1e-6:raise ValueError('Supplied IC mismatch')
    def tensor(self,x):return torch.tensor(x,dtype=torch.float64,device=self.device)
    def batch(self):
        col=self.rng.uniform(0,1,(self.n_pde,3));col[:len(col)//2,2]=10**self.rng.uniform(-4,0,len(col)//2);self.col=self.tensor(col)
    def weights(self):return parameters_to_vector(self.params).detach().cpu().numpy().copy()
    def fg(self,z):
        vector_to_parameters(self.tensor(z.copy()),self.params);m=self.coef(self.xy).ravel();R=field_regularizer(m,self.solver)
        if self.method=='pinn':
            q=self.col.clone().requires_grad_(True);data=((temperature_value(self.temp,self.coef,self.xd)-self.yd)/self.scale).square().mean();pde=(residual(self.temp,self.coef,q,self.ds.T)/self.scale).square().mean();loss=data+pde+self.gamma*R
            gs=torch.autograd.grad(loss,self.params);value=float(loss.detach());self.parts=dict(data=float(data.detach()),pde=float(pde.detach()),reg=float((self.gamma*R).detach()))
        else:
            J,gm=self.solver.field_objective(m.detach().cpu().numpy(),0.);surrogate=(m*self.tensor(gm*self.factor)).sum()+self.gamma*R
            gs=torch.autograd.grad(surrogate,self.params);value=J*self.factor+float((self.gamma*R).detach());self.parts=dict(data=float(J*self.factor),pde=0.,reg=float((self.gamma*R).detach()))
        g=torch.cat([v.ravel() for v in gs]).detach().cpu().numpy();return value,g


def train(a):
    torch.set_num_threads(2);root=Path(a.data);entry=next(e for e in json.loads((root/'manifest.json').read_text()) if e['key']==a.key);ds=Dataset(root,entry)
    dest=Path(a.out)/a.method;dest.mkdir(parents=True,exist_ok=False);started=time.monotonic();obj=PairObjective(ds,a.method,a.device,a.n_pde);initial=obj.weights().copy();accepted=initial.copy();evals=0;adam_done=0;rounds=[];history=[];status='complete'
    def fg(z):
        nonlocal evals
        if time.monotonic()-started>a.budget:raise TimeoutError('Per-arm bounded budget')
        evals+=1;return obj.fg(z)
    opt=torch.optim.Adam(obj.params,lr=1e-3);scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=max(1,a.adam//4),gamma=.5)
    try:
        for k in range(a.adam):
            if k%100==0:obj.batch()
            value,g=fg(obj.weights());opt.zero_grad();offset=0
            for p in obj.params:p.grad=obj.tensor(g[offset:offset+p.numel()].reshape(p.shape));offset+=p.numel()
            opt.step();scheduler.step();accepted=obj.weights();adam_done=k+1
            if k%100==0:print('Adam',k,value,obj.parts,flush=True)
        x=accepted.copy();H=np.eye(len(x))
        for o in range(a.outer):
            obj.batch();before=evals;x,H,record=quasi_newton(fg,x,H,a.inner);accepted=x.copy();H=.5*(H+H.T)
            try:np.linalg.cholesky(H);record['hessian_retained']=True
            except np.linalg.LinAlgError:H=np.eye(len(x));record['hessian_retained']=False
            value,_=obj.fg(accepted);record.update(round=o+1,function_evaluations=evals-before);rounds.append(record);history.append(obj.parts.copy());print('SSBroyden',record,obj.parts,flush=True)
    except TimeoutError:status='budget_timeout'
    # Restore last completed Adam step or outer solve, never a line-search trial.
    value,_=obj.fg(accepted)
    with torch.no_grad():
        D=obj.coef.diffusivity(obj.xy).cpu().numpy().ravel();fit=temperature_value(obj.temp,obj.coef,obj.xd).cpu().numpy() if a.method=='pinn' else None
    replay=obj.solver.predict(obj.solver.forward(D)[0]);rms=float(np.sqrt(np.mean((replay-ds.u)**2))/ds.sigma)
    metric=dict(key=a.key,method=a.method,variant='calibrated_hard_physics_v12',state_model='gaussian_ic_bc_locald',status=status,seed=42,n_steps=obj.solver.disc.n_steps,sigma=ds.sigma,D_const=obj.D_const,rms_const_over_sigma=obj.const_rms/ds.sigma,rms_replay_over_sigma=rms,rms_unet_over_sigma=float(np.sqrt(np.mean((fit-ds.u)**2))/ds.sigma) if fit is not None else rms,wall_s=time.monotonic()-started,adam_completed=adam_done,optimizer_diagnostics=rounds,history=history,final_parts=obj.parts,objective=value,function_evaluations=evals,initial_coefficient_sha256=obj.initial_weights_sha256,gamma=obj.gamma,alpha=1.,scale=obj.scale,initial_ic_relative_error=obj.ic_initial_relative_error,settings=vars(a),parameter_count=len(accepted))
    coefficient=obj.coef.checkpoint();coefficient['state_dict']={k:v.cpu() for k,v in coefficient['state_dict'].items()}
    saved=dict(key=a.key,method=a.method,coefficient=coefficient,final_training_points=obj.col.cpu(),T=ds.T)
    if a.method=='pinn':saved.update(temperature_config=obj.temp_config,temperature_state_dict={k:v.cpu() for k,v in obj.temp.state_dict().items()})
    torch.save(saved,dest/'model.pt');np.savez_compressed(dest/'D_hat.npz',D_hat=D.reshape(ds.N,ds.N));(dest/'metrics.json').write_text(json.dumps(metric,indent=2));print(json.dumps(metric),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--key',required=True);p.add_argument('--out',required=True);p.add_argument('--method',choices=['pinn','adjoint_nn'],required=True);p.add_argument('--device',default='cpu');p.add_argument('--adam',type=int,default=1000);p.add_argument('--outer',type=int,default=10);p.add_argument('--inner',type=int,default=400);p.add_argument('--n-pde',type=int,default=4096);p.add_argument('--budget',type=float,default=6500);train(p.parse_args())
