"""Versioned Zhang et al. v2 transfer to supplied transient diffusion data.

Research adaptation, not a reproduction of any of their four different PDEs.
No answer keys are loaded. Dense SSBroyden runs on CPU; residuals may use CUDA.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import line_search
from torch import nn
from torch.nn.utils import parameters_to_vector,vector_to_parameters
from adjoint_inversion import Dataset,AdjointSolver
from adjoint_nn import grid_coordinates
from compare_neural import TemperatureNetwork,residual,derivative
from pinn_papers import ssbroyden_update


class PaperCoefficient(nn.Module):
    def __init__(self,width=32,depth=3,seed=42):
        super().__init__();self.config=dict(width=width,depth=depth,seed=seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);layers=[];n=2
            for _ in range(depth):
                layer=nn.Linear(n,width,dtype=torch.float64);nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
                layers.extend([layer,nn.Tanh()]);n=width
            layer=nn.Linear(n,1,dtype=torch.float64);nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
            self.net=nn.Sequential(*layers,layer)
    def forward(self,xy):return self.net(2*xy-1).ravel()
    def diffusivity(self,xy):return self(xy).exp()
    def checkpoint(self):return dict(kind='zhang_exp',config=self.config,state_dict=self.state_dict())
    @classmethod
    def from_checkpoint(cls,c):
        model=cls(**c['config']);model.load_state_dict(c['state_dict']);return model


def quasi_newton(fg,x,H,iterations):
    f,g=fg(x);done=0;resets=0;reason='iteration_cap'
    for _ in range(iterations):
        p=-H@g
        if not np.isfinite(p).all() or g@p>=0:H=np.eye(len(x));p=-g;resets+=1
        cache={}
        def pair(z):
            key=z.tobytes()
            if key not in cache:cache[key]=fg(z)
            return cache[key]
        fv=lambda z:pair(z)[0];gv=lambda z:pair(z)[1]
        a=line_search(fv,gv,x,p,gfk=g,old_fval=f,c1=1e-4,c2=.9,maxiter=30)[0]
        if a is None:
            H=np.eye(len(x));p=-g;resets+=1
            a=line_search(fv,gv,x,p,gfk=g,old_fval=f,c1=1e-4,c2=.9,maxiter=30)[0]
        if a is None:reason='line_search_failed';break
        z=x+a*p;fn,gn=pair(z)
        if not np.isfinite(fn) or not np.isfinite(gn).all():reason='nonfinite';break
        H=ssbroyden_update(H,a*p,gn-g,g,a)
        x,f,g=z,fn,gn;done+=1
        if np.linalg.norm(g)<1e-12:reason='gradient_tolerance';break
    return x,H,dict(iterations=done,requested_iterations=iterations,reason=reason,loss=float(f),gradient_max_abs=float(np.max(np.abs(g))),resets=resets)


def train(a):
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    entries={e['key']:e for e in json.loads((Path(a.data)/'manifest.json').read_text())};ds=Dataset(Path(a.data),entries[a.key]);device=torch.device(a.device)
    rng=np.random.default_rng(a.seed);started=time.monotonic();dest=Path(a.out)/a.key/'pinn__s42';dest.mkdir(parents=True,exist_ok=False)
    temp=TemperatureNetwork(width=32,depth=3,seed=a.seed).to(device)
    # Match the paper's Xavier rule also on the state output layer.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(a.seed+1);nn.init.xavier_uniform_(temp.net[-1].weight);nn.init.zeros_(temp.net[-1].bias)
    coef=PaperCoefficient(seed=a.seed).to(device);params=list(temp.parameters())+list(coef.parameters())
    tensor=lambda x:torch.tensor(x,device=device,dtype=torch.float64)
    xd=tensor(np.c_[ds.x,ds.y,ds.t/ds.T]);yd=tensor(ds.u)
    xy=rng.uniform(0,1,(a.n_ic,2));xi=tensor(np.c_[xy,np.zeros(len(xy))])
    # Bilinear interpolation of supplied initial field, never synthetic answer keys.
    from scipy.interpolate import RegularGridInterpolator
    ui=tensor(RegularGridInterpolator((np.linspace(0,1,ds.N),)*2,ds.u0)(xy))
    def batch():
        col=tensor(rng.uniform(0,1,(a.n_pde,3)));wall=rng.uniform(0,1,(a.n_bc,3));edges=rng.integers(0,4,len(wall));targets=np.zeros(len(wall))
        for i,name in enumerate(('left','right','bottom','top')):
            mask=edges==i;wall[mask,0 if i<2 else 1]=i%2;targets[mask]=[ds.bc.at(t*ds.T)[name] for t in wall[mask,2]]
        return col,tensor(wall),tensor(targets)
    def losses(b):
        col,wall,targets=b;col=col.detach().requires_grad_(True)
        r=residual(temp,coef,col,ds.T);m=coef(col[:,:2]);dm=derivative(m,col)
        parts=dict(pde=r.square().mean(),bc=(temp(wall)-targets).square().mean(),ic=(temp(xi)-ui).square().mean(),data=(temp(xd)-yd).square().mean(),reg=.5*a.gamma*dm[:,:2].square().sum(1).mean())
        return sum(parts.values()),parts
    evaluations=0
    def check_budget():
        if time.monotonic()-started>a.budget:raise TimeoutError('Bounded paper transfer budget expired')
    b=batch();opt=torch.optim.Adam(params,lr=1e-3);scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=max(1,a.adam//4),gamma=.5);history=[];rounds=[];status='complete'
    try:
        for k in range(a.adam):
            check_budget()
            if k%100==0:b=batch()
            opt.zero_grad();loss,parts=losses(b);loss.backward();opt.step();scheduler.step()
            if k%250==0:print('Adam',k,float(loss.detach()),flush=True)
        x=parameters_to_vector(params).detach().cpu().numpy().copy();H=np.eye(len(x))
        for o in range(a.outer):
            b=batch();calls=0
            def fg(z):
                nonlocal calls,evaluations
                check_budget();calls+=1;evaluations+=1
                vector_to_parameters(tensor(z.copy()),params);loss,_=losses(b);g=torch.autograd.grad(loss,params)
                value=float(loss.detach());v=torch.cat([p.ravel() for p in g]).detach().cpu().numpy()
                return value,v
            x,H,record=quasi_newton(fg,x,H,a.inner);vector_to_parameters(tensor(x.copy()),params)
            H=.5*(H+H.T)
            try:np.linalg.cholesky(H);record['hessian_retained']=True
            except np.linalg.LinAlgError:H=np.eye(len(x));record['hessian_retained']=False
            record.update(round=o+1,function_evaluations=calls);rounds.append(record);print('SSBroyden',record,flush=True)
            loss,parts=losses(b);history.append({k:float(v.detach()) for k,v in parts.items()})
    except TimeoutError:status='budget_timeout'
    with torch.no_grad():
        D=coef.diffusivity(grid_coordinates(ds.N).to(device)).cpu().numpy();fit=temp(xd).cpu().numpy()
    solver=AdjointSolver(ds);constant=solver.best_constant();Dconst=float(np.exp(constant).mean());const_rms=float(solver._last['rms']);replay=solver.predict(solver.forward(D)[0]);wall_s=time.monotonic()-started
    metrics=dict(key=a.key,variant='zhang_v2_heat_transfer',state_model='plain',status=status,sigma=ds.sigma,n_steps=solver.disc.n_steps,D_const=Dconst,rms_const_over_sigma=const_rms/ds.sigma,rms_replay_over_sigma=float(np.sqrt(np.mean((replay-ds.u)**2))/ds.sigma),rms_unet_over_sigma=float(np.sqrt(np.mean((fit-ds.u)**2))/ds.sigma),wall_s=wall_s,optimizer_diagnostics=rounds,history=history,gamma=a.gamma,parameter_count=len(parameters_to_vector(params)),quasi_newton_function_evaluations=evaluations,settings=vars(a))
    torch.save(dict(temperature_config=dict(width=32,depth=3,seed=a.seed),temperature_state_dict={k:v.cpu() for k,v in temp.state_dict().items()},coefficient={**coef.checkpoint(),'state_dict':{k:v.cpu() for k,v in coef.state_dict().items()}},T=ds.T,final_training_points=b[0].cpu()),dest/'model.pt')
    np.savez_compressed(dest/'D_hat.npz',D_hat=D.reshape(ds.N,ds.N));(dest/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--key',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=42);p.add_argument('--adam',type=int,default=1000);p.add_argument('--outer',type=int,default=10);p.add_argument('--inner',type=int,default=400);p.add_argument('--n-pde',type=int,default=20000);p.add_argument('--n-bc',type=int,default=512);p.add_argument('--n-ic',type=int,default=512);p.add_argument('--gamma',type=float,default=.1);p.add_argument('--budget',type=float,default=6500);train(p.parse_args())
