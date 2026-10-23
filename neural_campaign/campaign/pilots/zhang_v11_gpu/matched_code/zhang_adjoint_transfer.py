"""Matched neural-adjoint arm of the separate Zhang heat-transfer pilot.

Same physical logD network, initial weights, mean data objective, gamma and
collocation-based H1 quadrature as the PINN; CN enforces the state equation.
No answer keys. Separate from the frozen broad discrepancy-path campaign.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector,vector_to_parameters
from adjoint_inversion import Dataset,AdjointSolver
from adjoint_nn import grid_coordinates
from compare_neural import derivative
from pinn_zhang_transfer import PaperCoefficient,quasi_newton


class MatchedObjective:
    def __init__(self,ds,seed=42,gamma=.1,n_pde=20000,n_bc=512,n_ic=512):
        self.ds=ds;self.coef=PaperCoefficient(seed=seed);self.params=list(self.coef.parameters());self.xy=grid_coordinates(ds.N)
        self.solver=AdjointSolver(ds);self.scale=2*self.solver.sigma_w**2/len(ds.u);self.gamma=gamma
        self.rng=np.random.default_rng(seed);self.rng.uniform(0,1,(n_ic,2))
        self.n_pde=n_pde;self.n_bc=n_bc;self.batch();self.parts={}
    def batch(self):
        # Match the PINN random stream, including unused BC draws. It draws one
        # initial batch before Adam and another at Adam step0.
        self.col=torch.tensor(self.rng.uniform(0,1,(self.n_pde,3)),dtype=torch.float64)
        self.rng.uniform(0,1,(self.n_bc,3));self.rng.integers(0,4,self.n_bc)
    def weights(self):return parameters_to_vector(self.params).detach().numpy().copy()
    def fg(self,z):
        vector_to_parameters(torch.tensor(z.copy(),dtype=torch.float64),self.params)
        m=self.coef(self.xy);data,gm=self.solver.field_objective(m.detach().numpy(),0.)
        col=self.col.clone().requires_grad_(True);logD=self.coef(col[:,:2]);dm=derivative(logD,col)
        reg=.5*self.gamma*dm[:,:2].square().sum(1).mean()
        surrogate=(m*torch.tensor(gm*self.scale,dtype=torch.float64)).sum()+reg
        gradients=torch.autograd.grad(surrogate,self.params)
        value=data*self.scale+float(reg.detach());self.parts=dict(data=float(data*self.scale),reg=float(reg.detach()),replay_sigma=float(self.solver._last['rms']/self.ds.sigma))
        return value,torch.cat([g.ravel() for g in gradients]).detach().numpy()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--key',required=True);p.add_argument('--out',required=True)
    p.add_argument('--adam',type=int,default=1000);p.add_argument('--outer',type=int,default=10);p.add_argument('--inner',type=int,default=400);p.add_argument('--n-pde',type=int,default=20000);p.add_argument('--budget',type=float,default=6500)
    a=p.parse_args();torch.set_num_threads(1);root=Path(a.data);entry=next(e for e in json.loads((root/'manifest.json').read_text()) if e['key']==a.key)
    ds=Dataset(root,entry);obj=MatchedObjective(ds,n_pde=a.n_pde);dest=Path(a.out);dest.mkdir(parents=True,exist_ok=False);started=time.monotonic();rounds=[];history=[];status='complete';evals=0;adam_done=0
    def fg(z):
        nonlocal evals
        if time.monotonic()-started>a.budget:raise TimeoutError('Bounded matched adjoint budget')
        evals+=1;return obj.fg(z)
    # NumPy derivative supplied to Adam is the exact reduced discrete gradient.
    opt=torch.optim.Adam(obj.params,lr=1e-3);sched=torch.optim.lr_scheduler.StepLR(opt,step_size=max(1,a.adam//4),gamma=.5)
    accepted=obj.weights()
    try:
        for k in range(a.adam):
            if k%100==0:obj.batch()
            value,g=fg(obj.weights());opt.zero_grad();offset=0
            for v in obj.params:v.grad=torch.tensor(g[offset:offset+v.numel()].reshape(v.shape),dtype=v.dtype);offset+=v.numel()
            opt.step();sched.step();accepted=obj.weights();adam_done=k+1
            if k%100==0:print('Adam',k,value,obj.parts,flush=True)
        x=accepted.copy();H=np.eye(len(x))
        for o in range(a.outer):
            obj.batch();before=evals;x,H,record=quasi_newton(fg,x,H,a.inner);accepted=x.copy()
            H=.5*(H+H.T)
            try:np.linalg.cholesky(H);record['hessian_retained']=True
            except np.linalg.LinAlgError:H=np.eye(len(x));record['hessian_retained']=False
            record.update(round=o+1,function_evaluations=evals-before);rounds.append(record);history.append(obj.parts.copy());print('SSBroyden',record,flush=True)
    except TimeoutError:status='budget_timeout'
    # Never retain the last line-search trial after a budget exception.
    value,_=obj.fg(accepted);wall=time.monotonic()-started
    D=obj.coef.diffusivity(obj.xy).detach().numpy().reshape(ds.N,ds.N)
    metrics=dict(key=a.key,method='adjoint_nn_zhang_matched',status=status,seed=42,gamma=obj.gamma,objective=value,parts=obj.parts,adam_completed=adam_done,optimizer_diagnostics=rounds,history=history,function_evaluations=evals,wall_s=wall,n_steps=obj.solver.disc.n_steps,settings=vars(a),rms_replay_over_sigma=obj.parts['replay_sigma'])
    np.savez_compressed(dest/'D_hat.npz',D_hat=D);torch.save(obj.coef.checkpoint(),dest/'model.pt');(dest/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
