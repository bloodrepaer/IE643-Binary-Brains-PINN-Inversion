"""v18: five dense inverse cases, temporal consistency and scale-matched control."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import hashlib
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import calibrated_neural_pair as base
from compare_neural import temperature_value, derivative, residual, field_regularizer
from neural_diffusivity import NeuralDiffusivity

FIELDS=('A_smooth','B_gaussian_bump','D_layered','E_grf','F_defects')
KEYS=tuple(f'{f}__Fo0.1__uniform8__n3969__nodes_nested__eta0.05' for f in FIELDS)
ARMS=('temporal_pinn','scaled_pde_pinn','adjoint_nn')
STEP=.01
PARENT=base.PairObjective


def spatial_operator(temp,coef,q,T):
    u=temperature_value(temp,coef,q)
    du=derivative(u,q)
    D=coef.diffusivity(q[:,:2]).ravel()
    return T*(derivative(D*du[:,0],q)[:,0]+derivative(D*du[:,1],q)[:,1])


class TemporalObjective(PARENT):
    def __init__(self,ds,method,device='cpu',n_pde=4096,seed=42,arm='temporal_pinn'):
        self.arm=arm
        super().__init__(ds,method,device,n_pde,seed)
        initial_path=Path(ds.root)/'initial'/f'{ds.key}.pt'
        if initial_path.exists():
            init=torch.load(initial_path,map_location='cpu',weights_only=True)
            self.coef=NeuralDiffusivity.from_checkpoint(init['coefficient']).to(self.device)
            self.D_const=init['sensor_fitted_constant']
            self.temp_config['D_base']=self.D_const
            self.temp=base.TemperatureNetwork(**self.temp_config).to(self.device)
            self.params=(list(self.temp.parameters()) if method=='pinn' else [])+list(self.coef.parameters())
            self.initial_weights_sha256=hashlib.sha256(parameters_to_vector(self.coef.parameters()).detach().cpu().numpy().tobytes()).hexdigest()
            if self.initial_weights_sha256!=init['initial_coefficient_sha256']:raise ValueError('Shared initial coefficient mismatch')

    def batch(self):
        # Same uniform measure for both PINN arms; both time endpoints in [0,1].
        col=self.rng.uniform(0,1,(self.n_pde,3));col[:,2]*=1-STEP
        self.col=self.tensor(col)

    def fg(self,z):
        if self.method!='pinn':return super().fg(z)
        vector_to_parameters(self.tensor(z.copy()),self.params)
        q0=self.col.clone().requires_grad_(True)
        q1=(self.col+self.tensor([0.,0.,STEP])).requires_grad_(True)
        data=((temperature_value(self.temp,self.coef,self.xd)-self.yd)/self.scale).square().mean()
        if self.arm=='temporal_pinn':
            err=temperature_value(self.temp,self.coef,q1)-temperature_value(self.temp,self.coef,q0)-STEP*spatial_operator(self.temp,self.coef,q1,self.ds.T)
        elif self.arm=='scaled_pde_pinn':
            err=STEP*residual(self.temp,self.coef,q1,self.ds.T)
        else:raise ValueError('Unknown physics objective')
        physics=(err/self.scale).square().mean()
        reg=self.gamma*field_regularizer(self.coef(self.xy).ravel(),self.solver)
        loss=data+physics+reg
        gs=torch.autograd.grad(loss,self.params)
        self.parts=dict(data=float(data.detach()),pde=float(physics.detach()),reg=float(reg.detach()))
        return float(loss.detach()),torch.cat([g.ravel() for g in gs]).detach().cpu().numpy()


def train(a):
    if a.key not in KEYS or a.arm not in ARMS:raise ValueError('Outside predeclared five dense cases')
    method='adjoint_nn' if a.arm=='adjoint_nn' else 'pinn'
    def factory(ds,method,device='cpu',n_pde=4096,seed=42):
        return TemporalObjective(ds,method,device,n_pde,seed,arm=a.arm)
    original=base.PairObjective
    args=argparse.Namespace(data=a.data,key=a.key,out=str(Path(a.out)/a.key/a.arm),
        method=method,device=a.device,budget=a.budget,adam=a.adam,outer=a.outer,inner=a.inner,n_pde=4096)
    try:
        base.PairObjective=factory
        base.train(args)
    finally:base.PairObjective=original
    p=Path(args.out)/method/'metrics.json';m=json.loads(p.read_text())
    m.update(arm=a.arm,variant='temporal_density_v18',n_sensors=3969,n_pde=4096,
        optimizer='ssb',time_step_tau=STEP,physics_measure='uniform tau0 in [0,0.99], uniform space',
        physics_loss=('backward_Euler_temperature_defect' if a.arm=='temporal_pinn' else 'h_squared_continuous_residual' if a.arm=='scaled_pde_pinn' else 'hard_discrete_state'),
        interpretation='Adapted temporal-consistency loss test, not a full reproduction of Thakur architecture/training. All outcomes retained; capped/partial fits are not converged optima or method losses.')
    p.write_text(json.dumps(m,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for x in ['data','key','out']:p.add_argument('--'+x,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--budget',type=float,required=True)
    p.add_argument('--adam',type=int,default=1000);p.add_argument('--outer',type=int,default=10)
    p.add_argument('--inner',type=int,default=400)
    train(p.parse_args())
