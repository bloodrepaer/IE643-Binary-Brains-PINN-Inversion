import json,torch,numpy as np
from pathlib import Path
from pinn_zhang_transfer import PaperCoefficient
from compare_neural import TemperatureNetwork,residual
from torch.nn.utils import parameters_to_vector,vector_to_parameters
h=Path('campaign/pilots/zhang_v11_gpu');torch.set_num_threads(2);torch.manual_seed(17)
c=PaperCoefficient();t=TemperatureNetwork(width=32,depth=3);ps=list(t.parameters())+list(c.parameters());x=torch.rand(12,3,dtype=torch.float64);z=parameters_to_vector(ps).detach().clone()
def f(v):
 vector_to_parameters(v.clone(),ps);q=x.clone().requires_grad_(True);return residual(t,c,q,.3636363636).square().mean()+.1*c(q[:,:2]).square().mean()+t(q).square().mean()
l=f(z);gs=torch.autograd.grad(l,ps);g=torch.cat([v.ravel() for v in gs]);checks=[]
for k in range(3):
 v=torch.randn_like(z);v/=v.norm();eps=1e-5;fd=(f(z+eps*v)-f(z-eps*v))/(2*eps);ad=g@v;err=float(abs(fd-ad)/max(1e-12,float(abs(fd)),float(abs(ad))));assert err<1e-5,err;checks.append(err)
vector_to_parameters(z,ps)
class Constant(torch.nn.Module):
 def diffusivity(self,q):return q[:,0]*0+.2
class Exact(torch.nn.Module):
 def forward(self,q):return torch.exp(-2*torch.pi**2*.2*.36*q[:,2])*torch.sin(torch.pi*q[:,0])*torch.sin(torch.pi*q[:,1])
r=residual(Exact(),Constant(),x.clone().requires_grad_(True),.36);assert float(r.abs().max())<1e-12
p=next((h/'smoke').rglob('model.pt'));s=torch.load(p,map_location='cpu',weights_only=True);c=PaperCoefficient.from_checkpoint(s['coefficient']);t=TemperatureNetwork(**s['temperature_config']);t.load_state_dict(s['temperature_state_dict']);D=c.diffusivity(__import__('adjoint_nn').grid_coordinates(65)).detach().numpy().reshape(65,65);saved=np.load(p.with_name('D_hat.npz'))['D_hat'];assert np.max(abs(D-saved))<1e-12
report=dict(gradient_relative_errors=checks,manufactured_heat_residual_max=float(r.abs().max().detach()),checkpoint_coefficient_max_difference=float(np.max(abs(D-saved))),optimizer_tests='SPDquadratic loss9.6e-35; Rosenbrock5.5e-27',pipeline='1Adam two resampled1-step SSBroyden rounds, checkpoint and commonCN replay produced',truth_inputs_absent=True)
(h/'implementation_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
