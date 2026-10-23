import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import json,numpy as np,torch
from pathlib import Path
from adjoint_inversion import Dataset
from calibrated_neural_pair import PairObjective
from compare_neural import temperature_value,residual
h=Path('campaign/pilots/calibrated_v12_gpu');h.mkdir(parents=True,exist_ok=True);root=Path('campaign/pilots/zhang_v11_gpu/dataset/data');ds=Dataset(root,json.loads((root/'manifest.json').read_text())[0]);torch.set_num_threads(2);report={};hashes=[]
for method in ['adjoint_nn','pinn']:
 obj=PairObjective(ds,method,n_pde=16);hashes.append(obj.initial_weights_sha256);z=obj.weights();rng=np.random.default_rng(20261019);probe=z+.01*rng.normal(size=len(z));f,g=obj.fg(probe);errors=[]
 for _ in range(2):
  v=rng.normal(size=len(z));v/=np.linalg.norm(v);eps=1e-5;fd=(obj.fg(probe+eps*v)[0]-obj.fg(probe-eps*v)[0])/(2*eps);ad=g@v;err=abs(fd-ad)/max(1e-10,abs(fd),abs(ad));assert err<2e-4,err;errors.append(float(err))
 obj.fg(z);xi=torch.cat([obj.xy,torch.zeros((len(obj.xy),1),dtype=torch.float64)],1)
 with torch.no_grad():ic=temperature_value(obj.temp,obj.coef,xi).numpy();wall=np.random.default_rng(4).uniform(0,1,(64,3));wall[:,0]=np.arange(64)%2;bc=temperature_value(obj.temp,obj.coef,torch.tensor(wall,dtype=torch.float64)).numpy()
 q=obj.col.clone().requires_grad_(True);r=residual(obj.temp,obj.coef,q,ds.T).detach().numpy();assert np.max(abs(r))<1e-8;assert np.max(abs(bc))<1e-12
 R=float(__import__('compare_neural').field_regularizer(obj.coef(obj.xy).ravel(),obj.solver).detach())
 if method=='adjoint_nn':J=obj.solver.field_objective(obj.coef(obj.xy).detach().numpy().ravel(),1.,False)
 else:
  fit=temperature_value(obj.temp,obj.coef,obj.xd).detach().numpy();J=.5*np.sum(((fit-ds.u)/obj.solver.sigma_w)**2)+R
 exact=obj.factor*J;actual=obj.parts['data']+obj.parts['reg'];assert abs(exact-actual)<1e-12
 report[method]=dict(gradient_relative_errors=errors,initial_ic_relative_error=float(np.linalg.norm(ic-ds.u0.ravel())/np.linalg.norm(ds.u0)),initial_bc_max=float(np.max(abs(bc))),constant_D_pde_max=float(np.max(abs(r))),sum_to_mean_objective_difference=abs(exact-actual),gamma=obj.gamma,initial_coefficient_sha256=obj.initial_weights_sha256)
assert hashes[0]==hashes[1];(h/'implementation_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
