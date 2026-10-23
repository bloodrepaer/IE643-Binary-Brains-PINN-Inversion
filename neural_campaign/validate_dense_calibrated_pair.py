"""Check full higher-density objectives without reading coefficient truth."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import json
from pathlib import Path
import numpy as np
import torch
from adjoint_inversion import Dataset
from calibrated_neural_pair import PairObjective


def main():
    torch.set_num_threads(1)
    home=Path('campaign/pilots/dense_calibrated_v14_gpu')
    protocol=json.loads((home/'pilot.json').read_text())
    root=home/'dataset/data'
    entries=json.loads((root/'manifest.json').read_text())
    results=[]
    for entry in entries:
        ds=Dataset(root,entry);hashes=[];gammas=[]
        for method in ['pinn','adjoint_nn']:
            obj=PairObjective(ds,method,n_pde=8)
            z=obj.weights();v,g=obj.fg(z);errors=[]
            rng=np.random.default_rng(20261036)
            for _ in range(2):
                direction=rng.normal(size=len(z));direction/=np.linalg.norm(direction)
                epsilon=1e-5
                numeric=(obj.fg(z+epsilon*direction)[0]-obj.fg(z-epsilon*direction)[0])/(2*epsilon)
                exact=float(g@direction)
                errors.append(abs(numeric-exact)/max(abs(numeric),abs(exact),1e-12))
            assert max(errors)<2e-5,(entry['key'],method,errors)
            obj.fg(z)
            m=obj.coef(obj.xy).detach().numpy().ravel()
            J,_=obj.solver.field_objective(m,1.)
            identity=abs(J*obj.factor-(obj.parts['data']+obj.parts['reg']))
            # PINN and CN states differ; identity checks use the adjoint data operator.
            if method=='adjoint_nn':assert identity<1e-15,identity
            assert len(ds.u)==entry['n_sensors']*8
            hashes.append(obj.initial_weights_sha256);gammas.append(obj.gamma)
            results.append(dict(key=entry['key'],n_sensors=entry['n_sensors'],
                method=method,full_observations=len(ds.u),gamma=obj.gamma,
                initial_sha=obj.initial_weights_sha256,gradient_relative_errors=errors,
                likelihood_conversion_error=identity if method=='adjoint_nn' else None))
            print(json.dumps(results[-1]),flush=True)
            del obj
        assert len(set(hashes))==1 and gammas[0]==gammas[1]
    (home/'implementation_validation.json').write_text(json.dumps(dict(uses_true_D=False,
        full_dense_objectives=True,shared_initialization_per_case=True,results=results),indent=2))


if __name__=='__main__':
    main()
