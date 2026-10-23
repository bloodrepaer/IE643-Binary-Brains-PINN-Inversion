"""No-truth full-data/full16384-point gradient checks and three short pipelines."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from adjoint_inversion import Dataset
from calibrated_neural_pair import PairObjective


def main():
    torch.set_num_threads(1);home=Path('campaign/pilots/dense_resolution_v15_gpu')
    protocol=json.loads((home/'pilot.json').read_text());root=Path(protocol['data_collection'])
    key=protocol['cases'][0];entry=next(e for e in json.loads((root/'manifest.json').read_text()) if e['key']==key)
    ds=Dataset(root,entry);control=PairObjective(ds,'pinn',n_pde=4096)
    repeat=PairObjective(ds,'pinn',n_pde=4096);dense=PairObjective(ds,'pinn',n_pde=16384)
    assert len(ds.u)==31752
    assert control.initial_weights_sha256==repeat.initial_weights_sha256==dense.initial_weights_sha256
    assert control.gamma==dense.gamma and control.scale==dense.scale
    for _ in range(3):
        assert torch.equal(control.col,repeat.col)
        control.batch();repeat.batch()
    z=control.weights();v,g=control.fg(z);w,h=repeat.fg(z)
    assert v==w and np.array_equal(g,h)
    rng=np.random.default_rng(20261120);z=dense.weights()+rng.normal(0,1e-3,len(z))
    v,g=dense.fg(z);errors=[]
    for _ in range(2):
        d=rng.normal(size=len(z));d/=np.linalg.norm(d);eps=1e-5
        numeric=(dense.fg(z+eps*d)[0]-dense.fg(z-eps*d)[0])/(2*eps)
        exact=float(g@d);errors.append(abs(numeric-exact)/max(abs(numeric),abs(exact),1e-12))
    assert max(errors)<1e-5,errors
    assert dense.col.shape==(16384,3) and float(dense.col.min())>=0 and float(dense.col.max())<=1
    assert float(dense.col[:8192,2].min())>=1e-4
    (home/'implementation_validation.json').write_text(json.dumps(dict(uses_true_D=False,
        full_observation_rows=len(ds.u),pde_points=16384,perturbed_state_and_coefficient=True,
        directional_gradient_relative_errors=errors,control_points_weights_objective_gradient_exact=True,
        initial_sha_shared=control.initial_weights_sha256,gamma=dense.gamma),indent=2))
    rows=[]
    for arm in protocol['arms']:
        cmd=[sys.executable,'-u','dense_resolution_pair.py','--data',str(root),'--key',key,
             '--out',str(home/'pipeline_smoke'),'--arm',arm,'--adam','1','--outer','1','--inner','1']
        with (home/(arm+'_smoke.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
        method='adjoint_nn' if arm=='adjoint_control' else 'pinn'
        m=json.loads((home/'pipeline_smoke'/arm/method/'metrics.json').read_text())
        assert m['status']=='complete' and m['adam_completed']==1
        rows.append(dict(arm=arm,n_pde=m['n_pde'],status=m['status'],initial_sha=m['initial_coefficient_sha256']))
    assert len({r['initial_sha'] for r in rows})==1
    (home/'pipeline_validation.json').write_text(json.dumps(rows,indent=2))
    print('Full16384-point/full-data gradients and all3 short pipelines PASS',errors,flush=True)


if __name__=='__main__':main()
