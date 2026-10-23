"""Preflight v13 sampler/control identities and bounded pipeline smokes."""
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from adjoint_inversion import Dataset
from calibrated_neural_pair import PairObjective
from sampling_neural_pair import SamplingObjective


def main():
    torch.set_num_threads(1)
    home = Path('campaign/pilots/sampling_v13_gpu')
    protocol = json.loads((home/'pilot.json').read_text())
    root = Path(protocol['data_collection'])
    entry = next(e for e in json.loads((root/'manifest.json').read_text()) if e['key']==protocol['cases'][0])
    ds = Dataset(root, entry)
    control = PairObjective(ds, 'pinn', n_pde=64)
    adapter = SamplingObjective(ds, 'pinn', n_pde=64, sampler='random')
    assert np.array_equal(control.weights(), adapter.weights())
    for _ in range(3):
        assert torch.equal(control.col, adapter.col)
        control.batch(); adapter.batch()
    z = control.weights(); value, grad = control.fg(z)
    other, other_grad = adapter.fg(z)
    assert value == other and np.array_equal(grad, other_grad)
    sobol = SamplingObjective(ds, 'pinn', n_pde=64, sampler='sobol')
    repeat = SamplingObjective(ds, 'pinn', n_pde=64, sampler='sobol')
    assert np.array_equal(z, sobol.weights())
    for _ in range(3):
        assert torch.equal(sobol.col, repeat.col)
        assert len(torch.unique(sobol.col, dim=0)) == 64
        assert float(sobol.col.min()) >= 0 and float(sobol.col.max()) <= 1
        assert float(sobol.col[:32, 2].min()) >= 1e-4
        sobol.batch(); repeat.batch()
    value, grad = sobol.fg(z)
    rng = np.random.default_rng(20261034); errors=[]
    for _ in range(2):
        direction = rng.normal(size=len(z)); direction /= np.linalg.norm(direction)
        eps = 1e-5
        numeric = (sobol.fg(z+eps*direction)[0]-sobol.fg(z-eps*direction)[0])/(2*eps)
        exact = float(grad@direction)
        errors.append(abs(numeric-exact)/max(abs(numeric), abs(exact), 1e-12))
    assert max(errors) < 1e-5, errors
    (home/'implementation_validation.json').write_text(json.dumps(dict(
        uses_true_D=False, random_control_points_weights_objective_gradient_exact=True,
        sobol_seed_reproducible_domain_unique=True, shared_initial_sha=sobol.initial_weights_sha256,
        directional_gradient_relative_errors=errors), indent=2))
    smokes=[]
    for arm in protocol['arms']:
        command=[sys.executable,'-u','sampling_neural_pair.py','--data',str(root),
                 '--key',protocol['cases'][0],'--out',str(home/'pipeline_smoke'),
                 '--arm',arm,'--adam','1','--outer','1','--inner','1','--n-pde','8']
        log=home/(arm+'_smoke.log')
        with log.open('w') as f:
            subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        method='adjoint_nn' if arm=='adjoint_control' else 'pinn'
        m=json.loads((home/'pipeline_smoke'/arm/method/'metrics.json').read_text())
        assert m['status']=='complete' and m['adam_completed']==1
        smokes.append(dict(arm=arm,initial_sha=m['initial_coefficient_sha256'],status=m['status']))
    assert len({r['initial_sha'] for r in smokes}) == 1
    (home/'pipeline_validation.json').write_text(json.dumps(smokes,indent=2))
    print('Implementation and all three bounded pipeline checks passed',errors,flush=True)


if __name__=='__main__':
    main()
