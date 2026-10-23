"""Preflight optimizer contract and unchanged SSB full-count controls, no truth."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from dense_optimizer_pair import ARMS,lbfgsb_round


def main():
    torch.set_num_threads(1);home=Path('campaign/pilots/optimizer_v16_gpu')
    protocol=json.loads((home/'pilot.json').read_text());root=Path(protocol['data_collection']);key=protocol['cases'][0]
    # A coupled positive-definite quadratic checks the fg/curvature/result contract.
    Q=np.array([[4.,1.],[1.,2.]]);target=np.array([.3,-.2])
    def fg(z):r=z-target;return float(.5*r@Q@r),Q@r
    x,H,r=lbfgsb_round(fg,np.zeros(2),np.eye(2),100)
    assert np.linalg.norm(x-target)<1e-7 and np.array_equal(H,np.eye(2))
    prior=json.loads(Path('campaign/pilots/dense_resolution_v15_gpu/implementation_validation.json').read_text())
    assert max(prior['directional_gradient_relative_errors'])<1e-8 and prior['full_observation_rows']==31752
    rows=[]
    for arm,(method,optimizer) in ARMS.items():
        out=home/'pipeline_smoke'
        command=[sys.executable,'-u','dense_optimizer_pair.py','--data',str(root),'--key',key,
                 '--out',str(out),'--arm',arm,'--adam','1','--outer','1','--inner','1']
        with (home/(arm+'_smoke.log')).open('w') as f:subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT,timeout=240)
        p=out/arm/method;m=json.loads((p/'metrics.json').read_text());assert m['status']=='complete' and m['adam_completed']==1
        saved=torch.load(p/'model.pt',map_location='cpu',weights_only=True)
        if optimizer=='ssb':
            reference=home/'baseline_controls'/arm
            command=[sys.executable,'-u','calibrated_neural_pair.py','--data',str(root),'--key',key,
                     '--out',str(reference),'--method',method,'--adam','1','--outer','1','--inner','1',
                     '--n-pde',str(m['n_pde'])]
            with (home/(arm+'_baseline.log')).open('w') as f:subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT,timeout=240)
            base=reference/method;old=torch.load(base/'model.pt',map_location='cpu',weights_only=True)
            assert all(torch.equal(v,old['coefficient']['state_dict'][k]) for k,v in saved['coefficient']['state_dict'].items())
            if method=='pinn':assert all(torch.equal(v,old['temperature_state_dict'][k]) for k,v in saved['temperature_state_dict'].items())
            assert m['history']==json.loads((base/'metrics.json').read_text())['history']
        rows.append(dict(arm=arm,initial_sha=m['initial_coefficient_sha256'],gamma=m['gamma'],n_pde=m['n_pde'],status=m['status']))
        print('Full-count pipeline PASS',arm,flush=True)
    assert len({r['initial_sha'] for r in rows})==1 and len({r['gamma'] for r in rows})==1
    (home/'implementation_validation.json').write_text(json.dumps(dict(uses_true_D=False,
        coupled_quadratic_solution_error=float(np.linalg.norm(x-target)),lbfgsb_result_contract=True,
        inherited_unchanged_v15_full_data_gradient_validation=prior,
        both_SSB_full_count_Adam_and_round_controls_bitwise_exact=True),indent=2))
    (home/'pipeline_validation.json').write_text(json.dumps(rows,indent=2))
    print('All4full-count pipelines and bothoriginalSSBcontrols PASS',flush=True)


if __name__=='__main__':main()
