"""Preflight all four frozen estimated coefficients without coefficient truth."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import dense_conditional_state as C


def main():
    torch.set_num_threads(2)
    home=Path('campaign/pilots/conditional_state_v17_gpu')
    home.mkdir(exist_ok=True)
    report=home/'preflight.json'
    if report.exists():raise ValueError('Preserve existing validation rather than overwrite')
    root=Path('datasets_v2_density_extended')
    entry=next(e for e in json.loads((root/'manifest.json').read_text()) if e['key']==C.KEY)
    ds=C.base.Dataset(root,entry)
    results=[]
    for arm in C.ARMS:
        started=time.monotonic()
        path=next((Path('campaign/pilots/optimizer_v16_gpu/artifacts/results')/arm).rglob('model.pt'))
        saved=torch.load(path,map_location='cpu',weights_only=True)
        obj=C.ConditionalObjective(ds,saved['coefficient'])
        initial_coef=C.tensor_sha(obj.coef)
        assert not any(p.requires_grad for p in obj.coef.parameters())
        assert len(obj.params)==len(list(obj.temp.parameters()))
        z=obj.weights();rng=np.random.default_rng(20261020)
        z=z+rng.normal(size=z.shape)*1e-4
        f,g=obj.fg(z);errors=[]
        for _ in range(2):
            v=rng.normal(size=z.shape);v/=np.linalg.norm(v);eps=1e-6
            fd=(obj.fg(z+eps*v)[0]-obj.fg(z-eps*v)[0])/(2*eps)
            ad=float(g@v);err=abs(fd-ad)/max(abs(fd),abs(ad),1e-12)
            assert err<2e-4,(arm,err,fd,ad)
            errors.append(dict(relative_error=err,finite_difference=fd,automatic=ad))
        assert C.tensor_sha(obj.coef)==initial_coef
        # Meaningful end-to-end full-count update; coefficient must be exact.
        args=argparse.Namespace(data=str(root),coefficient=str(path),key=C.KEY,
            arm=arm,out=str(home/'pipeline_smoke'),device='cpu',budget=600,
            adam=1,outer=1,inner=1)
        C.train(args)
        dest=home/'pipeline_smoke'/arm/'pinn'
        m=json.loads((dest/'metrics.json').read_text())
        after=torch.load(dest/'model.pt',map_location='cpu',weights_only=True)
        assert m['coefficient_frozen'] and not m['inverse_fit']
        assert m['adam_completed']==1 and m['optimizer_diagnostics'][0]['iterations']==1
        assert C.tensor_sha(C.NeuralDiffusivity.from_checkpoint(after['coefficient']))==initial_coef
        initial_temp=C.base.TemperatureNetwork(**after['temperature_config'])
        assert any(not torch.equal(v,after['temperature_state_dict'][k])
                   for k,v in initial_temp.state_dict().items()), 'State weights did not update'
        results.append(dict(arm=arm,state_gradient_directions=errors,
            frozen_coefficient_sha256=initial_coef,coefficient_unchanged=True,
            full_count_pipeline_passed=True,wall_s=time.monotonic()-started))
        report.write_text(json.dumps(dict(complete=False,uses_true_D=False,results=results),indent=2))
    report.write_text(json.dumps(dict(complete=True,uses_true_D=False,smoke_not_scientific_result=True,
        source_sha256=hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest(),results=results),indent=2))


if __name__=='__main__':main()
