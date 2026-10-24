"""Full8000-reading v22 gradients/pipelines without coefficient answer keys."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse,json
from pathlib import Path
import numpy as np
import torch
from adjoint_inversion import Dataset
from adjoint_nn import grid_coordinates
from neural_diffusivity import NeuralDiffusivity
import dense1000_pair as V

def main():
    torch.set_num_threads(2);h=Path('campaign/pilots/dense1000_v22_kaggle');root=h/'supplied_data'
    ds=Dataset(root,json.loads((root/'manifest.json').read_text())[0]);assert len(ds.u)==8000
    assert len(np.unique(np.c_[ds.x,ds.y],axis=0))==1000
    rows=[];shared=[]
    for arm in V.ARMS:
        method='pinn' if arm=='temporal_pinn' else 'adjoint_nn';o=V.objective(ds,method)
        shared.append((o.initial_weights_sha256,o.gamma,o.scale))
        z=o.weights();rng=np.random.default_rng(20261021);z+=rng.normal(0,1e-3,len(z))
        f,g=o.fg(z);errors=[]
        for _ in range(2):
            v=rng.normal(size=z.shape);v/=np.linalg.norm(v);e=1e-6
            fd=(o.fg(z+e*v)[0]-o.fg(z-e*v)[0])/(2*e)
            err=abs(fd-g@v)/max(abs(fd),abs(g@v),1e-12);assert err<3e-4,(arm,err)
            errors.append(float(err))
        a=argparse.Namespace(data=str(root),key=V.KEY,out=str(h/'pipeline_smoke'),arm=arm,
            device='cpu',budget=900.,adam=1,outer=1,inner=1)
        V.train(a);p=h/'pipeline_smoke'/V.KEY/arm/method;m=json.loads((p/'metrics.json').read_text())
        assert m['status']=='complete' and m['adam_completed']==1 and m['optimizer_diagnostics'][0]['iterations']==1
        s=torch.load(p/'model.pt',weights_only=True,map_location='cpu');c=NeuralDiffusivity.from_checkpoint(s['coefficient'])
        with torch.no_grad():D=c.diffusivity(grid_coordinates(ds.N)).numpy().reshape(ds.N,ds.N)
        with np.load(p/'D_hat.npz') as a:assert np.max(abs(D-a['D_hat']))<1e-12
        rows.append(dict(arm=arm,gradient_errors=errors,pipeline='PASS',checkpoint_reproduction='PASS'))
        print(arm,'PASS',flush=True)
    assert len(set(shared))==1
    (h/'preflight.json').write_text(json.dumps(dict(complete=True,smokes_not_results=True,coefficient_truth_used=False,
        full_readings=8000,n_sensors=1000,shared_initial_sha_gamma_scale=shared[0],results=rows),indent=2))

if __name__=='__main__':main()
