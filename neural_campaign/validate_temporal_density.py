"""Predeclared v18 manufactured-operator, full-count gradient and pipeline checks."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import temporal_density_pair as V


class Wave(torch.nn.Module):
    def forward(self,q):
        return torch.exp(-2*np.pi**2*.25*.4*q[:,2])*torch.sin(np.pi*q[:,0])*torch.sin(np.pi*q[:,1])


class Constant(torch.nn.Module):
    def diffusivity(self,q):return q[:,0:1]*0+.25


def main():
    torch.set_num_threads(2)
    h=Path('campaign/pilots/temporal_v18_gpu');h.mkdir(exist_ok=True)
    q=torch.tensor(np.random.default_rng(12).uniform(.1,.8,(31,3)),dtype=torch.float64,requires_grad=True)
    w=Wave();c=Constant();f=V.spatial_operator(w,c,q,.4)
    exact=-2*np.pi**2*.25*.4*w(q)
    assert float((f-exact).abs().max())<1e-13
    q1=(q.detach()+torch.tensor([0.,0.,V.STEP])).requires_grad_(True)
    defect=w(q1)-w(q)-V.STEP*V.spatial_operator(w,c,q1,.4)
    expected=w(q)*((1+V.STEP*2*np.pi**2*.25*.4)*np.exp(-V.STEP*2*np.pi**2*.25*.4)-1)
    assert float((defect-expected).abs().max())<1e-13
    assert float(V.residual(w,c,q1,.4).abs().max())<1e-13
    rows=[];root=Path('datasets_v2_density_extended')
    entries={e['key']:e for e in json.loads((root/'manifest.json').read_text())}
    for key in V.KEYS:
        shared=[]
        for arm in V.ARMS:
            method='adjoint_nn' if arm=='adjoint_nn' else 'pinn'
            ds=V.base.Dataset(root,entries[key]);obj=V.TemporalObjective(ds,method,arm=arm)
            shared.append((obj.initial_weights_sha256,obj.gamma,obj.scale))
            assert obj.col[:,2].min()>=0 and obj.col[:,2].max()+V.STEP<=1
            z=obj.weights();rng=np.random.default_rng(31415);z+=rng.normal(size=z.shape)*1e-3
            f,g=obj.fg(z);checks=[]
            for _ in range(2):
                v=rng.normal(size=z.shape);v/=np.linalg.norm(v);eps=1e-6
                fd=(obj.fg(z+eps*v)[0]-obj.fg(z-eps*v)[0])/(2*eps)
                ad=float(g@v);err=abs(fd-ad)/max(abs(fd),abs(ad),1e-12)
                assert err<3e-4,(key,arm,err)
                checks.append(dict(relative_error=err,finite_difference=fd,automatic=ad))
            if method=='adjoint_nn':
                original=V.PARENT(ds,method);f0,g0=original.fg(z)
                assert f==f0 and np.array_equal(g,g0)
            elif arm=='temporal_pinn':
                cloud=obj.col.clone()
            else:assert torch.equal(cloud,obj.col)
            args=argparse.Namespace(data=str(root),key=key,out=str(h/'pipeline_smoke'),arm=arm,
                device='cpu',budget=900,adam=1,outer=1,inner=1)
            V.train(args)
            m=json.loads((h/'pipeline_smoke'/key/arm/method/'metrics.json').read_text())
            assert m['adam_completed']==1 and m['optimizer_diagnostics'][0]['iterations']==1
            rows.append(dict(key=key,arm=arm,gradient_checks=checks,full_count_pipeline='PASS'))
            (h/'preflight.json').write_text(json.dumps(dict(complete=False,uses_true_D=False,results=rows),indent=2))
            print(key,arm,'PASS',flush=True)
        assert len(set(shared))==1
    (h/'preflight.json').write_text(json.dumps(dict(complete=True,uses_true_D=False,
        manufactured_divergence_and_backward_Euler='PASS',shared_initial_SHA_gamma_scale='PASS',
        full_observations=31752,n_pde=4096,smokes_not_results=True,results=rows),indent=2))


if __name__=='__main__':main()
