"""No-answer-key verification of the matched backward-Euler neural control."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from integrators import BoundaryCondition
from adjoint_inversion import Dataset
from adjoint_nn import grid_coordinates
from neural_diffusivity import NeuralDiffusivity
import matched_temporal_adjoint as V

def main():
    torch.set_num_threads(2);h=V.HOME;h.mkdir(exist_ok=True)
    N=9;T=.2;D=.25;x=np.linspace(0,1,N);u0=np.sin(np.pi*x[:,None])*np.sin(np.pi*x[None,:])
    ds=SimpleNamespace(N=N,T=T,bc=BoundaryCondition(),u0=u0,x=np.array([.25,.5,.75]),
        y=np.array([.5,.5,.5]),t=np.array([0.,.071,T]),u=np.array([.1,.1,.1]),sigma=.01)
    solver=V.BackwardEulerSolver(ds);W,c,L,solve=solver.forward(np.full(N*N,D));d=solver.disc
    eigen=-8*D/d.h**2*np.sin(np.pi/(2*(N-1)))**2
    exact=np.array([(1-d.dt*eigen)**(-k)*u0.ravel() for k in range(101)])
    assert np.max(abs(W-exact))<1e-12
    for k in range(101):assert d.state_of_step(k)==k
    assert np.allclose(np.asarray(solver.Q.sum(axis=1)).ravel(),1)
    assert np.max(abs(solver.predict(W)-solver.predict(exact)))<1e-12
    rng=np.random.default_rng(20261021);m=np.log(D)+rng.normal(0,.1,N*N)
    f,g=solver.field_objective(m,1e-4);checks=[]
    for _ in range(2):
        v=rng.normal(size=m.shape);v/=np.linalg.norm(v);e=1e-6
        fd=(solver.field_objective(m+e*v,1e-4,False)-solver.field_objective(m-e*v,1e-4,False))/(2*e)
        error=abs(fd-g@v)/max(abs(fd),abs(g@v),1e-12);assert error<1e-5
        checks.append(float(error))
    W,c,_,_=solver.forward(np.exp(m));Lf=d.full_operator(c)
    defect=W[1:]-W[:-1]-d.dt*(Lf@W[1:].T).T
    assert np.max(abs(defect[:,d.I]))<1e-12
    report=dict(smokes_not_results=True,coefficient_truth_used=False,
        manufactured_discrete_mode='PASS',observation_remapping='PASS',
        nonuniform_D_discrete_residual_max=float(np.max(abs(defect[:,d.I]))),
        field_gradient_errors=checks,cases=[])
    entries={e['key']:e for e in json.loads((V.SUPPLIED/'manifest.json').read_text())}
    initial=json.loads((Path('campaign/pilots/temporal_v18_gpu')/'pilot.json').read_text())['initial_coefficient_sha256_by_key']
    for key in V.KEYS:
        ds=Dataset(V.SUPPLIED,entries[key]);obj=V.MatchedObjective(ds)
        assert obj.initial_weights_sha256==initial[key]
        assert obj.solver.disc.n_steps==100 and np.all(obj.solver.disc.th_sub==1)
        z=obj.weights();z+=rng.normal(0,1e-3,len(z));f,g=obj.fg(z);errors=[]
        for _ in range(2):
            v=rng.normal(size=z.shape);v/=np.linalg.norm(v);e=1e-6
            fd=(obj.fg(z+e*v)[0]-obj.fg(z-e*v)[0])/(2*e)
            error=abs(fd-g@v)/max(abs(fd),abs(g@v),1e-12);assert error<3e-4,(key,error)
            errors.append(float(error))
        a=argparse.Namespace(key=key,out=str(h/'pipeline_smoke'),budget=900.,adam=1,outer=1,inner=1)
        V.train(a);p=h/'pipeline_smoke'/key/'adjoint_nn';metrics=json.loads((p/'metrics.json').read_text())
        assert metrics['status']=='complete' and metrics['adam_completed']==1 and metrics['optimizer_diagnostics'][0]['iterations']==1
        saved=torch.load(p/'model.pt',weights_only=True,map_location='cpu');coef=NeuralDiffusivity.from_checkpoint(saved['coefficient'])
        with torch.no_grad():Dhat=coef.diffusivity(grid_coordinates(ds.N)).numpy().reshape(ds.N,ds.N)
        with np.load(p/'D_hat.npz') as a:assert np.max(abs(Dhat-a['D_hat']))<1e-12
        assert metrics['initial_coefficient_sha256']==initial[key]
        report['cases'].append(dict(key=key,gradient_errors=errors,pipeline='PASS',initial_sha=initial[key]))
        (h/'preflight.json').write_text(json.dumps(report,indent=2));print(key,'PASS',flush=True)
    report['complete']=True;(h/'preflight.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
