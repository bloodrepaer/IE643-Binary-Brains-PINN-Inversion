"""v20: backward-Euler neural-adjoint controls matched to v18 temporal step.

No prior numerical implementation or scientific outputs are overwritten.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from adjoint_inversion import AdjointSolver, Dataset, Discretization
from temporal_density_pair import TemporalObjective, KEYS, STEP
import calibrated_neural_pair as base

HOME=Path('campaign/pilots/matched_temporal_v20_local')
SUPPLIED=Path('campaign/pilots/temporal_v18_gpu/supplied_data')

class BackwardEulerDiscretization(Discretization):
    def __init__(self,N,T,n_steps,bc):
        super().__init__(N,T,n_steps,bc)
        self.h_sub=np.full(n_steps,self.dt);self.th_sub=np.ones(n_steps)
        self.t_state=np.arange(n_steps+1)*self.dt;self.S=n_steps+1
        self.G=np.stack([self.g(t) for t in self.t_state])

    def state_of_step(self,k):return k

    def observation_operator(self,x,y,t):
        # Reuse unchanged bilinear/linear interpolation, then remove the
        # Rannacher half-step column from the original indexing convention.
        old_size=self.S
        try:
            self.S=old_size+1
            q=super().observation_operator(x,y,t).tocoo()
        finally:self.S=old_size
        size=self.N**2;old_state=q.col//size
        cols=(old_state-(old_state>0))*size+q.col%size
        return sp.csr_matrix((q.data,(q.row,cols)),shape=(len(x),self.S*size))

class BackwardEulerSolver(AdjointSolver):
    def __init__(self, ds, n_steps=None, **kwargs):
        steps=int(round(1/STEP))
        if n_steps is not None and n_steps!=steps:raise ValueError('v20 fixed T/100 time step')
        super().__init__(ds,n_steps=steps,**kwargs)
        self.disc=BackwardEulerDiscretization(ds.N,ds.T,steps,ds.bc)
        self.Q=self.disc.observation_operator(ds.x,ds.y,ds.t)

    def forward(self,D):
        d=self.disc;c=d.conductance(D)
        Lf=self.full_operator=d.full_operator(c)
        LII=Lf[d.I][:,d.I];LIB=Lf[d.I][:,d.Bd]
        M=(sp.identity(d.nI,format='csc')-d.dt*LII).tocsc()
        solve=spla.splu(M,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0,
                       options=dict(SymmetricMode=True)).solve
        W=d.G.copy();W[0,d.I]=self.u0_full[d.I];u=W[0,d.I]
        has_bc=np.any(d.G[:,d.Bd]);b=(LIB@d.G[:,d.Bd].T).T if has_bc else None
        for j in range(d.S-1):
            rhs=u.copy()
            if has_bc:rhs+=d.dt*b[j+1]
            u=solve(rhs);W[j+1,d.I]=u
        return W,c,LII,solve

class MatchedObjective(TemporalObjective):
    def __init__(self,ds,method='adjoint_nn',device='cpu',n_pde=4096,seed=42):
        if method!='adjoint_nn':raise ValueError('Only the predeclared BE control')
        original=base.AdjointSolver
        try:
            base.AdjointSolver=BackwardEulerSolver
            super().__init__(ds,method,device,n_pde,seed,arm='adjoint_nn')
        finally:base.AdjointSolver=original

def train(a):
    assert a.key in KEYS
    original=base.PairObjective
    try:
        base.PairObjective=MatchedObjective
        base.train(argparse.Namespace(data=str(SUPPLIED),key=a.key,
            out=str(Path(a.out)/a.key),method='adjoint_nn',device='cpu',budget=a.budget,
            adam=a.adam,outer=a.outer,inner=a.inner,n_pde=4096))
    finally:base.PairObjective=original
    p=Path(a.out)/a.key/'adjoint_nn';m=json.loads((p/'metrics.json').read_text())
    entry=next(e for e in json.loads((SUPPLIED/'manifest.json').read_text()) if e['key']==a.key)
    ds=Dataset(SUPPLIED,entry)
    with np.load(p/'D_hat.npz') as z:D=z['D_hat'].ravel()
    cn=AdjointSolver(ds)
    common=float(np.sqrt(np.mean((cn.predict(cn.forward(D)[0])-ds.u)**2))/ds.sigma)
    m.update(variant='matched_backward_euler_v20',arm='be_adjoint_nn',time_integrator='backward_euler',
        time_step_tau=STEP,training_fit_be_sigma=m['rms_replay_over_sigma'],
        common_cn_replay_sigma=common,comparison='v18 temporal PINN: same time-step size, coefficient initialization, data and optimizer schedule; soft AD versus hard FD spatial/state enforcement remain intrinsic')
    (p/'metrics.json').write_text(json.dumps(m,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--key',required=True)
    p.add_argument('--out',default=str(HOME/'results'));p.add_argument('--budget',type=float,default=7200)
    p.add_argument('--adam',type=int,default=1000);p.add_argument('--outer',type=int,default=10);p.add_argument('--inner',type=int,default=400)
    train(p.parse_args())
