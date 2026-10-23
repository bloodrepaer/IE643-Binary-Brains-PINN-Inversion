"""v16: controlled optimizer comparison for both neural inverse methods."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import calibrated_neural_pair as baseline

ARMS={'ssb_pinn':('pinn','ssb'),'lbfgsb_pinn':('pinn','lbfgsb'),
      'ssb_adjoint':('adjoint_nn','ssb'),'lbfgsb_adjoint':('adjoint_nn','lbfgsb')}


def lbfgsb_round(fg,x,H,iterations):
    result=minimize(fg,x,jac=True,method='L-BFGS-B',options=dict(maxiter=iterations,
        maxcor=50,maxls=30,gtol=1e-12,ftol=0.,maxfun=max(10000,iterations*50)))
    if result.nit>=iterations:reason='iteration_cap'
    elif result.success:reason='lbfgsb_stopping'
    else:reason='line_search_or_numerical_stop'
    record=dict(iterations=int(result.nit),requested_iterations=iterations,reason=reason,
        loss=float(result.fun),gradient_max_abs=float(np.max(abs(result.jac))),
        gradient_l2=float(np.linalg.norm(result.jac)),resets=0,optimizer_success=bool(result.success),
        optimizer_message=str(result.message),maxcor=50,maxls=30,gtol_infinity=1e-12,ftol=0.,
        curvature='Limited-memory updates reset per outer refresh; common trainer dense identity bookkeeping remains')
    return np.asarray(result.x).copy(),H,record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','key','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--budget',type=float,default=9000);p.add_argument('--adam',type=int,default=1000)
    p.add_argument('--outer',type=int,default=10);p.add_argument('--inner',type=int,default=400)
    p.add_argument('--smoke-n-pde',type=int)
    a=p.parse_args()
    if '__n3969__' not in a.key:raise ValueError('This predeclared pilot requires3969sensors')
    method,optimizer=ARMS[a.arm];n_pde=16384 if method=='pinn' else 4096
    if a.smoke_n_pde is not None:
        if a.adam>1 or a.outer>1 or a.inner>1:raise ValueError('Point override is smoke-only')
        n_pde=a.smoke_n_pde
    if optimizer=='lbfgsb':baseline.quasi_newton=lbfgsb_round
    args=argparse.Namespace(data=a.data,key=a.key,out=str(Path(a.out)/a.arm),method=method,
        device=a.device,budget=a.budget,adam=a.adam,outer=a.outer,inner=a.inner,n_pde=n_pde)
    baseline.train(args)
    path=Path(args.out)/method/'metrics.json';m=json.loads(path.read_text())
    m.update(arm=a.arm,optimizer=optimizer,variant='dense_optimizer_v16',n_sensors=3969,n_pde=n_pde,
        parent_protocol='dense_resolution_v15',
        optimizer_limitations='Different curvature memory/stopping conventions intrinsic; shared trainer retains dense bookkeeping also forL-BFGS-B. No memory or unqualifiedspeed claim.')
    path.write_text(json.dumps(m,indent=2))


if __name__=='__main__':main()
