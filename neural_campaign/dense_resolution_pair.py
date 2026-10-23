"""v15: change only PINN collocation count at3969 observed sensors."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse,json
from pathlib import Path
import calibrated_neural_pair as baseline

ARMS={'pde4096_control':('pinn',4096),'pde16384':('pinn',16384),'adjoint_control':('adjoint_nn',4096)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','key','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--budget',type=float,default=13000)
    p.add_argument('--adam',type=int,default=1000);p.add_argument('--outer',type=int,default=10)
    p.add_argument('--inner',type=int,default=400)
    p.add_argument('--smoke-n-pde',type=int,default=None)
    a=p.parse_args();method,n=ARMS[a.arm]
    if a.smoke_n_pde is not None:
        if a.adam>1 or a.outer>1 or a.inner>1:raise ValueError('Point override is smoke-only')
        n=a.smoke_n_pde
    args=argparse.Namespace(data=a.data,key=a.key,out=str(Path(a.out)/a.arm),method=method,
        device=a.device,budget=a.budget,adam=a.adam,outer=a.outer,inner=a.inner,n_pde=n)
    baseline.train(args)
    path=Path(args.out)/method/'metrics.json';m=json.loads(path.read_text())
    m.update(arm=a.arm,variant='dense_resolution_v15',n_sensors=3969,n_pde=n,
             parent_protocol='calibrated_hard_physics_v12',sampler='random',
             research_scope='Point-count-only PINN comparison; same full observed data and mixed-time measure.')
    path.write_text(json.dumps(m,indent=2))


if __name__=='__main__':main()
