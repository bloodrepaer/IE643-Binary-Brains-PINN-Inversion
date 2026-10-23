"""v14 higher-density pair; unchanged calibrated v12 numerical training code."""
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import argparse
import json
from pathlib import Path
import calibrated_neural_pair as baseline


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--key',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--method',choices=['pinn','adjoint_nn'],required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--budget',type=float,default=6500)
    p.add_argument('--adam',type=int,default=1000)
    p.add_argument('--outer',type=int,default=10)
    p.add_argument('--inner',type=int,default=400)
    p.add_argument('--n-pde',type=int,default=4096)
    a=p.parse_args()
    entry=next(e for e in json.loads((Path(a.data)/'manifest.json').read_text()) if e['key']==a.key)
    count=entry['n_sensors']
    a.out=str(Path(a.out)/f'n{count}')
    baseline.train(a)
    path=Path(a.out)/a.method/'metrics.json'
    m=json.loads(path.read_text())
    m.update(variant='dense_calibrated_v14',n_sensors=count,
             parent_protocol='calibrated_hard_physics_v12',
             density_limitations='Fixed likelihood alpha1; gamma conversion and fitted constant initialization depend on each input. Full observation objective, no density-dependent minibatching.')
    path.write_text(json.dumps(m,indent=2))


if __name__=='__main__':
    main()
