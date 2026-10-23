"""Fixed estimated-D conditional state diagnostic; never an inverse fit."""
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn.utils import parameters_to_vector
import calibrated_neural_pair as base
from neural_diffusivity import NeuralDiffusivity

ARMS = ('ssb_pinn', 'lbfgsb_pinn', 'ssb_adjoint', 'lbfgsb_adjoint')
KEY = 'A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05'


def tensor_sha(coef):
    return hashlib.sha256(parameters_to_vector(coef.parameters()).detach().cpu().numpy().tobytes()).hexdigest()


class ConditionalObjective(base.PairObjective):
    def __init__(self, ds, coefficient_checkpoint, device='cpu', n_pde=16384, seed=42):
        super().__init__(ds, 'pinn', device, n_pde, seed)
        # Preserve the original temperature architecture and sensor-fitted base
        # configuration; only the supplied retained coefficient is substituted.
        self.coef = NeuralDiffusivity.from_checkpoint(coefficient_checkpoint).to(self.device)
        self.coef.requires_grad_(False)
        self.params = list(self.temp.parameters())
        self.initial_weights_sha256 = tensor_sha(self.coef)
        with torch.no_grad():
            xi=torch.cat([self.xy,torch.zeros((len(self.xy),1),dtype=torch.float64,device=self.device)],1)
            ic=base.temperature_value(self.temp,self.coef,xi).cpu().numpy()
            self.ic_initial_relative_error=float(base.np.linalg.norm(ic-ds.u0.ravel())/base.np.linalg.norm(ds.u0))
        if self.ic_initial_relative_error>1e-6:raise ValueError('Fixed-D supplied IC mismatch')


def train(args):
    if args.key != KEY or args.arm not in ARMS:
        raise ValueError('Only the predeclared dense case and all-four source arms')
    source = Path(args.coefficient)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    saved = torch.load(source, map_location='cpu', weights_only=True)
    if saved.get('key', KEY) != KEY:
        raise ValueError('Wrong coefficient case')
    fixed = saved['coefficient']
    expected = tensor_sha(NeuralDiffusivity.from_checkpoint(fixed))
    original = base.PairObjective
    def factory(ds, method, device='cpu', n_pde=16384, seed=42):
        if method != 'pinn':raise ValueError('Conditional diagnostic is state-only')
        return ConditionalObjective(ds, fixed, device, n_pde, seed)
    a=argparse.Namespace(data=args.data,key=args.key,out=str(Path(args.out)/args.arm),
        method='pinn',device=args.device,n_pde=16384,budget=args.budget,
        adam=args.adam,outer=args.outer,inner=args.inner)
    try:
        base.PairObjective=factory
        base.train(a)
    finally:
        base.PairObjective=original
    dest=Path(a.out)/'pinn'
    after=torch.load(dest/'model.pt',map_location='cpu',weights_only=True)
    assert tensor_sha(NeuralDiffusivity.from_checkpoint(after['coefficient'])) == expected
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_digest
    for name,value in fixed['state_dict'].items():
        assert torch.equal(value.cpu(),after['coefficient']['state_dict'][name])
    metric=json.loads((dest/'metrics.json').read_text())
    metric.update(variant='dense_conditional_state_v17',arm=args.arm,
        diagnostic='fixed_estimated_coefficient_forward_only',inverse_fit=False,
        coefficient_frozen=True,coefficient_tensor_sha256=expected,
        coefficient_source_sha256=source_digest,coefficient_source_arm=args.arm,
        n_sensors=3969,n_pde=16384,optimizer='ssb',uses_true_D=False)
    (dest/'metrics.json').write_text(json.dumps(metric,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--coefficient',required=True)
    p.add_argument('--key',default=KEY);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--adam',type=int,default=1000);p.add_argument('--outer',type=int,default=10)
    p.add_argument('--inner',type=int,default=400);p.add_argument('--budget',type=float,required=True)
    train(p.parse_args())
