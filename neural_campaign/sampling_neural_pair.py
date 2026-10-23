"""v13 single-factor collocation comparison using unchanged v12 optimizer/loss."""
import argparse
import json
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
from pathlib import Path
import torch
import calibrated_neural_pair as baseline


class SamplingObjective(baseline.PairObjective):
    def __init__(self, *args, sampler='random', **kwargs):
        self.sampler = sampler
        self.draw_index = 0
        self.sampling_seed = kwargs.get('seed', 42)
        super().__init__(*args, **kwargs)

    def batch(self):
        if self.sampler == 'random':
            super().batch()
        else:
            # Independently scrambled power-of-two blocks; identical mixture measure.
            if self.n_pde < 2 or self.n_pde & (self.n_pde-1):
                raise ValueError('Sobol point count must be a power of two')
            engine = torch.quasirandom.SobolEngine(3, scramble=True,
                        seed=self.sampling_seed+self.draw_index)
            q = engine.draw_base2(self.n_pde.bit_length()-1, dtype=torch.float64)
            q[:len(q)//2, 2] = 10**(-4+4*q[:len(q)//2, 2])
            self.col = q.to(self.device)
        self.draw_index += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--key', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--arm', choices=['random_control', 'sobol', 'adjoint_control'], required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--budget', type=float, default=6500)
    ap.add_argument('--adam', type=int, default=1000)
    ap.add_argument('--outer', type=int, default=10)
    ap.add_argument('--inner', type=int, default=400)
    ap.add_argument('--n-pde', type=int, default=4096)
    a = ap.parse_args()
    method = 'adjoint_nn' if a.arm == 'adjoint_control' else 'pinn'
    sampler = 'sobol' if a.arm == 'sobol' else 'random'
    baseline.PairObjective = lambda *args, **kwargs: SamplingObjective(*args, sampler=sampler, **kwargs)
    train_args = argparse.Namespace(data=a.data, key=a.key, out=str(Path(a.out)/a.arm),
                    method=method, device=a.device, budget=a.budget, adam=a.adam,
                    outer=a.outer, inner=a.inner, n_pde=a.n_pde)
    baseline.train(train_args)
    path = Path(train_args.out)/method/'metrics.json'
    metrics = json.loads(path.read_text())
    metrics.update(variant='sampling_'+sampler+'_v13', arm=a.arm,
                   sampler=sampler, parent_protocol='calibrated_hard_physics_v12')
    path.write_text(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
