"""Fixed-checkpoint integration audit; no training, answer keys or selection."""
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from compare_neural import TemperatureNetwork, residual
from neural_diffusivity import NeuralDiffusivity


def main():
    torch.set_num_threads(1)
    home = Path('campaign/pilots/calibrated_v12_gpu')
    model = home/'artifacts/results/pinn/model.pt'
    output = home/'quadrature_audit'
    output.mkdir(exist_ok=True)
    protocol = dict(uses_true_D=False, training=False, checkpoint_selection=False,
                    checkpoint_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    measure='uniform space; half uniform time, half log time 1e-4..1',
                    reference_points=65536, replicate_points=4096,
                    seeds=[20261030, 20261031, 20261032, 20261033],
                    reference_seed=20261101,
                    purpose='Compare integration variability of random and scrambled Sobol points on one retained checkpoint. Four replicates are descriptive, not a statistical guarantee or method win.')
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    saved = torch.load(model, map_location='cpu', weights_only=True)
    temp = TemperatureNetwork(**saved['temperature_config'])
    temp.load_state_dict(saved['temperature_state_dict'])
    coef = NeuralDiffusivity.from_checkpoint(saved['coefficient'])

    def points(n, seed, kind):
        if kind == 'random':
            rng = np.random.default_rng(seed)
            q = rng.uniform(0, 1, (n, 3))
            q[:n//2, 2] = 10**rng.uniform(-4, 0, n//2)
        else:
            q = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed).draw_base2(
                int(np.log2(n)), dtype=torch.float64).numpy()
            q[:n//2, 2] = 10**(-4+4*q[:n//2, 2])
        assert np.isfinite(q).all() and q.min() >= 0 and q.max() <= 1
        return q

    def evaluate(q):
        values = []
        for offset in range(0, len(q), 256):
            x = torch.tensor(q[offset:offset+256], dtype=torch.float64, requires_grad=True)
            values.append(residual(temp, coef, x, saved['T']).detach().numpy())
        return np.concatenate(values)

    started = time.monotonic()
    reference_points = points(protocol['reference_points'], protocol['reference_seed'], 'random')
    reference = evaluate(reference_points)
    reference_energy = float(np.mean(reference**2))
    # A noisy integration reference is explicitly not a known exact integral.
    block_means = np.mean(reference.reshape(16, -1)**2, axis=1)
    bins = []
    for low, high in [(0, .001), (.001, .01), (.01, .1), (.1, 1)]:
        mask = (reference_points[:, 2] >= low) & (reference_points[:, 2] < high)
        bins.append(dict(tau_low=low, tau_high=high, points=int(mask.sum()),
                         fraction_of_residual_energy=float(np.sum(reference[mask]**2)/np.sum(reference**2))))
    rows = []
    for kind in ['random', 'sobol']:
        for seed in protocol['seeds']:
            r = evaluate(points(protocol['replicate_points'], seed, kind))
            row = dict(sampler=kind, seed=seed, points=protocol['replicate_points'],
                       mean_residual_squared=float(np.mean(r**2)), max_abs=float(np.max(abs(r))))
            row['relative_difference_from_MC_reference'] = row['mean_residual_squared']/reference_energy-1
            rows.append(row)
            print(json.dumps(row), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output/'replicates.csv', index=False)
    summary = []
    for kind, group in table.groupby('sampler'):
        summary.append(dict(sampler=kind, replicates=len(group),
                            mean_residual_squared=float(group.mean_residual_squared.mean()),
                            std_of_mean_residual_squared=float(group.mean_residual_squared.std(ddof=1)),
                            mean_absolute_relative_difference_from_MC_reference=float(group.relative_difference_from_MC_reference.abs().mean())))
    assert hashlib.sha256(model.read_bytes()).hexdigest() == protocol['checkpoint_sha256']
    result = dict(protocol=protocol, reference_mean_residual_squared=reference_energy,
                  reference_block_mean_std=float(block_means.std(ddof=1)),
                  reference_time_bin_energy=bins, results=summary,
                  wall_seconds=time.monotonic()-started, checkpoint_unchanged=True)
    (output/'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
