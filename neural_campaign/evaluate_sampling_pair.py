"""Verify all v13 arms before answer-key scoring; never select a best seed."""
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from adjoint_inversion import Dataset, AdjointSolver
from adjoint_nn import grid_coordinates
from neural_diffusivity import NeuralDiffusivity


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', default='campaign/pilots/sampling_v13_gpu')
    home = Path(parser.parse_args().pilot_dir)
    protocol = json.loads((home/'pilot.json').read_text())
    validation = json.loads((home/'validation.json').read_text())
    if not validation['coverage_complete']:
        raise ValueError('Both PINN variants require independent constraint checks')
    root = Path(protocol['data_collection'])
    entries = {e['key']: e for e in json.loads((root/'manifest.json').read_text())}
    arms = {}
    checks = []
    for path in (home/'artifacts').rglob('metrics.json'):
        m = json.loads(path.read_text())
        arm = m['arm']
        if arm in arms:
            raise ValueError('Duplicate arm')
        saved = torch.load(path.with_name('model.pt'), map_location='cpu', weights_only=True)
        coef = NeuralDiffusivity.from_checkpoint(saved['coefficient'])
        ds = Dataset(root, entries[m['key']])
        with torch.no_grad():
            D = coef.diffusivity(grid_coordinates(ds.N)).numpy().reshape(ds.N, ds.N)
        with np.load(path.with_name('D_hat.npz')) as z:
            error = float(np.max(abs(D-z['D_hat'])))
        if not np.isfinite(D).all() or D.min() < .001-1e-12 or D.max() > 10+1e-12 or error > 1e-12:
            raise ValueError('Invalid bounded coefficient/checkpoint')
        solver = AdjointSolver(ds, n_steps=m['n_steps'])
        replay = float(np.sqrt(np.mean((solver.predict(solver.forward(D.ravel())[0])-ds.u)**2))/ds.sigma)
        if abs(replay-m['rms_replay_over_sigma']) > 1e-9:
            raise ValueError('Common replay mismatch')
        arms[arm] = (m, D, entries[m['key']])
        checks.append(dict(arm=arm, checkpoint_difference=error, replay_sigma=replay))
    if set(arms) != {'random_control', 'sobol', 'adjoint_control'}:
        raise ValueError('All three retained arm outputs required')
    for field in ['key', 'initial_coefficient_sha256', 'gamma', 'seed', 'alpha', 'scale']:
        if len({m[field] for m, _, _ in arms.values()}) != 1:
            raise ValueError('Shared protocol mismatch: '+field)
    (home/'pair_integrity.json').write_text(json.dumps(dict(results=checks,
        coefficient_truth_used=False, shared_initialization_objective_verified=True), indent=2))
    rows = []
    # Answer keys are accessed only after all reconstruction and protocol checks.
    for arm, (m, D, entry) in arms.items():
        with np.load(root/entry['files']['ground_truth']) as z:
            truth = z['D_inv'].copy()
        rows.append(dict(arm=arm, method=m['method'], key=m['key'],
            rel_L2_D=float(np.linalg.norm(D-truth)/np.linalg.norm(truth)),
            replay_sigma=m['rms_replay_over_sigma'], sensor_fit_sigma=m['rms_unet_over_sigma'],
            status=m['status'], wall_s=m['wall_s'], adam_completed=m['adam_completed'],
            quasi_newton_iterations=sum(r['iterations'] for r in m['optimizer_diagnostics']),
            capped_rounds=sum(r['reason']=='iteration_cap' for r in m['optimizer_diagnostics']),
            function_evaluations=m['function_evaluations'], gamma=m['gamma']))
    table = pd.DataFrame(rows)
    table.to_csv(home/'primary_matched_neural_comparison.csv', index=False)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
