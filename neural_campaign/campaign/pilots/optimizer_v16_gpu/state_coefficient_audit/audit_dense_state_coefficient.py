"""All retained v16 state/coefficient combinations; no fit or answer-key access."""
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from adjoint_inversion import Dataset, AdjointSolver
from adjoint_nn import grid_coordinates
from compare_neural import TemperatureNetwork, temperature_value, residual, field_regularizer
from neural_diffusivity import NeuralDiffusivity


def main():
    torch.set_num_threads(1)
    home = Path('campaign/pilots/optimizer_v16_gpu')
    out = home/'state_coefficient_audit'
    out.mkdir(exist_ok=False)
    arms = ['ssb_pinn', 'lbfgsb_pinn', 'ssb_adjoint', 'lbfgsb_adjoint']
    states = ['ssb_pinn', 'lbfgsb_pinn']
    protocol = dict(states=states, coefficients=arms, combinations=8,
        uses_true_D=False, refitting=False, accepted_optimizer_steps=0,
        collocation='Each retained state uses its own unchanged final training cloud for every coefficient.',
        scope='Evaluate every Cartesian combination, preserving the local-D coupled base. No best combination selected.',
        limitations='Frozen-state compatibility test, not conditional state representability, a new inverse solution, '
                    'an optimization trajectory, a converged optimum or a unique-cause test.')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    source=Path(__file__)
    (out/source.name).write_bytes(source.read_bytes())
    saved, metrics, paths = {}, {}, {}
    for arm in arms:
        paths[arm] = next((home/'artifacts/results'/arm).rglob('model.pt'))
        saved[arm] = torch.load(paths[arm], map_location='cpu', weights_only=True)
        metrics[arm] = json.loads(paths[arm].with_name('metrics.json').read_text())
    digests = {a:hashlib.sha256(p.read_bytes()).hexdigest() for a,p in paths.items()}
    root=Path('datasets_v2_density_extended')
    entries={e['key']:e for e in json.loads((root/'manifest.json').read_text())}
    m=metrics[states[0]]
    assert len({v['key'] for v in metrics.values()}) == 1
    ds=Dataset(root,entries[m['key']]);solver=AdjointSolver(ds,n_steps=m['n_steps'])
    xy=grid_coordinates(ds.N)
    xd=torch.tensor(np.c_[ds.x,ds.y,ds.t/ds.T],dtype=torch.float64)
    y=torch.tensor(ds.u,dtype=torch.float64)
    rows=[]
    for coefficient_arm in arms:
        coef=NeuralDiffusivity.from_checkpoint(saved[coefficient_arm]['coefficient'])
        with torch.no_grad():D=coef.diffusivity(xy).numpy().ravel()
        replay=solver.predict(solver.forward(D)[0])
        replay_sigma=float(np.sqrt(np.mean((replay-ds.u)**2))/ds.sigma)
        for state_arm in states:
            sm=metrics[state_arm]
            temp=TemperatureNetwork(**saved[state_arm]['temperature_config'])
            temp.load_state_dict(saved[state_arm]['temperature_state_dict'])
            q=saved[state_arm]['final_training_points'].clone().requires_grad_(True)
            with torch.no_grad():
                obs=temperature_value(temp,coef,xd)
                data=float(((obs-y)/sm['scale']).square().mean())
                reg=float(sm['gamma']*field_regularizer(coef(xy).ravel(),solver))
            pde=float((residual(temp,coef,q,ds.T)/sm['scale']).square().mean().detach())
            parts=dict(data=data,pde=pde,reg=reg)
            if state_arm==coefficient_arm:
                assert max(abs(parts[k]-sm['final_parts'][k]) for k in parts)<1e-12
            row=dict(state_arm=state_arm,coefficient_arm=coefficient_arm,
                objective_parts=parts,objective=sum(parts.values()),
                state_sensor_fit_sigma=float(np.sqrt(np.mean((obs.numpy()-ds.u)**2))/ds.sigma),
                coefficient_common_CN_replay_sigma=replay_sigma,
                state_common_CN_sensor_gap_sigma=float(np.sqrt(np.mean((obs.numpy()-replay)**2))/ds.sigma))
            rows.append(row)
            print(json.dumps(row),flush=True)
    assert digests=={a:hashlib.sha256(p.read_bytes()).hexdigest() for a,p in paths.items()}
    (out/'results.json').write_text(json.dumps(dict(protocol=protocol,checkpoint_sha256=digests,
        checkpoints_unchanged=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),results=rows),indent=2))


if __name__=='__main__':main()
