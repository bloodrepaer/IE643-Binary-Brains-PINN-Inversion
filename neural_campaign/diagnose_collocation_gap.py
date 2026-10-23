"""Post-fit training/held-out PDE checks without coefficient truth or refitting."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from compare_neural import TemperatureNetwork, residual
from neural_diffusivity import NeuralDiffusivity
import campaign as C


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dirs',nargs='+',required=True)
    parser.add_argument('--aggregate',type=Path,default=C.HOME/'diagnostics/pilot_collocation_gap.csv')
    args=parser.parse_args()
    torch.set_num_threads(1)
    rng=np.random.default_rng(20261023)
    heldout=rng.uniform(0,1,(16384,3))
    heldout[:len(heldout)//2,2]=10**rng.uniform(-4,0,len(heldout)//2)
    rows=[]
    for name in args.pilot_dirs:
        home=Path(name)
        reports=[]
        for path in sorted((home/'artifacts').rglob('model.pt')):
            saved=torch.load(path,map_location='cpu',weights_only=True)
            # Matched pilot archives also contain adjoint coefficients without a state net.
            if 'temperature_config' not in saved:
                continue
            temperature=TemperatureNetwork(**saved['temperature_config'])
            temperature.load_state_dict(saved['temperature_state_dict'])
            coefficient=NeuralDiffusivity.from_checkpoint(saved['coefficient'])
            metrics=json.loads(path.with_name('metrics.json').read_text())
            def score(points):
                values=[]
                for offset in range(0,len(points),256):
                    x=torch.tensor(np.asarray(points[offset:offset+256]),dtype=torch.float64,requires_grad=True)
                    values.append(residual(temperature,coefficient,x,saved['T']).detach().numpy())
                return np.concatenate(values)
            validation=score(heldout)
            energy=validation**2
            order=np.argsort(energy)[::-1]
            artifact=str(path.relative_to(home/'artifacts'))
            row=dict(pilot=home.name,key=metrics['key'],artifact=artifact,
                     arm=metrics.get('arm'),variant=metrics.get('variant'),heldout_points=len(heldout),
                     heldout_rms=float(np.sqrt(energy.mean())),heldout_max=float(np.abs(validation).max()),
                     top_one_percent_energy=float(energy[order[:len(order)//100]].sum()/energy.sum()),
                     replay_sigma=metrics['rms_replay_over_sigma'],
                     sensor_fit_sigma=metrics['rms_unet_over_sigma'])
            retained=saved.get('final_training_points')
            if retained is not None:
                points=retained['collocation'] if isinstance(retained,dict) else retained
                train=score(points.numpy())
                row.update(training_points=len(train),training_rms=float(np.sqrt(np.mean(train**2))))
                row['heldout_to_training_rms']=row['heldout_rms']/row['training_rms'] if row['training_rms'] else None
            table=pd.DataFrame(heldout[order[:64]],columns=['x','y','tau'])
            table['residual']=validation[order[:64]]
            tag=hashlib.sha256(artifact.encode()).hexdigest()[:12]
            table.to_csv(home/f'collocation_gap_{tag}_outliers.csv',index=False)
            report=dict(updated=C.now(),uses_true_D=False,
                validation_seed=20261023,sampling='half uniform time, half log time; uniform space',
                caveat='Compare ratios only with the declared training measure. Adaptive pilots change that measure; v12 uses the same half-uniform/half-log-time measure as this independent probe. Finite-sample ratios do not establish causation or convergence.',
                results=row,optimizer_diagnostics=metrics.get('optimizer_diagnostics',[]))
            C.atomic_json(home/f'collocation_gap_{tag}.json',report)
            reports.append(report)
            rows.append(row)
            print(json.dumps(row),flush=True)
        if len(reports)==1:
            C.atomic_json(home/'collocation_gap.json',reports[0])
            table.to_csv(home/'collocation_gap_outliers.csv',index=False)
        elif reports:
            C.atomic_json(home/'collocation_gap.json',dict(updated=C.now(),uses_true_D=False,
                validation_seed=20261023,results=[r['results'] for r in reports],
                scope='All retained PINN arms, identical independent probe; per-artifact reports and outliers preserve every arm.'))
    aggregate=args.aggregate
    fresh=pd.DataFrame(rows)
    if aggregate.exists() and rows:
        previous=pd.read_csv(aggregate)
        previous=previous[~previous['pilot'].isin(fresh['pilot'])]
        fresh=pd.concat([previous,fresh],ignore_index=True)
    if rows:
        fresh.to_csv(aggregate,index=False)


if __name__=='__main__':main()
