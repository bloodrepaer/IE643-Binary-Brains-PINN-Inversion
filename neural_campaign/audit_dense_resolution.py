"""Predeclared nested quadrature-size audit of the retained3969-sensor PINN."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from compare_neural import TemperatureNetwork,residual
from neural_diffusivity import NeuralDiffusivity


def main():
    torch.set_num_threads(1)
    home=Path('campaign/pilots/dense_calibrated_v14_gpu')
    model=home/'artifacts/results/n3969/pinn/model.pt'
    out=home/'resolution_audit';out.mkdir(exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('Preserve completed audit; do not repeat')
    protocol=dict(uses_true_D=False,training=False,checkpoint_selection=False,
        checkpoint_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        sizes=[4096,16384,65536],seeds=[20261110,20261111,20261112,20261113],
        measure='uniform space; half uniform time, half log time1e-4..1',
        purpose='Describe point-count integration variability, not inverse accuracy or an optimal count.',
        caveat='Sizes are nested within each independent replicate; no exact integral or best-seed selection.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    saved=torch.load(model,map_location='cpu',weights_only=True)
    temp=TemperatureNetwork(**saved['temperature_config']);temp.load_state_dict(saved['temperature_state_dict'])
    coef=NeuralDiffusivity.from_checkpoint(saved['coefficient'])
    rows=[];started=time.monotonic();largest=max(protocol['sizes'])
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed)
        # Interleaving preserves exactly half of each time stratum in every prefix.
        q=rng.uniform(0,1,(largest//2,2,3));q[:,0,2]=10**rng.uniform(-4,0,largest//2)
        q=q.reshape(-1,3);values=[]
        for offset in range(0,largest,256):
            if time.monotonic()-started>480:raise TimeoutError('Read-only audit budget; retain completed replicates')
            x=torch.tensor(q[offset:offset+256],dtype=torch.float64,requires_grad=True)
            values.append(residual(temp,coef,x,saved['T']).detach().numpy())
        r=np.concatenate(values)
        for n in protocol['sizes']:
            e=r[:n]**2
            strata=e.reshape(-1,2)
            # Independent strata, equal mixture weights. Descriptive finite-sample SE.
            se=float(np.sqrt(.25*sum(np.var(strata[:,k],ddof=1)/len(strata) for k in range(2))))
            rows.append(dict(seed=seed,points=n,energy=float(e.mean()),energy_standard_error=se,
                max_abs=float(np.max(abs(r[:n]))),
                top_one_percent_energy=float(np.sort(e)[-max(1,n//100):].sum()/e.sum())))
        assert hashlib.sha256(model.read_bytes()).hexdigest()==protocol['checkpoint_sha256']
        (out/'partial_replicates.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(rows[-3:]),flush=True)
    summary=[]
    for n in protocol['sizes']:
        energies=np.array([r['energy'] for r in rows if r['points']==n])
        summary.append(dict(points=n,mean_energy=float(energies.mean()),
            replicate_std=float(energies.std(ddof=1)),replicate_cv=float(energies.std(ddof=1)/energies.mean())))
    (out/'results.json').write_text(json.dumps(dict(protocol=protocol,results=summary,
        replicates=rows,wall_s=time.monotonic()-started,checkpoint_unchanged=True),indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
