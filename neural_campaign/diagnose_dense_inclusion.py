"""Post-fit localization of dense inclusion coefficient errors; no refitting.

The answer key is used ONLY for error scoring of already completed fits.
All method masks share one constant-coefficient forward sensitivity per input.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import json
from pathlib import Path
import numpy as np
import pandas as pd
from adjoint_inversion import Dataset,AdjointSolver,sensitivity_map


def main():
    root=Path('datasets_v2_density_extended')
    entries={e['key']:e for e in json.loads((root/'manifest.json').read_text())}
    jobs=json.loads(Path('campaign/jobs.json').read_text())
    selected=[j for j in jobs if j['collection']==root.name and j['key'].startswith('C_inclusion__') and j['n_sensors']==3969 and j['method']=='adjoint_nn']
    out=Path('campaign/diagnostics/dense_inclusion_localization');out.mkdir(exist_ok=True)
    rows=[]
    for j in selected:
        key=j['key'];ds=Dataset(root,entries[key])
        nn_dir=Path(j['output'])/key/'adjoint_nn__s42'
        nn_metrics=json.loads((nn_dir/'metrics.json').read_text())
        solver=AdjointSolver(ds,n_steps=nn_metrics['n_steps'])
        W=solver.forward(np.full(ds.N**2,nn_metrics['D_const']))[0]
        sensitivity=sensitivity_map(solver,W)
        inner=np.zeros((ds.N,ds.N),dtype=bool);inner[1:-1,1:-1]=True
        high=inner&(sensitivity>=np.median(sensitivity[inner]));low=inner&~high;boundary=~inner
        # Answer-key access begins only after the shared evaluation masks exist.
        with np.load(root/entries[key]['files']['ground_truth']) as z:truth=z['D_inv'].copy()
        gridjob=next(g for g in jobs if g['collection']==root.name and g['key']==key and g['method']=='grid')
        grid_dir=Path(gridjob['output'])/key/'grid__s42'
        fields={}
        for method,folder in [('grid',grid_dir),('adjoint_nn',nn_dir)]:
            with np.load(folder/'D_hat.npz') as z:D=z['D_hat'].copy()
            fields[method]=D
            metrics=json.loads((folder/'metrics.json').read_text())
            energy=(D-truth)**2;total=energy.sum();order=np.argsort(energy.ravel())[::-1]
            relative=float(np.linalg.norm(D-truth)/np.linalg.norm(truth))
            if abs(relative-metrics['rel_L2_D'])>1e-10:raise ValueError('Coefficient score does not reproduce stored metric')
            rows.append(dict(key=key,eta=j['eta'],method=method,rel_L2_D=relative,replay_sigma=metrics['rms_replay_over_sigma'],
                D_min=float(D.min()),D_median=float(np.median(D)),D_p99=float(np.percentile(D,99)),D_max=float(D.max()),
                high_sensitivity_error_energy_fraction=float(energy[high].sum()/total),low_sensitivity_error_energy_fraction=float(energy[low].sum()/total),
                boundary_error_energy_fraction=float(energy[boundary].sum()/total),
                worst_one_percent_error_energy_fraction=float(energy.ravel()[order[:max(1,int(.01*energy.size))]].sum()/total),
                common_high_sensitivity_rel_error=float(np.linalg.norm((D-truth)[high])/np.linalg.norm(truth[high])),
                common_low_sensitivity_rel_error=float(np.linalg.norm((D-truth)[low])/np.linalg.norm(truth[low])),
                coefficient_error_without_boundary=float(np.linalg.norm((D-truth)[inner])/np.linalg.norm(truth[inner]))))
        np.savez_compressed(out/f'maps_eta{j["eta"]:g}.npz',truth=truth,grid=fields['grid'],adjoint_nn=fields['adjoint_nn'],
                            common_sensitivity=sensitivity,high_mask=high,low_mask=low,boundary_mask=boundary)
    table=pd.DataFrame(rows);table.to_csv(out/'error_localization.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(post_fit_evaluation_only=True,training_or_checkpoint_selection=False,
        mask='Same fitted-constant forward field per observation case, median integrated gradient sensitivity on interior; boundary separate',
        answer_key_use='Only after shared mask generation, for already-completed coefficient error localization; never refitting or settings selection',
        caveat='A small sensitivity proxy is not proof of nonidentifiability. Masks from each fitted state differ in original metrics; this evaluation uses common masks for method comparison.'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    with np.load(out/'maps_eta0.01.npz') as z:
        fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
        for ax,key,title in zip(axes[0],['truth','grid','adjoint_nn'],['Coefficient truth (evaluation only)','Grid estimate','Neural-adjoint estimate']):
            im=ax.imshow(z[key].T,origin='lower',extent=[0,1,0,1],norm=LogNorm(.05,5),cmap='viridis');ax.set_title(title)
        fig.colorbar(im,ax=list(axes[0]),label='Diffusivity D (log scale)',shrink=.85)
        s=z['common_sensitivity']
        im=axes[1,0].imshow(np.maximum(s.T,1e-12),origin='lower',extent=[0,1,0,1],norm=LogNorm(max(s.max()*1e-8,1e-12),s.max()),cmap='magma')
        axes[1,0].set_title('Shared constant-forward sensitivity proxy');fig.colorbar(im,ax=axes[1,0],shrink=.8)
        for ax,key,title in zip(axes[1,1:],['grid','adjoint_nn'],['Grid absolute coefficient error','Neural absolute coefficient error']):
            im=ax.imshow(np.maximum(np.abs(z[key]-z['truth']).T,1e-5),origin='lower',extent=[0,1,0,1],norm=LogNorm(1e-4,5),cmap='inferno');ax.set_title(title)
        fig.colorbar(im,ax=list(axes[1,1:]),label='Absolute error (log scale)',shrink=.85)
        for ax in axes.ravel():ax.set(xlabel='x',ylabel='y')
        fig.suptitle('Post-fit inclusion diagnostic: 3969 sensors, 1% noise\nNo refitting; same sensitivity proxy and colour scales for both methods')
        fig.savefig(out/'inclusion_dense_localization.png',dpi=150);plt.close(fig)
    print(table.to_string(index=False))


if __name__=='__main__':main()
