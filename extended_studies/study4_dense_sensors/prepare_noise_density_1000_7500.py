"""v24_1000_7500: fixed nested observations/noise, clean concentration for generation only."""
import csv, hashlib, json, shutil, zipfile
from pathlib import Path
import numpy as np

H=Path('campaign/pilots/noise_density_v24_1000_7500')
DATA=Path('datasets_v2_noise_density_v24_1000_7500')

def main():
    assert not DATA.exists() and not (H/'supplied_data').exists()
    parent=Path('datasets_v2_dense10000');e=json.loads((parent/'manifest.json').read_text())[0]
    assert e['integrator']=='rk4'
    with (parent/e['files']['sensors_csv']).open(newline='') as f:
        reader=csv.reader(f);header=next(reader);original=list(reader)
    obs=np.array(original,dtype=float).reshape(8,10000,5)
    # Only clean concentration is read. Coefficient answer keys are not opened.
    with np.load(parent/e['files']['sensors_npz']) as z:clean=z['u_clean'].reshape(8,10000).copy()
    small=np.array(json.loads(Path('campaign/pilots/dense1000_v22_kaggle/dataset_validation.json').read_text())['parent_indices'])
    remaining=np.setdiff1d(np.arange(10000),small)
    middle=np.sort(np.r_[small,np.random.default_rng(20261024).permutation(remaining)[:2969]])
    permutation=np.random.default_rng(20261024).permutation(remaining)
    sets={n:np.sort(np.r_[small,permutation[:n-1000]]) for n in (1000,3000,5000,7500)}
    assert len(middle)==3969 and set(small)<=set(middle)
    for p in ('sensors','problem'):(DATA/p).mkdir(parents=True)
    supplied=H/'supplied_data'
    for p in ('sensors','problem','initial'):(supplied/p).mkdir(parents=True)
    old=Path('campaign/pilots/dense10000_v21_gpu/supplied_data')
    for p in (old/'problem').iterdir():
        shutil.copy2(p,DATA/'problem'/p.name);shutil.copy2(p,supplied/'problem'/p.name)
    entries=[];minimal=[];cases=[];proof=[]
    for n in (3000,5000,7500):
        for eta in (.01,.05,.1):
            if eta==.05 and n in (1000,10000):continue
            ids=sets[n];key=f'A_smooth__Fo0.1__uniform8__n{n}__fine_nested__eta{eta:g}'
            selected=[];ratio=eta/.05;sigma=float(obs[0,0,4])*ratio
            for t in range(8):
                for i in ids:
                    row=original[t*10000+int(i)].copy()
                    if eta!=.05:
                        row[3]=f'{clean[t,i]+ratio*(obs[t,i,3]-clean[t,i]):.8g}'
                        row[4]=f'{sigma:.8g}'
                    selected.append(row)
            path=DATA/'sensors'/f'{key}.csv'
            with path.open('w',newline='') as f:
                w=csv.writer(f,lineterminator='\n');w.writerow(header);w.writerows(selected)
            a=np.array(selected,dtype=float).reshape(8,n,5)
            assert len(np.unique(a[0,:,:2],axis=0))==n
            assert np.array_equal(a[:,:,:3],obs[:,ids,:3])
            delta=(a[:,:,3]-clean[:,ids])-ratio*(obs[:,ids,3]-clean[:,ids])
            rounding=float(np.max(abs(delta)));assert rounding<5.1e-9
            entry={**e,'key':key,'n_sensors':n,'n_readings':8*n,'sensor_set':str(n),'eta':eta,
                'sigma':float(a[0,0,4]),'snr_db':float(20*np.log10(1/eta)),
                'family':'NOISE_DENSITY_V24_1000_7500','parent_collection':parent.name,'parent_key':e['key'],
                'seed_subset':20261023,'seed_middle_extension':20261024,
                'files':{k:v for k,v in e['files'].items() if k!='sensors_npz'}}
            entry['files']['sensors_csv']=f'sensors/{key}.csv'
            entries.append(entry)
            shutil.copy2(path,supplied/'sensors'/path.name)
            shutil.copy2(old/'initial'/f"{e['key']}.pt",supplied/'initial'/f'{key}.pt')
            minimal.append(dict(key=key,problem=e['problem'],n_sensors=n,eta=eta,
                files=dict(sensors_csv=entry['files']['sensors_csv'],problem=e['files']['problem'])))
            cases.append(dict(key=key,n_sensors=n,eta=eta,status='queued',arms=['temporal_pinn','be_adjoint_nn']))
            proof.append(dict(key=key,rows=8*n,distinct_sites=n,sigma=float(a[0,0,4]),
                parent5percent_preserved_exact=eta==.05,noise_rescale_roundoff=rounding))
    assert len(cases)==9
    (DATA/'manifest.json').write_text(json.dumps(entries,indent=2))
    (supplied/'manifest.json').write_text(json.dumps(minimal,indent=2))
    manifest={p.relative_to(supplied).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in supplied.rglob('*') if p.is_file()}
    (H/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
    (H/'dataset_validation.json').write_text(json.dumps(dict(generator='retained RK4 fine-grid data; no forward regeneration',
        clean_concentration_generation_only=True,coefficient_truth_opened=False,
        nested=True,subset_indices={str(n):ids.tolist() for n,ids in sets.items()},results=proof),indent=2))
    protocol=dict(version=24,status='prepared_not_dispatched',cases=cases,data_collection=DATA.name,
        initial_coefficient_sha256=json.loads(Path('campaign/pilots/dense10000_v21_gpu/pilot.json').read_text())['initial_coefficient_sha256'],
        adam=1000,outer=10,inner=400,n_pde=4096,time_step_tau=.01,seed=42,
        per_arm_internal_s=12600,per_arm_external_s=12900,total_pair_internal_s=25800,
        hardware='Both arms sequentialCPU2threads on same freeKaggleCPU host; cross-cell/reference hosts differ',
        budget_rationale='GPUquota1265s cannot cover full fit. CPUcap3x4200 gives fulliteration opportunity without truncating; caps/timeouts retained. Pairloop25800s bounded, no paidcompute.',
        reference_pilots=['dense1000_v22_kaggle','noise_density_v23'])
    (H/'pilot.json').write_text(json.dumps(protocol,indent=2))
    print('Prepared nine fresh pairs; nested/commonnoise proof PASS, coefficient answer keys not opened')

if __name__=='__main__':main()
