"""v21: extend the fixed smooth reference to 10,000 distinct sensor sites.

Generation may read the retained clean simulation. Training input excludes all
answer keys and clean readings; the original 3969 CSV observations are exact.
"""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np

HOME=Path('campaign/pilots/dense10000_v21_gpu')
DATA=Path('datasets_v2_dense10000')
KEY='A_smooth__Fo0.1__uniform8__n10000__fine_nested__eta0.05'
PARENT='A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05'
SOURCE=Path('datasets_v2_density_extended')

def main():
    assert not (DATA/'manifest.json').exists(),'Existing dataset must be preserved'
    HOME.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
    e=next(x for x in json.loads((SOURCE/'manifest.json').read_text()) if x['key']==PARENT)
    csv_parent=np.loadtxt(SOURCE/e['files']['sensors_csv'],delimiter=',',skiprows=1).reshape(8,3969,5)
    with np.load(SOURCE/e['files']['sensors_npz']) as z:clean_parent=z['u_clean'].reshape(8,3969).copy()
    with np.load(SOURCE/e['files']['ground_truth']) as z:
        x=z['x_data'].copy();y=z['y_data'].copy();times=z['t'].copy();U=z['u'].copy()
    assert len(x)==len(y)==129
    parent_xy=csv_parent[0,:,:2];grid=np.rint(parent_xy*128).astype(int)
    old_ids=grid[:,0]*129+grid[:,1]
    old_set=set(old_ids)
    pool=np.array([i*129+j for i in range(1,128) for j in range(1,128) if i*129+j not in old_set])
    added=np.random.default_rng(20261021).permutation(pool)[:6031]
    ids=np.r_[old_ids,added];xy=np.c_[x[ids//129],y[ids%129]]
    assert len(np.unique(xy,axis=0))==10000
    ti=[int(np.argmin(abs(times-t))) for t in csv_parent[:,0,2]]
    assert np.max(abs(times[ti]-csv_parent[:,0,2]))<1e-8
    clean=U[ti][:,ids//129,ids%129]
    # Original retained snapshots are explicitly stored float32 by the dataset
    # generator; old clean sensor readings retain float64 precision.
    assert np.array_equal(clean[:,:3969],clean_parent.astype(np.float32))
    snapshot_roundoff=float(np.max(abs(clean[:,:3969]-clean_parent)))
    clean=clean.astype(np.float64);clean[:,:3969]=clean_parent
    sigma=float(csv_parent[0,0,4]);noise=np.random.default_rng(20261022).standard_normal((8,6031))*sigma
    obs=np.zeros((8,10000,5));obs[:,:3969]=csv_parent
    obs[:,3969:,0]=xy[3969:,0];obs[:,3969:,1]=xy[3969:,1]
    obs[:,3969:,2]=csv_parent[:,0,2,None];obs[:,3969:,3]=clean[:,3969:]+noise;obs[:,3969:,4]=sigma
    # Match CSV round-trip convention while retaining every supplied old value.
    for part in ('sensors','problem'):(DATA/part).mkdir(exist_ok=True)
    dest=DATA/'sensors'/f'{KEY}.csv'
    with dest.open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['x','y','t','u','sigma'])
        w.writerows([[f'{v:.8g}' for v in row] for row in obs.reshape(-1,5)])
    supplied=np.loadtxt(dest,delimiter=',',skiprows=1).reshape(8,10000,5)
    assert np.array_equal(supplied[:,:3969],csv_parent)
    assert len(np.unique(supplied[0,:,:2],axis=0))==10000
    data={n:supplied[:,:,i].ravel() for i,n in enumerate(('x','y','t','u','sigma'))}
    data.update(u_clean=clean.ravel(),sensor_id=np.tile(ids,8),time_index=np.repeat(ti,10000))
    np.savez_compressed(DATA/'sensors'/f'{KEY}.npz',**data)
    for p in (SOURCE/'problem').glob(e['problem']+'*'):shutil.copy2(p,DATA/'problem'/p.name)
    entry={**e,'key':KEY,'n_sensors':10000,'n_readings':80000,'layout':'fine_nested','sensor_set':'10000',
        'seed_locations':20261021,'seed_added_noise':20261022,'parent_collection':SOURCE.name,'parent_key':PARENT,
        'nested_parent_csv_exact':True,'generation_note':'Retained129-grid clean snapshots;3969oldCSVrows preserved pertime,6031newdistinctfine-grid locations.65-gridinversionunchanged.',
        'files':{**e['files'],'sensors_csv':f'sensors/{KEY}.csv','sensors_npz':f'sensors/{KEY}.npz',
                 'ground_truth':'../datasets_v2_density_extended/'+e['files']['ground_truth']}}
    (DATA/'manifest.json').write_text(json.dumps([entry],indent=2))
    root=HOME/'supplied_data'
    for part in ('sensors','problem','initial'):(root/part).mkdir(parents=True,exist_ok=True)
    shutil.copy2(dest,root/'sensors'/dest.name)
    old_supplied=Path('campaign/pilots/temporal_v18_gpu/supplied_data')
    for name in (Path(e['files']['problem']).name,e['problem']+'__ic.csv'):
        shutil.copy2(old_supplied/'problem'/name,root/'problem'/name)
    shutil.copy2(old_supplied/'initial'/f'{PARENT}.pt',root/'initial'/f'{KEY}.pt')
    minimal=dict(key=KEY,problem=e['problem'],n_sensors=10000,eta=.05,
        files=dict(sensors_csv=f'sensors/{KEY}.csv',problem=e['files']['problem']))
    (root/'manifest.json').write_text(json.dumps([minimal],indent=2))
    manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    (HOME/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
    proof=dict(distinct_sensors=10000,readings=80000,old_observation_rows_exact=31752,added_sensors=6031,
        same_noise_sigma=sigma,source_truth_used_for_generation_only=True,training_input_contains_coefficient_truth=False,
        initial_coefficient='Unchanged v18 observation-fitted initial network, shared with parent and both new arms',
        clean_parent_difference=float(np.max(abs(clean[:,:3969]-clean_parent))),
        original_float32_snapshot_roundoff=snapshot_roundoff,
        added_clean_snapshot_precision='Original stored float32; maximum verified rounding difference at parent nodes recorded')
    (HOME/'dataset_validation.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))

if __name__=='__main__':main()
