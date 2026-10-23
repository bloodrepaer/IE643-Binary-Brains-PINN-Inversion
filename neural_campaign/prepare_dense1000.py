"""Prepare the explicitly requested v22 subset without reading coefficient truth."""
import csv, hashlib, json, shutil, zipfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

H=Path('campaign/pilots/dense1000_v22_kaggle')
DATA=Path('datasets_v2_dense1000')
PARENT=Path('datasets_v2_dense10000')
KEY='A_smooth__Fo0.1__uniform8__n1000__fine_nested__eta0.05'

def main():
    assert not H.exists() and not DATA.exists(), 'Preserve existing versions'
    H.mkdir(parents=True); DATA.mkdir()
    e=json.loads((PARENT/'manifest.json').read_text())[0]
    with (PARENT/e['files']['sensors_csv']).open(newline='') as f:
        reader=csv.reader(f); header=next(reader); rows=list(reader)
    assert len(rows)==80000
    ids=np.sort(np.random.default_rng(20261023).choice(10000,1000,replace=False))
    selected=[rows[t*10000+int(i)] for t in range(8) for i in ids]
    a=np.asarray(selected,dtype=float).reshape(8,1000,5)
    assert len(np.unique(a[0,:,:2],axis=0))==1000
    (DATA/'sensors').mkdir(); (DATA/'problem').mkdir()
    dest=DATA/'sensors'/f'{KEY}.csv'
    with dest.open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(header);w.writerows(selected)
    for p in (PARENT/'problem').iterdir():
        if p.is_file():shutil.copy2(p,DATA/'problem'/p.name)
    entry={**e,'key':KEY,'n_sensors':1000,'n_readings':8000,'sensor_set':'1000',
        'parent_collection':PARENT.name,'parent_key':e['key'],'seed_subset':20261023,
        'generation_note':'Fixed1000-site subset of10000; supplied CSV strings/noise preserved exactly; no forward regeneration or truth access',
        'files':{k:v for k,v in e['files'].items() if k!='sensors_npz'}}
    entry['files']['sensors_csv']=f'sensors/{KEY}.csv'
    (DATA/'manifest.json').write_text(json.dumps([entry],indent=2))
    supplied=H/'supplied_data'
    for part in ('sensors','problem','initial'):(supplied/part).mkdir(parents=True)
    shutil.copy2(dest,supplied/'sensors'/dest.name)
    old=Path('campaign/pilots/dense10000_v21_gpu/supplied_data')
    for p in (old/'problem').iterdir():shutil.copy2(p,supplied/'problem'/p.name)
    shutil.copy2(old/'initial'/f"{e['key']}.pt",supplied/'initial'/f'{KEY}.pt')
    minimal=dict(key=KEY,problem=e['problem'],n_sensors=1000,eta=.05,
        files=dict(sensors_csv=f'sensors/{KEY}.csv',problem=e['files']['problem']))
    (supplied/'manifest.json').write_text(json.dumps([minimal],indent=2))
    for name in ('dense10000_pair.py','validate_dense10000.py','evaluate_dense10000.py'):
        s=Path(name).read_text().replace('80000','8000').replace('10000','1000').replace('v21','v22')
        s=s.replace('dense1000_v22_gpu','dense1000_v22_kaggle')
        Path(name.replace('10000','1000')).write_text(s)
    manifest={p.relative_to(supplied).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in supplied.rglob('*') if p.is_file()}
    (H/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
    proof=dict(created_utc=datetime.now(timezone.utc).isoformat(),distinct_sensors=1000,readings=8000,
        subset_seed=20261023,parent_indices=ids.tolist(),parent_csv_rows_exact=True,
        coefficient_truth_accessed=False,training_contains_answerkeys=False,
        initial_network='Exact unchanged v21 parent initialization, not refitted/selected',
        limitation='Subset includes fine-grid locations; fixed alpha implies gamma changes with N')
    (H/'dataset_validation.json').write_text(json.dumps(proof,indent=2))
    modules=['adjoint_inversion.py','adjoint_nn.py','calibrated_neural_pair.py','compare_neural.py','integrators.py',
        'neural_diffusivity.py','pinn_papers.py','pinn_zhang_transfer.py','temporal_density_pair.py',
        'matched_temporal_adjoint.py','dense1000_pair.py']
    package=H/'dataset';package.mkdir()
    for name in modules:shutil.copy2(name,package/name)
    shutil.copy2(H/'input_manifest.json',package/'input_manifest.json')
    with zipfile.ZipFile(package/'data.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in supplied.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(supplied).as_posix())
    (package/'dataset-metadata.json').write_text(json.dumps(dict(title='IE643 dense1000 matched inputs',
        id='mihirgune123/ie643-dense1000-matched-data',licenses=[dict(name='CC0-1.0')]),indent=2))
    p=json.loads(Path('campaign/pilots/dense10000_v21_gpu/pilot.json').read_text())
    for k in ('failed_kernel_version','submitted_utc','submitted_v2_utc','loader_repair','last_status_checked','evaluated_utc','scientific_outcome','kernel_sha256'):
        p.pop(k,None)
    p.update(created=proof['created_utc'],status='prepared_not_submitted',cases=[KEY],n_sensors=1000,n_readings=8000,
        data_collection=DATA.name,variants=['dense1000_matched_temporal_v22'],coverage_complete=False,
        input_dataset='mihirgune123/ie643-dense1000-matched-data',input_dataset_version=1,input_dataset_status='not_uploaded',
        kernel_version=1,last_kernel_status='NOT_SUBMITTED',kernel=None,
        fairness='Separate Kaggle sessions: PINNCUDA+CPUcurvature and BECPU,2threads; accuracy only, no equal-host timing claim',
        total_external_s=9000,total_internal_s=8400,
        density_limitation=proof['limitation'],source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in modules})
    (H/'pilot.json').write_text(json.dumps(p,indent=2))
    print('Prepared',KEY,'without answerkey access; package bytes',(package/'data.zip').stat().st_size)

if __name__=='__main__':main()
