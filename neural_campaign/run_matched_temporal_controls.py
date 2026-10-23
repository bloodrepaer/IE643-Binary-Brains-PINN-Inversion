"""Bounded five-field BE neural controls for the human-requested fairness audit."""
import datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path

def main():
    root=Path.cwd();h=root/'campaign/pilots/matched_temporal_v20_local'
    p=json.loads((h/'pilot.json').read_text())
    assert json.loads((h/'preflight.json').read_text())['complete']
    assert json.loads((h/'validation_smoke.json').read_text())['coverage_complete']
    assert not (h/'worker.json').exists(),'Refuse duplicate worker'
    snapshot=h/'execution';snapshot.mkdir(exist_ok=False)
    names=['matched_temporal_adjoint.py','temporal_density_pair.py','calibrated_neural_pair.py',
        'adjoint_inversion.py','adjoint_nn.py','compare_neural.py','integrators.py',
        'neural_diffusivity.py','pinn_papers.py','pinn_zhang_transfer.py']
    hashes={}
    for name in names:
        shutil.copy2(root/name,snapshot/name)
        hashes[name]=hashlib.sha256((snapshot/name).read_bytes()).hexdigest()
    (h/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    logs=h/'logs';logs.mkdir(exist_ok=False);finished=[]
    def save(status,active=None):
        temp=h/'worker.tmp';temp.write_text(json.dumps(dict(pid=os.getpid(),status=status,active=active,
            finished=finished,updated=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
        temp.replace(h/'worker.json')
    for key in p['cases']:
        dest=h/'results'/key/'adjoint_nn'
        if dest.exists():raise RuntimeError('Prepared/completed work already exists: '+str(dest))
        if shutil.disk_usage(h).free<128*1024**2:save('disk_guard');return
        assert all(hashlib.sha256((snapshot/n).read_bytes()).hexdigest()==v for n,v in hashes.items())
        save('running',key)
        cmd=[sys.executable,'-u',str(snapshot/'matched_temporal_adjoint.py'),'--key',key,
             '--out',str(h/'results'),'--budget','7200','--adam','1000','--outer','10','--inner','400']
        try:
            with (logs/(key+'.log')).open('w') as log:
                r=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=7800,
                    env={**os.environ,'MKL_THREADING_LAYER':'SEQUENTIAL'})
            record=dict(key=key,returncode=r.returncode)
        except subprocess.TimeoutExpired:record=dict(key=key,status='external_timeout')
        if (dest/'metrics.json').exists():record['fit_status']=json.loads((dest/'metrics.json').read_text())['status']
        finished.append(record);save('running')
    save('validating')
    with (h/'evaluation.log').open('w') as log:
        r=subprocess.run([sys.executable,'-u','evaluate_matched_temporal.py'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
    save('finished' if r.returncode==0 else 'finished_requires_evaluation_review')

if __name__=='__main__':main()
