"""Private bounded v22 1000sensor pair, two arms on one Kaggle host."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import hashlib,json,platform,shutil,stat,subprocess,sys,time,zipfile
from pathlib import Path
import torch,numpy as np

ARM='temporal_pinn'
KEY='A_smooth__Fo0.1__uniform8__n1000__fine_nested__eta0.05'
def main():
    source=Path(os.environ['INPUT_DIR']).resolve() if 'INPUT_DIR' in os.environ else next(Path('/kaggle/input').rglob('dense1000_pair.py')).parent
    work=Path(os.environ.get('WORK','/kaggle/working/dense1000_work')).resolve();work.mkdir(parents=True,exist_ok=False)
    for p in source.glob('*.py'):shutil.copy2(p,work/p.name)
    data=work/'data'
    if (source/'data').is_dir():shutil.copytree(source/'data',data)
    else:
        data.mkdir()
        with zipfile.ZipFile(source/'data.zip') as z:
            assert z.testzip() is None
            for m in z.infolist():
                assert (data/m.filename).resolve().is_relative_to(data.resolve())
                assert not stat.S_ISLNK(m.external_attr>>16) and '\\' not in m.filename and ':' not in m.filename
            z.extractall(data)
    manifest=json.loads((source/'input_manifest.json').read_text())
    normalized=set()
    for name,digest in manifest.items():
        portable=name.replace('\\','/')
        path=(data/portable).resolve()
        assert portable not in normalized and ':' not in portable and path.is_relative_to(data.resolve())
        normalized.add(portable)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    (work/'environment.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
        cpu=platform.processor(),cpu_threads=2,cuda=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),indent=2))
    start=time.monotonic();outcomes=[]
    def pack():
        (work/'job_status.json').write_text(json.dumps(outcomes,indent=2))
        with zipfile.ZipFile(work.parent/('dense1000_'+ARM+'_results.zip'),'w',zipfile.ZIP_DEFLATED) as z:
            for p in work.rglob('*'):
                if p.is_file() and ('results' in p.relative_to(work).parts or p.suffix=='.log' or p.name in ('environment.json','job_status.json')):
                    z.write(p,p.relative_to(work))
    for arm in (ARM,):
        left=4500-(time.monotonic()-start)
        if left<120 or shutil.disk_usage(work).free<128*1024**2:
            outcomes.append(dict(arm=arm,status='skipped_budget_or_disk'));pack();return
        device=os.environ.get('DEVICE','cuda') if arm=='temporal_pinn' else 'cpu'
        cmd=[sys.executable,'-u','dense1000_pair.py','--data','data','--key',KEY,'--out','results',
            '--arm',arm,'--device',device,'--budget',str(min(4200,left-60))]
        if os.environ.get('PACKAGING_SMOKE')=='1':cmd+=['--adam','0','--outer','0']
        try:
            with (work/(arm+'.log')).open('w') as log:
                r=subprocess.run(cmd,cwd=work,stdout=log,stderr=subprocess.STDOUT,timeout=min(4400,left))
            record=dict(arm=arm,status='exited' if r.returncode==0 else 'failed',returncode=r.returncode,command=cmd)
        except subprocess.TimeoutExpired:record=dict(arm=arm,status='external_timeout',command=cmd)
        outcomes.append(record);pack();print(record,flush=True)

if __name__=='__main__':main()
