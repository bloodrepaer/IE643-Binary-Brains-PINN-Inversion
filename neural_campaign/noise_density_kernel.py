"""Private v23 single-case pair, sequential full CPU fits with bounded runtime."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import hashlib,json,platform,shutil,stat,subprocess,sys,time,zipfile
from pathlib import Path
import torch,numpy as np

KEY='__CASE_KEY__'
def main():
    source=Path(os.environ['INPUT_DIR']).resolve() if 'INPUT_DIR' in os.environ else next(Path('/kaggle/input').rglob('noise_density_pair.py')).parent
    work=Path(os.environ.get('WORK','/kaggle/working/noise_density_work')).resolve();work.mkdir(parents=True,exist_ok=False)
    code_manifest=json.loads((source/'source_manifest.json').read_text())
    for name,digest in code_manifest.items():
        assert Path(name).name==name and name.endswith('.py')
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
        shutil.copy2(source/name,work/name)
    data=work/'data'
    if (source/'data').is_dir():shutil.copytree(source/'data',data)
    else:
        data.mkdir()
        with zipfile.ZipFile(source/'data.zip') as z:
            assert z.testzip() is None;seen=set()
            for m in z.infolist():
                name=m.filename
                assert name.casefold() not in seen and '\\' not in name and ':' not in name
                assert (data/name).resolve().is_relative_to(data.resolve()) and not stat.S_ISLNK(m.external_attr>>16)
                seen.add(name.casefold())
            z.extractall(data)
    names=set()
    for name,digest in json.loads((source/'input_manifest.json').read_text()).items():
        portable=name.replace('\\','/');path=(data/portable).resolve()
        assert portable.casefold() not in names and ':' not in portable and path.is_relative_to(data.resolve())
        names.add(portable.casefold());assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    assert KEY in {e['key'] for e in json.loads((data/'manifest.json').read_text())}
    torch.set_num_threads(2)
    (work/'environment.json').write_text(json.dumps(dict(key=KEY,python=sys.version,torch=torch.__version__,numpy=np.__version__,
        cpu=platform.processor(),cpu_threads=2,cuda_used=False,resources='Both full arms sequential CPU2; no paid compute'),indent=2))
    start=time.monotonic();outcomes=[]
    def pack():
        (work/'job_status.json').write_text(json.dumps(outcomes,indent=2))
        with zipfile.ZipFile(work.parent/'noise_density_results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in work.rglob('*'):
                if p.is_file() and ('results' in p.relative_to(work).parts or p.suffix=='.log' or p.name in ('environment.json','job_status.json')):
                    z.write(p,p.relative_to(work).as_posix())
    for arm in ('temporal_pinn','be_adjoint_nn'):
        left=25800-(time.monotonic()-start)
        if left<120 or shutil.disk_usage(work).free<128*1024**2:
            outcomes.append(dict(arm=arm,status='skipped_budget_or_disk'));pack();return
        cmd=[sys.executable,'-u','noise_density_pair.py','--data','data','--key',KEY,'--out','results',
            '--arm',arm,'--device','cpu','--budget',str(min(12600,left-60))]
        if os.environ.get('PACKAGING_SMOKE')=='1':cmd+=['--adam','0','--outer','0']
        try:
            with (work/(arm+'.log')).open('w') as log:
                r=subprocess.run(cmd,cwd=work,stdout=log,stderr=subprocess.STDOUT,timeout=min(12900,left))
            record=dict(arm=arm,status='exited' if r.returncode==0 else 'failed',returncode=r.returncode,command=cmd)
        except subprocess.TimeoutExpired:record=dict(arm=arm,status='external_timeout',command=cmd)
        outcomes.append(record);pack();print(record,flush=True)

if __name__=='__main__':main()
