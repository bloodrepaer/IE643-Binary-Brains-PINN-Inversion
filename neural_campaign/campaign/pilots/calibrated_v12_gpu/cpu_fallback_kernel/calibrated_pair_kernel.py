"""Bounded private v12 pair on one shared Kaggle host; partial results kept."""
import json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path


def main():
    os.environ['DEVICE']='cpu'  # Free CPU fallback; numerical protocol unchanged.
    source=Path(os.environ['INPUT_DIR']).resolve() if 'INPUT_DIR' in os.environ else next(Path('/kaggle/input').rglob('protocol.json')).parent
    work=Path(os.environ.get('WORK','/kaggle/working/calibrated_work')).resolve();work.mkdir(parents=True,exist_ok=True)
    for p in source.glob('*.py'):shutil.copy2(p,work/p.name)
    if (source/'data').exists():shutil.copytree(source/'data',work/'data',dirs_exist_ok=True)
    else:
        dest=(work/'data').resolve();dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(source/'data.zip') as z:
            for i in z.infolist():
                if not (dest/i.filename).resolve().is_relative_to(dest):raise ValueError('Unsafe input')
            z.extractall(dest)
    protocol=json.loads((source/'protocol.json').read_text());key=protocol['cases'][0];outcomes=[];started=time.monotonic()
    def pack():
        (work/'job_status.json').write_text(json.dumps(outcomes,indent=2))
        with zipfile.ZipFile(work.parent/'calibrated_pair_results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in work.rglob('*'):
                rel=p.relative_to(work)
                if p.is_file() and ('results' in rel.parts or p.suffix=='.log' or p.name=='job_status.json'):z.write(p,rel)
    for method in ['pinn','adjoint_nn']:
        remaining=13500-(time.monotonic()-started)
        if remaining<120:outcomes.append(dict(method=method,status='skipped_budget'));pack();break
        command=[sys.executable,'-u','calibrated_neural_pair.py','--data','data','--key',key,'--out','results','--method',method,'--device',os.environ.get('DEVICE','cuda') if method=='pinn' else 'cpu','--budget',str(min(6500,remaining-60))]
        if os.environ.get('PACKAGING_SMOKE')=='1':command+=['--adam','0','--outer','0','--n-pde','8']
        status='failed';code=None;log=work/(method+'.log')
        try:
            with log.open('w',encoding='utf-8') as f:r=subprocess.run(command,cwd=work,stdout=f,stderr=subprocess.STDOUT,timeout=min(6900,remaining))
            code=r.returncode;status='exited' if code==0 else 'failed'
        except subprocess.TimeoutExpired:status='external_timeout'
        outcomes.append(dict(method=method,status=status,returncode=code,command=command));pack();print(method,status,log.read_text(encoding='utf-8',errors='replace')[-3000:],flush=True)


if __name__=='__main__':main()
