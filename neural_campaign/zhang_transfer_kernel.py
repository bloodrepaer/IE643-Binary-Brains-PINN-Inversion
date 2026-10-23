"""Private bounded runner for the separately versioned paper transfer."""
import json,os,shutil,subprocess,sys,zipfile
from pathlib import Path


def main():
    source=Path(os.environ['INPUT_DIR']).resolve() if 'INPUT_DIR' in os.environ else next(Path('/kaggle/input').rglob('protocol.json')).parent
    work=Path(os.environ.get('WORK','/kaggle/working/zhang_work')).resolve();work.mkdir(parents=True,exist_ok=True)
    for p in source.glob('*.py'):shutil.copy2(p,work/p.name)
    if (source/'data').exists():shutil.copytree(source/'data',work/'data',dirs_exist_ok=True)
    else:
        dest=(work/'data').resolve();dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(source/'data.zip') as z:
            for item in z.infolist():
                if not (dest/item.filename).resolve().is_relative_to(dest):raise ValueError('Unsafe input archive')
            z.extractall(dest)
    protocol=json.loads((source/'protocol.json').read_text());key=protocol['cases'][0]
    command=[sys.executable,'-u','pinn_zhang_transfer.py','--data','data','--key',key,'--out','results','--device',os.environ.get('DEVICE','cuda'),'--budget','6500']
    if os.environ.get('PACKAGING_SMOKE')=='1':command+=['--adam','0','--outer','0','--n-pde','8','--n-bc','8','--n-ic','8']
    log=work/'training.log';status='failed';code=None
    try:
        with log.open('w',encoding='utf-8') as f:
            r=subprocess.run(command,cwd=work,stdout=f,stderr=subprocess.STDOUT,timeout=6900,env={**os.environ,'MKL_THREADING_LAYER':'SEQUENTIAL'})
        code=r.returncode;status='complete' if code==0 else 'failed'
    except subprocess.TimeoutExpired:status='external_timeout'
    finally:
        (work/'job_status.json').write_text(json.dumps(dict(status=status,returncode=code,command=command)))
        with zipfile.ZipFile(work.parent/'zhang_transfer_results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in work.rglob('*'):
                rel=p.relative_to(work)
                if p.is_file() and ('results' in rel.parts or p.name in ['training.log','job_status.json']):z.write(p,rel)
        print(log.read_text(encoding='utf-8',errors='replace')[-6000:],flush=True)
    if status!='complete':raise RuntimeError('Paper-transfer failed; logs retained')


if __name__=='__main__':main()
