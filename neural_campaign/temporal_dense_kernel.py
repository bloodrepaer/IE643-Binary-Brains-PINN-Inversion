"""Private bounded five-case v18 GPU worker; source adapter embedded at packaging."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import json,shutil,stat,subprocess,sys,time,zipfile,platform
from pathlib import Path
import torch,numpy as np

ADAPTER_SOURCE = ''
CASE_KEYS = []


def main():
    source=Path(os.environ['INPUT_DIR']).resolve() if 'INPUT_DIR' in os.environ else next(Path('/kaggle/input').rglob('calibrated_neural_pair.py')).parent
    work=Path(os.environ.get('WORK','/kaggle/working/temporal_dense_work')).resolve();work.mkdir(parents=True,exist_ok=False)
    for p in source.glob('*.py'):shutil.copy2(p,work/p.name)
    dest=work/'data';dest.mkdir()
    if (source/'data').is_dir():
        shutil.copytree(source/'data',dest,dirs_exist_ok=True)
    else:
        with zipfile.ZipFile(source/'data.zip') as z:
            assert z.testzip() is None
            for m in z.infolist():
                assert not stat.S_ISLNK(m.external_attr>>16) and '\\' not in m.filename and ':' not in m.filename
                assert (dest/m.filename).resolve().is_relative_to(dest.resolve())
            z.extractall(dest)
    (work/'temporal_density_pair.py').write_text(ADAPTER_SOURCE,encoding='utf-8')
    (work/'environment.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
        cpu_model=platform.processor(),torch_cpu_threads_per_arm=2,
        cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),indent=2))
    started=time.monotonic();outcomes=[]
    def pack():
        (work/'job_status.json').write_text(json.dumps(outcomes,indent=2))
        with zipfile.ZipFile(work.parent/'temporal_dense_results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in work.rglob('*'):
                rel=p.relative_to(work)
                if p.is_file() and ('results' in rel.parts or p.suffix=='.log' or p.name in ['job_status.json','environment.json']):z.write(p,rel)
    for key in CASE_KEYS:
        for arm in ['temporal_pinn','scaled_pde_pinn']:
            remaining=37500-(time.monotonic()-started)
            if remaining<120 or shutil.disk_usage(work).free<128*1024**2:
                outcomes.append(dict(key=key,arm=arm,status='skipped_budget_or_disk_guard'));pack();return
            cmd=[sys.executable,'-u','temporal_density_pair.py','--data','data','--key',key,'--out','results',
                '--arm',arm,'--device',os.environ.get('DEVICE','cuda'),'--budget',str(min(3600,remaining-60))]
            if os.environ.get('PACKAGING_SMOKE')=='1':cmd+=['--adam','0','--outer','0']
            status='failed';code=None;log=work/(key+'__'+arm+'.log')
            try:
                with log.open('w',encoding='utf-8') as f:r=subprocess.run(cmd,cwd=work,stdout=f,stderr=subprocess.STDOUT,timeout=min(3900,remaining))
                code=r.returncode;status='exited' if code==0 else 'failed'
            except subprocess.TimeoutExpired:status='external_timeout'
            outcomes.append(dict(key=key,arm=arm,status=status,returncode=code,command=cmd));pack()
            print(key,arm,status,flush=True)


if __name__=='__main__':main()
