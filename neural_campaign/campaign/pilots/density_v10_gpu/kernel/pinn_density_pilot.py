"""Bounded matched nested50/full3969 sensor PINN diagnostic; no answer keys."""
import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def main():
    import torch
    device=os.environ.get('DEVICE','cuda')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required')
    source=Path(os.environ['INPUT_DIR']).resolve() if 'INPUT_DIR' in os.environ else next(Path('/kaggle/input').rglob('protocol.json')).parent
    work=Path(os.environ.get('WORK','/kaggle/working/density_work')).resolve();work.mkdir(parents=True,exist_ok=True)
    for p in source.glob('*.py'):shutil.copy2(p,work/p.name)
    if (source/'data/manifest.json').exists():shutil.copytree(source/'data',work/'data',dirs_exist_ok=True)
    else:
        dest=(work/'data').resolve();dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(source/'data.zip') as z:
            for item in z.infolist():
                if not (dest/item.filename).resolve().is_relative_to(dest):raise ValueError('Unsafe archive')
            z.extractall(dest)
        hits=list(dest.rglob('manifest.json'))
        if len(hits)!=1:raise ValueError('Expected one data manifest')
        if hits[0].parent!=dest:raise ValueError('Unexpected data archive layout')
    protocol=json.loads((source/'protocol.json').read_text());started=time.monotonic();outcomes=[]
    def pack():
        (work/'job_status.json').write_text(json.dumps(outcomes,indent=2)+'\n')
        with zipfile.ZipFile(work.parent/'pinn_density_results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in work.rglob('*'):
                relative=p.relative_to(work)
                if p.is_file() and ('results' in relative.parts or p.suffix=='.log' or p.name=='job_status.json'):z.write(p,relative)
    for key in protocol['cases'][:int(os.environ.get('CASE_LIMIT','2'))]:
        remaining=float(os.environ.get('BUDGET_SECONDS','10000'))-(time.monotonic()-started)
        if remaining<120:outcomes.append(dict(key=key,status='skipped_budget'));pack();break
        cmd=[sys.executable,'-u','compare_neural.py','--data','data','--keys',key,'--out','results/'+key,
             '--methods','pinn','--state-model','gaussian_ic_bc_locald','--device',device,'--seed','42','--width','32','--depth','3',
             '--adam-steps',os.environ.get('STEPS','15000'),'--lbfgs-steps',os.environ.get('LBFGS','750'),
             '--n-collocation','2048','--n-boundary','256','--data-batch','4096','--physics-weight','1000','--condition-weight','1000',
             '--pinn-alpha','1','--pde-sampler','mixed','--threads','2']
        log=work/(key+'.log');print('Starting',key,flush=True)
        status='failed';code=None
        try:
            with log.open('w',encoding='utf-8') as f:result=subprocess.run(cmd,cwd=work,stdout=f,stderr=subprocess.STDOUT,timeout=min(6500,remaining),env={**os.environ,'MKL_THREADING_LAYER':'SEQUENTIAL'})
            code=result.returncode;status='complete' if code==0 else 'failed'
        except subprocess.TimeoutExpired:status='timeout'
        outcomes.append(dict(key=key,status=status,returncode=code,command=cmd));pack()
        print(status,key,log.read_text(encoding='utf-8',errors='replace')[-1200:],flush=True)
        if status!='complete':raise RuntimeError('Density diagnostic failed; retained archive/logs must be inspected')


if __name__=='__main__':main()
