"""Private bounded v14 higher-density matched neural pairs."""
import json,os,shutil,subprocess,sys,time,zipfile,platform
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
from pathlib import Path
import torch,numpy as np

def main():
 source=Path(os.environ['INPUT_DIR']).resolve() if 'INPUT_DIR' in os.environ else next(Path('/kaggle/input').rglob('dense_calibrated_pair.py')).parent
 work=Path(os.environ.get('WORK','/kaggle/working/dense_calibrated_work')).resolve();work.mkdir(parents=True,exist_ok=True)
 for p in source.glob('*.py'):shutil.copy2(p,work/p.name)
 if (source/'data').exists():shutil.copytree(source/'data',work/'data',dirs_exist_ok=True)
 else:
  dest=(work/'data').resolve();dest.mkdir(exist_ok=True)
  with zipfile.ZipFile(source/'data.zip') as z:
   if z.testzip() is not None:raise ValueError('Corrupt input archive')
   for member in z.infolist():
    if not (dest/member.filename).resolve().is_relative_to(dest):raise ValueError('Unsafe input archive')
   z.extractall(dest)
 protocol=json.loads((source/'protocol.json').read_text());outcomes=[];started=time.monotonic()
 (work/'environment.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cpu_model=platform.processor(),torch_cpu_threads_per_arm=2,cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),indent=2))
 def pack():
  (work/'job_status.json').write_text(json.dumps(outcomes,indent=2))
  with zipfile.ZipFile(work.parent/'dense_calibrated_results.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in work.rglob('*'):
    rel=p.relative_to(work)
    if p.is_file() and ('results' in rel.parts or p.suffix=='.log' or p.name in ['job_status.json','environment.json']):z.write(p,rel)
 for key in protocol['cases']:
  count=next(e['n_sensors'] for e in json.loads((work/'data/manifest.json').read_text()) if e['key']==key)
  for method in ['pinn','adjoint_nn']:
   remaining=27000-(time.monotonic()-started)
   if remaining<120:outcomes.append(dict(key=key,method=method,status='skipped_budget'));pack();return
   device=os.environ.get('DEVICE','cuda') if method=='pinn' else 'cpu'
   command=[sys.executable,'-u','dense_calibrated_pair.py','--data','data','--key',key,'--out','results','--method',method,'--device',device,'--budget',str(min(6500,remaining-60))]
   if os.environ.get('PACKAGING_SMOKE')=='1':command+=['--adam','0','--outer','0','--n-pde','8']
   status='failed';code=None;log=work/(str(count)+'_'+method+'.log')
   try:
    with log.open('w',encoding='utf-8') as f:r=subprocess.run(command,cwd=work,stdout=f,stderr=subprocess.STDOUT,timeout=min(6900,remaining))
    code=r.returncode;status='exited' if code==0 else 'failed'
   except subprocess.TimeoutExpired:status='external_timeout'
   outcomes.append(dict(key=key,method=method,status=status,returncode=code,command=command));pack();print(count,method,status,log.read_text(encoding='utf-8',errors='replace')[-2000:],flush=True)

if __name__=='__main__':main()
