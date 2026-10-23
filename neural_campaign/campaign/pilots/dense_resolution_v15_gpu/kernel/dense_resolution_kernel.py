ADAPTER_SOURCE = '"""v15: change only PINN collocation count at3969 observed sensors."""\nimport os\nos.environ.setdefault(\'MKL_THREADING_LAYER\',\'SEQUENTIAL\')\nimport argparse,json\nfrom pathlib import Path\nimport calibrated_neural_pair as baseline\n\nARMS={\'pde4096_control\':(\'pinn\',4096),\'pde16384\':(\'pinn\',16384),\'adjoint_control\':(\'adjoint_nn\',4096)}\n\n\ndef main():\n    p=argparse.ArgumentParser(description=__doc__)\n    for name in [\'data\',\'key\',\'out\']:p.add_argument(\'--\'+name,required=True)\n    p.add_argument(\'--arm\',choices=ARMS,required=True)\n    p.add_argument(\'--device\',default=\'cpu\');p.add_argument(\'--budget\',type=float,default=13000)\n    p.add_argument(\'--adam\',type=int,default=1000);p.add_argument(\'--outer\',type=int,default=10)\n    p.add_argument(\'--inner\',type=int,default=400)\n    p.add_argument(\'--smoke-n-pde\',type=int,default=None)\n    a=p.parse_args();method,n=ARMS[a.arm]\n    if a.smoke_n_pde is not None:\n        if a.adam>1 or a.outer>1 or a.inner>1:raise ValueError(\'Point override is smoke-only\')\n        n=a.smoke_n_pde\n    args=argparse.Namespace(data=a.data,key=a.key,out=str(Path(a.out)/a.arm),method=method,\n        device=a.device,budget=a.budget,adam=a.adam,outer=a.outer,inner=a.inner,n_pde=n)\n    baseline.train(args)\n    path=Path(args.out)/method/\'metrics.json\';m=json.loads(path.read_text())\n    m.update(arm=a.arm,variant=\'dense_resolution_v15\',n_sensors=3969,n_pde=n,\n             parent_protocol=\'calibrated_hard_physics_v12\',sampler=\'random\',\n             research_scope=\'Point-count-only PINN comparison; same full observed data and mixed-time measure.\')\n    path.write_text(json.dumps(m,indent=2))\n\n\nif __name__==\'__main__\':main()\n'
"""Private bounded v15 dense-observation collocation-count test."""
import json,os,shutil,subprocess,sys,time,zipfile,platform,stat
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
from pathlib import Path
import torch,numpy as np

def main():
 source=Path(os.environ['INPUT_DIR']).resolve() if 'INPUT_DIR' in os.environ else next(Path('/kaggle/input').rglob('dense_calibrated_pair.py')).parent
 work=Path(os.environ.get('WORK','/kaggle/working/dense_resolution_work')).resolve();work.mkdir(parents=True,exist_ok=False)
 for p in source.glob('*.py'):shutil.copy2(p,work/p.name)
 if (source/'data').exists():shutil.copytree(source/'data',work/'data')
 else:
  dest=(work/'data').resolve();dest.mkdir()
  with zipfile.ZipFile(source/'data.zip') as z:
   if z.testzip() is not None:raise ValueError('Corrupt input archive')
   for member in z.infolist():
    if not (dest/member.filename).resolve().is_relative_to(dest) or stat.S_ISLNK(member.external_attr>>16):raise ValueError('Unsafe archive')
   z.extractall(dest)
 (work/'dense_resolution_pair.py').write_text(ADAPTER_SOURCE,encoding='utf-8')
 key='A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05'
 outcomes=[];started=time.monotonic()
 (work/'environment.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cpu_model=platform.processor(),torch_cpu_threads_per_arm=2,cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),indent=2))
 def pack():
  (work/'job_status.json').write_text(json.dumps(outcomes,indent=2))
  with zipfile.ZipFile(work.parent/'dense_resolution_results.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in work.rglob('*'):
    rel=p.relative_to(work)
    if p.is_file() and ('results' in rel.parts or p.suffix=='.log' or p.name in ['job_status.json','environment.json']):z.write(p,rel)
 for arm in ['pde4096_control','pde16384','adjoint_control']:
  remaining=40000-(time.monotonic()-started)
  if remaining<120:outcomes.append(dict(arm=arm,status='skipped_budget'));pack();return
  device=os.environ.get('DEVICE','cuda') if arm!='adjoint_control' else 'cpu'
  command=[sys.executable,'-u','dense_resolution_pair.py','--data','data','--key',key,'--out','results','--arm',arm,'--device',device,'--budget',str(min(13000,remaining-60))]
  if os.environ.get('PACKAGING_SMOKE')=='1':command+=['--adam','0','--outer','0']
  status='failed';code=None;log=work/(arm+'.log')
  try:
   with log.open('w',encoding='utf-8') as f:r=subprocess.run(command,cwd=work,stdout=f,stderr=subprocess.STDOUT,timeout=min(13600,remaining))
   code=r.returncode;status='exited' if code==0 else 'failed'
  except subprocess.TimeoutExpired:status='external_timeout'
  outcomes.append(dict(arm=arm,status=status,returncode=code,command=command));pack();print(arm,status,log.read_text(encoding='utf-8',errors='replace')[-2000:],flush=True)

if __name__=='__main__':main()
