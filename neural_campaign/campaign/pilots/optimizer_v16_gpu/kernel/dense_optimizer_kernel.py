ADAPTER_SOURCE = '"""v16: controlled optimizer comparison for both neural inverse methods."""\nimport os\nos.environ.setdefault(\'MKL_THREADING_LAYER\',\'SEQUENTIAL\')\nimport argparse,json\nfrom pathlib import Path\nimport numpy as np\nfrom scipy.optimize import minimize\nimport calibrated_neural_pair as baseline\n\nARMS={\'ssb_pinn\':(\'pinn\',\'ssb\'),\'lbfgsb_pinn\':(\'pinn\',\'lbfgsb\'),\n      \'ssb_adjoint\':(\'adjoint_nn\',\'ssb\'),\'lbfgsb_adjoint\':(\'adjoint_nn\',\'lbfgsb\')}\n\n\ndef lbfgsb_round(fg,x,H,iterations):\n    result=minimize(fg,x,jac=True,method=\'L-BFGS-B\',options=dict(maxiter=iterations,\n        maxcor=50,maxls=30,gtol=1e-12,ftol=0.,maxfun=max(10000,iterations*50)))\n    if result.nit>=iterations:reason=\'iteration_cap\'\n    elif result.success:reason=\'lbfgsb_stopping\'\n    else:reason=\'line_search_or_numerical_stop\'\n    record=dict(iterations=int(result.nit),requested_iterations=iterations,reason=reason,\n        loss=float(result.fun),gradient_max_abs=float(np.max(abs(result.jac))),\n        gradient_l2=float(np.linalg.norm(result.jac)),resets=0,optimizer_success=bool(result.success),\n        optimizer_message=str(result.message),maxcor=50,maxls=30,gtol_infinity=1e-12,ftol=0.,\n        curvature=\'Limited-memory updates reset per outer refresh; common trainer dense identity bookkeeping remains\')\n    return np.asarray(result.x).copy(),H,record\n\n\ndef main():\n    p=argparse.ArgumentParser(description=__doc__)\n    for name in [\'data\',\'key\',\'out\']:p.add_argument(\'--\'+name,required=True)\n    p.add_argument(\'--arm\',choices=ARMS,required=True);p.add_argument(\'--device\',default=\'cpu\')\n    p.add_argument(\'--budget\',type=float,default=9000);p.add_argument(\'--adam\',type=int,default=1000)\n    p.add_argument(\'--outer\',type=int,default=10);p.add_argument(\'--inner\',type=int,default=400)\n    p.add_argument(\'--smoke-n-pde\',type=int)\n    a=p.parse_args()\n    if \'__n3969__\' not in a.key:raise ValueError(\'This predeclared pilot requires3969sensors\')\n    method,optimizer=ARMS[a.arm];n_pde=16384 if method==\'pinn\' else 4096\n    if a.smoke_n_pde is not None:\n        if a.adam>1 or a.outer>1 or a.inner>1:raise ValueError(\'Point override is smoke-only\')\n        n_pde=a.smoke_n_pde\n    if optimizer==\'lbfgsb\':baseline.quasi_newton=lbfgsb_round\n    args=argparse.Namespace(data=a.data,key=a.key,out=str(Path(a.out)/a.arm),method=method,\n        device=a.device,budget=a.budget,adam=a.adam,outer=a.outer,inner=a.inner,n_pde=n_pde)\n    baseline.train(args)\n    path=Path(args.out)/method/\'metrics.json\';m=json.loads(path.read_text())\n    m.update(arm=a.arm,optimizer=optimizer,variant=\'dense_optimizer_v16\',n_sensors=3969,n_pde=n_pde,\n        parent_protocol=\'dense_resolution_v15\',\n        optimizer_limitations=\'Different curvature memory/stopping conventions intrinsic; shared trainer retains dense bookkeeping also forL-BFGS-B. No memory or unqualifiedspeed claim.\')\n    path.write_text(json.dumps(m,indent=2))\n\n\nif __name__==\'__main__\':main()\n'
"""Private bounded v16 dense method-by-optimizer comparison."""
import json,os,shutil,subprocess,sys,time,zipfile,platform,stat
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
from pathlib import Path
import torch,numpy as np

def main():
 source=Path(os.environ['INPUT_DIR']).resolve() if 'INPUT_DIR' in os.environ else next(Path('/kaggle/input').rglob('dense_calibrated_pair.py')).parent
 work=Path(os.environ.get('WORK','/kaggle/working/dense_optimizer_work')).resolve();work.mkdir(parents=True,exist_ok=False)
 for p in source.glob('*.py'):shutil.copy2(p,work/p.name)
 if (source/'data').exists():shutil.copytree(source/'data',work/'data')
 else:
  dest=(work/'data').resolve();dest.mkdir()
  with zipfile.ZipFile(source/'data.zip') as z:
   if z.testzip() is not None:raise ValueError('Corrupt input archive')
   for member in z.infolist():
    if not (dest/member.filename).resolve().is_relative_to(dest) or stat.S_ISLNK(member.external_attr>>16):raise ValueError('Unsafe archive')
   z.extractall(dest)
 (work/'dense_optimizer_pair.py').write_text(ADAPTER_SOURCE,encoding='utf-8')
 key='A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05'
 outcomes=[];started=time.monotonic()
 (work/'environment.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,scipy=__import__('scipy').__version__,cpu_model=platform.processor(),torch_cpu_threads_per_arm=2,cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),indent=2))
 def pack():
  (work/'job_status.json').write_text(json.dumps(outcomes,indent=2))
  with zipfile.ZipFile(work.parent/'dense_optimizer_results.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in work.rglob('*'):
    rel=p.relative_to(work)
    if p.is_file() and ('results' in rel.parts or p.suffix=='.log' or p.name in ['job_status.json','environment.json']):z.write(p,rel)
 for arm in ['ssb_pinn','lbfgsb_pinn','ssb_adjoint','lbfgsb_adjoint']:
  remaining=36800-(time.monotonic()-started)
  if remaining<120:outcomes.append(dict(arm=arm,status='skipped_budget'));pack();return
  device=os.environ.get('DEVICE','cuda') if arm.endswith('_pinn') else 'cpu'
  command=[sys.executable,'-u','dense_optimizer_pair.py','--data','data','--key',key,'--out','results','--arm',arm,'--device',device,'--budget',str(min(9000,remaining-60))]
  if os.environ.get('PACKAGING_SMOKE')=='1':command+=['--adam','0','--outer','0']
  status='failed';code=None;log=work/(arm+'.log')
  try:
   with log.open('w',encoding='utf-8') as f:r=subprocess.run(command,cwd=work,stdout=f,stderr=subprocess.STDOUT,timeout=min(9600,remaining))
   code=r.returncode;status='exited' if code==0 else 'failed'
  except subprocess.TimeoutExpired:status='external_timeout'
  outcomes.append(dict(arm=arm,status=status,returncode=code,command=command));pack();print(arm,status,log.read_text(encoding='utf-8',errors='replace')[-2000:],flush=True)

if __name__=='__main__':main()
