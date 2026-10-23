"""Private bounded v13 three-arm controlled sampling pilot."""
SAMPLING_SOURCE = '"""v13 single-factor collocation comparison using unchanged v12 optimizer/loss."""\nimport argparse\nimport json\nimport os\nos.environ.setdefault(\'MKL_THREADING_LAYER\', \'SEQUENTIAL\')\nfrom pathlib import Path\nimport torch\nimport calibrated_neural_pair as baseline\n\n\nclass SamplingObjective(baseline.PairObjective):\n    def __init__(self, *args, sampler=\'random\', **kwargs):\n        self.sampler = sampler\n        self.draw_index = 0\n        self.sampling_seed = kwargs.get(\'seed\', 42)\n        super().__init__(*args, **kwargs)\n\n    def batch(self):\n        if self.sampler == \'random\':\n            super().batch()\n        else:\n            # Independently scrambled power-of-two blocks; identical mixture measure.\n            if self.n_pde < 2 or self.n_pde & (self.n_pde-1):\n                raise ValueError(\'Sobol point count must be a power of two\')\n            engine = torch.quasirandom.SobolEngine(3, scramble=True,\n                        seed=self.sampling_seed+self.draw_index)\n            q = engine.draw_base2(self.n_pde.bit_length()-1, dtype=torch.float64)\n            q[:len(q)//2, 2] = 10**(-4+4*q[:len(q)//2, 2])\n            self.col = q.to(self.device)\n        self.draw_index += 1\n\n\ndef main():\n    ap = argparse.ArgumentParser(description=__doc__)\n    ap.add_argument(\'--data\', required=True)\n    ap.add_argument(\'--key\', required=True)\n    ap.add_argument(\'--out\', required=True)\n    ap.add_argument(\'--arm\', choices=[\'random_control\', \'sobol\', \'adjoint_control\'], required=True)\n    ap.add_argument(\'--device\', default=\'cpu\')\n    ap.add_argument(\'--budget\', type=float, default=6500)\n    ap.add_argument(\'--adam\', type=int, default=1000)\n    ap.add_argument(\'--outer\', type=int, default=10)\n    ap.add_argument(\'--inner\', type=int, default=400)\n    ap.add_argument(\'--n-pde\', type=int, default=4096)\n    a = ap.parse_args()\n    method = \'adjoint_nn\' if a.arm == \'adjoint_control\' else \'pinn\'\n    sampler = \'sobol\' if a.arm == \'sobol\' else \'random\'\n    baseline.PairObjective = lambda *args, **kwargs: SamplingObjective(*args, sampler=sampler, **kwargs)\n    train_args = argparse.Namespace(data=a.data, key=a.key, out=str(Path(a.out)/a.arm),\n                    method=method, device=a.device, budget=a.budget, adam=a.adam,\n                    outer=a.outer, inner=a.inner, n_pde=a.n_pde)\n    baseline.train(train_args)\n    path = Path(train_args.out)/method/\'metrics.json\'\n    metrics = json.loads(path.read_text())\n    metrics.update(variant=\'sampling_\'+sampler+\'_v13\', arm=a.arm,\n                   sampler=sampler, parent_protocol=\'calibrated_hard_physics_v12\')\n    path.write_text(json.dumps(metrics, indent=2))\n\n\nif __name__ == \'__main__\':\n    main()\n'

import json,os,shutil,subprocess,sys,time,zipfile,platform
from pathlib import Path
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import torch,numpy as np

def main():
 source=Path(os.environ['INPUT_DIR']).resolve() if 'INPUT_DIR' in os.environ else next(Path('/kaggle/input').rglob('calibrated_neural_pair.py')).parent
 work=Path(os.environ.get('WORK','/kaggle/working/sampling_work')).resolve();work.mkdir(parents=True,exist_ok=True)
 for p in source.glob('*.py'):shutil.copy2(p,work/p.name)
 if not (source/'data').exists():raise ValueError('Expected existing expanded private v12 supplied-input dataset')
 shutil.copytree(source/'data',work/'data',dirs_exist_ok=True)
 (work/'sampling_neural_pair.py').write_text(SAMPLING_SOURCE,encoding='utf-8')
 key=json.loads((source/'protocol.json').read_text())['cases'][0]
 (work/'environment.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cpu_model=platform.processor(),torch_cpu_threads_per_arm=2,cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),indent=2))
 outcomes=[];started=time.monotonic()
 def pack():
  (work/'job_status.json').write_text(json.dumps(outcomes,indent=2))
  with zipfile.ZipFile(work.parent/'sampling_pair_results.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in work.rglob('*'):
    rel=p.relative_to(work)
    if p.is_file() and ('results' in rel.parts or p.suffix=='.log' or p.name in ['job_status.json','environment.json']):z.write(p,rel)
 for arm in ['random_control','sobol','adjoint_control']:
  remaining=20000-(time.monotonic()-started)
  if remaining<120:outcomes.append(dict(arm=arm,status='skipped_budget'));pack();break
  device=os.environ.get('DEVICE','cuda') if arm!='adjoint_control' else 'cpu'
  command=[sys.executable,'-u','sampling_neural_pair.py','--data','data','--key',key,'--out','results','--arm',arm,'--device',device,'--budget',str(min(6500,remaining-60))]
  if os.environ.get('PACKAGING_SMOKE')=='1':command+=['--adam','0','--outer','0','--n-pde','8']
  status='failed';code=None;log=work/(arm+'.log')
  try:
   with log.open('w',encoding='utf-8') as f:r=subprocess.run(command,cwd=work,stdout=f,stderr=subprocess.STDOUT,timeout=min(6900,remaining))
   code=r.returncode;status='exited' if code==0 else 'failed'
  except subprocess.TimeoutExpired:status='external_timeout'
  outcomes.append(dict(arm=arm,status=status,returncode=code,command=command));pack();print(arm,status,log.read_text(encoding='utf-8',errors='replace')[-2000:],flush=True)

if __name__=='__main__':main()
