"""Private bounded five-case v18 GPU worker; source adapter embedded at packaging."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import json,shutil,stat,subprocess,sys,time,zipfile,platform
from pathlib import Path
import torch,numpy as np

ADAPTER_SOURCE = '"""v18: five dense inverse cases, temporal consistency and scale-matched control."""\nimport os\nos.environ.setdefault(\'MKL_THREADING_LAYER\',\'SEQUENTIAL\')\nimport argparse\nimport json\nfrom pathlib import Path\nimport numpy as np\nimport torch\nimport hashlib\nfrom torch.nn.utils import vector_to_parameters, parameters_to_vector\nimport calibrated_neural_pair as base\nfrom compare_neural import temperature_value, derivative, residual, field_regularizer\nfrom neural_diffusivity import NeuralDiffusivity\n\nFIELDS=(\'A_smooth\',\'B_gaussian_bump\',\'D_layered\',\'E_grf\',\'F_defects\')\nKEYS=tuple(f\'{f}__Fo0.1__uniform8__n3969__nodes_nested__eta0.05\' for f in FIELDS)\nARMS=(\'temporal_pinn\',\'scaled_pde_pinn\',\'adjoint_nn\')\nSTEP=.01\nPARENT=base.PairObjective\n\n\ndef spatial_operator(temp,coef,q,T):\n    u=temperature_value(temp,coef,q)\n    du=derivative(u,q)\n    D=coef.diffusivity(q[:,:2]).ravel()\n    return T*(derivative(D*du[:,0],q)[:,0]+derivative(D*du[:,1],q)[:,1])\n\n\nclass TemporalObjective(PARENT):\n    def __init__(self,ds,method,device=\'cpu\',n_pde=4096,seed=42,arm=\'temporal_pinn\'):\n        self.arm=arm\n        super().__init__(ds,method,device,n_pde,seed)\n        initial_path=Path(ds.root)/\'initial\'/f\'{ds.key}.pt\'\n        if initial_path.exists():\n            init=torch.load(initial_path,map_location=\'cpu\',weights_only=True)\n            self.coef=NeuralDiffusivity.from_checkpoint(init[\'coefficient\']).to(self.device)\n            self.D_const=init[\'sensor_fitted_constant\']\n            self.temp_config[\'D_base\']=self.D_const\n            self.temp=base.TemperatureNetwork(**self.temp_config).to(self.device)\n            self.params=(list(self.temp.parameters()) if method==\'pinn\' else [])+list(self.coef.parameters())\n            self.initial_weights_sha256=hashlib.sha256(parameters_to_vector(self.coef.parameters()).detach().cpu().numpy().tobytes()).hexdigest()\n            if self.initial_weights_sha256!=init[\'initial_coefficient_sha256\']:raise ValueError(\'Shared initial coefficient mismatch\')\n\n    def batch(self):\n        # Same uniform measure for both PINN arms; both time endpoints in [0,1].\n        col=self.rng.uniform(0,1,(self.n_pde,3));col[:,2]*=1-STEP\n        self.col=self.tensor(col)\n\n    def fg(self,z):\n        if self.method!=\'pinn\':return super().fg(z)\n        vector_to_parameters(self.tensor(z.copy()),self.params)\n        q0=self.col.clone().requires_grad_(True)\n        q1=(self.col+self.tensor([0.,0.,STEP])).requires_grad_(True)\n        data=((temperature_value(self.temp,self.coef,self.xd)-self.yd)/self.scale).square().mean()\n        if self.arm==\'temporal_pinn\':\n            err=temperature_value(self.temp,self.coef,q1)-temperature_value(self.temp,self.coef,q0)-STEP*spatial_operator(self.temp,self.coef,q1,self.ds.T)\n        elif self.arm==\'scaled_pde_pinn\':\n            err=STEP*residual(self.temp,self.coef,q1,self.ds.T)\n        else:raise ValueError(\'Unknown physics objective\')\n        physics=(err/self.scale).square().mean()\n        reg=self.gamma*field_regularizer(self.coef(self.xy).ravel(),self.solver)\n        loss=data+physics+reg\n        gs=torch.autograd.grad(loss,self.params)\n        self.parts=dict(data=float(data.detach()),pde=float(physics.detach()),reg=float(reg.detach()))\n        return float(loss.detach()),torch.cat([g.ravel() for g in gs]).detach().cpu().numpy()\n\n\ndef train(a):\n    if a.key not in KEYS or a.arm not in ARMS:raise ValueError(\'Outside predeclared five dense cases\')\n    method=\'adjoint_nn\' if a.arm==\'adjoint_nn\' else \'pinn\'\n    def factory(ds,method,device=\'cpu\',n_pde=4096,seed=42):\n        return TemporalObjective(ds,method,device,n_pde,seed,arm=a.arm)\n    original=base.PairObjective\n    args=argparse.Namespace(data=a.data,key=a.key,out=str(Path(a.out)/a.key/a.arm),\n        method=method,device=a.device,budget=a.budget,adam=a.adam,outer=a.outer,inner=a.inner,n_pde=4096)\n    try:\n        base.PairObjective=factory\n        base.train(args)\n    finally:base.PairObjective=original\n    p=Path(args.out)/method/\'metrics.json\';m=json.loads(p.read_text())\n    m.update(arm=a.arm,variant=\'temporal_density_v18\',n_sensors=3969,n_pde=4096,\n        optimizer=\'ssb\',time_step_tau=STEP,physics_measure=\'uniform tau0 in [0,0.99], uniform space\',\n        physics_loss=(\'backward_Euler_temperature_defect\' if a.arm==\'temporal_pinn\' else \'h_squared_continuous_residual\' if a.arm==\'scaled_pde_pinn\' else \'hard_discrete_state\'),\n        interpretation=\'Adapted temporal-consistency loss test, not a full reproduction of Thakur architecture/training. All outcomes retained; capped/partial fits are not converged optima or method losses.\')\n    p.write_text(json.dumps(m,indent=2))\n\n\nif __name__==\'__main__\':\n    p=argparse.ArgumentParser(description=__doc__)\n    for x in [\'data\',\'key\',\'out\']:p.add_argument(\'--\'+x,required=True)\n    p.add_argument(\'--arm\',choices=ARMS,required=True);p.add_argument(\'--device\',default=\'cpu\')\n    p.add_argument(\'--budget\',type=float,required=True)\n    p.add_argument(\'--adam\',type=int,default=1000);p.add_argument(\'--outer\',type=int,default=10)\n    p.add_argument(\'--inner\',type=int,default=400)\n    train(p.parse_args())\n'
CASE_KEYS = ['A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.05', 'B_gaussian_bump__Fo0.1__uniform8__n3969__nodes_nested__eta0.05', 'D_layered__Fo0.1__uniform8__n3969__nodes_nested__eta0.05', 'E_grf__Fo0.1__uniform8__n3969__nodes_nested__eta0.05', 'F_defects__Fo0.1__uniform8__n3969__nodes_nested__eta0.05']


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
