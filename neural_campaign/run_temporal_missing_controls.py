"""Finish only the two never-dispatched v18 controls, preserving prior partials."""
import datetime,json,os,shutil,subprocess,sys,time
from pathlib import Path

def main():
    h=Path('campaign/pilots/temporal_v18_gpu').resolve()
    out=h/'missing_controls_20261020';out.mkdir(exist_ok=False)
    keys=[f'{f}__Fo0.1__uniform8__n3969__nodes_nested__eta0.05' for f in ('E_grf','F_defects')]
    start=time.monotonic();finished=[]
    def save(status,active=None):
        (out/'worker.json').write_text(json.dumps(dict(pid=os.getpid(),status=status,active=active,
            finished=finished,updated=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
    for key in keys:
        dest=h/'local_results'/key/'adjoint_nn'/'adjoint_nn'
        if dest.exists(): raise RuntimeError('Refuse to duplicate prepared or existing output: '+str(dest))
        if shutil.disk_usage(h).free<128*1024**2: raise RuntimeError('128MiB disk guard')
        remaining=15000-(time.monotonic()-start)
        if remaining<120: break
        save('running',key)
        cmd=[sys.executable,'-u',str(h/'execution/temporal_density_pair.py'),'--data',str(h/'supplied_data'),
             '--key',key,'--out',str(h/'local_results'),'--arm','adjoint_nn','--device','cpu',
             '--budget',str(min(7200,remaining-60))]
        try:
            with (out/(key+'.log')).open('w') as log:
                r=subprocess.run(cmd,cwd=h/'execution',stdout=log,stderr=subprocess.STDOUT,
                    timeout=min(7800,remaining),env={**os.environ,'MKL_THREADING_LAYER':'SEQUENTIAL'})
            record=dict(key=key,returncode=r.returncode)
        except subprocess.TimeoutExpired: record=dict(key=key,status='external_timeout')
        if (dest/'metrics.json').exists(): record['fit_status']=json.loads((dest/'metrics.json').read_text())['status']
        finished.append(record);save('running')
    save('finished')

if __name__=='__main__': main()
