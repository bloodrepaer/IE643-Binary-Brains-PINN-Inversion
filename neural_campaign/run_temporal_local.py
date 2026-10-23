"""Bounded fresh matched neural-adjoint controls for v18; never broad queue reload."""
import concurrent.futures,json,os,subprocess,sys,time,threading,datetime,shutil
from pathlib import Path
from temporal_density_pair import KEYS


def main():
    h=Path('campaign/pilots/temporal_v18_gpu').resolve()
    logs=h/'local_logs';logs.mkdir(exist_ok=False)
    started=time.monotonic();active=set();finished=[];lock=threading.Lock()
    def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
    def save(status='running'):
        (h/'local_worker.json').write_text(json.dumps(dict(pid=os.getpid(),started_utc=begin,
            updated=now(),status=status,active=sorted(active),finished=finished,budget_seconds=21600),indent=2))
    begin=now();save()
    def run(key):
        remaining=21600-(time.monotonic()-started)
        if remaining<120:return dict(key=key,status='skipped_budget')
        if shutil.disk_usage(h).free<128*1024**2:return dict(key=key,status='skipped_disk_guard')
        with lock:active.add(key);save()
        cmd=[sys.executable,'-u',str(h/'execution/temporal_density_pair.py'),'--data',str(h/'supplied_data'),
            '--key',key,'--out',str(h/'local_results'),'--arm','adjoint_nn','--device','cpu','--budget',str(min(7200,remaining-60))]
        status='failed';code=None
        try:
            with (logs/(key+'.log')).open('w') as f:
                r=subprocess.run(cmd,cwd=h/'execution',stdout=f,stderr=subprocess.STDOUT,
                    timeout=min(7800,remaining),env={**os.environ,'MKL_THREADING_LAYER':'SEQUENTIAL'})
            code=r.returncode;status='exited' if code==0 else 'failed'
        except subprocess.TimeoutExpired:status='external_timeout'
        result=dict(key=key,status=status,returncode=code,finished=now())
        with lock:active.remove(key);finished.append(result);save()
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        list(pool.map(run,KEYS))
    save('finished')


if __name__=='__main__':main()
