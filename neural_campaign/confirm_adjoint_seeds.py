"""Bounded fixed-observation seed confirmation, separate from the broad plan.

Reuse canonical seed42 fits and run seeds43/44 on smooth, bump and defect
fields with unchanged frozen neural-adjoint settings. No true-D model tuning.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import pandas as pd
import campaign as C


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget-seconds',type=int,default=7200)
    parser.add_argument('--group',choices=['canonical','density'],default='canonical')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    home=C.HOME/('replicates/adjoint_seed_confirmation' if args.group=='canonical' else 'replicates/adjoint_density_seed_confirmation');home.mkdir(parents=True,exist_ok=True)
    all_jobs=json.loads((C.HOME/'jobs.json').read_text());jobs=[];original=[]
    cases=[('datasets_v2_rk4_zero',tier,tier+'__Fo0.1__uniform8__n50__random__eta0.05') for tier in ['A_smooth','B_gaussian_bump','F_defects']] if args.group=='canonical' else [
        ('datasets_v2_density_extended',tier,tier+f'__Fo0.1__uniform8__n3969__nodes_nested__eta{eta}')
        for tier,eta in [('B_gaussian_bump','0.05'),('D_layered','0.01'),('E_grf','0.05')]]
    suffix='confirmation_v1' if args.group=='canonical' else 'density_confirmation_v1'
    for collection,tier,key in cases:
        parent=next(j for j in all_jobs if j['collection']==collection and j['key']==key and j['method']=='adjoint_nn')
        if not C.completed(parent):raise ValueError('Baseline seed42 must be complete before confirmation')
        original.append(parent)
        for seed in [43,44]:
            job=dict(parent,seed=seed,id=hashlib.sha256((key+'|adjoint_nn|'+str(seed)+'|'+suffix).encode()).hexdigest()[:16])
            job['output']=str((home/'runs'/job['id']).relative_to(C.ROOT));jobs.append(job)
    if (home/'protocol.json').exists():
        existing=json.loads((home/'protocol.json').read_text())
        if existing['jobs']!=jobs:raise ValueError('Refuse to overwrite a different seed-confirmation protocol')
    else:
        C.atomic_json(home/'protocol.json',dict(created=C.now(),scope='Fixed-data initialization-seed confirmation; not new sensor/noise realizations and not part of the1510 seed42 broad fits',group=args.group,source_code='campaign/code/48b49b1f17a0',reuse_seed42=[j['id'] for j in original],new_seeds=[43,44],fields=[c[1] for c in cases],settings_unchanged=True,jobs=jobs,hardware=platform.platform(),training_threads=1,concurrent_main_adjoint_workers=2,
            rationale='Confirm dense B/D reported advantages and E negative comparison at prespecified full density/noise; retain all seeds, no parameter tuning or best-seed selection.' if args.group=='density' else 'Canonical fixed-data initialization confirmation'))
    if args.prepare_only:
        print('Validated protocol',home,'new jobs',len(jobs),flush=True);return
    started=time.monotonic();outcomes=[]
    with C.FileLock(home/'worker.lock'):
        for job in jobs:
            if C.completed(job):continue
            previous=home/(job['id']+'.json')
            if previous.exists() and json.loads(previous.read_text()).get('status') in ['failed','timeout']:continue
            remaining=args.budget_seconds-(time.monotonic()-started)
            if remaining<120 or shutil.disk_usage(C.ROOT).free<128*1024**2:break
            command=C.command_for(job)
            C.atomic_json(home/'worker.json',dict(pid=os.getpid(),status='running',updated=C.now(),active=job['id'],finished=outcomes))
            log=home/(job['id']+'.log');record=dict(id=job['id'],seed=job['seed'],key=job['key'],started=C.now(),command=command,
                sensors_sha256=hashlib.sha256((C.ROOT/job['collection']/ 'sensors'/(job['key']+'.csv')).read_bytes()).hexdigest())
            try:
                with log.open('a',encoding='utf-8') as stream:
                    result=subprocess.run(command,cwd=C.ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=min(1800,remaining),env={**os.environ,'MKL_THREADING_LAYER':'SEQUENTIAL'})
                record.update(status='complete' if result.returncode==0 and C.completed(job) else 'failed',returncode=result.returncode)
            except subprocess.TimeoutExpired:record['status']='timeout'
            except Exception as error:record.update(status='failed',error=repr(error))
            record['finished']=C.now();C.atomic_json(previous,record);outcomes.append(record)
        rows=[]
        for job in original+jobs:
            if not C.completed(job):continue
            m=json.loads(C.metric_path(job).read_text())
            rows.append(dict(key=job['key'],seed=job['seed'],rel_L2_D=m['rel_L2_D'],replay_sigma=m['rms_replay_over_sigma'],flag=m['flag'],wall_s=m['wall_s'],metrics=str(C.metric_path(job).relative_to(C.ROOT))))
        pd.DataFrame(rows).to_csv(home/'summary.csv',index=False)
        C.atomic_json(home/'worker.json',dict(pid=os.getpid(),status='finished',updated=C.now(),finished=outcomes,completed_new=sum(C.completed(j) for j in jobs),planned_new=len(jobs)))
    print('Saved seed confirmation summary; new fits:',sum(C.completed(j) for j in jobs),flush=True)


if __name__=='__main__':main()
