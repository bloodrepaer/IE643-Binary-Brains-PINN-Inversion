"""Colab orchestration only: frozen scientific modules remain byte-identical."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import csv,hashlib,json,platform,shutil,subprocess,sys,time,uuid,zipfile
from datetime import datetime,timezone
from pathlib import Path

ARMS=('temporal_pinn','be_adjoint_nn')
TRAIN=dict(adam=1000,outer=10,inner=400,n_pde=4096,seed=42,budget=12600,external_budget=12900)

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def verify_inputs(work):
    work=Path(work)
    for name,digest in json.loads((work/'source_manifest.json').read_text()).items():
        assert Path(name).name==name and sha(work/name)==digest, name
    for name,digest in json.loads((work/'input_manifest.json').read_text()).items():
        p=(work/'data'/name).resolve()
        assert p.is_relative_to((work/'data').resolve()) and sha(p)==digest,name
    return sha(work/'source_manifest.json')+':'+sha(work/'input_manifest.json')

def environment(device):
    import numpy,scipy,torch
    return dict(recorded_utc=now(),python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,
        torch=torch.__version__,platform=platform.platform(),cpu=platform.processor(),cpu_threads=2,
        requested_device=device,cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        resources='PINN observations/residual on CUDA; curvature and BE adjoint on CPU. Accuracy comparison only.')

def completed_pair(case_dir):
    case_dir=Path(case_dir);path=case_dir/'pair_status.json'
    if not (path.exists() and json.loads(path.read_text()).get('validation_complete',False)):return False
    for arm in ARMS:
        status=json.loads((case_dir/(arm+'_status.json')).read_text())
        assert status['status']=='complete'
        for name,digest in status['files'].items():
            target=(case_dir/name).resolve()
            assert target.is_relative_to(case_dir.resolve()) and sha(target)==digest,name
    return True

def next_case(plan,root,phase):
    assert phase in ('density','noise')
    by_id={c['id']:c for c in plan['cases']}
    if phase=='noise':
        assert all(completed_pair(Path(root)/'pairs'/c['id']) for c in plan['cases'] if c['eta']==.05), \
            'Finish and export all three 5%-noise density pairs before the noise phase.'
    for ident in plan['dispatch_order']:
        c=by_id[ident]
        if (c['eta']==.05)!=(phase=='density'):continue
        if not completed_pair(Path(root)/'pairs'/c['id']):return c
    return None

def run_pair(work,root,case,*,require_gpu=True,smoke=False):
    """Skip saved completed arms. Never silently retry a partial/interrupted arm."""
    import torch
    work=Path(work).resolve();root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    identity=verify_inputs(work);key=case['key'];case_dir=root/'pairs'/case['id']
    if smoke:case_dir=root/'SMOKE_DO_NOT_SCORE'/case['id']
    case_dir.mkdir(parents=True,exist_ok=True)
    settings=dict(TRAIN)
    if smoke:settings.update(adam=1,outer=1,inner=1,budget=900,external_budget=1200)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if require_gpu and device!='cuda':
        raise RuntimeError('No GPU allocated. Select a GPU in Runtime > Change runtime type, then reconnect. No fit was launched.')
    config=dict(key=key,source_input_identity=identity,settings=settings,smoke=smoke)
    config_path=case_dir/'run_configuration.json'
    if config_path.exists():assert json.loads(config_path.read_text())==config,'Existing run configuration differs; preserve it.'
    else:atomic(config_path,config)
    # A local exclusive lock prevents overlapping execution in the same runtime.
    lock=work/('active_'+hashlib.sha256(str(case_dir).encode()).hexdigest()[:16]+'.lock')
    with lock.open('x') as f:f.write(json.dumps(dict(pid=os.getpid(),case_dir=str(case_dir),started=now())))
    try:
        for arm in ARMS:
            method='pinn' if arm=='temporal_pinn' else 'adjoint_nn';pointer=case_dir/(arm+'_status.json')
            if pointer.exists():
                saved=json.loads(pointer.read_text())
                if saved.get('status')=='complete':
                    for name,digest in saved['files'].items():assert sha(case_dir/name)==digest,name
                    print('Using saved completed arm:',arm,flush=True);continue
                raise RuntimeError(f'{arm} has a previous partial/interrupted attempt. It is preserved; export it for review instead of silently retraining.')
            attempt_id=uuid.uuid4().hex[:12]
            local=work/'a'/attempt_id;local.mkdir(parents=True,exist_ok=False)
            evidence=case_dir/'attempts'/arm/attempt_id;evidence.mkdir(parents=True,exist_ok=False)
            atomic(pointer,dict(status='running',attempt=attempt_id,started=now(),configuration=config))
            actual_device=device if arm=='temporal_pinn' else 'cpu'
            atomic(evidence/'environment.json',environment(actual_device))
            cmd=[sys.executable,'-u',str(work/'noise_density_1000_7500_pair.py'),
                '--data',str(work/'data'),'--key',key,'--out',str(local/'results'),
                '--arm',arm,'--device',actual_device,'--budget',str(settings['budget']),
                '--adam',str(settings['adam']),'--outer',str(settings['outer']),'--inner',str(settings['inner'])]
            atomic(evidence/'command.json',dict(command=cmd,started=now(),source_input_identity=identity))
            print('\nStarting',key,arm,'on',actual_device,flush=True)
            log_path=local/'training.log';started=time.monotonic();proc=None;code=None;error=None;offset=0
            try:
                with log_path.open('w') as log:
                    proc=subprocess.Popen(cmd,cwd=work,stdout=log,stderr=subprocess.STDOUT)
                    while proc.poll() is None:
                        if time.monotonic()-started>settings['external_budget']:
                            proc.kill();proc.wait();raise TimeoutError('External per-arm budget reached')
                        time.sleep(.2 if smoke else 2)
                        with log_path.open() as stream:
                            stream.seek(offset);chunk=stream.read();offset=stream.tell()
                        if chunk:print(chunk,end='',flush=True)
                    code=proc.returncode
            except BaseException as exc:
                error=f'{type(exc).__name__}: {exc}'
                if proc is not None and proc.poll() is None:proc.kill();proc.wait()
            finally:
                for path in local.rglob('*'):
                    if path.is_file():
                        rel=path.relative_to(local)
                        # Compact evidence paths also work with Windows path-length limits.
                        target=evidence/('checkpoint' if rel.parts[0]=='results' else '')/path.name
                        target.parent.mkdir(parents=True,exist_ok=True)
                        assert not target.exists();shutil.copy2(path,target)
                atomic(evidence/'process_status.json',dict(returncode=code,error=error,finished=now()))
            source=local/'results'/key/arm/method
            if not all((source/n).exists() for n in ('model.pt','D_hat.npz','metrics.json')):
                atomic(pointer,dict(status='interrupted_or_failed_no_checkpoint',attempt=attempt_id,error=error,returncode=code))
                raise RuntimeError(f'{arm} did not save a full checkpoint. Evidence retained; this is not a method loss.')
            m=json.loads((source/'metrics.json').read_text());target=case_dir/'artifacts/results'/key/arm/method
            target.mkdir(parents=True,exist_ok=False);digests={}
            for name in ('model.pt','D_hat.npz','metrics.json'):
                shutil.copy2(source/name,target/name);digests[(target/name).relative_to(case_dir).as_posix()]=sha(target/name)
            atomic(pointer,dict(status=m['status'],attempt=attempt_id,returncode=code,error=error,
                adam_completed=m['adam_completed'],completed_rounds=len(m['optimizer_diagnostics']),files=digests))
            print('Saved to Drive:',arm,m['status'],'Adam',m['adam_completed'],'rounds',len(m['optimizer_diagnostics']),flush=True)
            if m['status']!='complete' or code!=0:
                raise RuntimeError('Partial/failed fit preserved. Export it for review; no automatic retry or method-loss claim.')
        validate_pair(work,case_dir,case)
        print('Pair saved and independently checked. Export results for local coefficient-error scoring.',flush=True)
        return case_dir
    finally:
        lock.unlink(missing_ok=True)  # Removes only this orchestration lock, never results.

def validate_pair(work,case_dir,case):
    work=Path(work);case_dir=Path(case_dir)
    plan=json.loads((work/'colab_plan.json').read_text())
    protocol=dict(cases=[case['key']],variants=['noise_density_temporal_v24_1000_7500'],
        data_collection=str((work/'data').resolve()),initial_coefficient_sha256=plan['initial_coefficient_sha256'])
    atomic(case_dir/'pilot.json',protocol)
    for args,name in [(['evaluate_ic_pilot.py','--pilot-dir',str(case_dir),'--data',str((work/'data').resolve())],'known_physics'),
        (['evaluate_noise_density_1000_7500.py','--pilot-dir',str(case_dir),'--key',case['key'],'--validate-only'],'paired_protocol')]:
        with (case_dir/(name+'_validation.log')).open('w') as log:
            subprocess.run([sys.executable,*[str(work/args[0]),*args[1:]]],cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True)
    integrity=json.loads((case_dir/'pair_integrity.json').read_text());assert integrity['coverage_complete']
    atomic(case_dir/'pair_status.json',dict(validation_complete=True,coefficient_truth_used=False,checked_utc=now(),key=case['key']))

def overview(root):
    rows=[]
    for path in sorted((Path(root)/'pairs').glob('*/artifacts/results/*/*/*/metrics.json')):
        m=json.loads(path.read_text());rows.append(dict(key=m['key'],arm=m['arm'],status=m['status'],n_sensors=m['n_sensors'],eta=m['eta'],
            sensor_fit_sigma=m['rms_unet_over_sigma'],replay_cn_sigma=m['common_cn_replay_sigma'],replay_be_sigma=m['common_be_replay_sigma'],
            adam=m['adam_completed'],rounds=len(m['optimizer_diagnostics'])))
    return rows

def export_results(root):
    root=Path(root);out=root/'exports';out.mkdir(exist_ok=True)
    name='IE643_Colab_results_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]+'.zip'
    path=out/name;manifest={}
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        for f in root.rglob('*'):
            if f.is_file() and 'exports' not in f.relative_to(root).parts:
                rel=f.relative_to(root).as_posix();manifest[rel]=sha(f);z.write(f,rel)
        z.writestr('EXPORT_MANIFEST.json',json.dumps(dict(files=manifest,coefficient_truth_used=False,exported_utc=now()),indent=2))
    return path
