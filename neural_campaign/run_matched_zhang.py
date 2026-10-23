"""One bounded, locked matched neural-adjoint research arm; no broad jobs."""
import json,subprocess,sys,time
from pathlib import Path
import campaign as C


def main():
    h=C.ROOT/'campaign/pilots/zhang_v11_gpu';state=h/'matched_adjoint_worker.json'
    with C.FileLock(h/'matched_adjoint.lock'):
        protocol=json.loads((h/'matched_adjoint_protocol.json').read_text())
        if (h/'matched_adjoint/metrics.json').exists():raise RuntimeError('Retained fit exists; do not duplicate')
        command=protocol['command'];record=dict(pid=__import__('os').getpid(),started=C.now(),status='running',command=command)
        C.atomic_json(state,record)
        try:
            with (h/'matched_adjoint_training.log').open('x',encoding='utf-8') as f:
                result=subprocess.run(command,cwd=C.ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=6900)
            record.update(status='exited',returncode=result.returncode)
        except subprocess.TimeoutExpired:record.update(status='external_timeout')
        except Exception as ex:record.update(status='failed',error=repr(ex))
        record['finished']=C.now();C.atomic_json(state,record)


if __name__=='__main__':main()
