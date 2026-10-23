"""Resumable IE643 experiment queue. See CAMPAIGN.md; no network calls here."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parent
HOME = ROOT/"campaign"
COLLECTIONS = ["datasets_v2_rk4_zero", "datasets_v2_pinnwin", "datasets_v2_pinnwin2",
               "datasets_v2_density", "datasets_v2_density_wideic", "datasets_v2_pinnwin3", "datasets_v2_density_extended"]
BASELINES = dict(zip(COLLECTIONS, ["results_tikhonov", "results_pinnwin_adjoint", "results_pinnwin2_adjoint",
                                  "results_density_adjoint", "results_density_wideic_adjoint", "results_pinnwin3_adjoint", "results_density_extended_adjoint"]))
CODE = ["compare_neural.py", "adjoint_nn.py", "neural_diffusivity.py", "adjoint_inversion.py", "integrators.py"]


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def metric_path(job):
    return ROOT/job["output"]/job["key"]/f"{job['method']}__s{job['seed']}"/"metrics.json"


def completed(job):
    path = metric_path(job)
    try:
        m = json.loads(path.read_text())
        return m["method"] == job["method"] and (path.parent/"D_hat.npz").exists() and (path.parent/"model.pt").exists() if job["method"] != "grid" else path.exists() and (path.parent/"D_hat.npz").exists()
    except (OSError, KeyError, ValueError):
        return False


def initialize():
    HOME.mkdir(exist_ok=True)
    archived = ROOT/"kaggle_thakur/dataset/data.zip"
    target = ROOT/"datasets_v2_pinnwin3/manifest.json"
    if not target.exists():
        with zipfile.ZipFile(archived) as archive:
            for member in archive.namelist():
                if member.startswith("datasets_v2_pinnwin3/"):
                    dest = (ROOT/member).resolve()
                    if not dest.is_relative_to(ROOT):
                        raise ValueError("Archive path leaves workspace")
                    archive.extract(member, ROOT)
    fingerprint = hashlib.sha256(b"".join((ROOT/name).read_bytes() for name in CODE)).hexdigest()[:12]
    snapshot = HOME/"code"/fingerprint; snapshot.mkdir(parents=True, exist_ok=True)
    for name in CODE:
        shutil.copy2(ROOT/name, snapshot/name)
    jobs = []
    for collection in COLLECTIONS:
        historical_index = {p.parent.name:p for p in (ROOT/BASELINES[collection]).rglob("D_hat.npz")}
        entries = json.loads((ROOT/collection/"manifest.json").read_text())
        for e in entries:
            hidden = e.get("bc_known", True) is False or e.get("source_known", True) is False
            priority = 0 if collection == COLLECTIONS[0] and e["key"] in [f"{t}__Fo0.1__uniform8__n50__random__eta0.05" for t in ["A_smooth", "B_gaussian_bump", "E_grf"]] else 1 if collection == COLLECTIONS[0] and e["key"] in [f"{t}__Fo0.1__uniform8__n50__random__eta0.05" for t in ["C_inclusion", "D_layered", "F_defects"]] else 2 if collection == COLLECTIONS[0] else 3
            historical = historical_index.get(e["key"], ROOT/BASELINES[collection]/e["key"]/"D_hat.npz")
            for method in (["adjoint_nn", "pinn"] if historical.exists() else ["grid", "adjoint_nn", "pinn"]):
                job_id = hashlib.sha256(f"{collection}/{e['key']}/{method}/42".encode()).hexdigest()[:16]
                jobs.append(dict(id=job_id, collection=collection, key=e["key"], method=method, seed=42,
                                 physics_mode="assumed_zero" if hidden else "matched", priority=priority,
                                 output=f"campaign/results/{collection}/{job_id}", code=f"campaign/code/{fingerprint}",
                                 family=e.get("family", "main"), n_sensors=e["n_sensors"], n_times=e["n_times"],
                                 eta=e["eta"], tier=e.get("base_tier", e["tier"]), historical_grid=str(historical.relative_to(ROOT)) if historical.exists() else None))
    jobs.sort(key=lambda j:(j["priority"], j["collection"], j["key"], j["method"]))
    # Reinitialisation never overwrites an existing campaign configuration.
    if (HOME/"jobs.json").exists():
        raise RuntimeError("campaign/jobs.json already exists; inspect before replacing the frozen campaign")
    atomic_json(HOME/"jobs.json", jobs)
    atomic_json(HOME/"settings.json", dict(code_hash=fingerprint, created=now(), coefficient_width=32,
                coefficient_depth=3, seed=42, reg="tikhonov", maxiter=150, adam_steps=15000, lbfgs_steps=750,
                n_collocation=2048, n_boundary=256, data_batch=4096, physics_weight=1000,
                condition_weight=1000, pinn_alpha=1, per_job_timeout_s=1800))
    atomic_json(HOME/"state.json", dict(status="prepared", created=now(), planned_jobs=len(jobs), collections=COLLECTIONS))
    print(f"Prepared {len(jobs)} jobs over {sum(len(json.loads((ROOT/c/'manifest.json').read_text())) for c in COLLECTIONS)} configurations; source {fingerprint}")


def command_for(job, device="cpu", smoke=False):
    settings = json.loads((HOME/"settings.json").read_text())
    return [sys.executable, "-u", str(ROOT/job["code"]/"compare_neural.py"),
            "--data", str(ROOT/job["collection"]), "--keys", job["key"], "--out", str(ROOT/job["output"]),
            "--methods", job["method"], "--physics-mode", job["physics_mode"], "--device", device,
            "--seed", str(job["seed"]), "--width", str(settings["coefficient_width"]), "--depth", str(settings["coefficient_depth"]),
            "--threads", "1", "--maxiter", "1" if smoke else str(settings["maxiter"]),
            "--adam-steps", "2" if smoke else str(settings["adam_steps"]), "--lbfgs-steps", "1" if smoke else str(settings["lbfgs_steps"]),
            "--n-collocation", "32" if smoke else str(settings["n_collocation"]), "--n-boundary", str(settings["n_boundary"]),
            "--data-batch", str(settings["data_batch"]), "--physics-weight", str(settings["physics_weight"]),
            "--condition-weight", str(settings["condition_weight"]), "--pinn-alpha", str(settings["pinn_alpha"])]


class FileLock:
    """OS lock is released automatically on process exit; no stale PID locks."""
    def __init__(self, path): self.path = path
    def __enter__(self):
        self.file = self.path.open("a+b")
        self.file.seek(0); self.file.write(b"0"); self.file.flush(); self.file.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self
    def __exit__(self, *args): self.file.close()


def run(args):
    jobs = json.loads((HOME/"jobs.json").read_text())
    settings = json.loads((HOME/"settings.json").read_text())
    candidates = [j for j in jobs if j["method"] == args.method and not completed(j)]
    if not args.retry_failed:
        def eligible(job):
            record=HOME/"job_state"/(job["id"]+".json")
            if not record.exists():return True
            return json.loads(record.read_text()).get("status") not in ("failed","timeout")
        candidates=[j for j in candidates if eligible(j)]
    if args.max_jobs: candidates = candidates[:args.max_jobs]
    started, mutex = time.monotonic(), threading.Lock()
    counter = dict(complete=0, failed=0, timeout=0, skipped_budget=0)
    logdir = HOME/"logs"; logdir.mkdir(exist_ok=True)
    statefile = HOME/f"worker_{args.method}.json"
    with FileLock(HOME/f"{args.method}.lock"):
        active = set()
        def save():
            atomic_json(statefile, dict(pid=os.getpid(), method=args.method, started=now(), updated=now(),
                                      status="running", active=sorted(active), finished_this_run=counter,
                                      pending_at_start=len(candidates), budget_seconds=args.budget_seconds))
        save()
        def execute(job):
            remaining = args.budget_seconds-(time.monotonic()-started)
            if remaining < 60: return job, "skipped_budget", None
            with mutex: active.add(job["id"]); save()
            command = command_for(job, args.device)
            record = dict(job_id=job["id"], started=now(), command=command,
                          sensors_sha256=hashlib.sha256((ROOT/job["collection"]/"sensors"/(job["key"]+".csv")).read_bytes()).hexdigest())
            try:
                with (logdir/(job["id"]+".log")).open("a", encoding="utf-8") as log:
                    r = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=min(settings["per_job_timeout_s"], remaining), env={**os.environ,"MKL_THREADING_LAYER":"SEQUENTIAL"})
                status = "complete" if r.returncode == 0 and completed(job) else "failed"
                record.update(status=status, returncode=r.returncode)
            except subprocess.TimeoutExpired:
                status = "timeout"; record.update(status=status)
            except Exception as ex:
                status = "failed"; record.update(status=status, error=repr(ex))
            record["finished"] = now()
            atomic_json(HOME/"job_state"/(job["id"]+".json"), record)
            with mutex: active.discard(job["id"]); counter[status] += 1; save()
            return job, status, record
        # Submit only a bounded number at a time, so a budget expiration does
        # not create hundreds of queued subprocesses or duplicate workers.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            iterator = iter(candidates)
            futures = {pool.submit(execute,j) for j in [next(iterator,None) for _ in range(args.workers)] if j}
            while futures:
                future = next(as_completed(futures)); futures.remove(future)
                job, status, _ = future.result()
                print(now(), job["collection"], job["key"], args.method, status, flush=True)
                if status != "skipped_budget" and time.monotonic()-started < args.budget_seconds-60:
                    nxt = next(iterator, None)
                    if nxt: futures.add(pool.submit(execute,nxt))
        atomic_json(statefile, dict(pid=os.getpid(), status="finished", updated=now(), finished_this_run=counter))
    summarize()


def import_gpu():
    archive = ROOT/"kaggle_neural/output/neural_gpu_results.zip"
    jobs = json.loads((HOME/"jobs.json").read_text())
    lookup = {(j["collection"],j["key"],j["method"]):j for j in jobs}
    count=0
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith("/metrics.json"):
                m=json.loads(z.read(name)); key=m["key"]
                job=lookup[(COLLECTIONS[0],key,"pinn")]
                dest=metric_path(job).parent; dest.mkdir(parents=True,exist_ok=True)
                prefix=name.rsplit("/",1)[0]+"/"
                for member in z.namelist():
                    if member.startswith(prefix) and "/" not in member[len(prefix):]:
                        (dest/member[len(prefix):]).write_bytes(z.read(member))
                m.update(physics_mode="matched", provenance="initial Kaggle v1: same coefficient, loss weights, budgets; full 400-reading data sum")
                atomic_json(dest/"metrics.json",m); count+=1
    print("Imported",count,"completed GPU runs")
    summarize()


def summarize():
    import csv
    jobs=json.loads((HOME/"jobs.json").read_text())
    rows=[]; seen=set()
    for j in jobs:
        if completed(j):
            m=json.loads(metric_path(j).read_text())
            rows.append(dict(collection=j["collection"],key=j["key"],method=j["method"],family=j["family"],tier=j["tier"],
                             eta=j["eta"],n_sensors=j["n_sensors"],n_times=j["n_times"],physics_mode=j["physics_mode"],
                             rel_L2_D=m.get("rel_L2_D"), replay=m.get("rms_replay_over_sigma"),
                             relative_replay=m.get("rms_replay_relative_signal"),u_fit=m.get("rms_unet_over_sigma"),
                             wall_s=m.get("wall_s"),flag=m.get("flag"),metrics=str(metric_path(j).relative_to(ROOT))))
        if j["historical_grid"] and (j["collection"],j["key"]) not in seen:
            seen.add((j["collection"],j["key"]))
            path=ROOT/j["historical_grid"]
            try:
                m=json.loads((path.parent/"metrics.json").read_text())
                rows.append(dict(collection=j["collection"],key=j["key"],method="grid_historical",family=j["family"],tier=j["tier"],
                                 eta=j["eta"],n_sensors=j["n_sensors"],n_times=j["n_times"],physics_mode=j["physics_mode"],
                                 rel_L2_D=m.get("rel_L2_D"),replay=m.get("rms_replay_over_sigma") if m.get("sigma",1)>0 else None,
                                 relative_replay=None,u_fit=None,wall_s=m.get("wall_s"),flag=m.get("flag"),
                                 metrics=str((path.parent/"metrics.json").relative_to(ROOT))))
            except (OSError,ValueError): pass
    if rows:
        with (HOME/"summary.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    state=dict(updated=now(),planned_jobs=len(jobs),completed_new_jobs=sum(completed(j) for j in jobs),
               completed_by_method={m:sum(completed(j) for j in jobs if j["method"]==m) for m in ["grid","adjoint_nn","pinn"]},
               historical_grid_rows=sum(r["method"]=="grid_historical" for r in rows))
    atomic_json(HOME/"state.json",state)
    lines=["# IE643 campaign findings (partial)","",f"Updated {state['updated']}","",f"Completed new jobs: {state['completed_new_jobs']} / {len(jobs)}.",
           "Historical grid results are retained separately; budgets, alpha selection and hardware are not identical.","",
           "| Collection | Dataset | Method | rel L2(D) | CN replay / sigma | u-net / sigma |",
           "|---|---|---|---:|---:|---:|"]
    for row in rows:
        if row["method"]=="grid_historical" and not any(r["key"]==row["key"] and r["collection"]==row["collection"] and r["method"]!="grid_historical" for r in rows):continue
        fmt=lambda v:"n/a" if v is None else f"{v:.4g}"
        lines.append(f"| {row['collection']} | {row['key']} | {row['method']} | {fmt(row['rel_L2_D'])} | {fmt(row['replay'])} | {fmt(row['u_fit'])} |")
    lines += ["", "Only paired completed cases support direct accuracy comparisons. Missing cases are not losses.",
              "Unknown-physics rows impose the same supplied zero assumptions in both new methods; their replay is under the told model, not the hidden true physics.",
              "The initial GPU PINNs fit sensors near the noise level but replay poorly: fitting u alone does not identify D.",
              "Optimizer budget caps and statistical/model-error floors must be distinguished before interpreting convergence."]
    (HOME/"PROGRESS.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(state),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="action",required=True)
    sub.add_parser("init");sub.add_parser("import-gpu");sub.add_parser("summary")
    runner=sub.add_parser("run");runner.add_argument("--method",choices=["grid","adjoint_nn","pinn"],required=True)
    runner.add_argument("--workers",type=int,default=2);runner.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    runner.add_argument("--max-jobs",type=int,default=0);runner.add_argument("--budget-seconds",type=int,default=21600)
    runner.add_argument("--retry-failed",action="store_true",help="Explicitly retry failures after inspecting and resolving their cause")
    args=parser.parse_args()
    if args.action=="init":initialize()
    elif args.action=="import-gpu":import_gpu()
    elif args.action=="summary":summarize()
    else:run(args)


if __name__=="__main__":main()
