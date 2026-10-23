"""
run_queue.py -- run a list of pinn_papers.py / adjoint jobs one after another, skipping finished ones,
and stop before a wall-clock budget runs out (so a capped background slot never kills a job half-way).

    python run_queue.py queue.json --budget-min 110
queue.json: [{"cmd": [...argv...], "done": "path/that/exists/when/finished", "est_min": 50}, ...]
"""
import json, os, subprocess, sys, time

jobs = json.load(open(sys.argv[1]))
budget = float(sys.argv[sys.argv.index("--budget-min") + 1]) if "--budget-min" in sys.argv else 110
t0 = time.time()
left = [j for j in jobs if not os.path.exists(j["done"])]
for j in left:
    used = (time.time() - t0) / 60
    if used + j["est_min"] > budget:
        print(f"stopping: {used:.0f} min used, next job needs ~{j['est_min']} min", flush=True)
        break
    print(f"[{time.strftime('%H:%M')}] start {' '.join(j['cmd'][-6:])}", flush=True)
    subprocess.run(j["cmd"], cwd="/home/claude/v2")
remaining = [j for j in jobs if not os.path.exists(j["done"])]
print(f"QUEUE: {len(jobs) - len(remaining)} of {len(jobs)} done, {len(remaining)} remaining", flush=True)
