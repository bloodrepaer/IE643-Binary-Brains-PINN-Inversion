"""Split all datasets into N chunks and run each chunk in its own process."""
import json, os, subprocess, sys
N   = int(sys.argv[1]) if len(sys.argv) > 1 else 4      # number of processes
REG = sys.argv[2] if len(sys.argv) > 2 else "tikhonov"  # tikhonov | tv
keys = [e["key"] for e in json.load(open("datasets_v2_rk4_zero/manifest.json"))]
env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
procs = []
for i in range(N):
    chunk = keys[i::N]                                   # interleaved: balances slow and fast datasets
    out = f"results_{REG}/part{i}"
    log = open(f"results_{REG}_part{i}.log", "w")
    procs.append(subprocess.Popen([sys.executable, "adjoint_inversion.py", "--reg", REG,
                                   "--out", out, "--keys", *chunk], stdout=log, stderr=log, env=env))
for p in procs:
    p.wait()
print("done")