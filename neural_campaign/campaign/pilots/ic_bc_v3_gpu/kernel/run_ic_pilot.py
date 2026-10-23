"""Train the identical shared D-network on three known-physics cases on GPU.

STEPS=2 LBFGS=1 CASE_LIMIT=1 DEVICE=cpu INPUT_DIR=... WORK=... enables a local
packaging smoke test. Kaggle defaults require CUDA, avoiding accidental CPU runs.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def main():
    import torch
    device = os.environ.get("DEVICE", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but unavailable; this run will not silently use CPU")
    print("Device:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU", flush=True)
    inp = os.environ.get("INPUT_DIR")
    if inp:
        source = Path(inp).resolve()
    else:
        hits = list(Path("/kaggle/input").rglob("compare_neural.py"))
        if len(hits) != 1:
            raise RuntimeError(f"Expected one shared-network input dataset, found {len(hits)}")
        source = hits[0].parent
    work = Path(os.environ.get("WORK", "/kaggle/working/neural_work")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.py"):
        shutil.copy2(path, work/path.name)
    if (source/"data.zip").exists():
        with zipfile.ZipFile(source/"data.zip") as archive:
            archive.extractall(work)
    else:
        # Kaggle sometimes expands uploaded archives automatically.
        candidates = list(source.rglob("datasets_v2_rk4_zero/manifest.json"))
        if len(candidates) != 1:
            raise RuntimeError("Benchmark archive or expanded manifest not found")
        shutil.copytree(candidates[0].parent, work/"datasets_v2_rk4_zero", dirs_exist_ok=True)
    manifest = json.loads((work/"datasets_v2_rk4_zero/manifest.json").read_text())
    # Explicit order: begin with the smooth sanity case.
    tiers = ("A_smooth", "B_gaussian_bump", "E_grf")
    keys = [next(e["key"] for e in manifest if e["key"].startswith(t+"__")) for t in tiers]
    keys = keys[:int(os.environ.get("CASE_LIMIT", "1"))]
    steps = os.environ.get("STEPS", "15000")
    lbfgs = os.environ.get("LBFGS", "750")
    budget = float(os.environ.get("BUDGET_SECONDS", "6500"))
    started = time.monotonic()
    outcomes = []
    zip_path = work.parent/"neural_gpu_results.zip"

    def pack():
        (work/"job_status.json").write_text(json.dumps(outcomes, indent=2))
        # Zip contains only finished artifacts, logs and configuration.
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(work.rglob("*")):
                if path.is_file() and ("results" in path.relative_to(work).parts or path.name == "job_status.json" or path.suffix == ".log"):
                    archive.write(path, path.relative_to(work))

    for key in keys:
        remaining = budget - (time.monotonic()-started)
        if remaining < 120:
            outcomes.append(dict(key=key, status="skipped_budget")); pack(); break
        command = [sys.executable, "-u", "compare_neural.py", "--methods", "pinn", "--state-model", "gaussian_ic_bc", "--device", device,
                   "--keys", key, "--out", "results/"+key, "--adam-steps", steps, "--lbfgs-steps", lbfgs,
                   "--n-collocation", os.environ.get("N_COLLOCATION", "2048"), "--n-boundary", "256",
                   "--physics-weight", "1000", "--condition-weight", "1000", "--pinn-alpha", "1",
                   "--width", "32", "--depth", "3", "--threads", "2", "--seed", "42"]
        print("Starting", key, " ".join(command), flush=True)
        log_path = work/(key+".log")
        status, exit_code = "failed", None
        try:
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
            exit_code = result.returncode
            status = "complete" if exit_code == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
        outcomes.append(dict(key=key, status=status, exit_code=exit_code, command=command))
        print(status, key, log_path.read_text(encoding="utf-8", errors="replace")[-2500:], flush=True)
        pack()
        if status != "complete":
            raise RuntimeError(f"Job {status}; available artifacts and log saved to {zip_path}")
    print("Saved", zip_path, flush=True)


if __name__ == "__main__":
    main()
