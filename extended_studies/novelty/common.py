"""
common.py -- shared pieces of the novelty pipeline: where the data live, scoring, result files.

Run every script from extended_studies/ through run.py, e.g.
    python run.py novelty/run_novelty.py --exp E1 --size quick
so core/ is importable and dataset folders resolve relative to extended_studies/.
"""
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

import adjoint_inversion as A            # core/ copy; identical maths to submission/adjoint_inversion.py

EXT = Path(__file__).resolve().parent.parent          # extended_studies/
REPO = EXT.parent
SUBMISSION = REPO / "submission"
RESULTS = Path(os.environ.get("NOVELTY_RESULTS", EXT / "novelty" / "results"))

COLLECTIONS = {
    "bench": "datasets_v2_rk4_zero",              # core/generate_dataset_v2.py (389 datasets)
    "density": "datasets_v2_density_extended",    # study1_representation/extend_sensor_density.py
    "pinnwin": "datasets_v2_pinnwin",             # study3_pinn_favoured/generate_pinn_favoured.py
    "arrhenius": "datasets_novelty_arrhenius",    # novelty/arrhenius.py
    "oed": "datasets_novelty_oed",                # novelty/sensor_design.py
}


def load_module(name, path):
    """Import a file under a chosen module name (the main PINN lives in submission/, and core/ has an
    older copy with the same file name)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PINN_MAIN = None


def pinn_main():
    """submission/pinn_inversion.py: the project's PINN (the one in the report and the deck)."""
    global _PINN_MAIN
    if _PINN_MAIN is None:
        _PINN_MAIN = load_module("pinn_main", SUBMISSION / "pinn_inversion.py")
    return _PINN_MAIN


_MANIFESTS = {}


def manifest(collection):
    root = EXT / COLLECTIONS[collection]
    if collection not in _MANIFESTS:
        path = root / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run  python run.py novelty/prepare_data.py --only {collection}")
        _MANIFESTS[collection] = {e["key"]: e for e in json.load(open(path))}
    return root, _MANIFESTS[collection]


def dataset(collection, key):
    """The adjoint's view of one dataset (sensor readings, IC, BC, sigma); truth only via ds.ground_truth()."""
    root, entries = manifest(collection)
    return A.Dataset(str(root), entries[key])


def entry(collection, key):
    return manifest(collection)[1][key]


# ---------------------------------------------------------------------------------------------
# scoring: every estimate, from every method, is scored by the same functions
# ---------------------------------------------------------------------------------------------
_REPLAY = {}


def replay_rms_over_sigma(ds, D, rows=None):
    """Run D through the standard CN solver of the inversion grid (the adjoint's forward model, as
    the dataset's problem file describes it) and compare with the readings: RMS / sigma."""
    k = (ds.root, ds.key)
    if k not in _REPLAY:
        _REPLAY[k] = A.AdjointSolver(ds)
    s = _REPLAY[k]
    W = s.forward(np.asarray(D, float).ravel())[0]
    r = s.predict(W) - ds.u
    if rows is not None:
        r = r[rows]
    return float(np.sqrt(np.mean(r ** 2)) / s.sigma_w)


def field_scores(D_hat, D_true):
    D_hat = np.asarray(D_hat, float).ravel()
    D_true = np.asarray(D_true, float).ravel()
    return dict(rel_L2_D=float(np.linalg.norm(D_hat - D_true) / np.linalg.norm(D_true)),
                rmse_logD=float(np.sqrt(np.mean((np.log(D_hat) - np.log(D_true)) ** 2))),
                D_min=float(D_hat.min()), D_max=float(D_hat.max()))


def best_constant_score(ds, D_true):
    """relL2 of the single constant that best fits the data: what any method has to beat."""
    s = A.AdjointSolver(ds)
    D_c = float(np.exp(s.best_constant()))
    return D_c, float(np.linalg.norm(D_c - D_true.ravel()) / np.linalg.norm(D_true.ravel()))


# ---------------------------------------------------------------------------------------------
# result files: results/<exp>/<arm>/<key>/{metrics.json, D_hat.npz, ...}
# ---------------------------------------------------------------------------------------------
def run_dir(exp, arm, key):
    return RESULTS / exp / arm / key


def is_done(exp, arm, key):
    return (run_dir(exp, arm, key) / "metrics.json").exists()


def save_run(exp, arm, key, metrics, **arrays):
    d = run_dir(exp, arm, key)
    d.mkdir(parents=True, exist_ok=True)
    if arrays:
        np.savez_compressed(d / "D_hat.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    tmp = d / "metrics.json.tmp"
    tmp.write_text(json.dumps(metrics, indent=1, default=_json_default))
    tmp.replace(d / "metrics.json")                    # atomic: a killed job never leaves half a file


def load_run(exp, arm, key):
    d = run_dir(exp, arm, key)
    m = json.loads((d / "metrics.json").read_text())
    arr = dict(np.load(d / "D_hat.npz")) if (d / "D_hat.npz").exists() else {}
    return m, arr


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0


def device_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
