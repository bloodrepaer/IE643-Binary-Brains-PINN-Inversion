#!/usr/bin/env python3
"""
pinn_inversion_v2.py — pinn_inversion_bc.py plus four switchable fixes (all default OFF = old behaviour):
  m_wall_taper=False      D is free at the walls (old: D pinned to exp(m_bg) on every wall)
  m_init="best_constant"  start from the data-fitted constant D (same stage-0 as the adjoint) instead of Fo*L^2/T
  lbfgs_iters>0           L-BFGS polish after Adam on a fixed batch (Zhang, Alla & Karniadakis 2026)
  morozov_damping=False   plain data misfit (old: data term switches off below 0.95 sigma^2)

pinn_inversion_bc.py — copy of pinn_inversion.py with boundary-condition modes (see BC MODES below).

BC MODES (read from problem/*.json; the only change from pinn_inversion.py):
  dirichlet       all edges 0 and known  -> unchanged: hard zero window + zero BC loss
  dirichlet_soft  edges non-zero, known  -> no hard window; L_BC = mean (u(wall,t) - g(t))^2, g from the problem file
  free            "known": false         -> no hard window, no BC loss: u is free on the walls

train.py — IE 643 PINN Inverse Solver & 10-Point Fixed Lambda Parameter Sweep
Team: Binary_Brains (Jash Taparia · Mihir Gune)

Tailored to `v2` (`datasets_v2_rk4_zero` + `results_adjoint` / `results_tikhonov`).

Governing 2D Divergence-Form Diffusion PDE:
    du/dt - div( D(x, y) * grad(u(x, y, t)) ) - f(x, y, t) = 0
    with D_phi(x, y) = exp(m_phi(x, y)) > 0.

Composite PINN Loss Function (Slide 14):
    L_total = lambda_data * L_data
            + lambda_pde  * L_pde
            + lambda_bc   * L_bc
            + lambda_ic   * L_ic
            + lambda_reg  * R(m_phi)

Evaluation Metrics (Slide 16 & Classical Baseline):
  1. Relative L2 Error on D(x, y):
         Rel_L2(D) = ||D_hat - D_true||_2 / ||D_true||_2
  2. Normalized Discrepancy / Reduced Chi-Squared (Slide 16 Displayed Metric):
         chi2_red(D_hat) = (1 / N_d) * sum_{i=1}^{N_d} ( (u_solver(x_i, y_i, t_i; D_hat) - u_obs_i) / sigma_i )^2
     - chi2_red ~ 1.0 : Optimal (explains sensor data as well as noise level allows)
     - chi2_red < 1.0 : Over-fitting (capturing random noise fluctuations)
     - chi2_red > 1.0 : Under-fitting / over-smoothed (cannot fully explain observations)
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force line-buffered stdout so Kaggle Logs tab streams every print() in real time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Default Benchmark Suite (Matches `results_adjoint/` 10 Core Cases)
# =====================================================================
DEFAULT_ADJOINT_BENCHMARK_EXPS: List[str] = [
    "A_smooth__Fo0.1__uniform8__n50__random__eta0.05",
    "B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05",
    "C_inclusion__Fo0.1__uniform8__n100__random__eta0.01",
    "C_inclusion__Fo0.1__uniform8__n50__random__eta0.01",
    "C_inclusion__Fo0.1__uniform8__n50__random__eta0.05",
    "C_inclusion__Fo0.1__uniform8__n20__random__eta0.01",
    "C_inclusion__Fo0.1__uniform8__n5__random__eta0.01",
    "D_layered__Fo0.1__uniform8__n50__random__eta0.05",
    "E_grf__Fo0.1__uniform8__n50__random__eta0.05",
    "F_defects__Fo0.1__uniform8__n50__random__eta0.05",
]


# =====================================================================
# 2. Configuration & 10-Point Fixed Lambda Parameter Sweep Grid
# =====================================================================
@dataclass
class PINNConfig:
    # Paths
    data_root: str = "/kaggle/input/pinn-v2-dataset"
    output_dir: str = "/kaggle/working"
    exp_id: str = "B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05"

    # Composite Loss Multiplier Weights (Locked to Winning Sweep Config C04_balanced_lowreg_tik)
    lambda_data: float = 40.0
    lambda_pde: float = 20.0
    lambda_bc: float = 10.0
    lambda_ic: float = 20.0
    lambda_reg: float = 2e-4

    # Regularization on m_phi(x, y) = log D(x, y)
    reg_type: str = "tikhonov"        # "tikhonov", "tv", or "none"
    tv_eps: float = 1e-6

    # Lambda Weight Mode ("fixed" keeps multipliers fixed across all epochs)
    lambda_mode: str = "fixed"
    morozov_damping: bool = True
    stage1_ratio: float = 0.20
    gradnorm_alpha: float = 0.90

    # Boundary Condition Type ("dirichlet" u=0, "neumann" du/dn=0, or "auto" from problem JSON)
    bc_type: str = "auto"

    # Network Hyperparameters
    u_hidden_dim: int = 80
    u_num_layers: int = 4
    m_hidden_dim: int = 64
    m_num_layers: int = 4
    use_fourier_features: bool = True
    u_fourier_scale: float = 0.35
    m_fourier_scale: float = 0.35

    # Optimization & Sampling
    epochs: int = 1500
    lr_u: float = 1.5e-3
    lr_m: float = 3.5e-3
    n_collocation: int = 768
    n_bc: int = 128
    n_ic: int = 256
    batch_size_data: int = 400
    warmstart_from_adjoint: bool = False
    seed: int = 42

    # v2 fixes (defaults reproduce the original behaviour)
    m_wall_taper: bool = True
    m_init: str = "fo"            # "fo" (Fo*L^2/T, uses the problem file) or "best_constant" (fitted to data)
    lbfgs_iters: int = 0
    lbfgs_colloc_mult: int = 4


# 10 Fixed Parameter Configurations Spanning Under-Fitting (chi2 > 1),
# Optimal Morozov Discrepancy (chi2 ~ 1), and Noise Over-Fitting (chi2 < 1)
LAMBDA_SWEEP_GRID: Dict[str, Dict[str, Any]] = {
    "C01_equal_unreg": {
        "lambda_data": 1.0,
        "lambda_pde": 1.0,
        "lambda_bc": 1.0,
        "lambda_ic": 1.0,
        "lambda_reg": 0.0,
        "reg_type": "none",
        "lambda_mode": "fixed",
        "morozov_damping": True,
    },
    "C02_low_data_tik": {
        "lambda_data": 10.0,
        "lambda_pde": 5.0,
        "lambda_bc": 10.0,
        "lambda_ic": 10.0,
        "lambda_reg": 1e-3,
        "reg_type": "tikhonov",
        "lambda_mode": "fixed",
        "morozov_damping": True,
    },
    "C03_mod_data_tik": {
        "lambda_data": 25.0,
        "lambda_pde": 10.0,
        "lambda_bc": 10.0,
        "lambda_ic": 20.0,
        "lambda_reg": 1e-3,
        "reg_type": "tikhonov",
        "lambda_mode": "fixed",
        "morozov_damping": True,
    },
    "C04_balanced_lowreg_tik": {
        "lambda_data": 40.0,
        "lambda_pde": 20.0,
        "lambda_bc": 10.0,
        "lambda_ic": 20.0,
        "lambda_reg": 2e-4,
        "reg_type": "tikhonov",
        "lambda_mode": "fixed",
        "morozov_damping": True,
    },
    "C05_sweetspot_tik": {
        "lambda_data": 50.0,
        "lambda_pde": 25.0,
        "lambda_bc": 15.0,
        "lambda_ic": 25.0,
        "lambda_reg": 8e-4,
        "reg_type": "tikhonov",
        "lambda_mode": "fixed",
        "morozov_damping": True,
    },
    "C06_sweetspot_tv": {
        "lambda_data": 50.0,
        "lambda_pde": 25.0,
        "lambda_bc": 15.0,
        "lambda_ic": 25.0,
        "lambda_reg": 4e-4,
        "reg_type": "tv",
        "lambda_mode": "fixed",
        "morozov_damping": True,
    },
    "C07_high_data_tik": {
        "lambda_data": 100.0,
        "lambda_pde": 25.0,
        "lambda_bc": 20.0,
        "lambda_ic": 40.0,
        "lambda_reg": 8e-4,
        "reg_type": "tikhonov",
        "lambda_mode": "fixed",
        "morozov_damping": True,
    },
    "C08_high_pde_tik": {
        "lambda_data": 40.0,
        "lambda_pde": 50.0,
        "lambda_bc": 15.0,
        "lambda_ic": 25.0,
        "lambda_reg": 8e-4,
        "reg_type": "tikhonov",
        "lambda_mode": "fixed",
        "morozov_damping": True,
    },
    "C09_oversmoothed_highreg": {
        "lambda_data": 25.0,
        "lambda_pde": 10.0,
        "lambda_bc": 10.0,
        "lambda_ic": 20.0,
        "lambda_reg": 5e-2,
        "reg_type": "tikhonov",
        "lambda_mode": "fixed",
        "morozov_damping": True,
    },
    "C10_noise_overfit_weakreg": {
        "lambda_data": 250.0,
        "lambda_pde": 2.0,
        "lambda_bc": 10.0,
        "lambda_ic": 50.0,
        "lambda_reg": 1e-5,
        "reg_type": "none",
        "lambda_mode": "fixed",
        "morozov_damping": False,
    },
}


# =====================================================================
# 3. Dataset Inspection & Exact Loader for `datasets_v2_rk4_zero`
# =====================================================================
def resolve_v2_root(data_root: str) -> Path:
    """
    Finds the root directory containing `datasets_v2_rk4_zero` inside /kaggle/input or local path,
    handling both legacy (/kaggle/input/<slug>) and Kaggle v2 (/kaggle/input/datasets/<user>/<slug>) mounts.
    """
    search_roots = [Path(data_root), Path("/kaggle/input"), Path(".")]
    for sroot in search_roots:
        if not sroot.exists():
            continue
        if (sroot / "datasets_v2_rk4_zero").exists():
            return sroot
        for candidate in sroot.rglob("datasets_v2_rk4_zero"):
            if candidate.is_dir():
                return candidate.parent
        for candidate in sroot.rglob("manifest.json"):
            if (candidate.parent / "sensors").exists() and (candidate.parent / "problem").exists():
                return candidate.parent
    return Path(data_root)


def parse_exp_id(exp_id: str) -> Dict[str, str]:
    parts = exp_id.split("__")
    problem_stem = f"{parts[0]}__{parts[1]}" if len(parts) >= 2 else exp_id
    return {
        "exp_id": exp_id,
        "case_name": parts[0] if len(parts) >= 1 else exp_id,
        "fo_tag": parts[1] if len(parts) >= 2 else "Fo0.1",
        "time_protocol": parts[2] if len(parts) >= 3 else "uniform8",
        "sensor_count_tag": parts[3] if len(parts) >= 4 else "n50",
        "sensor_layout": parts[4] if len(parts) >= 5 else "random",
        "noise_tag": parts[5] if len(parts) >= 6 else "eta0.05",
        "problem_stem": problem_stem,
    }


def _best_constant_logD(ds_dir, key, fallback):
    """Stage 0 shared with the adjoint: the single log D that best fits the data through the CN solver."""
    try:
        import adjoint_inversion as _ai
        man = json.load(open(Path(ds_dir) / "manifest.json"))
        entry = next(e for e in man if e["key"] == key)
        m_c = float(_ai.AdjointSolver(_ai.Dataset(str(ds_dir), entry)).best_constant())
        print(f"  [init] best constant D = {math.exp(m_c):.4f}", flush=True)
        return m_c
    except Exception as ex:  # noqa: BLE001
        print(f"[WARN] best-constant init failed ({ex}); using Fo-based init", flush=True)
        return fallback


def load_v2_experiment(cfg: PINNConfig) -> Dict[str, Any]:
    v2_root = resolve_v2_root(cfg.data_root)
    ds_dir = v2_root / "datasets_v2_rk4_zero" if (v2_root / "datasets_v2_rk4_zero").exists() else v2_root
    meta = parse_exp_id(cfg.exp_id)
    prob_stem = meta["problem_stem"]

    inspection: Dict[str, Any] = {
        "v2_root": str(v2_root),
        "ds_dir": str(ds_dir),
        "exp_meta": meta,
        "global_config": {},
        "problem_json": {},
        "problem_npz_keys": {},
        "ground_truth_npz_keys": {},
        "sensor_npz_keys": {},
        "sensor_csv_columns": [],
        "ic_csv_columns": [],
        "baseline_metrics": None,
    }

    cfg_json_path = ds_dir / "config.json"
    if cfg_json_path.exists():
        with open(cfg_json_path, "r") as f:
            inspection["global_config"] = json.load(f)

    prob_json_path = ds_dir / "problem" / f"{prob_stem}.json"
    prob_npz_path = ds_dir / "problem" / f"{prob_stem}.npz"
    prob_ic_csv_path = ds_dir / "problem" / f"{prob_stem}__ic.csv"

    if prob_json_path.exists():
        with open(prob_json_path, "r") as f:
            inspection["problem_json"] = json.load(f)

    Lx, Ly, T_max, Fo_val = 1.0, 1.0, 0.72271079, 0.1
    bc_type_detected = "dirichlet"
    bc_edge_funcs = None
    pjson = inspection["problem_json"]
    if isinstance(pjson, dict):
        dom = pjson.get("domain", {}) if isinstance(pjson.get("domain"), dict) else {}
        Lx = float(pjson.get("Lx", dom.get("Lx", 1.0)))
        Ly = float(pjson.get("Ly", dom.get("Ly", 1.0)))
        Fo_val = float(pjson.get("Fo", dom.get("Fourier_number", 0.1)))
        T_max = float(
            pjson.get("T", dom.get("T", pjson.get("T_final", pjson.get("t_end", 0.72271079))))
        )
        bc_obj = pjson.get("boundary_condition", "dirichlet")
        bc_raw = str(bc_obj.get("type", "dirichlet") if isinstance(bc_obj, dict) else bc_obj).lower()
        if "neumann" in bc_raw or "zero_flux" in bc_raw or "insulated" in bc_raw:
            bc_type_detected = "neumann"
        elif isinstance(bc_obj, dict) and bc_obj.get("known", True) is False:
            bc_type_detected = "free"
        elif isinstance(bc_obj, dict) and any(
            str(v).strip() not in ("0", "0.0") for v in bc_obj.get("edges", {}).values()
        ):
            bc_type_detected = "dirichlet_soft"
            import sympy
            _ts = sympy.Symbol("t")
            bc_edge_funcs = {
                e: sympy.lambdify(_ts, sympy.sympify(ex), "numpy") for e, ex in bc_obj["edges"].items()
            }

    x_grid = np.linspace(0.0, Lx, 65, dtype=np.float32)
    y_grid = np.linspace(0.0, Ly, 65, dtype=np.float32)
    u0_grid: Optional[np.ndarray] = None

    if prob_npz_path.exists():
        p_npz = np.load(prob_npz_path, allow_pickle=True)
        for k in p_npz.files:
            inspection["problem_npz_keys"][k] = {
                "shape": list(p_npz[k].shape),
                "dtype": str(p_npz[k].dtype),
            }
        if "Lx" in p_npz.files:
            Lx = float(p_npz["Lx"])
        if "Ly" in p_npz.files:
            Ly = float(p_npz["Ly"])
        if "T" in p_npz.files:
            T_max = float(p_npz["T"])
        if "Fo" in p_npz.files:
            Fo_val = float(p_npz["Fo"])
        for xk in ("x_inv", "x", "X", "x_grid", "xs", "x_data"):
            if xk in p_npz.files:
                arr = p_npz[xk]
                x_grid = np.unique(arr).astype(np.float32)
                Lx = float(x_grid.max())
                break
        for yk in ("y_inv", "y", "Y", "y_grid", "ys", "y_data"):
            if yk in p_npz.files:
                arr = p_npz[yk]
                y_grid = np.unique(arr).astype(np.float32)
                Ly = float(y_grid.max())
                break
        for u0k in ("u0_inv", "u0", "ic", "u_ic", "u_init"):
            if u0k in p_npz.files:
                u0_grid = p_npz[u0k].astype(np.float32)
                break

    X_ic_np: Optional[np.ndarray] = None
    u_ic_np: Optional[np.ndarray] = None
    if prob_ic_csv_path.exists():
        ic_df = pd.read_csv(prob_ic_csv_path)
        inspection["ic_csv_columns"] = list(ic_df.columns)
        x_col = next((c for c in ic_df.columns if c.lower() in ("x", "x_coord")), ic_df.columns[0])
        y_col = next((c for c in ic_df.columns if c.lower() in ("y", "y_coord")), ic_df.columns[1])
        u_col = next(
            (c for c in ic_df.columns if c.lower() in ("u0", "u", "u_ic", "ic", "val", "temperature")),
            ic_df.columns[-1],
        )
        x_ic = ic_df[x_col].to_numpy(dtype=np.float32)[:, None]
        y_ic = ic_df[y_col].to_numpy(dtype=np.float32)[:, None]
        t_ic = np.zeros_like(x_ic, dtype=np.float32)
        X_ic_np = np.hstack([x_ic, y_ic, t_ic])
        u_ic_np = ic_df[u_col].to_numpy(dtype=np.float32)[:, None]
        if u0_grid is None and len(u_ic_np) == len(x_grid) * len(y_grid):
            u0_grid = u_ic_np.reshape(len(x_grid), len(y_grid))
    elif u0_grid is not None:
        XX, YY = np.meshgrid(x_grid, y_grid, indexing="ij")
        if u0_grid.shape == XX.shape:
            X_ic_np = np.stack([XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())], axis=-1).astype(np.float32)
            u_ic_np = u0_grid.ravel()[:, None].astype(np.float32)

    gt_npz_path = ds_dir / "ground_truth" / f"{prob_stem}.npz"
    D_true: Optional[np.ndarray] = None
    u_true_full: Optional[np.ndarray] = None
    if gt_npz_path.exists():
        gt_npz = np.load(gt_npz_path, allow_pickle=True)
        for k in gt_npz.files:
            inspection["ground_truth_npz_keys"][k] = {
                "shape": list(gt_npz[k].shape),
                "dtype": str(gt_npz[k].dtype),
            }
        for dk in ("D_inv", "D", "D_true", "D_exact", "diffusivity", "D_data", "k"):
            if dk in gt_npz.files:
                D_true = gt_npz[dk].astype(np.float32)
                break
        if D_true is None:
            for k in gt_npz.files:
                if gt_npz[k].ndim == 2:
                    D_true = gt_npz[k].astype(np.float32)
                    break
        for uk in ("u", "U", "u_true", "u_exact", "trajectory"):
            if uk in gt_npz.files and gt_npz[uk].ndim == 3:
                u_true_full = gt_npz[uk].astype(np.float32)
                break

    sens_csv_path = ds_dir / "sensors" / f"{cfg.exp_id}.csv"
    sens_npz_path = ds_dir / "sensors" / f"{cfg.exp_id}.npz"
    X_data_np: Optional[np.ndarray] = None
    u_data_np: Optional[np.ndarray] = None
    sigma_np: Optional[np.ndarray] = None
    sensor_xy_unique: Optional[np.ndarray] = None
    sigma_sq: float = 2.36e-6

    if sens_npz_path.exists():
        s_npz = np.load(sens_npz_path, allow_pickle=True)
        for k in s_npz.files:
            inspection["sensor_npz_keys"][k] = {
                "shape": list(s_npz[k].shape),
                "dtype": str(s_npz[k].dtype),
            }
        if "sigma" in s_npz.files:
            sig_arr = s_npz["sigma"].astype(np.float32).ravel()[:, None]
            sigma_np = sig_arr
            sigma_sq = float(np.mean(sig_arr.astype(np.float64) ** 2))

    if sens_csv_path.exists():
        s_df = pd.read_csv(sens_csv_path)
        inspection["sensor_csv_columns"] = list(s_df.columns)
        cols_lower = {c.lower(): c for c in s_df.columns}
        if "sigma" in cols_lower:
            sig_col = s_df[cols_lower["sigma"]].to_numpy(dtype=np.float32)[:, None]
            sigma_np = sig_col
            sigma_sq = float(np.mean(sig_col.astype(np.float64) ** 2))
        if "x" in cols_lower and "y" in cols_lower and "t" in cols_lower:
            xc = cols_lower["x"]
            yc = cols_lower["y"]
            tc = cols_lower["t"]
            uc = None
            for cand in ("u_noisy", "u_obs", "y_obs", "u_meas", "val", "u"):
                if cand in cols_lower:
                    uc = cols_lower[cand]
                    break
            if uc is None:
                uc = s_df.columns[-1]
            X_data_np = s_df[[xc, yc, tc]].to_numpy(dtype=np.float32)
            u_data_np = s_df[[uc]].to_numpy(dtype=np.float32)
            T_max = max(T_max, float(X_data_np[:, 2].max()))
            sensor_xy_unique = np.unique(X_data_np[:, :2], axis=0)

    if X_data_np is None and sens_npz_path.exists():
        s_npz = np.load(sens_npz_path, allow_pickle=True)
        if "x" in s_npz.files and "y" in s_npz.files and "t" in s_npz.files and "u" in s_npz.files:
            X_data_np = np.stack(
                [s_npz["x"].ravel(), s_npz["y"].ravel(), s_npz["t"].ravel()], axis=-1
            ).astype(np.float32)
            u_data_np = s_npz["u"].ravel()[:, None].astype(np.float32)
            sensor_xy_unique = np.unique(X_data_np[:, :2], axis=0)

    D_adjoint: Optional[np.ndarray] = None
    baseline_search_dirs = [
        v2_root / "results_adjoint" / cfg.exp_id,
        v2_root / "results_adjoint_tv" / cfg.exp_id,
        v2_root / "results_defects_tv" / cfg.exp_id,
        v2_root / "results_defects_tikhonov" / cfg.exp_id,
        v2_root / "results_test" / cfg.exp_id,
    ] + [v2_root / "results_tikhonov" / f"part{i}" / cfg.exp_id for i in range(6)]

    for bdir in baseline_search_dirs:
        if bdir.exists():
            m_file = bdir / "metrics.json"
            d_file = bdir / "D_hat.npz"
            if m_file.exists():
                with open(m_file, "r") as f:
                    inspection["baseline_metrics"] = json.load(f)
            if d_file.exists():
                d_npz = np.load(d_file, allow_pickle=True)
                for k in ("D_hat", "D", "D_pred", "D_rec"):
                    if k in d_npz.files:
                        D_adjoint = d_npz[k].astype(np.float32)
                        break
                if D_adjoint is None and len(d_npz.files) > 0:
                    D_adjoint = d_npz[d_npz.files[0]].astype(np.float32)
            break

    if X_data_np is None or u_data_np is None:
        print(f"[WARN] Could not find sensor files for {cfg.exp_id} under {v2_root}. Using synthetic fallback.")
        rng = np.random.default_rng(cfg.seed)
        XX, YY = np.meshgrid(x_grid, y_grid, indexing="ij")
        D_true = (0.08 + 0.42 * np.exp(-((XX - 0.5) ** 2 + (YY - 0.5) ** 2) / (2 * 0.18**2))).astype(np.float32)
        n_pts = 400
        xs = rng.uniform(0.05, 0.95 * Lx, size=(n_pts, 1)).astype(np.float32)
        ys = rng.uniform(0.05, 0.95 * Ly, size=(n_pts, 1)).astype(np.float32)
        ts = rng.uniform(0.05 * T_max, T_max, size=(n_pts, 1)).astype(np.float32)
        us = (0.08 * np.sin(np.pi * xs / Lx) * np.sin(np.pi * ys / Ly) * np.exp(-1.5 * ts)).astype(np.float32)
        X_data_np = np.hstack([xs, ys, ts])
        u_data_np = us

    if X_ic_np is None or u_ic_np is None:
        XX, YY = np.meshgrid(x_grid, y_grid, indexing="ij")
        g1 = 1.0 * np.exp(-((XX - 0.3) ** 2 + (YY - 0.3) ** 2) / (2 * 0.05**2))
        g2 = 0.7 * np.exp(-((XX - 0.7) ** 2 + (YY - 0.65) ** 2) / (2 * 0.06**2))
        u0_synth = (g1 + g2).astype(np.float32)
        u0_synth[0, :] = u0_synth[-1, :] = u0_synth[:, 0] = u0_synth[:, -1] = 0.0
        u0_grid = u0_synth
        X_ic_np = np.stack([XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())], axis=-1).astype(np.float32)
        u_ic_np = u0_synth.ravel()[:, None]

    if sigma_np is None:
        sigma_np = np.full_like(u_data_np, math.sqrt(sigma_sq), dtype=np.float32)

    D_char = float(Fo_val * (max(Lx, Ly) ** 2) / max(T_max, 1e-6))
    m_init_bias = float(math.log(max(0.85 * D_char, 1e-4)))
    if getattr(cfg, "m_init", "fo") == "best_constant":
        m_init_bias = _best_constant_logD(ds_dir, cfg.exp_id, m_init_bias)

    return {
        "X_data": torch.tensor(X_data_np, dtype=torch.float32),
        "u_data": torch.tensor(u_data_np, dtype=torch.float32),
        "sigma_data": torch.tensor(sigma_np, dtype=torch.float32),
        "X_ic": torch.tensor(X_ic_np, dtype=torch.float32),
        "u_ic": torch.tensor(u_ic_np, dtype=torch.float32),
        "u0_grid": u0_grid,
        "D_true": D_true,
        "D_adjoint": D_adjoint,
        "u_true_full": u_true_full,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "Lx": Lx,
        "Ly": Ly,
        "T_max": T_max,
        "Fo": Fo_val,
        "D_char": D_char,
        "m_init_bias": m_init_bias,
        "sigma_sq": sigma_sq,
        "bc_type": bc_type_detected if cfg.bc_type == "auto" else cfg.bc_type,
        "bc_edge_funcs": bc_edge_funcs,
        "sensor_xy": sensor_xy_unique,
        "inspection": inspection,
    }


# =====================================================================
# 4. Dual-Network Architecture: Field u_theta(x,y,t) & Log-Diffusivity m_phi(x,y)
# =====================================================================
class FourierMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        use_fourier: bool = True,
        fourier_scale: float = 0.5,
        init_bias: float = 0.0,
        out_gain: float = 0.02,
    ):
        super().__init__()
        self.use_fourier = use_fourier
        eff_in = in_dim
        if use_fourier:
            B = torch.randn(in_dim, hidden_dim // 2) * fourier_scale
            self.register_buffer("fourier_B", B)
            eff_in = hidden_dim

        layers: List[nn.Module] = []
        d_in = eff_in
        for _ in range(num_layers - 1):
            lin = nn.Linear(d_in, hidden_dim)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(nn.Tanh())
            d_in = hidden_dim

        out_lin = nn.Linear(d_in, out_dim)
        nn.init.xavier_normal_(out_lin.weight, gain=out_gain)
        nn.init.constant_(out_lin.bias, init_bias)
        layers.append(out_lin)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fourier:
            proj = 2.0 * math.pi * (x @ self.fourier_B)
            x = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(x)


class InverseDiffusionPINN(nn.Module):
    """
    Two-network inverse PINN (Slide 14):
      - u_net(x, y, t) : predicts temperature field u_theta(x, y, t)
                         around the 2D diffusing Gaussian heat-kernel base solution
      - m_net(x, y)    : predicts log-diffusivity m_phi(x, y) = log D(x, y),
                         so D_phi(x, y) = exp(m_phi(x, y)) > 0 everywhere.
    """

    def __init__(
        self,
        cfg: PINNConfig,
        Lx: float,
        Ly: float,
        T_max: float,
        m_init_bias: float = -2.14,
        bc_type: str = "dirichlet",
    ):
        super().__init__()
        self.Lx = float(max(Lx, 1e-6))
        self.Ly = float(max(Ly, 1e-6))
        self.T_max = float(max(T_max, 1e-6))
        self.bc_type = bc_type
        self.register_buffer("scale_xyt", torch.tensor([self.Lx, self.Ly, self.T_max]))
        self.register_buffer("scale_xy", torch.tensor([self.Lx, self.Ly]))
        self.m_bg = nn.Parameter(torch.tensor(float(m_init_bias)))
        self.u_base_log_d = nn.Parameter(torch.tensor(float(m_init_bias)))
        self.m_wall_taper = bool(getattr(cfg, "m_wall_taper", True))

        self.u_net = FourierMLP(
            in_dim=3,
            out_dim=1,
            hidden_dim=cfg.u_hidden_dim,
            num_layers=cfg.u_num_layers,
            use_fourier=cfg.use_fourier_features,
            fourier_scale=cfg.u_fourier_scale,
            init_bias=0.0,
            out_gain=0.02,
        )
        self.m_net = FourierMLP(
            in_dim=2,
            out_dim=1,
            hidden_dim=cfg.m_hidden_dim,
            num_layers=cfg.m_num_layers,
            use_fourier=cfg.use_fourier_features,
            fourier_scale=cfg.m_fourier_scale,
            init_bias=0.0,
            out_gain=0.01,
        )

    def diffusing_heat_kernel(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        D_eff = torch.exp(torch.clamp(self.u_base_log_d, min=-4.5, max=0.0))
        s1_sq = 0.05**2
        s2_sq = 0.06**2
        v1 = s1_sq + 2.0 * D_eff * t
        v2 = s2_sq + 2.0 * D_eff * t
        g1 = 1.0 * (s1_sq / v1) * torch.exp(-((x - 0.30) ** 2 + (y - 0.30) ** 2) / (2.0 * v1))
        g2 = 0.7 * (s2_sq / v2) * torch.exp(-((x - 0.70) ** 2 + (y - 0.65) ** 2) / (2.0 * v2))
        return g1 + g2

    def forward_u(self, xyt: torch.Tensor) -> torch.Tensor:
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]
        raw_u = self.u_net(xyt / self.scale_xyt)
        if self.bc_type == "dirichlet":
            xn = x / self.Lx
            yn = y / self.Ly
            bc_win = 16.0 * xn * (1.0 - xn) * yn * (1.0 - yn)
            edge_taper = (
                torch.tanh(25.0 * xn)
                * torch.tanh(25.0 * (1.0 - xn))
                * torch.tanh(25.0 * yn)
                * torch.tanh(25.0 * (1.0 - yn))
            )
            u_base = self.diffusing_heat_kernel(x, y, t) * edge_taper
            t_gate = torch.tanh(8.0 * t / self.T_max)
            return u_base + t_gate * bc_win * raw_u
        if self.bc_type in ("dirichlet_soft", "free"):
            # IC still built in at t = 0; walls left to the BC loss (soft) or to the data (free)
            u_base = self.diffusing_heat_kernel(x, y, t)
            t_gate = torch.tanh(8.0 * t / self.T_max)
            return u_base + t_gate * raw_u
        return raw_u

    def forward_m(self, xy: torch.Tensor) -> torch.Tensor:
        xn = xy[:, 0:1] / self.Lx
        yn = xy[:, 1:2] / self.Ly
        interior_taper = (
            torch.tanh(6.0 * xn)
            * torch.tanh(6.0 * (1.0 - xn))
            * torch.tanh(6.0 * yn)
            * torch.tanh(6.0 * (1.0 - yn))
        )
        if not self.m_wall_taper:
            interior_taper = 1.0
        delta_m = interior_taper * self.m_net(xy / self.scale_xy)
        return self.m_bg + delta_m

    def forward_D(self, xy: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.clamp(self.forward_m(xy), min=-5.0, max=2.0))


# =====================================================================
# 5. Physics Residual, Boundary, Initial, and Regularization Losses
# =====================================================================
def d_wrt(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def compute_pde_and_reg_loss(
    model: InverseDiffusionPINN,
    X_pde: torch.Tensor,
    cfg: PINNConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    2D Divergence-Form Diffusion Residual:
        R_pde = u_t - div( D_phi(x, y) * grad_xy u_theta(x, y, t) )
              = u_t - [ D_x * u_x + D_y * u_y + D * (u_xx + u_yy) ]
    Returns:
        loss_pde_raw : raw unweighted mean(residual^2) for logging
        loss_pde_opt : time-balanced, asymmetric-gradient PDE loss so m_phi receives
                       85% of the PDE residual gradient without u_net distorting laplacian(u)
        loss_reg     : Tikhonov H1 or Total Variation regularization on m_phi
    """
    x = X_pde[:, 0:1].requires_grad_(True)
    y = X_pde[:, 1:2].requires_grad_(True)
    t = X_pde[:, 2:3].requires_grad_(True)

    xyt = torch.cat([x, y, t], dim=1)
    xy = torch.cat([x, y], dim=1)

    u = model.forward_u(xyt)
    m = torch.clamp(model.forward_m(xy), min=-5.0, max=2.0)
    D = torch.exp(m)

    u_x, u_y, u_t = torch.autograd.grad(
        outputs=u,
        inputs=[x, y, t],
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )

    u_xx = d_wrt(u_x, x)
    u_yy = d_wrt(u_y, y)

    m_x, m_y = torch.autograd.grad(
        outputs=m,
        inputs=[x, y],
        grad_outputs=torch.ones_like(m),
        create_graph=True,
        retain_graph=True,
    )
    D_x = D * m_x
    D_y = D * m_y

    residual_raw = u_t - (D_x * u_x + D_y * u_y + D * (u_xx + u_yy))
    loss_pde_raw = torch.mean(residual_raw**2)

    # Time-scale compensation for 2D heat kernel decay so sensor times t in [0.09, 0.72]
    # contribute strongly to discovering D(x, y)
    t_norm = t / model.T_max
    time_scale = torch.clamp((0.02 + t_norm) / 0.12, min=0.25, max=8.0).detach()

    # Asymmetric gradient routing: m_net receives 90% direct PDE sensitivity against detached u derivatives;
    # u_net receives 10% PDE smoothing against detached D so u_net stays anchored to sensor measurements.
    res_for_m = u_t.detach() - (
        D_x * u_x.detach() + D_y * u_y.detach() + D * (u_xx.detach() + u_yy.detach())
    )
    res_for_u = u_t - (
        D_x.detach() * u_x + D_y.detach() * u_y + D.detach() * (u_xx + u_yy)
    )
    loss_pde_opt = 0.90 * torch.mean((time_scale * res_for_m) ** 2) + 0.10 * torch.mean(
        (time_scale * res_for_u) ** 2
    )

    grad_m_sq = m_x**2 + m_y**2
    delta_m_sq = (m - model.m_bg) ** 2
    if cfg.reg_type == "tikhonov":
        loss_reg = torch.mean(grad_m_sq + 0.05 * delta_m_sq)
    elif cfg.reg_type == "tv":
        loss_reg = torch.mean(torch.sqrt(grad_m_sq + cfg.tv_eps) + 0.02 * delta_m_sq)
    else:
        loss_reg = torch.zeros(1, device=X_pde.device).squeeze()

    return loss_pde_raw, loss_pde_opt, loss_reg


def compute_bc_loss(
    model: InverseDiffusionPINN,
    Lx: float,
    Ly: float,
    T_max: float,
    n_bc: int,
    bc_type: str,
    device: torch.device,
    bc_edge_funcs=None,
) -> torch.Tensor:
    if bc_type == "free":
        return torch.zeros((), device=device)
    n_edge = max(1, n_bc // 4)
    s = torch.rand(n_edge, 1, device=device)
    t_e = T_max * torch.rand(n_edge * 4, 1, device=device)

    xy_left = torch.cat([torch.zeros_like(s), Ly * s], dim=1)
    xy_right = torch.cat([torch.full_like(s, Lx), Ly * s], dim=1)
    xy_bot = torch.cat([Lx * s, torch.zeros_like(s)], dim=1)
    xy_top = torch.cat([Lx * s, torch.full_like(s, Ly)], dim=1)
    xy_all = torch.cat([xy_left, xy_right, xy_bot, xy_top], dim=0)

    if bc_type == "dirichlet":
        X_bc = torch.cat([xy_all, t_e], dim=1)
        u_bc_pred = model.forward_u(X_bc)
        return torch.mean(u_bc_pred**2)
    if bc_type == "dirichlet_soft":
        X_bc = torch.cat([xy_all, t_e], dim=1)
        u_bc_pred = model.forward_u(X_bc)
        t_np = t_e.detach().cpu().numpy().ravel()
        g = np.concatenate([
            np.broadcast_to(np.asarray(bc_edge_funcs[e](t_np[k * n_edge:(k + 1) * n_edge]), dtype=np.float64), (n_edge,))
            for k, e in enumerate(("left", "right", "bottom", "top"))
        ])
        g_t = torch.tensor(g, dtype=u_bc_pred.dtype, device=device).view(-1, 1)
        return torch.mean((u_bc_pred - g_t) ** 2)

    x_b = xy_all[:, 0:1].requires_grad_(True)
    y_b = xy_all[:, 1:2].requires_grad_(True)
    X_bc = torch.cat([x_b, y_b, t_e], dim=1)
    u_b = model.forward_u(X_bc)
    u_x = d_wrt(u_b, x_b)
    u_y = d_wrt(u_b, y_b)
    flux_lr = u_x[: 2 * n_edge]
    flux_bt = u_y[2 * n_edge :]
    return 0.5 * (torch.mean(flux_lr**2) + torch.mean(flux_bt**2))


# =====================================================================
# 6. Trusted Forward Solver Evaluation for Slide 16 Metric (chi2_red)
# =====================================================================
@torch.no_grad()
def evaluate_slide16_chi2_metric(
    D_hat: np.ndarray,
    model: InverseDiffusionPINN,
    bundle: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Computes the Slide 16 Displayed Performance Metric:
        chi2_red = (1 / N_d) * sum_{i=1}^{N_d} ( (u(D_hat)(x_i, y_i, t_i) - u_obs_i) / sigma_i )^2
    using both:
      1. Trusted 2D finite-difference RK4 forward simulation with the recovered D_hat(x, y)
      2. The trained PINN field network u_theta(x_i, y_i, t_i)
    """
    X_data = bundle["X_data"].to(device)
    u_obs = bundle["u_data"].to(device).squeeze(-1)
    sigma = bundle["sigma_data"].to(device).squeeze(-1)

    # 1. PINN state network chi2_red(u_theta)
    u_pinn = model.forward_u(X_data).squeeze(-1)
    chi2_u_net = float(torch.mean(((u_pinn - u_obs) / (sigma + 1e-12)) ** 2).item())

    # 2. Trusted 2D Finite-Difference RK4 Forward Solver with recovered D_hat(x, y)
    x_grid = bundle["x_grid"]
    y_grid = bundle["y_grid"]
    Lx, Ly, T_max = bundle["Lx"], bundle["Ly"], bundle["T_max"]
    u0_grid = bundle.get("u0_grid")
    if u0_grid is None or u0_grid.shape != D_hat.shape:
        XX, YY = np.meshgrid(x_grid, y_grid, indexing="ij")
        g1 = 1.0 * np.exp(-((XX - 0.3) ** 2 + (YY - 0.3) ** 2) / (2 * 0.05**2))
        g2 = 0.7 * np.exp(-((XX - 0.7) ** 2 + (YY - 0.65) ** 2) / (2 * 0.06**2))
        u0_grid = (g1 + g2).astype(np.float32)
        u0_grid[0, :] = u0_grid[-1, :] = u0_grid[:, 0] = u0_grid[:, -1] = 0.0

    nx, ny = D_hat.shape
    dx = float(Lx / max(nx - 1, 1))
    dy = float(Ly / max(ny - 1, 1))
    D_t = torch.tensor(np.clip(D_hat, 1e-4, 5.0), dtype=torch.float32, device=device)
    u_cur = torch.tensor(u0_grid, dtype=torch.float32, device=device)

    # Harmonic/arithmetic face averages D_{i+1/2, j} and D_{i, j+1/2}
    Dx_face = 0.5 * (D_t[1:, :] + D_t[:-1, :])  # (nx-1, ny)
    Dy_face = 0.5 * (D_t[:, 1:] + D_t[:, :-1])  # (nx, ny-1)

    def rhs_div_D_grad_u(u_field: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(u_field)
        flux_x = Dx_face[:, 1:-1] * (u_field[1:, 1:-1] - u_field[:-1, 1:-1]) / (dx * dx)
        flux_y = Dy_face[1:-1, :] * (u_field[1:-1, 1:] - u_field[1:-1, :-1]) / (dy * dy)
        out[1:-1, 1:-1] = (flux_x[1:, :] - flux_x[:-1, :]) + (flux_y[:, 1:] - flux_y[:, :-1])
        return out

    d_max = float(D_t.max().item())
    dt_cfl = 0.20 * min(dx * dx, dy * dy) / max(d_max, 1e-4)
    n_steps = max(200, min(6000, int(math.ceil(T_max / dt_cfl))))
    dt = T_max / n_steps

    # Sensor timestamps & normalized coordinates for bilinear grid_sample
    t_obs_np = X_data[:, 2].cpu().numpy()
    unique_t = np.unique(np.round(t_obs_np, 7))
    u_sim_at_obs = torch.zeros_like(u_obs)

    # grid_sample expects coordinates in [-1, 1] with (y, x) ordering for (1, 1, nx, ny)
    x_norm = 2.0 * (X_data[:, 0] / Lx) - 1.0
    y_norm = 2.0 * (X_data[:, 1] / Ly) - 1.0
    sample_grid = torch.stack([y_norm, x_norm], dim=-1).view(1, -1, 1, 2)

    t_cur = 0.0
    for step in range(1, n_steps + 1):
        k1 = rhs_div_D_grad_u(u_cur)
        k2 = rhs_div_D_grad_u(u_cur + 0.5 * dt * k1)
        k3 = rhs_div_D_grad_u(u_cur + 0.5 * dt * k2)
        k4 = rhs_div_D_grad_u(u_cur + dt * k3)
        u_cur = u_cur + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        u_cur[0, :] = u_cur[-1, :] = u_cur[:, 0] = u_cur[:, -1] = 0.0
        t_next = step * dt

        # Check if any sensor snapshot falls in (t_cur, t_next]
        for ut in unique_t:
            if t_cur <= ut <= t_next + 0.5 * dt:
                mask = torch.tensor(np.abs(t_obs_np - ut) < 1e-5, device=device)
                if mask.any():
                    interp_all = F.grid_sample(
                        u_cur.view(1, 1, nx, ny),
                        sample_grid,
                        mode="bilinear",
                        align_corners=True,
                    ).view(-1)
                    u_sim_at_obs[mask] = interp_all[mask]
        t_cur = t_next

    chi2_solver = float(torch.mean(((u_sim_at_obs - u_obs) / (sigma + 1e-12)) ** 2).item())
    chi2_dist_from_1 = abs(chi2_solver - 1.0)
    u_Net_chi2_dist_from_1 = abs(chi2_u_net - 1.0)

    if chi2_u_net < 0.75:
        regime = "Over-fitting (<1: fitting noise)"
    elif chi2_u_net <= 2.50:
        regime = "Optimal (~1: matches noise floor)"
    else:
        regime = "Under-fitting (>1: over-smoothed)"

    return {
        "chi2_red_solver": round(chi2_solver, 4),
        "chi2_red_u_net": round(chi2_u_net, 4),
        "chi2_solver_dist_from_1": round(chi2_dist_from_1, 4),
        "chi2_u_net_dist_from_1": round(u_Net_chi2_dist_from_1, 4),
        "slide16_regime": regime,
    }


# =====================================================================
# 7. Single Configuration Training & Evaluation
# =====================================================================
def compute_field_metrics(D_pred: np.ndarray, D_true: np.ndarray) -> Dict[str, float]:
    diff = D_pred - D_true
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(D_true) + 1e-12))
    rel_l1 = float(np.sum(np.abs(diff)) / (np.sum(np.abs(D_true)) + 1e-12))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_err = float(np.max(np.abs(diff)))
    return {
        "rel_l2_D": round(rel_l2, 5),
        "rel_l1_D": round(rel_l1, 5),
        "mae_D": round(mae, 5),
        "rmse_D": round(rmse, 5),
        "max_err_D": round(max_err, 5),
    }


def train_single_experiment(cfg: PINNConfig, run_tag: Optional[str] = None) -> Dict[str, Any]:
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_out_dir = Path(cfg.output_dir) / (run_tag if run_tag else cfg.exp_id)
    exp_out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_v2_experiment(cfg)
    with open(exp_out_dir / "v2_inspection_report.json", "w") as f:
        json.dump(bundle["inspection"], f, indent=2)

    Lx, Ly, T_max = bundle["Lx"], bundle["Ly"], bundle["T_max"]
    bc_type = bundle["bc_type"]
    x_grid, y_grid = bundle["x_grid"], bundle["y_grid"]
    D_true, D_adjoint = bundle["D_true"], bundle["D_adjoint"]

    X_data_all = bundle["X_data"].to(device)
    u_data_all = bundle["u_data"].to(device)
    X_ic_all = bundle["X_ic"].to(device)
    u_ic_all = bundle["u_ic"].to(device)

    print(
        f"\n=== [{run_tag or cfg.exp_id}] (device={device}) Fixed Lambdas: "
        f"data={cfg.lambda_data}, pde={cfg.lambda_pde}, bc={cfg.lambda_bc}, "
        f"ic={cfg.lambda_ic}, reg={cfg.lambda_reg} ({cfg.reg_type}) ===",
        flush=True,
    )

    model = InverseDiffusionPINN(
        cfg, Lx, Ly, T_max, m_init_bias=bundle["m_init_bias"], bc_type=bc_type
    ).to(device)

    if cfg.warmstart_from_adjoint and D_adjoint is not None:
        XX, YY = np.meshgrid(x_grid, y_grid, indexing="ij")
        if D_adjoint.shape == XX.shape:
            xy_adj = torch.tensor(
                np.stack([XX.ravel(), YY.ravel()], axis=-1), dtype=torch.float32, device=device
            )
            m_adj = torch.tensor(
                np.log(np.clip(D_adjoint.ravel(), 1e-4, None))[:, None],
                dtype=torch.float32,
                device=device,
            )
            ws_opt = torch.optim.Adam(list(model.m_net.parameters()) + [model.m_bg], lr=2e-3)
            for _ in range(300):
                ws_opt.zero_grad()
                ws_loss = torch.mean((model.forward_m(xy_adj) - m_adj) ** 2)
                ws_loss.backward()
                ws_opt.step()

    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.u_net.parameters()) + [model.u_base_log_d], "lr": cfg.lr_u},
            {"params": list(model.m_net.parameters()) + [model.m_bg], "lr": cfg.lr_m},
        ],
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)
    mse = nn.MSELoss()
    u_data_msq = float(torch.mean(u_data_all**2).item()) + 1e-6
    u_ic_msq = float(torch.mean(u_ic_all**2).item()) + 1e-6
    sigma_sq = float(bundle.get("sigma_sq", 2.36e-6))

    lambdas = {
        "data": cfg.lambda_data,
        "pde": cfg.lambda_pde,
        "bc": cfg.lambda_bc,
        "ic": cfg.lambda_ic,
        "reg": cfg.lambda_reg,
    }

    history: List[Dict[str, float]] = []
    best_total = float("inf")
    warmup_m_epochs = int(0.15 * cfg.epochs)

    def _losses(polish=False):
            # 1. Sensor Data Mismatch Loss (L_data)
            n_obs = X_data_all.shape[0]
            if n_obs > cfg.batch_size_data and not polish:
                idx_d = torch.randint(0, n_obs, (cfg.batch_size_data,), device=device)
                Xd, ud = X_data_all[idx_d], u_data_all[idx_d]
            else:
                Xd, ud = X_data_all, u_data_all
            loss_data = mse(model.forward_u(Xd), ud)

            # 2. Initial Condition Loss (L_ic)
            n_ic_tot = X_ic_all.shape[0]
            if n_ic_tot > cfg.n_ic:
                idx_ic = torch.randint(0, n_ic_tot, (cfg.n_ic,), device=device)
                Xic, uic = X_ic_all[idx_ic], u_ic_all[idx_ic]
            else:
                Xic, uic = X_ic_all, u_ic_all
            loss_ic = mse(model.forward_u(Xic), uic)

            # 3. Boundary Condition Loss (L_bc)
            loss_bc = compute_bc_loss(model, Lx, Ly, T_max, cfg.n_bc, bc_type, device, bundle.get("bc_edge_funcs"))

            # 4. Interior + Sensor-Neighborhood Collocation PDE Residual Loss (L_pde)
            n_col = cfg.n_collocation * (cfg.lbfgs_colloc_mult if polish else 1)
            n_focus = int(0.60 * n_col)
            n_rest = n_col - n_focus
            x_focus = 0.05 * Lx + 0.90 * Lx * torch.rand(n_focus, 1, device=device)
            y_focus = 0.05 * Ly + 0.90 * Ly * torch.rand(n_focus, 1, device=device)
            t_focus = T_max * (0.02 + 0.65 * (torch.rand(n_focus, 1, device=device) ** 1.4))
            x_rest = Lx * torch.rand(n_rest, 1, device=device)
            y_rest = Ly * torch.rand(n_rest, 1, device=device)
            t_rest = T_max * torch.rand(n_rest, 1, device=device)
            X_pde = torch.cat(
                [
                    torch.cat([x_focus, y_focus, t_focus], dim=1),
                    torch.cat([x_rest, y_rest, t_rest], dim=1),
                    Xd.detach(),
                ],
                dim=0,
            )
            loss_pde_raw, loss_pde_opt, loss_reg = compute_pde_and_reg_loss(model, X_pde, cfg)

            if cfg.morozov_damping:
                data_term = (torch.relu(loss_data - 0.95 * sigma_sq) + 0.05 * loss_data) / u_data_msq
            else:
                data_term = loss_data / u_data_msq

            # 5. Composite PINN Loss with fixed multipliers:
            loss_total = (
                lambdas["data"] * data_term
                + lambdas["pde"] * loss_pde_opt
                + lambdas["bc"] * loss_bc
                + lambdas["ic"] * (loss_ic / u_ic_msq)
                + lambdas["reg"] * loss_reg
            )
            return loss_total, loss_data, loss_pde_raw, loss_bc, loss_ic, loss_reg

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Smooth learning-rate warmup on m_net over first 15% of epochs + cosine annealing to 1e-5
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(1, cfg.epochs)))
        m_lr_scale = min(1.0, max(0.02, (epoch - 50) / max(1, warmup_m_epochs)))
        optimizer.param_groups[0]["lr"] = 1e-5 + (cfg.lr_u - 1e-5) * cos_decay
        optimizer.param_groups[1]["lr"] = (1e-5 + (cfg.lr_m - 1e-5) * cos_decay) * m_lr_scale

        loss_total, loss_data, loss_pde_raw, loss_bc, loss_ic, loss_reg = _losses()

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        row = {
            "epoch": epoch,
            "loss_total": float(loss_total.item()),
            "loss_data": float(loss_data.item()),
            "loss_pde": float(loss_pde_raw.item()),
            "loss_bc": float(loss_bc.item()),
            "loss_ic": float(loss_ic.item()),
            "loss_reg": float(loss_reg.item()),
            "lambda_data": float(lambdas["data"]),
            "lambda_pde": float(lambdas["pde"]),
            "lambda_bc": float(lambdas["bc"]),
            "lambda_ic": float(lambdas["ic"]),
            "lambda_reg": float(lambdas["reg"]),
        }
        history.append(row)

        if epoch > int(0.65 * cfg.epochs) and row["loss_total"] < best_total:
            best_total = row["loss_total"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "u_net_state_dict": model.u_net.state_dict(),
                    "m_net_state_dict": model.m_net.state_dict(),
                    "config": asdict(cfg),
                    "metrics": row,
                },
                exp_out_dir / "pinn_best.pt",
            )

        if epoch == 1 or epoch % max(1, cfg.epochs // 5) == 0 or epoch == cfg.epochs:
            print(
                f"  Epoch [{epoch:04d}/{cfg.epochs:04d}] "
                f"L_tot={row['loss_total']:.3e} | "
                f"L_data={row['loss_data']:.3e} | "
                f"L_pde={row['loss_pde']:.3e} | "
                f"L_reg={row['loss_reg']:.3e}",
                flush=True,
            )

    if cfg.lbfgs_iters > 0:
        best_ckpt_path = exp_out_dir / "pinn_best.pt"
        if best_ckpt_path.exists():
            model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
        lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=cfg.lbfgs_iters, history_size=50,
                               line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)
        def _closure():
            lb.zero_grad()
            torch.manual_seed(cfg.seed + 777)      # identical batch on every evaluation (deterministic objective)
            lt = _losses(polish=True)[0]
            lt.backward()
            return lt
        l0 = float(_closure().item())
        lb.step(_closure)
        l1 = float(_closure().item())
        print(f"  [L-BFGS] {cfg.lbfgs_iters} iters: loss {l0:.4e} -> {l1:.4e}", flush=True)
        if np.isfinite(l1) and l1 <= l0:
            torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, best_ckpt_path)

    # Restore best late-stage checkpoint so final evaluation is immune to single-batch collocation spikes
    best_ckpt_path = exp_out_dir / "pinn_best.pt"
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # Evaluate D_hat(x, y) on the 65x65 inversion grid (x_grid, y_grid)
    model.eval()
    XX, YY = np.meshgrid(x_grid, y_grid, indexing="ij")
    xy_eval = torch.tensor(
        np.stack([XX.ravel(), YY.ravel()], axis=-1), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        D_hat_pinn = model.forward_D(xy_eval).cpu().numpy().reshape(XX.shape)

    np.savez_compressed(
        exp_out_dir / "D_hat.npz",
        D_hat=D_hat_pinn,
        x=x_grid,
        y=y_grid,
        D_true=D_true if D_true is not None else np.zeros_like(D_hat_pinn),
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "u_net_state_dict": model.u_net.state_dict(),
            "m_net_state_dict": model.m_net.state_dict(),
            "config": asdict(cfg),
        },
        exp_out_dir / "pinn_final.pt",
    )

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(exp_out_dir / "loss_history.csv", index=False)

    # Compute Slide 16 Chi-Squared Discrepancy Metric & Ground-Truth Diffusivity Error Metrics
    slide16_metrics = evaluate_slide16_chi2_metric(D_hat_pinn, model, bundle, device)
    elapsed_sec = time.time() - t0

    metrics_out: Dict[str, Any] = {
        "exp_id": cfg.exp_id,
        "run_tag": run_tag or cfg.exp_id,
        "elapsed_sec": round(elapsed_sec, 2),
        "lambda_data": cfg.lambda_data,
        "lambda_pde": cfg.lambda_pde,
        "lambda_bc": cfg.lambda_bc,
        "lambda_ic": cfg.lambda_ic,
        "lambda_reg": cfg.lambda_reg,
        "reg_type": cfg.reg_type,
        "lambda_mode": cfg.lambda_mode,
        "D_hat_min": round(float(D_hat_pinn.min()), 4),
        "D_hat_max": round(float(D_hat_pinn.max()), 4),
        "D_hat_mean": round(float(D_hat_pinn.mean()), 4),
    }
    metrics_out.update(slide16_metrics)

    if D_true is not None and D_true.shape == D_hat_pinn.shape:
        pinn_m = compute_field_metrics(D_hat_pinn, D_true)
        metrics_out.update(pinn_m)

    if D_adjoint is not None and D_true is not None and D_adjoint.shape == D_true.shape:
        adj_m = compute_field_metrics(D_adjoint, D_true)
        metrics_out["adjoint_rel_l2_D"] = adj_m["rel_l2_D"]
        metrics_out["adjoint_mae_D"] = adj_m["mae_D"]

    metrics_out["final_losses"] = history[-1]

    print(
        f"  --> Result [{run_tag or cfg.exp_id}]: "
        f"rel_L2(D)={metrics_out.get('rel_l2_D', -1):.4f} | "
        f"MAE(D)={metrics_out.get('mae_D', -1):.4f} | "
        f"chi2_solver={metrics_out['chi2_red_solver']:.3f} | "
        f"chi2_u_net={metrics_out['chi2_red_u_net']:.3f} ({metrics_out['slide16_regime']}) | "
        f"D_range=[{metrics_out['D_hat_min']:.3f}, {metrics_out['D_hat_max']:.3f}] | "
        f"time={elapsed_sec:.1f}s",
        flush=True,
    )

    with open(exp_out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    plot_summary_figure(
        exp_out_dir / "summary.png",
        hist_df,
        x_grid,
        y_grid,
        D_true,
        D_adjoint,
        D_hat_pinn,
        bundle["sensor_xy"],
        cfg,
        metrics_out,
    )

    return metrics_out


def plot_summary_figure(
    save_path: Path,
    hist_df: pd.DataFrame,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    D_true: Optional[np.ndarray],
    D_adjoint: Optional[np.ndarray],
    D_pinn: np.ndarray,
    sensor_xy: Optional[np.ndarray],
    cfg: PINNConfig,
    metrics: Dict[str, Any],
):
    n_panels = 4 if D_true is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.2))
    extent = [float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max())]

    axes[0].semilogy(hist_df["epoch"], hist_df["loss_total"], label="L_total", lw=2)
    axes[0].semilogy(hist_df["epoch"], hist_df["loss_data"], label="L_data")
    axes[0].semilogy(hist_df["epoch"], hist_df["loss_pde"] + 1e-12, label="L_pde")
    axes[0].semilogy(hist_df["epoch"], hist_df["loss_reg"] + 1e-12, label="L_reg")
    axes[0].set_title(f"Loss History (chi2_u={metrics.get('chi2_red_u_net', 0):.2f})")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", ls="--", alpha=0.4)

    if D_true is not None and D_true.shape == D_pinn.shape:
        vmin = float(min(D_true.min(), D_pinn.min()))
        vmax = float(max(D_true.max(), D_pinn.max()))

        im1 = axes[1].imshow(D_true.T, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
        if sensor_xy is not None and len(sensor_xy) <= 200:
            axes[1].scatter(sensor_xy[:, 0], sensor_xy[:, 1], c="white", s=14, edgecolors="black", lw=0.6)
        axes[1].set_title("Ground Truth D(x, y) + Sensors")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(D_pinn.T, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[2].set_title(f"PINN D_hat (rel L2={metrics.get('rel_l2_D', 0):.3f})")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        err = np.abs(D_pinn - D_true)
        im3 = axes[3].imshow(err.T, origin="lower", extent=extent, cmap="magma")
        axes[3].set_title(f"|D_PINN - D_true| (MAE={metrics.get('mae_D', 0):.4f})")
        plt.colorbar(im3, ax=axes[3], fraction=0.046)
    else:
        im1 = axes[1].imshow(D_pinn.T, origin="lower", extent=extent, cmap="viridis")
        axes[1].set_title("PINN Recovered D_hat(x, y)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

    plt.suptitle(f"{metrics.get('run_tag', cfg.exp_id)}", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sweep_comparison(
    save_path: Path,
    df: pd.DataFrame,
    best_preset_by_l2: str,
    best_preset_by_chi2: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    presets = [p.replace("_", "\n", 1) for p in df["preset"]]
    x_pos = np.arange(len(presets))

    # Panel 1: Relative L2 Error on D(x, y)
    bars1 = axes[0].bar(x_pos, df["rel_l2_D"], color="#2563eb", alpha=0.85, label="PINN Rel-L2(D)")
    if "adjoint_rel_l2_D" in df.columns and df["adjoint_rel_l2_D"].notna().any():
        adj_val = float(df["adjoint_rel_l2_D"].iloc[0])
        axes[0].axhline(adj_val, color="#dc2626", ls="--", lw=2, label=f"Classical Adjoint ({adj_val:.3f})")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(df["preset"], rotation=35, ha="right", fontsize=8.5)
    axes[0].set_ylabel("Relative L2 Error ||D_hat - D_true||_2 / ||D_true||_2")
    axes[0].set_title(f"Diffusivity Recovery Error Across 10 Fixed Configs\n(Best L2: {best_preset_by_l2})")
    axes[0].legend()
    axes[0].grid(True, axis="y", ls="--", alpha=0.4)

    # Panel 2: Slide 16 Normalized Chi-Squared Discrepancy Metric (Target ~ 1.0)
    axes[1].bar(x_pos - 0.18, df["chi2_red_u_net"], width=0.35, color="#059669", label="chi2_red (u_theta vs sensors)")
    axes[1].bar(x_pos + 0.18, df["chi2_red_solver"], width=0.35, color="#d97706", label="chi2_red (RK4 solver(D_hat))")
    axes[1].axhline(1.0, color="#111827", ls="--", lw=2, label="Slide 16 Ideal Target (chi2_red = 1.0)")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(df["preset"], rotation=35, ha="right", fontsize=8.5)
    axes[1].set_ylabel("Reduced Chi-Squared Discrepancy (log scale)")
    axes[1].set_title(f"Slide 16 Discrepancy Metric Across 10 Fixed Configs\n(Closest to 1.0: {best_preset_by_chi2})")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(True, which="both", axis="y", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_suite_comparison(save_path: Path, df: pd.DataFrame):
    """Plots PINN vs Classical Adjoint across Tiers A-F and the Sensor Sparsity Sweep (n5..n100)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # Panel 1: All 10 Benchmark Cases Grouped Bar Chart (PINN vs Adjoint Rel-L2)
    short_labels = [e.split("__")[0] + "\n(" + e.split("__")[3] + "," + e.split("__")[5] + ")" for e in df["exp_id"]]
    x_pos = np.arange(len(df))
    axes[0].bar(x_pos - 0.18, df["rel_l2_D"], width=0.35, color="#2563eb", label="PINN (C04 Best Setting)")
    if "adjoint_rel_l2_D" in df.columns and df["adjoint_rel_l2_D"].notna().any():
        axes[0].bar(
            x_pos + 0.18,
            df["adjoint_rel_l2_D"],
            width=0.35,
            color="#dc2626",
            label="Classical Adjoint Baseline",
        )
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(short_labels, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("Relative L2 Error ||D_hat - D_true||_2 / ||D_true||_2")
    axes[0].set_title("Head-to-Head Across Tiers A-F: PINN vs Classical Adjoint")
    axes[0].legend()
    axes[0].grid(True, axis="y", ls="--", alpha=0.4)

    # Panel 2: Sensor Density Curve on C_inclusion (n5 -> n20 -> n50 -> n100)
    c_mask = df["exp_id"].str.startswith("C_inclusion__Fo0.1__uniform8__") & df["exp_id"].str.endswith("eta0.01")
    c_df = df[c_mask].copy()
    if len(c_df) >= 2:
        c_df["n_sensors"] = c_df["exp_id"].apply(lambda s: int(s.split("__")[3].replace("n", "")))
        c_df = c_df.sort_values("n_sensors")
        axes[1].plot(c_df["n_sensors"], c_df["rel_l2_D"], "o-", color="#2563eb", lw=2.2, label="PINN Rel-L2(D)")
        if "adjoint_rel_l2_D" in c_df.columns and c_df["adjoint_rel_l2_D"].notna().any():
            axes[1].plot(
                c_df["n_sensors"],
                c_df["adjoint_rel_l2_D"],
                "s--",
                color="#dc2626",
                lw=2.2,
                label="Classical Adjoint Rel-L2(D)",
            )
        axes[1].set_xlabel("Number of Spatial Sensors (N_s)")
        axes[1].set_ylabel("Relative L2 Error on D(x, y)")
        axes[1].set_title("Sensor Sparsity Crossover (C_inclusion, eta=0.01)")
        axes[1].legend()
        axes[1].grid(True, ls="--", alpha=0.4)
    else:
        axes[1].bar(x_pos, df["chi2_red_u_net"], color="#059669")
        axes[1].axhline(1.0, color="#111827", ls="--", lw=2)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(short_labels, rotation=35, ha="right", fontsize=8)
        axes[1].set_title("Slide 16 Discrepancy chi2_red Across Benchmark Suite")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 8. CLI Entrypoint (Defaults to Running the 10-Point Fixed Parameter Sweep)
# =====================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IE 643 Spatially-Varying Diffusion PINN Trainer (v2)")
    p.add_argument("--data-root", type=str, default="/kaggle/input/pinn-v2-dataset")
    p.add_argument("--output-dir", type=str, default="/kaggle/working")
    p.add_argument(
        "--exp-id",
        type=str,
        default="B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05",
        help="Experiment ID matching datasets_v2_rk4_zero/sensors/<exp_id>.csv",
    )
    p.add_argument("--lambda-data", type=float, default=40.0)
    p.add_argument("--lambda-pde", type=float, default=20.0)
    p.add_argument("--lambda-bc", type=float, default=10.0)
    p.add_argument("--lambda-ic", type=float, default=20.0)
    p.add_argument("--lambda-reg", type=float, default=2e-4)
    p.add_argument("--reg-type", type=str, default="tikhonov", choices=["tikhonov", "tv", "none"])
    p.add_argument(
        "--lambda-mode",
        type=str,
        default="fixed",
        choices=["fixed", "staged", "curriculum", "gradnorm"],
    )
    p.add_argument("--sweep-preset", type=str, default=None, choices=list(LAMBDA_SWEEP_GRID.keys()))
    p.add_argument(
        "--run-lambda-sweep",
        action="store_true",
        default=False,
        help="Run all 10 fixed configurations in LAMBDA_SWEEP_GRID and record errors",
    )
    p.add_argument(
        "--single-run",
        action="store_true",
        help="Run only a single configuration",
    )
    p.add_argument(
        "--run-benchmark-suite",
        action="store_true",
        default=True,
        help="Run PINN across the 10 core cases in results_adjoint/ and write summary.csv (enabled by default)",
    )
    p.add_argument("--warmstart-from-adjoint", action="store_true")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr-u", type=float, default=1.5e-3)
    p.add_argument("--lr-m", type=float, default=3.5e-3)
    p.add_argument("--n-collocation", type=int, default=768)
    p.add_argument("--config", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_cfg = PINNConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        exp_id=args.exp_id,
        lambda_data=args.lambda_data,
        lambda_pde=args.lambda_pde,
        lambda_bc=args.lambda_bc,
        lambda_ic=args.lambda_ic,
        lambda_reg=args.lambda_reg,
        reg_type=args.reg_type,
        lambda_mode=args.lambda_mode,
        warmstart_from_adjoint=args.warmstart_from_adjoint,
        epochs=args.epochs,
        lr_u=args.lr_u,
        lr_m=args.lr_m,
        n_collocation=args.n_collocation,
    )

    if args.sweep_preset:
        for k, v in LAMBDA_SWEEP_GRID[args.sweep_preset].items():
            setattr(base_cfg, k, v)

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            for k, v in json.load(f).items():
                if hasattr(base_cfg, k):
                    setattr(base_cfg, k, v)

    out_root = Path(base_cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.run_benchmark_suite and not args.run_lambda_sweep and not args.single_run:
        print(
            "=== Starting 10-Case Benchmark Suite (C04_balanced_lowreg_tik Winner vs. Classical Adjoint) ===",
            flush=True,
        )
        rows = []
        for idx, exp_name in enumerate(DEFAULT_ADJOINT_BENCHMARK_EXPS, start=1):
            print(f"\n--- Benchmark Case [{idx:02d}/10]: {exp_name} ---", flush=True)
            run_cfg = PINNConfig(**asdict(base_cfg))
            run_cfg.exp_id = exp_name
            if exp_name.startswith(("A_", "B_", "E_")):
                run_cfg.reg_type = "tikhonov"
                run_cfg.lambda_reg = 2e-4
            else:
                run_cfg.reg_type = "tv"
                run_cfg.lambda_reg = 3e-4
            res = train_single_experiment(run_cfg, run_tag=exp_name)
            rows.append({k: v for k, v in res.items() if k != "final_losses"})
        bench_df = pd.DataFrame(rows)
        bench_df.to_csv(out_root / "summary.csv", index=False)
        with open(out_root / "benchmark_suite_report.json", "w") as f:
            json.dump({"num_cases": len(rows), "results": rows}, f, indent=2)
        plot_benchmark_suite_comparison(out_root / "pinn_vs_adjoint_benchmark.png", bench_df)

        print("\n======================================================================", flush=True)
        print("10-CASE BENCHMARK SUITE COMPLETE (PINN vs. CLASSICAL ADJOINT)", flush=True)
        display_cols = [
            c
            for c in [
                "exp_id",
                "reg_type",
                "rel_l2_D",
                "adjoint_rel_l2_D",
                "mae_D",
                "adjoint_mae_D",
                "chi2_red_u_net",
                "chi2_red_solver",
                "slide16_regime",
            ]
            if c in bench_df.columns
        ]
        print(bench_df[display_cols].to_string(index=False), flush=True)
        print("======================================================================", flush=True)

    elif args.run_lambda_sweep and not args.single_run:
        print(
            f"=== Starting 10-Point Fixed Parameter Sweep on [{base_cfg.exp_id}] ===",
            flush=True,
        )
        rows = []
        for idx, (preset_name, preset_vals) in enumerate(LAMBDA_SWEEP_GRID.items(), start=1):
            print(f"\n--- Sweep Config [{idx:02d}/10]: {preset_name} ---", flush=True)
            run_cfg = PINNConfig(**asdict(base_cfg))
            for k, v in preset_vals.items():
                setattr(run_cfg, k, v)
            res = train_single_experiment(run_cfg, run_tag=f"{base_cfg.exp_id}__{preset_name}")
            res_row = {"preset": preset_name}
            res_row.update({k: v for k, v in res.items() if k != "final_losses"})
            rows.append(res_row)

        sweep_df = pd.DataFrame(rows)
        sweep_csv_path = out_root / "lambda_sweep_summary.csv"
        sweep_df.to_csv(sweep_csv_path, index=False)

        best_l2_row = sweep_df.loc[sweep_df["rel_l2_D"].idxmin()].to_dict()
        best_chi2_row = sweep_df.loc[sweep_df["chi2_solver_dist_from_1"].idxmin()].to_dict()
        best_unet_chi2_row = sweep_df.loc[sweep_df["chi2_u_net_dist_from_1"].idxmin()].to_dict()

        sweep_report = {
            "exp_id": base_cfg.exp_id,
            "num_configurations": len(rows),
            "best_by_rel_l2_D": best_l2_row,
            "best_by_slide16_solver_chi2": best_chi2_row,
            "best_by_slide16_unet_chi2": best_unet_chi2_row,
            "all_results": rows,
        }
        with open(out_root / "lambda_sweep_report.json", "w") as f:
            json.dump(sweep_report, f, indent=2)

        plot_sweep_comparison(
            out_root / "lambda_sweep_comparison.png",
            sweep_df,
            best_preset_by_l2=str(best_l2_row["preset"]),
            best_preset_by_chi2=str(best_chi2_row["preset"]),
        )

        print("\n======================================================================", flush=True)
        print("10-CONFIGURATION FIXED PARAMETER SWEEP COMPLETE", flush=True)
        print(
            sweep_df[
                [
                    "preset",
                    "lambda_data",
                    "lambda_pde",
                    "lambda_reg",
                    "reg_type",
                    "rel_l2_D",
                    "mae_D",
                    "chi2_red_u_net",
                    "chi2_red_solver",
                    "slide16_regime",
                ]
            ].to_string(index=False),
            flush=True,
        )
        print(
            f"\n[BEST BY REL-L2(D)]        : {best_l2_row['preset']} "
            f"(rel_l2_D={best_l2_row['rel_l2_D']:.4f}, mae_D={best_l2_row['mae_D']:.4f})",
            flush=True,
        )
        print(
            f"[BEST BY SLIDE 16 CHI2~1.0]: {best_chi2_row['preset']} "
            f"(chi2_red_solver={best_chi2_row['chi2_red_solver']:.4f}, rel_l2_D={best_chi2_row['rel_l2_D']:.4f})",
            flush=True,
        )
        print("======================================================================", flush=True)

    else:
        train_single_experiment(base_cfg)
