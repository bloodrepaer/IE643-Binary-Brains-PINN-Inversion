"""Extra fixed-alpha optimization at a retained dense NN fit; no true D.

This warm continuation is a separate convergence diagnostic, not an equal-
budget campaign replacement. Case/alpha are chosen from inputs and retained
training/discrepancy information, never coefficient error.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import minimize
import sys
# Keep the diagnostic's numerical solver identical to the broad campaign.
sys.path.insert(0,str(Path(__file__).resolve().parent/'campaign/code/48b49b1f17a0'))
from adjoint_nn import NeuralAdjointSolver
from adjoint_inversion import Dataset
from neural_diffusivity import NeuralDiffusivity


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--maxiter',type=int,default=750)
    ap.add_argument('--budget-seconds',type=int,default=1800)
    ap.add_argument('--out',default='campaign/diagnostics/adjoint_dense_convergence_v1')
    args=ap.parse_args()
    torch.set_num_threads(1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    # Reject duplicate scheduling and preserve existing outputs.
    lock=out/'launched.json'
    with lock.open('x') as f:json.dump(dict(pid=os.getpid(),started=time.time(),arguments=vars(args)),f,indent=2)
    key='A_smooth__Fo0.1__uniform8__n3969__nodes_nested__eta0.01'
    jobs=json.loads(Path('campaign/jobs.json').read_text())
    job=next(j for j in jobs if j['collection']=='datasets_v2_density_extended' and j['key']==key and j['method']=='adjoint_nn')
    parent=Path(job['output'])/key/'adjoint_nn__s42'
    baseline=json.loads((parent/'metrics.json').read_text())
    reference=parent/'model.pt';saved=torch.load(reference,map_location='cpu',weights_only=True)
    coefficient=NeuralDiffusivity.from_checkpoint(saved['coefficient'])
    root=Path(job['collection']);entry=next(e for e in json.loads((root/'manifest.json').read_text()) if e['key']==key)
    solver=NeuralAdjointSolver(Dataset(root,entry),coefficient=coefficient,n_steps=baseline['n_steps'],control=baseline['control'],reg=baseline['reg'])
    theta=solver.weights();alpha=baseline['alpha']
    initial_objective,initial_gradient=solver.objective(theta,alpha)
    protocol=dict(key=key,uses_true_D=False,initial_checkpoint=str(reference),initial_checkpoint_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  source='campaign/code/48b49b1f17a0',alpha=alpha,additional_maxiter=args.maxiter,budget_seconds=args.budget_seconds,
                  selection='Largest sensor density and lowest noise in first completed field; fixed retained alpha. Investigate capped optimization, not coefficient-truth selection.',
                  limitation='Warm continuation at selected alpha, not a repeated discrepancy path or equal-budget replacement. Original frozen result retained.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.monotonic();last=theta.copy();trace=[]
    def objective(x):
        if time.monotonic()-start>args.budget_seconds:raise TimeoutError('Diagnostic wall budget')
        return solver.objective(x,alpha)
    def callback(x):
        nonlocal last
        last=x.copy();trace.append(dict(iteration=len(trace)+1,replay_sigma=float(solver._last['rms']/solver.sigma_w),objective=float(solver._last['misfit'])))
        if len(trace)%50==0:print(trace[-1],flush=True)
    try:
        result=minimize(objective,theta,args=(),jac=True,method='L-BFGS-B',callback=callback,
                        options=dict(maxiter=args.maxiter,ftol=1e-8,gtol=1e-6,maxls=30))
        last=result.x;status='converged' if result.success else 'optimizer_stopped'
        optimizer=dict(iterations=int(result.nit),function_evaluations=int(result.nfev),success=bool(result.success),message=str(result.message))
    except TimeoutError:
        status='wall_budget_exhausted';optimizer=dict(iterations=len(trace),success=False,message='Diagnostic wall budget; last accepted iterate retained')
    final_objective,gradient=solver.objective(last,alpha)
    report=dict(status=status,key=key,uses_true_D=False,initial_checkpoint_sha256=protocol['initial_checkpoint_sha256'],alpha=alpha,
                baseline_replay_sigma=baseline['rms_replay_over_sigma'],continued_replay_sigma=float(solver._last['rms']/solver.sigma_w),
                initial_objective=float(initial_objective),final_objective=float(final_objective),initial_gradient_inf=float(np.max(np.abs(initial_gradient))),
                final_gradient_inf=float(np.max(np.abs(gradient))),optimizer=optimizer,additional_seconds=time.monotonic()-start,trace=trace)
    torch.save(dict(coefficient=coefficient.checkpoint(),key=key,method='adjoint_convergence_diagnostic'),out/'model.pt')
    np.savez_compressed(out/'D_hat.npz',D_hat=solver._last['D'].reshape(solver.ds.N,solver.ds.N))
    (out/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report|{'trace':f'{len(trace)} retained iterations'},indent=2),flush=True)


if __name__=='__main__':main()
