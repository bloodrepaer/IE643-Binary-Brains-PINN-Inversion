"""Read-only saved comparison and checkpoint inference helpers for the interface."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import io,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from neural_diffusivity import NeuralDiffusivity
from compare_neural import TemperatureNetwork,temperature_value

ROOT=Path(__file__).resolve().parent


def comparison_catalog():
    result=[];p=ROOT/'campaign/primary_neural_comparison.csv'
    if p.exists():
        for row in pd.read_csv(p).to_dict('records'):
            result.append(dict(label=f"Frozen · {row['key']}",key=row['key'],collection=row['collection'],mode=row['physics_mode'],protocol='Frozen broad campaign; shared coefficient, differing optimization/alpha selection.',paths={m:ROOT/row['metrics_'+m] for m in ['adjoint_nn','pinn']}))
    # Retained density pilot: match each PINN to exactly the same broad NN case.
    base=ROOT/'campaign/analysis_results.csv'
    if base.exists():
        rows=pd.read_csv(base)
        for p in (ROOT/'campaign/pilots/density_v10_gpu/artifacts').rglob('metrics.json'):
            m=json.loads(p.read_text());matches=rows[(rows.collection=='datasets_v2_density_extended')&(rows.key==m['key'])&(rows.method=='adjoint_nn')]
            if len(matches)==1:result.append(dict(label='Density v10 · '+m['key'],key=m['key'],collection='datasets_v2_density_extended',mode='matched',protocol='v10 hard-IC/BC PINN versus retained broad neural adjoint; optimizer/regularization selection differ.',paths=dict(pinn=p,adjoint_nn=ROOT/matches.iloc[0].metrics)))
    for name in ['zhang_v11_gpu','calibrated_v12_gpu']:
        home=ROOT/'campaign/pilots'/name;pair=home/'primary_matched_neural_comparison.csv'
        if not pair.exists():continue  # Never present queued/incomplete work as a comparison.
        protocol=json.loads((home/'pilot.json').read_text());paths={}
        for p in (home/'artifacts').rglob('metrics.json'):
            m=json.loads(p.read_text());method=m.get('method','pinn')
            if method in ['pinn','adjoint_nn']:paths[method]=p
        if (home/'matched_adjoint/metrics.json').exists():paths['adjoint_nn']=home/'matched_adjoint/metrics.json'
        if len(paths)==2:result.append(dict(label=name+' · '+protocol['cases'][0],key=protocol['cases'][0],collection=protocol['data_collection'],mode='matched',protocol='Separate matched research pair. Read fit status, constraints, budgets and hardware before interpreting accuracy.',paths=paths))
    return result


def load_comparison(item):
    fields={};metrics={};records=[]
    manifest={e['key']:e for e in json.loads((ROOT/item['collection']/'manifest.json').read_text())};entry=manifest[item['key']]
    for method,path in item['paths'].items():
        m=json.loads(path.read_text());metrics[method]=m
        with np.load(path.with_name('D_hat.npz')) as z:fields[method]=np.asarray(z['D_hat'],float)
        diagnostics=m.get('optimizer_diagnostics',[]);path_stages=m.get('alpha_path',[])
        capped=(any(r.get('iterations',0)>=r.get('requested_iterations',10**9) for r in diagnostics) or any(r.get('iters',0)>=150 for r in path_stages)) if diagnostics or path_stages else None
        records.append(dict(method=method,coefficient_error_percent=None,replay_sigma=m.get('rms_replay_over_sigma'),sensor_fit_sigma=m.get('rms_unet_over_sigma',m.get('rms_replay_over_sigma') if method=='adjoint_nn' else None),status=m.get('status',m.get('flag','not recorded')),wall_seconds=m.get('wall_s',m.get('elapsed_sec')),optimizer_capped=capped))
    truth=None
    answer=entry.get('files',{}).get('ground_truth')
    if answer and (ROOT/item['collection']/answer).exists():
        with np.load(ROOT/item['collection']/answer) as z:truth=z['D_inv'].copy()
        for r in records:
            D=fields[r['method']]
            if D.shape==truth.shape:r['coefficient_error_percent']=100*float(np.linalg.norm(D-truth)/np.linalg.norm(truth))
    return fields,pd.DataFrame(records),truth,metrics


def infer_checkpoint(model_bytes,points):
    """No training, file writes, or coefficient answer key access."""
    if not {'x','y'}<=set(points.columns):raise ValueError('Points require numeric x and y columns.')
    if len(points)==0 or len(points)>100000:raise ValueError('Use between 1 and 100000 points.')
    xy=points[['x','y']].to_numpy(dtype=float)
    if not np.isfinite(xy).all() or np.any(xy<0) or np.any(xy>1):raise ValueError('This campaign uses x,y within [0,1].')
    saved=torch.load(io.BytesIO(model_bytes),map_location='cpu',weights_only=True)
    c=saved.get('coefficient',saved)
    if c.get('kind')=='zhang_exp':
        from pinn_zhang_transfer import PaperCoefficient
        coefficient=PaperCoefficient.from_checkpoint(c)
    else:coefficient=NeuralDiffusivity.from_checkpoint(c)
    temperature=None;times=None
    if 'temperature_state_dict' in saved and 't' in points:
        T=float(saved['T']);times=points.t.to_numpy(dtype=float)
        if not np.isfinite(times).all() or np.any(times<0) or np.any(times>T*(1+1e-7)):raise ValueError(f'Temperature inputs require physical time t within [0,{T:.8g}].')
        times=np.clip(times,0,T)/T;temperature=TemperatureNetwork(**saved['temperature_config']);temperature.load_state_dict(saved['temperature_state_dict'])
    D=[];u=[]
    with torch.no_grad():
        for start in range(0,len(points),512):
            q=torch.tensor(xy[start:start+512],dtype=torch.float64);D.extend(coefficient.diffusivity(q).numpy().ravel())
            if temperature is not None:
                xyt=torch.cat([q,torch.tensor(times[start:start+512,None],dtype=torch.float64)],1);u.extend(temperature_value(temperature,coefficient,xyt).numpy().ravel())
    result=points.copy();result['D_pred']=D
    if not np.isfinite(result.D_pred).all() or np.any(result.D_pred<=0):raise ValueError('Checkpoint produced invalid diffusivity.')
    if temperature is not None:result['u_pred']=u
    metrics={}
    if 'D_reference' in result:
        ref=result.D_reference.to_numpy(dtype=float)
        if not np.isfinite(ref).all() or np.linalg.norm(ref)==0:raise ValueError('Reference D must be finite and have nonzero norm.')
        metrics['relative_L2_D_on_input_points']=float(np.linalg.norm(result.D_pred-ref)/np.linalg.norm(ref))
    if 'u_pred' in result and 'u' in result:
        observed=result.u.to_numpy(dtype=float)
        if not np.isfinite(observed).all():raise ValueError('Observed u must be finite.')
        metrics['temperature_network_RMSE']=float(np.sqrt(np.mean((result.u_pred-observed)**2)))
    return result,metrics,temperature is not None
