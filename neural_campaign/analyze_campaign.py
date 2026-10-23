"""Aggregate paired campaign results and evaluate hidden physics after training."""
import argparse
import hashlib
import json
from pathlib import Path
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sympy
from scipy.interpolate import RegularGridInterpolator
import campaign as C
import adjoint_inversion as ai
from domain import Domain2D
from integrators import BoundaryCondition, ForwardModel, integrate, EDGES


def true_physics_prediction(root, entry, D, n_steps=200):
    """Answer key is permitted here ONLY for post-fit evaluation."""
    root=Path(root);ds=ai.Dataset(root,entry)
    with np.load(root/entry["files"]["ground_truth"]) as gt:
        source_expression=str(gt["source_true"]) if "source_true" in gt else "0"
        bc=ds.bc
        if entry.get("bc_known",True) is False:
            paired=entry["problem"]
            for prefix in ("H2","H7","H10"):paired=paired.replace(prefix+"u-",prefix+"k-")
            specfile=root/"problem"/(paired+".json")
            if specfile.exists():
                spec=json.loads(specfile.read_text());t=sympy.Symbol("t")
                bc=BoundaryCondition({e:sympy.lambdify(t,sympy.sympify(ex),"numpy") for e,ex in spec["boundary_condition"]["edges"].items()})
            elif "bc_true_edges" in gt:
                ts=gt["bc_true_t"].copy();values=gt["bc_true_edges"].copy()
                bc=BoundaryCondition({e:(lambda t,i=i:np.interp(t,ts,values[:,i])) for i,e in enumerate(EDGES)})
            else:raise ValueError("True boundary specification unavailable")
    dom=Domain2D(Nx=ds.N,Ny=ds.N);fm=ForwardModel(dom,D,bc=bc)
    x,y=sympy.symbols("x y")
    src=sympy.lambdify((x,y),sympy.sympify(source_expression),"numpy")
    forcing=np.broadcast_to(np.asarray(src(dom.X,dom.Y),dtype=float),dom.shape)[1:-1,1:-1].ravel()
    times,interior,_=integrate(fm.L,lambda t:fm.b(t)+forcing,ds.u0[1:-1,1:-1].ravel(),
                              ds.T,n_steps,"cn",dt_target=ds.T/n_steps)
    U=np.stack([bc.field(dom.shape,t) for t in times]);U[:,1:-1,1:-1]=interior.reshape(n_steps+1,ds.N-2,ds.N-2)
    # CSVs use eight significant figures: the final reading can round just
    # above T. Match the inversion observation operator's endpoint clipping.
    if np.any(ds.t > ds.T+1e-6*max(ds.T,1e-12)) or np.any(ds.t < -1e-6*max(ds.T,1e-12)):
        raise ValueError("Observation times genuinely leave the problem horizon")
    prediction=RegularGridInterpolator((times,dom.x,dom.y),U)(np.c_[np.clip(ds.t,0,ds.T),ds.x,ds.y])
    return prediction,ds


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--true-physics",action="store_true",help="Add common true-physics replay after fitting, cached by field hash")
    args=parser.parse_args()
    C.summarize()
    df=pd.read_csv(C.HOME/"summary.csv")
    manifests={name:{e["key"]:e for e in json.loads((C.ROOT/name/"manifest.json").read_text())} for name in C.COLLECTIONS}
    for axis in ("Fo","protocol","layout","bc_known","source_known"):
        df[axis]=[manifests[r.collection][r.key].get(axis,True if axis.endswith("known") else None) for r in df.itertuples()]
    if args.true_physics:
        cache_dir=C.HOME/"evaluation";cache_dir.mkdir(exist_ok=True)
        true_replays=[]
        for _,row in df.iterrows():
            if row["physics_mode"]!="assumed_zero":true_replays.append(row["replay"]);continue
            metrics_path=C.ROOT/row["metrics"]
            Dfile=metrics_path.parent/"D_hat.npz"
            metrics=json.loads(metrics_path.read_text());n_steps=metrics.get("n_steps",200)
            fingerprint=hashlib.sha256(Dfile.read_bytes()+str(n_steps).encode()+row["collection"].encode()).hexdigest()
            cache=cache_dir/(fingerprint+".json")
            if cache.exists():value=json.loads(cache.read_text())["rms_true_over_sigma"]
            else:
                with np.load(Dfile) as z:D=np.asarray(z["D_hat"],float)
                pred,ds=true_physics_prediction(C.ROOT/row["collection"],manifests[row["collection"]][row["key"]],D,n_steps)
                rms=float(np.sqrt(np.mean((pred-ds.u)**2)))
                value=rms/ds.sigma if ds.sigma>0 else None
                C.atomic_json(cache,dict(rms_true_over_sigma=value,rms_true=rms,n_steps=n_steps,field_sha256=fingerprint))
            true_replays.append(value)
        df["replay_true_physics"]=true_replays
    df.to_csv(C.HOME/"analysis_results.csv",index=False)
    # Primary scientific question: neural coefficient representations in BOTH
    # inverse methods. Grid adjoints remain a secondary contextual baseline.
    neural_columns=['collection','key','physics_mode','tier','eta','n_sensors','n_times','rel_L2_D','replay','u_fit','wall_s','flag','metrics']
    neural_pairs=df[df.method=='adjoint_nn'][neural_columns].merge(
        df[df.method=='pinn'][neural_columns],on=['collection','key','physics_mode'],suffixes=('_adjoint_nn','_pinn'),validate='one_to_one')
    neural_pairs.to_csv(C.HOME/'primary_neural_comparison.csv',index=False)
    grouped=df.groupby(["collection","physics_mode","method"],dropna=False).agg(
        cases=("key","count"),median_rel_L2_D=("rel_L2_D","median"),median_replay=("replay","median"),
        median_wall_s=("wall_s","median"))
    grouped.to_csv(C.HOME/"grouped_results.csv")
    for axis in ("n_sensors","eta","n_times","Fo","protocol","layout","tier"):
        table=df.groupby(["collection","physics_mode","method",axis],dropna=False).agg(
            cases=("key","count"),median_rel_L2_D=("rel_L2_D","median"),median_replay=("replay","median"))
        table.to_csv(C.HOME/f"effects_{axis}.csv")
    density=df[df.collection=="datasets_v2_density_extended"].copy()
    complete_density=[]
    if not density.empty:
        density_figs=C.HOME/"figures";density_figs.mkdir(exist_ok=True)
        density.to_csv(C.HOME/"density_results.csv",index=False)
        path_diagnostics=[]
        maxiter=json.loads((C.HOME/'settings.json').read_text())['maxiter']
        for row in density.itertuples():
            metrics=json.loads((C.ROOT/row.metrics).read_text())
            path=metrics.get('alpha_path',[])
            selected=next((p for p in path if np.isclose(p['alpha'],metrics['alpha'])),{})
            path_diagnostics.append(dict(key=row.key,method=row.method,tier=row.tier,eta=row.eta,
                n_sensors=row.n_sensors,selected_alpha=metrics['alpha'],flag=row.flag,
                selected_iterations=selected.get('iters'),selected_hit_iteration_cap=selected.get('iters',-1)>=maxiter,
                capped_path_stages=sum(p.get('iters',0)>=maxiter for p in path),
                path_stages=len(path),selected_replay_sigma=row.replay,
                minimum_path_replay_sigma=min((p['rms_over_sigma'] for p in path),default=np.nan)))
        pd.DataFrame(path_diagnostics).to_csv(C.HOME/'density_path_diagnostics.csv',index=False)
        expected={5,20,50,100,200,400,800,1600,3200,3969}
        for (tier,eta,method),part in density.groupby(['tier','eta','method']):
            if set(part.n_sensors)!=expected:continue
            curve=part.sort_values('n_sensors')
            complete_density.append(dict(tier=tier,eta=eta,method=method,cases=len(curve),
                error_5=float(curve.iloc[0].rel_L2_D),error_3969=float(curve.iloc[-1].rel_L2_D),
                relative_error_reduction_percent=100*(1-float(curve.iloc[-1].rel_L2_D)/float(curve.iloc[0].rel_L2_D)),
                error_increases_between_adjacent_densities=int((np.diff(curve.rel_L2_D)>0).sum())))
        pd.DataFrame(complete_density).to_csv(C.HOME/'density_complete_curves.csv',index=False)
        for eta,part in density.groupby("eta"):
            fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
            for ax,(tier,subset) in zip(axes.flat,part.groupby("tier")):
                for method,curve in subset.groupby("method"):
                    curve=curve.sort_values("n_sensors")
                    ax.plot(curve.n_sensors,curve.rel_L2_D,"o-",label=method,ms=3)
                ax.set(xscale="log",xlabel="Spatial sensors",ylabel="Relative L2(D)",title=tier.replace("_"," "));ax.grid(alpha=.2)
            handles,labels=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc="outside lower center",ncol=3)
            fig.suptitle(f"Nested sensor-density study (completed fits): noise {100*eta:g}%, eight observation times")
            for ax in list(axes.flat)[part.tier.nunique():]:ax.set_visible(False)
            fig.savefig(density_figs/f"density_noise_{eta:g}.png",dpi=150)
            plt.close(fig)
    paired=df.pivot_table(index=["collection","key","physics_mode"],columns="method",values="rel_L2_D",aggfunc="first")
    paired.to_csv(C.HOME/"paired_errors.csv")
    join_keys=['collection','key','physics_mode']
    grid=df[df.method.isin(['grid','grid_historical'])][join_keys+['rel_L2_D','replay','flag']].rename(columns={c:'grid_'+c for c in ['rel_L2_D','replay','flag']})
    neural=df[df.method=='adjoint_nn'][join_keys+['tier','eta','n_sensors','rel_L2_D','replay','flag']].rename(columns={c:'neural_'+c for c in ['rel_L2_D','replay','flag']})
    adjoint_pairs=neural.merge(grid,on=join_keys,validate='one_to_one')
    adjoint_pairs['neural_lower_error']=adjoint_pairs.neural_rel_L2_D<adjoint_pairs.grid_rel_L2_D
    adjoint_pairs['grid_lower_error']=adjoint_pairs.grid_rel_L2_D<adjoint_pairs.neural_rel_L2_D
    adjoint_pairs.to_csv(C.HOME/'paired_adjoint_results.csv',index=False)
    adjoint_grouped=adjoint_pairs.groupby(['physics_mode','tier']).agg(pairs=('key','count'),
        neural_lower_error=('neural_lower_error','sum'),grid_lower_error=('grid_lower_error','sum'),
        median_neural_error=('neural_rel_L2_D','median'),median_grid_error=('grid_rel_L2_D','median'))
    adjoint_grouped.to_csv(C.HOME/'paired_adjoint_summary.csv')
    density_pairs=adjoint_pairs[adjoint_pairs.collection=='datasets_v2_density_extended'].copy()
    density_comparisons=[]
    if not density_pairs.empty:
        density_pairs['neural_to_grid_error_ratio']=density_pairs.neural_rel_L2_D/density_pairs.grid_rel_L2_D
        density_pairs.to_csv(C.HOME/'density_paired_results.csv',index=False)
        expected_counts={5,20,50,100,200,400,800,1600,3200,3969}
        for (tier,eta),curve in density_pairs.groupby(['tier','eta']):
            full=curve[curve.n_sensors==3969]
            density_comparisons.append(dict(tier=tier,eta=eta,paired_densities=len(curve),
                complete_curve=set(curve.n_sensors)==expected_counts,
                neural_lower_error=int(curve.neural_lower_error.sum()),grid_lower_error=int(curve.grid_lower_error.sum()),
                median_error_ratio=float(curve.neural_to_grid_error_ratio.median()),
                full_density_neural_error=float(full.neural_rel_L2_D.iloc[0]) if len(full) else None,
                full_density_grid_error=float(full.grid_rel_L2_D.iloc[0]) if len(full) else None))
        pd.DataFrame(density_comparisons).to_csv(C.HOME/'density_paired_summary.csv',index=False)
    defect_rows=[];defect_details=[]
    for row in df[df.tier.astype(str).str.contains('F_defects',regex=False)].itertuples():
        metrics=json.loads((C.ROOT/row.metrics).read_text())
        if 'defect_dice' not in metrics:continue
        item={name:getattr(row,name) for name in ['collection','key','method','physics_mode','eta','n_sensors','n_times','rel_L2_D']}
        item.update({name:metrics.get(name) for name in ['defect_dice','defects_detected','n_defects','background_rel_err']})
        defect_rows.append(item)
        for index,defect in enumerate(metrics.get('defects',[])):
            defect_details.append(dict(**item,defect_index=index,**defect))
    defects=pd.DataFrame(defect_rows)
    defects.to_csv(C.HOME/'defect_results.csv',index=False)
    pd.DataFrame(defect_details).to_csv(C.HOME/'defect_details.csv',index=False)
    if not defects.empty:
        density_defects=defects[defects.collection=='datasets_v2_density_extended']
        if not density_defects.empty:
            fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
            for (method,eta),curve in density_defects.groupby(['method','eta']):
                curve=curve.sort_values('n_sensors');label=f'{method}, {100*eta:g}% noise'
                axes[0].plot(curve.n_sensors,curve.defect_dice,'o-',ms=3,label=label)
                axes[1].plot(curve.n_sensors,curve.defects_detected/curve.n_defects,'o-',ms=3,label=label)
            for ax,label in zip(axes,['Defect-mask Dice','Fraction of defects detected']):
                ax.set(xscale='log',xlabel='Spatial sensors',ylabel=label,ylim=(-.03,1.03));ax.grid(alpha=.2)
            axes[0].legend(fontsize=8);fig.suptitle('Nested-density defect recovery (completed fits)')
            (C.HOME/'figures').mkdir(exist_ok=True);fig.savefig(C.HOME/'figures/density_defects.png',dpi=150);plt.close(fig)
    shared=paired.dropna(subset=["adjoint_nn","pinn"]) if {"adjoint_nn","pinn"}<=set(paired.columns) else paired.iloc[:0]
    figs=C.HOME/"figures";figs.mkdir(exist_ok=True)
    if not shared.empty:
        fig,ax=plt.subplots(figsize=(6,5))
        for mode,part in shared.groupby(level="physics_mode"):
            ax.scatter(part["adjoint_nn"],part["pinn"],label=mode,alpha=.75)
        lim=max(float(shared[["adjoint_nn","pinn"]].max().max()),.1)
        ax.plot([0,lim],[0,lim],"k--",lw=1);ax.set(xlabel="Neural adjoint: relative L2(D)",ylabel="PINN: relative L2(D)",title="Paired coefficient recovery (completed cases)");ax.legend();fig.tight_layout();fig.savefig(figs/"paired_coefficients.png",dpi=160);plt.close(fig)
        representatives=list(shared.index[:6])
        fig,axes=plt.subplots(len(representatives),3,figsize=(11,3.1*len(representatives)),squeeze=False,constrained_layout=True)
        for i,(collection,key,mode) in enumerate(representatives):
            entry=manifests[collection][key]
            with np.load(C.ROOT/collection/entry["files"]["ground_truth"]) as gt:truth=gt["D_inv"].copy()
            fields=[truth]
            for method in ("adjoint_nn","pinn"):
                row=df[(df.collection==collection)&(df.key==key)&(df.method==method)].iloc[0]
                with np.load((C.ROOT/row.metrics).parent/"D_hat.npz") as z:fields.append(z["D_hat"].copy())
            lo=min(a.min() for a in fields);hi=max(a.max() for a in fields)
            for j,(field,label) in enumerate(zip(fields,["True D","Neural adjoint","PINN"])):
                im=axes[i,j].imshow(field.T,origin="lower",extent=[0,1,0,1],vmin=lo,vmax=hi,cmap="plasma")
                axes[i,j].set_title(label if i==0 else "");axes[i,j].set_xlabel("x")
            axes[i,0].set_ylabel(entry.get("base_tier",entry["tier"]).replace("_"," ")+"\ny")
            fig.colorbar(im,ax=list(axes[i,:]),shrink=.8,label="D")
        fig.suptitle("Completed paired reconstructions (shared colour scale per row)")
        fig.savefig(figs/"paired_fields.png",dpi=150);plt.close(fig)
    # Temperature sensor fitting and coefficient replay are distinct checks.
    p=df[df.method=="pinn"].dropna(subset=["u_fit","replay"])
    if not p.empty:
        fig,ax=plt.subplots(figsize=(7,4))
        if len(p)<=12:
            x=np.arange(len(p));ax.bar(x-.18,p.u_fit,.36,label="Temperature-network fit");ax.bar(x+.18,p.replay,.36,label="Coefficient CN replay")
            ax.axhline(1,color="k",ls="--",lw=1);ax.set_xticks(x,[f"{r.tier}\nn={r.n_sensors}" for r in p.itertuples()],rotation=45,ha="right");ax.set(ylabel="RMS / noise sigma")
        else:
            for collection,part in p.groupby("collection"):
                ax.scatter(part.u_fit.clip(lower=1e-8),part.replay.clip(lower=1e-8),label=collection,alpha=.65,s=18)
            ax.set(xscale="log",yscale="log",xlabel="Temperature-network RMS / sigma",ylabel="CN replay RMS / sigma")
            ax.axhline(1,color="k",ls="--",lw=1);ax.axvline(1,color="k",ls=":",lw=1)
        ax.set(title="Sensor fit versus physics replay");ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figs/"fit_vs_replay.png",dpi=160);plt.close(fig)
    lines=["# IE643 experiment findings (partial)","",f"Updated {C.now()}.","",
           "Primary comparison: neural adjoint versus neural PINN. Grid adjoint is a secondary baseline. Both primary methods represent the unknown coefficient by a neural network; the neural adjoint obtains temperature through the discrete forward solver, while the PINN learns a temperature network. Versioned pilot protocols with different coefficient networks or objectives are reported separately until matched adjoint runs exist.","",
           f"Completed new fits: {sum(C.completed(j) for j in json.loads((C.HOME/'jobs.json').read_text()))} / {len(json.loads((C.HOME/'jobs.json').read_text()))}.",
           f"Paired completed neural comparisons: {len(shared)}.","",
           "## Primary neural comparison","","Full paired rows, coefficient error, common replay, sensor fit, convergence flags and qualified timings: primary_neural_comparison.csv. Only identical collection/key/physics rows are paired. Sparse pilot coverage and failed PINN consistency checks limit conclusions; unfinished PINNs are not method losses.","",
           "## Current evidence","","```text",grouped.to_string(),"```","",
           "The initial GPU PINNs fit the sensor temperatures near the noise level, yet their coefficients fail common CN replay (approximately 5.7-7.3 sigma on A/B/E). Sensor interpolation alone is not reliable coefficient recovery.","",
           "## Interpretation limits","",
           "Only completed pairs count toward a method comparison. Pending or failed runs are not losses. Broad single-seed coverage is exploratory; representative conclusions need additional seeds.","",
           "Grid and neural adjoints use discrepancy paths; the PINN uses fixed alpha 1. The two neural models share coefficient architecture and initial field, but differ in state representation, optimizer and physics enforcement. Historical and current hardware timings are not directly comparable.","",
           "Matched-physics and assumed-zero hidden-physics rows must be discussed separately. Told-model replay can look good for the wrong coefficient under misspecification. Optional true-physics replay reads hidden BC/source only after fitting.","",
           "Convergence caps are recorded. A discrepancy flag alone does not prove optimizer convergence or a physical/statistical noise floor."]
    lines += ["","## Added sensor-density experiment","",
              "180 additional configurations cover six canonical fields, 5/20/50/100/200/400/800/1600/3200/3969 sensors, and 1/5/10% noise. Fo=0.1 and eight uniform observation times are fixed. Sensor locations, clean values and noise draws are nested within each field/noise group; original problem and answer-key files are unchanged.",
              "This uses grid-node sensors and is analyzed separately from the historical off-grid layouts. Compare full density curves at fixed field/noise before pooling. Larger datasets can require more optimization, so budget caps and data-minibatch sampling must be reported. Low-noise layered fields may have a discretization-error floor even with the true coefficient."]
    if not shared.empty:
        lines += ["","## Paired descriptive results",""]
        for mode,part in shared.groupby(level="physics_mode"):
            wins=int((part.adjoint_nn<part.pinn).sum());losses=int((part.pinn<part.adjoint_nn).sum())
            lines.append(f"{mode}: {len(part)} pairs; lower coefficient error for neural adjoint in {wins}, PINN in {losses}; equal in {len(part)-wins-losses}. This is descriptive of the completed subset.")
    if not adjoint_pairs.empty:
        lines += ['', '## Paired neural and grid adjoints', '', '```text',adjoint_grouped.to_string(),'```','',
                  'Only identical observation cases and physics assumptions are paired. These are descriptive single-seed comparisons under the retained budgets and discrepancy paths, not a controlled claim that the representation alone causes each difference. Historical grid and current neural timings use different execution conditions. Full rows are in paired_adjoint_results.csv.']
    if density_comparisons:
        lines += ['', '## Paired nested-density comparisons', '', '```text',pd.DataFrame(density_comparisons).to_string(index=False),'```','',
                  'Each row fixes field and noise; only identical nested observations are paired. Complete curves require all ten prespecified densities. Partial curve counts are coverage, not complete-curve method rankings. An error ratio below one favors the neural adjoint. These single-seed, retained-budget comparisons do not establish converged optima.']
    inclusion_path=C.HOME/'diagnostics/dense_inclusion_localization/error_localization.csv'
    if inclusion_path.exists():
        inclusion=pd.read_csv(inclusion_path)
        lines += ['', '## Dense inclusion failure localization (post-fit evaluation)', '', '```text',
                  inclusion[['eta','method','rel_L2_D','replay_sigma','D_max','low_sensitivity_error_energy_fraction','boundary_error_energy_fraction','worst_one_percent_error_energy_fraction']].to_string(index=False),'```','',
                  'Both methods are evaluated against the answer key only after fitting, using the same fitted-constant forward sensitivity proxy and disjoint interior-high/interior-low/boundary masks. At 1% noise, the neural estimate concentrates about 73% of squared coefficient error in its worst 1% of grid nodes, and about 83% in the common low-sensitivity interior mask; boundary nodes contribute less than 0.001%. The maps show a sharp central overshoot. This is not a boundary-only artifact. The sensitivity proxy does not establish nonidentifiability, and iteration caps prevent separating optimization from representation limitations. Original per-fit high/low sensitivity metrics use different masks and should not be treated as a common-mask comparison. No checkpoint is refitted or selected here.']
    if not defects.empty:
        full_defects=defects[(defects.collection=='datasets_v2_density_extended')&(defects.n_sensors==3969)]
        if not full_defects.empty:
            lines += ['', '## Local defect recovery at full sensor density', '', '```text',
                      full_defects[['method','eta','rel_L2_D','defect_dice','defects_detected','n_defects','background_rel_err']].to_string(index=False),'```','',
                      'Global coefficient error can hide missed small defects. These metrics use the retained evaluation definitions: detected means recovered signed log-contrast at least 25% of the true contrast, and Dice uses the original anomaly threshold. Neither threshold is tuned to these results. Full density curves and per-defect contrast/localization scores are saved in defect_results.csv, defect_details.csv and figures/density_defects.png. These are synthetic post-fit evaluations, not real-data detection guarantees.']
    if complete_density:
        lines += ['', '## Completed density curves', '', '```text',pd.DataFrame(complete_density).to_string(index=False),'```','',
                  'These are complete ten-density curves for the listed method/field/noise groups only. Method comparisons require matched completed curves. Sensor noise and locations are nested; there is one observation realization per field/noise group.',
                  'Nonmonotonic coefficient error can reflect regularization selection as well as optimization and noise. The discrepancy rule stops at the first alpha reaching RMS/sigma <=1. If none reaches it before the path plateau rule, it selects the strongest alpha whose RMS is within 2% of the best path value. A small crossing of one can therefore change the selected regularization substantially. The floor_above_sigma flag does not establish a physical error floor. Inspect density_path_diagnostics.csv for selected alpha, path minimum and iteration caps; grid outputs do not record optimizer success, so an iteration count alone cannot prove convergence.']
    truth_diagnostic=C.HOME/'diagnostics/density_true_D_replay.csv'
    if truth_diagnostic.exists():
        table=pd.read_csv(truth_diagnostic)
        lines += ['', '## Post-fit full-density forward consistency', '',
                  'Evaluation only: replay the true supplied coefficient through the same 200-step CN solver. This does not select inverse models or training settings. The clean-field discrepancy isolates forward discretization and observation mismatch; noisy replay also includes the actual noise realization.',
                  '', '```text',table[['tier','noise','true_D_CN_replay_sigma','true_D_CN_vs_clean_sigma','selected_grid_replay_sigma','selected_grid_coefficient_error']].to_string(index=False),'```', '',
                  'Coefficient error and temperature discrepancy quantify different things. A noise-level temperature replay can coexist with substantial coefficient error in weakly observed regions or restricted coefficient representations. A path flag alone cannot attribute the coefficient error to forward discretization.']
    refinement=C.HOME/'diagnostics/density_time_refinement.csv'
    if refinement.exists():
        lines += ['', '### Forward time-step refinement (evaluation only)', '', '```text',pd.read_csv(refinement).to_string(index=False),'```','',
                  'The layered-field mismatch barely changes from 200 to 800 CN steps. Temporal refinement alone does not resolve it; spatial/operator and data-generation differences remain to be separated. This is a post-fit diagnostic, not truth-based selection of inverse training settings.']
    spatial=C.HOME/'diagnostics/density_spatial_resolution.csv'
    if spatial.exists():
        lines += ['', '### Forward spatial-grid check (evaluation only)', '', '```text',pd.read_csv(spatial).to_string(index=False),'```','',
                  'For the layered field at 1% noise, switching from the inversion grid (65x65) to the original data-generation grid (129x129) reduces true-D clean-data discrepancy from about 1.849 to 0.007 sensor sigma at 200 CN steps. At 800 steps it falls from 1.845 to 0.000437 sigma. This isolates the coarser spatial discretization as the dominant mismatch in these forward checks; simply increasing CN time steps on the coarse grid is insufficient. Original observations, inverse estimates and training settings are preserved.']
    transferred=C.HOME/'diagnostics/learned_layered_grid_transfer.csv'
    if transferred.exists():
        transfer=pd.read_csv(transferred)
        lines += ['', '### Retained layered estimates on a finer replay grid', '', '```text',
                  transfer[['eta','method','grid_N','deployment','replay_sigma']].to_string(index=False),'```','',
                  'These are post-fit replays, not refits, and use no coefficient truth or answer-key snapshots. The supplied analytic IC is evaluated on129-grid after checking agreement with the original65-grid IC. Common transfer bilinearly interpolates logD for both methods; native neural evaluation is a separate deployment. All transferred fields preserve original65-grid node values. At1% noise and3969 sensors, fine common-transfer replay is1.171 neural versus1.888 grid sigma; native neural replay is1.106. This does not replace the common65-grid campaign comparison or establish convergence.']
    state_transfer=C.HOME/'diagnostics/pinn_state_grid_transfer.csv'
    if state_transfer.exists():
        lines += ['', '### PINN state consistency on a finer replay grid', '', '```text',pd.read_csv(state_transfer).to_string(index=False),'```','',
                  'This post-fit check uses only the supplied analytic IC and each retained coefficient function, without coefficient truth or refitting. Refining65 to129 nodes changes v6 state-versus-CN mismatch from2.343 to2.346 noise sigma, v7 from.723 to.724 and v8 from.774 to.776. The observed mismatches are therefore not removed by this spatial refinement. The analytic IC check introduces only the recorded small CSV-rounding difference relative to original replay.']
    constraint_file=C.HOME/"diagnostics/pinn_constraints.csv"
    if constraint_file.exists():
        constraints=pd.read_csv(constraint_file)
        lines += ["","## Known-initial-condition diagnosis", "",
                  "The completed plain-state PINNs were checked against the supplied initial field, without reading the true coefficient. These v0 models substantially underfit the initial condition despite matching later sensor measurements.",
                  "```text",constraints[["key","ic_relative_L2","ic_true_peak","ic_predicted_peak"]].to_string(index=False),"```","",
                  "Further plain-state batch preparation is held by campaign/pinn_review_gate.json. Separate versioned pilots retain the same coefficient network and observations and are evaluated using known constraints and replay, not coefficient truth."]
        gate=C.HOME/'pinn_review_gate.json'
        if gate.exists():lines += ['',json.loads(gate.read_text()).get('latest_action','')]
    for validation in sorted((C.HOME/'pilots').glob('*/validation.json')):
        pilot=json.loads(validation.read_text())
        pilot_view=pd.DataFrame(pilot['results'])
        pilot_columns=[c for c in ['variant','key','ic_relative_L2','replay_sigma','bc_rms_over_sensor_sigma','replay_improvement_over_constant_percent','consistency_screen_passed'] if c in pilot_view]
        lines += ['', '## Separate PINN pilot: '+validation.parent.name, '',
                  'These pilot fits are retained separately and are not counted as frozen v0 campaign completions. Validation uses supplied physics and observations, without true coefficient selection.',
                  '', '```text',pilot_view[pilot_columns].to_string(index=False),'```','',pilot['decision']]
        metadata=validation.parent/'pilot.json'
        if metadata.exists():
            protocol=json.loads(metadata.read_text())
            if protocol.get('cumulative_lbfgs_budget'):
                lines += ['',f"This continuation starts from a prior pilot checkpoint: cumulative Adam budget {protocol['cumulative_adam_steps']}, cumulative L-BFGS budget {protocol['cumulative_lbfgs_budget']}. It is not an equal-budget fresh fit."]
        coefficient_scores=validation.parent/'coefficient_scores.csv'
        if coefficient_scores.exists():
            lines += ['', 'Post-fit coefficient scores (answer-key evaluation only; all retained cases/arms scored, no selection):', '',
                      '```text',pd.read_csv(coefficient_scores).to_string(index=False),'```']
    frozen=C.HOME/'pilots/ic_bc_v3_gpu/frozen_state_diagnostic/diagnostic.json'
    if frozen.exists():
        diagnostic=json.loads(frozen.read_text())
        lines += ['', '## PINN optimization diagnostic', '',
                  'The v3 temperature network was frozen and only its coefficient network optimized against PDE residual plus the same spatial regularizer, without reading true D. This is a separate block diagnostic, not a completed campaign fit.',
                  f"Common replay changed from {diagnostic['before']['replay_sigma']:.3f} to {diagnostic['after']['replay_sigma']:.3f} sigma. Independent PDE RMS changed from {diagnostic['before']['validation_pde_rms']:.4f} to {diagnostic['after']['validation_pde_rms']:.4f}; neither resolves the physical-consistency failure."]
    concentration=C.HOME/'pilots/ic_bc_v3_gpu/residual_concentration.json'
    if concentration.exists():
        diagnostic=json.loads(concentration.read_text());top1=next(v for v in diagnostic['concentration'] if v['top_fraction']==.01)
        lines += ['', f"An independent {diagnostic['n_points']:,}-point probe using the original mixed sampling distribution found the largest 1% of residuals carry {100*top1['energy_fraction']:.1f}% of residual energy. This motivates investigating collocation coverage; it does not prove adaptive sampling will fix recovery. No true D is used in this diagnosis."]
    incident=C.HOME/'diagnostics/disk_incident_recovery.json'
    if incident.exists():
        recovery=json.loads(incident.read_text())
        lines += ['', '## Infrastructure failure and recovery', '',
                  f"The C: drive filled during an edit and grid visualization. The root source was restored from its retained validated copy; frozen campaign source was unaffected. Integrity checks passed for {recovery['completed_core_artifacts_verified']} completed core artifacts (JSON, array finiteness and NPZ/model CRCs). One missing grid PNG was rebuilt from the saved estimate without refitting or changing coefficient/metrics hashes. Its original failure log and state remain under failed_attempts.",
                  'Only SHA256-verified duplicate inputs and a reproducible input-only upload container were reclaimed; canonical data, all scientific source versions and trained results were retained. Input folders also received lossless NTFS compression. The grid coordinator was restarted after confirming no surviving grid process; new coordinators stop dispatch below 128 MiB free disk space. This is an infrastructure incident, not a scientific method failure. See diagnostics/disk_incident_recovery.json and the input-deduplication manifests.']
    seed_homes=[C.HOME/'replicates/adjoint_seed_confirmation',C.HOME/'replicates/adjoint_density_seed_confirmation']
    continuation_path=C.HOME/'diagnostics/adjoint_dense_convergence_v1/diagnostic.json'
    if continuation_path.exists():
        continuation=json.loads(continuation_path.read_text())
        lines += ['', '## Dense adjoint convergence continuation (separate budget)', '',
                  f"A_smooth at 3969 sensors and 1% noise was continued from its retained neural-adjoint checkpoint at unchanged alpha {continuation['alpha']:.6g}. Replay changed from {continuation['baseline_replay_sigma']:.4f} to {continuation['continued_replay_sigma']:.4f} sensor sigma over {continuation['optimizer']['iterations']} additional iterations; optimizer status: {continuation['optimizer']['message']}.",
                  'This diagnostic was preselected from sensor density, noise and iteration caps, without coefficient truth. It warm-starts the selected-alpha checkpoint rather than repeating the discrepancy path. Additional compute is separate from the frozen campaign; the original fit is preserved and no coefficient-error result is used for selection.']
    forward_rows=[]
    for forward_home in [C.HOME/'pilots/fixed_d_forward_v7_gpu',C.HOME/'pilots/fixed_d_refresh_v8_gpu']:
        forward_validation=forward_home/'artifacts/independent_validation.json'
        if not forward_validation.exists():continue
        validated=json.loads(forward_validation.read_text())
        measured=validated['measurements']
        forward_metrics=json.loads((forward_home/'artifacts/metrics.json').read_text())
        forward_rows.append(dict(pilot=forward_home.name,adam_steps=forward_metrics['adam_steps'],lbfgs_steps=forward_metrics['lbfgs_steps'],lbfgs_rounds=forward_metrics.get('lbfgs_rounds',1),
                                 **measured,training_pde_rms=forward_metrics['training_pde_rms'],heldout_pde_rms=forward_metrics['heldout_pde_rms'],
                                 training_seconds=forward_metrics['training_seconds'],device_name=forward_metrics['device_name'],
                                 lbfgs_function_evaluations=sum(r['function_evaluations'] for r in forward_metrics.get('optimizer_rounds',[forward_metrics['optimizer_diagnostics']]))))
        lines += ['', '## Fixed-coefficient forward-state diagnostic: '+forward_home.name, '',
                  'The coefficient is the retained A neural-adjoint fit, held fixed bitwise. Only the fresh temperature network is optimized against the PDE; sensors and CN are evaluation only, and no answer key is supplied.',
                  f"State-versus-CN sensor RMS is {measured['state_vs_CN_sensor_sigma']:.4f} noise sigma; state-versus-observations {measured['sensor_fit_sigma']:.4f}, fixed-coefficient CN-versus-observations {measured['fixed_coefficient_CN_sensor_sigma']:.4f}. Checkpoint metrics independently reproduced: {validated['checkpoint_metrics_reproduced']}; coefficient unchanged: {validated['coefficient_bitwise_unchanged']}.",
                  f"This forward-only diagnostic uses {forward_metrics['adam_steps']} Adam and {forward_metrics['lbfgs_steps']} requested L-BFGS steps in {forward_metrics.get('lbfgs_rounds',1)} round(s) on {forward_metrics['device_name']}. The initial coefficient is spatially variable, unlike fresh v6; it is not an equal-initial-function inverse comparison. Refreshed rounds change both collocation and optimizer history. Success or failure does not by itself reopen the inverse gate or prove state nonexpressivity."]
    if forward_rows:
        pd.DataFrame(forward_rows).to_csv(C.HOME/'diagnostics/forward_state_comparison.csv',index=False)
        if len(forward_rows)==2:
            original,refreshed=forward_rows
            lines += ['', '### Controlled forward collocation-refresh comparison', '',
                      f"Refreshing collocation and optimizer history reduced mixed held-out PDE RMS by {100*(1-refreshed['heldout_pde_rms']/original['heldout_pde_rms']):.1f}%, while state-versus-CN sensor RMS changed from {original['state_vs_CN_sensor_sigma']:.4f} to {refreshed['state_vs_CN_sensor_sigma']:.4f} sigma. Neither temperature fit is an inverse coefficient recovery result. The two retained GPU Adam histories are identical; each used750 L-BFGS iterations and819 function evaluations. Times were {original['training_seconds']:.1f} and {refreshed['training_seconds']:.1f} seconds on {original['device_name']}. Lower held-out PDE RMS alone does not guarantee lower state error at sensor locations; the inverse gate remains held."]
    gradient_file=C.HOME/'diagnostics/pinn_loss_gradient_components.csv'
    if gradient_file.exists():
        lines += ['', '## Retained-checkpoint loss and gradient reconstruction', '', '```text',pd.read_csv(gradient_file).to_string(index=False),'```','',
                  'The v5/v6 fixed-batch objectives reproduce saved losses and gradients without refitting. V6 data/PDE full-gradient cosine is about -0.991, with comparable component norms. Such opposition is normal at a penalized compromise and depends on parameterization. It does not establish a correct penalty or prove the cause of inverse failure. Component and geometry files are retained per pilot; no true coefficient is used.']
    gap_path=C.HOME/'diagnostics/pilot_collocation_gap.csv'
    if gap_path.exists():
        gap=pd.read_csv(gap_path)
        lines += ['', '## Independent collocation and optimization checks', '', '```text',gap.to_string(index=False),'```','',
                  'The listed checkpoints use the same independent 16384-point half-log/half-uniform-time probe; retained training-cloud statistics appear where available. Adaptive v5 training changes the sampling measure, whereas v12 uses the same mixed measure as the probe. V6 held-out RMS is about1.51 times training RMS and v12 about12.71 times. Both retained optimization limits and generalization must be considered; these checks do not establish a sole cause of replay failure. No coefficient truth is read and no checkpoint is refitted or selected by this diagnostic.']
    quadrature_path=C.HOME/'pilots/calibrated_v12_gpu/quadrature_audit/results.json'
    if quadrature_path.exists():
        quadrature=json.loads(quadrature_path.read_text())
        lines += ['', '## Fixed-checkpoint residual integration audit', '', '```text',
                  pd.DataFrame(quadrature['results']).to_string(index=False),'```','',
                  'Four predeclared4096-point replicates per sampler estimate the same mixed-time residual integral on the unchanged v12 checkpoint. The65536-point random reference is also noisy, not an exact integral. Lower replicate variability alone does not establish greater accuracy or a better training method; no model is retrained, selected or scored against coefficient truth. All replicates and reference time-bin energies are retained in the pilot quadrature_audit folder.']
    dense_resolution=C.HOME/'pilots/dense_calibrated_v14_gpu/resolution_audit/results.json'
    if dense_resolution.exists():
        resolution=json.loads(dense_resolution.read_text())
        lines += ['', '## Dense PINN fixed-checkpoint quadrature resolution', '', '```text',
                  pd.DataFrame(resolution['results']).to_string(index=False),'```','',
                  'The retained3969-sensor PINN is evaluated on four predeclared independent random replicates, each with nested4096/16384/65536 prefixes and exactly half uniform/half log-time points. Replicate coefficients of variation are23.23%,9.30%,3.31%; all replicates are retained. These descriptive four-replicate estimates have no exact-integral reference. Reduced integration variability is not a verified training or inverse-accuracy improvement, nor an optimal point-count claim. The checkpoint is unchanged; no coefficient truth, optimizer or checkpoint selection is used.']
    for seed_home in seed_homes:
        if not (seed_home/'protocol.json').exists():continue
        protocol=json.loads((seed_home/'protocol.json').read_text())
        originals={j['id']:j for j in json.loads((C.HOME/'jobs.json').read_text())}
        seed_rows=[]
        for job in [originals[i] for i in protocol['reuse_seed42']]+protocol['jobs']:
            if not C.completed(job):continue
            metrics=json.loads(C.metric_path(job).read_text())
            seed_rows.append(dict(collection=job['collection'],key=job['key'],seed=job['seed'],rel_L2_D=metrics['rel_L2_D'],
                replay_sigma=metrics['rms_replay_over_sigma'],flag=metrics['flag']))
        seed_frame=pd.DataFrame(seed_rows)
        seed_frame.to_csv(seed_home/'partial_summary.csv',index=False)
        if not seed_frame.empty:
            seed_groups=seed_frame.groupby('key').agg(completed_seeds=('seed','count'),mean_coefficient_error=('rel_L2_D','mean'),
                std_coefficient_error=('rel_L2_D','std'),min_coefficient_error=('rel_L2_D','min'),max_coefficient_error=('rel_L2_D','max'),
                mean_replay_sigma=('replay_sigma','mean'))
            seed_groups.to_csv(seed_home/'grouped_summary.csv')
            seed_comparison=seed_frame.merge(adjoint_pairs[['collection','key','grid_rel_L2_D','grid_replay']],on=['collection','key'],how='left',validate='many_to_one')
            seed_comparison['neural_lower_error']=seed_comparison.rel_L2_D < seed_comparison.grid_rel_L2_D
            seed_comparison.to_csv(seed_home/'paired_grid_comparison.csv',index=False)
            expected_seeds=len(protocol['reuse_seed42'])+len(protocol['jobs'])
            phase='complete' if len(seed_frame)==expected_seeds else 'partial'
            lines += ['', f'## Fixed-observation adjoint seed confirmation: {seed_home.name} ({phase})', '', '```text',seed_groups.to_string(),'```','',
                      'These initialization seeds share exactly the same observations and frozen neural-adjoint settings. They quantify optimization/representation variability conditional on one observation realization, not uncertainty across new sensor locations or noise draws. A single completed seed has no estimated standard deviation. These additional fits are retained separately from the 1510-fit broad plan.']
            for key,part in seed_comparison.groupby('key'):
                if part.grid_rel_L2_D.notna().all():
                    lines.append(f"{key}: neural coefficient error {part.rel_L2_D.min():.4f}--{part.rel_L2_D.max():.4f}, fixed grid error {part.grid_rel_L2_D.iloc[0]:.4f}; neural lower in {int(part.neural_lower_error.sum())}/{len(part)} seeds. Replay range {part.replay_sigma.min():.3f}--{part.replay_sigma.max():.3f} sigma. All seeds are retained; none is selected as the preferred fit. Timing differences also reflect worker contention, so these seed timings are not a speed comparison.")
    for pilot_name,pair_title in [('zhang_v11_gpu','paper-transfer'),('calibrated_v12_gpu','calibrated hard-physics'),('sampling_v13_gpu','controlled collocation sampling'),('dense_calibrated_v14_gpu','higher-density calibrated'),('dense_resolution_v15_gpu','dense collocation-count control'),('optimizer_v16_gpu','dense method-by-optimizer control'),('temporal_v18_gpu','five dense temporal-consistency cases')]:
        paper_pair=C.HOME/f'pilots/{pilot_name}/primary_matched_neural_comparison.csv'
        if not paper_pair.exists() and pilot_name=='temporal_v18_gpu':
            paper_pair=paper_pair.with_name('partial_complete_case_comparison.csv')
        if not paper_pair.exists():continue
        lines += ['', f'## Matched {pair_title} neural adjoint versus neural PINN', '',
                  '```text',pd.read_csv(paper_pair).to_string(index=False),'```','',
                  'Separate predeclared research protocol. Each comparison shares coefficient network, initial weights, mean data objective and regularization as declared in its versioned protocol; v16 crosses both methods with two optimizer implementations under common iteration limits. See the versioned protocol for cases, coefficient bounds and state construction. Different state enforcement and finite-dimensional objectives remain intrinsic. Check coverage, fit status, optimizer caps, independent constraints and replay before interpreting accuracy. Record actual hardware, resource use and CPU contention before comparing timings. These research fits are outside the frozen broad count.']
        integrity=paper_pair.with_name('pair_integrity.json')
        if integrity.exists():
            coverage=json.loads(integrity.read_text()).get('coverage_complete')
            if coverage is not None:lines.append(f'All predeclared paired cases covered: {coverage}. Missing or unpaired cases are not method losses.')
    (C.HOME/"FINDINGS.md").write_text("\n".join(lines),encoding="utf-8")
    print("Saved analysis tables, figures and campaign/FINDINGS.md")


if __name__=="__main__":main()
