import sys; sys.path.insert(0,'/home/claude/work')
from style import *
import numpy as np, pandas as pd, glob, os, json
from scipy.stats import spearmanr
O='/home/claude/work/fig/'; V2='/home/claude/v2/'
r=pd.read_csv('/home/claude/work/rescored.csv')
S=r[(r.method=='PINN')&(r.preset.str.startswith('C'))].copy().sort_values('preset')
adj=r[(r.method=='Adjoint')&(r.preset=='results_adjoint')&(r.run.str.startswith('B_'))].iloc[0]
fig,axs=plt.subplots(1,2,figsize=(11,4.2))
ax=axs[0]
ax.scatter(S.rms_replay_over_sigma,S.rel_L2_D,s=60,color=ORANGE,edgecolor=SURF,zorder=4,label='PINN, D̂ replayed (CN)')
ax.scatter(np.sqrt(S.pinn_chi2_unet),S.rel_L2_D,s=60,marker='D',color=SURF,edgecolor=ORANGE,linewidth=1.6,zorder=4,label="PINN, own network u_θ")
for _,q in S.iterrows(): ax.annotate(q.preset[:3],(q.rms_replay_over_sigma,q.rel_L2_D),xytext=(5,-3),textcoords='offset points',fontsize=7.5,color=INK2)
ax.scatter([adj.rms_replay_over_sigma],[adj.rel_L2_D],s=90,color=BLUE,edgecolor=SURF,zorder=5,label='Adjoint (Tikhonov)')
rho,pv=spearmanr(S.rms_replay_over_sigma,S.rel_L2_D)
ax.set_xscale('log'); ax.set_xticks([0.5,1,2,5,10,20]); ax.set_xticklabels(['0.5','1','2','5','10','20']); ax.axvline(1,color=INK,lw=0.8,ls=':')
ax.set_xlabel('RMS misfit / σ'); ax.set_ylabel('relL2(D)'); ax.set_ylim(0,0.95); ax.legend(fontsize=8,loc='lower right')
ax.set_title(f'B bump λ-sweep: replay ranks configs (Spearman ρ={rho:.2f}),\nnetwork χ is ~1 for 9/10 and says nothing',fontsize=10)
ax=axs[1]
lab=[p[:3] for p in S.preset]; x=np.arange(len(S))
ax.bar(x,S.rel_L2_D,0.6,color=ORANGE); ax.axhline(adj.rel_L2_D,color=BLUE,lw=2,label=f'adjoint {adj.rel_L2_D:.2f}')
ax.axhline(0.58,color=INK,lw=1.2,ls='--',label='best constant D 0.58')
for i,(v,m) in enumerate(zip(S.rel_L2_D,S.mean_ratio)): ax.text(i,v+0.01,f'{v:.2f}\n×{m:.2f}',ha='center',fontsize=7,color=INK2)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0,1.0); ax.set_ylabel('relL2(D)'); ax.legend(fontsize=8,loc='upper left'); ax.grid(axis='x',visible=False)
ax.set_title('Best preset C01 (all λ=1, no reg) — "winning" C04 is 7th of 10\nlabel ×k = mean(D̂)/mean(D_true)',fontsize=10)
fig.tight_layout(); fig.savefig(O+'L1_lambda_sweep.png'); plt.close()
print('rho',rho,pv)
# loss histories for benchmark runs
runs=sorted([d for d in glob.glob(V2+'results_pinn/*eta0.0[15]')])
fig,axs=plt.subplots(1,3,figsize=(12,3.6))
cols=[BLUE,ORANGE,AQUA,YELLOW,MAGENTA,GREEN,VIOLET,RED]
sel=[d for d in runs if any(t in d for t in ['A_smooth','B_gauss','C_inclusion__Fo0.1__uniform8__n100','D_lay','E_grf','F_def'])]
for d,c in zip(sel,cols):
    h=pd.read_csv(d+'/loss_history.csv'); k=os.path.basename(d).split('__')[0]
    sig2=pd.read_csv(V2+'datasets_v2_rk4_zero/sensors/'+os.path.basename(d)+'.csv').sigma.iloc[0]**2
    sm=lambda v: pd.Series(v).rolling(25,min_periods=1).median()
    axs[0].plot(h.epoch,sm(h.loss_data/sig2),color=c,lw=1.4,label=k.split('_')[0])
    axs[1].plot(h.epoch,sm(h.loss_pde),color=c,lw=1.4)
    axs[2].plot(h.epoch,sm(h.loss_reg),color=c,lw=1.4)
axs[0].axhline(1,color=INK,lw=1,ls='--'); axs[0].set_yscale('log'); axs[0].set_title('data MSE / σ²  (pinned at ~0.95 by design)'); axs[0].legend(fontsize=7.5,ncol=2)
axs[1].set_yscale('log'); axs[1].set_title('raw PDE residual (mean r²)'); axs[2].set_yscale('log'); axs[2].set_title('regulariser R(m): jumps 10⁵× by epoch 250, then flat')
for a in axs: a.set_xlabel('epoch')
fig.tight_layout(); fig.savefig(O+'L2_pinn_training_curves.png'); plt.close()
