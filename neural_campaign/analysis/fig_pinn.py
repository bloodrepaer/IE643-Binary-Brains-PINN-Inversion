import sys; sys.path.insert(0,'/home/claude/work')
from style import *
import numpy as np, pandas as pd, json
O='/home/claude/work/fig/'; V2='/home/claude/v2/'
r=pd.read_csv('/home/claude/work/rescored.csv'); F=np.load('/home/claude/work/fields.npz')
P=r[(r.method=='PINN')&(r.preset=='benchmark')].set_index('run')
A=r[(r.method=='Adjoint')&(r.preset=='results_adjoint')].set_index('run')
ATV=r[(r.method=='Adjoint')&(r.preset.isin(['results_adjoint_tv','results_defects_tv']))].set_index('run')
cases=list(P.index)
short=lambda k: k.replace('__Fo0.1__uniform8','').replace('__random','').replace('_gaussian_bump',' bump').replace('_smooth',' smooth').replace('_inclusion',' incl.').replace('_layered',' layered').replace('_grf',' GRF').replace('_defects',' defects').replace('__n','\nn').replace('__eta0.05',' η5%').replace('__eta0.01',' η1%')
order=['A_smooth','B_gaussian','C_inclusion__Fo0.1__uniform8__n5_','C_inclusion__Fo0.1__uniform8__n20','C_inclusion__Fo0.1__uniform8__n50__random__eta0.01','C_inclusion__Fo0.1__uniform8__n100','C_inclusion__Fo0.1__uniform8__n50__random__eta0.05','D_layered','E_grf','F_defects']
cases=[next(c for c in cases if c.startswith(o)) for o in order]
tik=pd.read_pickle('/tmp/claude-0/-home-claude/1faf916c-94e1-5c39-9cb1-4dbe1a6ae754/scratchpad/tik.pkl').set_index('key')
# P1 accuracy bars
fig,ax=plt.subplots(figsize=(11,4)); x=np.arange(len(cases)); w=0.26
a=[A.loc[c,'rel_L2_D'] for c in cases]; p=[P.loc[c,'rel_L2_D'] for c in cases]
tv=[ATV.loc[c,'rel_L2_D'] if c in ATV.index else np.nan for c in cases]
cg=[tik.loc[c,'rel_L2_D_constant_guess'] for c in cases]
ax.bar(x-w,a,w*0.92,color=BLUE,label='Adjoint · Tikhonov'); ax.bar(x,tv,w*0.92,color=AQUA,label='Adjoint · TV (where run)'); ax.bar(x+w,p,w*0.92,color=ORANGE,label='PINN (reg. as run)')
ax.scatter(x,cg,marker='_',s=700,color=INK,zorder=5,label='best constant D')
for i,(pp,cc) in enumerate(zip(p,cg)): ax.text(i+w,pp+0.01,f'{pp:.2f}',ha='center',fontsize=7.5,color=INK2)
for i,v in enumerate(a): ax.text(i-w,v+0.01,f'{v:.2f}',ha='center',fontsize=7.5,color=INK2)
ax.set_xticks(x); ax.set_xticklabels([short(c)+('\nPINN: TV' if P.loc[c,'reg']=='tv' else '') for c in cases],fontsize=7.8)
ax.set_ylabel('relL2(D)'); ax.set_ylim(0,0.9); ax.legend(ncol=4,fontsize=8,loc='upper left'); ax.grid(axis='x',visible=False)
ax.set_title(f'Adjoint beats the PINN on all 10 benchmark cases (median relL2 {np.median(a):.2f} vs {np.median(p):.2f}); PINN is worse than a constant D on A, E, F')
fig.tight_layout(); fig.savefig(O+'P1_accuracy_pinn_vs_adjoint.png'); plt.close()
# P2 replay consistency
fig,ax=plt.subplots(figsize=(11,3.9))
un=[np.sqrt(P.loc[c,'pinn_chi2_unet']) for c in cases]; pr=[P.loc[c,'rms_replay_over_sigma'] for c in cases]; ar=[A.loc[c,'rms_replay_over_sigma'] for c in cases]
ax.scatter(x-0.12,un,s=60,marker='D',color=SURF,edgecolor=ORANGE,linewidth=1.8,zorder=4,label="PINN's own network u_θ at sensors")
ax.scatter(x+0.12,pr,s=60,color=ORANGE,edgecolor=SURF,linewidth=1.2,zorder=4,label='PINN D̂ replayed through CN solver')
ax.scatter(x,ar,s=60,color=BLUE,edgecolor=SURF,linewidth=1.2,zorder=5,label='Adjoint D̂ replayed through CN solver')
for i in range(len(cases)): ax.plot([i-0.12,i+0.12],[un[i],pr[i]],color=ORANGE,lw=1,alpha=0.6)
ax.axhline(1,color=INK,lw=1,ls='--'); ax.set_yscale('log'); ax.set_yticks([0.5,1,2,5,10,20]); ax.set_yticklabels(['0.5','1','2','5','10','20'])
ax.set_xticks(x); ax.set_xticklabels([short(c) for c in cases],fontsize=7.8); ax.set_ylabel('RMS misfit / σ'); ax.grid(axis='x',visible=False)
ax.legend(fontsize=8,loc='upper left'); ax.set_title("PINN's u and D disagree: its network fits to ~1σ, but its D only reproduces the data to 2–16σ")
fig.tight_layout(); fig.savefig(O+'P2_replay_consistency.png'); plt.close()
# P3 field gallery
gal=['A_smooth__Fo0.1__uniform8__n50__random__eta0.05','B_gaussian_bump__Fo0.1__uniform8__n50__random__eta0.05','C_inclusion__Fo0.1__uniform8__n100__random__eta0.01','D_layered__Fo0.1__uniform8__n50__random__eta0.05','E_grf__Fo0.1__uniform8__n50__random__eta0.05','F_defects__Fo0.1__uniform8__n50__random__eta0.05']
fig,axs=plt.subplots(len(gal),5,figsize=(12.5,15))
for i,k in enumerate(gal):
    Dt=F[k+'|Dt']; Da=F['results_adjoint/'+k+'|D']; Dp=F[k+'|D']
    s=pd.read_csv(V2+f'datasets_v2_rk4_zero/sensors/{k}.csv')[['x','y']].drop_duplicates()
    vmin,vmax=min(Dt.min(),Da.min(),Dp.min()),max(Dt.max(),Da.max(),Dp.max())
    ext=[0,1,0,1]
    for j,(M,tt) in enumerate([(Dt,'true D'),(Da,f"adjoint  relL2 {A.loc[k,'rel_L2_D']:.2f}"),(Dp,f"PINN  relL2 {P.loc[k,'rel_L2_D']:.2f}")]):
        im=axs[i,j].imshow(M.T,origin='lower',extent=ext,cmap=SEQ,vmin=vmin,vmax=vmax); axs[i,j].set_title(tt,fontsize=9.5)
    axs[i,0].scatter(s.x,s.y,s=7,color=ORANGE,edgecolor='white',linewidth=0.4)
    cb=fig.colorbar(im,ax=axs[i,2],fraction=0.046,pad=0.03); cb.ax.tick_params(labelsize=7)
    for j,(M,tt) in enumerate([(Da,'log(adjoint / true)'),(Dp,'log(PINN / true)')]):
        im2=axs[i,3+j].imshow(np.log(M/Dt).T,origin='lower',extent=ext,cmap=DIV,vmin=-1.2,vmax=1.2); axs[i,3+j].set_title(tt,fontsize=9.5)
    cb=fig.colorbar(im2,ax=axs[i,4],fraction=0.046,pad=0.03); cb.ax.tick_params(labelsize=7)
    axs[i,0].set_ylabel(short(k).replace('\n',' '),fontsize=10,color=INK,fontweight='bold')
    for ax in axs[i]: ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle('Recovered D fields (shared colour scale per row; dots = sensors) and log-error maps (blue = too low, red = too high)',x=0.01,ha='left',fontsize=11,fontweight='bold',y=0.995)
fig.tight_layout(); fig.savefig(O+'P3_field_gallery.png'); plt.close()
# P4 scale bias & decomposition
fig,axs=plt.subplots(1,2,figsize=(11,3.9))
mrA=[A.loc[c,'mean_ratio'] for c in cases]; mrP=[P.loc[c,'mean_ratio'] for c in cases]
axs[0].bar(x-0.2,mrA,0.38,color=BLUE,label='Adjoint'); axs[0].bar(x+0.2,mrP,0.38,color=ORANGE,label='PINN'); axs[0].axhline(1,color=INK,lw=1,ls='--')
axs[0].set_ylim(0,1.15); axs[0].set_ylabel('mean(D̂) / mean(D_true)'); axs[0].set_title('PINN underestimates the overall level of D by 23–42%')
axs[0].set_xticks(x); axs[0].set_xticklabels([short(c) for c in cases],fontsize=7); axs[0].legend(fontsize=8,loc='lower left'); axs[0].grid(axis='x',visible=False)
raw=np.array(p); sc=np.array([P.loc[c,'relL2_scale_corrected'] for c in cases])
axs[1].bar(x,sc,0.6,color=ORANGE,label='shape error (after best rescale c·D̂)'); axs[1].bar(x,raw-sc,0.6,bottom=sc,color='#f6b89d',label='extra error from wrong scale')
axs[1].scatter(x,a,marker='_',s=350,color=BLUE,lw=2.5,zorder=5,label='adjoint relL2 (unscaled)')
axs[1].set_xticks(x); axs[1].set_xticklabels([short(c) for c in cases],fontsize=7); axs[1].set_ylabel('PINN relL2(D)'); axs[1].legend(fontsize=7.5,loc='upper left'); axs[1].grid(axis='x',visible=False)
axs[1].set_title('Even with the scale fixed, PINN shape error > adjoint total'); axs[1].set_ylim(0,0.95)
fig.tight_layout(); fig.savefig(O+'P4_scale_bias.png'); plt.close()
# P5 data-limited vs method-limited (C, eta 1%)
ns=[5,20,50,100]; ks=[f'C_inclusion__Fo0.1__uniform8__n{n}__random__eta0.01' for n in ns]
fig,ax=plt.subplots(figsize=(6.5,3.8))
ax.plot(range(4),[A.loc[k,'rel_L2_D'] for k in ks],marker='o',color=BLUE,label='Adjoint · Tikhonov')
ax.plot(range(4),[P.loc[k,'rel_L2_D'] for k in ks],marker='o',color=ORANGE,label='PINN · TV')
ax.plot([3],[ATV.loc[ks[3],'rel_L2_D']],marker='s',color=AQUA,ls='none',ms=8,label='Adjoint · TV')
ax.axhline(tik.loc[ks[0],'rel_L2_D_constant_guess'],color=GREY,ls='--',lw=1.2,label='best constant D')
ax.set_xticks(range(4)); ax.set_xticklabels(ns); ax.set_xlabel('sensors'); ax.set_ylabel('relL2(D)'); ax.set_ylim(0,0.85)
ax.set_title('C inclusion, η 1%: adjoint uses more data, PINN does not'); ax.legend(fontsize=8,loc='lower left')
fig.tight_layout(); fig.savefig(O+'P5_density_C.png'); plt.close()
# P6 slices
fig,axs=plt.subplots(1,4,figsize=(12,3.3)); xs=np.linspace(0,1,65)
for ax,k,lab in zip(axs,[gal[0],gal[2],gal[3],gal[5]],['A smooth','C incl. n100 η1%','D layered','F defects']):
    Dt=F[k+'|Dt']; Da=F['results_adjoint/'+k+'|D']; Dp=F[k+'|D']
    j=int(np.argmax(np.abs(Dt-np.median(Dt)).sum(0))) if 'C_' in k or 'F_' in k else 32
    ax.plot(xs,Dt[:,j],color=INK,lw=2,label='true'); ax.plot(xs,Da[:,j],color=BLUE,label='adjoint'); ax.plot(xs,Dp[:,j],color=ORANGE,label='PINN')
    ax.set_title(f'{lab} · y = {xs[j]:.2f}'); ax.set_xlabel('x'); ax.set_ylim(bottom=0)
axs[0].set_ylabel('D'); axs[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(O+'P6_slices.png'); plt.close()
print('ok')
