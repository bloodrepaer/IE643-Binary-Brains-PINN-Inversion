import sys; sys.path.insert(0,'/home/claude/work')
from style import *
import numpy as np, pandas as pd
d = pd.read_pickle('/tmp/claude-0/-home-claude/1faf916c-94e1-5c39-9cb1-4dbe1a6ae754/scratchpad/tik.pkl')
O='/home/claude/work/fig/'
core = d[d.tier.isin(TIERS)]
ETA_C = {0.0:VIOLET, 0.01:BLUE, 0.05:ORANGE, 0.1:AQUA}
ETA_L = {0.0:'η 0', 0.01:'η 1%', 0.05:'η 5%', 0.1:'η 10%'}
# ---- A1 replay calibration
fig, ax = plt.subplots(1,2, figsize=(10,3.4), gridspec_kw=dict(width_ratios=[1.3,1]))
noisy = d[d.eta>0]
bins = np.linspace(0.85,1.25,41)
for flag,c in [('reached_sigma',BLUE),('floor_above_sigma',ORANGE)]:
    v = noisy[noisy.flag==flag].rms_replay_over_sigma
    ax[0].hist(v.clip(upper=1.249), bins=bins, color=c, label=f'{flag} (n={len(v)})', alpha=0.9, edgecolor=SURF, linewidth=0.5)
ax[0].axvline(1, color=INK, lw=1, ls='--'); ax[0].set_xlabel('RMS_replay / σ  (values >1.25 clipped into last bin)'); ax[0].set_ylabel('datasets')
ax[0].set_title(f'Replay ratio, {len(noisy)} noisy datasets: median {noisy.rms_replay_over_sigma.median():.3f}'); ax[0].legend(loc='upper right')
v = noisy.rms_vs_clean_over_sigma
ax[1].hist(v, bins=30, color=BLUE, edgecolor=SURF, linewidth=0.5)
ax[1].set_xlabel('RMS(replay − clean data) / σ'); ax[1].set_title(f'Error vs noise-free signal: median {v.median():.2f}σ')
fig.savefig(O+'A1_replay_calibration.png'); plt.close()
# ---- A2 S1 sensors x noise
s1 = core[(core.Fo==0.1)&(core.protocol=='uniform8')&(core.layout.isin(['random','nodes']))]
fig, axs = plt.subplots(2,3, figsize=(11,6), sharex=True)
xs = [5,20,50,100,3969]
for ax,t in zip(axs.flat,TIERS):
    g = s1[s1.tier==t]
    for eta in [0.0,0.01,0.05,0.1]:
        h = g[g.eta==eta].set_index('n_sensors').reindex(xs)
        ax.plot(range(5), h.rel_L2_D.values, marker='o', color=ETA_C[eta], label=ETA_L[eta])
    ax.axhline(g.rel_L2_D_constant_guess.median(), color=GREY, ls='--', lw=1.2, label='best constant D')
    ax.set_title(TNAME[t]); ax.set_xticks(range(5)); ax.set_xticklabels(['5','20','50','100','full'])
    ax.set_ylim(0, max(0.8, min(1.6, g.rel_L2_D.max()*1.05)))
for ax in axs[1]: ax.set_xlabel('sensors')
for ax in axs[:,0]: ax.set_ylabel('relL2(D)')
axs[0,0].legend(fontsize=8, loc='upper right')
fig.suptitle('S1 · Accuracy vs sensor count and noise (Fo 0.1, uniform8, random layout; full = all 3,969 nodes)', x=0.01, ha='left', fontsize=11, fontweight='bold')
fig.tight_layout(); fig.savefig(O+'A2_S1_sensors_noise.png'); plt.close()
# ---- A3 S2 horizon x protocol heatmaps
s2 = core[(core.n_sensors==50)&(core.layout=='random')&(core.eta==0.05)&(core.protocol.isin(['uniform8','uniform40','log8','early8','late8','final1']))]
prots = ['uniform8','uniform40','log8','early8','late8','final1']; Fos=[0.02,0.05,0.1,0.2]
fig, axs = plt.subplots(2,3, figsize=(12,7.6)); fig.subplots_adjust(hspace=0.62, wspace=0.28, right=0.86, top=0.88)
for ax,t in zip(axs.flat,TIERS):
    g = s2[s2.tier==t]
    M = g.pivot_table(index='Fo', columns='protocol', values='rel_L2_D').reindex(index=Fos, columns=prots)
    C = g.rel_L2_D_constant_guess.median()
    R = M.values / C
    im = ax.imshow(R, cmap=DIV, vmin=0, vmax=2, aspect='auto')
    for i in range(len(Fos)):
        for j in range(len(prots)):
            v = M.values[i,j]
            if np.isfinite(v): ax.text(j,i,f'{v:.2f}', ha='center', va='center', fontsize=8, color=INK if 0.35<R[i,j]<1.6 else 'white')
    ax.set_xticks(range(len(prots))); ax.set_xticklabels(prots, rotation=35, fontsize=8); ax.set_yticks(range(4)); ax.set_yticklabels(Fos)
    ax.set_title(f'{TNAME[t]} · const. D {C:.2f}'); ax.grid(False)
for ax in axs[:,0]: ax.set_ylabel('Fourier number Fo')
cax=fig.add_axes([0.89,0.2,0.015,0.6]); cb = fig.colorbar(im, cax=cax); cb.set_label('relL2 ÷ constant-D relL2   (1 = no better than a constant)')
fig.suptitle('S2 · Observation horizon × time protocol (50 random sensors, η 5%) — cell text = relL2(D)', x=0.01, ha='left', fontsize=11, fontweight='bold')
fig.savefig(O+'A3_S2_horizon_protocol.png'); plt.close()
# ---- A4 S3 temporal density x noise
s3 = core[(core.Fo==0.1)&(core.n_sensors==50)&(core.layout=='random')&(core.protocol.str.startswith('uniform'))]
s3 = s3.assign(nt=s3.protocol.str.replace('uniform','').astype(int))
fig, axs = plt.subplots(2,3, figsize=(11,6), sharex=True)
for ax,t in zip(axs.flat,TIERS):
    g = s3[s3.tier==t]
    for eta in [0.01,0.05,0.1]:
        h = g[g.eta==eta].sort_values('nt'); ax.plot(h.nt, h.rel_L2_D, marker='o', color=ETA_C[eta], label=ETA_L[eta])
    ax.axhline(g.rel_L2_D_constant_guess.median(), color=GREY, ls='--', lw=1.2, label='best constant D')
    ax.set_xscale('log'); ax.set_xticks([4,8,16,40,80,200]); ax.set_xticklabels([4,8,16,40,80,200]); ax.set_title(TNAME[t]); ax.set_ylim(bottom=0)
for ax in axs[1]: ax.set_xlabel('time levels (uniform)')
for ax in axs[:,0]: ax.set_ylabel('relL2(D)')
axs[0,0].legend(fontsize=8)
fig.suptitle('S3 · Temporal density × noise (Fo 0.1, 50 random sensors)', x=0.01, ha='left', fontsize=11, fontweight='bold')
fig.tight_layout(); fig.savefig(O+'A4_S3_temporal_density.png'); plt.close()
# ---- A5 S4 layout x count
s4 = core[(core.Fo==0.1)&(core.protocol=='uniform8')&(core.eta==0.05)&(core.n_sensors.isin([20,50]))]
lays = ['random','grid','clustered','boundary']; LC = dict(zip(lays,[BLUE,ORANGE,AQUA,VIOLET]))
fig, axs = plt.subplots(1,2, figsize=(11,3.6), sharey=True)
w=0.2
for ax,n in zip(axs,[20,50]):
    for k,l in enumerate(lays):
        vals = [s4[(s4.tier==t)&(s4.n_sensors==n)&(s4.layout==l)].rel_L2_D.mean() for t in TIERS]
        ax.bar(np.arange(6)+(k-1.5)*w, np.minimum(vals,1.5), w*0.92, color=LC[l], label=l)
        for i,v in enumerate(vals):
            if v>1.5: ax.text(i+(k-1.5)*w, 1.5, f'{v:.2f}', ha='center', va='bottom', fontsize=7.5, color=INK)
    cg=[core[core.tier==t].rel_L2_D_constant_guess.median() for t in TIERS]
    ax.scatter(np.arange(6), cg, marker='_', s=500, color=INK, zorder=5, label='best constant D')
    ax.set_xticks(range(6)); ax.set_xticklabels([TNAME[t] for t in TIERS], fontsize=8.5); ax.set_title(f'{n} sensors'); ax.grid(axis='x', visible=False)
axs[0].set_ylabel('relL2(D)  (capped at 1.5)'); axs[1].legend(fontsize=8, ncol=2)
fig.suptitle('S4 · Sensor layout × count (Fo 0.1, uniform8, η 5%)', x=0.01, ha='left', fontsize=11, fontweight='bold')
fig.tight_layout(); fig.savefig(O+'A5_S4_layout.png'); plt.close()
# ---- A6 fit vs accuracy: replay can't see D error
fig, axs = plt.subplots(1,2, figsize=(11,3.8))
nz = core[core.eta>0]
ratio = nz.rel_L2_D/nz.rel_L2_D_constant_guess
isD = nz.tier=='D_layered'
axs[0].scatter(nz.rms_replay_over_sigma[~isD], ratio[~isD], s=22, color=BLUE, alpha=0.75, edgecolor=SURF, linewidth=0.6, label='tiers A,B,C,E,F')
axs[0].scatter(nz.rms_replay_over_sigma[isD], ratio[isD], s=22, color=ORANGE, alpha=0.9, edgecolor=SURF, linewidth=0.6, label='tier D layered')
axs[0].axhline(1, color=GREY, ls='--', lw=1.2); axs[0].axvline(1, color=INK, lw=0.8, ls=':')
axs[0].set_xlim(0.75,1.45); axs[0].set_xlabel('RMS_replay / σ'); axs[0].set_ylabel('relL2 ÷ constant-guess relL2'); axs[0].legend(fontsize=8)
axs[0].set_title(f'Fitting to σ does not certify D: {(ratio>1).sum()} of {len(nz)} worse than a constant')
ax=axs[1]
hi, lo = core.rel_L2_D_high_sensitivity, core.rel_L2_D_low_sensitivity
ax.scatter(hi, lo, s=18, color=BLUE, alpha=0.7, edgecolor=SURF, linewidth=0.5)
m=1.6; ax.plot([0,m],[0,m], color=INK, lw=0.8, ls='--'); ax.set_xlim(0,m); ax.set_ylim(0,m)
frac=(lo>hi).mean(); ax.set_xlabel('relL2 in high-sensitivity half'); ax.set_ylabel('relL2 in low-sensitivity half')
ax.set_title(f'Low-flux half is worse in {frac:.0%} of {len(core)} datasets'); ax.set_aspect('equal')
fig.tight_layout(); fig.savefig(O+'A6_fit_vs_accuracy_sensitivity.png'); plt.close()
# ---- A7 cost
fig, ax = plt.subplots(figsize=(6.5,3.6))
for t,c in zip(['D_layered'],[ORANGE]): pass
ax.scatter(d.n_readings, d.wall_s/60, s=16, color=BLUE, alpha=0.6, edgecolor=SURF, linewidth=0.5, label='others')
dd=d[d.tier=='D_layered']; ax.scatter(dd.n_readings, dd.wall_s/60, s=18, color=ORANGE, alpha=0.9, edgecolor=SURF, linewidth=0.5, label='tier D')
ax.set_xscale('log'); ax.set_xlabel('readings in dataset'); ax.set_ylabel('wall-clock (min, 1 core)')
ax.set_title(f'Adjoint cost: median {d.wall_s.median()/60:.1f} min, total {d.wall_s.sum()/3600:.1f} core-h'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(O+'A7_cost.png'); plt.close()
print('done')
