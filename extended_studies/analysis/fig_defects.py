import sys; sys.path.insert(0,'/home/claude/work')
from style import *
import numpy as np, pandas as pd
r=pd.read_csv('/home/claude/work/rescored.csv'); O='/home/claude/work/fig/'
var=['F_defects','F_defects_single','F_defects_many','F_defects_small','F_defects_low_contrast','F_defects_voids']
lab=['default (3)','single','many (6)','small r','low contrast','voids']
k=lambda v: f'{v}__Fo0.1__uniform8__n50__random__eta0.05'
T=r[r.preset=='results_defects_tikhonov'].set_index('run'); V=r[r.preset=='results_defects_tv'].set_index('run')
P=r[(r.method=='PINN')&(r.preset=='benchmark')].set_index('run')
fig,ax=plt.subplots(figsize=(10,3.8)); x=np.arange(len(var)); w=0.27
for off,src,c,name in [(-w,T,BLUE,'Adjoint · Tikhonov'),(0,V,AQUA,'Adjoint · TV')]:
    dice=[src.loc[k(v),'defect_dice'] for v in var]; det=[f"{int(src.loc[k(v),'defects_detected'])}/{int(src.loc[k(v),'n_defects'])}" for v in var]
    ax.bar(x+off,dice,w*0.92,color=c,label=name)
    for i,(dd,tt) in enumerate(zip(dice,det)): ax.text(i+off,dd+0.02,tt,ha='center',fontsize=8,color=INK2)
pd_=P.loc[k('F_defects')]
ax.bar([w],[pd_.defect_dice],w*0.92,color=ORANGE,label='PINN · TV (default only)'); ax.text(w,0.02,f"{int(pd_.defects_detected)}/3",ha='center',fontsize=8,color=INK2)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel('Dice of anomaly map'); ax.set_ylim(0,1); ax.legend(fontsize=8,loc='upper center'); ax.grid(axis='x',visible=False)
ax.set_title('Tier F (50 sensors, η 5%) — labels = defects detected. TV ≥ Tikhonov; small / low-contrast defects are below the data limit')
fig.tight_layout(); fig.savefig(O+'F1_defects.png'); plt.close()
