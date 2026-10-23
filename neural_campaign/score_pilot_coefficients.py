"""Separate answer-key scoring AFTER pilot fitting/consistency evaluation.

Scores all retained arms/cases; never selects or refits a checkpoint.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pilot-dir',required=True);ap.add_argument('--data',default='datasets_v2_rk4_zero')
    a=ap.parse_args();home=Path(a.pilot_dir);root=Path(a.data)
    if not (home/'validation.json').exists():raise ValueError('Run independent consistency evaluation first')
    entries={e['key']:e for e in json.loads((root/'manifest.json').read_text())};rows=[]
    for path in sorted((home/'artifacts').rglob('metrics.json')):
        m=json.loads(path.read_text());entry=entries[m['key']]
        with np.load(path.with_name('D_hat.npz')) as z:D=z['D_hat'].copy()
        with np.load(root/entry['files']['ground_truth']) as z:truth=z['D_inv'].copy()
        if D.shape!=truth.shape or not np.isfinite(D).all() or np.any(D<=0):raise ValueError('Invalid coefficient')
        rows.append(dict(key=m['key'],variant=m.get('variant',m.get('state_model','legacy')),artifact=str(path.relative_to(home/'artifacts')),
            n_sensors=entry.get('n_sensors'),eta=entry.get('eta'),rel_L2_D=float(np.linalg.norm(D-truth)/np.linalg.norm(truth)),
            rmse_logD=float(np.sqrt(np.mean(np.log(D/truth)**2))),replay_sigma=m['rms_replay_over_sigma'],sensor_fit_sigma=m['rms_unet_over_sigma']))
    pd.DataFrame(rows).to_csv(home/'coefficient_scores.csv',index=False)
    (home/'coefficient_scoring_protocol.json').write_text(json.dumps(dict(answer_key_use='Post-fit scoring only, after independent consistency checks',
        scores_all_retained_cases=True,checkpoint_or_setting_selection=False,training=False),indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
