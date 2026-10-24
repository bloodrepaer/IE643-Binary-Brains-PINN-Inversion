"""Aggregate already validated v24 scores; no training, forward solve or truth access."""
from pathlib import Path
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('campaign/pilots/noise_density_v24_1000_7500')
DENSITIES = [1000, 3000, 5000, 7500]
NOISES = [.01, .05, .1]
COLORS = {'temporal_pinn': '#1768AC', 'be_adjoint_nn': '#D66B20'}
LABELS = {'temporal_pinn': 'Temporal PINN', 'be_adjoint_nn': 'BE neural adjoint'}


def main():
    data = pd.read_csv(ROOT / 'primary_matched_neural_comparison.csv')
    assert len(data) == 24 and not data.duplicated(['n_sensors', 'eta', 'arm']).any()
    assert data[['key', 'sigma', 'replay_cn_rms', 'replay_be_rms']].notna().all().all()
    assert (data.status == 'complete').all()
    assert (data.adam_completed == 1000).all()
    assert (data.quasi_newton_iterations == 4000).all()
    assert (data.capped_rounds == 10).all()
    rows = []
    for eta in NOISES:
        for n in DENSITIES:
            pair = data[(data.eta == eta) & (data.n_sensors == n)]
            assert len(pair) == 2
            a = pair[pair.arm == 'temporal_pinn'].iloc[0]
            b = pair[pair.arm == 'be_adjoint_nn'].iloc[0]
            apass = bool(a.replay_cn_sigma <= 1.1 and a.replay_be_sigma <= 1.1)
            bpass = bool(b.replay_cn_sigma <= 1.1 and b.replay_be_sigma <= 1.1)
            rows.append(dict(
                n_sensors=n, eta=eta, status='complete',
                pinn_error_percent=100*a.rel_L2_D, adjoint_error_percent=100*b.rel_L2_D,
                pinn_cn_sigma=a.replay_cn_sigma, pinn_be_sigma=a.replay_be_sigma,
                adjoint_cn_sigma=b.replay_cn_sigma, adjoint_be_sigma=b.replay_be_sigma,
                pinn_both_replay_pass=apass, adjoint_both_replay_pass=bpass,
                pinn_consistency_qualified_advantage=apass and a.rel_L2_D < b.rel_L2_D,
                adjoint_consistency_qualified_advantage=bpass and b.rel_L2_D < a.rel_L2_D))
    summary = pd.DataFrame(rows)
    assert len(summary) == 12
    summary.to_csv(ROOT / 'paired_cell_summary.csv', index=False)
    coverage = dict(expected_cells=12, available_complete_cells=12, complete_fit_outputs=24,
                    missing_cells=[], coverage_complete=True, agent_scope_complete=True,
                    pinn_consistency_qualified_advantage_cells=int(summary.pinn_consistency_qualified_advantage.sum()),
                    adjoint_consistency_qualified_advantage_cells=int(summary.adjoint_consistency_qualified_advantage.sum()),
                    all_available_fits_full_but_capped=True,
                    coefficient_truth_used_for_training_or_selection=False)
    (ROOT / 'report_coverage.json').write_text(json.dumps(coverage, indent=2))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, eta in zip(axes, NOISES):
        for arm in COLORS:
            s = data[(data.eta == eta) & (data.arm == arm)].sort_values('n_sensors')
            ax.plot(s.n_sensors, 100*s.rel_L2_D, 'o-', color=COLORS[arm], label=LABELS[arm], lw=2)
        ax.set_title(f'{eta*100:g}% noise')
        ax.set_xticks(DENSITIES, ['1k', '3k', '5k', '7.5k'])
        ax.set_xlabel('Distinct sensor locations')
        ax.grid(alpha=.2)
        ax.set_ylim(0, float((100*data.rel_L2_D).max())*1.08)
    axes[0].set_ylabel('Coefficient relative L2 error (%)')
    axes[0].legend(frameon=False)
    fig.suptitle('Density trend — all 12 paired cells complete', fontsize=14)
    fig.text(.5, .01, 'One field / realization / seed; capped optimization. Gamma changes with density and noise; coefficient accuracy alone is not a physics-qualified win.', ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(ROOT / 'density_trend.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(15, 9.5), sharex=True, sharey='row')
    metrics = [('rel_L2_D', 'Coefficient error (%)'), ('replay_be_rms', 'Common BE raw RMS (log)'), ('replay_be_sigma', 'Common BE RMS / sigma')]
    for col, n in enumerate(DENSITIES):
        for row, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            for arm in COLORS:
                s = data[(data.n_sensors == n) & (data.arm == arm)].sort_values('eta')
                values = s[metric] * (100 if metric == 'rel_L2_D' else 1)
                ax.plot(100*s.eta, values, 'o-', color=COLORS[arm], label=LABELS[arm], lw=2)
            if row == 0:
                ax.set_title(f'{n:,} sensors')
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_yscale('log')
            if row == 2:
                ax.axhline(1.1, color='#888888', ls='--', lw=1)
                ax.set_xlabel('Noise (%)')
            ax.set_xticks([1, 5, 10])
            ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle('Noise trend — all four densities; identical vertical scales within each row', fontsize=13)
    fig.text(.5, .01, 'Dashed replay limit: 1.1 sigma. Common CN and BE, raw RMS and RMS/sigma are both retained in the tables. Noise also changes gamma.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(ROOT / 'noise_trend.png', dpi=180)
    plt.close(fig)

    text = ['# Completed noise / density sweep', '',
            f'Updated {datetime.now(timezone.utc).isoformat()}. **12/12 paired cells complete; 24 full fits. No active jobs, missing cells or further dispatch.**', '',
            'This scoped sweep is complete. The broader campaign and its held experiments are separate. All available fits completed 1,000 Adam + 4,000 SSB; every outer round hit its iteration cap. No budget timeouts. Independent supplied-physics, protocol, checkpoint and both common-solver checks preceded coefficient truth scoring. Aggregation only reads validated result tables; no training, solver evaluation or coefficient-truth access.', '',
            '| Sensors | Noise | PINN coefficient error | BE adjoint error | PINN CN / BE replay (sigma) | Adjoint CN / BE replay (sigma) | PINN qualified advantage |',
            '|---:|---:|---:|---:|---|---|---|']
    for x in rows:
        text.append(f'| {x["n_sensors"]:,} | {x["eta"]*100:g}% | {x["pinn_error_percent"]:.2f}% | {x["adjoint_error_percent"]:.2f}% | {x["pinn_cn_sigma"]:.4f} / {x["pinn_be_sigma"]:.4f} | {x["adjoint_cn_sigma"]:.4f} / {x["adjoint_be_sigma"]:.4f} | {"Yes, single case" if x["pinn_consistency_qualified_advantage"] else "No"} |')
    text += ['', 'Observed findings:', '',
             '- At 5% noise, increasing density from 1,000 to 7,500 lowers PINN coefficient error from 35.37% to 8.62%; BE adjoint error rises from 21.45% to 25.09%. PINN has lower coefficient error at 3,000, 5,000 and 7,500, but only the 7,500-sensor fit passes both replay screens.',
             '- The new 5,000/5% PINN has 13.03% error versus adjoint 23.79%. PINN BE replay passes (1.068481 sigma), while CN replay narrowly fails (1.102375 > 1.1). At 7,500/5%, PINN CN replay narrowly passes (1.097229). Keep continuous values visible: a narrow fixed-threshold pass/fail difference is not evidence of a robust population boundary.',
             '- At 1% noise, PINN coefficient error improves from 9.40% to 5.61% with density, but all PINNs fail both replays. BE adjoint coefficient error rises from 32.04% to 45.64%; every BE replay passes while CN replay fails. This mismatch does not establish a unique cause, convergence or general method superiority.',
             '- At 10% noise, PINN error falls from 69.10% to 22.64% with density. Adjoint errors are 21.58%–23.26%, and all its replays pass both screens. PINN at 7,500 has slightly lower coefficient error than adjoint but fails BE replay (1.121848), despite passing CN.',
             f'- PINN has a consistency-qualified coefficient advantage in {coverage["pinn_consistency_qualified_advantage_cells"]} of 12 primary cells; adjoint has lower coefficient error with both replay screens passing in {coverage["adjoint_consistency_qualified_advantage_cells"]} cells. Remaining cells have no such qualified advantage. These counts describe this fixed exploratory matrix, not population win rates.', '',
             'Definition: a consistency-qualified coefficient advantage requires completed fits, lower relative L2 coefficient error against the matched neural method, independent supplied-physics/protocol checks, and both common CN and BE replay <=1.1 sigma. Thresholds, evaluation solvers, checkpoints and seeds were not selected after seeing coefficient truth. Passing implementation checks is distinct from replay consistency; passing the replay screen is not exact PDE satisfaction.', '',
             'Limits: one smooth field, one common observation/noise realization, one optimizer seed and capped optimization (not proven convergence). Gamma changes with noise/count at fixed alpha. The parent observation-fitted 5%-data warm start is shared. Nested locations sample a fine-grid RK4 forward dataset with a 65-grid inversion; the data were not CN-generated. AD/FD, shifted random pairs/fixed grid and soft/hard state constraints remain different finite objectives. Colab T4/CPU and Kaggle CPU resources differ between cells; no isolated causal density/noise, speed or bitwise trajectory claim. Continuous PDE probes, IC/BC, raw and sigma-normalized CN/BE replays, actual caps/evaluations/resources and statuses are retained in individual case reports and the full CSV. The existing 10,000-sensor reference lies outside this primary matrix and is excluded from the counts/plots.', '',
             'Legacy v22 aggregation: key, eta and sigma come from its unchanged frozen protocol/metrics; raw RMS is its existing validated normalized replay times sigma. No re-evaluation/refit. Both Colab returns preserve original export manifests/notebooks/attempts, and the duplicated 3,000-sensor result in the second archive was bitwise identical and reused. Earlier partial tables/reports/figures remain retained.', '',
             '![Density trend](density_trend.png)', '', '![Noise trend](noise_trend.png)', '',
             'No additional fits are authorized by completion of this report. Held v20/other fields/UI remain held.']
    (ROOT / 'SWEEP_REPORT.md').write_text('\n'.join(text) + '\n')
    print(json.dumps(coverage))


if __name__ == '__main__':
    main()
