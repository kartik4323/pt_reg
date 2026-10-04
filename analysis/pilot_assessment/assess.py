"""Read-only analysis of the uploaded pilot reports; writes adjacent artifacts."""
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
documents = {split: json.loads((ROOT / f'pilot_report_{split}.json').read_text())
             for split in ('test', 'cut_holdout')}
summary = {'inputs': {}, 'evaluations': {}}
for split, document in documents.items():
    source = ROOT / f'pilot_report_{split}.json'
    summary['inputs'][split] = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                               'advance_eligible': document['advance_eligible']}
    for report in document['reports']:
        if report.get('kind') != 'evaluation' or report['purpose'] != 'pilot':
            continue
        pairs = [pair for row in report['samples'] for pair in row['diagnostics']['pairs'].values()]
        rows = report['samples']
        summary['evaluations'][f'{split}/{report["condition"]}'] = {
            'examples': len(rows), 'sources': len({r['source_id'] for r in rows}),
            'successes': sum(r['success'] for r in rows),
            'failure_breakdown': report['failure_breakdown'],
            'pairs': len(pairs), 'pairs_below_total_mass_0.05': sum(p['raw_mass'] < .05 for p in pairs),
            'pair_candidates': sum(p['candidate_count'] for p in pairs),
            'assembly_hypotheses': sum(r['diagnostics']['hypotheses'] for r in rows),
            'sdf_l1': report['predicted_sdf_l1'],
        }
training = [r for r in documents['test']['reports'] if r.get('kind') == 'training']
summary['training'] = [{key: r[key] for key in ('condition', 'stage', 'seconds', 'metrics', 'cuda_peak_reserved_bytes')}
                       for r in training]
preflight = next(r for r in documents['test']['reports'] if r.get('kind') == 'preflight')
summary['resources'] = {'gpu': preflight['attempts'][-1]['gpu_name'],
                        'preflight_reserved_gib': preflight['attempts'][-1]['max_reserved_bytes'] / 2**30,
                        'six_stage_training_hours': sum(r['seconds'] for r in training) / 3600,
                        'managed_mib_at_end_of_training': training[-1]['storage']['managed_bytes'] / 2**20}
(HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3))
for stage, axis in enumerate(axes, 1):
    for prefix, label, color in [('overfit', 'Fixed 16: same examples', '#267C68'),
                                  ('predicted', 'Pilot: held-out validation', '#C54F37')]:
        log = (ROOT / 'logs' / f'{prefix}-s{stage}.log').read_text()
        values = re.findall(r'stage=\d+ update=(\d+)/\d+ train=([-\d.eE+]+) val=([-\d.eE+]+)', log)
        # Appended logs may contain several attempts. Keep the last occurrence.
        curve = {int(step): float(val) for step, _, val in values}
        xs = sorted(curve)
        axis.plot(xs, [curve[x] for x in xs], label=label, color=color, linewidth=2)
    axis.set_title(['Stage 1: geometry', 'Stage 2: scaffold', 'Stage 3: matching'][stage-1])
    axis.set_ylabel(['Composite validation loss', 'Validation SDF L1', 'Composite matching loss'][stage-1])
    axis.set_xlabel('Optimizer updates')
    axis.grid(alpha=.2)
    axis.spines[['top', 'right']].set_visible(False)
    axis.set_xlim(100, 2000)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle('Fixed-example and held-out validation curves', fontsize=15, x=.05, ha='left')
fig.text(.05, .015, 'Different data/pose regimes; stage-specific losses are not comparable across panels. '
         'Held-out assembly success: 0/69 test, 0/11 unseen cuts.', fontsize=9)
fig.tight_layout(rect=(0, .055, 1, .94))
fig.savefig(HERE / 'training_curves.png', dpi=170)
print(json.dumps(summary['resources']))
print('Wrote summary.json and training_curves.png')
