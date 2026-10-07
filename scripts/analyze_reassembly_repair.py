"""Audit extracted repair results and replay saved XYZ-only matches on CPU.

Run from the repository root. Canonical coordinates are used only for metrics,
never for correspondence filtering, candidate generation or pose selection.
Saved tensors cover a selected subset at samples_poses/4101, not the full gate.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from reassembly.evaluation import chamfer
from reassembly.repair.assembly import build_candidates_from_matches, solve_candidates


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def counts(rows):
    return dict(count=len(rows), geometric_successes=sum(r['success'] for r in rows),
                accepted_successes=sum(r['success'] and r['status'] == 'ok' for r in rows),
                ok=sum(r['status'] == 'ok' for r in rows),
                low_confidence=sum(r['status'] == 'low_confidence' for r in rows),
                candidate_coverage=sum(r['candidate_coverage']['candidate_oracle_success'] for r in rows))


def metrics(poses, arrays):
    if poses[0] is None:
        return dict(success=False, max_part_chamfer=None)
    rotation, translation = poses
    count = int(arrays['fragment_mask'].sum())
    aligned = np.einsum('fni,fji->fnj', arrays['points'][:count], rotation) + translation[:, None]
    cds = [chamfer(a, b) for a, b in zip(aligned, arrays['canonical_points'][:count])]
    whole = chamfer(aligned.reshape(-1, 3), arrays['canonical_points'][:count].reshape(-1, 3))
    return dict(success=max(cds) <= .01 and whole <= .01, max_part_chamfer=max(cds), whole_chamfer=whole)


def matches_from_arrays(arrays, mutual=False):
    matches = []
    for key in arrays.files:
        if not key.endswith('_weights'):
            continue
        prefix = key[:-len('weights')]
        _, i, j, _ = prefix.split('_')
        i, j = int(i), int(j)
        a, b = arrays[prefix+'source_indices'], arrays[prefix+'target_indices']
        weights = arrays[key].copy()
        if mutual:
            # Keep only mutual maxima; preserve original absolute mass.
            rows = np.arange(len(weights)); cols = weights.argmax(1)
            keep = np.zeros_like(weights, dtype=bool)
            keep[rows, cols] = weights.argmax(0)[cols] == rows
            weights *= keep
        matches.append(dict(i=i, j=j, source_xyz=arrays['points'][i, a],
            target_xyz=arrays['points'][j, b], weights=weights,
            source_matchability=(1-arrays[prefix+'source_prob'][:, -1])*arrays['fracture_probability'][0, i, a],
            target_matchability=(1-arrays[prefix+'target_prob'][:, -1])*arrays['fracture_probability'][0, j, b]))
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = dict(diagnostic_only=True, threshold=.01, variants={}, replay=[],
                  replay_scope='Selected saved tensors, samples_poses/4101; not gate acceptance')
    for experiment in sorted((args.results/'experiments').iterdir()):
        root = experiment/'seed42'
        summary = {}
        for phase in ('gate/overfit', 'gate/training', 'contact_validation'):
            rows = read_rows(root/phase/'examples.jsonl')
            summary[phase] = {mode: counts([r for r in rows if r['mode'] == mode])
                              for mode in sorted({r['mode'] for r in rows})}
            summary[phase]['by_pieces'] = {str(n): counts([r for r in rows if r['pieces'] == n]) for n in (2, 3)}
        summary['histories'] = {}
        for phase in ('overfit', 'contacts'):
            summary['histories'][phase] = [dict(update=r['update'], **{k:r['validation'][k] for k in
                ('assembly_success', 'matching_top1_recall', 'retrieval_top5', 'matching_mass', 'matching_localization')})
                for r in read_rows(root/phase/'history.jsonl')]
        fixed = [r for r in read_rows(root/'gate/overfit/examples.jsonl') if r['mode'] == 'unchanged']
        summary['fixed_failures'] = [dict(pattern_id=r['pattern_id'], status=r['status'],
            geometric_success=r['success'], max_part_chamfer=max(r['per_part_chamfer']) if r['per_part_chamfer'] else None,
            rotation_deg=r['rotation_deg'], candidate_coverage=r['candidate_coverage'])
            for r in fixed if not r['success'] or r['status'] != 'ok']
        output['variants'][experiment.name] = summary

    root = args.results/'experiments/revised__resampled_contrastive/seed42'
    cfg = json.loads((root/'overfit/config.resolved.json').read_text())
    for phase in ('gate/overfit', 'gate/training', 'contact_validation'):
        originals = {r['pattern_id']:r for r in read_rows(root/phase/'examples.jsonl')
                     if r['mode'] == 'samples_poses' and r['seed'] == 4101}
        for path in sorted((root/phase/'tensors').glob('*.npz')):
            with np.load(path, allow_pickle=False) as arrays:
                original = originals[path.stem]
                for variant in ('baseline', 'no_refinement', 'huber_005', 'mutual'):
                    options = copy.deepcopy(cfg)
                    if variant == 'no_refinement': options['solver']['refinement_iterations'] = 0
                    if variant == 'huber_005': options['solver']['huber_delta'] = .005
                    cache = build_candidates_from_matches(matches_from_arrays(arrays, variant == 'mutual'),
                        int(arrays['fragment_mask'].sum()), original['diagnostics']['anchor_index'], cfg=options)
                    result = solve_candidates(cache, options)
                    measure = metrics((result['rotations'], result['translations']), arrays)
                    candidate_metrics = [metrics(h['poses'], arrays) for h in cache.hypotheses]
                    row = dict(phase=phase, pattern_id=path.stem, variant=variant, **measure,
                        status=result['status'], confidence=result['confidence'],
                        contact_rms=result['diagnostics'].get('contact_rms'),
                        candidates=len(cache.hypotheses), coverage=any(r['success'] for r in candidate_metrics),
                        original_success=original['success'], original_status=original['status'],
                        original_confidence=original['confidence'])
                    output['replay'].append(row)
            print(f'replayed {phase}/{path.stem}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
