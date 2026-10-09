"""Compare baseline and orientation on saved contact tensors, without inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
from analyze_reassembly_repair import read_rows, matches_from_arrays, metrics
from reassembly.repair.assembly import build_candidates_from_matches, solve_candidates
from reassembly.repair.orientation import attach_normals, validate_overrides
from reassembly.repair.provenance import code_inventory


def replay(experiment, overrides):
    root = Path(experiment)
    config_path = root/'contacts/config.resolved.json'
    if not config_path.exists(): config_path = root/'overfit/config.resolved.json'
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    output = dict(diagnostic_only=True, threshold=.01, replay_scope='Selected tensors at samples_poses/4101; not gate acceptance',
                  solver_overrides=overrides, code=code_inventory(),
                  input_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), rows=[])
    for phase in ('gate/overfit', 'gate/training', 'contact_validation'):
        row_path = root/phase/'examples.jsonl'
        if not row_path.exists(): continue
        originals = {r['pattern_id']:r for r in read_rows(row_path)
                     if r['mode']=='samples_poses' and r['seed']==4101 and r['condition']=='contact_only'}
        for path in sorted((root/phase/'tensors').glob('*.npz')):
            original = originals[path.stem]
            with np.load(path, allow_pickle=False) as arrays:
                for label in ('baseline', 'orientation'):
                    options = copy.deepcopy(cfg)
                    if label=='orientation': options['solver'].update(overrides)
                    else: options['solver'].pop('contact_orientation', None)
                    count = int(arrays['fragment_mask'].sum())
                    matches = attach_normals(matches_from_arrays(arrays), arrays['points'][:count], options)
                    cache = build_candidates_from_matches(matches, count,
                        original['diagnostics']['anchor_index'], cfg=options)
                    result = solve_candidates(cache, options)
                    measurement = metrics((result['rotations'], result['translations']), arrays)
                    coverage = any(metrics(h['poses'], arrays)['success'] for h in cache.hypotheses)
                    output['rows'].append(dict(phase=phase, pattern_id=path.stem, pieces=count, source_id=original['source_id'],
                        variant=label, **measurement, status=result['status'], confidence=result['confidence'],
                        candidate_covered=coverage, diagnostics=result['diagnostics'],
                        tensor_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        original_success=original['success'], original_status=original['status']))
            print('replayed',phase,path.stem,flush=True)
    if not output['rows']: raise ValueError('No saved samples_poses/4101 correspondence tensors found')
    output['summaries'] = {}
    for phase in sorted({r['phase'] for r in output['rows']}):
        output['summaries'][phase] = {}
        for label in ('baseline','orientation'):
            rows = [r for r in output['rows'] if r['phase']==phase and r['variant']==label]
            output['summaries'][phase][label] = dict(count=len(rows), geometric_successes=sum(r['success'] for r in rows),
                accepted_successes=sum(r['success'] and r['status']=='ok' for r in rows),
                candidate_coverage=sum(r['candidate_covered'] for r in rows))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--solver-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    document = yaml.safe_load(args.solver_config.read_text(encoding='utf-8'))
    if not isinstance(document, dict) or set(document) != {'solver'}:
        raise ValueError('Solver config must contain only a solver mapping')
    overrides = validate_overrides(document['solver'])
    if args.output.exists(): raise FileExistsError('Choose a fresh replay output')
    result = replay(args.experiment, overrides)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result['summaries'],indent=2))


if __name__ == '__main__': main()
