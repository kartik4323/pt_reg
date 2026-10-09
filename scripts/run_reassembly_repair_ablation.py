"""Evaluate both existing contact checkpoints and validation without retraining."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from reassembly.repair.checkpoints import load_checkpoint
from reassembly.repair.evaluation import contact_gate, evaluate
from reassembly.repair.orientation import validate_overrides
from reassembly.resources import make_guard


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--managed-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--solver-config', type=Path)
    p.add_argument('--query-cache', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    source, output = args.experiment.expanduser().resolve(), args.output.expanduser().resolve()
    root, prepared = args.managed_root.expanduser().resolve(), args.manifest.resolve().parent
    if (root not in output.parents or output == prepared or prepared in output.parents or output in prepared.parents
            or source == output or source in output.parents or output in source.parents):
        raise ValueError('Choose a dedicated output under managed-root, separate from prepared data and the original experiment')
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError('Use a fresh evaluation output or --resume with unchanged inputs')
    checkpoint, overfit_checkpoint = source/'contacts/best.pt', source/'overfit/best.pt'
    cfg = copy.deepcopy(load_checkpoint(checkpoint, stage=1, purpose='experiment')['cfg'])
    overrides = None
    if args.solver_config:
        document = yaml.safe_load(args.solver_config.read_text(encoding='utf-8'))
        if not isinstance(document, dict) or set(document) != {'solver'}:
            raise ValueError('Solver config must contain only a solver mapping')
        overrides = validate_overrides(document['solver'])
        cfg['solver'].update(overrides)
    cfg['resources']['managed_root'] = str(root)
    guard = make_guard(cfg, output, prepared)
    guard.check()
    query_cache = args.query_cache
    if query_cache is None and (source/'queries/query_cache.json').exists():
        query_cache = source/'queries/query_cache.json'
    gate = contact_gate(checkpoint, overfit_checkpoint, args.manifest, output/'gate', args.device,
        guard, resume=args.resume, solver_overrides=overrides)
    # Gate failure is still evidence: always complete the validation comparison.
    validation = evaluate(checkpoint, args.manifest, output/'contact_validation', args.device,
        conditions=('contact_only',), query_cache=query_cache, guard=guard,
        resume=args.resume, solver_overrides=overrides)
    summaries = list(validation['summaries'].values())
    print(json.dumps(dict(gate_passed=gate['passed'], diagnostic_only=overrides is not None,
        validation_success=sum(s['successes'] for s in summaries)/sum(s['count'] for s in summaries),
        gate_report=str(output/'gate/contact_gate.json')), indent=2))
    return 0 if gate['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
