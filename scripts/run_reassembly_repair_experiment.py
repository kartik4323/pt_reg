"""Run one configured contact experiment using the existing preflight and gates.

Unlike the factorial comparisons phase, this runs only the requested geometry
and view-supervision variant. Each changed configuration needs a fresh output.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reassembly.repair.config import load_config
from reassembly.repair.data import supplement_queries
from reassembly.repair.workflow import _contact_experiment
from reassembly.resources import make_guard


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--managed-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--field-diagnostic-report', type=Path, required=True)
    p.add_argument('--query-cache', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    cfg = load_config(args.config)
    root, output = args.managed_root.expanduser().resolve(), args.output.expanduser().resolve()
    prepared = args.manifest.resolve().parent
    if root not in output.parents or output == prepared or prepared in output.parents or output in prepared.parents:
        raise ValueError('Use a dedicated output under managed-root, outside prepared data')
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError('Use a fresh experiment output, or --resume for an unchanged configuration')
    cfg['resources']['managed_root'] = str(root)
    guard = make_guard(cfg, output, prepared)
    guard.check()
    cache = args.query_cache
    if cache is None:
        cache = output/'queries/query_cache.json'
        if not cache.exists():
            supplement_queries(args.manifest, cache.parent, guard)
    result = _contact_experiment(cfg, args.manifest, output, args.device,
        args.field_diagnostic_report, cache, args.resume, guard)
    print(json.dumps({k:result[k] for k in ('variant', 'training_seed', 'gate_passed', 'contact_success', 'gate_report')}, indent=2))
    return 0 if result['gate_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
