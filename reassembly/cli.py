"""One explicit, bounded workflow. Run ``python -m reassembly --help``."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .precision import NUMERICS_VERSION
from .resources import jsonable, make_guard, write_json


def profile_signature(cfg: dict) -> str:
    """Bind allocation results to every input/model/solver/loss size and precision."""
    relevant = {key: cfg[key] for key in ('version', 'data', 'model', 'loss', 'solver')}
    relevant['numerics_version'] = NUMERICS_VERSION
    relevant['train'] = {key: cfg['train'][key] for key in ('batch_size', 'grad_accum_steps', 'amp')}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def require_preflight(path: Path | None, cfg: dict, manifest: Path, device: torch.device):
    if path is None:
        raise ValueError('Training requires --preflight-report from this configuration and dataset')
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    fingerprint = json.loads(manifest.read_text(encoding='utf-8'))['fingerprint']
    if report.get('kind') != 'preflight' or report.get('profile_signature') != profile_signature(cfg):
        raise ValueError('Preflight does not match this configuration; use its resolved config or reprofile')
    if report.get('dataset_fingerprint') != fingerprint or not report.get('correctness_passed'):
        raise ValueError('Prepared-data preflight and passing geometry/solver checks are required')
    expected = 'passed' if device.type == 'cuda' else 'cpu_verified_cuda_unmeasured'
    if report.get('status') != expected:
        raise ValueError(f'Preflight must have status {expected} for this training device')
    if device.type == 'cuda':
        attempt = report['attempts'][-1]
        if attempt.get('gpu_name') != torch.cuda.get_device_name(device):
            raise ValueError('Preflight was measured on a different GPU model; reprofile this host')
        if attempt['max_reserved_bytes'] >= cfg['resources']['max_vram_gib'] * 1024**3:
            raise ValueError('Preflight does not satisfy the strict VRAM limit')


def require_overfit(path: Path | None, manifest: Path, cfg: dict | None = None):
    if path is None:
        raise ValueError('Start with the fixed 16-pattern overfit; supply its passing --overfit-report before a fresh held-out pilot')
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    fingerprint = json.loads(manifest.read_text(encoding='utf-8'))['fingerprint']
    if (report.get('kind') != 'evaluation' or not report.get('fixed_fit_passed')
            or report.get('dataset_fingerprint') != fingerprint or report.get('purpose') != 'overfit'
            or report.get('condition') != 'predicted'):
        raise ValueError('A matching 16-pattern assembly overfit has not passed; inspect its failure diagnostics')
    if cfg is not None and any(report.get('config', {}).get(key) != cfg[key] for key in ('data', 'model', 'loss', 'solver')):
        raise ValueError('Overfit report configuration differs from this experiment; repeat the fixed check')


def parser():
    root = argparse.ArgumentParser(description='Fresh coarse-scaffold rigid assembly v2 (2–3 complete fragments)')
    sub = root.add_subparsers(dest='command', required=True)
    for name in ('acquire', 'prepare', 'preflight', 'train', 'evaluate', 'infer', 'report'):
        p = sub.add_parser(name)
        p.add_argument('--config', type=Path, help='v2 YAML or resolved JSON config')
        p.add_argument('--managed-root', type=Path, help='Override managed root (default ~/reassembly_v2)')
        if name in ('preflight', 'train', 'evaluate', 'infer'):
            p.add_argument('--device', default='cuda', help='cuda, cuda:0 or cpu; no silent fallback')
        if name in ('preflight', 'train', 'evaluate'):
            p.add_argument('--manifest', type=Path, required=name != 'preflight')
        if name != 'train':
            p.add_argument('--output', type=Path)
        if name == 'acquire':
            p.add_argument('--dry-run', action='store_true', help='Access/size check only')
        elif name == 'prepare':
            p.add_argument('--source', type=Path, required=True, help='Bottle category ZIP or mesh directory')
        elif name == 'train':
            p.add_argument('--stage', type=int, choices=(1, 2, 3), required=True)
            p.add_argument('--run-dir', type=Path, required=True)
            p.add_argument('--initialize-from', type=Path)
            p.add_argument('--resume', type=Path)
            p.add_argument('--overfit', action='store_true')
            p.add_argument('--condition', choices=('predicted', 'contact_only'), default='predicted')
            p.add_argument('--preflight-report', type=Path, required=True)
            p.add_argument('--overfit-report', type=Path)
        elif name == 'evaluate':
            p.add_argument('--checkpoint', type=Path, required=True)
            p.add_argument('--condition', choices=('contact_only', 'predicted', 'gt', 'perturbed'), default='predicted')
            p.add_argument('--split', choices=('val', 'test', 'cut_holdout'), default='test')
            p.add_argument('--limit', type=int)
            p.add_argument('--overfit', action='store_true')
        elif name == 'infer':
            p.add_argument('--checkpoint', type=Path, required=True)
            p.add_argument('--fragment', type=Path, action='append', required=True, help='Repeat 2–3 times; each file is an Nx3 .npy array')
            p.add_argument('--condition', choices=('predicted', 'contact_only'), default='predicted')
            p.add_argument('--save-scaffold', action='store_true')
        elif name == 'report':
            p.add_argument('--input', type=Path, action='append', required=True, help='Repeat for preparation/preflight/training/evaluation JSONs')
    return root


def dispatch(args):
    cfg = load_config(args.config)
    if args.managed_root:
        cfg['resources']['managed_root'] = str(args.managed_root.expanduser().resolve())
    managed = Path(cfg['resources']['managed_root']).expanduser().resolve()
    cfg['resources']['managed_root'] = str(managed)
    device = torch.device(args.device) if hasattr(args, 'device') else None
    if device and device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable on this host. Use --device cpu for correctness checks; measure the pilot on the A5000.')
    if args.command == 'train':
        output = args.run_dir.resolve()
    else:
        default = {'acquire': managed/'sources', 'prepare': managed/'prepared',
                   'preflight': managed/'preflight', 'evaluate': managed/'evaluation',
                   'infer': managed/'inference', 'report': managed/'pilot_report.json'}[args.command]
        output = (args.output or default).expanduser().resolve()
    guard = make_guard(cfg, output if args.command != 'report' else output.parent)
    storage = guard.check()
    if args.command == 'acquire':
        from .acquire import acquire_bottles
        report = acquire_bottles(output, cfg, guard, args.dry_run)
        report['storage'] = guard.check()
        write_json(output/'acquisition.json', report, guard)
        return report, 0 if report['status'] in ('available', 'downloaded', 'already_present') else 2
    if args.command == 'prepare':
        from .prepare import prepare_dataset
        started = time.perf_counter()
        report = prepare_dataset(args.source.resolve(), output, cfg, guard)
        report.update(elapsed_seconds=time.perf_counter()-started, storage=guard.check())
        report['source_yield'] = report['accepted_sources'] / max(1, report['candidates'])
        write_json(output/'preparation_report.json', report, guard)
        return report, 0 if report['learning_ready'] else 2
    if args.command == 'preflight':
        from .training import collate_samples, preflight
        from .validation import run_correctness_checks
        correctness = run_correctness_checks()
        if not correctness['passed']:
            raise RuntimeError(f'Geometry/solver correctness checks failed: {correctness}')
        sample_batch, fingerprint = None, None
        if args.manifest:
            from .data import FractureDataset, verify_manifest
            integrity = verify_manifest(args.manifest)
            dataset = FractureDataset(args.manifest, 'all', cfg, fixed=True)
            indices = [i for i, record in enumerate(dataset.records) if record['pieces'] == 3]
            if not indices:
                raise ValueError('Prepare at least one valid 3-piece pattern to profile the largest supported case')
            fingerprint = dataset.fingerprint
            sample_batch = lambda size: collate_samples([dataset[indices[i % len(indices)]] for i in range(size)], device)
        report, resolved = preflight(cfg, device, sample_batch=sample_batch)
        report.update(kind='preflight', schema_version=2, correctness=correctness,
                      correctness_passed=correctness['passed'], dataset_fingerprint=fingerprint,
                      input_kind='prepared_patterns' if args.manifest else 'synthetic_allocation_fixture',
                      profile_signature=profile_signature(resolved), storage=storage)
        report['dataset_integrity'] = integrity if args.manifest else None
        write_json(output/'config.resolved.json', resolved, guard)
        write_json(output/'preflight.json', report, guard)
        return report, 0 if report['status'] != 'failed' else 2
    if args.command == 'train':
        from .training import train_stage
        require_preflight(args.preflight_report, cfg, args.manifest, device)
        if not args.overfit and args.stage == 1 and not args.resume:
            require_overfit(args.overfit_report, args.manifest, cfg)
        report = train_stage(cfg, args.manifest, output, args.stage, device,
                             initialize_from=args.initialize_from, resume=args.resume,
                             overfit=args.overfit, condition=args.condition, guard=guard)
        report['storage'] = guard.check()
        rate = report['seconds'] / max(1, report.get('updates_this_invocation', report['updates']))
        report['projected_10000_update_hours'] = rate * 10000 / 3600
        report['projection_note'] = 'Linear extrapolation including periodic validation; no longer run was launched.'
        write_json(output/'training_report.json', report, guard)
        return report, 0
    if args.command == 'evaluate':
        from .evaluation import evaluate
        if args.limit is not None and args.limit <= 0:
            raise ValueError('--limit must be positive')
        report = evaluate(args.checkpoint, args.manifest, output, cfg, device, args.split,
                          args.condition, args.limit, args.overfit, guard)
        return report, 0
    if args.command == 'infer':
        from .evaluation import infer_fragments
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Select a fresh inference output directory; existing artifacts are preserved')
        if len(args.fragment) not in (2, 3):
            raise ValueError('Supply exactly 2 or 3 complete XYZ fragments')
        fragments = [np.load(path, allow_pickle=False) for path in args.fragment]
        report = infer_fragments(fragments, args.checkpoint, cfg, device, args.condition)
        aligned, field = report.pop('aligned_fragments'), report.pop('scaffold', None)
        output.mkdir(parents=True, exist_ok=True)
        if aligned is not None:
            arrays = {f'fragment_{i}': points for i, points in enumerate(aligned)}
            arrays.update(rotations=report['rotations'], translations=report['translations'], transforms=report['transforms'])
            guard.check(additional_bytes=sum(value.nbytes for value in arrays.values()))
            np.savez_compressed(output/'assembly.npz', **arrays)
            report['aligned_fragments_file'] = str(output/'assembly.npz')
        else:
            report['aligned_fragments_file'] = None
        if args.save_scaffold and field:
            guard.check(additional_bytes=sum(value.nbytes for value in field.values() if isinstance(value, np.ndarray)))
            np.savez_compressed(output/'scaffold.npz', **field)
            report['scaffold_file'] = str(output/'scaffold.npz')
            from .visualization import plot_scaffold
            guard.check(additional_bytes=1024**2)
            plot_scaffold(field, output/'scaffold.png')
            report['scaffold_visualization'] = str(output/'scaffold.png')
        write_json(output/'result.json', report, guard)
        return report, 0 if report['status'] == 'ok' else 2
    if args.command == 'report':
        from .evaluation import pilot_report
        report = pilot_report(args.input, output, guard)
        return report, 0
    raise ValueError('Unsupported command')


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        report, status = dispatch(args)
        # Full sample arrays/row details are in bounded artifacts, not the console.
        console = {k: v for k, v in report.items() if k not in ('samples', 'reports', 'config', 'calibration_bins')}
        print(json.dumps(jsonable(console), indent=2, allow_nan=False))
        return status
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, OSError) as exc:
        print(f'reassembly: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
