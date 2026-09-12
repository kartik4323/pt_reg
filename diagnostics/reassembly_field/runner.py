"""Inventory, incremental jobs, resource limits and compact follow-up reporting."""
from collections import defaultdict
import json
from pathlib import Path
import tarfile
import time

import numpy as np
import torch

from diagnostics.reassembly_v2.contracts import matching_contract, sample_contract
from diagnostics.reassembly_v2.runtime import Limits, job_id, read, sha256, write
from diagnostics.reassembly_v2.variants import check_adapter, choose_subset
from diagnostics.reassembly_v2.worker import inventory as original_inventory
from reassembly.checkpoints import load_checkpoint
from reassembly.data import FractureDataset, make_oracle_field, verify_manifest
from reassembly.model import ReassemblyModel
from reassembly.resources import seed_all, tree_bytes
from reassembly.training import collate_samples
from reassembly.repair.fields import ContinuousGTField, ContinuousNeuralField
from . import probes


CHECKPOINT = 'pilot/predicted/s3/best.pt'
FINGERPRINT = 'b7f2b84a20964276c894300a7ece4d8f97f232903bace0f14131535594462043'
REPO = Path(__file__).resolve().parents[2]


def source_documents(path):
    """Read only bounded JSON members; never extract user-provided archives."""
    path = Path(path)
    names = ('inventory.json', 'experiments.json', 'summary.json')
    if path.is_dir():
        docs = {name: read(path / name) for name in names}
        hashes = {name: {'path': str((path / name).resolve()), 'sha256': sha256(path / name)} for name in names}
    else:
        docs = {}
        with tarfile.open(path, 'r:*') as archive:
            for name in names:
                members = [m for m in archive.getmembers() if m.isfile() and
                           (m.name == name or m.name.endswith('/' + name))]
                if len(members) != 1 or members[0].size > 16 * 1024 ** 2:
                    raise ValueError(f'Original diagnostic bundle must contain one bounded {name}')
                docs[name] = json.load(archive.extractfile(members[0]))
        hashes = {'source_bundle': {'path': str(path.resolve()), 'sha256': sha256(path)}}
    if docs['summary.json'].get('status') != 'complete':
        raise ValueError('Source diagnostics must be a completed original run')
    if docs['inventory.json']['dataset_integrity']['dataset_fingerprint'] != FINGERPRINT:
        raise ValueError('Source diagnostics fingerprint differs from the recorded VM dataset')
    return docs, hashes


def selected_records(dataset, source=None):
    if source is None:
        indices = choose_subset(dataset.records)
        return indices, {'selection': 'Original choose_subset, independent of model performance'}, {}
    docs, hashes = source_documents(source)
    records = docs['experiments.json']['validation_subset']
    requested = [r['pattern_id'] for r in records]
    positions = {r['pattern_id']: i for i, r in enumerate(dataset.records)}
    if len(requested) != 12 or len(set(requested)) != 12 or any(p not in positions for p in requested):
        raise ValueError('Original diagnostic subset must contain 12 distinct patterns from this VM validation split')
    selected = [positions[p] for p in requested]
    for actual, recorded in zip((dataset.records[i] for i in selected), records):
        for key in ('source_id', 'pattern_id', 'split', 'band', 'cut_family'):
            if actual.get(key) != recorded.get(key):
                raise ValueError(f'Original diagnostic subset metadata changed: {key}')
    return selected, {'selection': 'Exact identities and order from original diagnostics',
                      'source': str(source), 'original_inventory': docs['inventory.json']}, hashes


def find_source(root, explicit=None):
    if explicit:
        return Path(explicit)
    candidates = sorted((root / 'diagnostics').glob('*/experiments.json'),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for file in candidates:
        summary = file.parent / 'summary.json'
        if summary.is_file() and read(summary).get('status') == 'complete':
            return file.parent
    return None


def validate_resume(output):
    inventory_path = Path(output) / 'inventory.json'
    if not inventory_path.exists():
        raise RuntimeError('Interrupted inventory cannot be resumed; choose a fresh output')
    inv = read(inventory_path)
    for section in ('files', 'checkpoints', 'production_files', 'diagnostic_files', 'followup_files', 'source_files'):
        for name, entry in inv.get(section, {}).items():
            path = Path(entry['path'])
            if not path.is_file() or sha256(path) != entry['sha256']:
                raise RuntimeError(f'Resume input changed: {section}/{name}')
    manifest = inv.get('prepared_manifest_path')
    if manifest and verify_manifest(manifest)['dataset_fingerprint'] != inv['dataset_integrity']['dataset_fingerprint']:
        raise RuntimeError('Prepared data changed before resume')


def build_jobs(dataset, indices):
    jobs = []
    for index in indices:
        record = dataset.records[index]
        base = {'index': index, 'pattern_id': record['pattern_id'], 'source_id': record['source_id'],
                'band': record['band'], 'checkpoint': CHECKPOINT}
        definitions = [{'kind': 'continuous', 'field': name} for name in ('predicted', 'gt')]
        definitions += [{'kind': 'conditioning'}, {'kind': 'contact_only_control'}]
        definitions += [{'kind': 'grid', 'field': name, 'resolution': resolution,
                         'conditional': resolution == 256} for resolution in (32, 64, 128, 256)
                        for name in ('predicted', 'gt')]
        for definition in definitions:
            job = {**base, **definition}
            jobs.append(dict(job, id=job_id(job)))
    return jobs


def conditional_256_needed(previous):
    errors = {name: value['field_metrics']['observed_original_surface']['interpolation_mae']
              for name, value in previous['variants'].items()}
    return any(value is not None and value > .001 for value in errors.values()), errors


class PeriodicCheck:
    def __init__(self, limits, device):
        self.limits, self.device, self.previous = limits, device, 0.

    def __call__(self, force=False):
        if self.limits.deadline is not None and time.time() >= self.limits.deadline:
            raise TimeoutError('Optional field diagnostic deadline reached')
        if self.device.type == 'cuda' and torch.cuda.max_memory_reserved(self.device) >= 20 * 1024 ** 3:
            raise RuntimeError('Peak reserved GPU memory reached 20 GiB')
        if force or time.monotonic() - self.previous >= 30:
            self.limits.check()
            self.previous = time.monotonic()


def save_visual(output, job, variants, sample):
    """Bounded field slices and original-surface points; no full grids persisted."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = probes.original.array(sample['target_points'])
    # A view through the object's target centroid, bounded around target extent.
    lo, hi = p.min(0) - .05, p.max(0) + .05
    axis = [np.linspace(lo[k], hi[k], 96) for k in range(2)]
    xx, yy = np.meshgrid(*axis, indexing='ij')
    query = np.stack((xx.ravel(), yy.ravel(), np.full(xx.size, p[:, 2].mean())), -1)
    fig, axes = plt.subplots(1, len(variants) + 1, figsize=(4 * (len(variants) + 1), 3.5))
    for ax, (name, field) in zip(axes, variants.items()):
        distance = field.sample(query)[0].reshape(xx.shape)
        ax.imshow(distance.T, origin='lower', extent=[lo[0], hi[0], lo[1], hi[1]],
                  cmap='coolwarm', vmin=-.1, vmax=.1)
        if distance.min() < 0 < distance.max():
            ax.contour(xx, yy, distance, levels=[0], colors='black', linewidths=.7)
        ax.set_title(name)
    axes[-1].scatter(p[:, 0], p[:, 1], s=1, alpha=.4)
    axes[-1].set_aspect('equal')
    axes[-1].set_title('Intact target XY')
    fig.suptitle(f"{job['pattern_id']} {job.get('field', '')} {job.get('resolution', 'continuous')}", fontsize=8)
    fig.tight_layout()
    path = output / 'visuals' / f"{job['id']}.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def remove_scratch(output, job_id_value):
    directory = (output / 'scratch' / job_id_value).resolve()
    expected = (output / 'scratch').resolve()
    if expected not in directory.parents:
        raise RuntimeError('Scratch cleanup escaped the diagnostic directory')
    for name in ('distance.npy', 'sigma.npy', 'cursor.json', 'cursor.json.tmp'):
        file = directory / name
        if file.exists():
            file.unlink()
    if directory.exists() and not any(directory.iterdir()):
        directory.rmdir()


def run(root, output, device, source=None, deadline=None, resume=False):
    root, output = Path(root), Path(output)
    work = root / 'bottles498'
    limits = Limits(root, output, deadline)
    check = PeriodicCheck(limits, device)
    torch.set_num_threads(4)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA unavailable; no CPU substitution for VM diagnostics')
        memory = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., (20 * 1024 ** 3 - 64 * 1024 ** 2) / memory), device)
        torch.cuda.reset_peak_memory_stats(device)
    if resume:
        validate_resume(output)
    check(force=True)
    write(output / 'progress.json', {'phase': 'artifact verification'})
    inv = original_inventory(work, output, limits, device)
    inv['prepared_manifest_path'] = str(work / 'prepared/manifest.json')
    files = list(Path(__file__).parent.glob('*.py')) + list((REPO / 'reassembly/repair').glob('*.py'))
    inv['followup_files'] = {p.relative_to(REPO).as_posix(): {'path': str(p), 'sha256': sha256(p)} for p in files}
    write(output / 'inventory.json', inv)
    cfg = read(work / 'eval/test/predicted/evaluation.json')['config']
    dataset = FractureDataset(work / 'prepared/manifest.json', 'val', cfg, fixed=True)
    if len(dataset) != 48 or dataset.fingerprint != FINGERPRINT or cfg['data']['points_per_fragment'] != 1024:
        raise RuntimeError('Expected the original 48 validation patterns, VM fingerprint and 1,024-point inputs')
    if resume and (output / 'experiments.json').exists():
        previous = read(output / 'experiments.json')
        source = previous.get('source_path')
    else:
        source = find_source(root, source)
    indices, selection, source_hashes = selected_records(dataset, source)
    if source:
        old = selection.pop('original_inventory')
        if old['checkpoints'][CHECKPOINT]['sha256'] != inv['checkpoints'][CHECKPOINT]['sha256']:
            raise RuntimeError('Selected checkpoint differs from the completed original diagnostic run')
    inv['source_files'] = source_hashes
    write(output / 'inventory.json', inv)
    jobs = build_jobs(dataset, indices)
    experiments = {'schema_version': 1, 'kind': 'field_followup_diagnosis', 'jobs': jobs,
                   'validation_subset': [dataset.records[i] for i in indices], 'selection': selection,
                   'source_path': str(source) if source else None, 'checkpoint': CHECKPOINT,
                   'dataset_fingerprint': dataset.fingerprint, 'batch_size': 1, 'points_per_fragment': 1024,
                   'field_extent': 2.25, 'field_chunk': 2048, 'seed': probes.SEED,
                   'perturbations': [list(p) for p in probes.PERTURBATIONS],
                   'conditional_256_rule': 'For the same example and field source, run 256 only if either 128 clamp variant has observed-original-surface interpolation MAE > 0.001 against the continuous field.',
                   'refinement': 'Original v2 five-step damped Gauss-Newton and unchanged production thresholds; fixed starts and oracle contacts across fields.',
                   'precision': 'FP32 neural queries, FP64 numerical refinement', 'optimizer_updates': 0,
                   'production_acceptance': False}
    if resume and (output / 'experiments.json').exists() and read(output / 'experiments.json') != experiments:
        raise RuntimeError('Resume experiment manifest changed')
    write(output / 'experiments.json', experiments)
    state = load_checkpoint(work / CHECKPOINT, dataset_fingerprint=dataset.fingerprint, stage=3, purpose='pilot')
    model = ReassemblyModel(state['cfg']).to(device)
    model.load_state_dict(state['model'])
    model.eval().requires_grad_(False)
    seed_all(cfg['train']['seed'])
    contract = read(output / 'adapter_checks.json') if (output / 'adapter_checks.json').exists() else {}
    current_index, sample, batch, encoded, fields, queries, regions, contacts, contact_status = (None,) * 9
    for position, job in enumerate(jobs):
        check(force=True)
        path = output / 'results' / f"{job['id']}.json"
        if path.exists() and not read(path).get('error'):
            remove_scratch(output, job['id'])
            continue
        write(output / 'progress.json', {'phase': job['kind'], 'job_id': job['id'],
                                         'completed_jobs': position, 'planned_jobs': len(jobs)})
        started = time.monotonic()
        try:
            if current_index != job['index']:
                current_index = job['index']
                fields = None
                sample = dataset[current_index]
                batch = collate_samples([sample], device)
                with torch.no_grad():
                    encoded = model.encode(batch['points'], batch['fragment_mask'], batch['anchor_index'])
                fields = {'predicted': ContinuousNeuralField(model, encoded, chunk_size=2048),
                          'gt': ContinuousGTField.from_sample(sample, truncation=model.field.truncation, chunk_size=512)}
                queries, regions = probes.probe_queries(sample)
                contacts, contact_status = probes.oracle_contacts(sample, cfg)
                checks = {'sample': sample_contract(sample), 'rebuild': check_adapter(dataset, current_index),
                          'matching': matching_contract(model, sample, cfg, device)}
                oracle = make_oracle_field(sample)
                np.testing.assert_allclose(fields['gt'].raw_distance(queries[:64]), oracle(queries[:64]), atol=1e-5, rtol=1e-4)
                checks['continuous_gt_matches_original_callable'] = True
                contract[sample['pattern_id']] = checks
                write(output / 'adapter_checks.json', contract)
            result = {'job': job, 'pattern_id': sample['pattern_id'], 'source_id': sample['source_id'],
                      'band': sample['band'], 'pieces': int(sample['fragment_mask'].sum()),
                      'oracle_contact_support': contact_status, 'production_acceptance': False}
            if job['kind'] == 'conditioning':
                result['conditioning'], arrays = probes.conditioning(model, encoded, fields['predicted'], fields['gt'])
                tensor_dir = output / 'tensors'
                tensor_dir.mkdir(exist_ok=True)
                np.savez_compressed(tensor_dir / f"{job['id']}.npz", **arrays)
            elif job['kind'] == 'contact_only_control':
                result['refinement'] = probes.refine_probes(None, sample, encoded, contacts, contact_status, cfg, check)
            else:
                direct = fields[job['field']]
                if job['kind'] == 'grid' and job['resolution'] == 256:
                    earlier = next(j for j in jobs if j['index'] == job['index'] and j.get('field') == job['field'] and j.get('resolution') == 128)
                    previous = read(output / 'results' / f"{earlier['id']}.json")
                    needed, errors = conditional_256_needed(previous)
                    result['conditional_128_exterior_mae'] = errors
                    if not needed:
                        result['skipped'] = '128 exterior interpolation MAE <= 0.001 for both clamp variants; predefined conditional rule'
                        result['seconds'] = time.monotonic() - started
                        write(path, result)
                        continue
                if job['kind'] == 'grid':
                    def progress(done, total):
                        write(output / 'progress.json', {'phase': 'fresh grid construction', 'job_id': job['id'],
                                                        'completed_nodes': done, 'total_nodes': total})
                    raw, sigma, bounds, result['grid_profile'] = probes.fresh_grid_nodes(
                        direct, job['resolution'], output / 'scratch' / job['id'],
                        inv['checkpoints'][CHECKPOINT]['sha256'] + sample['pattern_id'] + job['field'], check, progress)
                    variants = probes.grid_variants(raw, sigma, bounds, direct.truncation)
                else:
                    variants = {'continuous': direct}
                result['variants'] = {}
                for name, field in variants.items():
                    check()
                    metrics, arrays = probes.field_metrics(field, direct, fields['gt'], queries, regions)
                    result['variants'][name] = {'field_metrics': metrics,
                        'refinement': probes.refine_probes(field, sample, encoded, contacts, contact_status, cfg, check)}
                    tensor_dir = output / 'tensors'
                    tensor_dir.mkdir(exist_ok=True)
                    np.savez_compressed(tensor_dir / f"{job['id']}_{name}.npz", **arrays)
                save_visual(output, job, variants, sample)
                if job['kind'] == 'grid':
                    # Close mmap-backed views before Windows scratch cleanup.
                    variants = None
                    del raw, sigma
            result['seconds'] = time.monotonic() - started
            result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None
            check()
            write(path, result)
            remove_scratch(output, job['id'])
        except Exception as exc:
            write(path, {'job': job, 'error': f'{type(exc).__name__}: {exc}'})
            raise
        print(f"completed {position + 1}/{len(jobs)} {job['kind']} {job['id']}", flush=True)
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Model parameter gradients unexpectedly accumulated')


def finalize(output, reason=None):
    output = Path(output)
    manifest = read(output / 'experiments.json') if (output / 'experiments.json').exists() else {'jobs': []}
    inv = read(output / 'inventory.json') if (output / 'inventory.json').exists() else {}
    jobs, rows, unrun = manifest['jobs'], [], []
    for job in jobs:
        path = output / 'results' / f"{job['id']}.json"
        if path.exists():
            rows.append(read(path))
        else:
            unrun.append(job)
    changed, verified = [], {}
    for section in ('files', 'checkpoints', 'production_files', 'diagnostic_files', 'followup_files', 'source_files'):
        for name, entry in inv.get(section, {}).items():
            file = Path(entry['path'])
            after = sha256(file) if file.is_file() else None
            same = after == entry['sha256']
            verified[f'{section}/{name}'] = {'before': entry['sha256'], 'after': after, 'unchanged': same}
            if not same:
                changed.append(f'{section}/{name}')
    prepared_after = None
    if inv.get('prepared_manifest_path'):
        try:
            prepared_after = verify_manifest(inv['prepared_manifest_path'])
            if prepared_after['dataset_fingerprint'] != inv['dataset_integrity']['dataset_fingerprint']:
                changed.append('prepared_dataset')
        except Exception as exc:
            changed.append('prepared_dataset')
            prepared_after = {'error': f'{type(exc).__name__}: {exc}'}
    errors = [r for r in rows if r.get('error')]
    complete = bool(jobs) and len(rows) == len(jobs) and not errors and not changed and reason is None
    groups = defaultdict(list)
    for row in rows:
        for variant, value in row.get('variants', {}).items():
            key = f"{row['job']['field']}/{row['job'].get('resolution', 'continuous')}/{variant}"
            groups[key].append((row, value))
    observed = {}
    for name, items in groups.items():
        errors_at_surface = [v['field_metrics']['observed_original_surface']['interpolation_mae'] for _, v in items]
        measurable = [value for value in errors_at_surface if value is not None]
        observed[name] = {'patterns': len(items), 'sources': len({r['source_id'] for r, _ in items}),
                          'patterns_with_sampled_exterior': len(measurable),
                          'mean_exterior_interpolation_mae': float(np.mean(measurable)) if measurable else None,
                          'per_pattern': {r['pattern_id']: v['field_metrics']['observed_original_surface'] for r, v in items}}
    # Paired drift summaries preserve start, contact and segmentation controls.
    pose_groups = defaultdict(list)
    for row in rows:
        definitions = list(row.get('variants', {}).items()) or [('no_field', {'refinement': row.get('refinement', [])})]
        for variant, values in definitions:
            for probe in values.get('refinement', []):
                if 'after' not in probe:
                    continue
                key = '/'.join(map(str, (row['job'].get('field', 'none'), row['job'].get('resolution', 'continuous'), variant,
                                        probe['exterior'], probe['contacts'], probe['confidence'], probe['degrees'])))
                pose_groups[key].append((row, probe))
    pose_observations = {}
    for name, items in pose_groups.items():
        improvements = [r['pattern_id'] for r, p in items if p['after']['whole_chamfer'] < p['before']['whole_chamfer'] - 1e-8]
        harms = [r['pattern_id'] for r, p in items if p['after']['whole_chamfer'] > p['before']['whole_chamfer'] + 1e-8]
        pose_observations[name] = {'patterns': len(items), 'sources': len({r['source_id'] for r, _ in items}),
                                   'improved_whole_chamfer': improvements, 'worsened_whole_chamfer': harms,
                                   'mean_before_chamfer': float(np.mean([p['before']['whole_chamfer'] for _, p in items])),
                                   'mean_after_chamfer': float(np.mean([p['after']['whole_chamfer'] for _, p in items]))}
    summary = {'kind': 'focused_field_diagnostic', 'status': 'complete' if complete else 'partial_or_blocked',
               'dataset_fingerprint': inv.get('dataset_integrity', {}).get('dataset_fingerprint'),
               'reason': reason, 'planned_jobs': len(jobs), 'completed_jobs': len(rows) - len(errors),
               'unrun_jobs': unrun, 'failed_jobs': errors,
               'conditional_skips': [r for r in rows if r.get('skipped')],
               'input_changes': changed, 'hash_verification': verified, 'prepared_verification_after': prepared_after,
               'checkpoint_changes': [name.removeprefix('checkpoints/') for name in changed if name.startswith('checkpoints/')],
               'checkpoint_verification': {name.removeprefix('checkpoints/'): value for name, value in verified.items() if name.startswith('checkpoints/')},
               'optimizer_updates': 0, 'production_acceptance': False,
               'observed': observed, 'pose_observations': pose_observations,
               'source_objects': sorted({r.get('source_id') for r in rows if r.get('source_id')}),
               'supported_explanation_tests': 'Compare continuous versus fresh grids on identical points, starts, contacts and exterior masks. Fixed-confidence refinements and fixed-sigma matcher controls isolate uncertainty from content.',
               'unresolved': ['Oracle contacts and poses are diagnostic controls, not attainable production performance.',
                              'These probes characterize the old checkpoint; they do not establish how a newly trained architecture performs.',
                              'No automatic architecture, acceptance threshold or training decision is made.']}
    write(output / 'summary.json', summary)
    lines = ['# Field diagnosis follow-up', '', f"Status: **{summary['status']}**; {summary['completed_jobs']}/{len(jobs)} jobs.",
             '', 'No optimizer updates. Existing checkpoints and prepared artifacts were hashed before and after.',
             '', '## Observed field interpolation', '', '| Field / resolution / clamp | Patterns | Sources | Exterior interpolation MAE |',
             '|---|---:|---:|---:|']
    for key, value in observed.items():
        mae = value['mean_exterior_interpolation_mae']
        display = '-' if mae is None else f'{mae:.6g}'
        lines.append(f"| {key} | {value['patterns']} | {value['sources']} | {display} |")
    lines += ['', '## Controlled pose changes', '', '| Field / resolution / clamp / exterior / contacts / confidence / degrees | Patterns | Improved | Worsened | Before CD | After CD |',
              '|---|---:|---:|---:|---:|---:|']
    for key, value in pose_observations.items():
        lines.append(f"| {key} | {value['patterns']} | {len(value['improved_whole_chamfer'])} | {len(value['worsened_whole_chamfer'])} | {value['mean_before_chamfer']:.6g} | {value['mean_after_chamfer']:.6g} |")
    lines += ['', '## Supported explanation versus unresolved', '',
              'A grid artifact explanation is supported only if the continuous control removes the error while other inputs remain fixed. Inspect per-example counters, gradients and source coverage before attributing a unique cause.', '',
              'Matcher content interventions hold each query’s predicted uncertainty fixed. Their probability matrices are in tensors; they do not use the legacy evaluation loss as a proxy for conditioning.', '',
              f"Conditional 256-grid skips: {len(summary['conditional_skips'])}. Unrun: {len(unrun)}. Failed: {len(errors)}. Input changes: {len(changed)}.", '',
              *summary['unresolved']]
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    bundle = output / 'field_diagnostic_bundle.tar.gz'
    with tarfile.open(bundle, 'w:gz') as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path != bundle and not path.is_symlink() and 'scratch' not in path.relative_to(output).parts and not path.name.endswith('.tmp'):
                archive.add(path, arcname=path.relative_to(output).as_posix(), recursive=False)
    if tree_bytes(output) > 2 * 1024 ** 3:
        raise RuntimeError('Field diagnosis including archive exceeded its 2 GiB artifact cap')
    return summary
