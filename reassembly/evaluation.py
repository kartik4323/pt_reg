"""Rigid assembly evaluation, XYZ inference and explicit pilot acceptance gates."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from .checkpoints import load_checkpoint
from . import ARCHITECTURE
from .data import FractureDataset, make_oracle_field
from .geometry import export_transforms, normalize_fragments
from .model import ReassemblyModel
from .resources import seed_all, write_json, jsonable
from .solver import solve_assembly
from .training import collate_samples


def chamfer(a, b) -> float:
    """Symmetric mean Euclidean NN distance (not squared), normalized units."""
    a, b = np.asarray(a), np.asarray(b)
    return float((cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()) / 2)


def assembly_metrics(sample: dict, result: dict, threshold: float) -> dict:
    count = int(sample['fragment_mask'].sum())
    failure = result['rotations'] is None
    metrics = {'status': result['status'], 'failed': failure, 'success': False,
               'confidence': float(result['confidence']),
               'contact_rms': result.get('diagnostics', {}).get('contact_rms'),
               'whole_chamfer': None, 'per_part_chamfer': None,
               'rotation_deg': None, 'translation_error': None}
    if failure:
        return metrics
    points = sample['points'][:count].numpy()
    target = sample['canonical_points'][:count].numpy()
    rotations, translations = np.asarray(result['rotations']), np.asarray(result['translations'])
    aligned = np.einsum('fni,fji->fnj', points, rotations) + translations[:, None]
    part_cd = [chamfer(a, b) for a, b in zip(aligned, target)]
    rotation_gt = sample['rotations_gt'][:count].numpy()
    cosine = np.clip((np.einsum('fij,fij->f', rotations, rotation_gt) - 1) / 2, -1, 1)
    translation_gt = sample['translations_gt'][:count].numpy()
    moving = np.arange(count) != int(sample['anchor_index'])
    metrics.update(whole_chamfer=chamfer(aligned.reshape(-1, 3), target.reshape(-1, 3)),
                   per_part_chamfer=part_cd, rotation_deg=np.degrees(np.arccos(cosine))[moving].tolist(),
                   translation_error=np.linalg.norm(translations - translation_gt, axis=-1)[moving].tolist())
    # Geometry, including each part, determines success when poses are symmetric.
    metrics['success'] = bool(max(part_cd) <= threshold and metrics['whole_chamfer'] <= threshold)
    return metrics


def aggregate(rows: list[dict]) -> dict:
    result = {'count': len(rows), 'success_rate': None, 'failure_rate': None,
              'low_confidence_rate': None}
    if not rows:
        return result
    result.update(success_rate=float(np.mean([r['success'] for r in rows])),
                  failure_rate=float(np.mean([r['failed'] for r in rows])),
                  low_confidence_rate=float(np.mean([r['status'] == 'low_confidence' for r in rows])))
    for key in ('whole_chamfer', 'per_part_chamfer', 'contact_rms', 'rotation_deg', 'translation_error'):
        values = [value for row in rows if row.get(key) is not None
                  for value in np.asarray(row[key]).reshape(-1) if np.isfinite(value)]
        result[key] = None if not values else {'mean': float(np.mean(values)),
                                               'median': float(np.median(values)),
                                               'p90': float(np.percentile(values, 90))}
    result['metric_denominator_note'] = 'Distance/pose means exclude failed solves; failure rate includes every sample.'
    return result


def _load_model(checkpoint: Path, cfg: dict, device: torch.device,
                fingerprint: str | None = None):
    state = load_checkpoint(checkpoint, cfg=cfg, dataset_fingerprint=fingerprint, stage=3)
    model = ReassemblyModel(cfg).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    return model, state


def infer_fragments(fragments: list[np.ndarray], checkpoint: Path, cfg: dict,
                    device: torch.device, condition: str = 'predicted') -> dict:
    """Accept only XYZ; matrices act on original-resolution input arrays."""
    if condition not in ('predicted', 'contact_only'):
        raise ValueError('XYZ inference supports predicted or contact_only, never oracle inputs')
    model, state = _load_model(checkpoint, cfg, device)
    trained = state['cfg']['train'].get('condition', 'predicted')
    if trained != condition:
        raise ValueError(f'Checkpoint was trained for {trained}, requested {condition}')
    normalized = normalize_fragments(fragments, cfg['data']['points_per_fragment'], cfg['data']['seed'])
    batch = {'points': torch.as_tensor(normalized['points'], device=device).unsqueeze(0),
             'fragment_mask': torch.as_tensor(normalized['fragment_mask'], device=device).unsqueeze(0),
             'anchor_index': torch.tensor([normalized['anchor_index']], device=device)}
    result = solve_assembly(model, batch, cfg, condition)
    if result['rotations'] is not None:
        result.update(export_transforms(result['rotations'], result['translations'], normalized))
    else:
        result.update(transforms=None, aligned_fragments=None)
    result.update(schema_version=2, convention='x_aligned = x @ R.T + t',
                  checkpoint=str(Path(checkpoint).resolve()), dataset_fingerprint=state['dataset_fingerprint'])
    result['scaffold_frame'] = {'name': 'normalized_reference', 'scale': normalized['scale'],
                               'reference_centroid': normalized['centroids'][normalized['anchor_index']],
                               'to_output_coordinates': 'x_output = scale * x_field + reference_centroid',
                               'to_output_distance': 'distance_output = scale * distance_field'}
    return result


def evaluate(checkpoint: Path, manifest: Path, output: Path, cfg: dict, device: torch.device,
             split='test', condition='predicted', limit: int | None = None, overfit=False,
             guard=None) -> dict:
    dataset = FractureDataset(manifest, 'train' if overfit else split, cfg, stage=3,
                              fixed=True, limit=cfg['train']['overfit_patterns'] if overfit else limit)
    if not len(dataset):
        raise ValueError('Requested evaluation split is empty')
    model, state = _load_model(checkpoint, cfg, device, dataset.fingerprint)
    trained = state['cfg']['train'].get('condition', 'predicted')
    if condition == 'contact_only' and trained != 'contact_only':
        raise ValueError('Contact-only comparison requires its own contact-only trained checkpoint')
    if condition != 'contact_only' and trained != 'predicted':
        raise ValueError('Scaffold comparisons require a checkpoint trained with predicted scaffolds')
    if overfit != (state.get('purpose') == 'overfit'):
        raise ValueError('Fixed overfit and held-out pilot checkpoints cannot be interchanged')
    seed_all(cfg['train']['seed'])
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Select a fresh evaluation output directory; existing reports are preserved')
    if guard:
        guard.check()
    output.mkdir(parents=True, exist_ok=True)
    rows, sdf_errors, calibration = [], [], []
    started = time.perf_counter()
    for index in range(len(dataset)):
        sample = dataset[index]
        batch = collate_samples([sample], device)
        oracle = make_oracle_field(sample) if condition == 'gt' else None
        result = solve_assembly(model, batch, cfg, condition, field_override=oracle)
        row = assembly_metrics(sample, result, cfg['solver']['success_threshold'])
        row.update(pattern_id=sample['pattern_id'], source_id=sample['source_id'],
                   band=sample['band'], cut_family=sample['cut_family'], pieces=int(sample['fragment_mask'].sum()),
                   reason=result.get('reason'), diagnostics=result['diagnostics'])
        # Supervision is used only for reporting, never for candidate poses.
        from .losses import correspondence_loss, segmentation_loss
        with torch.no_grad():
            encoded = model.encode(batch['points'], batch['fragment_mask'], batch['anchor_index'])
            pairs = model.match(encoded, use_scaffold=condition != 'contact_only')
            matching_diagnostics = {}
            matching, positives = correspondence_loss(
                pairs, batch, float(cfg['train'].get('contact_radius', .05)),
                localization_sigma=float(cfg['train'].get('contact_sigma', .01)),
                localization_weight=float(cfg['loss'].get('matching_localization', 1.0)), diagnostics=matching_diagnostics)
            row.update(predicted_model_matching_loss=float(matching), positive_contact_targets=int(positives),
                       fracture_segmentation_loss=float(segmentation_loss(encoded['fracture_logits'], batch['fracture_labels'], batch['fragment_mask'])))
            row.update({name: float(value) for name, value in matching_diagnostics.items()})
        # Measure the learned reconstruction independently of the oracle ablation.
        if condition != 'contact_only':
            with torch.no_grad():
                field = model.scaffold(encoded, batch['sdf_queries'])
                tau = cfg['model']['truncation']
                errors = (field['distance'].clamp(-tau, tau) - batch['sdf_values'].clamp(-tau, tau)).abs()
                sigma = field['log_scale'].exp()
                row.update(sdf_l1=float(errors.mean()), uncertainty_mae=float((errors-sigma).abs().mean()))
                sdf_errors.append(row['sdf_l1'])
                # Bounded aggregate calibration bins; no per-query artifact growth.
                for lower, upper in ((0, .005), (.005, .02), (.02, 1)):
                    mask = (sigma >= lower) & (sigma < upper)
                    if mask.any():
                        calibration.append({'bin': [lower, upper], 'count': int(mask.sum()),
                                            'error': float(errors[mask].mean()), 'sigma': float(sigma[mask].mean())})
        rows.append(row)
        if index < min(8, int(cfg.get('evaluation', {}).get('qualitative_limit', 8))):
            arrays = {'points': sample['points'].numpy(), 'mask': sample['fragment_mask'].numpy(),
                      'target': sample['canonical_points'].numpy()}
            if result['rotations'] is not None:
                arrays.update(rotations=result['rotations'], translations=result['translations'])
            if result.get('scaffold'):
                arrays.update(scaffold_distance=result['scaffold']['distance'],
                              scaffold_uncertainty=result['scaffold']['uncertainty'],
                              scaffold_bounds=result['scaffold']['bounds'])
            if guard:
                guard.check(additional_bytes=sum(v.nbytes for v in arrays.values()))
            np.savez_compressed(output / f'example_{index:02d}.npz', **arrays)
        print(f'evaluate {condition} {index+1}/{len(dataset)}: {row["status"]}, success={row["success"]}', flush=True)
        # CLI summaries omit the sample array. Persist the actual failure
        # evidence in tee'd logs as well, so a logs-only handoff is actionable.
        print('sample_diagnostic ' + json.dumps(jsonable(row), allow_nan=False), flush=True)
    summary = aggregate(rows)
    report = {'schema_version': 2, 'architecture': ARCHITECTURE, 'kind': 'evaluation', 'purpose': state['purpose'],
              'condition': condition, 'split': 'fixed_train_16' if overfit else split,
              'dataset_fingerprint': dataset.fingerprint, 'checkpoint': str(Path(checkpoint).resolve()),
              'trained_updates': state['step'], 'training_lineage': state.get('training_lineage', {}), 'config': cfg,
              'metric_convention': 'symmetric mean Euclidean NN distance in shared normalized units',
              'success_threshold': cfg['solver']['success_threshold'], 'summary': summary,
              'by_band': {band: aggregate([r for r in rows if r['band'] == band])
                          for band in ('easy', 'intermediate', 'hard')},
              'by_pieces': {str(count): aggregate([r for r in rows if r['pieces'] == count]) for count in (2, 3)},
              'samples': rows, 'elapsed_seconds': time.perf_counter()-started,
              'predicted_sdf_l1': float(np.mean(sdf_errors)) if sdf_errors else None,
              'calibration_bins': calibration,
              'diagnostic_note': 'Matching/segmentation/SDF diagnostics measure the learned model. GT and perturbed fields are used only in their explicitly labeled solver ablations. Pre/post contact residuals are in each sample diagnostics.',
              'fixed_fit_passed': bool(overfit and condition == 'predicted' and len(rows) == 16 and all(r['success'] for r in rows)
                                       and all(r['status'] == 'ok' for r in rows))}
    report['failure_breakdown'] = {
        'no_solution': sum(r['failed'] for r in rows),
        'geometrically_inaccurate_solution': sum(not r['failed'] and not r['success'] for r in rows),
        'geometrically_correct_low_confidence': sum(r['success'] and r['status'] != 'ok' for r in rows),
        'accepted': sum(r['success'] and r['status'] == 'ok' for r in rows),
    }
    write_json(output / 'evaluation.json', report, guard)
    return report


def pilot_report(report_paths: list[Path], output: Path, guard=None) -> dict:
    reports = [json.loads(Path(path).read_text(encoding='utf-8')) for path in report_paths]
    evals = [r for r in reports if r.get('kind') == 'evaluation' and r.get('purpose') == 'pilot']
    predicted = next((r for r in evals if r['condition'] == 'predicted'), None)
    baseline = next((r for r in evals if r['condition'] == 'contact_only'), None)
    comparable, benefit, delta = False, False, None
    if predicted and baseline:
        def budget(report):
            # Best selected steps may differ; allocated budgets must match.
            return {stage: {k: v for k, v in entry.items() if k not in ('condition', 'checkpoint_step')}
                    for stage, entry in report.get('training_lineage', {}).items()}
        comparable = (predicted['dataset_fingerprint'] == baseline['dataset_fingerprint']
                      and set(budget(predicted)) == {'1', '2', '3'}
                      and budget(predicted) == budget(baseline)
                      and all(predicted['config'][k] == baseline['config'][k] for k in ('data', 'model', 'loss', 'solver'))
                      and {k: v for k, v in predicted['config']['train'].items() if k != 'condition'}
                          == {k: v for k, v in baseline['config']['train'].items() if k != 'condition'}
                      and predicted['split'] == baseline['split']
                      and [r['pattern_id'] for r in predicted['samples']] == [r['pattern_id'] for r in baseline['samples']])
        if comparable:
            delta = predicted['summary']['success_rate'] - baseline['summary']['success_rate']
            pred_cd = predicted['summary'].get('whole_chamfer')
            base_cd = baseline['summary'].get('whole_chamfer')
            benefit = bool(delta > 0 and pred_cd and base_cd and pred_cd['mean'] <= base_cd['mean']
                           and predicted['summary']['failure_rate'] <= baseline['summary']['failure_rate'])
    from .cli import profile_signature
    fingerprint = predicted['dataset_fingerprint'] if predicted else None
    matching = [r for r in reports if fingerprint and r.get('dataset_fingerprint') == fingerprint]
    fixed_pass = any(r.get('fixed_fit_passed') and r.get('purpose') == 'overfit' and r.get('condition') == 'predicted'
                     and all(r.get('config', {}).get(key) == predicted['config'][key] for key in ('data', 'model', 'loss', 'solver'))
                     for r in matching)
    cuda_pass = any(r.get('kind') == 'preflight' and r.get('status') == 'passed'
                    and r.get('correctness_passed') and r.get('profile_signature') == profile_signature(predicted['config'])
                    and r.get('attempts') and r['attempts'][-1].get('max_reserved_bytes') is not None
                    and r['attempts'][-1]['max_reserved_bytes'] < predicted['config']['resources']['max_vram_gib'] * 1024**3
                    for r in matching)
    geometry_ready = any(r.get('kind') == 'preparation' and r.get('learning_ready') for r in matching)
    controls = {}
    for condition in ('gt', 'perturbed'):
        controls[condition] = any(r.get('kind') == 'evaluation' and r.get('purpose') == 'pilot'
                                 and r.get('condition') == condition and r.get('checkpoint') == predicted['checkpoint']
                                 and r.get('split') == predicted['split']
                                 and r.get('config') == predicted['config']
                                 and [s['pattern_id'] for s in r['samples']] == [s['pattern_id'] for s in predicted['samples']]
                                 for r in matching)
    if not comparable:
        outcome = 'incomplete: matched held-out contact-only and predicted-scaffold evaluations are required'
    elif not benefit:
        outcome = 'scaffold benefit not demonstrated; inspect reconstruction, contact matching, and pose refinement separately'
    else:
        outcome = 'useful scaffold guidance observed on this pilot; longer training remains a separate decision'
    report = {'schema_version': 2, 'kind': 'pilot_report', 'outcome': outcome,
              'comparisons_are_matched': comparable, 'scaffold_benefit': benefit,
              'success_rate_delta': delta, 'fixed_examples_fit': fixed_pass,
              'cuda_and_correctness_passed': cuda_pass, 'geometry_learning_ready': geometry_ready,
              'controls_complete': controls,
              'advance_eligible': bool(comparable and benefit and fixed_pass and cuda_pass and geometry_ready
                                       and all(controls.values()) and predicted['split'] in ('test', 'cut_holdout')),
              'input_reports': [str(Path(p).resolve()) for p in report_paths],
              'reports': reports, 'full_training_launched': False}
    write_json(output, report, guard)
    return report
