"""Paired field interventions. All oracle information stays in this package."""
from pathlib import Path
import time

import numpy as np
import torch

from diagnostics.reassembly_v2 import probes as original
from diagnostics.reassembly_v2.runtime import read, write
from reassembly import solver
from reassembly.evaluation import assembly_metrics
from reassembly.geometry import so3_exp
from reassembly.repair.fields import ClippedGridField


SEED = 4101
PERTURBATIONS = ((0, 0.), (5, .02), (15, .05))


def pose_starts(sample):
    """Reproduce the original diagnostic's three starts, including its RNG order."""
    count, anchor = int(sample['fragment_mask'].sum()), int(sample['anchor_index'])
    base_r = original.array(sample['rotations_gt'])[:count].astype(np.float64)
    base_t = original.array(sample['translations_gt'])[:count].astype(np.float64)
    for degrees, translation in PERTURBATIONS:
        rng = np.random.default_rng(SEED)
        r, t = base_r.copy(), base_t.copy()
        for i in range(count):
            if i == anchor:
                continue
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            update = so3_exp(axis * np.deg2rad(degrees))
            offset = rng.normal(size=3)
            offset *= translation / np.linalg.norm(offset)
            r[i], t[i] = update @ r[i], update @ t[i] + offset
        yield {'degrees': degrees, 'translation': translation, 'seed': SEED}, (r, t)


def oracle_contacts(sample, cfg):
    """Fixed independently sampled same-interface contacts, without learned gates.

    Use the union of directional nearest neighbors on the *original* 1,024 point
    samples. No artificial coincident points or target-coordinate substitution.
    Thresholds and non-collinearity checks remain the production defaults.
    """
    canonical, local = original.array(sample['canonical_points']), original.array(sample['points'])
    identities = original.array(sample['interface_ids'])
    count = int(sample['fragment_mask'].sum())
    radius, sigma = float(cfg['train']['contact_radius']), float(cfg['train']['contact_sigma'])
    contacts, traces = [], {}
    for i in range(count):
        for j in range(i + 1, count):
            a, b = np.flatnonzero(identities[i] >= 0), np.flatnonzero(identities[j] >= 0)
            label = f'{i}-{j}'
            if not len(a) or not len(b):
                traces[label] = {'reason': 'no_observed_interface', 'correspondences': 0}
                continue
            distance = np.linalg.norm(canonical[i, a, None] - canonical[j, b][None], axis=-1)
            allowed = (identities[i, a, None] == identities[j, b][None]) & (distance <= radius)
            masked = np.where(allowed, distance, np.inf)
            rows, cols = np.arange(len(a)), np.arange(len(b))
            ids = np.unique(np.concatenate((np.stack((rows, masked.argmin(1)), -1),
                                             np.stack((masked.argmin(0), cols), -1))), axis=0)
            ids = ids[allowed[ids[:, 0], ids[:, 1]]]
            if not len(ids):
                traces[label] = {'reason': 'noncontacting_or_no_local_support', 'correspondences': 0}
                continue
            source, target = a[ids[:, 0]], b[ids[:, 1]]
            weights = np.exp(-distance[ids[:, 0], ids[:, 1]] ** 2 / (2 * sigma ** 2))
            keep = weights >= float(cfg['solver'].get('min_correspondence_weight', 1e-3))
            source, target, weights = source[keep], target[keep], weights[keep]
            support = {'i': i, 'j': j, 'source': local[i, source].astype(np.float64),
                       'target': local[j, target].astype(np.float64), 'weights': weights,
                       'source_indices': source, 'target_indices': target, 'confidence': 1.}
            fits = solver._pair_candidates(support, cfg['solver']) if len(weights) >= 3 and weights.sum() >= cfg['solver'].get('min_pair_mass', .05) else []
            traces[label] = {'correspondences': len(weights), 'mass': float(weights.sum()),
                             'interfaces': sorted(set(identities[i, source].tolist())),
                             'reason': None if fits else 'insufficient_or_degenerate_support'}
            if fits:
                contacts.append(support)
    reached = {int(sample['anchor_index'])}
    for _ in range(count):
        for p in contacts:
            if p['i'] in reached or p['j'] in reached:
                reached.update((p['i'], p['j']))
    return contacts, {'pairs': traces, 'connected': len(reached) == count,
                      'method': 'same-interface directional-nearest union, Gaussian distance weights; original independent samples'}


def probe_queries(sample):
    """Stored training queries plus all observed original-surface points at GT poses."""
    count = int(sample['fragment_mask'].sum())
    p = original.array(sample['points'])[:count]
    r, t = original.array(sample['rotations_gt'])[:count], original.array(sample['translations_gt'])[:count]
    aligned = np.einsum('fni,fji->fnj', p, r) + t[:, None]
    exterior = original.array(sample['fracture_labels'])[:count] < .5
    q = original.array(sample['sdf_queries'])
    near = original.array(sample['sdf_near_mask']).astype(bool)
    extra = aligned[exterior]
    queries = np.concatenate((q, extra))
    n, total = len(q), len(queries)
    region = {
        'stored_queries': np.arange(total) < n,
        'near_surface': np.r_[near, np.zeros(len(extra), bool)],
        'surrounding': np.r_[~near, np.zeros(len(extra), bool)],
        'observed_original_surface': np.arange(total) >= n}
    return queries, region


def calibration(distance, sigma, target, mask):
    d, s, y = distance[mask], sigma[mask], target[mask]
    error = np.abs(d - y)
    bins = []
    for lo, hi in zip(np.linspace(-6., -2., 9)[:-1], np.linspace(-6., -2., 9)[1:]):
        chosen = (s >= np.exp(lo)) & (s <= np.exp(hi) if hi == -2. else s < np.exp(hi))
        bins.append({'sigma_low': float(np.exp(lo)), 'sigma_high': float(np.exp(hi)),
                     'count': int(chosen.sum()), 'mean_sigma': float(s[chosen].mean()) if chosen.any() else None,
                     'mae': float(error[chosen].mean()) if chosen.any() else None})
    return {'count': len(d), 'mae': float(error.mean()) if len(d) else None,
            'uncertainty_mae': float(np.abs(error - s).mean()) if len(d) else None,
            'fraction_error_below_sigma': float((error <= s).mean()) if len(d) else None,
            'bins': bins}


def field_metrics(field, direct, gt, queries, regions):
    distance, confidence, gradient = field.sample(queries)
    reference, _, direct_gradient = direct.sample(queries)
    truth = gt.values(queries)[0]
    sigma = .01 * (1 / np.maximum(confidence, 1e-12) - 1)
    regions = dict(regions, inside=truth < 0, outside=truth >= 0)
    result = {}
    for name, mask in regions.items():
        norms = np.linalg.norm(gradient[mask], axis=-1)
        ours, target_gradient = gradient[mask], direct_gradient[mask]
        denom = np.linalg.norm(ours, axis=-1) * np.linalg.norm(target_gradient, axis=-1)
        useful = denom > 1e-10
        result[name] = {
            **calibration(distance, sigma, truth, mask),
            'interpolation_mae': float(np.abs(distance[mask] - reference[mask]).mean()) if mask.any() else None,
            'sign_error_fraction': float(((distance[mask] < 0) != (truth[mask] < 0)).mean()) if mask.any() else None,
            'gradient_norm': original.distribution(norms),
            'gradient_error_norm': original.distribution(np.linalg.norm(ours - target_gradient, axis=-1)),
            'gradient_cosine': original.distribution((ours[useful] * target_gradient[useful]).sum(-1) / denom[useful]),
            'gradient_cosine_valid_points': int(useful.sum())}
    return result, {'query_xyz': queries.astype(np.float32), 'distance': distance.astype(np.float32),
                    'continuous_distance': reference.astype(np.float32), 'gt_distance': truth.astype(np.float32),
                    'gradient': gradient.astype(np.float32), 'sigma': sigma.astype(np.float32),
                    **{name + '_mask': mask for name, mask in regions.items()}}


class FixedConfidence:
    def __init__(self, field, confidence=.8):
        self.field, self.confidence = field, confidence

    def sample(self, points):
        distance, _, gradient = self.field.sample(points)
        return distance, np.full(len(points), self.confidence), gradient


def refine_probes(field, sample, encoded, contacts, contact_status, cfg, check=lambda: None):
    outputs = []
    anchor = int(sample['anchor_index'])
    def metric(poses):
        r, t = poses
        return assembly_metrics(sample, {'rotations': r, 'translations': t,
                                         'status': 'diagnostic_pose', 'confidence': 0}, cfg['solver']['success_threshold'])
    for exterior_kind in ('learned', 'oracle'):
        points, weights = original.exterior(sample, encoded, exterior_kind == 'oracle')
        for start, poses in pose_starts(sample):
            before = metric(poses)
            for contact_kind in ('field_only', 'fixed_oracle_contacts'):
                pairs = [] if contact_kind == 'field_only' else contacts
                if field is None and contact_kind == 'field_only':
                    continue
                if contact_kind == 'fixed_oracle_contacts' and not contact_status['connected']:
                    outputs.append({**start, 'exterior': exterior_kind, 'contacts': contact_kind,
                                    'skipped': 'oracle contact support is not connected'})
                    continue
                for confidence in (('native', 'fixed_0.8') if field is not None else ('none',)):
                    check()
                    used = FixedConfidence(field) if confidence == 'fixed_0.8' else field
                    score_before = solver._score(poses, pairs, points, weights, used, cfg['solver'])
                    refined, accepted = solver._refine(poses, anchor, pairs, points, weights, used, cfg['solver'])
                    np.testing.assert_array_equal(refined[0][anchor], poses[0][anchor])
                    np.testing.assert_array_equal(refined[1][anchor], poses[1][anchor])
                    np.testing.assert_allclose(np.linalg.det(refined[0]), 1., atol=1e-5)
                    after = metric(refined)
                    outputs.append({**start, 'exterior': exterior_kind, 'contacts': contact_kind,
                                    'confidence': confidence, 'before': before, 'after': after,
                                    'accepted_steps': accepted, 'before_score': score_before,
                                    'after_score': solver._score(refined, pairs, points, weights, used, cfg['solver']),
                                    'before_rotations': poses[0], 'before_translations': poses[1],
                                    'after_rotations': refined[0], 'after_translations': refined[1],
                                    'production_acceptance': False})
    return outputs


def fresh_grid_nodes(field, resolution, directory, token, check, progress, extent=2.25, chunk=2048):
    """Chunk-resumable fresh nodes; never interpolate a coarser grid to enlarge it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    description = {'token': token, 'resolution': resolution, 'extent': extent,
                   'chunk': chunk, 'truncation': field.truncation}
    meta = directory / 'cursor.json'
    total, completed = resolution ** 3, 0
    paths = [directory / 'distance.npy', directory / 'sigma.npy']
    if meta.exists():
        saved = read(meta)
        if saved['description'] != description:
            raise RuntimeError('Partial grid input differs; use a fresh output directory')
        completed = int(saved['completed_nodes'])
        if completed < 0 or completed > total:
            raise RuntimeError('Invalid partial grid cursor')
        arrays = [np.lib.format.open_memmap(p, mode='r+') for p in paths]
        if any(a.shape != (total,) or a.dtype != np.float32 for a in arrays):
            raise RuntimeError('Partial grid array shape or dtype changed')
    else:
        arrays = [np.lib.format.open_memmap(p, mode='w+', dtype=np.float32, shape=(total,)) for p in paths]
    started = time.monotonic()
    initial = completed
    bounds = np.array([[-extent] * 3, [extent] * 3])
    for begin in range(completed, total, chunk):
        check()
        end = min(total, begin + chunk)
        indices = np.arange(begin, end)
        coordinates = np.stack(np.unravel_index(indices, (resolution,) * 3), -1)
        xyz = bounds[0] + coordinates / (resolution - 1) * (bounds[1] - bounds[0])
        distance, sigma = field.values(xyz, untruncated=True)
        if not np.isfinite(distance).all() or not np.isfinite(sigma).all() or (sigma < 0).any():
            raise RuntimeError('Non-finite/invalid fresh grid node values')
        arrays[0][begin:end], arrays[1][begin:end] = distance, sigma
        for a in arrays:
            a.flush()
        write(meta, {'description': description, 'completed_nodes': end})
        progress(end, total)
    elapsed = time.monotonic() - started
    measured = total - initial
    timing = {'seconds': elapsed, 'new_nodes': measured, 'resumed_nodes': initial,
              'nodes_per_second': measured / elapsed if measured and elapsed else None,
              'projected_256_grid_seconds': (256 ** 3) * elapsed / measured if measured else None}
    return arrays[0].reshape((resolution,) * 3), arrays[1].reshape((resolution,) * 3), bounds, timing


def grid_variants(distance, sigma, bounds, truncation):
    before = solver.ScaffoldGrid(np.clip(distance, -truncation, truncation), sigma, bounds, truncation)
    after = ClippedGridField(solver.ScaffoldGrid(distance, sigma, bounds, truncation))
    return {'clamp_before': before, 'clamp_after': after}


def conditioning(model, encoded, predicted, gt):
    """Change field content while holding input encoding, queries, and sigma fixed."""
    rows = torch.arange(len(encoded['anchor_index']), device=encoded['anchor_index'].device)
    q = torch.cat((model.conditioning_queries[None], encoded['token_xyz'][rows, encoded['anchor_index']]), 1)
    xyz = original.array(q)[0]
    mu, sigma = predicted.values(xyz)
    shift = np.array([3 * 4.5 / 31, 0., 0.])
    content = {'predicted': mu, 'gt': gt.values(xyz)[0],
               'perturbed': np.clip(predicted.values(xyz - shift)[0] + .035, -predicted.truncation, predicted.truncation),
               'disabled': None}
    matrices, observations, baseline = {}, {}, None
    with torch.no_grad():
        production = model.match(encoded, use_scaffold=True)
        for kind, distance in content.items():
            prior = None if distance is None else torch.cat((q, torch.as_tensor(distance, device=q.device, dtype=q.dtype)[None, :, None],
                                                            torch.as_tensor(np.log(sigma), device=q.device, dtype=q.dtype)[None, :, None]), -1)
            pairs = model.match(encoded, use_scaffold=kind != 'disabled', prior_override=prior)
            if baseline is None:
                baseline = pairs
                for a, b in zip(production, pairs):
                    for key in ('source_prob', 'target_prob', 'weights'):
                        torch.testing.assert_close(a[key], b[key], atol=1e-5, rtol=1e-4)
            observations[kind] = {}
            for base, pair in zip(baseline, pairs):
                if not bool(pair['valid'][0]):
                    continue
                i, j = pair['i'], pair['j']
                key = f'{i}-{j}'
                for name in ('source_indices', 'target_indices'):
                    torch.testing.assert_close(base[name], pair[name], atol=0, rtol=0)
                    if kind == 'predicted':
                        matrices[f'{key}_{name}'] = original.array(pair[name])[0]
                if kind == 'predicted':
                    for name in ('source_xyz', 'target_xyz'):
                        matrices[f'{key}_{name}'] = original.array(pair[name])[0].astype(np.float32)
                change = {}
                for name in ('source_prob', 'target_prob', 'weights'):
                    a, b = original.array(base[name])[0], original.array(pair[name])[0]
                    matrices[f'{kind}_{key}_{name}'] = b.astype(np.float32)
                    change[name] = {'mean_absolute_change': float(np.abs(a - b).mean()),
                                    'max_absolute_change': float(np.abs(a - b).max()),
                                    'changed_argmax_fraction': float((a.argmax(-1) != b.argmax(-1)).mean())}
                observations[kind][key] = change
    matrices.update(query_xyz=xyz.astype(np.float32), fixed_sigma=sigma.astype(np.float32),
                    predicted_distance=mu.astype(np.float32), gt_distance=content['gt'].astype(np.float32),
                    perturbed_distance=content['perturbed'].astype(np.float32))
    return {'controls': observations, 'uncertainty': 'Predicted per-query sigma held identical for all enabled contents',
            'perturbation': {'translation': shift.tolist(), 'distance_offset': .035},
            'unchanged_path_reproduced': True, 'production_acceptance': False}, matrices
