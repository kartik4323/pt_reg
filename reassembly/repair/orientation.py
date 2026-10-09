"""Optional XYZ-only contact orientation; no mesh, labels or shape field access."""
from __future__ import annotations

import copy
import math
import numpy as np
from scipy.spatial import cKDTree

from reassembly import solver
from reassembly.geometry import weighted_kabsch


DEFAULTS = dict(enabled=False, neighbors=32, flatness_ratio=.2,
                tangent_ratio=.1, radial_cosine=.2, weight=.0004, offset=.02)


def options(cfg=None):
    raw = (cfg or {}).get('solver', cfg or {}).get('contact_orientation', {})
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError('contact_orientation must contain only declared orientation settings')
    result = {**DEFAULTS, **raw}
    if not isinstance(result['enabled'], bool):
        raise ValueError('contact_orientation.enabled must be boolean')
    if type(result['neighbors']) is not int or result['neighbors'] < 3:
        raise ValueError('contact_orientation.neighbors must be an integer >= 3')
    for name in ('flatness_ratio', 'tangent_ratio', 'radial_cosine'):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f'contact_orientation.{name} must lie strictly between zero and one')
    for name in ('weight', 'offset'):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'contact_orientation.{name} must be finite and positive')
    return result


def validate_overrides(overrides):
    """The ablation interface cannot change confidence or acceptance thresholds."""
    if overrides is None:
        return None
    if not isinstance(overrides, dict) or set(overrides) != {'contact_orientation'}:
        raise ValueError('Solver overrides may contain only contact_orientation')
    options({'solver': overrides})
    return copy.deepcopy(overrides)


def estimate_normals(points, queries, config):
    points, queries = np.asarray(points, dtype=float), np.asarray(queries, dtype=float)
    if (points.ndim != 2 or queries.ndim != 2 or points.shape[1:] != (3,)
            or queries.shape[1:] != (3,) or not np.isfinite(points).all() or not np.isfinite(queries).all()):
        raise ValueError('Normal estimation requires finite XYZ arrays')
    if len(points) < 3:
        return np.zeros_like(queries), np.zeros(len(queries), dtype=bool)
    # Canonical ordering stabilizes centroid reductions and KD-tree tie order.
    points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
    k = min(config['neighbors'], len(points))
    neighbors = cKDTree(points).query(queries, k=k)[1]
    patches = points[neighbors]
    centered = patches - patches.mean(1, keepdims=True)
    eigenvalues, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', centered, centered)/k)
    normals = vectors[:, :, 0].copy()
    radial = queries - points.mean(0)
    projection = (normals * radial).sum(-1)
    normals *= np.where(projection >= 0, 1., -1.)[:, None]
    reliable = ((eigenvalues[:, 1] > 1e-12)
        & (eigenvalues[:, 0]/np.maximum(eigenvalues[:, 1], 1e-12) < config['flatness_ratio'])
        & (eigenvalues[:, 1]/np.maximum(eigenvalues[:, 2], 1e-12) > config['tangent_ratio'])
        & (np.abs(projection)/np.maximum(np.linalg.norm(radial, axis=1), 1e-12) > config['radial_cosine']))
    return normals, reliable


def attach_normals(matches, fragment_points, cfg):
    """Return copied matches; normals use complete local fragments, including context."""
    config = options(cfg)
    if not config['enabled']:
        return matches
    result = []
    for pair in matches:
        item = dict(pair)
        for side, part in (('source', pair['i']), ('target', pair['j'])):
            item[side+'_normals'], item[side+'_normal_reliability'] = estimate_normals(
                fragment_points[part], pair[side+'_xyz'], config)
        result.append(item)
    return result


def attach_selected(selected, pair):
    names = ('source_normals', 'target_normals', 'source_normal_reliability', 'target_normal_reliability')
    if not any(name in pair for name in names):
        return
    if not all(name in pair for name in names):
        raise ValueError('Contact normals require both directions and reliability arrays')
    for side in ('source', 'target'):
        normals = np.asarray(pair[side+'_normals'], dtype=float)
        reliability = np.asarray(pair[side+'_normal_reliability'])
        count = len(pair[side+'_xyz'])
        if (normals.shape != (count, 3) or reliability.shape != (count,)
                or reliability.dtype != np.bool_ or not np.isfinite(normals).all()
                or not np.allclose(np.linalg.norm(normals[reliability], axis=1), 1., atol=1e-6)):
            raise ValueError('Reliable contact normals must be finite unit vectors with boolean reliability')
        ids = selected[side+'_indices']
        selected[side+'_normals'] = normals[ids]
        selected[side+'_normal_reliability'] = reliability[ids]


def reliable_support(pair, min_mass=.05):
    if 'source_normals' not in pair:
        return np.zeros(len(pair['weights']), dtype=bool)
    reliable = pair['source_normal_reliability'] & pair['target_normal_reliability']
    if (pair['weights'][reliable].sum() < min_mass
            or any(len(np.unique(pair[side+'_indices'][reliable])) < 3 for side in ('source', 'target'))):
        reliable[:] = False
    return reliable


def score(poses, correspondences, config, min_mass=.05):
    if not config['enabled']:
        return 0., dict(orientation_score=0., orientation_reliable_mass=0., orientation_reliable_fraction=0.)
    rotations, _ = poses
    numerator = mass = total = 0.
    for pair in correspondences:
        total += float(pair['weights'].sum())
        reliable = reliable_support(pair, min_mass)
        if not reliable.any():
            continue
        a = pair['source_normals'][reliable] @ rotations[pair['i']].T
        b = pair['target_normals'][reliable] @ rotations[pair['j']].T
        dot = np.clip((a*b).sum(-1), -1., 1.)
        weights = pair['weights'][reliable]
        numerator += float((weights*((1+dot)/2)**2).sum())
        mass += float(weights.sum())
    raw = numerator/max(mass, 1e-12)
    return config['weight']*raw, dict(orientation_score=raw, orientation_reliable_mass=mass,
                                      orientation_reliable_fraction=mass/max(total, 1e-12))


def _fit(source, target, weights, pair, ids, config, min_mass, oriented):
    if not oriented:
        return weighted_kabsch(source[ids], target[ids], weights[ids], min_mass=min_mass)
    reliable = reliable_support(pair, min_mass)[ids]
    if (weights[ids][reliable].sum() < min_mass
            or any(len(np.unique(pair[side+'_indices'][ids][reliable])) < 3 for side in ('source', 'target'))):
        return dict(valid=False)
    a, b, w = source[ids], target[ids], weights[ids]
    na, nb = pair['source_normals'][ids], pair['target_normals'][ids]
    extra = w*reliable/3
    # The symmetric offsets leave centers and total probability mass unchanged.
    return weighted_kabsch(np.concatenate((a, a+config['offset']*na, a-config['offset']*na)),
        np.concatenate((b, b-config['offset']*nb, b+config['offset']*nb)),
        np.concatenate((w-2*extra, extra, extra)), min_mass=min_mass)


def pair_candidates(pair, solver_options, config):
    """Use bounded legacy and normal-constrained proposals; keep real fit support."""
    minimum = float(solver_options.get('min_pair_mass', .05))
    reliable = reliable_support(pair, minimum)
    limit = min(16, max(1, int(solver_options.get('max_pair_candidates', 16))))
    if not config['enabled'] or not reliable.any() or limit < 2:
        return solver._pair_candidates(pair, solver_options)
    legacy_limit, normal_limit = (limit+1)//2, limit//2
    source, target, weights = (pair[k] for k in ('source', 'target', 'weights'))
    subsets = [np.arange(len(weights))]
    seeds = []
    for seed in range(len(weights)):
        if len(subsets) >= max(legacy_limit, normal_limit): break
        if seeds and np.min(np.linalg.norm(source[seeds]-source[seed], axis=1)) < .01: continue
        seeds.append(seed)
        subsets.append(np.argsort(np.linalg.norm(source-source[seed], axis=1), kind='stable')[:min(12,len(weights))])
    threshold = float(solver_options.get('inlier_threshold', .04))
    fits = []
    for oriented, budget in ((False, legacy_limit), (True, normal_limit)):
        for ids in subsets[:budget]:
            fit = _fit(source, target, weights, pair, ids, config, minimum, oriented)
            if not fit['valid']: continue
            residual = np.linalg.norm(source @ fit['rotation'].T+fit['translation']-target, axis=1)
            inliers = residual <= threshold
            if any(len(np.unique(pair[side+'_indices'][inliers])) < 3 for side in ('source', 'target')): continue
            fit = _fit(source, target, weights, pair, np.flatnonzero(inliers), config, minimum, oriented)
            if not fit['valid']: continue
            residual = np.linalg.norm(source @ fit['rotation'].T+fit['translation']-target, axis=1)
            fit.update(proposal_type='normal_constrained' if oriented else 'legacy',
                supported_mass=float(weights[residual<=threshold].sum()))
            pair_poses = (np.stack([fit['rotation'], np.eye(3)]), np.stack([fit['translation'], np.zeros(3)]))
            local_pair = {**pair, 'i':0, 'j':1}
            penalty, detail = score(pair_poses, [local_pair], config, minimum)
            fit.update(detail)
            fit['score'] = float((weights*np.minimum(residual**2, threshold**2)).sum()/weights.sum()) + penalty
            fits.append(fit)
    fits.sort(key=lambda x: (x['score'], -x['supported_mass']))
    distinct = []
    for fit in fits:
        if any(np.linalg.norm(fit['rotation']-old['rotation']) < .05 and np.linalg.norm(fit['translation']-old['translation']) < .01 for old in distinct): continue
        distinct.append(fit)
        if len(distinct) >= min(4, max(1, int(solver_options.get('keep_pair_candidates', 4)))): break
    return distinct
