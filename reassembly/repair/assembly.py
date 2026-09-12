"""Field-independent contact candidates with continuous-field scoring/refinement.

Pose arrays use ``x_aligned = x_local @ R.T + t`` in shared normalized units.
The reference is fixed to identity. Candidate fitting and spanning-tree assembly
are performed once, then the identical candidate cache is reused for paired
contact-only/predicted/GT/perturbed evaluations. GT is never read by this path.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field as dataclass_field
from itertools import combinations, product

import numpy as np
import torch

from reassembly import solver
from .fields import FieldSampler


_CANDIDATE_DEFAULTS = dict(min_correspondence_weight=.001, min_pair_mass=.05,
                         max_pair_candidates=16, keep_pair_candidates=4, inlier_threshold=.04)


def _candidate_options(cfg):
    options = solver._options(cfg)
    return {key: options.get(key, default) for key, default in _CANDIDATE_DEFAULTS.items()}


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _readonly(value):
    value = np.array(_numpy(value), copy=True)
    value.setflags(write=False)
    return value


@dataclass
class CandidateCache:
    num_fragments: int
    anchor_index: int
    correspondences: list[dict]
    pair_candidates: dict
    hypotheses: list[dict]
    exterior_points: list[np.ndarray] | None
    exterior_weights: list[np.ndarray] | None
    candidate_options: dict
    diagnostics: dict
    fingerprint: str
    encoded: dict | None = None
    matches: list[dict] = dataclass_field(default_factory=list)
    pairs: list[dict] = dataclass_field(default_factory=list)

    @property
    def poses(self):
        return [item['poses'] for item in self.hypotheses]

    @property
    def initial_poses(self):
        return self.poses

    @property
    def assemblies(self):
        return self.poses


def build_candidates_from_matches(matches, num_fragments, anchor_index=0,
                                  exterior_points=None, exterior_weights=None, cfg=None):
    """Fit the existing rigid solver's candidates without consulting any field.

    Matches carry i,j,source_xyz,target_xyz,weights and optional directional
    matchability. This model-independent adapter also supports explicit oracle
    diagnostic controls without introducing labels into learned inference.
    """
    if num_fragments not in (2, 3) or not 0 <= anchor_index < num_fragments:
        raise ValueError("candidate construction requires two or three fragments and a valid reference")
    options = solver._options(cfg)
    if exterior_points is not None:
        exterior_points = [_readonly(np.asarray(points, dtype=np.float64)) for points in exterior_points]
        if len(exterior_points) != num_fragments or any(p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() for p in exterior_points):
            raise ValueError("one finite XYZ exterior sample array is required per fragment")
    if exterior_weights is not None:
        if exterior_points is None or len(exterior_weights) != num_fragments:
            raise ValueError("exterior weights require one matching exterior array per fragment")
        exterior_weights = [_readonly(np.asarray(w, dtype=np.float64)) for w in exterior_weights]
        if any(w.shape != (len(p),) or not np.isfinite(w).all() or (w < 0).any() or (w > 1).any()
               for p, w in zip(exterior_points, exterior_weights)):
            raise ValueError("exterior weights must be finite probabilities")
    correspondences, candidates, seen, pairs = [], {}, set(), {}
    copied_matches = []
    for pair in matches:
        i, j = int(pair['i']), int(pair['j'])
        if not 0 <= i < j < num_fragments or (i, j) in seen:
            raise ValueError("each unordered pair must appear once with 0 <= i < j < fragment count")
        seen.add((i, j))
        pair = {key: (_readonly(value) if isinstance(value, (np.ndarray, torch.Tensor)) else copy.deepcopy(value))
                for key, value in pair.items()}
        copied_matches.append(pair)
        selected = solver._correspondences(pair, options)
        key = f'{i}-{j}'
        pairs[key] = dict(source_points=len(pair['source_xyz']), target_points=len(pair['target_xyz']),
                          raw_mass=float(np.asarray(pair['weights']).sum()), selected_correspondences=0,
                          candidate_count=0, reason='insufficient_correspondence_support')
        if selected is None:
            continue
        correspondences.append(selected)
        fits = solver._pair_candidates(selected, options)
        pairs[key].update(selected_correspondences=len(selected['weights']),
                          unique_source_points=len(np.unique(selected['source_indices'])),
                          unique_target_points=len(np.unique(selected['target_indices'])),
                          selected_mass=float(selected['weights'].sum()), confidence=selected['confidence'],
                          candidate_count=len(fits), reason=None if fits else 'degenerate_or_inconsistent_support')
        if fits:
            candidates[(i, j)] = fits
    hypotheses = []
    for edges in combinations(candidates, num_fragments-1):
        for indices in product(*(range(len(candidates[edge])) for edge in edges)):
            fits = tuple(candidates[edge][index] for edge, index in zip(edges, indices))
            poses = solver._compose_tree(list(edges), fits, num_fragments, anchor_index)
            if poses is not None:
                hypotheses.append(dict(id=len(hypotheses), edges=list(edges), pair_choices=list(indices),
                                       poses=tuple(_readonly(value) for value in poses)))
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(count=num_fragments, anchor=anchor_index, options=_candidate_options(cfg)), sort_keys=True).encode())
    for pair in correspondences:
        digest.update(f"{pair['i']}-{pair['j']}".encode())
        for key in ('source', 'target', 'weights'):
            digest.update(np.ascontiguousarray(pair[key], dtype=np.float64).tobytes())
    for item in hypotheses:
        for value in item['poses']:
            digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
    fingerprint = digest.hexdigest()
    diagnostics = dict(pair_candidate_counts={f'{i}-{j}': len(values) for (i, j), values in candidates.items()},
                       pairs=pairs, anchor_index=int(anchor_index), hypotheses=len(hypotheses),
                       refinement_accepted_steps=0, candidate_fingerprint=fingerprint,
                       connected_candidate_graph=bool(hypotheses), candidate_generation_uses_field=False,
                       tree_count=len({tuple(item['edges']) for item in hypotheses}))
    return CandidateCache(num_fragments, anchor_index, correspondences, candidates, hypotheses,
                          exterior_points, exterior_weights, _candidate_options(cfg), diagnostics,
                          fingerprint, matches=copied_matches)


def build_candidates(model, batch, cfg, encoded=None):
    """XYZ-only batch-size-one contact prediction; fixed 256 exterior samples."""
    points, mask, anchor = batch['points'], batch['fragment_mask'], batch['anchor_index']
    if points.ndim != 4 or points.shape[0] != 1 or points.shape[-1] != 3:
        raise ValueError("build_candidates accepts one padded XYZ sample")
    count = int(_numpy(mask)[0].sum())
    if not np.array_equal(_numpy(mask)[0].astype(bool), np.arange(points.shape[1]) < count):
        raise ValueError("active fragments must precede padding")
    anchor_value = int(_numpy(anchor).reshape(-1)[0])
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        with torch.no_grad():
            encoded = model.encode(points, mask, anchor) if encoded is None else encoded
            predictions = model.match(encoded, use_scaffold=False)
            matches = []
            for pair in predictions:
                if pair.get('valid') is not None and not bool(_numpy(pair['valid']).reshape(-1)[0]):
                    continue
                match = dict(i=int(pair['i']), j=int(pair['j']))
                for name in ('source_xyz', 'target_xyz', 'weights', 'source_matchability', 'target_matchability'):
                    if name in pair:
                        match[name] = _numpy(pair[name])[0]
                matches.append(match)
            probabilities = _numpy(torch.sigmoid(-encoded['fracture_logits']))[0]
            exterior, exterior_weights = [], []
            for i in range(count):
                selected = np.linspace(0, points.shape[2]-1, min(256, points.shape[2]), dtype=int)
                exterior.append(_numpy(points)[0, i, selected])
                exterior_weights.append(probabilities[i, selected])
            cache = build_candidates_from_matches(matches, count, anchor_value, exterior, exterior_weights, cfg)
            cache.encoded = {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}
            cache.pairs = [{key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in pair.items()}
                           for pair in predictions]
            return cache
    finally:
        for module, training in modes.items():
            module.training = training


def solve_candidates(cache: CandidateCache, cfg, field: FieldSampler | None = None):
    """Score/refine cached poses using original confidence thresholds and caps."""
    if _candidate_options(cfg) != cache.candidate_options:
        raise ValueError("candidate thresholds changed: rebuild candidates instead of reusing incompatible support")
    options = solver._options(cfg)
    diagnostics = copy.deepcopy(cache.diagnostics)
    failure = dict(status='failed', confidence=0., rotations=None, translations=None,
                   anchor_index=cache.anchor_index, diagnostics=diagnostics)
    if not cache.hypotheses:
        return {**failure, 'reason': 'no_valid_connected_assembly'}
    scored = []
    for hypothesis in cache.hypotheses:
        score, detail = solver._score(hypothesis['poses'], cache.correspondences, cache.exterior_points,
                                     cache.exterior_weights, field, options)
        scored.append((score, hypothesis, detail))
    scored.sort(key=lambda item: item[0])
    finalists = []
    for initial_score, hypothesis, initial_detail in scored[:min(4, max(1, int(options.get('refine_candidates', 4))))]:
        refined, accepted = solver._refine(hypothesis['poses'], cache.anchor_index, cache.correspondences,
                                          cache.exterior_points, cache.exterior_weights, field, options)
        score, detail = solver._score(refined, cache.correspondences, cache.exterior_points,
                                     cache.exterior_weights, field, options)
        finalists.append((score, refined, detail, accepted, initial_score, initial_detail, hypothesis['id']))
    score, (rotations, translations), detail, accepted, initial_score, initial_detail, candidate_id = min(finalists, key=lambda item: item[0])
    diagnostics.update(detail)
    diagnostics.update(score=float(score), refinement_accepted_steps=accepted,
                       selected_candidate_id=candidate_id,
                       refined_candidate_ids=[value[-1] for value in finalists],
                       pre_refinement_score=float(initial_score),
                       pre_refinement_contact_score=initial_detail['contact_score'],
                       pre_refinement_prior_score=initial_detail['prior_score'],
                       pre_refinement_contact_rms=initial_detail['contact_rms'],
                       candidate_scores=[dict(id=item[1]['id'], score=float(item[0]), **item[2]) for item in scored])
    if not (np.isfinite(rotations).all() and np.isfinite(translations).all()):
        return {**failure, 'reason': 'nonfinite_refinement'}
    if not (np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-6)
            and np.allclose(np.linalg.det(rotations), 1., atol=1e-6)
            and np.allclose(rotations[cache.anchor_index], np.eye(3), atol=1e-8)
            and np.allclose(translations[cache.anchor_index], 0., atol=1e-8)):
        return {**failure, 'reason': 'invalid_rigid_refinement'}
    edge_confidence = {(pair['i'], pair['j']): pair['confidence'] for pair in cache.correspondences}
    mass_confidence = max(min(edge_confidence[edge] for edge in item['edges']) for item in cache.hypotheses)
    diagnostics['contact_confidence'] = float(mass_confidence)
    confidence = float(mass_confidence*np.exp(-detail['contact_rms']/max(float(options.get('huber_delta', .02)), 1e-6)))
    status = 'ok' if confidence >= float(options.get('min_assembly_confidence', .1)) else 'low_confidence'
    return dict(status=status, reason=None if status == 'ok' else 'weak_or_inconsistent_contact_support',
                confidence=confidence, rotations=rotations, translations=translations,
                anchor_index=cache.anchor_index, diagnostics=diagnostics)


def candidate_coverage(cache, sample, threshold=.01):
    """Evaluation-only geometric coverage; never changes selection or confidence."""
    from reassembly.evaluation import assembly_metrics
    rows = []
    for hypothesis in cache.hypotheses:
        rotations, translations = hypothesis['poses']
        metrics = assembly_metrics(sample, dict(rotations=rotations, translations=translations,
                                               status='diagnostic_candidate', confidence=0.), threshold)
        rows.append(dict(candidate_id=hypothesis['id'], **metrics))
    return dict(diagnostic_only=True, candidate_fingerprint=cache.fingerprint,
                candidates=len(rows), geometrically_valid_candidates=sum(row['success'] for row in rows),
                any_geometrically_valid=any(row['success'] for row in rows),
                best_whole_chamfer=min((row['whole_chamfer'] for row in rows), default=None),
                best_max_part_chamfer=min((max(row['per_part_chamfer']) for row in rows), default=None),
                rows=rows)


def candidate_pose_metrics(cache, sample, threshold=.01):
    coverage = candidate_coverage(cache, sample, threshold)
    return dict(candidate_count=coverage['candidates'],
                geometrically_correct_candidates=coverage['geometrically_valid_candidates'],
                best_candidate_whole_chamfer=coverage['best_whole_chamfer'],
                candidate_oracle_success=coverage['any_geometrically_valid'],
                candidate_fingerprint=cache.fingerprint, diagnostic_only=True)
