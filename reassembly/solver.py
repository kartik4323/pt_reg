"""Confidence-preserving contact hypotheses and scaffold-guided rigid assembly.

This module deliberately separates neural prediction from discrete hypothesis
selection and numerical refinement. Neither the solver nor XYZ-only inference
uses fracture labels, source meshes, or ground-truth poses.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Callable

import numpy as np

from .geometry import skew, so3_exp, weighted_kabsch


def _options(cfg: dict | None) -> dict:
    cfg = cfg or {}
    return cfg.get("solver", cfg)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass
class ScaffoldGrid:
    """Signed distance and detached uncertainty, with xyz-indexed grid axes."""

    distance: np.ndarray
    uncertainty: np.ndarray
    bounds: np.ndarray
    truncation: float = 0.1

    def __post_init__(self) -> None:
        self.distance = np.asarray(self.distance, dtype=np.float64)
        self.uncertainty = np.asarray(self.uncertainty, dtype=np.float64)
        self.bounds = np.asarray(self.bounds, dtype=np.float64)
        if self.distance.ndim != 3 or min(self.distance.shape) < 2:
            raise ValueError("scaffold distance must be a three-dimensional grid of resolution >= 2")
        if self.uncertainty.shape != self.distance.shape:
            raise ValueError("scaffold uncertainty and distance grids must have matching shapes")
        if self.bounds.shape != (2, 3) or not np.all(self.bounds[1] > self.bounds[0]):
            raise ValueError("scaffold bounds must have shape [2,3] with positive extents")
        if not (np.isfinite(self.distance).all() and np.isfinite(self.uncertainty).all() and np.isfinite(self.bounds).all()):
            raise ValueError("scaffold contains non-finite values")
        if (self.uncertainty < 0).any():
            raise ValueError("scaffold uncertainty cannot be negative")

    def sample(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trilinear distance, confidence and analytic coordinate derivative.

        Points outside the field receive a small-confidence boundary residual,
        preventing unsupported poses from escaping the grid at zero cost.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("field sample coordinates must have shape [N,3]")
        shape = np.asarray(self.distance.shape)
        spacing = (shape - 1) / (self.bounds[1] - self.bounds[0])
        clipped = np.clip(points, self.bounds[0], self.bounds[1])
        position = (clipped - self.bounds[0]) * spacing
        lower = np.minimum(np.floor(position).astype(np.int64), shape - 2)
        fraction = position - lower
        distance = np.zeros(len(points))
        sigma = np.zeros(len(points))
        gradient = np.zeros_like(points)
        for corner in product((0, 1), repeat=3):
            offset = np.asarray(corner)
            index = lower + offset
            values = self.distance[tuple(index.T)]
            factors = np.where(offset, fraction, 1 - fraction)
            weight = factors.prod(axis=1)
            distance += weight * values
            sigma += weight * self.uncertainty[tuple(index.T)]
            for axis in range(3):
                others = [a for a in range(3) if a != axis]
                gradient[:, axis] += values * factors[:, others].prod(axis=1) * (1 if offset[axis] else -1) * spacing[axis]
        confidence = np.clip(0.01 / (0.01 + sigma), 0.02, 1.0)
        displacement = points - clipped
        outside_distance = np.linalg.norm(displacement, axis=1)
        outside = outside_distance > 0
        distance[outside] = self.truncation + outside_distance[outside]
        gradient[outside] = displacement[outside] / outside_distance[outside, None]
        confidence[outside] = 0.05
        return distance, confidence, gradient

    def as_dict(self) -> dict[str, Any]:
        return {"distance": self.distance.astype(np.float32),
                "uncertainty": self.uncertainty.astype(np.float32),
                "bounds": self.bounds, "truncation": self.truncation}


def _grid_from_function(
    function: Callable, resolution: int, extent: float, chunk: int, truncation: float
) -> ScaffoldGrid:
    axes = [np.linspace(-extent, extent, resolution, dtype=np.float32)] * 3
    queries = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    distances, uncertainties = [], []
    for start in range(0, len(queries), chunk):
        result = function(queries[start:start + chunk])
        if isinstance(result, dict):
            distance = _numpy(result["distance"]).reshape(-1)
            if "uncertainty" in result:
                uncertainty = _numpy(result["uncertainty"]).reshape(-1)
            elif "log_scale" in result:
                uncertainty = np.exp(_numpy(result["log_scale"]).reshape(-1))
            else:
                uncertainty = np.full_like(distance, 0.0025)
        else:
            distance = _numpy(result).reshape(-1)
            uncertainty = np.full_like(distance, 0.0025)
        if len(distance) != len(queries[start:start + chunk]):
            raise ValueError("field callable must return one signed distance per query")
        distances.append(np.clip(distance, -truncation, truncation))
        uncertainties.append(uncertainty)
    shape = (resolution,) * 3
    return ScaffoldGrid(np.concatenate(distances).reshape(shape),
                        np.concatenate(uncertainties).reshape(shape),
                        np.asarray([[-extent] * 3, [extent] * 3]), truncation)


def _make_field(model, encoded, cfg: dict, field_override=None) -> ScaffoldGrid:
    import torch

    options = _options(cfg)
    if isinstance(field_override, ScaffoldGrid):
        return field_override
    if isinstance(field_override, dict):
        distance = field_override["distance"]
        sigma = field_override.get("uncertainty", np.full_like(distance, 0.0025))
        return ScaffoldGrid(distance, sigma, field_override["bounds"], float(field_override.get("truncation", 0.1)))
    device = encoded["token_xyz"].device
    dtype = encoded["token_xyz"].dtype
    field_module = getattr(model, "field", None)
    with torch.no_grad():
        context = field_module.context(encoded) if field_override is None and field_module is not None else None

    def predict(query):
        tensor = torch.as_tensor(query, device=device, dtype=dtype).unsqueeze(0)
        with torch.no_grad():
            return field_module(encoded, tensor, context=context) if context is not None else model.scaffold(encoded, tensor)

    return _grid_from_function(
        field_override if callable(field_override) else predict,
        int(options.get("resolution", 32)), float(options.get("field_extent", 1.25)),
        int(options.get("field_chunk", 2048)), float(getattr(field_module, "truncation", 0.1)),
    )


def _correspondences(pair: dict[str, Any], options: dict) -> dict[str, Any] | None:
    """Take the union of row and column maxima without re-normalizing mass."""
    source = np.asarray(pair["source_xyz"], dtype=np.float64)
    target = np.asarray(pair["target_xyz"], dtype=np.float64)
    weights = np.asarray(pair["weights"], dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("pair coordinates must have shape [K,3]")
    if weights.shape != (len(source), len(target)):
        raise ValueError("pair weights must have shape [source_count,target_count]")
    if len(source) < 3 or len(target) < 3:
        return None
    if not (np.isfinite(weights).all() and np.isfinite(source).all() and np.isfinite(target).all()) or (weights < 0).any():
        return None
    # Includes non-mutual alternatives, useful on independently sampled mating
    # surfaces. Geometric hypothesis scoring subsequently rejects bad matches.
    rows = np.arange(len(source))
    cols = np.arange(len(target))
    indices = np.unique(np.concatenate([
        np.stack([rows, weights.argmax(axis=1)], axis=1),
        np.stack([weights.argmax(axis=0), cols], axis=1),
    ]), axis=0)
    values = weights[indices[:, 0], indices[:, 1]]
    keep = values >= float(options.get("min_correspondence_weight", 1e-3))
    indices, values = indices[keep], values[keep]
    if len(values) < 3 or float(values.sum()) < float(options.get("min_pair_mass", 0.05)):
        return None
    order = np.argsort(-values, kind="stable")
    indices, values = indices[order], values[order]
    support_confidence = float(values.sum() / max(len(np.unique(indices[:, 0])), len(np.unique(indices[:, 1]))))
    if 'source_matchability' in pair and 'target_matchability' in pair:
        a, b = np.asarray(pair['source_matchability']), np.asarray(pair['target_matchability'])
        if (a.shape != (len(source),) or b.shape != (len(target),)
                or not np.isfinite(a).all() or not np.isfinite(b).all()
                or (a < 0).any() or (a > 1).any() or (b < 0).any() or (b > 1).any()):
            raise ValueError('Matchability must be finite per-point probabilities in [0,1]')
        # Use retained matched-vs-dustbin mass for contact existence confidence.
        # Individual correspondence probabilities remain the actual fit weights;
        # their spread over several valid independent samples is not a dustbin.
        support_confidence = min(float(a[np.unique(indices[:, 0])].mean()),
                                 float(b[np.unique(indices[:, 1])].mean()))
    return {"i": int(pair["i"]), "j": int(pair["j"]),
            "source": source[indices[:, 0]], "target": target[indices[:, 1]],
            "weights": values, "source_indices": indices[:, 0],
            "target_indices": indices[:, 1],
            # Exterior/dustbin points are not evidence against a small contact.
            # Preserve absolute probability mass: weak selected weights still
            # produce weak confidence, without renormalizing them to sum to one.
            "confidence": float(np.clip(support_confidence, 0, 1))}


def _pair_candidates(correspondences: dict, options: dict) -> list[dict]:
    source, target, weights = (correspondences[k] for k in ("source", "target", "weights"))
    limit = min(16, max(1, int(options.get("max_pair_candidates", 16))))
    keep_count = min(4, max(1, int(options.get("keep_pair_candidates", 4))))
    subsets = [np.arange(len(weights))]
    # Broad fit plus local neighborhoods around spatially distinct strong seeds.
    used_seeds: list[int] = []
    for seed in range(len(weights)):
        if len(subsets) >= limit:
            break
        if used_seeds and np.min(np.linalg.norm(source[used_seeds] - source[seed], axis=1)) < 0.01:
            continue
        used_seeds.append(seed)
        neighborhood = np.argsort(np.linalg.norm(source - source[seed], axis=1), kind="stable")
        subsets.append(neighborhood[:min(12, len(neighborhood))])
    threshold = float(options.get("inlier_threshold", 0.04))
    candidates = []
    for subset in subsets:
        fit = weighted_kabsch(source[subset], target[subset], weights[subset],
                              min_mass=float(options.get("min_pair_mass", 0.05)))
        if not fit["valid"]:
            continue
        residual = np.linalg.norm(source @ fit["rotation"].T + fit["translation"] - target, axis=1)
        inliers = residual <= threshold
        if np.unique(correspondences["source_indices"][inliers]).size < 3 or np.unique(correspondences["target_indices"][inliers]).size < 3:
            continue
        refined = weighted_kabsch(source[inliers], target[inliers], weights[inliers],
                                  min_mass=float(options.get("min_pair_mass", 0.05)))
        if not refined["valid"]:
            continue
        residual = np.linalg.norm(source @ refined["rotation"].T + refined["translation"] - target, axis=1)
        supported_mass = float(weights[residual <= threshold].sum())
        refined["score"] = float(np.sum(weights * np.minimum(residual**2, threshold**2)) / weights.sum())
        refined["supported_mass"] = supported_mass
        candidates.append(refined)
    candidates.sort(key=lambda value: (value["score"], -value["supported_mass"]))
    distinct: list[dict] = []
    for candidate in candidates:
        if any(np.linalg.norm(candidate["rotation"] - other["rotation"]) < 0.05
               and np.linalg.norm(candidate["translation"] - other["translation"]) < 0.01 for other in distinct):
            continue
        distinct.append(candidate)
        if len(distinct) >= keep_count:
            break
    return distinct


def _compose_tree(edges: list[tuple[int, int]], candidates: tuple[dict, ...], count: int, anchor: int):
    rotation = np.repeat(np.eye(3)[None], count, axis=0)
    translation = np.zeros((count, 3))
    reached = {anchor}
    while len(reached) < count:
        previous = len(reached)
        for (i, j), candidate in zip(edges, candidates):
            relative_r, relative_t = candidate["rotation"], candidate["translation"]
            # Candidate maps local i -> local j; Ri = Rj Rij.
            if j in reached and i not in reached:
                rotation[i] = rotation[j] @ relative_r
                translation[i] = rotation[j] @ relative_t + translation[j]
                reached.add(i)
            elif i in reached and j not in reached:
                rotation[j] = rotation[i] @ relative_r.T
                translation[j] = translation[i] - rotation[j] @ relative_t
                reached.add(j)
        if len(reached) == previous:
            return None
    return rotation, translation


def _robust_square(norm: np.ndarray, delta: float) -> np.ndarray:
    return np.where(norm <= delta, norm**2, 2 * delta * norm - delta**2)


def _score(poses, correspondences, exterior, exterior_weights, field, options):
    rotations, translations = poses
    delta = float(options.get("huber_delta", 0.02))
    contact_scores = []
    contact_squared_errors = []
    contact_rms = []
    contact_masses = []
    for pair in correspondences:
        i, j = pair["i"], pair["j"]
        residual = pair["source"] @ rotations[i].T + translations[i] - pair["target"] @ rotations[j].T - translations[j]
        norm = np.linalg.norm(residual, axis=1)
        weights = pair["weights"]
        contact_scores.append(float(np.sum(weights * _robust_square(norm, delta))))
        contact_squared_errors.append(float(np.sum(weights * norm**2)))
        contact_masses.append(float(weights.sum()))
        contact_rms.append(float(np.sqrt(np.sum(weights * norm**2) / weights.sum())))
    # One normalization for the contact family preserves relative confidence
    # between edges. Normalizing every edge separately would give a nearly
    # unmatched pair the same influence as a well-supported mating surface.
    total_contact_mass = max(float(np.sum(contact_masses)), 1e-12)
    contact = float(np.sum(contact_scores) / total_contact_mass)
    prior_scores = []
    if field is not None and exterior is not None:
        for i, points in enumerate(exterior):
            moved = points @ rotations[i].T + translations[i]
            distance, confidence, _ = field.sample(moved)
            probability = np.ones(len(points)) if exterior_weights is None else exterior_weights[i]
            # Normalize using possible exterior mass, not confidence mass: an
            # uncertain scaffold genuinely exerts less force on the assembly.
            denominator = max(float(np.sum(probability)), 1e-8)
            prior_scores.append(float(np.sum(probability * confidence * _robust_square(np.abs(distance), delta)) / denominator))
    prior = float(np.mean(prior_scores)) if prior_scores else 0.0
    coefficient = min(0.25, max(0.0, float(options.get("prior_weight", 0.25))))
    return contact + coefficient * prior, {
        "contact_score": contact, "prior_score": prior,
        "contact_rms": float(np.sqrt(np.sum(contact_squared_errors) / total_contact_mass)),
        "pair_contact_rms": contact_rms, "pair_contact_mass": contact_masses,
    }


def _refine(poses, anchor, correspondences, exterior, exterior_weights, field, options):
    rotations, translations = (value.copy() for value in poses)
    count = len(rotations)
    moving = [i for i in range(count) if i != anchor]
    columns = {fragment: 6 * index for index, fragment in enumerate(moving)}
    dimensions = 6 * len(moving)
    delta = float(options.get("huber_delta", 0.02))
    coefficient = min(0.25, max(0.0, float(options.get("prior_weight", 0.25))))
    damping = float(options.get("damping", 1e-4))
    iterations = min(5, max(0, int(options.get("refinement_iterations", 5))))
    accepted = 0
    total_contact_mass = max(float(sum(pair["weights"].sum() for pair in correspondences)), 1e-12)
    for _ in range(iterations):
        jacobians, residuals = [], []
        for pair in correspondences:
            i, j = pair["i"], pair["j"]
            source = pair["source"] @ rotations[i].T + translations[i]
            target = pair["target"] @ rotations[j].T + translations[j]
            residual = source - target
            norm = np.linalg.norm(residual, axis=1)
            irls = np.minimum(1.0, delta / np.maximum(norm, 1e-12))
            weight = np.sqrt(pair["weights"] / total_contact_mass * irls)
            jacobian = np.zeros((len(source), 3, dimensions))
            if i != anchor:
                offset = columns[i]
                jacobian[:, :, offset:offset + 3] = -skew(source)
                jacobian[:, :, offset + 3:offset + 6] = np.eye(3)
            if j != anchor:
                offset = columns[j]
                jacobian[:, :, offset:offset + 3] = skew(target)
                jacobian[:, :, offset + 3:offset + 6] = -np.eye(3)
            jacobians.append((jacobian * weight[:, None, None]).reshape(-1, dimensions))
            residuals.append((residual * weight[:, None]).reshape(-1))
        if field is not None and exterior is not None and coefficient > 0:
            for i in moving:
                source = exterior[i] @ rotations[i].T + translations[i]
                distance, confidence, gradient = field.sample(source)
                probability = np.ones(len(source)) if exterior_weights is None else exterior_weights[i]
                irls = np.minimum(1.0, delta / np.maximum(np.abs(distance), 1e-12))
                weight = np.sqrt(coefficient * probability * confidence * irls / max(float(probability.sum()), 1e-8) / count)
                jacobian = np.zeros((len(source), dimensions))
                offset = columns[i]
                jacobian[:, offset:offset + 3] = np.einsum("ni,nij->nj", gradient, -skew(source))
                jacobian[:, offset + 3:offset + 6] = gradient
                jacobians.append(jacobian * weight[:, None])
                residuals.append(distance * weight)
        jacobian = np.concatenate(jacobians)
        residual = np.concatenate(residuals)
        if not (np.isfinite(jacobian).all() and np.isfinite(residual).all()):
            break
        normal = jacobian.T @ jacobian
        try:
            step = np.linalg.solve(normal + damping * np.eye(dimensions), -(jacobian.T @ residual))
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(step).all():
            break
        # Trust region in normalized units avoids single ill-conditioned jumps.
        magnitude = max(1.0, float(np.max(np.abs(step))) / 0.15)
        step /= magnitude
        old_score, _ = _score((rotations, translations), correspondences, exterior, exterior_weights, field, options)
        candidate_r, candidate_t = rotations.copy(), translations.copy()
        for i in moving:
            offset = columns[i]
            update = so3_exp(step[offset:offset + 3])
            candidate_r[i] = update @ rotations[i]
            candidate_t[i] = update @ translations[i] + step[offset + 3:offset + 6]
        new_score, _ = _score((candidate_r, candidate_t), correspondences, exterior, exterior_weights, field, options)
        if new_score < old_score:
            rotations, translations = candidate_r, candidate_t
            damping = max(damping / 2, 1e-7)
            accepted += 1
        else:
            damping *= 10
        if np.linalg.norm(step) < 1e-8:
            break
    return (rotations, translations), accepted


def solve_from_matches(
    matches: list[dict[str, Any]], num_fragments: int, anchor_index: int = 0,
    exterior_points: list[np.ndarray] | None = None,
    exterior_weights: list[np.ndarray] | None = None,
    field: ScaffoldGrid | None = None, cfg: dict | None = None,
) -> dict[str, Any]:
    """Model-independent assembly API for confidence-weighted pair matches.

    Each pair contains ``i``, ``j``, ``source_xyz[K,3]``, ``target_xyz[L,3]``
    and ``weights[K,L]`` in [0,1]. The returned transforms operate in normalized
    fragment coordinates. Failure is explicit and has ``None`` transforms.
    """
    if num_fragments not in (2, 3) or not 0 <= anchor_index < num_fragments:
        raise ValueError("solver requires two or three fragments and a valid reference index")
    options = _options(cfg)
    if exterior_points is not None:
        exterior_points = [np.asarray(points, dtype=np.float64) for points in exterior_points]
        if len(exterior_points) != num_fragments:
            raise ValueError("exterior point arrays must match the fragment count")
    if exterior_weights is not None:
        if exterior_points is None or len(exterior_weights) != num_fragments:
            raise ValueError("exterior probabilities require one matching point array per fragment")
        exterior_weights = [np.asarray(weights, dtype=np.float64) for weights in exterior_weights]
        for points, weights in zip(exterior_points, exterior_weights):
            if weights.shape != (len(points),) or (weights < 0).any() or not np.isfinite(weights).all():
                raise ValueError("invalid exterior probabilities")
    correspondences, candidates, seen, pair_diagnostics = [], {}, set(), {}
    for pair in matches:
        i, j = int(pair["i"]), int(pair["j"])
        if not (0 <= i < j < num_fragments):
            raise ValueError("each pair must have 0 <= i < j < num_fragments")
        if (i, j) in seen:
            raise ValueError("duplicate fragment pair")
        seen.add((i, j))
        selected = _correspondences(pair, options)
        key = f"{i}-{j}"
        pair_diagnostics[key] = {"source_points": len(pair['source_xyz']), "target_points": len(pair['target_xyz']),
                                 "raw_mass": float(np.asarray(pair['weights']).sum()),
                                 "selected_correspondences": 0, "candidate_count": 0,
                                 "reason": "insufficient_correspondence_support"}
        if selected is None:
            continue
        correspondences.append(selected)
        fits = _pair_candidates(selected, options)
        pair_diagnostics[key].update(selected_correspondences=len(selected['weights']),
                                     unique_source_points=len(np.unique(selected['source_indices'])),
                                     unique_target_points=len(np.unique(selected['target_indices'])),
                                     selected_mass=float(selected['weights'].sum()), confidence=selected['confidence'],
                                     candidate_count=len(fits), reason=None if fits else 'degenerate_or_inconsistent_support')
        if fits:
            candidates[(i, j)] = fits
    diagnostics = {"pair_candidate_counts": {f"{i}-{j}": len(values) for (i, j), values in candidates.items()},
                   "pairs": pair_diagnostics,
                   "anchor_index": int(anchor_index), "hypotheses": 0, "refinement_accepted_steps": 0}
    failure = {"status": "failed", "confidence": 0.0, "rotations": None, "translations": None,
               "anchor_index": int(anchor_index), "diagnostics": diagnostics}
    hypotheses = []
    # For three fragments, these are precisely the three possible spanning
    # trees. Missing/non-contacting edges are naturally skipped.
    for edges in combinations(candidates, num_fragments - 1):
        for choices in product(*(candidates[edge] for edge in edges)):
            poses = _compose_tree(list(edges), choices, num_fragments, anchor_index)
            if poses is not None:
                score, detail = _score(poses, correspondences, exterior_points, exterior_weights, field, options)
                hypotheses.append((score, poses, detail))
    diagnostics["hypotheses"] = len(hypotheses)
    if not hypotheses:
        return {**failure, "reason": "no_valid_connected_assembly"}
    hypotheses.sort(key=lambda item: item[0])
    finalists = []
    for initial_score, poses, initial_detail in hypotheses[:min(4, max(1, int(options.get("refine_candidates", 4))))]:
        refined, accepted = _refine(poses, anchor_index, correspondences, exterior_points, exterior_weights, field, options)
        score, detail = _score(refined, correspondences, exterior_points, exterior_weights, field, options)
        finalists.append((score, refined, detail, accepted, initial_score, initial_detail))
    score, (rotations, translations), detail, accepted, initial_score, initial_detail = min(finalists, key=lambda item: item[0])
    diagnostics.update(detail)
    diagnostics.update({"score": float(score), "refinement_accepted_steps": accepted,
                        "pre_refinement_score": float(initial_score),
                        "pre_refinement_contact_score": initial_detail["contact_score"],
                        "pre_refinement_prior_score": initial_detail["prior_score"],
                        "pre_refinement_contact_rms": initial_detail["contact_rms"]})
    if not (np.isfinite(rotations).all() and np.isfinite(translations).all()):
        return {**failure, "reason": "nonfinite_refinement"}
    # A weak bridge must not be hidden by a strong edge. Non-contacting pairs
    # cannot lower confidence merely by existing in a complete three-part set.
    edge_confidence = {(pair['i'], pair['j']): pair['confidence'] for pair in correspondences}
    mass_confidence = max(min(edge_confidence[edge] for edge in edges)
                          for edges in combinations(candidates, num_fragments - 1))
    diagnostics['contact_confidence'] = float(mass_confidence)
    confidence = float(mass_confidence * np.exp(-detail["contact_rms"] / max(float(options.get("huber_delta", 0.02)), 1e-6)))
    status = "ok" if confidence >= float(options.get("min_assembly_confidence", 0.1)) else "low_confidence"
    return {"status": status, "reason": None if status == "ok" else "weak_or_inconsistent_contact_support",
            "confidence": confidence, "rotations": rotations, "translations": translations,
            "anchor_index": int(anchor_index), "diagnostics": diagnostics}


def solve_assembly(model, batch: dict, cfg: dict, condition: str = "predicted", field_override=None) -> dict:
    """Run XYZ-only prediction and rigid assembly for exactly one sample.

    ``gt`` is an evaluation-only oracle: it requires an explicit signed-field
    override. ``contact_only`` disables both matcher scaffold conditioning and
    all scaffold scoring/refinement. No training labels are inspected here.
    """
    import torch

    if condition not in {"predicted", "contact_only", "gt", "perturbed"}:
        raise ValueError(f"unknown scaffold condition: {condition}")
    if condition == "gt" and field_override is None:
        raise ValueError("GT evaluation requires an explicit ground-truth signed field")
    if condition != "gt" and field_override is not None:
        raise ValueError("field_override is permitted only in explicitly labeled GT evaluation")
    points, mask, anchor = batch["points"], batch["fragment_mask"], batch["anchor_index"]
    if points.ndim != 4 or points.shape[0] != 1:
        raise ValueError("solve_assembly accepts exactly one padded sample")
    count = int(_numpy(mask)[0].sum())
    if not np.array_equal(_numpy(mask)[0].astype(bool), np.arange(points.shape[1]) < count):
        raise ValueError("active fragments must precede padding")
    anchor_value = int(_numpy(anchor).reshape(-1)[0])
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            encoded = model.encode(points, mask, anchor)
            field = None if condition == "contact_only" else _make_field(model, encoded, cfg, field_override)
            if condition == "perturbed":
                # A reproducible, deliberately misleading signed offset plus
                # spatial roll; uncertainty remains unchanged for a hard test.
                field = ScaffoldGrid(np.clip(np.roll(field.distance, 3, axis=0) + 0.035,
                                              -field.truncation, field.truncation),
                                     field.uncertainty.copy(), field.bounds.copy(), field.truncation)
            prior_override = None
            if condition in {"gt", "perturbed"}:
                anchor_queries = encoded["token_xyz"][:, anchor_value]
                conditioning_queries = getattr(model, "conditioning_queries", None)
                query = anchor_queries if conditioning_queries is None else torch.cat([
                    conditioning_queries[None].to(anchor_queries), anchor_queries], dim=1)
                values, confidence, _ = field.sample(_numpy(query)[0])
                # Invert the confidence conversion to retain interpolated sigma.
                sigma = np.maximum(0.01 * (1 / confidence - 1), np.exp(-6))
                values = torch.as_tensor(values, device=query.device, dtype=query.dtype)[None, :, None]
                log_scale = torch.as_tensor(np.log(sigma), device=query.device, dtype=query.dtype)[None, :, None]
                prior_override = torch.cat([query, values, log_scale], dim=-1)
            predictions = model.match(encoded, use_scaffold=condition != "contact_only", prior_override=prior_override)
            matches = []
            for pair in predictions:
                valid = pair.get("valid")
                if valid is not None and not bool(_numpy(valid).reshape(-1)[0]):
                    continue
                matches.append({"i": int(pair["i"]), "j": int(pair["j"]),
                                "source_xyz": _numpy(pair["source_xyz"])[0],
                                "target_xyz": _numpy(pair["target_xyz"])[0],
                                "weights": _numpy(pair["weights"])[0]})
                for name in ('source_matchability', 'target_matchability'):
                    if name in pair:
                        matches[-1][name] = _numpy(pair[name])[0]
            # Only predicted original-surface probabilities enter the solver.
            probabilities = torch.sigmoid(-encoded["fracture_logits"])
            exterior = [_numpy(points)[0, i] for i in range(count)]
            exterior_weights = [_numpy(probabilities)[0, i] for i in range(count)]
            # Bound numerical work without changing the neural input resolution.
            for i in range(count):
                selected = np.linspace(0, len(exterior[i]) - 1, min(256, len(exterior[i])), dtype=int)
                exterior[i], exterior_weights[i] = exterior[i][selected], exterior_weights[i][selected]
        result = solve_from_matches(matches, count, anchor_value, exterior, exterior_weights, field, cfg)
        result["condition"] = condition
        result["scaffold"] = None if field is None else field.as_dict()
        return result
    finally:
        model.train(was_training)
