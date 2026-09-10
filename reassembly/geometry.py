"""Coordinate contracts and rigid geometry for the versioned assembly pipeline.

All transforms use column-rotation/row-array convention: ``y = x @ R.T + t``.
The network never sees independently rescaled fragments.  Its output frame is
the centered reference fragment, in the one common normalization scale.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _xyz(value: Any, name: str = "points") -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 3:
        raise ValueError(f"{name} must contain at least three XYZ points")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return value


def deterministic_fps(points: np.ndarray, count: int) -> np.ndarray:
    """Farthest-point indices with geometric tie-breaking and repeat padding.

    Lexicographic ordering makes the selected coordinates independent of input
    point order (indices still refer to the caller's original array).  Duplicate
    coordinates can have different indices, which has no geometric effect.
    """
    points = _xyz(points)
    if count < 1:
        raise ValueError("sample count must be positive")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    ordered = points[order]
    centered = ordered - ordered.mean(axis=0)
    selected = np.empty(min(count, len(points)), dtype=np.int64)
    distance = np.full(len(points), np.inf)
    available = np.ones(len(points), dtype=bool)
    current = int(np.argmax(np.einsum("ij,ij->i", centered, centered)))
    for index in range(len(selected)):
        selected[index] = current
        available[current] = False
        delta = ordered - ordered[current]
        distance = np.minimum(distance, np.einsum("ij,ij->i", delta, delta))
        current = int(np.argmax(np.where(available, distance, -1.0)))
    selected = order[selected]
    if len(selected) < count:
        selected = np.resize(selected, count)
    return selected


def normalize_fragments(
    fragments: Sequence[np.ndarray], num_points: int = 1024, seed: int = 42
) -> dict[str, Any]:
    """Prepare two or three complete XYZ fragments without losing their scale.

    ``seed`` is retained in the public interface for reproducibility metadata;
    geometric FPS is deterministic and does not consume random state.
    """
    if len(fragments) not in (2, 3):
        raise ValueError("assembly requires exactly two or three complete fragments")
    if num_points < 3:
        raise ValueError("num_points must be at least three")
    original = [_xyz(points, f"fragment {i}").copy() for i, points in enumerate(fragments)]
    # Sorting before reductions avoids point-permutation-dependent roundoff.
    centroids = np.stack([
        p[np.lexsort((p[:, 2], p[:, 1], p[:, 0]))].mean(axis=0) for p in original
    ])
    centered = [p - center for p, center in zip(original, centroids)]
    radii = np.asarray([np.linalg.norm(p, axis=1).max() for p in centered])
    if (radii <= np.finfo(np.float64).eps).any():
        raise ValueError("each fragment must have nonzero geometric extent")
    scale = float(radii.sum())
    rms = np.asarray([np.sqrt(np.mean(np.sum(p * p, axis=1))) for p in centered])
    # Treat numerical ties as ties; choose the lowest original fragment index.
    tied = np.flatnonzero(np.isclose(rms, rms.max(), rtol=1e-12, atol=1e-15))
    anchor = int(tied[0])
    indices = [deterministic_fps(p, num_points) for p in centered]
    points = np.zeros((3, num_points, 3), dtype=np.float32)
    for i, (p, selected) in enumerate(zip(centered, indices)):
        points[i] = p[selected] / scale
    return {
        "points": points,
        "fragment_mask": np.arange(3) < len(original),
        "anchor_index": anchor,
        "centroids": centroids,
        "scale": scale,
        "original_points": original,
        "selected_indices": indices,
        "rms_radii": rms,
        "bounding_radii": radii,
        "seed": int(seed),
    }


def weighted_kabsch(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
    min_mass: float = 1e-8,
    rank_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Fit a proper rigid transform; planar support is valid, a line is not.

    Failures carry no fallback transform. Weights are never normalized before
    the minimum-confidence check, so a near-certain dustbin is not a match.
    """
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    failure = {"valid": False, "rotation": None, "translation": None}
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        return {**failure, "reason": "invalid_correspondence_shape"}
    weights = np.ones(len(source)) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(source),):
        return {**failure, "reason": "invalid_weight_shape"}
    if not (np.isfinite(source).all() and np.isfinite(target).all() and np.isfinite(weights).all()):
        return {**failure, "reason": "nonfinite_correspondences"}
    if (weights < 0).any():
        return {**failure, "reason": "negative_correspondence_weights"}
    keep = weights > 0
    source, target, weights = source[keep], target[keep], weights[keep]
    mass = float(weights.sum())
    if len(source) < 3 or mass < min_mass:
        return {**failure, "reason": "insufficient_correspondence_support", "mass": mass}
    normalized = weights / mass
    source_center = normalized @ source
    target_center = normalized @ target
    a, b = source - source_center, target - target_center
    source_spectrum = np.linalg.svd(a * np.sqrt(normalized[:, None]), compute_uv=False)
    target_spectrum = np.linalg.svd(b * np.sqrt(normalized[:, None]), compute_uv=False)
    for spectrum in (source_spectrum, target_spectrum):
        if spectrum[0] <= 1e-12 or spectrum[1] <= rank_tolerance * spectrum[0]:
            return {**failure, "reason": "collinear_correspondence_support", "mass": mass}
    u, singular, vt = np.linalg.svd(a.T @ (normalized[:, None] * b))
    # Cross covariance can degenerate even when both point sets have rank two.
    if singular[0] <= 1e-15 or singular[1] <= rank_tolerance * singular[0]:
        return {**failure, "reason": "degenerate_cross_covariance", "mass": mass}
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = target_center - rotation @ source_center
    residual = source @ rotation.T + translation - target
    rms = float(np.sqrt(np.sum(normalized * np.sum(residual * residual, axis=1))))
    return {"valid": True, "rotation": rotation, "translation": translation,
            "rms": rms, "mass": mass, "support": int(len(source)), "reason": None}


def export_transforms(
    rotations: np.ndarray, translations: np.ndarray, normalization: dict[str, Any]
) -> dict[str, Any]:
    """Compose network poses back into the original reference input's units."""
    original = normalization["original_points"]
    count = len(original)
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    if rotations.shape != (count, 3, 3) or translations.shape != (count, 3):
        raise ValueError("pose count must match the number of input fragments")
    if not (np.isfinite(rotations).all() and np.isfinite(translations).all()):
        raise ValueError("cannot export non-finite poses")
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5):
        raise ValueError("rotations must be orthogonal")
    if not np.allclose(np.linalg.det(rotations), 1, atol=1e-5):
        raise ValueError("rotations must have determinant +1")
    anchor = int(normalization["anchor_index"])
    if not np.allclose(rotations[anchor], np.eye(3), atol=1e-6) or not np.allclose(translations[anchor], 0, atol=1e-6):
        raise ValueError("reference pose must be identity")
    centers = np.asarray(normalization["centroids"], dtype=np.float64)
    scale = float(normalization["scale"])
    exported_rotation = rotations.copy()
    exported_translation = centers[anchor] + scale * translations - np.einsum("fij,fj->fi", rotations, centers)
    exported_rotation[anchor] = np.eye(3)
    exported_translation[anchor] = 0
    transforms = np.repeat(np.eye(4)[None], count, axis=0)
    transforms[:, :3, :3] = exported_rotation
    transforms[:, :3, 3] = exported_translation
    aligned = [np.asarray(p) @ r.T + t for p, r, t in zip(original, exported_rotation, exported_translation)]
    return {"rotations": exported_rotation, "translations": exported_translation,
            "transforms": transforms, "aligned_fragments": aligned,
            "anchor_index": anchor}


def skew(vector: np.ndarray) -> np.ndarray:
    """Cross-product matrix (supports an arbitrary leading batch shape)."""
    vector = np.asarray(vector, dtype=np.float64)
    result = np.zeros(vector.shape[:-1] + (3, 3))
    x, y, z = np.moveaxis(vector, -1, 0)
    result[..., 0, 1], result[..., 0, 2] = -z, y
    result[..., 1, 0], result[..., 1, 2] = z, -x
    result[..., 2, 0], result[..., 2, 1] = -y, x
    return result


def so3_exp(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    cross = skew(vector)
    if angle < 1e-8:
        return np.eye(3) + cross + 0.5 * cross @ cross
    return np.eye(3) + np.sin(angle) / angle * cross + (1 - np.cos(angle)) / angle**2 * cross @ cross
