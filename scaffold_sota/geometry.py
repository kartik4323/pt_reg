"""Common rigid frames and geometry metrics for all study recipients."""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from reassembly.geometry import deterministic_fps, normalize_fragments, weighted_kabsch, export_transforms


def as_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def chamfer(a, b):
    """Average of two mean *unsquared* Euclidean nearest-neighbor distances."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1:] != (3,) or b.shape[1:] != (3,) or not len(a) or not len(b):
        raise ValueError("Chamfer requires nonempty XYZ arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Chamfer requires finite coordinates")
    return float((cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()) / 2)


def validate_poses(rotations, translations, count, atol=1e-4):
    rotations, translations = as_numpy(rotations).astype(float), as_numpy(translations).astype(float)
    if rotations.shape != (count, 3, 3) or translations.shape != (count, 3):
        raise ValueError("Pose count/shape does not match observed fragments")
    if not np.isfinite(rotations).all() or not np.isfinite(translations).all():
        raise ValueError("Nonfinite pose")
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=atol) or not np.allclose(np.linalg.det(rotations), 1, atol=atol):
        raise ValueError("Predicted rotations are not proper rigid transforms")
    return rotations, translations


def reference_gauge(rotations, translations, anchor_index):
    """Remove a single common SE(3) gauge using only the predicted anchor pose."""
    rotations, translations = validate_poses(rotations, translations, len(rotations))
    anchor_rotation, anchor_translation = rotations[anchor_index], translations[anchor_index]
    return anchor_rotation.T[None] @ rotations, (translations - anchor_translation) @ anchor_rotation


def transform_points(points, rotations, translations):
    return np.einsum("fni,fji->fnj", as_numpy(points), as_numpy(rotations)) + as_numpy(translations)[:, None]


def fixed_query_bank(count=512, extent=1.5, device=None, dtype=torch.float32):
    """Input-only multiscale spatial support; no predicted/GT surface sampling.

    Identical support across every field/content condition in normalized anchor
    coordinates: half near the centered reference, half in the wider volume.
    """
    if count < 8 or extent <= 0:
        raise ValueError("Invalid spatial field support")
    bank = torch.quasirandom.SobolEngine(3, scramble=True, seed=42).draw(count)
    scales = torch.full((count, 1), float(extent))
    scales[:count // 2] = float(extent) / 3
    return ((bank * 2 - 1) * scales).to(device=device, dtype=dtype)
