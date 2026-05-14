"""
utils/point_cloud_utils.py
──────────────────────────
Core point-cloud manipulation helpers used across the pipeline.
"""

from typing import Optional, Tuple

import numpy as np
import torch


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_point_cloud_np(pts: np.ndarray) -> np.ndarray:
    """Centre to origin, scale to unit sphere. Input: (N,3) float32."""
    pts = pts - pts.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 1e-8:
        pts = pts / scale
    return pts.astype(np.float32)


def normalize_point_cloud(pts: torch.Tensor) -> torch.Tensor:
    """Centre to origin, scale to unit sphere. Input: (N,3) or (B,N,3)."""
    if pts.dim() == 2:
        pts = pts - pts.mean(dim=0, keepdim=True)
        scale = pts.norm(dim=1).max()
    else:  # batched
        pts = pts - pts.mean(dim=1, keepdim=True)
        scale = pts.norm(dim=2).max(dim=1, keepdim=True).values.unsqueeze(-1)
    return pts / (scale + 1e-8)


# ── Sampling ──────────────────────────────────────────────────────────────────

def fps(pts: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Farthest Point Sampling (numpy).
    Returns indices of `num_samples` points from pts (N,3).
    """
    N = pts.shape[0]
    if num_samples >= N:
        return np.arange(N)
    selected = np.zeros(num_samples, dtype=np.int64)
    dist = np.full(N, np.inf)
    current = np.random.randint(0, N)
    for i in range(num_samples):
        selected[i] = current
        d = np.linalg.norm(pts - pts[current], axis=1)
        dist = np.minimum(dist, d)
        current = np.argmax(dist)
    return selected


def random_sample(pts: np.ndarray, num_samples: int,
                  replace: bool = False) -> np.ndarray:
    N = pts.shape[0]
    if num_samples >= N:
        return pts
    idx = np.random.choice(N, num_samples, replace=replace)
    return pts[idx]


def ensure_n_points(pts: np.ndarray, n: int) -> np.ndarray:
    """
    Ensure pts has exactly n points by random sampling (with or w/o replacement).
    """
    N = pts.shape[0]
    if N == n:
        return pts
    if N > n:
        idx = np.random.choice(N, n, replace=False)
        return pts[idx]
    # N < n: oversample with replacement to fill
    extra = n - N
    idx = np.random.choice(N, extra, replace=True)
    return np.concatenate([pts, pts[idx]], axis=0)


# ── Geometric transforms ──────────────────────────────────────────────────────

def random_rotation_matrix() -> np.ndarray:
    """Uniformly random SO(3) rotation matrix (3,3)."""
    q = np.random.randn(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R


def apply_rotation(pts: np.ndarray, R: np.ndarray) -> np.ndarray:
    return (R @ pts.T).T


def random_rotation(pts: np.ndarray) -> np.ndarray:
    return apply_rotation(pts, random_rotation_matrix())


def jitter_points(pts: np.ndarray, std: float = 0.01,
                  clip: float = 0.05) -> np.ndarray:
    noise = np.clip(np.random.randn(*pts.shape) * std, -clip, clip)
    return (pts + noise).astype(np.float32)


def random_translate(pts: np.ndarray, max_shift: float = 0.05) -> np.ndarray:
    shift = np.random.uniform(-max_shift, max_shift, (1, 3)).astype(np.float32)
    return pts + shift


def dropout_points(pts: np.ndarray, prob: float = 0.05,
                   max_fraction: float = 0.10) -> np.ndarray:
    """Randomly drop up to max_fraction of points with probability prob."""
    if np.random.rand() > prob:
        return pts
    n_drop = int(len(pts) * np.random.uniform(0, max_fraction))
    if n_drop == 0:
        return pts
    keep = np.random.choice(len(pts), len(pts) - n_drop, replace=False)
    return pts[keep]


# ── Plane helpers ─────────────────────────────────────────────────────────────

def signed_distance_to_plane(pts: np.ndarray,
                              normal: np.ndarray,
                              offset: float) -> np.ndarray:
    """
    Returns signed distance of each point to the plane defined by
    normal · x = offset.  Shape: (N,)
    """
    return pts @ normal - offset


def split_by_plane(pts: np.ndarray,
                   normal: Optional[np.ndarray] = None,
                   offset: Optional[float] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split pts into two halves by a plane (normal, offset).
    If not provided, generates a random plane through the centroid.
    Returns (part_A, part_B).  Both may have 0 points in degenerate cases.
    """
    if normal is None:
        normal = np.random.randn(3).astype(np.float32)
        normal /= np.linalg.norm(normal) + 1e-8
    if offset is None:
        # plane through centroid with small random shift
        centroid = pts.mean(axis=0)
        offset = float(centroid @ normal) + np.random.uniform(-0.05, 0.05)

    d = signed_distance_to_plane(pts, normal, offset)
    mask_a = d >= 0
    mask_b = ~mask_a
    return pts[mask_a], pts[mask_b]


# ── Tensor conversion ─────────────────────────────────────────────────────────

def to_tensor(pts: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(pts.astype(np.float32))


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
