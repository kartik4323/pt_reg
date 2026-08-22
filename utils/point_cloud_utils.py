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


def deterministic_fps_indices(points: torch.Tensor, count: int) -> torch.Tensor:
    """Deterministic batched farthest-point indices for ``(B,N,3)`` tensors.

    The first point is the one farthest from the cloud centroid, rather than a
    random seed.  It is therefore stable across restarts and suitable for a
    recorded experiment protocol.  The routine is intentionally simple because
    it runs only on Stage-3 target clouds (at most 5,000 points).
    """
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (B,N,3)")
    batch, total, _ = points.shape
    if count <= 0 or count >= total:
        return torch.arange(total, device=points.device).unsqueeze(0).expand(batch, -1)
    indices = torch.empty((batch, count), dtype=torch.long, device=points.device)
    centroid = points.mean(dim=1, keepdim=True)
    current = ((points - centroid).square().sum(dim=-1)).argmax(dim=1)
    minimum_distance = torch.full((batch, total), float("inf"), device=points.device, dtype=points.dtype)
    batch_index = torch.arange(batch, device=points.device)
    for step in range(count):
        indices[:, step] = current
        selected = points[batch_index, current].unsqueeze(1)
        distance = (points - selected).square().sum(dim=-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        current = minimum_distance.argmax(dim=1)
    return indices


def gather_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather batched point indices, preserving ``(B,K,3)`` layout."""
    if points.shape[0] != indices.shape[0]:
        raise ValueError("points and indices must have equal batch size")
    return torch.gather(points, 1, indices[..., None].expand(-1, -1, points.shape[-1]))


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


# ── Occupancy / voxel helpers ─────────────────────────────────────────────────
#
# Used for the coarse shape prior: a fixed-resolution occupancy grid over the
# canonical [-1, 1]^3 frame. Deliberately low resolution -- the prior only has to
# be coarsely right, so it is generated and consumed as a volume, and sampled to
# points when a point-cloud consumer needs one.

def grid_centers(resolution: int, dtype=np.float32) -> np.ndarray:
    """Centers of an R^3 voxel grid over [-1,1]^3, shape (R^3, 3), C-order (x,y,z)."""
    lin = ((np.arange(resolution) + 0.5) / resolution * 2.0 - 1.0).astype(dtype)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)


def points_to_occupancy(pts: np.ndarray, resolution: int = 32) -> np.ndarray:
    """Rasterize points into a boolean R^3 occupancy grid over [-1,1]^3.

    Surface-only (a cell is occupied iff a point falls in it). Use for shells or
    as the fallback when a mesh is not watertight.
    """
    pitch = 2.0 / resolution
    idx = np.floor((np.asarray(pts) + 1.0) / pitch).astype(np.int64)
    keep = ((idx >= 0) & (idx < resolution)).all(axis=1)
    grid = np.zeros((resolution,) * 3, dtype=bool)
    idx = idx[keep]
    if len(idx):
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


def occupancy_from_mesh(mesh, resolution: int = 32, surface_samples: int = 50000) -> np.ndarray:
    """Solid occupancy on a fixed R^3 grid over [-1,1]^3 for a (normalized) mesh.

    Uses trimesh voxelization + interior fill, which needs no ``rtree`` (unlike
    ``mesh.contains``). Falls back to surface-point rasterization when the mesh is
    not watertight -- common for Breaking Bad fragments.
    """
    import trimesh  # local import: keeps this module importable without trimesh

    pitch = 2.0 / resolution
    try:
        vox = mesh.voxelized(pitch=pitch)
        if getattr(mesh, "is_watertight", False):
            try:
                vox = vox.fill()
            except Exception:
                pass
        return points_to_occupancy(np.asarray(vox.points), resolution)
    except Exception:
        pts, _ = trimesh.sample.sample_surface(mesh, surface_samples)
        return points_to_occupancy(np.asarray(pts), resolution)


def occupancy_surface(grid: np.ndarray) -> np.ndarray:
    """Boundary cells of an occupancy grid: occupied cells with an empty 6-neighbour.

    The *filled* interior of a solid grid contains no real surface, so sampling it
    produces points that lie nowhere on the object. Anything that consumes the prior
    as geometry -- registration, FPFH/normals, Chamfer against a surface point cloud
    -- must use this boundary instead.
    """
    occ = grid.astype(bool)
    if not occ.any():
        return occ
    empty_neighbour = np.zeros_like(occ)
    for axis in range(3):
        for shift in (-1, 1):
            rolled = np.roll(occ, shift, axis=axis)
            # cells rolled in from outside the volume count as empty
            idx = [slice(None)] * 3
            idx[axis] = 0 if shift == 1 else occ.shape[axis] - 1
            rolled[tuple(idx)] = False
            empty_neighbour |= ~rolled
    return occ & empty_neighbour


def occupancy_to_points(
    grid: np.ndarray,
    num_points: int,
    jitter: bool = True,
    surface_only: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample ``num_points`` from an occupancy grid -> (num_points, 3).

    This is the bridge that lets every existing point-cloud consumer (the Stage-3
    estimators, the classical registration baseline, chamfer metrics) keep working
    against a volumetric prior without any change.

    ``surface_only=True`` (the default) samples the occupancy *boundary*, which is
    the analogue of the object's surface and the only sensible input for
    registration; set it False to sample the solid interior (e.g. for volumetric
    containment scoring).
    """
    resolution = grid.shape[0]
    pitch = 2.0 / resolution
    source = occupancy_surface(grid) if surface_only else grid.astype(bool)
    occ = np.argwhere(source)
    if len(occ) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    rng = rng or np.random.default_rng()
    pick = rng.integers(0, len(occ), size=num_points)
    centers = (occ[pick].astype(np.float32) + 0.5) * pitch - 1.0
    if jitter:
        centers = centers + rng.uniform(-0.5, 0.5, size=centers.shape).astype(np.float32) * pitch
    return centers.astype(np.float32)


def occupancy_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Intersection-over-union between two boolean occupancy grids."""
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(inter / union) if union else 1.0


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
