"""Classical (non-learned) point-cloud registration helpers.

Shared by ``scripts/classical_registration_baseline.py`` (the measurement CLI) and
``models.pose.ClassicalRegistrationPoseEstimator`` (the Stage-3 module), so the
registration math lives in exactly one place.

All routines use the repo's row-vector convention: ``aligned = source @ R.T + t``,
matching ``models.pose.apply_fragment_transforms`` and the dataset's stored
``(align_rotations, align_translations)``.

Measured on real ShapeNet fragments vs the ground-truth object: rotation p50 6.0
deg overall, 0.08 deg on large fragments, with no training at all.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from models.pose import kabsch_align

try:  # optional dependency; the multi-init ICP fallback needs no extras
    import open3d as o3d  # type: ignore

    HAS_OPEN3D = True
except Exception:  # pragma: no cover
    HAS_OPEN3D = False


def resolve_method(method: str = "auto") -> str:
    """Pick the registration backend: FPFH+RANSAC when open3d is available."""
    if method == "auto":
        return "ransac_fpfh" if HAS_OPEN3D else "multi_icp"
    if method == "ransac_fpfh" and not HAS_OPEN3D:
        return "multi_icp"
    return method


def chamfer(a: torch.Tensor, b: torch.Tensor) -> float:
    """Symmetric mean nearest-neighbour distance between (M,3) and (K,3)."""
    d = torch.cdist(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
    return float(d.min(dim=1).values.mean() + d.min(dim=0).values.mean())


def subsample(points: torch.Tensor, n: int) -> torch.Tensor:
    if points.shape[0] <= n:
        return points
    idx = torch.randperm(points.shape[0])[:n]
    return points[idx]


def icp_once(src: torch.Tensor, tgt: torch.Tensor, iters: int):
    """Point-to-point ICP from the current frame. Returns (R, t, aligned)."""
    rotation = torch.eye(3, dtype=src.dtype)
    translation = torch.zeros(3, dtype=src.dtype)
    current = src
    for _ in range(iters):
        dist = torch.cdist(current, tgt)
        matched = tgt[dist.argmin(dim=1)]
        step_r, step_t = kabsch_align(current.unsqueeze(0), matched.unsqueeze(0))
        step_r, step_t = step_r[0], step_t[0]
        current = current @ step_r.transpose(-1, -2) + step_t
        rotation = step_r @ rotation
        translation = step_r @ translation + step_t
    return rotation, translation, current


def _random_rotation(dtype: torch.dtype) -> torch.Tensor:
    q = torch.randn(4)
    q = q / q.norm()
    w, x, y, z = q
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=dtype,
    )


def register_multi_icp(
    src: torch.Tensor,
    tgt: torch.Tensor,
    inits: int = 30,
    iters: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi-init ICP: try `inits` random initial rotations, keep lowest-Chamfer.

    Dependency-free fallback. Weaker than FPFH+RANSAC (it only explores the
    basin of convergence) but always runnable.
    """
    best = None
    best_cd = float("inf")
    for k in range(inits):
        r0 = torch.eye(3, dtype=src.dtype) if k == 0 else _random_rotation(src.dtype)
        src0 = src @ r0.transpose(-1, -2)
        r_icp, t_icp, current = icp_once(src0, tgt, iters)
        cd = chamfer(current, tgt)
        if cd < best_cd:
            best_cd = cd
            best = (r_icp @ r0, t_icp)
    return best[0], best[1]


def register_fpfh(
    src_np: np.ndarray,
    tgt_np: np.ndarray,
    voxel: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Open3D FPFH + RANSAC global registration, refined by ICP.

    Returns (R (3,3), t (3,)) with ``aligned = src @ R.T + t``.
    """
    if not HAS_OPEN3D:
        raise RuntimeError("register_fpfh requires open3d (pip install open3d)")
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_np.astype(np.float64))
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_np.astype(np.float64))
    src_d = src.voxel_down_sample(voxel)
    tgt_d = tgt.voxel_down_sample(voxel)
    normal_r = voxel * 2.0
    feat_r = voxel * 5.0
    for pcd in (src_d, tgt_d):
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_r, max_nn=30))
    src_f = o3d.pipelines.registration.compute_fpfh_feature(
        src_d, o3d.geometry.KDTreeSearchParamHybrid(radius=feat_r, max_nn=100)
    )
    tgt_f = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_d, o3d.geometry.KDTreeSearchParamHybrid(radius=feat_r, max_nn=100)
    )
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, src_f, tgt_f, True, voxel * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    icp = o3d.pipelines.registration.registration_icp(
        src, tgt, voxel * 1.0, ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    m = np.asarray(icp.transformation)
    return torch.from_numpy(m[:3, :3]).float(), torch.from_numpy(m[:3, 3]).float()


def register_fragment(
    source: torch.Tensor,
    target: torch.Tensor,
    method: str = "auto",
    voxel: float = 0.05,
    icp_inits: int = 30,
    icp_iters: int = 30,
    icp_src_points: int = 256,
    icp_tgt_points: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Register one fragment (N,3) onto a target cloud (M,3) -> (R (3,3), t (3,)).

    Dispatches to FPFH+RANSAC+ICP or the multi-init ICP fallback. CPU-only.
    """
    resolved = resolve_method(method)
    source = source.detach().to("cpu", torch.float32)
    target = target.detach().to("cpu", torch.float32)
    if resolved == "ransac_fpfh":
        return register_fpfh(source.numpy(), target.numpy(), voxel)
    return register_multi_icp(
        subsample(source, icp_src_points),
        subsample(target, icp_tgt_points),
        icp_inits,
        icp_iters,
    )
