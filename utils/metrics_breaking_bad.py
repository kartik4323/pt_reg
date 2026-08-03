"""Breaking Bad benchmark metrics, matching the reference protocol.

Implements the four metrics every Breaking Bad paper reports so our numbers are
directly comparable to published ones:

* ``RMSE(R)``  -- root-mean-square error over **Euler-angle triplets in degrees**,
  with the 360-degree wrap. This is a weak, gauge-dependent metric, but it is the
  published convention; we also expose the geodesic angle as an honest secondary.
* ``RMSE(T)``  -- root-mean-square translation error (papers report x1e-2).
* ``PA``       -- Part Accuracy: fraction of parts whose symmetric mean-point
  Chamfer distance between "fragment under predicted pose" and "fragment under GT
  pose" is below ``0.01``.
* ``CD``       -- Chamfer distance of the whole assembly vs ground truth
  (papers report x1e-3).

Two protocol details that are easy to get wrong and that we handle explicitly:

1. **Per-object grouping.** Every metric is averaged over the valid parts of an
   object *first*, then over objects. Pooling all parts from all objects into one
   flat list gives a different (part-count-weighted) number.
2. **Gauge fixing.** Jigsaw / PuzzleFusion++ anchor the prediction to the ground
   truth using the **largest fragment** before measuring, which frees a global
   SE(3); the original Global/LSTM/DGL baselines do not. The two are not
   comparable, so :func:`evaluate_assembly` reports both.

Conventions follow the repo: row-vector transforms, ``placed = points @ R.T + t``
(see ``models.pose.apply_fragment_transforms``), and a rotation that maps a
fragment from its own (recentred) frame into the canonical assembled frame.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch


# --- rotation conversions ---------------------------------------------------

def matrix_to_euler_degrees(rot: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrices -> (..., 3) Euler degrees, scipy ``'xyz'``.

    Matches ``scipy.spatial.transform.Rotation.as_euler('xyz', degrees=True)``,
    which is what the reference Breaking Bad implementation uses. Note scipy's
    LOWERCASE ``'xyz'`` is **extrinsic** (rotations about fixed axes x, then y,
    then z), i.e. ``R = Rz(c) @ Ry(b) @ Rx(a)`` -- not the intrinsic convention.
    """
    rot = rot.to(torch.float64)
    r00, r01, r02 = rot[..., 0, 0], rot[..., 0, 1], rot[..., 0, 2]
    r10, r11, r12 = rot[..., 1, 0], rot[..., 1, 1], rot[..., 1, 2]
    r20, r21, r22 = rot[..., 2, 0], rot[..., 2, 1], rot[..., 2, 2]
    # Extrinsic xyz: r20 = -sin(b); r21 = sa*cb; r22 = ca*cb; r00 = cb*cc; r10 = cb*sc
    sb = (-r20).clamp(-1.0, 1.0)
    b = torch.asin(sb)
    gimbal = sb.abs() > 1.0 - 1e-7
    a = torch.atan2(r21, r22)
    c = torch.atan2(r10, r00)
    # Degenerate (cos b == 0): a and c are not separable; fold into one angle.
    # With c := 0 the matrix reduces to r01 = sa*sb, r11 = ca, so the sign of
    # sin(b) must be carried through (getting this wrong breaks b = -90).
    a_g = torch.atan2(torch.sign(sb) * r01, r11)
    c_g = torch.zeros_like(c)
    a = torch.where(gimbal, a_g, a)
    c = torch.where(gimbal, c_g, c)
    return torch.stack([a, b, c], dim=-1) * (180.0 / math.pi)


def euler_degree_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Absolute per-axis Euler difference in degrees with the 360 wrap applied."""
    diff = (a - b).abs()
    return torch.minimum(diff, 360.0 - diff)


def geodesic_degrees(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Geodesic (true SO(3)) angle in degrees between rotations. (..., 3, 3) -> (...)."""
    delta = pred.transpose(-1, -2).to(torch.float64) @ gt.to(torch.float64)
    trace = delta.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cos) * (180.0 / math.pi)


# --- chamfer ----------------------------------------------------------------

def _symmetric_mean_chamfer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """mean-of-min both directions, for (..., N, 3) vs (..., M, 3) -> (...).

    Uses UNSQUARED distances, matching the ``mean(dist1) + mean(dist2)`` form the
    reference Part-Accuracy code applies before thresholding at 0.01.
    """
    dist = torch.cdist(a, b)
    return dist.min(-1).values.mean(-1) + dist.min(-2).values.mean(-1)


# --- gauge fixing -----------------------------------------------------------

def _largest_part_index(points: torch.Tensor, mask: torch.Tensor) -> int:
    """Index of the 'largest' valid fragment, measured by bounding-box diagonal."""
    extent = (points.max(dim=-2).values - points.min(dim=-2).values).norm(dim=-1)
    extent = torch.where(mask, extent, torch.full_like(extent, -1.0))
    return int(extent.argmax())


def _anchor_to_largest(
    pred_rot: torch.Tensor,
    pred_trans: torch.Tensor,
    gt_rot: torch.Tensor,
    gt_trans: torch.Tensor,
    fragments: torch.Tensor,
    mask: torch.Tensor,
):
    """Remove the global SE(3) gauge by aligning on the largest fragment.

    Composes every predicted pose with the correction that maps the largest
    fragment's predicted placement onto its ground-truth placement. This is the
    Jigsaw / PuzzleFusion++ convention.
    """
    k = _largest_part_index(fragments, mask)
    # correction C such that  C o pred_k == gt_k   (row-vector convention)
    r_corr = gt_rot[k] @ pred_rot[k].transpose(-1, -2)
    t_corr = gt_trans[k] - pred_trans[k] @ r_corr.transpose(-1, -2)
    new_rot = r_corr @ pred_rot
    new_trans = pred_trans @ r_corr.transpose(-1, -2) + t_corr
    return new_rot, new_trans


# --- the metric ------------------------------------------------------------

def evaluate_object(
    fragments: torch.Tensor,
    fragment_mask: torch.Tensor,
    pred_rot: torch.Tensor,
    pred_trans: torch.Tensor,
    gt_rot: torch.Tensor,
    gt_trans: torch.Tensor,
    pa_threshold: float = 0.01,
    anchor_largest: bool = False,
) -> Dict[str, float]:
    """Metrics for ONE object. Shapes: fragments (F,N,3), mask (F,), poses (F,3,3)/(F,3).

    Returns per-object scalars: ``rmse_r`` (deg), ``mae_r`` (deg), ``geodesic_r``
    (deg), ``rmse_t``, ``mae_t``, ``part_accuracy`` (0..1), ``assembly_cd``.
    """
    fragments = fragments.to(torch.float64)
    mask = fragment_mask.bool()
    if anchor_largest and int(mask.sum()) > 0:
        pred_rot, pred_trans = _anchor_to_largest(
            pred_rot.to(torch.float64), pred_trans.to(torch.float64),
            gt_rot.to(torch.float64), gt_trans.to(torch.float64), fragments, mask,
        )
    pred_rot = pred_rot.to(torch.float64)
    pred_trans = pred_trans.to(torch.float64)
    gt_rot = gt_rot.to(torch.float64)
    gt_trans = gt_trans.to(torch.float64)

    valid = mask
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {k: float("nan") for k in
                ("rmse_r", "mae_r", "geodesic_r", "rmse_t", "mae_t", "part_accuracy", "assembly_cd")}

    # --- rotation: per-axis Euler degrees, wrapped (published convention) ---
    e_pred = matrix_to_euler_degrees(pred_rot)[valid]          # (V,3)
    e_gt = matrix_to_euler_degrees(gt_rot)[valid]
    d_eul = euler_degree_difference(e_pred, e_gt)               # (V,3)
    rmse_r = float(d_eul.pow(2).mean(-1).sqrt().mean())         # per part, then mean
    mae_r = float(d_eul.mean(-1).mean())
    geo = geodesic_degrees(pred_rot, gt_rot)[valid]
    geodesic_r = float(geo.mean())

    # --- translation ---
    d_t = (pred_trans - gt_trans)[valid]                        # (V,3)
    rmse_t = float(d_t.pow(2).mean(-1).sqrt().mean())
    mae_t = float(d_t.abs().mean(-1).mean())

    # --- part accuracy: CD between the SAME points under pred vs GT pose ---
    placed_pred = fragments @ pred_rot.transpose(-1, -2) + pred_trans.unsqueeze(-2)
    placed_gt = fragments @ gt_rot.transpose(-1, -2) + gt_trans.unsqueeze(-2)
    per_part_cd = _symmetric_mean_chamfer(placed_pred[valid], placed_gt[valid])   # (V,)
    part_accuracy = float((per_part_cd < pa_threshold).to(torch.float64).mean())

    # --- whole-assembly chamfer ---
    union_pred = placed_pred[valid].reshape(-1, 3)
    union_gt = placed_gt[valid].reshape(-1, 3)
    assembly_cd = float(_symmetric_mean_chamfer(union_pred, union_gt))

    return {
        "rmse_r": rmse_r,
        "mae_r": mae_r,
        "geodesic_r": geodesic_r,
        "rmse_t": rmse_t,
        "mae_t": mae_t,
        "part_accuracy": part_accuracy,
        "assembly_cd": assembly_cd,
    }


def evaluate_batch(
    fragments: torch.Tensor,
    fragment_mask: torch.Tensor,
    pred_rot: torch.Tensor,
    pred_trans: torch.Tensor,
    gt_rot: torch.Tensor,
    gt_trans: torch.Tensor,
    pa_threshold: float = 0.01,
    anchor_largest: bool = False,
) -> list:
    """Per-object metrics for a batch. fragments (B,F,N,3) etc. -> list of dicts."""
    out = []
    for b in range(fragments.shape[0]):
        out.append(
            evaluate_object(
                fragments[b], fragment_mask[b], pred_rot[b], pred_trans[b],
                gt_rot[b], gt_trans[b], pa_threshold, anchor_largest,
            )
        )
    return out


def aggregate(per_object: list, prefix: str = "") -> Dict[str, float]:
    """Mean over objects of the per-object metrics, skipping NaNs.

    Also emits the paper-scaled variants: ``rmse_t_x100`` and ``assembly_cd_x1000``,
    since Breaking Bad tables report RMSE(T) x1e-2 and CD x1e-3.
    """
    if not per_object:
        return {}
    keys = [k for k in per_object[0].keys()]
    result: Dict[str, float] = {}
    for k in keys:
        vals = [o[k] for o in per_object if o[k] == o[k]]  # drop NaN
        result[f"{prefix}{k}"] = float(sum(vals) / len(vals)) if vals else float("nan")
    if f"{prefix}rmse_t" in result:
        result[f"{prefix}rmse_t_x100"] = result[f"{prefix}rmse_t"] * 100.0
    if f"{prefix}assembly_cd" in result:
        result[f"{prefix}assembly_cd_x1000"] = result[f"{prefix}assembly_cd"] * 1000.0
    result[f"{prefix}num_objects"] = float(len(per_object))
    return result
