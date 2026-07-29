#!/usr/bin/env python3
"""Classical (non-learned) fragment->object registration baseline.

Measures the *geometric ceiling* of Stage 3: can a fragment's SE(3) pose be
recovered from geometry alone, without any learning? For each fragment we
register the randomly-posed fragment to the ground-truth object and compare the
estimated transform to the stored GT pose. If classical registration succeeds,
the fragments ARE geometrically alignable and the learned pose failures are a
feature problem; if it also fails, the fragments are genuinely ambiguous
(symmetry / size) and no pose head will help.

Two methods, auto-selected:
  * ransac_fpfh: Open3D FPFH + RANSAC + ICP (uses handcrafted geometric
    features -- the direct analog of "are geometric features enough"). Used if
    open3d imports.
  * multi_icp: pure-torch ICP from many random initial rotations, keep the
    lowest-Chamfer result (no dependencies; tests basin-of-convergence).

Writes ``classical_baseline_report.json`` to the config's output.dir with
overall / per-category / per-fragment-size error distributions and a verdict
(fraction of fragments under 20 deg). Reuses existing helpers only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.assembly_dataset import AssemblyObjectDataset
from models.pose import apply_fragment_transforms, kabsch_align, rotation_geodesic_error
from utils.run_artifacts import build_results_markdown, summarize_distribution

try:
    import open3d as o3d  # type: ignore
    _HAS_O3D = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_O3D = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classical fragment->object registration baseline")
    p.add_argument("--config", default="configs/two_stage_shapenet_direct.yaml")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--method", default="auto", choices=["auto", "ransac_fpfh", "multi_icp"])
    p.add_argument("--icp-inits", type=int, default=30)
    p.add_argument("--icp-iters", type=int, default=30)
    p.add_argument("--icp-src-points", type=int, default=256, help="Fragment points subsampled for the fallback ICP.")
    p.add_argument("--icp-tgt-points", type=int, default=1024, help="Object points subsampled for the fallback ICP.")
    p.add_argument("--voxel", type=float, default=0.05, help="Open3D voxel size (objects are unit-normalized).")
    p.add_argument(
        "--oracle-check",
        action="store_true",
        help="Register the CANONICAL (already-aligned) fragment; error should be ~0. Validates convention.",
    )
    return p.parse_args()


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _chamfer(a: torch.Tensor, b: torch.Tensor) -> float:
    d = torch.cdist(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
    return float(d.min(dim=1).values.mean() + d.min(dim=0).values.mean())


def _subsample(points: torch.Tensor, n: int) -> torch.Tensor:
    if points.shape[0] <= n:
        return points
    idx = torch.randperm(points.shape[0])[:n]
    return points[idx]


def _icp_once(src: torch.Tensor, tgt: torch.Tensor, iters: int):
    """Point-to-point ICP from the current frame. Returns (R, t) with aligned = src @ R.T + t."""
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


def register_multi_icp(src: torch.Tensor, tgt: torch.Tensor, inits: int, iters: int):
    """Multi-init ICP: try `inits` random initial rotations, keep lowest-Chamfer."""
    best = None
    best_cd = float("inf")
    for k in range(inits):
        if k == 0:
            r0 = torch.eye(3, dtype=src.dtype)
        else:
            q = torch.randn(4)
            q = q / q.norm()
            w, x, y, z = q
            r0 = torch.tensor([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ], dtype=src.dtype)
        src0 = src @ r0.transpose(-1, -2)
        r_icp, t_icp, current = _icp_once(src0, tgt, iters)
        cd = _chamfer(current, tgt)
        if cd < best_cd:
            best_cd = cd
            best = (r_icp @ r0, t_icp)
    return best[0], best[1]


def register_fpfh(src_np: np.ndarray, tgt_np: np.ndarray, voxel: float):
    """Open3D FPFH + RANSAC + ICP. Returns (R (3,3), t (3,)) with aligned = src @ R.T + t."""
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


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    method = args.method
    if method == "auto":
        method = "ransac_fpfh" if _HAS_O3D else "multi_icp"
    if method == "ransac_fpfh" and not _HAS_O3D:
        print("[warn] open3d not available; falling back to multi_icp")
        method = "multi_icp"
    print(f"Method: {method}  (open3d available: {_HAS_O3D})  oracle_check: {args.oracle_check}")

    dataset = AssemblyObjectDataset(
        data_root=cfg["data"]["shapenet_root"], split=args.split, cfg=cfg, epoch_size=args.num_samples
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    records = []  # (synset, extent, rot_deg, trans_err, chamfer)
    processed = 0
    for sample in loader:
        if processed >= args.num_samples:
            break
        fragments = sample["fragments"][0]              # (F, N, 3)
        canonical = sample["canonical_fragments"][0]    # (F, N, 3)
        gt_rot = sample["align_rotations"][0]           # (F, 3, 3)
        gt_trans = sample["align_translations"][0]      # (F, 3)
        mask = sample["fragment_mask"][0]               # (F,)
        target = sample["target"][0]                    # (T, 3)
        synset = sample["synset"][0]

        for f in range(fragments.shape[0]):
            if not bool(mask[f]):
                continue
            source = canonical[f] if args.oracle_check else fragments[f]
            true_rot = torch.eye(3) if args.oracle_check else gt_rot[f]
            true_trans = torch.zeros(3) if args.oracle_check else gt_trans[f]

            if method == "ransac_fpfh":
                r_est, t_est = register_fpfh(source.numpy(), target.numpy(), args.voxel)
            else:
                r_est, t_est = register_multi_icp(
                    _subsample(source, args.icp_src_points),
                    _subsample(target, args.icp_tgt_points),
                    args.icp_inits, args.icp_iters,
                )

            rot_deg = float(rotation_geodesic_error(r_est, true_rot) * 180.0 / np.pi)
            trans_err = float(torch.linalg.vector_norm(t_est - true_trans))
            aligned = source @ r_est.transpose(-1, -2) + t_est
            extent = float((canonical[f].max(0).values - canonical[f].min(0).values).norm())
            records.append((synset, extent, rot_deg, trans_err, _chamfer(aligned, target)))
        processed += 1

    if not records:
        print("[error] no fragments processed")
        return

    rot = [r[2] for r in records]
    trans = [r[3] for r in records]
    cham = [r[4] for r in records]

    # per-category
    per_cat = {}
    for synset in sorted(set(r[0] for r in records)):
        cat_rot = [r[2] for r in records if r[0] == synset]
        per_cat[synset] = {"count": len(cat_rot), "rotation_error_deg": summarize_distribution(cat_rot)}

    # per-extent tercile
    extents = sorted(r[1] for r in records)
    lo, hi = extents[len(extents) // 3], extents[2 * len(extents) // 3]
    buckets = {"small": [], "medium": [], "large": []}
    for r in records:
        b = "small" if r[1] <= lo else ("medium" if r[1] <= hi else "large")
        buckets[b].append(r[2])
    per_extent = {
        b: {"count": len(v), "rotation_error_deg": summarize_distribution(v)}
        for b, v in buckets.items() if v
    }

    frac_good = float(np.mean([1.0 if x < 20.0 else 0.0 for x in rot]))
    report = {
        "method": method,
        "open3d_available": _HAS_O3D,
        "oracle_check": args.oracle_check,
        "config": args.config,
        "split": args.split,
        "num_samples": processed,
        "num_fragments": len(records),
        "rotation_error_deg": summarize_distribution(rot),
        "translation_error": summarize_distribution(trans),
        "aligned_chamfer": summarize_distribution(cham),
        "fraction_rot_below_20deg": frac_good,
        "per_category": per_cat,
        "per_fragment_extent": per_extent,
        "verdict": {
            "geometry_recoverable": bool(frac_good >= 0.5),
            "note": "geometry_recoverable => features are the learned bottleneck (Branch A); "
                    "otherwise fragments are ambiguous (Branch B).",
        },
    }

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / ("classical_baseline_oracle.json" if args.oracle_check else "classical_baseline_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    build_results_markdown(out_dir)

    rd = report["rotation_error_deg"]
    print(f"Report written to {report_path}")
    print(f"  fragments={len(records)}  method={method}")
    print(f"  rotation_error_deg: mean={rd['mean']:.1f} p50={rd['p50']:.1f} p90={rd['p90']:.1f} min={rd['min']:.2f}")
    print(f"  fraction < 20deg  : {frac_good:.3f}")
    print(f"  translation_error : mean={report['translation_error']['mean']:.3f}")
    if args.oracle_check:
        print("  (oracle-check: rotation mean should be ~0 if the convention is correct)")
    else:
        print(f"  VERDICT geometry_recoverable={report['verdict']['geometry_recoverable']} "
              f"(>=50% of fragments under 20deg)")


if __name__ == "__main__":
    main()
