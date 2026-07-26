#!/usr/bin/env python3
"""Pre-flight sanity check for Stage 3 pose labels and the SE(3) convention.

Run this BEFORE spending GPU hours on Stage 3. It verifies, using only the
ground-truth labels the dataset produces (no trained model), that:

  1. Applying the stored ``(align_rotations, align_translations)`` to the posed
     fragments reproduces the canonical fragments  -> oracle fragment MSE ~= 0.
  2. A Kabsch fit from posed -> canonical fragments recovers those same labels
     -> oracle rotation/translation error ~= 0.
  3. The oracle-aligned union matches the target object, and does so far better
     than the un-aligned (randomly posed) union  -> the labels are informative.

If (1) or (2) are not ~0, the transform convention is wrong and no amount of
Stage 3 training will help -- fix the labels first. Reuses
``models.pose.apply_fragment_transforms`` / ``kabsch_align`` so it exercises the
exact same math the trainer relies on.

Writes ``oracle_alignment_report.json`` to the configured ``output.dir``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from torch.utils.data import DataLoader

from data.assembly_dataset import AssemblyObjectDataset
from models.pose import (
    apply_fragment_transforms,
    kabsch_align,
    rotation_geodesic_error,
)
from utils.run_artifacts import build_results_markdown, summarize_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 oracle alignment check")
    parser.add_argument("--config", default="configs/two_stage_mock.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _chamfer(a: torch.Tensor, b: torch.Tensor) -> float:
    """Symmetric mean nearest-neighbour distance between (M,3) and (K,3)."""
    dist = torch.cdist(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
    return float(dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean())


def _union(fragments: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    parts = [fragments[i] for i in range(fragments.shape[0]) if bool(mask[i])]
    return torch.cat(parts, dim=0) if parts else fragments.reshape(-1, 3)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    device = torch.device(args.device)

    dataset = AssemblyObjectDataset(
        data_root=cfg["data"]["shapenet_root"],
        split=args.split,
        cfg=cfg,
        epoch_size=args.num_samples,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    frag_mse = []
    rot_err_deg = []
    trans_err = []
    cd_oracle = []
    cd_unaligned = []

    processed = 0
    for batch in loader:
        if processed >= args.num_samples:
            break
        fragments = batch["fragments"].to(device)            # (1, F, N, 3)
        canonical = batch["canonical_fragments"].to(device)  # (1, F, N, 3)
        gt_rot = batch["align_rotations"].to(device)          # (1, F, 3, 3)
        gt_trans = batch["align_translations"].to(device)     # (1, F, 3)
        mask = batch["fragment_mask"].to(device)              # (1, F)
        target = batch["target"].to(device)                   # (1, T, 3)

        # (1) Fragment MSE: posed -> canonical using stored labels.
        aligned = apply_fragment_transforms(fragments, gt_rot, gt_trans)
        valid = mask.reshape(-1).bool()
        per_pt_sq = ((aligned - canonical) ** 2).sum(dim=-1)  # (1, F, N)
        frag_sq = per_pt_sq.reshape(per_pt_sq.shape[1], -1).mean(dim=-1)  # (F,)
        frag_mse.extend(frag_sq[valid].cpu().tolist())

        # (2) Kabsch recovery: fit posed -> canonical, compare to stored labels.
        flat_src = fragments.reshape(-1, fragments.shape[2], 3)
        flat_dst = canonical.reshape(-1, canonical.shape[2], 3)
        rec_rot, rec_trans = kabsch_align(flat_src, flat_dst)
        rec_rot = rec_rot.reshape(gt_rot.shape)
        rec_trans = rec_trans.reshape(gt_trans.shape)
        r_deg = rotation_geodesic_error(rec_rot, gt_rot).reshape(-1) * 180.0 / math.pi
        t_l2 = torch.linalg.vector_norm(rec_trans - gt_trans, dim=-1).reshape(-1)
        rot_err_deg.extend(r_deg[valid].cpu().tolist())
        trans_err.extend(t_l2[valid].cpu().tolist())

        # (3) Union chamfer vs target: oracle-aligned vs un-aligned baseline.
        oracle_union = _union(aligned[0], mask[0])
        unaligned_union = _union(fragments[0], mask[0])
        cd_oracle.append(_chamfer(oracle_union, target[0]))
        cd_unaligned.append(_chamfer(unaligned_union, target[0]))
        processed += 1

    report = {
        "config": args.config,
        "split": args.split,
        "num_samples": processed,
        "num_fragments": len(frag_mse),
        "fragment_mse": summarize_distribution(frag_mse),
        "rotation_error_deg": summarize_distribution(rot_err_deg),
        "translation_error": summarize_distribution(trans_err),
        "union_chamfer_oracle": summarize_distribution(cd_oracle),
        "union_chamfer_unaligned": summarize_distribution(cd_unaligned),
    }
    # Verdict: labels are valid iff recovery is ~0 and oracle beats un-aligned.
    frag_ok = report["fragment_mse"].get("max", 1.0) < 1e-6
    rot_ok = report["rotation_error_deg"].get("max", 999.0) < 1.0
    trans_ok = report["translation_error"].get("max", 999.0) < 1e-3
    informative = report["union_chamfer_oracle"].get("mean", 1.0) < 0.5 * report[
        "union_chamfer_unaligned"
    ].get("mean", 1.0)
    report["verdict"] = {
        "fragment_mse_near_zero": frag_ok,
        "kabsch_rotation_recovered": rot_ok,
        "kabsch_translation_recovered": trans_ok,
        "oracle_union_beats_unaligned": informative,
        "labels_valid": bool(frag_ok and rot_ok and trans_ok and informative),
    }

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "oracle_alignment_report.json"
    import json

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    build_results_markdown(out_dir)

    print(f"Oracle alignment report written to {report_path}")
    print(
        "  fragment MSE (max)        = "
        f"{report['fragment_mse'].get('max', float('nan')):.3e}  -> "
        f"{'OK' if frag_ok else 'FAIL'}"
    )
    print(
        "  Kabsch rot err deg (max)  = "
        f"{report['rotation_error_deg'].get('max', float('nan')):.3e}  -> "
        f"{'OK' if rot_ok else 'FAIL'}"
    )
    print(
        "  Kabsch trans err (max)    = "
        f"{report['translation_error'].get('max', float('nan')):.3e}  -> "
        f"{'OK' if trans_ok else 'FAIL'}"
    )
    print(
        "  union chamfer oracle/unaligned mean = "
        f"{report['union_chamfer_oracle'].get('mean', float('nan')):.4f} / "
        f"{report['union_chamfer_unaligned'].get('mean', float('nan')):.4f}  -> "
        f"{'OK' if informative else 'FAIL'}"
    )
    print(f"  LABELS VALID: {report['verdict']['labels_valid']}")


if __name__ == "__main__":
    main()
