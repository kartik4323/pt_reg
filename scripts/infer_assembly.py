#!/usr/bin/env python3
"""Inference entrypoint for reconstruction and rigid fragment assembly."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from data.assembly_dataset import AssemblyObjectDataset, MultiFragmentGenerator
from data.mock_dataset import generate_mock_dataset
from models.assembly import build_assembly_model
from models.pose import build_pose_estimator, differentiable_icp_initialization
from utils.point_cloud_utils import (
    ensure_n_points,
    normalize_point_cloud_np,
    random_rotation_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fragment reconstruction and assembly inference")
    parser.add_argument("--config", default="configs/two_stage_mock.yaml")
    parser.add_argument("--checkpoint", default=None, help="Backward-compatible alias for --stage2-checkpoint")
    parser.add_argument("--stage1-checkpoint", default=None, help="Optional Stage 1 checkpoint path for metadata")
    parser.add_argument("--stage2-checkpoint", default=None, help="Path to stage2_assembly.pt")
    parser.add_argument("--stage3-checkpoint", default=None, help="Optional learned stage3_pose.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0, help="Dataset index for dataset-backed inference")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of dataset samples to run")
    parser.add_argument(
        "--fragments",
        nargs="*",
        default=None,
        help="Optional .npy fragment files. If provided, dataset sampling is skipped.",
    )
    parser.add_argument(
        "--object-point-cloud",
        default=None,
        help="Complete object .npy point cloud to fracture, reconstruct, assemble, and compare against.",
    )
    parser.add_argument("--num-fragments", type=int, default=None, help="Number of synthetic fragments to cut")
    parser.add_argument(
        "--cut-strategy",
        default=None,
        choices=["plane", "voronoi", "multi_plane", "irregular", "physics_proxy"],
        help="Optional fracture strategy for --object-point-cloud",
    )
    parser.add_argument(
        "--pose-mode",
        default="auto",
        choices=["auto", "learned", "icp", "none"],
        help="Stage 3 assembly mode. auto uses learned Stage 3 if provided, otherwise ICP.",
    )
    parser.add_argument("--icp-iterations", type=int, default=None)
    parser.add_argument("--prefix", default=None, help="Optional output filename prefix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-inputs", action="store_true", help="Save input fragments and target when available")
    parser.add_argument(
        "--make-mock-data",
        action="store_true",
        help="Generate mock data when the selected config has mock_data.enabled=true",
    )
    return parser.parse_args()


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def maybe_make_mock_data(cfg: dict, force: bool = False) -> None:
    mock_cfg = cfg.get("mock_data", {})
    if not mock_cfg.get("enabled", False):
        return

    data_root = Path(cfg["data"]["shapenet_root"])
    metadata_path = data_root / "metadata.json"
    if force or not metadata_path.exists():
        data_root.mkdir(parents=True, exist_ok=True)
        generate_mock_dataset(
            output_dir=str(data_root),
            n_objects=mock_cfg.get("n_objects", 60),
            num_points=cfg["data"].get("num_points_per_object", 512),
        )


def resolve_stage2_checkpoint(args: argparse.Namespace) -> str:
    checkpoint = args.stage2_checkpoint or args.checkpoint
    if checkpoint is None:
        raise ValueError("Provide --stage2-checkpoint, or the legacy --checkpoint argument")
    return checkpoint


def load_model(cfg: dict, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_assembly_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_pose_model(cfg: dict, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_pose_estimator(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def write_ply(path: Path, points: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


def save_points(base: Path, points: np.ndarray) -> None:
    np.save(base.with_suffix(".npy"), points.astype(np.float32))
    write_ply(base.with_suffix(".ply"), points.astype(np.float32))


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def chamfer_metrics(pred: np.ndarray, target: np.ndarray, tau: float = 0.05) -> dict:
    pred_t = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0)
    target_t = torch.from_numpy(target.astype(np.float32)).unsqueeze(0)
    dist = torch.cdist(pred_t, target_t, p=2)
    pred_to_target = dist.min(dim=2).values
    target_to_pred = dist.min(dim=1).values
    precision = (pred_to_target < tau).float().mean()
    recall = (target_to_pred < tau).float().mean()
    fscore = 2.0 * precision * recall / (precision + recall + 1.0e-8)
    return {
        "chamfer": float((pred_to_target.mean() + target_to_pred.mean()).item()),
        "pred_to_target": float(pred_to_target.mean().item()),
        "target_to_pred": float(target_to_pred.mean().item()),
        "fscore_tau": float(fscore.item()),
        "tau": tau,
    }


def apply_row_transform_np(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (points @ rotation.T + translation[None, :]).astype(np.float32)


def random_pose_np(points: np.ndarray, translation_scale: float, random_rotation: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if random_rotation:
        pose_rotation = random_rotation_matrix()
    else:
        pose_rotation = np.eye(3, dtype=np.float32)
    pose_translation = np.random.uniform(
        -translation_scale,
        translation_scale,
        size=(3,),
    ).astype(np.float32)
    posed = apply_row_transform_np(points, pose_rotation, pose_translation)
    align_rotation = pose_rotation.T.astype(np.float32)
    align_translation = (-pose_translation @ pose_rotation).astype(np.float32)
    return posed, align_rotation, align_translation


def prepare_object_point_cloud(path: str, cfg: dict, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    object_path = Path(path)
    pts = np.load(str(object_path)).astype(np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{object_path} must contain an array of shape (N, 3)")

    pts = normalize_point_cloud_np(pts)
    data_cfg = cfg.get("data", {})
    frag_cfg = cfg.get("fragment", {})
    max_fragments = frag_cfg.get("max_fragments", 6)
    n_fragments = args.num_fragments or min(max(frag_cfg.get("min_fragments", 3), 2), max_fragments)
    if n_fragments > max_fragments:
        raise ValueError(f"--num-fragments={n_fragments} exceeds config fragment.max_fragments={max_fragments}")

    generator = MultiFragmentGenerator(
        num_points_per_fragment=data_cfg.get("num_points_per_fragment", 512),
        num_target_points=data_cfg.get("num_points_per_object", 1024),
        min_fragments=max(2, min(n_fragments, max_fragments)),
        max_fragments=max_fragments,
        strategies=frag_cfg.get("strategies", ["plane", "voronoi", "irregular"]),
        boundary_eps=frag_cfg.get("boundary_eps", 0.04),
        irregular_noise_scale=frag_cfg.get("irregular_noise_scale", 0.06),
    )
    frag_set = generator(pts, num_fragments=n_fragments, strategy=args.cut_strategy)

    n_points = generator.num_points_per_fragment
    fragments = np.zeros((max_fragments, n_points, 3), dtype=np.float32)
    canonical_fragments = np.zeros_like(fragments)
    fragment_mask = np.zeros((max_fragments,), dtype=bool)
    rotations = np.tile(np.eye(3, dtype=np.float32), (max_fragments, 1, 1))
    translations = np.zeros((max_fragments, 3), dtype=np.float32)
    translation_scale = cfg.get("augmentation", {}).get("fragment_pose_translation", 0.7)
    random_rotation = cfg.get("augmentation", {}).get("random_fragment_rotation", True)

    for idx, fragment in enumerate(frag_set.fragments[:max_fragments]):
        posed, align_rot, align_trans = random_pose_np(fragment, translation_scale, random_rotation)
        fragments[idx] = posed
        canonical_fragments[idx] = fragment
        fragment_mask[idx] = True
        rotations[idx] = align_rot
        translations[idx] = align_trans

    target = ensure_n_points(pts, data_cfg.get("num_points_per_object", 1024)).astype(np.float32)
    meta = {
        "source": "object_point_cloud",
        "object_point_cloud": str(object_path),
        "num_fragments": int(fragment_mask.sum()),
        "cut_strategy": frag_set.strategy,
        "random_fragment_rotation": bool(random_rotation),
        "fragment_pose_translation": float(translation_scale),
    }
    sample = {
        "target": torch.from_numpy(target),
        "canonical_fragments": torch.from_numpy(canonical_fragments),
        "align_rotations": torch.from_numpy(rotations),
        "align_translations": torch.from_numpy(translations),
    }
    return torch.from_numpy(fragments).unsqueeze(0), torch.from_numpy(fragment_mask).unsqueeze(0), meta, sample


def prepare_fragment_files(paths: Iterable[str], cfg: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    fragment_paths = [Path(path) for path in paths]
    if not fragment_paths:
        raise ValueError("--fragments was provided but no fragment files were listed")

    max_fragments = cfg.get("fragment", {}).get("max_fragments", 6)
    if len(fragment_paths) > max_fragments:
        raise ValueError(f"Received {len(fragment_paths)} fragments, but config max_fragments={max_fragments}")

    n_points = cfg["data"].get("num_points_per_fragment", 512)
    fragments = np.zeros((max_fragments, n_points, 3), dtype=np.float32)
    mask = np.zeros((max_fragments,), dtype=bool)

    for idx, path in enumerate(fragment_paths):
        pts = np.load(str(path)).astype(np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"{path} must contain an array of shape (N, 3)")
        pts = ensure_n_points(normalize_point_cloud_np(pts), n_points)
        fragments[idx] = pts
        mask[idx] = True

    meta = {
        "source": "fragment_files",
        "fragment_files": [str(path) for path in fragment_paths],
        "num_fragments": int(mask.sum()),
    }
    return torch.from_numpy(fragments).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0), meta


def prepare_dataset_sample(dataset: AssemblyObjectDataset, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    sample = dataset[idx]
    meta = {
        "source": "dataset",
        "dataset_index": idx,
        "object_id": sample.get("object_id"),
        "synset": sample.get("synset"),
        "strategy": sample.get("strategy"),
        "num_fragments": int(sample["num_fragments"]),
    }
    return (
        sample["fragments"].unsqueeze(0),
        sample["fragment_mask"].unsqueeze(0),
        meta,
        sample,
    )


@torch.no_grad()
def run_one(
    model: torch.nn.Module,
    fragments: torch.Tensor,
    fragment_mask: torch.Tensor,
    device: torch.device,
    pose_model: Optional[torch.nn.Module] = None,
    pose_mode: str = "none",
    icp_iterations: int = 10,
) -> dict:
    output = model(fragments.to(device), fragment_mask.to(device))
    result = {
        "reconstruction": output.point_cloud[0].detach().cpu().numpy(),
        "compatibility_scores": output.compatibility_scores[0].detach().cpu().numpy(),
        "fragment_mask": fragment_mask[0].detach().cpu().numpy(),
    }
    if pose_mode == "learned":
        if pose_model is None:
            raise ValueError("pose_mode='learned' requires --stage3-checkpoint")
        pose_architecture = getattr(pose_model, "__class__").__name__
        if icp_iterations > 0 and pose_architecture != "TargetSegmentationPoseEstimator":
            init_rotations, init_translations, _ = differentiable_icp_initialization(
                fragments.to(device),
                output.point_cloud,
                fragment_mask.to(device),
                iterations=icp_iterations,
            )
        else:
            init_rotations = None
            init_translations = None
        pose_output = pose_model(
            fragments.to(device),
            output.point_cloud,
            fragment_mask.to(device),
            initial_rotations=init_rotations,
            initial_translations=init_translations,
        )
        result.update(
            {
                "rotations": pose_output.rotations[0].detach().cpu().numpy(),
                "translations": pose_output.translations[0].detach().cpu().numpy(),
                "aligned_fragments": pose_output.aligned_fragments[0].detach().cpu().numpy(),
                "aligned_union": pose_output.aligned_union[0].detach().cpu().numpy(),
                "pose_mode": "learned",
            }
        )
    elif pose_mode == "icp":
        rotations, translations, aligned = differentiable_icp_initialization(
            fragments.to(device),
            output.point_cloud,
            fragment_mask.to(device),
            iterations=icp_iterations,
        )
        result.update(
            {
                "rotations": rotations[0].detach().cpu().numpy(),
                "translations": translations[0].detach().cpu().numpy(),
                "aligned_fragments": aligned[0].detach().cpu().numpy(),
                "aligned_union": aligned.reshape(aligned.shape[0], -1, 3)[0].detach().cpu().numpy(),
                "pose_mode": "icp",
            }
        )
    else:
        result["pose_mode"] = "none"
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = load_cfg(args.config)
    maybe_make_mock_data(cfg, force=args.make_mock_data)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage2_checkpoint = resolve_stage2_checkpoint(args)
    model = load_model(cfg, stage2_checkpoint, device)
    pose_mode = args.pose_mode
    if pose_mode == "auto":
        pose_mode = "learned" if args.stage3_checkpoint is not None else "icp"
    pose_model = load_pose_model(cfg, args.stage3_checkpoint, device) if pose_mode == "learned" else None
    icp_iterations = args.icp_iterations or cfg.get("stage3", {}).get("icp_iterations", 10)

    if args.object_point_cloud is not None:
        fragments, fragment_mask, meta, sample = prepare_object_point_cloud(args.object_point_cloud, cfg, args)
        result = run_one(
            model,
            fragments,
            fragment_mask,
            device,
            pose_model=pose_model,
            pose_mode=pose_mode,
            icp_iterations=icp_iterations,
        )
        prefix = args.prefix or Path(args.object_point_cloud).stem
        meta.update(
            {
                "stage1_checkpoint": args.stage1_checkpoint,
                "stage2_checkpoint": stage2_checkpoint,
                "stage3_checkpoint": args.stage3_checkpoint,
                "pose_mode": result["pose_mode"],
                "icp_iterations": icp_iterations if result["pose_mode"] == "icp" else None,
            }
        )
        save_inference_outputs(output_dir, prefix, result, meta, fragments, sample, True)
        print_comparison(prefix, output_dir / f"{prefix}_comparison.json")
        return

    if args.fragments is not None and len(args.fragments) > 0:
        fragments, fragment_mask, meta = prepare_fragment_files(args.fragments, cfg)
        result = run_one(
            model,
            fragments,
            fragment_mask,
            device,
            pose_model=pose_model,
            pose_mode=pose_mode,
            icp_iterations=icp_iterations,
        )
        prefix = args.prefix or "fragments"
        save_inference_outputs(output_dir, prefix, result, meta, fragments, None, args.save_inputs)
        print(f"Saved inference output: {output_dir / prefix}_reconstruction.ply")
        return

    dataset = AssemblyObjectDataset(
        data_root=cfg["data"]["shapenet_root"],
        split=args.split,
        cfg=cfg,
        epoch_size=max(args.index + args.num_samples, 1),
    )

    for offset in range(args.num_samples):
        idx = args.index + offset
        fragments, fragment_mask, meta, sample = prepare_dataset_sample(dataset, idx)
        result = run_one(
            model,
            fragments,
            fragment_mask,
            device,
            pose_model=pose_model,
            pose_mode=pose_mode,
            icp_iterations=icp_iterations,
        )
        prefix = args.prefix or f"{args.split}_{idx:04d}"
        if args.num_samples > 1 and args.prefix:
            prefix = f"{args.prefix}_{idx:04d}"
        save_inference_outputs(output_dir, prefix, result, meta, fragments, sample, args.save_inputs)
        print(f"Saved inference output: {output_dir / prefix}_reconstruction.ply")


def save_inference_outputs(
    output_dir: Path,
    prefix: str,
    result: dict,
    meta: dict,
    fragments: torch.Tensor,
    sample: Optional[dict],
    save_inputs: bool,
) -> None:
    recon_base = output_dir / f"{prefix}_reconstruction"
    save_points(recon_base, result["reconstruction"])
    np.save(output_dir / f"{prefix}_compatibility_scores.npy", result["compatibility_scores"].astype(np.float32))
    comparison = {}
    if sample is not None and "target" in sample:
        target = sample["target"].numpy().astype(np.float32)
        comparison["reconstruction_vs_ground_truth"] = chamfer_metrics(result["reconstruction"], target)
        if "aligned_union" in result:
            comparison["aligned_union_vs_ground_truth"] = chamfer_metrics(result["aligned_union"], target)
            comparison["aligned_union_vs_reconstruction"] = chamfer_metrics(
                result["aligned_union"],
                result["reconstruction"],
            )
        save_json(output_dir / f"{prefix}_comparison.json", comparison)

    pose_outputs = {}
    if "rotations" in result:
        rotations_path = output_dir / f"{prefix}_rotations.npy"
        translations_path = output_dir / f"{prefix}_translations.npy"
        aligned_fragments_path = output_dir / f"{prefix}_aligned_fragments.npy"
        aligned_union_base = output_dir / f"{prefix}_aligned_union"
        np.save(rotations_path, result["rotations"].astype(np.float32))
        np.save(translations_path, result["translations"].astype(np.float32))
        np.save(aligned_fragments_path, result["aligned_fragments"].astype(np.float32))
        save_points(aligned_union_base, result["aligned_union"].astype(np.float32))
        pose_outputs = {
            "rotations": str(rotations_path),
            "translations": str(translations_path),
            "aligned_fragments": str(aligned_fragments_path),
            "aligned_union_npy": str(aligned_union_base.with_suffix(".npy")),
            "aligned_union_ply": str(aligned_union_base.with_suffix(".ply")),
        }

    save_json(
        output_dir / f"{prefix}_metadata.json",
        {
            **meta,
            "reconstruction_npy": str(recon_base.with_suffix(".npy")),
            "reconstruction_ply": str(recon_base.with_suffix(".ply")),
            "compatibility_scores": str(output_dir / f"{prefix}_compatibility_scores.npy"),
            "valid_fragment_mask": result["fragment_mask"].astype(bool).tolist(),
            "pose_mode": result.get("pose_mode", "none"),
            "pose_outputs": pose_outputs,
            "comparison": comparison,
        },
    )

    if not save_inputs:
        return

    np.save(output_dir / f"{prefix}_input_fragments.npy", fragments[0].numpy().astype(np.float32))
    if sample is not None:
        save_points(output_dir / f"{prefix}_target", sample["target"].numpy().astype(np.float32))
        np.save(
            output_dir / f"{prefix}_canonical_fragments.npy",
            sample["canonical_fragments"].numpy().astype(np.float32),
        )
        if "align_rotations" in sample:
            np.save(output_dir / f"{prefix}_gt_rotations.npy", sample["align_rotations"].numpy().astype(np.float32))
        if "align_translations" in sample:
            np.save(output_dir / f"{prefix}_gt_translations.npy", sample["align_translations"].numpy().astype(np.float32))


def print_comparison(prefix: str, path: Path) -> None:
    if not path.exists():
        print(f"Saved inference output: {path.parent / (prefix + '_reconstruction.ply')}")
        return
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    recon = metrics.get("reconstruction_vs_ground_truth", {})
    print(f"Saved reconstructed object: {path.parent / (prefix + '_reconstruction.ply')}")
    if recon:
        print(
            "Reconstruction vs ground truth: "
            f"CD={recon.get('chamfer', 0.0):.6f}, "
            f"F@{recon.get('tau', 0.05):.3f}={recon.get('fscore_tau', 0.0):.4f}"
        )
    aligned = metrics.get("aligned_union_vs_ground_truth")
    if aligned:
        print(
            "Aligned union vs ground truth: "
            f"CD={aligned.get('chamfer', 0.0):.6f}, "
            f"F@{aligned.get('tau', 0.05):.3f}={aligned.get('fscore_tau', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
