#!/usr/bin/env python3
"""Inference entrypoint for the Stage 2 fragment assembly model."""

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

from data.assembly_dataset import AssemblyObjectDataset
from data.mock_dataset import generate_mock_dataset
from models.assembly import build_assembly_model
from utils.point_cloud_utils import ensure_n_points, normalize_point_cloud_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 fragment assembly inference")
    parser.add_argument("--config", default="configs/two_stage_mock.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to stage2_assembly.pt")
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


def load_model(cfg: dict, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_assembly_model(cfg).to(device)
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
) -> dict:
    output = model(fragments.to(device), fragment_mask.to(device))
    return {
        "reconstruction": output.point_cloud[0].detach().cpu().numpy(),
        "compatibility_scores": output.compatibility_scores[0].detach().cpu().numpy(),
        "fragment_mask": fragment_mask[0].detach().cpu().numpy(),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = load_cfg(args.config)
    maybe_make_mock_data(cfg, force=args.make_mock_data)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(cfg, args.checkpoint, device)

    if args.fragments is not None and len(args.fragments) > 0:
        fragments, fragment_mask, meta = prepare_fragment_files(args.fragments, cfg)
        result = run_one(model, fragments, fragment_mask, device)
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
        result = run_one(model, fragments, fragment_mask, device)
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
    save_json(
        output_dir / f"{prefix}_metadata.json",
        {
            **meta,
            "reconstruction_npy": str(recon_base.with_suffix(".npy")),
            "reconstruction_ply": str(recon_base.with_suffix(".ply")),
            "compatibility_scores": str(output_dir / f"{prefix}_compatibility_scores.npy"),
            "valid_fragment_mask": result["fragment_mask"].astype(bool).tolist(),
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


if __name__ == "__main__":
    main()
