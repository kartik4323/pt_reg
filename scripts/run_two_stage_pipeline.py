#!/usr/bin/env python3
"""CLI for the compatibility-driven two-stage assembly pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from data.mock_dataset import generate_mock_dataset
from training.two_stage_trainer import Stage1Trainer, Stage2Trainer, run_pose_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage 3D fragment assembly")
    parser.add_argument("--config", default="configs/two_stage_mock.yaml")
    parser.add_argument(
        "--mode",
        choices=["mock-data", "stage1", "stage2", "stage3", "all", "smoke"],
        default="smoke",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage1-checkpoint", default=None)
    parser.add_argument("--stage2-checkpoint", default=None)
    parser.add_argument("--no-freeze", action="store_true")
    parser.add_argument("--make-mock-data", action="store_true")
    return parser.parse_args()


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_mock_data(cfg: dict, force: bool = False) -> None:
    data_root = Path(cfg["data"]["shapenet_root"])
    if force or not (data_root / "metadata.json").exists():
        mock_cfg = cfg.get("mock_data", {})
        data_root.mkdir(parents=True, exist_ok=True)
        generate_mock_dataset(
            output_dir=str(data_root),
            n_objects=mock_cfg.get("n_objects", 60),
            num_points=cfg["data"].get("num_points_per_object", 512),
        )


def shrink_for_smoke(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["stage1"] = dict(cfg.get("stage1", {}))
    cfg["stage2"] = dict(cfg.get("stage2", {}))
    cfg["stage3"] = dict(cfg.get("stage3", {}))
    cfg["stage1"].update({"epochs": 1, "epoch_size": 12, "batch_size": 3, "num_workers": 0})
    cfg["stage2"].update({"epochs": 1, "epoch_size": 8, "batch_size": 2, "num_workers": 0})
    cfg["stage3"].update({"epoch_size": 4, "batch_size": 2})
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    device = torch.device(args.device)

    mock_cfg = cfg.get("mock_data", {})
    mock_enabled = mock_cfg.get("enabled", False)
    metadata_path = Path(cfg["data"]["shapenet_root"]) / "metadata.json"

    if args.mode == "mock-data" or args.make_mock_data or (args.mode == "smoke" and mock_enabled):
        ensure_mock_data(cfg, force=args.mode == "mock-data")
        if args.mode == "mock-data":
            return
    elif not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} was not found. For mock data, use a config with "
            "mock_data.enabled: true or pass --make-mock-data. For real data, "
            "run scripts/prepare_shapenet_points.py first."
        )

    if args.mode == "smoke":
        cfg = shrink_for_smoke(cfg)

    stage1_ckpt = args.stage1_checkpoint
    stage2_ckpt = args.stage2_checkpoint

    if args.mode in {"stage1", "all", "smoke"}:
        stage1_ckpt = str(Stage1Trainer(cfg, device).train())
        print(f"Saved Stage 1 checkpoint: {stage1_ckpt}")

    if args.mode in {"stage2", "all", "smoke"}:
        if stage1_ckpt is None:
            default_stage1 = Path(cfg["output"]["dir"]) / "stage1_pretrained.pt"
            stage1_ckpt = str(default_stage1) if default_stage1.exists() else None
        freeze_pretrained = cfg.get("stage2", {}).get("freeze_pretrained", True)
        if args.no_freeze:
            freeze_pretrained = False
        stage2_ckpt = str(
            Stage2Trainer(
                cfg,
                device,
                stage1_checkpoint=stage1_ckpt,
                freeze_pretrained=freeze_pretrained,
            ).train()
        )
        print(f"Saved Stage 2 checkpoint: {stage2_ckpt}")

    if args.mode in {"stage3", "all", "smoke"}:
        if stage2_ckpt is None:
            default_stage2 = Path(cfg["output"]["dir"]) / "stage2_assembly.pt"
            if not default_stage2.exists():
                raise FileNotFoundError(
                    "Stage 2 checkpoint not provided and outputs/stage2_assembly.pt was not found."
                )
            stage2_ckpt = str(default_stage2)
        metrics = run_pose_stage(
            cfg,
            device,
            stage2_checkpoint=stage2_ckpt,
            split="test",
            icp_iterations=cfg.get("stage3", {}).get("icp_iterations", 5),
        )
        print(
            "Stage 3 pose metrics: "
            + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        )


if __name__ == "__main__":
    main()
