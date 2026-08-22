#!/usr/bin/env python3
"""Run only the fixed Stage-3 GT-target robustness curves."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from utils.target_noise import evaluate_target_noise


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed PartNet target-noise curves")
    parser.add_argument("--config", default="configs/partnet_gpat_full.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--stage3-checkpoint", required=True)
    parser.add_argument("--output", required=True, help="JSON output inside an immutable run directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["data"]["shapenet_root"] = str(Path(args.data_root).resolve())
    result = evaluate_target_noise(
        cfg,
        str(Path(args.stage3_checkpoint).resolve()),
        torch.device(args.device),
        repetitions=args.repetitions,
        max_samples=args.max_samples,
    )
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
