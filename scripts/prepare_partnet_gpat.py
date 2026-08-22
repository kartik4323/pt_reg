#!/usr/bin/env python3
"""Convert GPAT-preprocessed PartNet samples into this pipeline's exact-part format.

This script intentionally does *not* download PartNet.  Run the official GPAT
preprocessor on the GPU VM first, then point ``--gpat-root`` at the resulting
sample tree.  The output is a small, portable manifest plus ``.npz`` files that
contain only the selected pilot/full samples; raw PartNet remains outside the
repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.gpat_partnet_dataset import _resample, apply_pose, contact_graph_and_boundaries


CATEGORY_ALIASES = {
    "03001627": "chair", "chair": "chair",
    "03636649": "lamp", "lamp": "lamp",
    "03325088": "faucet", "faucet": "faucet",
    "04379243": "table", "table": "table",
    "03211117": "display", "display": "display", "monitor": "display",
}
SEEN_CATEGORIES = ("chair", "lamp", "faucet")
UNSEEN_CATEGORIES = ("table", "display")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare exact PartNet samples from GPAT-format files")
    parser.add_argument("--raw-root", default=None, help="PartNet raw root (recorded only; never modified)")
    parser.add_argument("--metadata-root", default=None, help="PartNet/GPAT metadata root (recorded only)")
    parser.add_argument(
        "--gpat-root", default=None,
        help="Root containing official GPAT train/val/test sample directories (default: sibling $DATA_ROOT/partnet).",
    )
    parser.add_argument("--output-root", default="./partnet_gpat_processed")
    parser.add_argument("--manifest", default="partnet-pilot-v1", help="partnet-pilot-v1 or full")
    parser.add_argument(
        "--split-file",
        default=None,
        help="Optional JSON override mapping {train:[sample-id], val:[sample-id], test:[sample-id]} for custom GPAT-format data.",
    )
    parser.add_argument("--category-map", default=None, help="Optional JSON mapping sample-id -> category")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Optional[str]) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_samples(root: Path, category_map: Dict[str, str]) -> List[dict]:
    samples: List[dict] = []
    for parts_path in sorted(root.rglob("parts.npy")):
        sample_dir = parts_path.parent
        target_path = sample_dir / "target.npy"
        poses_path = sample_dir / "poses.npy"
        if not target_path.exists() or not poses_path.exists():
            continue
        sample_id = sample_dir.relative_to(root).as_posix()
        category = category_map.get(sample_id)
        if category is None:
            tokens = [piece.lower() for piece in sample_dir.relative_to(root).parts]
            category = next((CATEGORY_ALIASES[token] for token in reversed(tokens) if token in CATEGORY_ALIASES), None)
        if category is None:
            continue
        try:
            parts = np.load(parts_path, mmap_mode="r")
            poses = np.load(poses_path, mmap_mode="r")
            if parts.ndim != 3 or parts.shape[-1] != 3 or len(parts) < 2 or len(parts) > 20:
                continue
            if poses.shape[0] != len(parts) or poses.shape[-1] != 7:
                continue
        except (OSError, ValueError):
            continue
        split = next((piece.lower() for piece in sample_dir.relative_to(root).parts if piece.lower() in {"train", "val", "test"}), None)
        samples.append(
            {
                "id": sample_id,
                "source_dir": sample_dir,
                "category": category,
                "num_parts": int(len(parts)),
                "source_split": split,
            }
        )
    return samples


def source_split(sample: dict, splits: dict) -> Optional[str]:
    sample_id = sample["id"]
    for split in ("train", "test"):
        if sample_id in set(splits.get(split, [])):
            return split
    if sample_id in set(splits.get("val", [])):
        return "val"
    return sample.get("source_split")


def select_pilot(samples: List[dict], splits: dict) -> List[dict]:
    """The documented 84-object integration pilot, stratified by category."""
    selected: List[dict] = []
    for category in SEEN_CATEGORIES:
        candidates = sorted(
            [sample for sample in samples if sample["category"] == category and source_split(sample, splits) == "train"],
            key=lambda sample: stable_key(sample["id"]),
        )
        for sample in candidates[:16]:
            selected.append({**sample, "split": "train"})
        val_candidates = sorted(
            [sample for sample in samples if sample["category"] == category and source_split(sample, splits) == "val"],
            key=lambda sample: stable_key(sample["id"]),
        )
        for sample in val_candidates[:4]:
            selected.append({**sample, "split": "val"})
        test_candidates = sorted(
            [sample for sample in samples if sample["category"] == category and source_split(sample, splits) == "test"],
            key=lambda sample: stable_key(sample["id"]),
        )
        for sample in test_candidates[:4]:
            selected.append({**sample, "split": "test"})
    for category in UNSEEN_CATEGORIES:
        candidates = sorted(
            [sample for sample in samples if sample["category"] == category and source_split(sample, splits) == "test"],
            key=lambda sample: stable_key(sample["id"]),
        )
        for sample in candidates[:6]:
            selected.append({**sample, "split": "test"})
    expected = {"train": 48, "val": 12, "test": 24}
    actual = {split: sum(sample["split"] == split for sample in selected) for split in expected}
    if actual != expected:
        raise RuntimeError(f"Pilot selection incomplete: expected {expected}, got {actual}. Check --split-file/data.")
    return selected


def select_full(samples: List[dict], splits: dict) -> List[dict]:
    selected: List[dict] = []
    for sample in sorted(samples, key=lambda item: stable_key(item["id"])):
        original = source_split(sample, splits)
        if original is None:
            continue
        if sample["category"] in SEEN_CATEGORIES and original == "train":
            split = "train"
        elif sample["category"] in SEEN_CATEGORIES and original == "val":
            split = "val"
        elif original == "test":
            split = "test"
        else:
            continue
        selected.append({**sample, "split": split})
    return selected


def write_sample(sample: dict, output_root: Path, overwrite: bool) -> dict:
    source = sample["source_dir"]
    destination = output_root / "samples" / f"{stable_key(sample['id'])}.npz"
    if destination.exists() and not overwrite:
        return {**sample, "path": destination.relative_to(output_root).as_posix(), "sha256": sha256_file(destination)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    parts = np.asarray(np.load(source / "parts.npy"), dtype=np.float32)
    poses = np.asarray(np.load(source / "poses.npy"), dtype=np.float32)
    target = np.asarray(np.load(source / "target.npy"), dtype=np.float32)
    # GPAT's documented interchange contract is exactly 1,000 points per
    # part.  Resampling deterministically still makes custom data explicit and
    # ensures that boundary masks have the same indexing as downstream parts.
    canonical = np.stack(
        [
            _resample(
                apply_pose(part, pose[:3], pose[3:]),
                1000,
                np.random.default_rng(int(stable_key(f"{sample['id']}:{part_idx}")[:8], 16)),
            )
            for part_idx, (part, pose) in enumerate(zip(parts, poses))
        ],
        axis=0,
    ).astype(np.float32)
    adjacency, contact_boundaries = contact_graph_and_boundaries(canonical, threshold=0.01)
    arrays = {
        "canonical_parts": canonical,
        "local_parts": parts,
        "target": target,
        "gpat_poses": poses,
        "adjacency": adjacency,
        "contact_boundaries": contact_boundaries,
        "eq_class": np.asarray(
            np.load(source / "eq_class.npy") if (source / "eq_class.npy").exists() else np.arange(len(parts)),
            dtype=np.int64,
        ),
    }
    if (source / "labels.npy").exists():
        arrays["target_labels"] = np.asarray(np.load(source / "labels.npy"), dtype=np.int64)
    np.savez_compressed(destination, **arrays)
    source_names = ["parts.npy", "target.npy", "poses.npy"]
    source_names.extend(name for name in ("labels.npy", "eq_class.npy") if (source / name).exists())
    return {
        **{key: value for key, value in sample.items() if key != "source_dir"},
        "path": destination.relative_to(output_root).as_posix(),
        "sha256": sha256_file(destination),
        "source_files": {name: sha256_file(source / name) for name in source_names},
    }


def main() -> None:
    args = parse_args()
    default_gpat_root = Path(args.metadata_root).resolve().parent / "partnet" if args.metadata_root else None
    source_root = Path(args.gpat_root).resolve() if args.gpat_root else default_gpat_root
    if source_root is None:
        raise ValueError("Pass --gpat-root, or use --metadata-root whose parent contains the GPAT 'partnet' directory")
    output_root = Path(args.output_root).resolve()
    category_map = load_json(args.category_map)
    splits = load_json(args.split_file)
    samples = discover_samples(source_root, category_map)
    if not samples:
        raise RuntimeError(f"No valid GPAT samples found under {source_root}")
    if args.manifest == "partnet-pilot-v1":
        selected = select_pilot(samples, splits)
    elif args.manifest == "full":
        selected = select_full(samples, splits)
    else:
        raise ValueError("--manifest must be 'partnet-pilot-v1' or 'full'")

    written = [write_sample(sample, output_root, args.overwrite) for sample in selected]
    preparation_identity = {
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "selection": args.manifest,
        "contact_threshold": 0.01,
        "part_points": 1000,
        "target_points": 5000,
        "max_parts": 20,
        "split_file_sha256": sha256_file(Path(args.split_file)) if args.split_file else None,
    }
    preprocessing_hash = hashlib.sha256(
        json.dumps(preparation_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    for entry in written:
        entry["preprocessing_hash"] = preprocessing_hash
    manifest = {
        "format_version": 1,
        "protocol": "ours_exact",
        "selection": args.manifest,
        "gpat_root": str(source_root),
        "raw_root": args.raw_root,
        "metadata_root": args.metadata_root,
        "split_file": str(Path(args.split_file).resolve()) if args.split_file else None,
        "point_counts": {"part": 1000, "target": 5000},
        "max_parts": 20,
        "preprocessing_hash": preprocessing_hash,
        "preprocessing_identity": preparation_identity,
        "samples": written,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    for split in ("train", "val", "test"):
        print(f"{split}: {sum(item['split'] == split for item in written)}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
