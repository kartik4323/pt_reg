"""PartNet samples prepared in the public GPAT point-cloud format.

The official GPAT preprocessor writes a directory per sample with ``parts.npy``
and ``target.npy`` (and usually ``poses.npy``, ``labels.npy`` and
``eq_class.npy``).  This module deliberately consumes that interchange format
instead of a private copy of PartNet: the reference implementation and this
pipeline therefore see the same source geometry.

``ours_exact`` turns each exact, target-frame part into an independently posed
fragment, matching this repository's Stage-1 -> Stage-2 -> Stage-3 contract.
The stored manifest is the split authority; no random re-splitting occurs in
the loader.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.assembly_dataset import _random_pose


def _seed_for(*values: object) -> int:
    digest = hashlib.sha256("|".join(map(str, values)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _resample(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic per-sample point resampling without inventing geometry."""
    points = np.asarray(points, dtype=np.float32)
    if len(points) == count:
        return points
    if len(points) > count:
        return points[rng.choice(len(points), count, replace=False)]
    if len(points) == 0:
        return np.zeros((count, 3), dtype=np.float32)
    extra = rng.choice(len(points), count - len(points), replace=True)
    return np.concatenate([points, points[extra]], axis=0)


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """GPAT's scalar-first quaternion -> row-vector compatible rotation matrix."""
    q = np.asarray(quaternion, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def apply_pose(points: np.ndarray, translation: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Map local GPAT part points into the target frame."""
    rotation = quaternion_to_matrix(quaternion)
    return (np.asarray(points, dtype=np.float32) @ rotation.T + translation).astype(np.float32)


def contact_graph_and_boundaries(parts: np.ndarray, threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric surface-contact graph and per-point contact masks.

    The points are already in the assembled target frame.  A pair is adjacent if
    either surface has a point within ``threshold`` of the other; masks mark the
    corresponding contact-region points.  The convention is intentionally
    identical for Stage 1 labels and Stage 2 diagnostics.
    """
    num_parts, num_points, _ = parts.shape
    adjacency = np.zeros((num_parts, num_parts), dtype=bool)
    boundaries = np.zeros((num_parts, num_points), dtype=bool)
    for first in range(num_parts):
        for second in range(first + 1, num_parts):
            distances = np.linalg.norm(
                parts[first, :, None, :] - parts[second, None, :, :], axis=-1
            )
            first_near = distances.min(axis=1) <= threshold
            second_near = distances.min(axis=0) <= threshold
            # Require the nearest-surface criterion in both directions.  With
            # the same distance matrix this is symmetric, but spelling it out
            # prevents one-sided contact labels if the implementation changes.
            if first_near.any() and second_near.any():
                adjacency[first, second] = adjacency[second, first] = True
                boundaries[first] |= first_near
                boundaries[second] |= second_near
    return adjacency, boundaries


class GpatPartNetAssemblyDataset(Dataset):
    """Padded exact-part object samples selected by a prepared manifest."""

    def __init__(self, data_root: str, split: str, cfg: dict, epoch_size: Optional[int] = None) -> None:
        self.cfg = cfg
        self.root = Path(data_root)
        data_cfg = cfg["data"]
        manifest_arg = data_cfg.get("manifest", "manifest.json")
        manifest_path = Path(manifest_arg)
        if not manifest_path.is_absolute():
            manifest_path = self.root / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"GPAT PartNet manifest not found: {manifest_path}. "
                "Run scripts/prepare_partnet_gpat.py on the GPU VM first."
            )
        with open(manifest_path, "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("protocol") != "ours_exact":
            raise ValueError("This loader requires a manifest with protocol='ours_exact'")

        configured_categories = set(data_cfg.get("categories", []))
        self.entries = [
            entry
            for entry in self.manifest.get("samples", [])
            if entry.get("split") == split
            and (not configured_categories or entry.get("category") in configured_categories)
        ]
        if not self.entries:
            raise RuntimeError(f"No GPAT PartNet samples for split={split!r} in {manifest_path}")

        frag_cfg = cfg.get("fragment", {})
        self.max_fragments = int(frag_cfg.get("max_fragments", 20))
        self.min_fragments = int(frag_cfg.get("min_fragments", 2))
        self.num_fragment_points = int(data_cfg.get("num_points_per_fragment", 1000))
        self.num_target_points = int(data_cfg.get("num_points_per_object", 5000))
        self.contact_threshold = float(data_cfg.get("contact_threshold", 0.01))
        augmentation = cfg.get("augmentation", {})
        self.translation_scale = float(augmentation.get("fragment_pose_translation", 0.7))
        self.random_rotation = bool(augmentation.get("random_fragment_rotation", True))
        self.seed = int(data_cfg.get("seed", 42))
        # Training callers explicitly provide an epoch size.  Evaluation callers
        # leave it as ``None`` so validation/test are a deterministic single
        # pass over the manifest, never a repeated training-sized sampler.
        self.epoch_size = len(self.entries) if epoch_size is None else int(epoch_size)

    def __len__(self) -> int:
        return self.epoch_size

    def _entry(self, index: int) -> dict:
        return self.entries[index % len(self.entries)]

    def _load(self, entry: dict) -> dict:
        sample_path = self.root / entry["path"]
        with np.load(sample_path, allow_pickle=False) as sample:
            return {key: sample[key] for key in sample.files}

    def __getitem__(self, index: int) -> dict:
        entry = self._entry(index)
        data = self._load(entry)
        sample_rng = np.random.default_rng(_seed_for(self.seed, entry["id"], index))
        parts = np.asarray(data["canonical_parts"], dtype=np.float32)
        target = np.asarray(data["target"], dtype=np.float32)
        if not (self.min_fragments <= len(parts) <= self.max_fragments):
            raise ValueError(f"Sample {entry['id']} has invalid part count {len(parts)}")

        num_parts = len(parts)
        fragments = np.zeros((self.max_fragments, self.num_fragment_points, 3), dtype=np.float32)
        canonical = np.zeros_like(fragments)
        rotations = np.tile(np.eye(3, dtype=np.float32), (self.max_fragments, 1, 1))
        translations = np.zeros((self.max_fragments, 3), dtype=np.float32)
        for part_idx, part in enumerate(parts):
            point_rng = np.random.default_rng(_seed_for(self.seed, entry["id"], "part", part_idx))
            canonical_part = _resample(part, self.num_fragment_points, point_rng)
            # _random_pose uses numpy's global RNG.  Save/restore it so dataset
            # samples remain deterministic even with DataLoader workers.
            state = np.random.get_state()
            np.random.seed(_seed_for(self.seed, entry["id"], index, "pose", part_idx))
            try:
                posed, rotation, translation = _random_pose(
                    canonical_part, self.translation_scale, self.random_rotation
                )
            finally:
                np.random.set_state(state)
            fragments[part_idx] = posed
            canonical[part_idx] = canonical_part
            rotations[part_idx] = rotation
            translations[part_idx] = translation

        # ``prepare_partnet_gpat.py`` computes these once from the canonical
        # 1,000-point GPAT parts.  Keep a fallback for manually-produced
        # manifests, but never silently use a different label convention.
        if "adjacency" in data and "contact_boundaries" in data:
            adjacency = np.asarray(data["adjacency"], dtype=bool)
            boundaries = np.asarray(data["contact_boundaries"], dtype=bool)
            if adjacency.shape != (num_parts, num_parts) or boundaries.shape != (num_parts, self.num_fragment_points):
                raise ValueError(
                    f"Stored contact labels for {entry['id']} do not match its exact 1,000-point parts"
                )
        else:
            adjacency, boundaries = contact_graph_and_boundaries(
                canonical[:num_parts], threshold=self.contact_threshold
            )
        padded_adjacency = np.zeros((self.max_fragments, self.max_fragments), dtype=bool)
        padded_adjacency[:num_parts, :num_parts] = adjacency
        padded_boundaries = np.zeros((self.max_fragments, self.num_fragment_points), dtype=np.float32)
        padded_boundaries[:num_parts] = boundaries.astype(np.float32)
        target = _resample(target, self.num_target_points, sample_rng)
        target_labels = np.asarray(data.get("target_labels", np.full((len(data["target"]),), num_parts, dtype=np.int64)), dtype=np.int64)
        if len(target_labels) != len(data["target"]):
            raise ValueError(f"target_labels length mismatch for {entry['id']}")
        # GPAT targets are already 5,000 points, so this normally preserves
        # index identity.  For custom data, reuse the deterministic resampling
        # indices by reconstructing a stable nearest copy of each output point.
        if len(data["target"]) == self.num_target_points:
            target_labels = target_labels.copy()
        else:
            original_target = np.asarray(data["target"], dtype=np.float32)
            nearest = np.linalg.norm(target[:, None, :] - original_target[None, :, :], axis=-1).argmin(axis=1)
            target_labels = target_labels[nearest]

        equivalent = np.asarray(data.get("eq_class", np.arange(num_parts)), dtype=np.int64)
        padded_equivalent = np.full((self.max_fragments,), -1, dtype=np.int64)
        padded_equivalent[:num_parts] = equivalent[:num_parts]
        gpat_poses = np.zeros((self.max_fragments, 7), dtype=np.float32)
        if "gpat_poses" in data:
            source_poses = np.asarray(data["gpat_poses"], dtype=np.float32)
            if source_poses.shape != (num_parts, 7):
                raise ValueError(f"gpat_poses shape mismatch for {entry['id']}")
            gpat_poses[:num_parts] = source_poses
        return {
            "fragments": torch.from_numpy(fragments),
            "canonical_fragments": torch.from_numpy(canonical),
            "fragment_boundary": torch.from_numpy(padded_boundaries),
            "fragment_mask": torch.from_numpy(
                np.arange(self.max_fragments, dtype=np.int64) < num_parts
            ),
            "adjacency": torch.from_numpy(padded_adjacency),
            "target": torch.from_numpy(target),
            "target_labels": torch.from_numpy(target_labels),
            "target_boundary": torch.zeros((self.num_target_points,), dtype=torch.float32),
            "align_rotations": torch.from_numpy(rotations),
            "align_translations": torch.from_numpy(translations),
            "equivalence_classes": torch.from_numpy(padded_equivalent),
            "gpat_poses": torch.from_numpy(gpat_poses),
            "num_fragments": torch.tensor(num_parts, dtype=torch.long),
            "object_id": entry["id"],
            "synset": entry.get("category", "unknown"),
            "strategy": "gpat_partnet_exact",
        }


class GpatPartNetPairDataset(Dataset):
    """Deterministic Stage-1 pairs from the PartNet contact graph."""

    def __init__(self, data_root: str, split: str, cfg: dict, epoch_size: Optional[int] = None) -> None:
        self.base = GpatPartNetAssemblyDataset(data_root, split, cfg, epoch_size=1)
        self.cfg = cfg
        stage_cfg = cfg.get("stage1", {})
        self.epoch_size = int(epoch_size or stage_cfg.get("epoch_size", len(self.base.entries) * 4))
        pairs = cfg.get("pairs", {})
        self.positive_ratio = float(pairs.get("positive_ratio", 0.5))
        self.hard_ratio = float(pairs.get("hard_negative_ratio", 0.25))
        self.seed = int(cfg.get("data", {}).get("seed", 42))

    def __len__(self) -> int:
        return self.epoch_size

    def _sample(self, entry_index: int, index: int) -> dict:
        # The base dataset resolves by manifest index; its epoch length is not
        # part of sample identity.  Avoid mutating shared Dataset state, which
        # would be unsafe when workers are enabled.
        return self.base[entry_index]

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng(_seed_for(self.seed, "pair", index))
        sample_index = int(rng.integers(0, len(self.base.entries)))
        sample = self._sample(sample_index, index)
        valid = np.flatnonzero(sample["fragment_mask"].numpy())
        adjacency = sample["adjacency"].numpy()
        positive = [(i, j) for i in valid for j in valid if i < j and adjacency[i, j]]
        negative = [(i, j) for i in valid for j in valid if i < j and not adjacency[i, j]]
        roll = rng.random()
        if roll < self.positive_ratio and positive:
            first, second = positive[int(rng.integers(0, len(positive)))]
            label, pair_type = 1.0, "positive"
            other = sample
        elif roll < self.positive_ratio + self.hard_ratio and negative:
            first, second = negative[int(rng.integers(0, len(negative)))]
            label, pair_type = 0.0, "hard_negative"
            other = sample
        else:
            first = int(valid[int(rng.integers(0, len(valid)))])
            other_index = int(rng.integers(0, len(self.base.entries)))
            if len(self.base.entries) > 1:
                # Cross-object negatives are required by the protocol.
                if other_index == sample_index:
                    other_index = (other_index + 1) % len(self.base.entries)
            other = self._sample(other_index, index + self.epoch_size)
            other_valid = np.flatnonzero(other["fragment_mask"].numpy())
            second = int(other_valid[int(rng.integers(0, len(other_valid)))])
            label, pair_type = 0.0, "easy_negative"
        return {
            "frag_a": sample["fragments"][first],
            "frag_b": other["fragments"][second],
            "label": torch.tensor(label, dtype=torch.float32),
            "pair_type": pair_type,
            "boundary_a": sample["fragment_boundary"][first],
            "boundary_b": other["fragment_boundary"][second],
        }
