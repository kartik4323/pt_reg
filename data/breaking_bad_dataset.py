"""Breaking Bad dataset, exposed with the same contract as ``AssemblyObjectDataset``.

Drop-in for the existing trainers (`Stage2Trainer`, `Stage3PoseTrainer`,
`run_pose_stage`, `scripts/oracle_stage3_alignment.py`,
`scripts/classical_registration_baseline.py`): same constructor signature, same
`__getitem__` keys/shapes, plus an ``occupancy`` key for the coarse shape prior.

Run ``scripts/get_breaking_bad_data.py`` first; this reads the ``.npz`` files it
writes, so training never touches a mesh.

**Protocol parity (the part that is easy to get wrong).** Breaking Bad does NOT
apply a random translation. Per fragment:

    m = centroid(canonical_points)          # in the assembled frame
    posed = (canonical - m) @ Rr.T          # Rr ~ uniform SO(3)

so the ground truth that maps posed -> canonical (this repo's row-vector
convention, ``canonical = posed @ R_align.T + t_align``) is

    R_align = Rr.T          t_align = m

i.e. **the GT translation IS the fragment's centroid in the canonical frame**, and
the GT rotation is the inverse of the applied one. Our previous synthetic pipeline
sampled random translations in +-0.7, which is a harder, non-comparable problem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.point_cloud_utils import random_rotation_matrix


def _pose_fragment(points: np.ndarray, random_rotation: bool = True):
    """Breaking Bad posing: recentre, then rotate. Returns (posed, R_align, t_align)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    rot = random_rotation_matrix() if random_rotation else np.eye(3, dtype=np.float32)
    posed = (centered @ rot.T).astype(np.float32)
    return posed, rot.T.astype(np.float32), centroid.astype(np.float32)


class BreakingBadAssemblyDataset(Dataset):
    """Padded multi-fragment samples from Breaking Bad fracture patterns."""

    def __init__(
        self,
        data_root: str,
        split: str,
        cfg: dict,
        epoch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        data_cfg = cfg["data"]
        self.root = Path(data_root)
        self.subset = data_cfg.get("subset", "everyday")
        manifest_path = self.root / f"{self.subset}_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. Run scripts/get_breaking_bad_data.py first."
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        splits = self.manifest.get("splits", {})
        # Breaking Bad ships train/val only; everyone uses val as the test set.
        alias = {"test": "val", "val": "val", "train": "train"}
        key = alias.get(split, split)
        keys = list(splits.get(key, []))
        if not keys:
            raise RuntimeError(
                f"No samples for split={split!r} (resolved {key!r}) in {manifest_path}. "
                f"Available: { {k: len(v) for k, v in splits.items()} }"
            )
        self.keys = sorted(keys)
        self.sample_dir = self.root / self.subset

        frag_cfg = cfg.get("fragment", {})
        self.max_fragments = int(frag_cfg.get("max_fragments", 20))
        self.min_fragments = int(frag_cfg.get("min_fragments", 2))
        aug_cfg = cfg.get("augmentation", {})
        self.random_rotation = bool(aug_cfg.get("random_fragment_rotation", True))
        self.num_points_per_fragment = int(data_cfg.get("num_points_per_fragment", 1000))
        self.num_points_per_object = int(data_cfg.get("num_points_per_object", 5000))
        self.epoch_size = epoch_size or len(self.keys)
        self.occupancy_resolution = int(self.manifest.get("occupancy_resolution", 32))

    def __len__(self) -> int:
        return self.epoch_size

    def _load(self, key: str) -> dict:
        with np.load(self.sample_dir / f"{key}.npz") as z:
            return {k: z[k] for k in z.files}

    def _resample(self, pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """Match the fixed per-fragment count the trainers require.

        Sub-sampling is without replacement. Up-sampling can only duplicate (the
        stored cloud is discrete), so the preparation script should be run with
        ``--sampling fixed_per_fragment --points-per-fragment N`` matching
        ``data.num_points_per_fragment`` to avoid it entirely.
        """
        count = len(pts)
        if count == n:
            return pts
        if count > n:
            idx = rng.choice(count, n, replace=False)
        else:
            idx = np.concatenate([np.arange(count), rng.choice(count, n - count, replace=True)])
        return pts[idx]

    def __getitem__(self, idx: int) -> dict:
        key = self.keys[idx % len(self.keys)]
        data = self._load(key)
        rng = np.random.default_rng()

        offsets = data["offsets"]
        n_frag_total = int(data["num_parts"])
        n_frag = min(n_frag_total, self.max_fragments)
        n_pts = self.num_points_per_fragment
        max_frag = self.max_fragments

        fragments = np.zeros((max_frag, n_pts, 3), dtype=np.float32)
        canonical = np.zeros_like(fragments)
        boundary = np.zeros((max_frag, n_pts), dtype=np.float32)
        mask = np.zeros((max_frag,), dtype=bool)
        rotations = np.tile(np.eye(3, dtype=np.float32), (max_frag, 1, 1))
        translations = np.zeros((max_frag, 3), dtype=np.float32)
        adjacency = np.zeros((max_frag, max_frag), dtype=bool)

        all_pts = data["points"]
        all_fracture = data["fracture"]
        for i in range(n_frag):
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            pts = all_pts[lo:hi]
            frac = all_fracture[lo:hi].astype(np.float32)
            # keep points and their fracture labels in step through resampling
            count = len(pts)
            if count == n_pts:
                sel = np.arange(count)
            elif count > n_pts:
                sel = rng.choice(count, n_pts, replace=False)
            else:
                sel = np.concatenate(
                    [np.arange(count), rng.choice(count, n_pts - count, replace=True)]
                )
            pts, frac = pts[sel], frac[sel]

            posed, r_align, t_align = _pose_fragment(pts, self.random_rotation)
            fragments[i] = posed
            canonical[i] = pts
            boundary[i] = frac
            mask[i] = True
            rotations[i] = r_align
            translations[i] = t_align

        src_adj = data["adjacency"][:n_frag, :n_frag]
        adjacency[:n_frag, :n_frag] = src_adj

        target = data["target"]
        target = self._resample(target, self.num_points_per_object, rng)
        # target_boundary is only used as a chamfer weight; fracture-ness of the
        # assembled surface is the union of the per-fragment labels, so approximate
        # it by nearest-fragment-point lookup would be costly -- zeros is honest.
        target_boundary = np.zeros((len(target),), dtype=np.float32)

        out = {
            "fragments": torch.from_numpy(fragments),
            "canonical_fragments": torch.from_numpy(canonical),
            "fragment_boundary": torch.from_numpy(boundary),
            "fragment_mask": torch.from_numpy(mask),
            "adjacency": torch.from_numpy(adjacency),
            "target": torch.from_numpy(target.astype(np.float32)),
            "target_boundary": torch.from_numpy(target_boundary),
            "align_rotations": torch.from_numpy(rotations),
            "align_translations": torch.from_numpy(translations),
            "num_fragments": torch.tensor(n_frag, dtype=torch.long),
            "object_id": key,
            "synset": key.split("__")[0],
            "strategy": key.split("__")[-1],
        }
        if "occupancy" in data:
            out["occupancy"] = torch.from_numpy(data["occupancy"].astype(np.float32))
        return out


class BreakingBadPairDataset(Dataset):
    """Fragment pairs for Stage-1 fracture-surface compatibility.

    Positives are genuinely adjacent pieces (real shared fracture surfaces, from the
    derived contact adjacency); negatives are non-adjacent pieces of the same object
    (hard) or pieces from different objects (easy).
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        cfg: dict,
        epoch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.base = BreakingBadAssemblyDataset(data_root, split, cfg, epoch_size=None)
        stage_cfg = cfg.get("stage1", {})
        self.epoch_size = epoch_size or stage_cfg.get("epoch_size", len(self.base.keys) * 4)
        pair_cfg = cfg.get("pairs", {})
        self.positive_ratio = float(pair_cfg.get("positive_ratio", 0.5))
        self.hard_ratio = float(pair_cfg.get("hard_negative_ratio", 0.25))
        self.n_pts = self.base.num_points_per_fragment

    def __len__(self) -> int:
        return self.epoch_size

    def _sample_object(self, rng) -> dict:
        return self.base[int(rng.integers(0, len(self.base.keys)))]

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng()
        roll = rng.random()
        sample = self._sample_object(rng)
        mask = sample["fragment_mask"].numpy()
        adj = sample["adjacency"].numpy()
        valid = np.flatnonzero(mask)

        pos_pairs = [(i, j) for i in valid for j in valid if i < j and adj[i, j]]
        neg_pairs = [(i, j) for i in valid for j in valid if i < j and not adj[i, j]]

        if roll < self.positive_ratio and pos_pairs:
            i, j = pos_pairs[int(rng.integers(0, len(pos_pairs)))]
            label = 1.0
            a, b = sample["fragments"][i], sample["fragments"][j]
            ba, bb = sample["fragment_boundary"][i], sample["fragment_boundary"][j]
        elif roll < self.positive_ratio + self.hard_ratio and neg_pairs:
            i, j = neg_pairs[int(rng.integers(0, len(neg_pairs)))]
            label = 0.0
            a, b = sample["fragments"][i], sample["fragments"][j]
            ba, bb = sample["fragment_boundary"][i], sample["fragment_boundary"][j]
        else:
            other = self._sample_object(rng)
            ov = np.flatnonzero(other["fragment_mask"].numpy())
            i = int(valid[rng.integers(0, len(valid))])
            j = int(ov[rng.integers(0, len(ov))])
            label = 0.0
            a, b = sample["fragments"][i], other["fragments"][j]
            ba, bb = sample["fragment_boundary"][i], other["fragment_boundary"][j]

        return {
            "frag_a": a,
            "frag_b": b,
            "label": torch.tensor(label, dtype=torch.float32),
            "pair_type": "positive" if label > 0.5 else "negative",
            "boundary_a": ba,
            "boundary_b": bb,
        }
