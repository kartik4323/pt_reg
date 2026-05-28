"""
Datasets for compatibility-driven 3D fragment assembly.

This module supplies the two data views used by the proposed pipeline:

1. CompatibilityPairDataset for Stage 1. Each item contains a fragment pair,
   a binary fit label, and a pair type.
2. AssemblyObjectDataset for Stage 2/3. Each item contains a padded set of
   posed fragments, canonical training targets, adjacency labels, and the
   ground-truth rigid transforms from posed fragment frames to the target
   object frame.

The fracture strategies are lightweight approximations that run directly on
point clouds. Plane and noisy Voronoi cuts are implemented now; physics-based
fracture can be plugged in later by replacing MultiFragmentGenerator.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.point_cloud_utils import (
    ensure_n_points,
    normalize_point_cloud_np,
    random_rotation_matrix,
)


def _load_point_cloud(path: Path) -> np.ndarray:
    pts = np.load(str(path)).astype(np.float32)
    return normalize_point_cloud_np(pts)


def _split_objects(
    data_root: Path,
    categories: Sequence[str],
    split: str,
    train_split: float,
    val_split: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], Dict[str, List[str]]]:
    meta_path = data_root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {data_root}. Generate mock data or "
            "run the ShapeNet preprocessing script first."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    rng = np.random.RandomState(seed)
    objects: List[Tuple[str, str]] = []
    by_synset: Dict[str, List[str]] = {}

    for synset in categories:
        obj_ids = list(metadata.get(synset, []))
        obj_ids.sort()
        rng.shuffle(obj_ids)
        n = len(obj_ids)
        n_train = int(n * train_split)
        n_val = int(n * val_split)

        if split == "train":
            chosen = obj_ids[:n_train]
        elif split == "val":
            chosen = obj_ids[n_train : n_train + n_val]
        elif split == "test":
            chosen = obj_ids[n_train + n_val :]
        else:
            raise ValueError("split must be one of: train, val, test")

        by_synset[synset] = chosen
        objects.extend((synset, obj_id) for obj_id in chosen)

    if not objects:
        raise RuntimeError(
            f"No objects found for split={split!r} in {data_root}. Check the "
            "configured categories and split fractions."
        )
    return objects, by_synset


def _ensure_points_and_mask(
    pts: np.ndarray,
    mask: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    count = pts.shape[0]
    if count == 0:
        pts = np.zeros((1, 3), dtype=np.float32)
        mask = np.zeros((1,), dtype=bool)
        count = 1

    if count == n:
        return pts.astype(np.float32), mask.astype(bool)

    replace = count < n
    idx = np.random.choice(count, n, replace=replace)
    return pts[idx].astype(np.float32), mask[idx].astype(bool)


def _apply_row_transform(pts: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply x' = R x + t to row-vector points."""
    return (pts @ rotation.T + translation[None, :]).astype(np.float32)


def _random_pose(
    pts: np.ndarray,
    translation_scale: float,
    random_rotation: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly pose a canonical fragment.

    Returns posed points and the transform that maps posed points back into
    the canonical object frame, using the same row-vector convention:
    canonical = posed @ R_align.T + t_align.
    """
    if random_rotation:
        rotation_pose = random_rotation_matrix()
    else:
        rotation_pose = np.eye(3, dtype=np.float32)
    translation_pose = np.random.uniform(
        -translation_scale, translation_scale, size=(3,)
    ).astype(np.float32)

    posed = _apply_row_transform(pts, rotation_pose, translation_pose)
    rotation_align = rotation_pose.T.astype(np.float32)
    translation_align = (-translation_pose @ rotation_pose).astype(np.float32)
    return posed, rotation_align, translation_align


def _centroids(parts: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([p.mean(axis=0) if len(p) else np.zeros(3) for p in parts]).astype(np.float32)


@dataclass
class FragmentSet:
    fragments: List[np.ndarray]
    boundary_masks: List[np.ndarray]
    adjacency: np.ndarray
    target: np.ndarray
    target_boundary_mask: np.ndarray
    strategy: str


class MultiFragmentGenerator:
    """Point-cloud fracture generator for 2 to 9 fragments."""

    def __init__(
        self,
        num_points_per_fragment: int = 512,
        num_target_points: int = 1024,
        min_fragments: int = 3,
        max_fragments: int = 6,
        strategies: Sequence[str] = ("plane", "voronoi", "irregular"),
        boundary_eps: float = 0.04,
        irregular_noise_scale: float = 0.06,
    ) -> None:
        if min_fragments < 2:
            raise ValueError("min_fragments must be at least 2")
        if max_fragments >= 10:
            raise ValueError("max_fragments must be < 10")
        self.num_points_per_fragment = num_points_per_fragment
        self.num_target_points = num_target_points
        self.min_fragments = min_fragments
        self.max_fragments = max_fragments
        self.strategies = list(strategies)
        self.boundary_eps = boundary_eps
        self.irregular_noise_scale = irregular_noise_scale

    def __call__(
        self,
        pts: np.ndarray,
        num_fragments: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> FragmentSet:
        pts = normalize_point_cloud_np(pts)
        if num_fragments is None:
            num_fragments = random.randint(self.min_fragments, self.max_fragments)
        if strategy is None:
            strategy = random.choice(self.strategies)

        if strategy == "plane":
            assignment, boundary, adjacency = self._plane_bins(pts, num_fragments)
        elif strategy in {"voronoi", "multi_plane", "irregular", "physics_proxy"}:
            noise = self.irregular_noise_scale if strategy in {"irregular", "physics_proxy"} else 0.0
            assignment, boundary, adjacency = self._voronoi(pts, num_fragments, noise)
        else:
            raise ValueError(f"Unknown fracture strategy: {strategy}")

        fragments: List[np.ndarray] = []
        boundary_masks: List[np.ndarray] = []
        for frag_idx in range(num_fragments):
            mask = assignment == frag_idx
            frag_pts = pts[mask]
            frag_boundary = boundary[mask]
            frag_pts, frag_boundary = _ensure_points_and_mask(
                frag_pts,
                frag_boundary,
                self.num_points_per_fragment,
            )
            fragments.append(frag_pts)
            boundary_masks.append(frag_boundary)

        target, target_boundary = _ensure_points_and_mask(
            pts, boundary, self.num_target_points
        )

        return FragmentSet(
            fragments=fragments,
            boundary_masks=boundary_masks,
            adjacency=adjacency.astype(bool),
            target=target,
            target_boundary_mask=target_boundary,
            strategy=strategy,
        )

    def _plane_bins(
        self,
        pts: np.ndarray,
        num_fragments: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        normal = np.random.randn(3).astype(np.float32)
        normal /= np.linalg.norm(normal) + 1e-8
        proj = pts @ normal
        quantiles = np.linspace(0.0, 1.0, num_fragments + 1)[1:-1]
        edges = np.quantile(proj, quantiles)
        assignment = np.digitize(proj, edges, right=False).astype(np.int64)

        if len(edges) == 0:
            boundary = np.zeros(len(pts), dtype=bool)
        else:
            d = np.min(np.abs(proj[:, None] - edges[None, :]), axis=1)
            boundary = d <= self.boundary_eps

        adjacency = np.zeros((num_fragments, num_fragments), dtype=bool)
        for idx in range(num_fragments - 1):
            adjacency[idx, idx + 1] = True
            adjacency[idx + 1, idx] = True
        return assignment, boundary, adjacency

    def _voronoi(
        self,
        pts: np.ndarray,
        num_fragments: int,
        noise_scale: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = pts.shape[0]
        seed_idx = np.random.choice(n, num_fragments, replace=False)
        seeds = pts[seed_idx]
        dists = np.linalg.norm(pts[:, None, :] - seeds[None, :, :], axis=-1)
        if noise_scale > 0:
            dists = dists + np.random.randn(*dists.shape) * noise_scale

        order = np.argsort(dists, axis=1)
        assignment = order[:, 0].astype(np.int64)
        best = np.take_along_axis(dists, order[:, :1], axis=1)[:, 0]
        second = np.take_along_axis(dists, order[:, 1:2], axis=1)[:, 0]
        boundary = (second - best) <= self.boundary_eps

        adjacency = np.zeros((num_fragments, num_fragments), dtype=bool)
        for first, second_idx, is_boundary in zip(order[:, 0], order[:, 1], boundary):
            if is_boundary and first != second_idx:
                adjacency[first, second_idx] = True
                adjacency[second_idx, first] = True

        if not adjacency.any():
            centers = _centroids([pts[assignment == i] for i in range(num_fragments)])
            center_dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
            np.fill_diagonal(center_dist, np.inf)
            for i in range(num_fragments):
                j = int(np.argmin(center_dist[i]))
                adjacency[i, j] = True
                adjacency[j, i] = True

        return assignment, boundary, adjacency


class _ObjectBackedDataset(Dataset):
    def __init__(self, data_root: str, split: str, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_root = Path(data_root)
        data_cfg = cfg["data"]
        self.objects, self.by_synset = _split_objects(
            data_root=self.data_root,
            categories=data_cfg["categories"],
            split=split,
            train_split=data_cfg.get("train_split", 0.8),
            val_split=data_cfg.get("val_split", 0.1),
            seed=data_cfg.get("seed", 42),
        )

        frag_cfg = cfg.get("fragment", {})
        self.generator = MultiFragmentGenerator(
            num_points_per_fragment=data_cfg.get("num_points_per_fragment", 512),
            num_target_points=data_cfg.get("num_points_per_object", 1024),
            min_fragments=frag_cfg.get("min_fragments", 3),
            max_fragments=frag_cfg.get("max_fragments", 6),
            strategies=frag_cfg.get("strategies", ["plane", "voronoi", "irregular"]),
            boundary_eps=frag_cfg.get("boundary_eps", 0.04),
            irregular_noise_scale=frag_cfg.get("irregular_noise_scale", 0.06),
        )

    def _load(self, synset: str, obj_id: str) -> np.ndarray:
        return _load_point_cloud(self.data_root / synset / f"{obj_id}.npy")


class CompatibilityPairDataset(_ObjectBackedDataset):
    """Pair samples for Stage 1 binary compatibility pretraining."""

    def __init__(
        self,
        data_root: str,
        split: str,
        cfg: dict,
        epoch_size: Optional[int] = None,
    ) -> None:
        super().__init__(data_root, split, cfg)
        stage_cfg = cfg.get("stage1", {})
        self.epoch_size = epoch_size or stage_cfg.get("epoch_size", len(self.objects) * 4)
        aug_cfg = cfg.get("augmentation", {})
        self.translation_scale = aug_cfg.get("fragment_pose_translation", 0.7)
        self.random_rotation = aug_cfg.get("random_fragment_rotation", True)
        pair_cfg = cfg.get("pairs", {})
        self.pair_ratios = {
            "positive": pair_cfg.get("positive_ratio", 0.5),
            "easy_negative": pair_cfg.get("easy_negative_ratio", 0.2),
            "hard_negative": pair_cfg.get("hard_negative_ratio", 0.2),
            "perturbed_positive": pair_cfg.get("perturbed_positive_ratio", 0.1),
        }
        total = sum(max(0.0, float(v)) for v in self.pair_ratios.values())
        if total <= 0:
            raise ValueError("At least one Stage 1 pair sampling ratio must be positive")
        self.pair_types = list(self.pair_ratios.keys())
        self.pair_probs = [max(0.0, float(self.pair_ratios[k])) / total for k in self.pair_types]
        self.perturb_noise = pair_cfg.get("perturb_jitter_std", 0.03)
        self.perturb_scale = pair_cfg.get("perturb_anisotropic_scale", 0.12)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> dict:
        pair_type = random.choices(self.pair_types, weights=self.pair_probs, k=1)[0]
        if pair_type == "positive":
            frag_a, mask_a, frag_b, mask_b = self._positive_pair()
            label = 1.0
        elif pair_type == "easy_negative":
            frag_a, mask_a, frag_b, mask_b = self._easy_negative_pair()
            label = 0.0
        elif pair_type == "hard_negative":
            frag_a, mask_a, frag_b, mask_b = self._hard_negative_pair()
            label = 0.0
        elif pair_type == "perturbed_positive":
            frag_a, mask_a, frag_b, mask_b = self._perturbed_positive_pair()
            label = 0.0
        else:
            raise ValueError(f"Unknown pair type: {pair_type}")

        frag_a, rot_a, trans_a = _random_pose(
            frag_a, self.translation_scale, self.random_rotation
        )
        frag_b, rot_b, trans_b = _random_pose(
            frag_b, self.translation_scale, self.random_rotation
        )

        return {
            "frag_a": torch.from_numpy(frag_a),
            "frag_b": torch.from_numpy(frag_b),
            "label": torch.tensor(label, dtype=torch.float32),
            "pair_type": pair_type,
            "boundary_a": torch.from_numpy(mask_a.astype(np.float32)),
            "boundary_b": torch.from_numpy(mask_b.astype(np.float32)),
            "align_rotations": torch.from_numpy(np.stack([rot_a, rot_b], axis=0)),
            "align_translations": torch.from_numpy(np.stack([trans_a, trans_b], axis=0)),
        }

    def _positive_pair(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        synset, obj_id = random.choice(self.objects)
        pts = self._load(synset, obj_id)
        frag_set = self.generator(
            pts,
            num_fragments=max(3, self.generator.min_fragments),
        )
        adjacent_pairs = np.argwhere(np.triu(frag_set.adjacency, k=1))
        if len(adjacent_pairs) == 0:
            idx_a, idx_b = 0, 1
        else:
            idx_a, idx_b = adjacent_pairs[np.random.randint(len(adjacent_pairs))]
            idx_a, idx_b = int(idx_a), int(idx_b)
        return (
            frag_set.fragments[idx_a],
            frag_set.boundary_masks[idx_a],
            frag_set.fragments[idx_b],
            frag_set.boundary_masks[idx_b],
        )

    def _easy_negative_pair(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        first = random.choice(self.objects)
        candidates = [obj for obj in self.objects if obj[0] != first[0]]
        if not candidates:
            candidates = [obj for obj in self.objects if obj != first]
        second = random.choice(candidates)
        frag_a, mask_a = self._random_fragment(*first)
        frag_b, mask_b = self._random_fragment(*second)
        return frag_a, mask_a, frag_b, mask_b

    def _hard_negative_pair(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        synset, obj_id = random.choice(self.objects)
        same_class = [oid for oid in self.by_synset.get(synset, []) if oid != obj_id]
        if not same_class:
            return self._easy_negative_pair()
        other_id = random.choice(same_class)
        frag_a, mask_a = self._random_fragment(synset, obj_id)
        frag_b, mask_b = self._random_fragment(synset, other_id)
        return frag_a, mask_a, frag_b, mask_b

    def _perturbed_positive_pair(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frag_a, mask_a, frag_b, mask_b = self._positive_pair()
        frag_b = self._perturb_fragment_geometry(frag_b)
        return frag_a, mask_a, frag_b, mask_b

    def _random_fragment(self, synset: str, obj_id: str) -> Tuple[np.ndarray, np.ndarray]:
        frag_set = self.generator(self._load(synset, obj_id), num_fragments=2)
        idx = random.randrange(len(frag_set.fragments))
        return frag_set.fragments[idx], frag_set.boundary_masks[idx]

    def _perturb_fragment_geometry(self, fragment: np.ndarray) -> np.ndarray:
        centered = fragment - fragment.mean(axis=0, keepdims=True)
        scales = np.random.uniform(
            1.0 - self.perturb_scale,
            1.0 + self.perturb_scale,
            size=(1, 3),
        ).astype(np.float32)
        noise = np.random.normal(0.0, self.perturb_noise, size=fragment.shape).astype(np.float32)
        return (centered * scales + fragment.mean(axis=0, keepdims=True) + noise).astype(np.float32)


CompatibilityTripletDataset = CompatibilityPairDataset


class AssemblyObjectDataset(_ObjectBackedDataset):
    """Padded multi-fragment object samples for Stage 2 reconstruction."""

    def __init__(
        self,
        data_root: str,
        split: str,
        cfg: dict,
        epoch_size: Optional[int] = None,
    ) -> None:
        super().__init__(data_root, split, cfg)
        stage_cfg = cfg.get("stage2", {})
        frag_cfg = cfg.get("fragment", {})
        self.epoch_size = epoch_size or stage_cfg.get("epoch_size", len(self.objects))
        self.max_fragments = frag_cfg.get("max_fragments", 6)
        aug_cfg = cfg.get("augmentation", {})
        self.translation_scale = aug_cfg.get("fragment_pose_translation", 0.7)
        self.random_rotation = aug_cfg.get("random_fragment_rotation", True)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> dict:
        synset, obj_id = self.objects[idx % len(self.objects)]
        frag_set = self.generator(self._load(synset, obj_id))
        n_frag = len(frag_set.fragments)

        n_pts = self.generator.num_points_per_fragment
        n_target = self.generator.num_target_points
        max_frag = self.max_fragments

        fragments = np.zeros((max_frag, n_pts, 3), dtype=np.float32)
        canonical_fragments = np.zeros_like(fragments)
        boundary_masks = np.zeros((max_frag, n_pts), dtype=np.float32)
        fragment_mask = np.zeros((max_frag,), dtype=bool)
        rotations = np.tile(np.eye(3, dtype=np.float32), (max_frag, 1, 1))
        translations = np.zeros((max_frag, 3), dtype=np.float32)
        adjacency = np.zeros((max_frag, max_frag), dtype=bool)

        for frag_idx, (frag, bmask) in enumerate(zip(frag_set.fragments, frag_set.boundary_masks)):
            if frag_idx >= max_frag:
                break
            posed, rot_align, trans_align = _random_pose(
                frag, self.translation_scale, self.random_rotation
            )
            fragments[frag_idx] = posed
            canonical_fragments[frag_idx] = frag
            boundary_masks[frag_idx] = bmask.astype(np.float32)
            fragment_mask[frag_idx] = True
            rotations[frag_idx] = rot_align
            translations[frag_idx] = trans_align

        adjacency[:n_frag, :n_frag] = frag_set.adjacency[:max_frag, :max_frag]

        target = ensure_n_points(frag_set.target, n_target).astype(np.float32)
        target_boundary = frag_set.target_boundary_mask.astype(np.float32)
        if target_boundary.shape[0] != n_target:
            _, target_boundary = _ensure_points_and_mask(
                frag_set.target, frag_set.target_boundary_mask, n_target
            )

        return {
            "fragments": torch.from_numpy(fragments),
            "canonical_fragments": torch.from_numpy(canonical_fragments),
            "fragment_boundary": torch.from_numpy(boundary_masks),
            "fragment_mask": torch.from_numpy(fragment_mask),
            "adjacency": torch.from_numpy(adjacency),
            "target": torch.from_numpy(target),
            "target_boundary": torch.from_numpy(target_boundary.astype(np.float32)),
            "align_rotations": torch.from_numpy(rotations),
            "align_translations": torch.from_numpy(translations),
            "num_fragments": torch.tensor(n_frag, dtype=torch.long),
            "object_id": obj_id,
            "synset": synset,
            "strategy": frag_set.strategy,
        }
