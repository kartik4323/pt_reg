"""Read-only study observations; supervision never crosses the inference boundary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ._observations import FractureDataset
from .manifests import verify_manifest


INPUT_KEYS = ("points", "fragment_mask", "anchor_index")


def sanitize_input(sample: dict, device=None) -> dict:
    """Whitelist an unbatched observation, dropping every label and source path."""
    result = {key: torch.as_tensor(sample[key], device=device).detach().clone() for key in INPUT_KEYS}
    points, mask, anchor = (result[key] for key in INPUT_KEYS)
    if points.ndim != 3 or points.shape[0] != 3 or points.shape[-1] != 3 or points.shape[1] < 3:
        raise ValueError("Study inputs require padded points[3,N,3], with N >= 3")
    if mask.shape != (3,) or anchor.numel() != 1:
        raise ValueError("Expected fragment_mask[3] and scalar anchor_index")
    if not torch.isfinite(points).all() or mask.sum() not in (2, 3):
        raise ValueError("Inputs must be finite, complete two/three-fragment sets")
    mask = mask.bool()
    anchor = anchor.long().reshape(())
    if not 0 <= int(anchor) < 3 or not bool(mask[anchor]):
        raise ValueError("Anchor must identify an observed fragment")
    result.update(points=points.float(), fragment_mask=mask, anchor_index=anchor)
    return result


def observation_fingerprint(sample: dict) -> str:
    digest = hashlib.sha256()
    for key, value in sanitize_input(sample).items():
        digest.update(key.encode())
        array = value.cpu().contiguous().numpy()
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class StudyDataset(FractureDataset):
    """All requested records, with fresh deterministic views at each train step.

    The shared reader supplies supervision and normalization only. Its implicit
    training curriculum is replaced with exact indexing; repair gates and caches
    are never imported. Construction verifies every manifest asset by default.
    """

    def __init__(self, manifest, split, points_per_fragment=1024, seed=42,
                 fixed=None, limit=None, sdf_queries=512, translation_range=.7,
                 hard_noise_std=.001, verify=True):
        if split not in ("train", "val", "test", "cut_holdout", "all"):
            raise ValueError("Unknown study split")
        if points_per_fragment < 3 or sdf_queries < 2:
            raise ValueError("Need at least three input points and two supervision queries")
        self.verification = verify_manifest(manifest) if verify else {"status": "not_verified"}
        cfg = {"data": {"points_per_fragment": int(points_per_fragment), "seed": int(seed),
                        "sdf_queries": int(sdf_queries), "translation_range": float(translation_range),
                        "hard_noise_std": float(hard_noise_std)}}
        super().__init__(manifest, split, cfg, stage=1, fixed=split != "train" if fixed is None else fixed, limit=limit)
        self.manifest_sha256 = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
        # Different identities must not disguise the same asset across splits.
        by_source = {r["source_id"]: r for r in self.manifest["sources"]}
        content_splits = {}
        for record in self.manifest["patterns"]:
            group = "test" if record["split"] == "cut_holdout" else record["split"]
            content_hash = by_source[record["source_id"]]["sha256"]
            if content_hash in content_splits and content_splits[content_hash] != group:
                raise ValueError("Identical source content leaks across study splits")
            content_splits[content_hash] = group

    def _record(self, index):
        return self.records[index]

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["observation_id"] = observation_fingerprint(sample)
        sample["dataset_fingerprint"] = self.fingerprint
        sample["split"] = self.records[index]["split"]
        return sample


def collate_samples(samples, device=None):
    """Stack tensor values while preserving identity and provenance metadata."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    return {key: torch.stack([s[key] for s in samples]).to(device) if isinstance(value, torch.Tensor)
            else [s[key] for s in samples] for key, value in samples[0].items()}
