from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .utils import byte_size, require_relative, sha256_file, stable_hash, write_json


PACKAGE_ROOT = Path(__file__).resolve().parent
PARTNET_ALIASES = {"03001627": "chair", "03636649": "lamp", "03325088": "faucet", "04379243": "table", "03211117": "display", "monitor": "display"}
PATTERN_PREFIXES = ("fractured", "fracture", "mode")


@dataclass(frozen=True)
class Candidate:
    track: str
    split: str
    category: str
    object_id: str
    relative_path: str
    size_bytes: int
    source_sha256: str
    num_parts: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def load_subset_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else PACKAGE_ROOT / "data" / "subsets" / "common_v1.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _split_lookup(split_root: Path, subset: str) -> set[str]:
    path = split_root / f"{subset}.train.txt"
    return {line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()} if path.exists() else set()


def _belongs(relative: str, entries: set[str]) -> bool:
    """Match an object path against the official split lists.

    Breaking Bad's artifact lists are published in both ``<object-id>`` and
    ``artifact/<object-id>`` forms.  The decompressed artifact root removes
    that leading directory, so compare both canonical spellings rather than
    defaulting every unmatched artifact to the training split.
    """
    relative = relative.replace("\\", "/").strip("/")
    candidates = {relative, *(f"{subset}/{relative}" for subset in ("everyday", "artifact"))}
    for entry in entries:
        value = entry.replace("\\", "/").strip("/")
        if any(candidate == value or candidate.startswith(value + "/") for candidate in candidates):
            return True
    return False


def _hash_tree(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def scan_breaking_bad(root: str | Path, subset: str, min_parts: int = 2, max_parts: int = 20) -> list[Candidate]:
    root = Path(root).resolve()
    base = root / subset
    if not base.exists():
        raise FileNotFoundError(f"Expected decompressed Breaking Bad subset: {base}")
    split_root = root / "data_split"
    train = _split_lookup(split_root, subset)
    val_path = split_root / f"{subset}.val.txt"
    val = {line.strip().replace("\\", "/") for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()} if val_path.exists() else set()
    objects: list[tuple[str, Path]] = []
    for first in sorted(path for path in base.iterdir() if path.is_dir()):
        children = [path for path in first.iterdir() if path.is_dir()]
        if any(child.name.startswith(PATTERN_PREFIXES) for child in children):
            objects.append(("artifact" if subset != "everyday" else "unknown", first))
        else:
            objects.extend((first.name, child) for child in sorted(children))
    candidates: list[Candidate] = []
    for category, object_dir in objects:
        relative_object = object_dir.relative_to(base).as_posix()
        split = "train" if _belongs(relative_object, train) else "val" if _belongs(relative_object, val) else "train"
        for pattern in sorted(path for path in object_dir.iterdir() if path.is_dir() and path.name.startswith(PATTERN_PREFIXES)):
            pieces = sorted(pattern.glob("piece_*.obj")) or sorted(pattern.glob("*.obj"))
            if not min_parts <= len(pieces) <= max_parts:
                continue
            candidates.append(Candidate(
                track=f"breaking_bad_{subset}", split=split, category=category, object_id=relative_object,
                relative_path=pattern.relative_to(root).as_posix(), size_bytes=byte_size(pattern),
                source_sha256=_hash_tree(pattern), num_parts=len(pieces),
            ))
    return candidates


def scan_partnet_gpat(root: str | Path, min_parts: int = 2, max_parts: int = 20) -> list[Candidate]:
    root = Path(root).resolve()
    candidates: list[Candidate] = []
    for parts in sorted(root.rglob("parts.npy")):
        sample = parts.parent
        required = [parts, sample / "target.npy", sample / "poses.npy"]
        if not all(path.exists() for path in required):
            continue
        import numpy as np
        shape = np.load(parts, mmap_mode="r").shape
        if len(shape) != 3 or not min_parts <= shape[0] <= max_parts:
            continue
        relative = sample.relative_to(root)
        tokens = [piece.lower() for piece in relative.parts]
        category = next((PARTNET_ALIASES[token] for token in reversed(tokens) if token in PARTNET_ALIASES), None)
        split = next((token for token in tokens if token in {"train", "val", "test"}), None)
        if not category or not split:
            continue
        candidates.append(Candidate(
            track="partnet_gpat", split=split, category=category, object_id=relative.as_posix(),
            relative_path=relative.as_posix(), size_bytes=byte_size(sample), source_sha256=_hash_tree(sample), num_parts=int(shape[0]),
        ))
    return candidates


def select_object_disjoint(candidates: Iterable[Candidate], cap_bytes: int, seed: int, allowed_splits: set[str], balanced: bool) -> list[Candidate]:
    grouped: dict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.split in allowed_splits:
            grouped[(candidate.category, candidate.object_id, candidate.split)].append(candidate)
    objects = []
    for key, patterns in grouped.items():
        objects.append((key[0], key[2], key[1], sorted(patterns, key=lambda item: item.relative_path), sum(item.size_bytes for item in patterns)))
    buckets: dict[tuple[str, str], deque[tuple[str, str, str, list[Candidate], int]]] = defaultdict(deque)
    for category, split, object_id, patterns, size in objects:
        buckets[(split, category if balanced else "all")].append((category, split, object_id, patterns, size))
    for key, values in buckets.items():
        buckets[key] = deque(sorted(values, key=lambda item: stable_hash(f"{item[1]}/{item[0]}/{item[2]}", seed)))
    selected: list[Candidate] = []
    used = 0
    keys = sorted(buckets)
    while keys:
        progressed = False
        for key in list(keys):
            values = buckets[key]
            while values and used + values[0][4] > cap_bytes:
                values.popleft()
            if not values:
                keys.remove(key)
                continue
            _, _, _, patterns, size = values.popleft()
            selected.extend(patterns)
            used += size
            progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda item: (item.track, item.split, item.category, item.object_id, item.relative_path))


def build_manifest(
    breaking_bad_root: str | Path,
    partnet_gpat_root: str | Path | None,
    output: str | Path,
    config_path: str | Path | None = None,
    *,
    include_partnet: bool = True,
) -> dict[str, Any]:
    """Build the deterministic common manifest.

    ``include_partnet=False`` intentionally creates a Breaking Bad-only
    comparative protocol for hosts that cannot stage the gated PartNet v0
    archive.  GPAT must not be run against that manifest.
    """
    config = load_subset_config(config_path)
    seed = int(config["seed"])
    bb_root = Path(breaking_bad_root).resolve()
    if include_partnet and partnet_gpat_root is None:
        raise ValueError("--partnet-gpat-root is required unless --without-partnet is supplied")
    partnet_root = Path(partnet_gpat_root).resolve() if partnet_gpat_root is not None else None
    selected: list[Candidate] = []
    for track, data in config["tracks"].items():
        if track == "partnet_gpat" and not include_partnet:
            continue
        cap = int(float(data["cap_gib"]) * 1024**3)
        if track.startswith("breaking_bad_"):
            candidates = scan_breaking_bad(bb_root, str(data["subset"]), int(data["min_parts"]), int(data["max_parts"]))
            chosen = select_object_disjoint(candidates, cap, seed, set(data["allowed_splits"]), bool(data["category_balanced"]))
        else:
            assert partnet_root is not None
            candidates = scan_partnet_gpat(partnet_root, int(data["min_parts"]), int(data["max_parts"]))
            allowed = {"train", "val", "test"}
            seen, unseen = set(data["seen_categories"]), set(data["unseen_categories"])
            candidates = [
                item for item in candidates
                if (item.category in seen and item.split in {"train", "val", "test"})
                or (item.category in unseen and item.split == "test")
            ]
            chosen = select_object_disjoint(candidates, cap, seed, allowed, True)
        selected.extend(chosen)
    total = sum(item.size_bytes for item in selected)
    max_total = int(float(config["retained_cap_gib"]) * 1024**3)
    if total > max_total:
        raise RuntimeError(f"Selected {total} bytes, above retained cap {max_total}")
    document = {
        "version": config["version"] if include_partnet else f"{config['version']}_breaking_bad_only",
        "seed": seed, "convention": config["convention"],
        "sources": {"breaking_bad_root": str(bb_root), **({"partnet_gpat_root": str(partnet_root)} if partnet_root else {})},
        "track_scope": "common" if include_partnet else "breaking_bad_only",
        "retained_cap_bytes": max_total, "selected_bytes": total,
        "samples": [item.to_dict() for item in selected],
    }
    write_json(Path(output), document)
    return document
