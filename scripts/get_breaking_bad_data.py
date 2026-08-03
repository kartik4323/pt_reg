#!/usr/bin/env python3
"""Prepare the Breaking Bad fracture dataset for this pipeline.

Breaking Bad ships *compressed* archives (everyday is only ~377 MB) that must be
expanded with the authors' own ``decompress.py`` into per-piece meshes (~60 GB for
everyday). This script takes it from there: it walks the decompressed tree and
converts each fracture pattern into a compact ``.npz`` holding per-fragment point
clouds, the ground-truth assembled-frame geometry, and a coarse occupancy grid --
so training never touches a mesh.

Get the raw data first (no registration required):
    Borealis Dataverse, DOI 10.5683/SP3/LZNPKB
    https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/LZNPKB
    -> everyday_compressed.zip  (+ data_split.tar.gz for the official splits)
    then:  python decompress.py --data_root <root> --subset everyday
           (needs: numpy scipy tqdm igl gpytoolbox==0.2.0)

Then:
    python scripts/get_breaking_bad_data.py \
        --raw-root <decompressed_root> --output-root ./breaking_bad_processed \
        --subset everyday --occupancy-resolution 32

Design notes that matter for protocol parity:
* Points are sampled **proportional to fragment surface area** with a floor, which
  is the Jigsaw convention, and we store the per-fragment count. We never pad a
  small piece up to a fixed count by duplicating points -- that silently destroys
  local geometry (normals, covariance, FPFH) and it is a defect we measured in the
  previous synthetic pipeline.
* Fragments are stored in the **assembled (canonical) frame**. Posing is applied at
  training time by the dataset, because Breaking Bad's protocol recentres each
  fragment to its own centroid and applies only a random rotation (no random
  translation). Baking a pose in here would prevent reproducing that.
* Fracture-surface labels are **not** in the released data. We derive them the
  standard way: a point is on a fracture surface iff it lies within ``eta`` (0.025)
  of another piece of the same pattern, measured in the assembled frame.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh

from utils.point_cloud_utils import occupancy_from_mesh

# Directory names for a fracture pattern. The reference loaders filter on the
# substring "fractured" and also accept "mode"; some trees show "fracture_*".
# Accept all three rather than guessing which release is on disk.
PATTERN_PREFIXES = ("fractured", "fracture", "mode")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Breaking Bad for this pipeline")
    p.add_argument("--raw-root", required=True, help="Root of the DECOMPRESSED Breaking Bad tree")
    p.add_argument("--output-root", default="./breaking_bad_processed")
    p.add_argument("--subset", default="everyday", choices=["everyday", "artifact", "other"])
    p.add_argument("--split-dir", default=None, help="Dir with <subset>.train.txt/.val.txt (default: <raw-root>/data_split)")
    p.add_argument("--sampling", default="fixed_per_fragment",
                   choices=["fixed_per_fragment", "area_proportional"],
                   help=(
                       "fixed_per_fragment: N distinct points per fragment (the Breaking Bad "
                       "baseline / PuzzleFusion++ convention, and a drop-in for the trainers' "
                       "fixed (F,N,3) tensor). area_proportional: total points split by surface "
                       "area with a floor (the Jigsaw convention). Both are duplication-free "
                       "because points are sampled from the MESH, not resampled from a cloud."
                   ))
    p.add_argument("--points-per-fragment", type=int, default=1000,
                   help="Points per fragment for --sampling fixed_per_fragment.")
    p.add_argument("--num-points-per-object", type=int, default=5000,
                   help="Total points per object for --sampling area_proportional.")
    p.add_argument("--min-points-per-fragment", type=int, default=30)
    p.add_argument("--max-parts", type=int, default=20, help="Skip patterns with more parts (protocol: 20).")
    p.add_argument("--min-parts", type=int, default=2)
    p.add_argument("--occupancy-resolution", type=int, default=32)
    p.add_argument("--fracture-eta", type=float, default=0.025,
                   help="Distance threshold for deriving fracture-surface labels.")
    p.add_argument("--patterns-per-object", type=int, default=None,
                   help="Cap fracture patterns per object (default: all).")
    p.add_argument("--max-objects", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load one piece; flatten Scenes and reject empties (same idiom as the ShapeNet prep)."""
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if len(g.vertices) > 0]
        if not geoms:
            raise ValueError("empty scene")
        loaded = trimesh.util.concatenate(geoms)
    if loaded.is_empty or len(loaded.vertices) == 0:
        raise ValueError("empty mesh")
    return loaded


def find_patterns(object_dir: Path) -> List[Path]:
    out = []
    for child in sorted(object_dir.iterdir()):
        if child.is_dir() and child.name.startswith(PATTERN_PREFIXES):
            if any(child.glob("piece_*.obj")) or any(child.glob("*.obj")):
                out.append(child)
    return out


def allocate_points(areas: np.ndarray, total: int, floor: int) -> np.ndarray:
    """Split `total` points across fragments proportional to area, with a floor.

    Greedy repair: give everyone the floor first, distribute the remainder by area,
    then fix rounding on the largest fragment.
    """
    n = len(areas)
    if total < n * floor:
        total = n * floor
    counts = np.full(n, floor, dtype=np.int64)
    remaining = total - counts.sum()
    if remaining > 0 and areas.sum() > 0:
        share = areas / areas.sum()
        extra = np.floor(share * remaining).astype(np.int64)
        counts += extra
        counts[int(np.argmax(areas))] += remaining - int(extra.sum())
    return counts


def derive_fracture_labels(clouds: List[np.ndarray], eta: float) -> List[np.ndarray]:
    """Per-point boolean: is this point on a fracture surface?

    A point is fracture iff within `eta` of ANY other piece, in the assembled frame.
    This is the standard derivation (the released data has no such label).
    """
    labels = []
    for i, ci in enumerate(clouds):
        others = [c for j, c in enumerate(clouds) if j != i]
        if not others:
            labels.append(np.zeros(len(ci), dtype=bool))
            continue
        rest = np.concatenate(others, axis=0)
        # chunk to keep the distance matrix bounded for large pieces
        near = np.zeros(len(ci), dtype=bool)
        step = 4096
        for s in range(0, len(ci), step):
            block = ci[s:s + step]
            d = np.linalg.norm(block[:, None, :] - rest[None, :, :], axis=-1)
            near[s:s + step] = d.min(axis=1) <= eta
        labels.append(near)
    return labels


def adjacency_from_labels(clouds: List[np.ndarray], eta: float) -> np.ndarray:
    """Piece-piece contact adjacency: do they share any point pair within eta?"""
    n = len(clouds)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = clouds[i], clouds[j]
            touch = False
            step = 4096
            for s in range(0, len(a), step):
                d = np.linalg.norm(a[s:s + step, None, :] - b[None, :, :], axis=-1)
                if bool((d.min() <= eta)):
                    touch = True
                    break
            adj[i, j] = adj[j, i] = touch
    return adj


def process_pattern(pattern_dir: Path, args: argparse.Namespace) -> Optional[dict]:
    """One fracture pattern -> dict of arrays, or None if it should be skipped."""
    piece_files = sorted(pattern_dir.glob("piece_*.obj")) or sorted(pattern_dir.glob("*.obj"))
    if not (args.min_parts <= len(piece_files) <= args.max_parts):
        return None

    meshes = []
    for f in piece_files:
        try:
            meshes.append(load_mesh(f))
        except Exception as exc:
            print(f"[warn] {f}: {exc}")
            return None

    # Normalize ONCE, using the assembled object, so every piece shares one frame.
    assembled = trimesh.util.concatenate(meshes)
    center = assembled.bounds.mean(axis=0)
    scale = float(np.abs(assembled.vertices - center).max())
    if scale <= 0:
        return None
    for m in meshes:
        m.vertices = (m.vertices - center) / scale

    areas = np.array([max(float(m.area), 1e-12) for m in meshes])
    if args.sampling == "fixed_per_fragment":
        counts = np.full(len(meshes), int(args.points_per_fragment), dtype=np.int64)
    else:
        counts = allocate_points(areas, args.num_points_per_object, args.min_points_per_fragment)

    clouds = []
    for m, k in zip(meshes, counts):
        pts, _ = trimesh.sample.sample_surface(m, int(k))
        clouds.append(np.asarray(pts, dtype=np.float32))

    fracture = derive_fracture_labels(clouds, args.fracture_eta)
    adjacency = adjacency_from_labels(clouds, args.fracture_eta)

    assembled_norm = trimesh.util.concatenate(meshes)
    occupancy = occupancy_from_mesh(assembled_norm, args.occupancy_resolution)
    target_pts, _ = trimesh.sample.sample_surface(assembled_norm, args.num_points_per_object)

    return {
        # ragged per-fragment arrays are stored concatenated + offsets
        "points": np.concatenate(clouds, axis=0).astype(np.float32),
        "offsets": np.cumsum([0] + [len(c) for c in clouds]).astype(np.int64),
        "fracture": np.concatenate(fracture, axis=0),
        "adjacency": adjacency,
        "target": np.asarray(target_pts, dtype=np.float32),
        "occupancy": occupancy,
        "num_parts": np.int64(len(clouds)),
    }


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    subset_root = raw_root / args.subset
    if not subset_root.exists():
        raise FileNotFoundError(
            f"{subset_root} not found. Decompress Breaking Bad first with the authors' "
            "decompress.py (see this file's docstring)."
        )
    out_root = Path(args.output_root) / args.subset
    out_root.mkdir(parents=True, exist_ok=True)

    split_dir = Path(args.split_dir) if args.split_dir else raw_root / "data_split"
    splits = {}
    for name in ("train", "val"):
        f = split_dir / f"{args.subset}.{name}.txt"
        if f.exists():
            splits[name] = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
        else:
            print(f"[warn] split file missing: {f}")
    if not splits:
        print("[warn] no official splits found; every object will be listed under 'train'")

    # everyday nests one category level; artifact/other do not
    object_dirs: List[Path] = []
    for child in sorted(subset_root.iterdir()):
        if not child.is_dir():
            continue
        if find_patterns(child):
            object_dirs.append(child)
        else:
            object_dirs.extend(d for d in sorted(child.iterdir()) if d.is_dir() and find_patterns(d))
    if args.max_objects:
        object_dirs = object_dirs[: args.max_objects]
    print(f"Found {len(object_dirs)} objects with fracture patterns under {subset_root}")

    def split_of(obj_dir: Path) -> str:
        rel = obj_dir.relative_to(subset_root).as_posix()
        for name, entries in splits.items():
            if any(rel in e or e in rel for e in entries):
                return name
        return "train"

    manifest: dict = {"subset": args.subset, "occupancy_resolution": args.occupancy_resolution,
                      "num_points_per_object": args.num_points_per_object,
                      "fracture_eta": args.fracture_eta, "splits": {}}
    written = skipped = 0
    for obj_dir in object_dirs:
        split = split_of(obj_dir)
        rel = obj_dir.relative_to(subset_root).as_posix().replace("/", "__")
        patterns = find_patterns(obj_dir)
        if args.patterns_per_object:
            patterns = patterns[: args.patterns_per_object]
        for pattern in patterns:
            key = f"{rel}__{pattern.name}"
            dest = out_root / f"{key}.npz"
            if dest.exists() and not args.overwrite:
                manifest["splits"].setdefault(split, []).append(key)
                continue
            try:
                data = process_pattern(pattern, args)
            except Exception as exc:
                print(f"[warn] {pattern}: {exc}")
                data = None
            if data is None:
                skipped += 1
                continue
            np.savez_compressed(dest, **data)
            manifest["splits"].setdefault(split, []).append(key)
            written += 1
            if written % 200 == 0:
                print(f"  wrote {written} patterns...")

    for split, keys in manifest["splits"].items():
        manifest["splits"][split] = sorted(set(keys))
    with open(Path(args.output_root) / f"{args.subset}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    counts = {k: len(v) for k, v in manifest["splits"].items()}
    print(f"[OK] wrote {written} patterns ({skipped} skipped) -> {out_root}")
    print(f"[OK] manifest: { {k: v for k, v in counts.items()} }")
    print(f"Use in YAML: data.shapenet_root: {args.output_root}  (subset: {args.subset})")


if __name__ == "__main__":
    main()
