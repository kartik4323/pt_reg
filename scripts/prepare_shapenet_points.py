#!/usr/bin/env python3
"""Prepare ShapeNet-style meshes as point-cloud .npy files.

Preferred input layout for ShapeNetCore:

    <input-root>/<synset>/<object-id>/models/model_normalized.obj

The script also searches recursively for synset folders, so layouts like
<input-root>/ShapeNetCore.v2/<synset>/... work too.

Output layout expected by the two-stage pipeline:

    <output-root>/<synset>/<object-id>.npy
    <output-root>/metadata.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh

from utils.point_cloud_utils import normalize_point_cloud_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ShapeNet meshes to point clouds")
    parser.add_argument("--input-root", required=True, help="Extracted ShapeNetCore root")
    parser.add_argument("--output-root", required=True, help="Output point-cloud dataset root")
    parser.add_argument(
        "--categories",
        nargs="+",
        required=True,
        help="ShapeNet synset IDs, e.g. 02691156 03001627 04379243",
    )
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--max-objects-per-category", type=int, default=None)
    return parser.parse_args()


def find_meshes(input_root: Path, synset: str) -> Iterable[Path]:
    candidate_dirs = []
    direct = input_root / synset
    if direct.exists():
        candidate_dirs.append(direct)
    candidate_dirs.extend(
        path for path in input_root.rglob(synset) if path.is_dir() and path not in candidate_dirs
    )

    meshes: list[Path] = []
    for synset_dir in candidate_dirs:
        preferred = sorted(synset_dir.glob("*/models/model_normalized.obj"))
        meshes.extend(preferred if preferred else sorted(synset_dir.rglob("*.obj")))
    return sorted(set(meshes))


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geometries = [geom for geom in loaded.geometry.values() if len(geom.vertices) > 0]
        if not geometries:
            raise ValueError("empty scene")
        loaded = trimesh.util.concatenate(geometries)
    if loaded.is_empty or len(loaded.vertices) == 0:
        raise ValueError("empty mesh")
    return loaded


def object_id_from_mesh(path: Path, synset: str) -> str:
    parts = path.parts
    if synset in parts:
        synset_index = parts.index(synset)
        if synset_index + 1 < len(parts):
            return parts[synset_index + 1]
    return path.stem


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, list[str]] = {}
    for synset in args.categories:
        out_dir = output_root / synset
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata[synset] = []

        meshes = list(find_meshes(input_root, synset))
        if args.max_objects_per_category is not None:
            meshes = meshes[: args.max_objects_per_category]

        print(f"{synset}: found {len(meshes)} mesh files")
        for index, mesh_path in enumerate(meshes, start=1):
            try:
                mesh = load_mesh(mesh_path)
                points = mesh.sample(args.num_points).astype(np.float32)
                points = normalize_point_cloud_np(points)
                obj_id = object_id_from_mesh(mesh_path, synset)
                np.save(out_dir / f"{obj_id}.npy", points)
                metadata[synset].append(obj_id)
            except Exception as exc:
                print(f"[WARN] skipped {mesh_path}: {exc}")
                continue

            if index % 100 == 0:
                print(f"  processed {index}/{len(meshes)}")

    metadata = {key: sorted(value) for key, value in metadata.items() if value}
    with open(output_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    total = sum(len(value) for value in metadata.values())
    print(f"[OK] wrote {total} point clouds to {output_root}")
    print(f"[OK] metadata: {output_root / 'metadata.json'}")


if __name__ == "__main__":
    main()
