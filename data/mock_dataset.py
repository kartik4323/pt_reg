"""
data/mock_dataset.py
─────────────────────
Synthetic point cloud generator for testing the pipeline without ShapeNet.

Generates primitive shapes (sphere, cube, cylinder, cone, torus) with random
parameters, normalises them, and saves as .npy files mirroring the ShapeNet
directory structure.

Usage:
    python data/mock_dataset.py \
        --output_dir ./mock_shapenet_data \
        --n_objects 500 \
        --num_points 4096
"""

import argparse
import json
import random
import uuid
from pathlib import Path

import numpy as np

# Fake synset IDs to mirror ShapeNet structure
MOCK_CATEGORIES = {
    "02691156": "sphere",      # airplane → sphere
    "03001627": "cube",        # chair    → cube
    "04379243": "cylinder",    # table    → cylinder
}


# ── Primitive samplers ─────────────────────────────────────────────────────────

def sample_sphere(n: int, r: float = 1.0) -> np.ndarray:
    pts = np.random.randn(n, 3).astype(np.float32)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8
    return pts * r * np.random.uniform(0.8, 1.2)


def sample_cube(n: int) -> np.ndarray:
    face = np.random.randint(0, 6, n)
    pts  = np.random.uniform(-1, 1, (n, 3)).astype(np.float32)
    axis = face // 2
    sign = (face % 2) * 2 - 1
    pts[np.arange(n), axis] = sign.astype(np.float32)
    return pts


def sample_cylinder(n: int, r: float = 1.0, h: float = 2.0) -> np.ndarray:
    theta = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
    y     = np.random.uniform(-h / 2, h / 2, n).astype(np.float32)
    x     = r * np.cos(theta)
    z     = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def sample_cone(n: int, r: float = 1.0, h: float = 2.0) -> np.ndarray:
    t     = np.random.uniform(0, 1, n).astype(np.float32)
    theta = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
    y = t * h - h / 2
    rad = r * (1 - t)
    x = rad * np.cos(theta)
    z = rad * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def sample_torus(n: int, R: float = 1.0, r: float = 0.3) -> np.ndarray:
    theta = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
    phi   = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.stack([x, y, z], axis=1)


SAMPLERS = [sample_sphere, sample_cube, sample_cylinder, sample_cone, sample_torus]


def sample_random_shape(n: int) -> np.ndarray:
    fn = random.choice(SAMPLERS)
    pts = fn(n)
    # Random rigid transform
    R = random_rotation()
    pts = (R @ pts.T).T
    return pts.astype(np.float32)


def random_rotation() -> np.ndarray:
    q = np.random.randn(4).astype(np.float32)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float32)


def normalize(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max()
    return pts / (scale + 1e-8)


# ── Generator ──────────────────────────────────────────────────────────────────

def generate_mock_dataset(
    output_dir: str,
    n_objects: int = 300,
    num_points: int = 4096,
) -> None:
    out = Path(output_dir)
    synsets = list(MOCK_CATEGORIES.keys())

    meta = {s: [] for s in synsets}

    for i in range(n_objects):
        syn = synsets[i % len(synsets)]
        obj_id = str(uuid.uuid4()).replace("-", "")[:16]

        pts = sample_random_shape(num_points)
        pts = normalize(pts)

        cat_dir = out / syn
        cat_dir.mkdir(parents=True, exist_ok=True)
        np.save(cat_dir / f"{obj_id}.npy", pts.astype(np.float32))
        meta[syn].append(obj_id)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{n_objects} mock objects...")

    # Save metadata.json
    meta_path = out / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total = sum(len(v) for v in meta.values())
    print(f"\n[OK] Mock dataset ready: {out}  ({total} objects)")
    print(f"   Categories: {list(MOCK_CATEGORIES.values())}")
    print(f"   metadata.json written.\n")
    print("To use this mock dataset, set in your config:")
    print(f"  data:")
    print(f"    shapenet_root: \"{output_dir}\"")
    print(f"    categories: {synsets}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate mock ShapeNet-like dataset")
    p.add_argument("--output_dir",  default="./mock_shapenet_data")
    p.add_argument("--n_objects",   type=int, default=300)
    p.add_argument("--num_points",  type=int, default=4096)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_mock_dataset(args.output_dir, args.n_objects, args.num_points)
