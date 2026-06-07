#!/usr/bin/env python3
"""Visualize inference outputs saved by scripts/infer_assembly.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


DEFAULT_COLORS = {
    "reconstruction": (0.10, 0.45, 1.00),
    "target": (0.10, 0.75, 0.25),
    "aligned_union": (1.00, 0.35, 0.10),
    "input_fragment": (0.85, 0.20, 0.85),
    "aligned_fragment": (1.00, 0.65, 0.05),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize saved inference point clouds")
    parser.add_argument("--dir", default="outputs/inference", help="Inference output directory")
    parser.add_argument("--prefix", default=None, help="Output prefix, e.g. object_smoke")
    parser.add_argument("--list", action="store_true", help="List available prefixes and exit")
    parser.add_argument("--backend", choices=["auto", "open3d", "matplotlib"], default="auto")
    parser.add_argument("--max-points", type=int, default=12000, help="Max points per displayed cloud")
    parser.add_argument("--legacy-scene", action="store_true", help="Use the old single-scene flag-based viewer")
    parser.add_argument("--show-fragments", action="store_true", help="Show per-fragment clouds if available")
    parser.add_argument("--show-target", action="store_true", help="Show ground-truth target if available")
    parser.add_argument("--show-aligned", action="store_true", help="Show aligned union/fragments if available")
    parser.add_argument("--no-reconstruction", action="store_true", help="Hide reconstructed object")
    parser.add_argument("--point-size", type=float, default=3.0)
    return parser.parse_args()


def available_prefixes(output_dir: Path) -> list[str]:
    prefixes = set()
    for path in output_dir.glob("*_metadata.json"):
        prefixes.add(path.name[: -len("_metadata.json")])
    for path in output_dir.glob("*_reconstruction.npy"):
        prefixes.add(path.name[: -len("_reconstruction.npy")])
    return sorted(prefixes)


def choose_prefix(output_dir: Path, requested: Optional[str]) -> str:
    if requested:
        return requested
    prefixes = available_prefixes(output_dir)
    if not prefixes:
        raise FileNotFoundError(f"No inference outputs found in {output_dir}")
    return prefixes[-1]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_points(path: Path, max_points: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    pts = np.load(path).astype(np.float32)
    if pts.ndim == 3:
        pts = pts.reshape(-1, pts.shape[-1])
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{path} does not contain points shaped (N, 3)")
    return downsample(pts, max_points)


def downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = np.random.choice(points.shape[0], max_points, replace=False)
    return points[idx]


def load_fragment_sets(path: Path, mask: Optional[Iterable[bool]], max_points: int) -> list[np.ndarray]:
    if not path.exists():
        return []
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"{path} does not contain fragments shaped (F, N, 3)")
    if mask is None:
        valid = [True] * arr.shape[0]
    else:
        valid = list(mask)
    return [downsample(arr[i], max_points) for i in range(min(len(valid), arr.shape[0])) if valid[i]]


def build_scene(output_dir: Path, prefix: str, args: argparse.Namespace) -> tuple[list[tuple[str, np.ndarray, tuple[float, float, float]]], dict]:
    metadata = load_json(output_dir / f"{prefix}_metadata.json")
    mask = metadata.get("valid_fragment_mask")
    clouds: list[tuple[str, np.ndarray, tuple[float, float, float]]] = []

    if not args.no_reconstruction:
        recon = load_points(output_dir / f"{prefix}_reconstruction.npy", args.max_points)
        if recon is not None:
            clouds.append(("reconstruction", recon, DEFAULT_COLORS["reconstruction"]))

    if args.show_target:
        target = load_points(output_dir / f"{prefix}_target.npy", args.max_points)
        if target is not None:
            clouds.append(("target", target, DEFAULT_COLORS["target"]))

    if args.show_aligned:
        aligned = load_points(output_dir / f"{prefix}_aligned_union.npy", args.max_points)
        if aligned is not None:
            clouds.append(("aligned_union", aligned, DEFAULT_COLORS["aligned_union"]))

    if args.show_fragments:
        per_fragment_max = max(1, args.max_points // 6)
        input_fragments = load_fragment_sets(output_dir / f"{prefix}_input_fragments.npy", mask, per_fragment_max)
        for idx, pts in enumerate(input_fragments):
            clouds.append((f"input_fragment_{idx}", pts, color_cycle(idx, base="input_fragment")))

        if args.show_aligned:
            aligned_fragments = load_fragment_sets(output_dir / f"{prefix}_aligned_fragments.npy", mask, per_fragment_max)
            for idx, pts in enumerate(aligned_fragments):
                clouds.append((f"aligned_fragment_{idx}", pts, color_cycle(idx, base="aligned_fragment")))

    if not clouds:
        raise FileNotFoundError(f"No displayable .npy point clouds found for prefix {prefix!r} in {output_dir}")
    return clouds, metadata


def build_standard_views(
    output_dir: Path,
    prefix: str,
    max_points: int,
) -> tuple[list[tuple[str, list[tuple[str, np.ndarray, tuple[float, float, float]]]]], dict]:
    metadata = load_json(output_dir / f"{prefix}_metadata.json")
    target = load_points(output_dir / f"{prefix}_target.npy", max_points)
    reconstruction = load_points(output_dir / f"{prefix}_reconstruction.npy", max_points)
    aligned_union = load_points(output_dir / f"{prefix}_aligned_union.npy", max_points)

    missing = []
    if target is None:
        missing.append(f"{prefix}_target.npy")
    if reconstruction is None:
        missing.append(f"{prefix}_reconstruction.npy")
    if aligned_union is None:
        missing.append(f"{prefix}_aligned_union.npy")
    if missing:
        raise FileNotFoundError(
            "The five-view visualizer needs these files: " + ", ".join(missing)
        )

    views = [
        (
            "1. Original object",
            [("original object", target, DEFAULT_COLORS["target"])],
        ),
        (
            "2. Stage 2 reconstructed object",
            [("stage 2 reconstruction", reconstruction, DEFAULT_COLORS["reconstruction"])],
        ),
        (
            "3. Stage 3 rigidly assembled fragments",
            [("stage 3 assembled fragments", aligned_union, DEFAULT_COLORS["aligned_union"])],
        ),
        (
            "4. Original + Stage 2 reconstruction",
            [
                ("original object", target, DEFAULT_COLORS["target"]),
                ("stage 2 reconstruction", reconstruction, DEFAULT_COLORS["reconstruction"]),
            ],
        ),
        (
            "5. Original + Stage 3 assembled fragments",
            [
                ("original object", target, DEFAULT_COLORS["target"]),
                ("stage 3 assembled fragments", aligned_union, DEFAULT_COLORS["aligned_union"]),
            ],
        ),
    ]
    return views, metadata


def color_cycle(idx: int, base: str) -> tuple[float, float, float]:
    palette = [
        (0.90, 0.15, 0.15),
        (0.15, 0.60, 0.95),
        (0.20, 0.75, 0.35),
        (0.95, 0.55, 0.10),
        (0.55, 0.25, 0.95),
        (0.95, 0.20, 0.65),
        (0.15, 0.75, 0.75),
        (0.65, 0.65, 0.15),
    ]
    color = palette[idx % len(palette)]
    if base == "aligned_fragment":
        return tuple(min(1.0, c * 0.75 + 0.25) for c in color)
    return color


def visualize_open3d(
    clouds: list[tuple[str, np.ndarray, tuple[float, float, float]]],
    window_name: str = "Inference point clouds",
) -> None:
    import open3d as o3d

    geometries = []
    for name, points, color in clouds:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)
        geometries.append(pcd)
        print(f"{name}: {points.shape[0]} points")
    o3d.visualization.draw_geometries(geometries, window_name=window_name)


def visualize_open3d_views(
    views: list[tuple[str, list[tuple[str, np.ndarray, tuple[float, float, float]]]]],
) -> None:
    for title, clouds in views:
        print(f"\n{title}")
        visualize_open3d(clouds, window_name=title)


def visualize_matplotlib(
    clouds: list[tuple[str, np.ndarray, tuple[float, float, float]]],
    point_size: float,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for name, points, color in clouds:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_size, c=[color], label=name, alpha=0.8)
        print(f"{name}: {points.shape[0]} points")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def visualize_matplotlib_views(
    views: list[tuple[str, list[tuple[str, np.ndarray, tuple[float, float, float]]]]],
    point_size: float,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20, 8))
    for idx, (title, clouds) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        for name, points, color in clouds:
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=point_size,
                c=[color],
                label=name,
                alpha=0.75,
            )
            print(f"{title} / {name}: {points.shape[0]} points")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect((1, 1, 1))
        if len(clouds) > 1:
            ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


def print_comparison(output_dir: Path, prefix: str) -> None:
    comparison = load_json(output_dir / f"{prefix}_comparison.json")
    if not comparison:
        return
    print("\nComparison metrics:")
    for name, metrics in comparison.items():
        chamfer = metrics.get("chamfer")
        fscore = metrics.get("fscore_tau")
        tau = metrics.get("tau", 0.05)
        if chamfer is not None and fscore is not None:
            print(f"  {name}: CD={chamfer:.6f}, F@{tau:.3f}={fscore:.4f}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.dir)
    if args.list:
        for prefix in available_prefixes(output_dir):
            print(prefix)
        return

    prefix = choose_prefix(output_dir, args.prefix)
    print(f"Visualizing prefix: {prefix}")
    if args.legacy_scene:
        clouds, metadata = build_scene(output_dir, prefix, args)
    else:
        views, metadata = build_standard_views(output_dir, prefix, args.max_points)
    if metadata:
        print(f"Source: {metadata.get('source', 'unknown')}, pose_mode: {metadata.get('pose_mode', 'unknown')}")
    print_comparison(output_dir, prefix)

    if args.backend in {"auto", "open3d"}:
        try:
            if args.legacy_scene:
                visualize_open3d(clouds)
            else:
                visualize_open3d_views(views)
            return
        except ImportError:
            if args.backend == "open3d":
                raise
            print("Open3D is not installed; falling back to matplotlib.")

    if args.legacy_scene:
        visualize_matplotlib(clouds, args.point_size)
    else:
        visualize_matplotlib_views(views, args.point_size)


if __name__ == "__main__":
    main()
