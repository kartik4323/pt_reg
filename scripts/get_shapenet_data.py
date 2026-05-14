#!/usr/bin/env python3
"""Download, extract, and prepare ShapeNetCore point clouds.

This script intentionally does not contain credentials. ShapeNetCore access is
gated; request/accept access on the hosting site first, then pass a token with
--hf-token or set HF_TOKEN in your shell.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ARCHIVE_EXTENSIONS = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Get ShapeNetCore from scratch and convert it to point clouds"
    )
    parser.add_argument(
        "--source",
        choices=["huggingface", "archive"],
        default="huggingface",
        help="Use Hugging Face download or local archives you already have.",
    )
    parser.add_argument("--repo-id", default="ShapeNet/ShapeNetCore")
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token. If omitted, HF_TOKEN or HUGGINGFACE_HUB_TOKEN is used.",
    )
    parser.add_argument(
        "--download-root",
        default="./downloads/shapenet_hf",
        help="Where Hugging Face files are downloaded.",
    )
    parser.add_argument(
        "--archive-root",
        default=None,
        help="Folder containing .zip/.tar archives for --source archive.",
    )
    parser.add_argument(
        "--archive",
        nargs="*",
        default=[],
        help="Specific archive file(s) for --source archive.",
    )
    parser.add_argument(
        "--extract-root",
        default="./downloads/ShapeNetCore",
        help="Where archives are extracted.",
    )
    parser.add_argument(
        "--output-root",
        default="./shape_processed",
        help="Final point-cloud dataset root used by the training YAML.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["02691156", "03001627", "04379243"],
        help="ShapeNet synset IDs to prepare.",
    )
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--max-objects-per-category", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument(
        "--hf-all-files",
        action="store_true",
        help="Download every file in the HF repo instead of category-matching files.",
    )
    return parser.parse_args()


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in ARCHIVE_EXTENSIONS)


def find_archives(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and is_archive(path))


def download_from_huggingface(args: argparse.Namespace) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --source huggingface. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. After ShapeNetCore access is approved, "
            "set $env:HF_TOKEN='hf_...' in PowerShell or pass --hf-token."
        )

    allow_patterns = None
    if not args.hf_all_files:
        allow_patterns = []
        for category in args.categories:
            allow_patterns.extend(
                [
                    f"*{category}*",
                    f"**/{category}/**",
                    f"**/{category}.zip",
                    f"**/{category}.tar*",
                    "README.md",
                    "DATA.md",
                    "taxonomy.json",
                    "*.json",
                ]
            )

    print(f"Downloading {args.repo_id} to {args.download_root}")
    return Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=args.download_root,
            token=token,
            allow_patterns=allow_patterns,
        )
    )


def archives_from_args(args: argparse.Namespace, downloaded_root: Optional[Path]) -> list[Path]:
    archives = [Path(path) for path in args.archive]
    if args.archive_root:
        archives.extend(find_archives(Path(args.archive_root)))
    if downloaded_root is not None:
        archives.extend(find_archives(downloaded_root))
    return sorted(set(path.resolve() for path in archives if path.exists()))


def extract_archive(path: Path, extract_root: Path) -> None:
    print(f"Extracting {path}")
    if path.name.lower().endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            for member in zf.infolist():
                target = (extract_root / member.filename).resolve()
                if not str(target).startswith(str(extract_root.resolve())):
                    raise RuntimeError(f"Unsafe zip member path: {member.filename}")
            zf.extractall(extract_root)
        return

    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as tf:
            for member in tf.getmembers():
                target = (extract_root / member.name).resolve()
                if not str(target).startswith(str(extract_root.resolve())):
                    raise RuntimeError(f"Unsafe tar member path: {member.name}")
            tf.extractall(extract_root)
        return

    print(f"[WARN] unsupported archive skipped: {path}")


def extract_archives(archives: Iterable[Path], extract_root: Path) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        extract_archive(archive, extract_root)

    nested = find_archives(extract_root)
    for archive in nested:
        marker = archive.with_suffix(archive.suffix + ".extracted")
        if marker.exists():
            continue
        extract_archive(archive, extract_root)
        marker.write_text("ok", encoding="utf-8")


def prepare_points(args: argparse.Namespace, input_root: Path) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("prepare_shapenet_points.py")),
        "--input-root",
        str(input_root),
        "--output-root",
        args.output_root,
        "--categories",
        *args.categories,
        "--num-points",
        str(args.num_points),
    ]
    if args.max_objects_per_category is not None:
        command.extend(["--max-objects-per-category", str(args.max_objects_per_category)])

    print("Preparing point clouds")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    downloaded_root: Optional[Path] = None

    if args.source == "huggingface" and not args.skip_download:
        downloaded_root = download_from_huggingface(args)
    elif args.source == "huggingface":
        downloaded_root = Path(args.download_root)

    extract_root = Path(args.extract_root)
    if not args.skip_extract:
        archives = archives_from_args(args, downloaded_root)
        if archives:
            extract_archives(archives, extract_root)
        elif downloaded_root is not None:
            print("No archives found; using downloaded folder directly as mesh root.")
            if extract_root.exists():
                shutil.rmtree(extract_root)
            shutil.copytree(downloaded_root, extract_root)
        else:
            raise RuntimeError("No archives found. Pass --archive-root or --archive.")

    if not args.skip_prepare:
        prepare_points(args, extract_root)

    print("[OK] ShapeNet point-cloud dataset is ready.")
    print(f"Use this in YAML: data.shapenet_root: {args.output_root}")


if __name__ == "__main__":
    main()
