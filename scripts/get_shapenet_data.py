#!/usr/bin/env python3
"""Download, extract, and prepare ShapeNetCore point clouds.

This script intentionally does not contain credentials. ShapeNetCore access is
gated; request/accept access on the hosting site first, then pass a token with
--hf-token or set HF_TOKEN in your shell.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--per-category",
        action="store_true",
        help=(
            "Stream one synset at a time: extract that category's archive, convert "
            "it to point clouds, then delete the extracted meshes before the next "
            "category. Caps peak disk usage at roughly a single category instead of "
            "extracting everything at once. Resumable: synsets already present in "
            "--output-root are skipped."
        ),
    )
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help=(
            "In --per-category mode, delete each source archive after its category "
            "has been successfully converted, to reclaim disk space. Off by default."
        ),
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


def prepare_points(
    args: argparse.Namespace,
    input_root: Path,
    categories: Optional[Iterable[str]] = None,
) -> None:
    cats = list(categories) if categories is not None else list(args.categories)
    command = [
        sys.executable,
        str(Path(__file__).with_name("prepare_shapenet_points.py")),
        "--input-root",
        str(input_root),
        "--output-root",
        args.output_root,
        "--categories",
        *cats,
        "--num-points",
        str(args.num_points),
    ]
    if args.max_objects_per_category is not None:
        command.extend(["--max-objects-per-category", str(args.max_objects_per_category)])

    print("Preparing point clouds")
    subprocess.run(command, check=True)


def gather_candidate_archives(
    args: argparse.Namespace, downloaded_root: Optional[Path]
) -> list[Path]:
    """All archive files we might extract, across explicit args and search roots."""
    candidates = [Path(path) for path in args.archive]
    for root in (downloaded_root, Path(args.archive_root) if args.archive_root else None):
        if root is not None:
            candidates.extend(find_archives(root))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def find_category_archive(candidates: Iterable[Path], synset: str) -> Optional[Path]:
    """Pick the archive for a synset, preferring an exact ``<synset>.<ext>`` name."""
    matches = [path for path in candidates if synset in path.name]
    if not matches:
        return None
    exact = [path for path in matches if path.name.split(".")[0] == synset]
    return exact[0] if exact else matches[0]


def rebuild_metadata(output_root: Path) -> dict:
    """(Re)write metadata.json from whatever ``<synset>/*.npy`` exist on disk.

    Necessary because prepare_shapenet_points.py rewrites metadata.json on every
    invocation with only the categories it was given; when we call it per-synset
    the file would otherwise end up listing only the last synset.
    """
    metadata: dict[str, list[str]] = {}
    if output_root.exists():
        for synset_dir in sorted(output_root.iterdir()):
            if synset_dir.is_dir():
                obj_ids = sorted(path.stem for path in synset_dir.glob("*.npy"))
                if obj_ids:
                    metadata[synset_dir.name] = obj_ids
    with open(output_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def run_per_category(args: argparse.Namespace, downloaded_root: Optional[Path]) -> None:
    """Extract -> convert -> delete, one synset at a time (bounded peak disk)."""
    candidates = gather_candidate_archives(args, downloaded_root)
    if not candidates:
        raise RuntimeError(
            "No archives found for --per-category. Pass --archive-root/--archive, "
            "or run the Hugging Face download first."
        )

    extract_base = Path(args.extract_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    done: list[str] = []
    skipped: list[str] = []
    for synset in args.categories:
        out_dir = output_root / synset
        if out_dir.exists() and any(out_dir.glob("*.npy")):
            print(f"[skip] {synset}: already prepared ({len(list(out_dir.glob('*.npy')))} files)")
            done.append(synset)
            continue

        archive = find_category_archive(candidates, synset)
        if archive is None:
            print(f"[warn] {synset}: no matching archive found; skipping")
            skipped.append(synset)
            continue

        # Extract into a synset-NEUTRAL working dir. Naming it after the synset
        # would duplicate the synset in the mesh path (the archive already has an
        # inner <synset>/ folder), which makes object_id_from_mesh label every
        # object with the synset id -> filename collisions -> lost objects.
        cat_extract = extract_base / "_per_category_work"
        if cat_extract.exists():
            shutil.rmtree(cat_extract)
        cat_extract.mkdir(parents=True, exist_ok=True)
        try:
            extract_archive(archive, cat_extract)
            prepare_points(args, cat_extract, categories=[synset])
            done.append(synset)
            if args.delete_archives:
                archive.unlink(missing_ok=True)
                print(f"[cleanup] deleted archive {archive.name}")
        except Exception as exc:  # keep going; a bad synset shouldn't abort the run
            print(f"[error] {synset}: {exc}")
            skipped.append(synset)
        finally:
            if cat_extract.exists():
                shutil.rmtree(cat_extract)
                print(f"[cleanup] removed extracted meshes for {synset}")

    metadata = rebuild_metadata(output_root)
    counts = {key: len(value) for key, value in metadata.items()}
    print(f"[OK] per-category prepare done. prepared={done} skipped={skipped}")
    print(f"[OK] metadata.json rebuilt with {counts}")


def main() -> None:
    args = parse_args()
    downloaded_root: Optional[Path] = None

    if args.source == "huggingface" and not args.skip_download:
        downloaded_root = download_from_huggingface(args)
    elif args.source == "huggingface":
        downloaded_root = Path(args.download_root)

    if args.per_category:
        run_per_category(args, downloaded_root)
        print("[OK] ShapeNet point-cloud dataset is ready.")
        print(f"Use this in YAML: data.shapenet_root: {args.output_root}")
        return

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
