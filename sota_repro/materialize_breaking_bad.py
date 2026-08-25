"""Stage only manifest-selected Breaking Bad objects through the official decompressor."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from .subset import _hash_tree
from .utils import byte_size, copytree_or_link, read_json, require_relative, safe_rmtree, write_json


def _selected_objects(manifest: dict, subset: str) -> tuple[set[str], list[dict]]:
    track = f"breaking_bad_{subset}"
    rows = [row for row in manifest["samples"] if row["track"] == track]
    objects = {str(row["object_id"]) for row in rows}
    if not rows:
        raise ValueError(f"Manifest has no {track} samples")
    return objects, rows


def _stage_objects(compressed_root: Path, stage: Path, subset: str, object_ids: set[str]) -> None:
    source_root = compressed_root / f"{subset}_compressed"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing compressed dataset directory: {source_root}")
    for object_id in sorted(object_ids):
        source = source_root / object_id
        require_relative(source, source_root)
        if not source.is_dir():
            raise FileNotFoundError(f"Manifest object is absent from compressed data: {source}")
        copytree_or_link(source, stage / f"{subset}_compressed" / object_id, link=True)


def _copy_selected(rows: list[dict], stage: Path, output: Path) -> int:
    copied = 0
    for row in rows:
        relative = Path(str(row["relative_path"]))
        source = stage / relative
        target = output / str(row["track"]) / relative
        require_relative(source, stage)
        require_relative(target, output)
        if not source.is_dir():
            raise FileNotFoundError(f"Official decompressor did not produce selected sample: {source}")
        expected_hash = str(row["source_sha256"])
        if target.exists():
            if _hash_tree(target) != expected_hash:
                raise RuntimeError(f"Existing materialized sample differs from manifest: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        if _hash_tree(target) != expected_hash:
            raise RuntimeError(f"Copied sample differs from manifest: {target}")
        copied += int(row["size_bytes"])
    return copied


def materialize_breaking_bad(
    manifest_path: str | Path,
    compressed_root: str | Path,
    official_repository: str | Path,
    output_root: str | Path,
    scratch_root: str | Path,
    decompressor_python: str = sys.executable,
    subsets: tuple[str, ...] = ("everyday", "artifact"),
    keep_staging: bool = False,
    dry_run: bool = False,
) -> dict[str, int | str]:
    """Decompress and retain selected objects only; all stage data is disposable."""
    manifest = read_json(Path(manifest_path))
    compressed = Path(compressed_root).resolve()
    official = Path(official_repository).resolve()
    decompressor = official / "decompress.py"
    output = Path(output_root).resolve()
    scratch = Path(scratch_root).resolve()
    if not decompressor.is_file():
        raise FileNotFoundError(f"Official Breaking Bad decompressor not found: {decompressor}")
    stage = scratch / f"breaking_bad_selected_{uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        split_source = compressed / "data_split"
        if split_source.is_dir():
            copytree_or_link(split_source, stage / "data_split", link=True)
        copied = 0
        for subset in subsets:
            object_ids, rows = _selected_objects(manifest, subset)
            _stage_objects(compressed, stage, subset, object_ids)
            command = [decompressor_python, str(decompressor), "--data_root", str(stage), "--subset", subset]
            if subset == "everyday":
                for category in sorted({Path(object_id).parts[0] for object_id in object_ids}):
                    category_command = [*command, "--category", category]
                    print("[decompress]", " ".join(category_command))
                    if not dry_run:
                        subprocess.run(category_command, cwd=official, check=True)
            else:
                print("[decompress]", " ".join(command))
                if not dry_run:
                    subprocess.run(command, cwd=official, check=True)
            if not dry_run:
                copied += _copy_selected(rows, stage, output)
                safe_rmtree(stage / subset, stage)
                safe_rmtree(stage / f"{subset}_compressed", stage)
        if not dry_run and (stage / "data_split").exists():
            copytree_or_link(stage / "data_split", output / "data_split", link=False)
        if not dry_run:
            # Keep the selected-only manifest beside the retained data.  Model
            # runners use it for split lists and immutable run provenance.
            retained_manifest = dict(manifest)
            retained_manifest["sources"] = {"materialized_root": str(output)}
            retained_manifest["materialized_bytes"] = byte_size(output)
            cap = retained_manifest.get("retained_cap_bytes")
            if cap is not None and retained_manifest["materialized_bytes"] > int(cap):
                raise RuntimeError("Materialized corpus exceeds configured retained-data cap")
            write_json(output / "common_v1_manifest.json", retained_manifest)
        return {"output": str(output), "copied_bytes": copied, "stage": str(stage)}
    finally:
        if not keep_staging:
            safe_rmtree(stage, scratch)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--compressed-root", required=True,
                        help="Directory containing data_split and *_compressed directories after archive extraction.")
    parser.add_argument("--official-repository", required=True,
                        help="Clone of Breaking-Bad-Dataset.github.io containing decompress.py.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--python", dest="decompressor_python", default=sys.executable,
                        help="Python from the authors' Breaking Bad decompressor environment.")
    parser.add_argument("--subset", choices=["everyday", "artifact", "both"], default="both")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    subsets = ("everyday", "artifact") if args.subset == "both" else (args.subset,)
    print(materialize_breaking_bad(
        args.manifest, args.compressed_root, args.official_repository, args.output_root, args.scratch_root,
        args.decompressor_python, subsets, args.keep_staging, args.dry_run,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
