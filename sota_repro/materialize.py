from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .subset import _hash_tree
from .utils import byte_size, copytree_or_link, read_json, safe_rmtree, write_json


def materialize_common(manifest_path: str | Path, destination: str | Path, link: bool = False) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    destination = Path(destination).resolve()
    sources = {key: Path(value).resolve() for key, value in manifest["sources"].items()}
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for sample in manifest["samples"]:
        root = sources["partnet_gpat_root"] if sample["track"] == "partnet_gpat" else sources["breaking_bad_root"]
        source = root / sample["relative_path"]
        target = destination / sample["track"] / sample["relative_path"]
        if target.exists():
            if _hash_tree(target) != sample["source_sha256"]:
                raise RuntimeError(f"Existing materialized sample differs from manifest: {target}")
            copied.append(str(target.relative_to(destination)))
            continue
        if not source.exists():
            raise FileNotFoundError(f"Selected source vanished: {source}")
        copytree_or_link(source, target, link=link)
        if _hash_tree(target) != sample["source_sha256"]:
            raise RuntimeError(f"Materialized sample differs from manifest: {target}")
        copied.append(str(target.relative_to(destination)))
    # Native Breaking Bad loaders need official split files in their ordinary location.
    bb_source = sources["breaking_bad_root"] / "data_split"
    if bb_source.exists():
        copytree_or_link(bb_source, destination / "data_split", link=link)
    written = dict(manifest)
    written["sources"] = {"materialized_root": str(destination)}
    written["materialized_bytes"] = byte_size(destination)
    if written["materialized_bytes"] > written["retained_cap_bytes"]:
        raise RuntimeError("Materialized corpus exceeds configured retained-data cap")
    write_json(destination / "common_v1_manifest.json", written)
    return {"destination": str(destination), "samples": len(copied), "bytes": written["materialized_bytes"], "by_track": Counter(item["track"] for item in manifest["samples"])}


def _write_breaking_bad_indexes(data_root: Path, view: Path, model: str = "") -> None:
    """Write selected-only split and pattern lists understood by the official loaders."""
    manifest_path = data_root / "common_v1_manifest.json"
    if not manifest_path.exists():
        return
    manifest = read_json(manifest_path)
    split_lists: dict[tuple[str, str], set[str]] = {}
    pattern_lists: dict[tuple[str, str], set[str]] = {}
    for row in manifest["samples"]:
        if not str(row["track"]).startswith("breaking_bad_"):
            continue
        subset = str(row["track"])[len("breaking_bad_"):]
        split = str(row["split"])
        object_id = str(row["object_id"])
        # DiffAssemble joins IDs directly to datasets/breaking-bad and reads
        # category from path component 1. Other adapters use category/object.
        if model == "diffassemble" and not object_id.startswith(subset + "/"):
            object_id = subset + "/" + object_id
        split_lists.setdefault((subset, split), set()).add(object_id)
        relative = str(row["relative_path"])
        pattern_lists.setdefault((subset, split), set()).add(f"{int(row['num_parts']):03d} {relative}")
    split_root = view / "breaking_bad" / "data_split"
    list_root = view / "data_lists"
    for (subset, split), objects in split_lists.items():
        destination = split_root / f"{subset}.{split}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(sorted(objects)) + "\n"
        destination.write_text(content, encoding="utf-8")
        (view / "breaking_bad" / f"{subset}.{split}.txt").write_text(content, encoding="utf-8")
    for (subset, split), patterns in pattern_lists.items():
        destination = list_root / f"{subset}_{split}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(sorted(patterns)) + "\n", encoding="utf-8")


def create_native_view(model: str, data_root: str | Path, scratch_root: str | Path) -> Path:
    """Expose a model-specific data root without duplicating the retained corpus."""
    data_root = Path(data_root).resolve()
    scratch_root = Path(scratch_root).resolve()
    view = scratch_root / model / "data"
    if (view / "view_manifest.json").is_file():
        if model == "diffassemble":
            # Refresh legacy cached lists too; do not touch retained meshes.
            _write_breaking_bad_indexes(data_root, view, model)
        return view
    # A failed index build can leave links and directories behind. Finish the
    # view on retry; only the manifest written at the end marks it complete.
    view.mkdir(parents=True, exist_ok=True)
    breaking_bad = data_root / "breaking_bad_everyday" / "everyday"
    artifact = data_root / "breaking_bad_artifact" / "artifact"
    partnet = data_root / "partnet_gpat"
    if breaking_bad.exists():
        copytree_or_link(breaking_bad, view / "breaking_bad" / "everyday", link=True)
    if artifact.exists():
        copytree_or_link(artifact, view / "breaking_bad" / "artifact", link=True)
    _write_breaking_bad_indexes(data_root, view, model)
    if (view / "breaking_bad" / "data_split").exists():
        copytree_or_link(view / "breaking_bad" / "data_split", view / "data_split", link=True)
    if partnet.exists():
        copytree_or_link(partnet, view / "partnet", link=True)
    write_json(view / "view_manifest.json", {"model": model, "retained_root": str(data_root), "scratch_root": str(scratch_root), "selected_only": True})
    return view


def clean_native_view(model: str, scratch_root: str | Path) -> None:
    scratch_root = Path(scratch_root).resolve()
    safe_rmtree(scratch_root / model, scratch_root)
