"""Read-only manifest discovery and verification without scientific packages.

Startup checks intentionally use only the Python standard library. Discovery
checks metadata and file availability; only ``verify_manifest`` hashes assets.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath


_SPLITS = ("train", "val", "test", "cut_holdout")
_SKIP_DIRECTORIES = {
    "envs", "venv", "virtualenv", "node_modules", "site-packages", "upstream",
    "cache", "caches", "__pycache__", "anaconda3", "miniconda3", "miniforge3",
    "mambaforge", "pkgs", "wandb", "native_source",
}
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024


def manifest_fingerprint(document):
    """Exactly match the immutable ``reassembly.prepare`` identity equation."""
    identity = {"schema_version": document["schema_version"], "records": document["patterns"],
                "sources": document["sources"], "seed": document["seed"], "config": document["config"]}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(manifest):
    path = Path(manifest).expanduser().resolve()
    if not path.is_file():
        raise ValueError(
            "Prepared manifest does not exist: {}. Replace the example "
            "'/absolute/path/to/bottle/manifest.json' with your real prepared manifest; "
            "use 'python -m scaffold_sota find-data --search-root /your/data/root' to locate it."
            .format(path))
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("Manifest exceeds the 64 MiB startup metadata limit: {}".format(path))
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Malformed prepared manifest JSON: {}".format(path)) from exc
    if not isinstance(document, dict):
        raise ValueError("Malformed prepared manifest: expected a JSON object")
    return path, document


def _asset_path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Manifest asset paths must be nonempty relative strings")
    if Path(relative).is_absolute() or PureWindowsPath(relative).drive or PureWindowsPath(relative).root:
        raise ValueError("Manifest paths must be relative to the prepared dataset")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Manifest contains a path outside the prepared dataset") from exc
    return path


def _metadata(path, document):
    if document.get("schema_version") != 2 or document.get("sdf_convention") != "negative_inside":
        raise ValueError("Manifest requires schema_version=2 and negative-inside SDF")
    if not isinstance(document.get("config"), dict) or not isinstance(document.get("seed"), int):
        raise ValueError("Malformed prepared manifest: expected preparation config and integer seed")
    for group in ("sources", "patterns"):
        if not isinstance(document.get(group), list) or not document[group]:
            raise ValueError("Malformed prepared manifest: {} must be a nonempty list".format(group))
    seen_paths, sources, pattern_ids, assets = set(), {}, set(), []
    identity_split, content_split = {}, {}
    for group in ("sources", "patterns"):
        for record in document[group]:
            if not isinstance(record, dict):
                raise ValueError("Malformed prepared manifest: asset records must be objects")
            source_id = record.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("Malformed prepared manifest: source_id must be a nonempty string")
            asset = _asset_path(path.parent, record.get("path"))
            if asset in seen_paths:
                raise ValueError("Manifest references duplicate asset path: {}".format(record["path"]))
            seen_paths.add(asset)
            expected = record.get("sha256")
            if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                raise ValueError("Malformed prepared manifest: asset SHA-256 must be 64 lowercase hexadecimal characters")
            assets.append((asset, expected, record["path"]))
            if group == "sources":
                if source_id in sources:
                    raise ValueError("Duplicate source identity in manifest")
                sources[source_id] = record
                continue
            if source_id not in sources:
                raise ValueError("Pattern references unknown source identity")
            pattern_id = record.get("pattern_id")
            if not isinstance(pattern_id, str) or not pattern_id:
                raise ValueError("Malformed prepared manifest: pattern_id must be a nonempty string")
            if pattern_id in pattern_ids:
                raise ValueError("Duplicate pattern identity in manifest")
            pattern_ids.add(pattern_id)
            if type(record.get("pieces")) is not int or record["pieces"] not in (2, 3):
                raise ValueError("Manifest contains an incomplete/unsupported fragment set; expected two or three pieces")
            split = record.get("split")
            if split not in _SPLITS:
                raise ValueError("Manifest has an unknown data split")
            if not isinstance(record.get("cut_family"), str) or not record["cut_family"]:
                raise ValueError("Malformed prepared manifest: cut_family is required")
            effective = "test" if split == "cut_holdout" else split
            content = sources[source_id]["sha256"]
            if source_id in identity_split and identity_split[source_id] != effective:
                raise ValueError("Source object leaks across dataset splits")
            if content in content_split and content_split[content] != effective:
                raise ValueError("Identical source content leaks across study splits")
            if record["cut_family"] == "heldout_radial" and effective != "test":
                raise ValueError("Held-out cut family leaked into training or validation")
            identity_split[source_id], content_split[content] = effective, effective
    return assets


def verify_manifest(manifest):
    """Verify all asset hashes, identity metadata, and study split separation."""
    path, document = _read_manifest(manifest)
    assets = _metadata(path, document)
    total_bytes = 0
    for asset, expected, relative in assets:
        if not asset.is_file():
            raise ValueError("Missing prepared asset: {}".format(relative))
        if _file_sha256(asset) != expected:
            raise ValueError("Prepared asset SHA-256 mismatch: {}".format(relative))
        total_bytes += asset.stat().st_size
    fingerprint = manifest_fingerprint(document)
    if fingerprint != document.get("fingerprint"):
        raise ValueError("Dataset fingerprint mismatch: assignments or preparation metadata changed")
    return {"status": "verified", "dataset_fingerprint": fingerprint,
            "verified_sources": len(document["sources"]), "verified_patterns": len(document["patterns"]),
            "verified_bytes": total_bytes, "source_split_separation": "verified",
            "content_split_separation": "verified"}


def check_data(manifest):
    """Return the CLI's original result schema without loading point clouds."""
    verification = verify_manifest(manifest)
    return {"fingerprint": verification["dataset_fingerprint"], "verification": verification,
            "patterns": verification["verified_patterns"], "sources": verification["verified_sources"]}


def inspect_manifest(manifest):
    """Inspect candidate metadata and asset availability, without hashing assets."""
    path, document = _read_manifest(manifest)
    return _inspect_document(path, document)


def _inspect_document(path, document):
    assets = _metadata(path, document)
    missing = [relative for asset, _, relative in assets if not asset.is_file()]
    fingerprint = manifest_fingerprint(document)
    return {"path": str(path), "fingerprint": document.get("fingerprint"),
            "fingerprint_matches": fingerprint == document.get("fingerprint"),
            "sources": len(document["sources"]), "patterns": len(document["patterns"]),
            "assets_present": not missing, "missing_asset_count": len(missing),
            "missing_assets": missing[:20], "asset_hashes_verified": False,
            "inspection": "metadata_and_availability_only"}


def find_manifests(search_roots=None, max_depth=8, max_files=50000):
    """Bounded read-only discovery; never select a candidate or follow dir links.

    Depth zero includes files directly under a search root. Reaching a limit or
    encountering an unreadable path is reported, so callers cannot mistake an
    incomplete search for a uniquely identified dataset.
    """
    if type(max_depth) is not int or max_depth < 0 or type(max_files) is not int or max_files < 1:
        raise ValueError("Search max_depth must be nonnegative and max_files must be positive")
    if search_roots is None:
        search_roots = [Path.home()]
        if os.environ.get("REASSEMBLY_ROOT"):
            search_roots.append(os.environ["REASSEMBLY_ROOT"])
    elif isinstance(search_roots, (str, os.PathLike)):
        search_roots = [search_roots]
    roots = list(dict.fromkeys(str(Path(root).expanduser().absolute()) for root in search_roots))
    if not roots:
        raise ValueError("At least one search root is required")
    report = {"search_roots": roots, "max_depth": max_depth, "max_files": max_files,
              "candidates": [], "truncated": False, "errors": [], "scanned_files": 0,
              "scanned_directories": 0, "skipped_directories": 0, "ignored_manifests": 0,
              "scanned_entries": 0, "max_entries": 2 * max_files,
              "inspection": "metadata_and_availability_only"}
    seen_dirs, seen_candidates = set(), set()

    def problem(path, message, incomplete=False):
        # A corrupt manifest remains an actionable report, even though it is not
        # a compatible candidate. Bound diagnostic storage during broad scans.
        if len(report["errors"]) < 100:
            report["errors"].append({"path": str(path), "error": str(message)})
        else:
            report["truncated"] = True
        if incomplete:
            report["truncated"] = True

    def inspect(path):
        if not (path.name.lower().endswith(".json") and "manifest" in path.name.lower()):
            return
        resolved = path.resolve()
        if resolved in seen_candidates:
            return
        seen_candidates.add(resolved)
        try:
            manifest_path, document = _read_manifest(path)
            if document.get("schema_version") != 2 or document.get("sdf_convention") != "negative_inside":
                report["ignored_manifests"] += 1
                return
            report["candidates"].append(_inspect_document(manifest_path, document))
        except (ValueError, OSError, RuntimeError) as exc:
            problem(path, exc)

    for root_string in roots:
        root = Path(root_string)
        if root.is_symlink():
            problem(root, "Directory/file symlink search roots are not traversed; provide the resolved data root", True)
            continue
        if root.is_file():
            if report["scanned_files"] >= max_files:
                report["truncated"] = True
                break
            report["scanned_files"] += 1
            inspect(root)
            continue
        if not root.is_dir():
            problem(root, "Search root does not exist or is not readable", True)
            continue
        stack = [(root, 0)]
        while stack:
            if (report["scanned_files"] >= max_files or report["scanned_directories"] >= max_files
                    or report["scanned_entries"] >= report["max_entries"]):
                report["truncated"] = True
                break
            directory, depth = stack.pop()
            resolved = directory.resolve()
            if resolved in seen_dirs:
                continue
            seen_dirs.add(resolved)
            report["scanned_directories"] += 1
            try:
                with os.scandir(str(directory)) as entries:
                    for entry in entries:
                        if report["scanned_entries"] >= report["max_entries"]:
                            report["truncated"] = True
                            break
                        report["scanned_entries"] += 1
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name.startswith(".") or entry.name.lower() in _SKIP_DIRECTORIES:
                                report["skipped_directories"] += 1
                                continue
                            if depth >= max_depth or len(stack) + report["scanned_directories"] >= max_files:
                                report["truncated"] = True
                                continue
                            stack.append((Path(entry.path), depth + 1))
                        elif entry.is_file(follow_symlinks=False):
                            if report["scanned_files"] >= max_files:
                                report["truncated"] = True
                                break
                            report["scanned_files"] += 1
                            inspect(Path(entry.path))
            except (OSError, RuntimeError) as exc:
                problem(directory, exc, True)
    report["candidates"].sort(key=lambda candidate: candidate["path"])
    return report
