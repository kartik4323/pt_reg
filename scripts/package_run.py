#!/usr/bin/env python3
"""Create a self-contained experiment archive without raw PartNet data."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


EXCLUDED_DIRECTORY_NAMES = {
    "partnet_raw", "partnet_meta", "partnet_gpat_processed", "shape_processed", "downloads",
}


def allowed(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(member.name).parts
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in parts):
        return None
    if member.name.endswith("run_bundle.tar.zst") or member.name.endswith("run_bundle.tar"):
        return None
    # A result bundle should contain materialized artifacts, not references to
    # a VM-only raw-data path outside the immutable run folder.
    if member.issym() or member.islnk():
        return None
    return member


def main() -> None:
    parser = argparse.ArgumentParser(description="Package an immutable run as run_bundle.tar.zst")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--keep-tar", action="store_true", help="Retain the intermediate uncompressed .tar")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not (run_dir / "RESULTS.md").exists():
        raise RuntimeError("Run directory lacks RESULTS.md; package only completed/inspected immutable runs")
    output = Path(args.output).resolve() if args.output else run_dir / "run_bundle.tar.zst"
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="partnet_run_bundle_") as temporary:
        tar_path = Path(temporary) / "run_bundle.tar"
        with tarfile.open(tar_path, "w") as archive:
            archive.add(run_dir, arcname=run_dir.name, recursive=True, filter=allowed)
        try:
            import zstandard  # type: ignore

            with open(tar_path, "rb") as source, open(output, "wb") as destination:
                zstandard.ZstdCompressor(level=10, threads=0).copy_stream(source, destination)
        except ImportError:
            zstd = shutil.which("zstd")
            if not zstd:
                raise RuntimeError("Need python package 'zstandard' or the 'zstd' executable to create .tar.zst")
            subprocess.run([zstd, "--quiet", "--force", "-10", "-o", str(output), str(tar_path)], check=True)
        if args.keep_tar:
            shutil.copy2(tar_path, output.with_suffix(""))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
