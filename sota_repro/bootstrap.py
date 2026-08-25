from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

from .registry import ModelSpec, load_registry


def source_dir(suite_root: Path, model: str) -> Path:
    return suite_root / "models" / model / "upstream"


def _run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True, env=env)
    return completed.stdout.strip()


def validate_source(spec: ModelSpec, source: Path) -> dict[str, str]:
    if spec.source_status != "released":
        raise ValueError(f"{spec.name} has no verified public source")
    if not (source / ".git").exists():
        raise FileNotFoundError(f"Missing upstream source for {spec.name}: {source}")
    head = _run(["git", "rev-parse", "HEAD"], source)
    if head != spec.revision:
        raise RuntimeError(f"{spec.name}: expected {spec.revision}, found {head}")
    dirty = _run(["git", "status", "--porcelain"], source)
    if dirty:
        raise RuntimeError(f"{spec.name}: official source is dirty; put changes in models/{spec.name}/patches")
    return {"source": str(source), "revision": head, "clean": "true"}


def bootstrap(suite_root: Path, names: Iterable[str] | None = None) -> list[dict[str, str]]:
    registry = load_registry()
    requested = list(names or [spec.name for spec in registry.values() if spec.source_status == "released"])
    results: list[dict[str, str]] = []
    for name in requested:
        if name not in registry:
            raise KeyError(f"Unknown model: {name}")
        spec = registry[name]
        if spec.source_status != "released":
            raise ValueError(f"{name}: {spec.data.get('audit_note', 'source unavailable')}")
        destination = source_dir(suite_root, name)
        if destination.exists():
            results.append({"model": name, **validate_source(spec, destination)})
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        clone_env = os.environ.copy()
        # Official repositories sometimes place multi-gigabyte checkpoints in LFS.
        # The lockfile intentionally tracks source only; weights are downloaded by
        # the model-specific test guide/checkpoint command instead.
        clone_env["GIT_LFS_SKIP_SMUDGE"] = "1"
        _run(["git", "clone", "--recursive", spec.repository, str(destination)], env=clone_env)
        _run(["git", "checkout", "--detach", spec.revision], destination)
        _run(["git", "submodule", "update", "--init", "--recursive"], destination)
        results.append({"model": name, **validate_source(spec, destination)})
    return results
