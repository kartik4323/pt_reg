from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def byte_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_relative(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        return resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{path} is outside {root}") from exc


def copytree_or_link(source: Path, destination: Path, link: bool = False) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
            return
        except OSError:
            pass
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def flatten_command(command: Iterable[str], variables: dict[str, str]) -> list[str]:
    return [str(item).format(**variables) for item in command]


def environment_path(value: str | None, fallback: Path) -> Path:
    return Path(value).expanduser().resolve() if value else fallback.resolve()


def safe_rmtree(path: Path, permitted_root: Path) -> None:
    require_relative(path, permitted_root)
    if path.exists():
        shutil.rmtree(path)
