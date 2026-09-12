"""Bounded disk usage, atomic reports, and reproducible experiment utilities."""
from __future__ import annotations

import json
import os
import random
import shutil
import stat
from pathlib import Path

import numpy as np
import torch

GIB = 1024 ** 3


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: str | Path, value, guard=None) -> None:
    path = Path(path)
    content = json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n"
    if guard:
        guard.check(additional_bytes=len(content.encode()))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def tree_bytes(root: Path) -> int:
    """Measure live files, tolerating removal during atomic report replacement.

    Directory enumeration is not a snapshot: a writer may rename a temporary
    file before we stat it. Ignore only vanished entries; permission and other
    I/O errors must still stop resource checks. Never follow symbolic links.
    """
    root = Path(root)
    try:
        info = root.lstat()
    except FileNotFoundError:
        return 0
    if stat.S_ISREG(info.st_mode):
        return info.st_size
    if stat.S_ISLNK(info.st_mode):
        return 0

    def walk_error(error):
        if not isinstance(error, FileNotFoundError):
            raise error

    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        for name in files:
            path = Path(directory) / name
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


class ResourceGuard:
    def __init__(self, roots, cap_gib=40, min_free_gib=50):
        candidates = sorted({Path(p).expanduser().resolve() for p in roots}, key=lambda p: len(p.parts))
        self.roots = []
        for path in candidates:
            if not any(path == root or root in path.parents for root in self.roots):
                self.roots.append(path)
        self.cap_bytes = int(float(cap_gib) * GIB)
        self.min_free_bytes = int(float(min_free_gib) * GIB)

    def check(self, additional_bytes=0) -> dict:
        used = sum(tree_bytes(root) for root in self.roots)
        additional = max(0, int(additional_bytes))
        if used + additional > self.cap_bytes:
            raise RuntimeError(f"Managed storage cap exceeded: {used + additional} > {self.cap_bytes} bytes")
        volumes = []
        for root in self.roots:
            existing = root
            while not existing.exists():
                existing = existing.parent
            free = shutil.disk_usage(existing).free
            if free - additional < self.min_free_bytes:
                raise RuntimeError(
                    f"Insufficient free space for {root}: need {self.min_free_bytes + additional} bytes, have {free}. "
                    "Choose a managed root on the filesystem with adequate free space; /data is not the default."
                )
            volumes.append({"path": str(root), "free_bytes": free})
        return {"managed_bytes": used, "cap_bytes": self.cap_bytes, "min_free_bytes": self.min_free_bytes, "volumes": volumes}


def make_guard(cfg: dict, *extra_roots) -> ResourceGuard:
    r = cfg["resources"]
    return ResourceGuard([r["managed_root"], *extra_roots], r["cap_gib"], r["min_free_gib"])
