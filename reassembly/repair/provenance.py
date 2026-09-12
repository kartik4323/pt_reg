"""Small, hash-bound experiment inventories and consistent artifact writing."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from reassembly.prepare import _file_sha256
from reassembly.resources import jsonable


def code_inventory():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "reassembly").rglob("*.py"))
    files = {str(p.relative_to(root)).replace("\\", "/"): _file_sha256(p) for p in paths}
    return {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def provenance(device):
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    device = torch.device(device)
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "git_revision": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"), "code": code_inventory(), "python": sys.version,
        "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda, "float32_matmul_precision": torch.get_float32_matmul_precision()}


def append_jsonl(path, value, guard=None):
    content = json.dumps(jsonable(value), allow_nan=False) + "\n"
    if guard:
        guard.check(additional_bytes=len(content.encode()))
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
