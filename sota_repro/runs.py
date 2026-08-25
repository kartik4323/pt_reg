from __future__ import annotations

import datetime as dt
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .bootstrap import validate_source
from .registry import ModelSpec
from .utils import read_json, sha256_file, write_json


def gpu_profile() -> dict[str, str]:
    try:
        value = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=True, text=True, capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        value = "unavailable"
    return {"nvidia_smi": value}


def environment_lock(spec: ModelSpec, source: Path) -> list[str]:
    environment = spec.data["environment"]
    manager = environment["manager"]
    if manager == "conda":
        command = ["conda", "run", "--no-capture-output", "-n", str(environment["name"]), "python", "-m", "pip", "freeze"]
    elif manager == "uv":
        command = ["uv", "run", "--project", str(source), "python", "-m", "pip", "freeze"]
    else:
        command = [sys.executable, "-m", "pip", "freeze"]
    try:
        text = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return sorted(line for line in text.splitlines() if line)


def create_run(
    spec: ModelSpec,
    source: Path,
    run_root: Path,
    action: str,
    seed: int,
    data_manifest: Path | None,
    track: str | None = None,
) -> Path:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = run_root / f"{now}_{spec.name}_{action}_seed{seed}"
    counter = 1
    while run.exists():
        counter += 1
        run = run_root / f"{now}_{spec.name}_{action}_seed{seed}_{counter}"
    run.mkdir(parents=True)
    source_info = validate_source(spec, source)
    payload: dict[str, Any] = {
        "model": spec.name,
        "action": action,
        "track": track,
        "created_at": now,
        "seed": seed,
        "source": source_info,
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": gpu_profile(),
        "environment": dict(spec.data["environment"]),
        "environment_lock": environment_lock(spec, source),
        "status": "running",
    }
    if data_manifest and data_manifest.exists():
        payload["subset_manifest"] = {"path": str(data_manifest.resolve()), "sha256": sha256_file(data_manifest)}
    write_json(run / "run_manifest.json", payload)
    return run


def finish_run(run: Path, status: str, **extra: Any) -> None:
    path = run / "run_manifest.json"
    payload = read_json(path)
    payload.update(extra)
    payload["status"] = status
    payload["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(path, payload)


def record_checkpoint_hashes(run: Path, **paths: str | None) -> None:
    """Record checkpoint identity without copying potentially large checkpoint files."""
    payload = read_json(run / "run_manifest.json")
    hashes: dict[str, dict[str, str | None]] = {}
    for label, value in paths.items():
        if not value:
            continue
        path = Path(value).resolve()
        hashes[label] = {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None}
    payload["checkpoint_hashes"] = hashes
    write_json(run / "run_manifest.json", payload)
