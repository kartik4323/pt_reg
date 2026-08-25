from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelSpec:
    name: str
    data: dict[str, Any]

    @property
    def repository(self) -> str:
        return str(self.data["repository"])

    @property
    def revision(self) -> str:
        return str(self.data["revision"])

    @property
    def source_status(self) -> str:
        return str(self.data.get("source_status", "released"))

    @property
    def tracks(self) -> tuple[str, ...]:
        return tuple(self.data.get("tracks", ()))


def lock_path() -> Path:
    return PACKAGE_ROOT / "models.lock.yaml"


def load_registry(path: str | Path | None = None) -> dict[str, ModelSpec]:
    source = Path(path) if path else lock_path()
    with source.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict) or not isinstance(document.get("models"), dict):
        raise ValueError(f"Invalid SOTA lockfile: {source}")
    registry = {name: ModelSpec(name, dict(data)) for name, data in document["models"].items()}
    for name, spec in registry.items():
        if spec.source_status == "released" and len(spec.revision) != 40:
            raise ValueError(f"{name}: released source must be pinned to a full 40-character revision")
    return registry


def released_models(registry: dict[str, ModelSpec] | None = None) -> list[ModelSpec]:
    registry = registry or load_registry()
    return [spec for spec in registry.values() if spec.source_status == "released"]
