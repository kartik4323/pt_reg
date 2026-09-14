"""Read-only pinned sources with study-owned runtime copies and adapters."""
from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path

from .io import REPO, sha256_file, write_json, read_json, owned_output, contained

MODEL_NAMES = ("jigsaw", "ccs", "garf", "pmtr", "puzzlefusion_pp", "diffassemble", "cmnet", "gpat")
CONDITIONS = ("native", "B0", "B1", "B2", "B3", "B4")


def registry():
    import yaml
    return yaml.safe_load((REPO / "sota_repro/models.lock.yaml").read_text(encoding="utf-8"))["models"]


def model_spec(name):
    if name not in MODEL_NAMES:
        raise ValueError("Unknown model: " + name)
    return registry()[name]


def source_info(name, source=None):
    spec = model_spec(name)
    source = Path(source or REPO / "sota_repro/models" / name / "upstream").resolve()
    if not source.is_dir():
        raise FileNotFoundError("Pinned source is missing: " + str(source))
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if revision != spec["revision"]:
        raise ValueError("%s source revision differs from pinned registry" % name)
    dirty = subprocess.run(["git", "-C", str(source), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise ValueError("Pinned native source must be clean: " + name)
    return {"path": str(source), "revision": revision, "model": name,
            "registry_sha256": sha256_file(REPO / "sota_repro/models.lock.yaml")}


def materialize_source(name, root, run, source=None):
    info = source_info(name, source)
    destination = owned_output(root, Path(run) / "native_source")
    marker = owned_output(root, Path(run) / "source.json")
    files_marker = owned_output(root, Path(run) / "source.files.json")
    # Hash pinned tracked files, not compiler products created in the owned copy.
    # A clean upstream checkout does not establish that an older copy is intact.
    original = Path(info["path"])
    tracked = subprocess.run(["git", "-C", str(original), "ls-files", "-z"],
                             capture_output=True, text=True, check=True).stdout.split("\0")
    files = {}
    for relative in filter(None, tracked):
        path = original / relative
        if not contained(path, original):
            raise ValueError("Pinned source contains a file link outside its checkout")
        if path.is_file():
            files[relative] = sha256_file(path)
    if destination.exists():
        if not marker.exists() or read_json(marker) != info:
            raise ValueError("Existing source copy has different provenance")
        if not files_marker.exists() or read_json(files_marker) != files:
            raise ValueError("Existing source copy lacks the matching pinned file manifest")
        for relative, expected in files.items():
            copied = destination / relative
            if not contained(copied, destination) or not copied.is_file() or sha256_file(copied) != expected:
                raise ValueError("Study native source copy changed: " + relative)
        return destination, info
    shutil.copytree(info["path"], destination, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".venv", "logs", "wandb"))
    write_json(files_marker, files)
    write_json(marker, info)
    return destination, info


def build_backend(name, source, options, device):
    if name not in MODEL_NAMES or options.get("condition", "native") not in CONDITIONS:
        raise ValueError("Unknown model/learned condition")
    module_name = "scaffold_sota.adapters." + name
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise RuntimeError("Native adapter not implemented: %s; no substitute model will be used" % name) from exc
        raise
    return module.build(Path(source), options, device)
