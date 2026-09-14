"""Exercise the user's startup commands with all site-packages disabled."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scaffold_sota.io import REPO, study_root


@pytest.fixture
def startup(tmp_path):
    root = study_root(tmp_path / "study", create=True)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    sources, patterns = [], []
    for kind, records in (("source", sources), ("pattern", patterns)):
        path = prepared / (kind + ".npz")
        # check-data verifies immutable bytes, not NPZ loading or training.
        path.write_bytes(("test-only-" + kind).encode())
        record = {"source_id": "source-a", "path": path.name,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if kind == "pattern":
            record.update(pattern_id="pattern-a", split="train", pieces=2, cut_family="random", band="hard")
        records.append(record)
    document = {"schema_version": 2, "sdf_convention": "negative_inside", "seed": 42,
                "config": {}, "sources": sources, "patterns": patterns}
    identity = {"schema_version": 2, "sources": sources, "records": patterns, "seed": 42, "config": {}}
    document["fingerprint"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = prepared / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"resources": {"min_free_gib": 0, "cap_gib": 1}}), encoding="utf-8")
    return root, prepared, manifest, config


def cli(*args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-S", "-m", "scaffold_sota", *map(str, args)],
                          cwd=str(REPO), env=env, capture_output=True, text=True, timeout=30)


def test_check_data_does_not_need_torch_or_numpy(startup):
    root, _, manifest, _ = startup
    result = cli("check-data", "--run-root", root, "--manifest", manifest)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["verification"]["status"] == "verified"
    assert report["sources"] == report["patterns"] == 1
    assert (root / "data_verification.json").is_file()


def test_placeholder_error_is_actionable_without_import_traceback(startup):
    root, _, _, _ = startup
    result = cli("check-data", "--run-root", root, "--manifest", "/absolute/path/to/bottle/manifest.json")
    assert result.returncode == 2
    assert "find-data" in result.stderr
    assert "Traceback" not in result.stderr
    assert "torch" not in result.stderr


def test_find_path_selects_only_a_single_candidate(startup):
    _, prepared, manifest, _ = startup
    result = cli("find-data", "--search-root", prepared, "--path")
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == manifest
    (prepared / "second_manifest.json").write_bytes(manifest.read_bytes())
    result = cli("find-data", "--search-root", prepared, "--path")
    assert result.returncode == 2
    assert not result.stdout.strip()
    assert "narrow --search-root" in result.stderr


def test_tools_plan_runs_without_ml_or_yaml(startup):
    root, _, _, config = startup
    result = cli("setup-tools", "--run-root", root, "--config", config)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "planned"


def test_missing_optional_dependency_points_to_separate_environment(startup):
    root, _, _, _ = startup
    result = cli("compare", "--run-root", root, "--baseline", "unused.jsonl",
                 "--treatment", "unused.jsonl", "--output", root / "report.json")
    assert result.returncode == 2
    assert "Missing dependency" in result.stderr
    assert "setup-tools" in result.stderr
    assert "Traceback" not in result.stderr
