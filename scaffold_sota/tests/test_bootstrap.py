"""Common CPU environment setup contracts; never installs packages in tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scaffold_sota import bootstrap
from scaffold_sota.config import load_config
from scaffold_sota.io import REPO, read_json, study_root, write_json


@pytest.fixture
def tools_fixture(tmp_path):
    root = study_root(tmp_path / "study", create=True)
    cfg = load_config()
    cfg["resources"].update(min_free_gib=0, cap_gib=1)
    return root, cfg


def test_planner_runs_without_any_third_party_dependencies(tmp_path):
    # -S excludes installed site-packages, reproducing the fresh VM failure.
    script = """import json, sys
from scaffold_sota.bootstrap import setup_tools
from scaffold_sota.config import load_config
from scaffold_sota.io import study_root
root = study_root(sys.argv[1], create=True)
cfg = load_config()
cfg['resources'].update(min_free_gib=0, cap_gib=1)
result = setup_tools(root, cfg)
assert not {'torch', 'numpy', 'yaml', 'trimesh', 'scipy'} & set(sys.modules)
print(json.dumps({'status': result['status'], 'python': result['python']}))
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO))
    result = subprocess.run([sys.executable, "-S", "-c", script, str(tmp_path / "dependency-free")],
                            cwd=str(REPO), env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["status"] == "planned"


def test_plan_targets_only_owned_prefix_and_cpu_wheels(tools_fixture, monkeypatch):
    root, cfg = tools_fixture
    monkeypatch.setattr(bootstrap.subprocess, "Popen", lambda *a, **k: pytest.fail("Planner must not run children"))
    result = bootstrap.setup_tools(root, cfg)
    prefix = root / "envs" / "tools"
    assert result["environment_prefix"] == str(prefix)
    assert result["python"] == str(bootstrap.tools_python(root))
    assert not prefix.exists()
    assert not (root / "environments/tools/ownership.json").exists()
    assert "--prefix" in result["commands"][0]
    assert str(prefix) in result["commands"][0]
    assert "python=3.10" in result["commands"][0]
    assert bootstrap.CPU_INDEX in result["commands"][1]
    assert bootstrap.TORCH_REQUIREMENT in result["commands"][1]
    assert all(command[0] == result["python"] for command in result["commands"][1:])
    assert not result["cpu_tools_verified"] and not result["native_models_verified"]
    assert not (root / ".active_job.lock").exists()


def test_existing_unowned_prefix_is_never_claimed(tools_fixture):
    root, cfg = tools_fixture
    prefix = root / "envs/tools"
    prefix.mkdir(parents=True)
    sentinel = prefix / "keep.txt"
    sentinel.write_text("unrelated environment", encoding="utf-8")
    with pytest.raises(ValueError, match="ownership marker"):
        bootstrap.setup_tools(root, cfg)
    assert sentinel.read_text(encoding="utf-8") == "unrelated environment"
    assert not (root / "environments/tools/ownership.json").exists()


def test_inherited_pip_redirection_cannot_modify_active_environment(tools_fixture, monkeypatch):
    root, _ = tools_fixture
    monkeypatch.setenv("PIP_TARGET", "/active/env/site-packages")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://unwanted.example")
    monkeypatch.setenv("PYTHONUSERBASE", "/active/user")
    original = dict(os.environ)
    env = bootstrap._tools_environment(root)
    assert "PIP_TARGET" not in env and "PIP_EXTRA_INDEX_URL" not in env
    assert "PYTHONUSERBASE" not in env
    assert env["PIP_CONFIG_FILE"] == os.devnull
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    for name in ("PIP_CACHE_DIR", "CONDA_PKGS_DIRS", "XDG_CACHE_HOME", "CONDA_ENVS_PATH", "TMPDIR"):
        Path(env[name]).relative_to(root)
    assert dict(os.environ) == original


def test_install_failure_records_log_and_preserves_owned_retry(tools_fixture, monkeypatch):
    root, cfg = tools_fixture
    monkeypatch.setattr(bootstrap.sys, "platform", "linux")
    monkeypatch.setattr(bootstrap, "_conda_executable", lambda execute: "test-only-conda")
    calls = []

    class FailedChild:
        returncode = 17

        def poll(self):
            return self.returncode

    def fail_child(command, **kwargs):
        owner = read_json(root / "environments/tools/ownership.json")
        assert owner["environment_prefix"] == str(root / "envs/tools")
        assert kwargs["start_new_session"] is True
        kwargs["stdout"].write("test-only solver failure\n")
        calls.append(command)
        return FailedChild()

    monkeypatch.setattr(bootstrap.subprocess, "Popen", fail_child)
    with pytest.raises(RuntimeError, match="setup-00.log"):
        bootstrap.setup_tools(root, cfg, execute=True)
    record = read_json(root / "environments/tools/setup.json")
    assert record["status"] == "failed" and not record["cpu_tools_verified"]
    assert "test-only solver failure" in (root / "environments/tools/setup-00.log").read_text(encoding="utf-8")
    assert len(calls) == 1 and not (root / ".active_job.lock").exists()
    # Ownership persists, but an incomplete prefix is never called installed.
    assert read_json(root / "environments/tools/smoke.json")["status"] == "pending"


def test_resource_abort_stops_only_own_process_group(tools_fixture, monkeypatch):
    root, cfg = tools_fixture
    monkeypatch.setattr(bootstrap.sys, "platform", "linux")
    monkeypatch.setattr(bootstrap, "_conda_executable", lambda execute: "test-only-conda")
    monkeypatch.setattr(bootstrap.time, "sleep", lambda seconds: None)
    calls, stopped = [], []

    class RunningChild:
        returncode = None

        def poll(self):
            return None

    child = RunningChild()

    class Guard:
        def __init__(self, *args):
            self.checks = 0

        def check(self, **kwargs):
            self.checks += 1
            if self.checks >= 3:
                raise RuntimeError("test-only storage limit")

    monkeypatch.setattr(bootstrap, "ResourceGuard", Guard)
    monkeypatch.setattr(bootstrap.subprocess, "Popen", lambda *a, **k: calls.append(a) or child)
    monkeypatch.setattr(bootstrap, "_stop_owned_child", lambda value: stopped.append(value))
    with pytest.raises(RuntimeError, match="storage limit"):
        bootstrap.setup_tools(root, cfg, execute=True)
    assert len(calls) == 1 and stopped == [child]
    assert read_json(root / "environments/tools/setup.json")["status"] == "failed"


def test_success_requires_fresh_smoke_and_keeps_native_unverified(tools_fixture, monkeypatch):
    root, cfg = tools_fixture
    monkeypatch.setattr(bootstrap.sys, "platform", "linux")
    monkeypatch.setattr(bootstrap, "_conda_executable", lambda execute: "test-only-conda")
    calls = []

    class SucceededChild:
        returncode = 0

        def poll(self):
            return 0

    def fake_child(command, **kwargs):
        calls.append(command)
        if "-c" in command:
            write_json(root / "environments/tools/smoke.json", {"status": "passed", "cpu_tools_verified": True})
        return SucceededChild()

    monkeypatch.setattr(bootstrap.subprocess, "Popen", fake_child)
    result = bootstrap.setup_tools(root, cfg, execute=True)
    assert result["status"] == "installed_cpu_tools_verified"
    assert result["cpu_tools_verified"] and not result["native_models_verified"]
    assert len(calls) == len(result["commands"])
    # A stale previous success cannot stand in for the new child import check.
    monkeypatch.setattr(bootstrap.subprocess, "Popen", lambda *a, **k: SucceededChild())
    with pytest.raises(RuntimeError, match="did not pass"):
        bootstrap.setup_tools(root, cfg, execute=True)
