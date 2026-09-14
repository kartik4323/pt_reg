"""Prerequisite planning and failure routing; no toolchain is installed here."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scaffold_sota import bootstrap, cuda_tools, setup
from scaffold_sota.config import load_config
from scaffold_sota.io import read_json, study_root


@pytest.fixture
def work(tmp_path):
    root = study_root(tmp_path / "study", create=True)
    cfg = load_config()
    cfg["resources"].update(min_free_gib=0, cap_gib=1)
    return root, cfg


def test_uv_bootstrap_pin_and_resolution(work, monkeypatch):
    root, _ = work
    assert "uv==0.9.26" in bootstrap.TOOLS_REQUIREMENTS
    assert "'-m', 'uv', '--version'" in bootstrap.SMOKE_SCRIPT
    monkeypatch.setattr(setup.shutil, "which", lambda *a, **k: None)
    assert setup._uv_executable(root) == "uv"
    with pytest.raises(RuntimeError, match="setup-tools"):
        setup._uv_executable(root, execute=True)
    binary = root / "envs/tools" / ("Scripts/uv.exe" if os.name == "nt" else "bin/uv")
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"test-only")
    assert setup._uv_executable(root, execute=True) == str(binary)


def test_private_compiler_found_without_shell_activation(work, monkeypatch):
    root, _ = work
    prefix = root / "toolchains/cuda-11.3"
    (prefix / "bin").mkdir(parents=True)
    for name in ("nvcc", "x86_64-conda-linux-gnu-gcc", "x86_64-conda-linux-gnu-g++"):
        (prefix / "bin" / name).write_bytes(b"test-only")
    monkeypatch.setattr(setup.shutil, "which", lambda *a, **k: None)
    monkeypatch.setattr(setup.subprocess, "check_output", lambda *a, **k: "Cuda compilation tools, release 11.3, V11.3.122")
    env = {"SCAFFOLD_SOTA_RUN_ROOT": str(root)}
    result = setup._cuda_compiler(env, root / "envs/ccs", "11.3")
    assert result["path"] == str(prefix)
    assert env["CC"].endswith("gcc") and env["CXX"].endswith("g++")
    assert str(prefix / "lib") in env["LD_LIBRARY_PATH"]


def test_toolchain_plan_is_private_no_driver(work, monkeypatch):
    root, cfg = work
    monkeypatch.setattr(cuda_tools.subprocess, "Popen", lambda *a, **k: pytest.fail("Planner must not execute"))
    plan = cuda_tools.setup_cuda(root, cfg)
    assert plan["prefix"] == str(root / "toolchains/cuda-11.3")
    assert "cuda-toolkit=11.3.1" in plan["command"]
    assert "gxx_linux-64=10" in plan["command"]
    assert not plan["installs_driver"] and not plan["gpu_verified"]
    assert not (root / "environments/cuda-11.3/ownership.json").exists()


def test_toolchain_rejects_unowned_existing_prefix(work):
    root, cfg = work
    (root / "toolchains/cuda-11.3").mkdir(parents=True)
    with pytest.raises(ValueError, match="ownership marker"):
        cuda_tools.setup_cuda(root, cfg)


def test_failure_logs_tail_and_maintains_retry_ownership(work, monkeypatch):
    root, cfg = work
    monkeypatch.setattr(cuda_tools.sys, "platform", "linux")
    monkeypatch.setattr(cuda_tools.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(cuda_tools, "_conda_executable", lambda execute: "test-only-conda")

    def fail(command, **kwargs):
        kwargs["stdout"].write("test-only unsatisfiable dependency\n")
        return SimpleNamespace(poll=lambda: 1, returncode=1)

    monkeypatch.setattr(cuda_tools.subprocess, "Popen", fail)
    with pytest.raises(RuntimeError, match="test-only unsatisfiable"):
        cuda_tools.setup_cuda(root, cfg, execute=True)
    assert read_json(root / "environments/cuda-11.3/setup.json")["status"] == "failed"
    assert (root / "environments/cuda-11.3/ownership.json").exists()
    assert not (root / ".active_job.lock").exists()


def test_zero_exit_without_compile_output_is_not_success(work, monkeypatch):
    root, cfg = work
    monkeypatch.setattr(cuda_tools.sys, "platform", "linux")
    monkeypatch.setattr(cuda_tools.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(cuda_tools, "_conda_executable", lambda execute: "test-only-conda")
    monkeypatch.setattr(cuda_tools.subprocess, "Popen", lambda *a, **k: SimpleNamespace(poll=lambda: 0, returncode=0))

    def compiler(env, prefix, release):
        env["CXX"] = str(prefix / "bin/g++")
        return {"path": str(prefix), "host_cxx": env["CXX"]}

    monkeypatch.setattr(cuda_tools, "_cuda_compiler", compiler)
    with pytest.raises(RuntimeError, match="no executable"):
        cuda_tools.setup_cuda(root, cfg, execute=True)
    assert not read_json(root / "environments/cuda-11.3/setup.json")["compiler_verified"]


def test_log_tail_keeps_actionable_end_only(tmp_path):
    path = tmp_path / "failure.log"
    path.write_text("\n".join("line%d" % i for i in range(100)), encoding="utf-8")
    assert setup._log_tail(path, count=2) == "line98\nline99"


def test_jigsaw_removes_only_deprecated_alias_and_retains_native_pins():
    path = setup.REPO / "sota_repro/models/jigsaw/upstream/environment.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    before = yaml.safe_dump(document)
    repaired = setup._jigsaw_environment(document)
    assert yaml.safe_dump(document) == before
    original_pip = next(item["pip"] for item in document["dependencies"] if isinstance(item, dict))
    repaired_pip = next(item["pip"] for item in repaired["dependencies"] if isinstance(item, dict))
    assert set(original_pip) - set(repaired_pip) == {"sklearn==0.0.post1"}
    assert [item for item in repaired["dependencies"] if isinstance(item, str)] == [item for item in document["dependencies"] if isinstance(item, str)]


def test_detected_v100_does_not_inherit_a5000_architecture(monkeypatch):
    monkeypatch.setattr(setup.subprocess, "check_output", lambda *a, **k: "Tesla V100-PCIE-32GB, 7.0\n")
    devices = setup._gpu_capabilities({})
    assert devices == [{"name": "Tesla V100-PCIE-32GB", "compute_capability": "7.0"}]
    with pytest.raises(RuntimeError, match="Ampere or newer"):
        setup._check_garf_hardware(devices, load_config())
    config = load_config()
    config["model"].update(encoder_flash=False, denoiser_flash=False)
    setup._check_garf_hardware(devices, config)
    setup._check_garf_hardware([{"name": "RTX A5000", "compute_capability": "8.6"}], load_config())


def test_cuda_visibility_is_respected(monkeypatch):
    calls = []
    monkeypatch.setattr(setup.subprocess, "check_output", lambda command, **kw: calls.append(command) or "A100, 8.0\n")
    assert setup._gpu_capabilities({"CUDA_VISIBLE_DEVICES": ""}) == []
    assert not calls
    setup._gpu_capabilities({"CUDA_VISIBLE_DEVICES": "1"})
    assert "--id=1" in calls[0]


def test_success_verifies_compiler_output_but_not_gpu(work, monkeypatch):
    root, cfg = work
    monkeypatch.setattr(cuda_tools.sys, "platform", "linux")
    monkeypatch.setattr(cuda_tools.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(cuda_tools, "_conda_executable", lambda execute: "test-only-conda")

    def compiler(env, prefix, release):
        env["CXX"] = str(prefix / "bin/g++")
        return {"path": str(prefix), "host_cxx": env["CXX"]}

    def child(command, **kwargs):
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"test-only-output")
        return SimpleNamespace(poll=lambda: 0, returncode=0)

    monkeypatch.setattr(cuda_tools, "_cuda_compiler", compiler)
    monkeypatch.setattr(cuda_tools.subprocess, "Popen", child)
    result = cuda_tools.setup_cuda(root, cfg, execute=True)
    assert result["compiler_verified"] and not result["gpu_verified"]
    assert result["status"] == "installed_compiler_verified"
    assert result["compile_probe_sha256"]
