"""Planner/setup contracts; no fixture represents a native CUDA model."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scaffold_sota import matrix, registry, setup
from scaffold_sota.config import load_config
from scaffold_sota.io import read_json, study_root, write_json, sha256_file


@pytest.fixture
def plan_fixture(tmp_path, monkeypatch):
    root = study_root(tmp_path / "study", create=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"test_only":true}', encoding="utf-8")
    monkeypatch.setattr(matrix, "_code_hash", lambda: "test-only-code")
    return root, manifest, load_config()


def seal_test_completion(root, job):
    folder = root / "runs" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    # Opaque placeholders test existence/hash bookkeeping only, never torch.load.
    for name in ("best.pt", "latest.pt"):
        (folder / name).write_bytes(b"test-only-checkpoint")
    write_json(folder / "run.json", {"status": "completed", "identity": matrix._expected_identity(job),
                                    "checkpoint_hashes": {name: sha256_file(folder / name)
                                                          for name in ("best.pt", "latest.pt")}})


def test_stage_profiles_and_pending_prior_are_explicit(plan_fixture):
    root, manifest, cfg = plan_fixture
    result = matrix.compile_matrix(root, manifest, cfg, models=["garf", "puzzlefusion_pp"])
    jobs = {job["id"]: job for job in result["jobs"]}
    gp = read_json(jobs["garf_pretrain_native_seed42"]["config_path"])
    ga = read_json(jobs["garf_assembly_native_seed42"]["config_path"])
    pp = read_json(jobs["puzzlefusion_pp_pretrain_native_seed42"]["config_path"])
    assert gp["train"]["amp"] and ga["train"]["amp"]
    assert gp["train"]["learning_rate"] == 1e-4 and ga["train"]["learning_rate"] == 2e-4
    assert ga["train"]["betas"] == [.95, .999]
    assert pp["train"]["learning_rate"] == 5e-4 and pp["data"]["points_per_fragment"] == 1000
    assert jobs["garf_assembly_native_seed42"]["depends_on"] == ["garf_pretrain_native_seed42"]
    assert jobs["garf_assembly_B0_seed42"]["initialize"].endswith("garf_assembly_native_seed42/best.pt") or jobs["garf_assembly_B0_seed42"]["initialize"].endswith("garf_assembly_native_seed42\\best.pt")
    assert jobs["garf_assembly_B3_seed42"]["status"] == "awaiting_v3"
    assert not cfg["train"]["amp"], "Planning must not mutate caller configuration"


@pytest.mark.parametrize("changes", [{"models": ["jigsaw", "jigsaw"]}, {"seeds": [42, 42]},
                                     {"conditions": ["B0", "B0"]}, {"models": ["gpat"]}])
def test_ambiguous_or_incompatible_matrix_fails_before_writes(plan_fixture, changes):
    root, manifest, cfg = plan_fixture
    with pytest.raises(ValueError):
        matrix.compile_matrix(root, manifest, cfg, **changes)
    assert not (root / "matrices").exists()


def test_changed_completed_lineage_is_not_reused(plan_fixture):
    root, manifest, cfg = plan_fixture
    plan = matrix.compile_matrix(root, manifest, cfg, models=["jigsaw"], conditions=[])
    job = plan["jobs"][0]
    seal_test_completion(root, job)
    result = matrix.execute_matrix(root / "matrices/pilot.json")
    assert result[0]["reused"]
    path = root / "runs" / job["id"] / "run.json"
    record = read_json(path)
    record["identity"]["prior_hash"] = "another-prior"
    write_json(path, record)
    with pytest.raises(ValueError, match="prior_hash"):
        matrix.execute_matrix(root / "matrices/pilot.json", resume=True)


def test_changed_prior_or_config_is_rejected_before_any_child(plan_fixture, monkeypatch):
    root, manifest, cfg = plan_fixture
    prior = manifest.with_name("prior.pt")
    prior.write_bytes(b"first")
    plan = matrix.compile_matrix(root, manifest, cfg, models=["jigsaw"], prior_v3=prior)
    monkeypatch.setattr(matrix.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected child"))
    prior.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="prior changed"):
        matrix.execute_matrix(root / "matrices/pilot.json")
    prior.write_bytes(b"first")
    config = read_json(plan["jobs"][0]["config_path"])
    config["data"]["hard_noise_std"] *= 2
    write_json(plan["jobs"][0]["config_path"], config)
    with pytest.raises(ValueError, match="config changed"):
        matrix.execute_matrix(root / "matrices/pilot.json")


def test_zero_exit_does_not_claim_training_completion(plan_fixture, monkeypatch):
    root, manifest, cfg = plan_fixture
    matrix.compile_matrix(root, manifest, cfg, models=["jigsaw"], conditions=[])
    python = setup.environment_python(root, "jigsaw")
    python.parent.mkdir(parents=True)
    python.write_bytes(b"not-an-interpreter")
    monkeypatch.setattr(matrix.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    with pytest.raises(RuntimeError, match="without a run record"):
        matrix.execute_matrix(root / "matrices/pilot.json")


def test_completed_job_requires_checkpoint_artifacts(plan_fixture):
    root, manifest, cfg = plan_fixture
    plan = matrix.compile_matrix(root, manifest, cfg, models=["jigsaw"], conditions=[])
    seal_test_completion(root, plan["jobs"][0])
    (root / "runs" / plan["jobs"][0]["id"] / "best.pt").unlink()
    with pytest.raises(RuntimeError, match="missing best.pt"):
        matrix.execute_matrix(root / "matrices/pilot.json")


@pytest.mark.parametrize("name", ["best.pt", "latest.pt"])
def test_completed_reuse_rejects_altered_checkpoint_bytes(plan_fixture, name):
    root, manifest, cfg = plan_fixture
    plan = matrix.compile_matrix(root, manifest, cfg, models=["jigsaw"], conditions=[])
    job = plan["jobs"][0]
    seal_test_completion(root, job)
    (root / "runs" / job["id"] / name).write_bytes(b"replacement-checkpoint")
    with pytest.raises(ValueError, match="checkpoint hash is missing or changed"):
        matrix.execute_matrix(root / "matrices/pilot.json", resume=True)


def test_completed_reuse_requires_recorded_checkpoint_hashes(plan_fixture):
    root, manifest, cfg = plan_fixture
    plan = matrix.compile_matrix(root, manifest, cfg, models=["jigsaw"], conditions=[])
    job = plan["jobs"][0]
    seal_test_completion(root, job)
    path = root / "runs" / job["id"] / "run.json"
    record = read_json(path)
    record.pop("checkpoint_hashes")
    write_json(path, record)
    with pytest.raises(ValueError, match="checkpoint hash is missing or changed"):
        matrix.execute_matrix(root / "matrices/pilot.json", resume=True)


def test_native_copy_rejects_changed_code_but_allows_build_products(tmp_path, monkeypatch):
    root = study_root(tmp_path / "study", create=True)
    source = tmp_path / "native"
    source.mkdir()
    (source / "model.py").write_text("NATIVE = True\n", encoding="utf-8")
    info = {"path": str(source), "revision": "test-only", "model": "jigsaw"}
    monkeypatch.setattr(registry, "source_info", lambda *args: info)
    monkeypatch.setattr(registry.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="model.py\0"))
    copied, _ = registry.materialize_source("jigsaw", root, root / "runs/native")
    (copied / "native_extension.so").write_bytes(b"test-only-build-product")
    registry.materialize_source("jigsaw", root, root / "runs/native")
    (copied / "model.py").write_text("NATIVE = False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="copy changed"):
        registry.materialize_source("jigsaw", root, root / "runs/native")
    assert (source / "model.py").read_text(encoding="utf-8") == "NATIVE = True\n"


def test_subprocess_python_isolation(tmp_path, monkeypatch):
    root = study_root(tmp_path / "study", create=True)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        monkeypatch.setenv(name, "/unrelated/pipeline")
    env = setup.subprocess_environment(root)
    assert env["PYTHONPATH"] == str(setup.REPO)
    assert env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONHOME" not in env and "PYTHONUSERBASE" not in env
    assert Path(env["TORCH_EXTENSIONS_DIR"]).parent == root


@pytest.mark.parametrize("model", ["garf", "ccs", "puzzlefusion_pp", "diffassemble", "gpat"])
def test_setup_plan_includes_real_dependency_chain(tmp_path, monkeypatch, model):
    root = study_root(tmp_path / "study", create=True)
    original = tmp_path / "explicit-pinned-source"
    original.mkdir()

    def materialize(name, run_root, work, passed_source):
        assert passed_source == original
        copy = Path(work) / "native_source"
        copy.mkdir(parents=True)
        for relative in ("pyproject.toml", "uv.lock", "requirements.txt", "environment.yml"):
            pinned = setup.REPO / "sota_repro/models" / name / "upstream" / relative
            (copy / relative).write_bytes(pinned.read_bytes() if pinned.exists() else b"test-only")
        return copy, {"path": str(original), "revision": "test-only"}

    monkeypatch.setattr(setup, "materialize_source", materialize)
    monkeypatch.setattr(setup, "ResourceGuard", lambda *a, **k: SimpleNamespace(check=lambda **kw: {}))
    plan = setup.setup_model(root, model, load_config(), source=original)
    commands = [" ".join(command) for command in plan["commands"]]
    assert plan["runtime_verified"] is False
    assert all(" -n sota-" not in command for command in commands)
    if model == "garf":
        assert any("uv pip check --python" in command for command in commands)
        assert not any("-m pip" in command for command in commands)
    elif model == "ccs":
        assert any("-c " in command and "constraints.repro.txt" in command for command in commands)
        assert any("pointnet2_ops_lib" in command for command in commands)
        assert any("chamfer" in command for command in commands)
        assert len(plan["cuda_build_steps"]) == 2
    elif model == "puzzlefusion_pp":
        for dependency in ("torch==2.0.1", "torch-cluster==", "pytorch3d.git@v0.7.4", "chamferdist==1.0.3"):
            assert any(dependency in command for command in commands)
        assert len(plan["cuda_build_steps"]) == 2
    elif model == "diffassemble":
        assert any("repair_runtime.py" in command for command in commands)
    elif model == "gpat":
        assert plan["status"] == "unsupported_legacy_runtime"
