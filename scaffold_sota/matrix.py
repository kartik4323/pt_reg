"""Sealed dependencies for matched native and learned-input comparisons."""
from __future__ import annotations

import copy
import subprocess
from pathlib import Path

from .config import validate_config
from .io import PACKAGE, REPO, digest_json, owned_output, read_json, sha256_file, study_root, write_json
from .registry import MODEL_NAMES, model_spec
from .setup import environment_python, subprocess_environment


def run_name(model, stage, condition, seed):
    return "%s_%s_%s_seed%d" % (model, stage, condition, seed)


def _code_hash():
    # Identical to training.code_hash, without importing Torch in the planner.
    files = {str(p.relative_to(REPO)): sha256_file(p)
             for p in sorted(PACKAGE.rglob("*.py")) if "__pycache__" not in p.parts}
    for relative in ("reassembly/__init__.py", "reassembly/model.py", "reassembly/geometry.py",
                     "reassembly/prepare.py", "reassembly/repair/__init__.py", "reassembly/repair/model.py"):
        if (REPO / relative).is_file():
            files[relative] = sha256_file(REPO / relative)
    return digest_json(files)


def _profile(cfg, model, stage, seed):
    options = copy.deepcopy(cfg)
    options["train"]["seed"] = options["data"]["seed"] = int(seed)
    if model in ("puzzlefusion_pp", "diffassemble"):
        options["data"]["points_per_fragment"] = 1000
    policy = "study AdamW and constant learning rate; not the native training schedule"
    if model in ("garf", "puzzlefusion_pp"):
        pretrain = stage == "pretrain"
        options["train"].update(
            learning_rate=(1e-4 if model == "garf" else 5e-4) if pretrain else 2e-4,
            weight_decay=1e-5 if model == "garf" and pretrain else 1e-6,
            betas=[.9, .999] if model == "garf" and pretrain else [.95, .999], eps=1e-8)
        policy = "native stage AdamW hyperparameters; study constant learning rate and update budget"
    if model == "garf":
        options["train"]["amp"] = True  # Native FracSeg always uses CUDA FP16.
    validate_config(options)
    return options, policy


def compile_matrix(root, manifest, cfg, *, models=("jigsaw", "ccs", "garf"), seeds=(42,),
                   prior_v3=None, prior_v2=None, conditions=("B0", "B1", "B2", "B3", "B4"), output=None):
    root = study_root(root)
    manifest = Path(manifest).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError("The immutable bottle manifest must exist when compiling an executable matrix")
    models, seeds, conditions = list(models), list(seeds), list(conditions)
    if not models or len(set(models)) != len(models) or any(m not in MODEL_NAMES for m in models):
        raise ValueError("Matrix models must be nonempty, known and unique")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Matrix seeds must be unique nonnegative integers")
    if len(set(conditions)) != len(conditions) or any(c not in ("B0", "B1", "B2", "B3", "B4") for c in conditions):
        raise ValueError("Matrix adaptation conditions must be known and unique")
    if "gpat" in models:
        raise ValueError("GPAT requires its separate semantic target protocol, not the bottle matrix")
    output = owned_output(root, output or root / "matrices" / "pilot.json")
    if output.exists():
        raise ValueError("Matrix already exists; choose a new name to preserve the planned comparison")
    priors = {}
    for name, path in (("v3", prior_v3), ("v2", prior_v2)):
        if path:
            path = Path(path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError("An explicit prior must exist; omit v3 to plan pending jobs: " + str(path))
            priors[name] = {"path": str(path), "sha256": sha256_file(path)}
    fixed = {"manifest_hash": sha256_file(manifest), "registry_hash": sha256_file(REPO / "sota_repro/models.lock.yaml"),
             "code_hash": _code_hash()}
    jobs, omissions = [], []
    for model in models:
        selected = conditions
        if model == "cmnet":
            selected = [c for c in conditions if c in ("B0", "B1", "B2")]
            omissions.extend({"model": model, "condition": c, "reason": "CMNet supports pooled conditioning only"}
                             for c in conditions if c not in selected)
        for seed in seeds:
            feature, prerequisite = None, []
            stages = ["pretrain", "assembly"] if model in ("garf", "puzzlefusion_pp") else ["assembly"]
            for stage in stages:
                options, policy = _profile(cfg, model, stage, seed)
                config_path = root / "matrices" / ("%s_%s_seed%d.json" % (model, stage, seed))
                if config_path.exists() and read_json(config_path) != options:
                    raise ValueError("Matrix config already exists with different settings")
                write_json(config_path, options)
                base = [str(environment_python(root, model)), "-m", "scaffold_sota", "train", "--run-root", str(root),
                        "--manifest", str(manifest), "--model", model, "--config", str(config_path), "--stage", stage]
                if feature:
                    base += ["--feature-checkpoint", str(feature)]
                native = run_name(model, stage, "native", seed)
                for condition in (["native"] if stage == "pretrain" else ["native"] + selected):
                    name = run_name(model, stage, condition, seed)
                    command = base + ["--run-id", name, "--condition", condition]
                    initialize = root / "runs" / native / "best.pt" if condition != "native" else None
                    if initialize:
                        command += ["--initialize", str(initialize)]
                    needs_prior = condition in ("B2", "B3", "B4")
                    prior = priors.get("v3") if needs_prior else None
                    if prior:
                        command += ["--prior", prior["path"]]
                    native_options = dict(options["model"], condition=condition, stage=stage,
                                          num_tokens=options["prior"]["query_count"])
                    if feature:
                        native_options["feature_checkpoint"] = str(feature)
                    identity = dict(fixed, experiment_id="scaffold_sota", run_id=name, model=model, stage=stage,
                                    condition=condition, seed=seed, source_revision=model_spec(model)["revision"],
                                    cfg_hash=digest_json(options), options_hash=digest_json(native_options),
                                    prior_hash=prior["sha256"] if prior else None, cpu_smoke=False)
                    jobs.append({"id": name, "depends_on": prerequisite if condition == "native" else [native],
                                 "argv": command, "argv_hash": digest_json(command), "expected_identity": identity,
                                 "config_path": str(config_path), "feature_checkpoint": str(feature) if feature else None,
                                 "initialize": str(initialize) if initialize else None, "optimizer_policy": policy,
                                 "prior_required": needs_prior, "status": "awaiting_v3" if needs_prior and not prior else "pending"})
                if stage == "pretrain":
                    prerequisite = [native]
                    feature = root / "runs" / native / "best.pt"
    result = {"schema_version": 2, "experiment_id": "scaffold_sota", "run_root": str(root),
              "manifest": str(manifest), "priors": priors, "sealed": fixed, "jobs": jobs, "omissions": omissions,
              "note": "Training only; frozen controls/evaluation are separate. Recompile to supply newly available v3. Completed jobs require identical lineage."}
    write_json(output, result)
    return result


def _expected_identity(job):
    result = dict(job["expected_identity"])
    for argument, key in (("feature_checkpoint", "feature_hash"), ("initialize", "native_parent_hash")):
        result[key] = sha256_file(job[argument]) if job.get(argument) else None
    return result


def _validate_record(job, path, completed=False):
    record = read_json(path)
    identity = record.get("identity", {})
    changed = [key for key, value in _expected_identity(job).items() if identity.get(key) != value]
    if changed:
        raise ValueError("Existing job %s differs from this matrix on %s" % (job["id"], ", ".join(changed)))
    if completed:
        if record.get("status") != "completed":
            raise RuntimeError("Job returned successfully without a completed run record: " + job["id"])
        for name in ("best.pt", "latest.pt"):
            checkpoint = Path(path).parent / name
            if not checkpoint.is_file():
                raise RuntimeError("Completed job is missing " + name + ": " + job["id"])
            expected_hash = record.get("checkpoint_hashes", {}).get(name)
            if not isinstance(expected_hash, str) or sha256_file(checkpoint) != expected_hash:
                raise ValueError("Completed checkpoint hash is missing or changed: %s/%s" % (job["id"], name))
    return record


def execute_matrix(path, *, resume=False):
    matrix = read_json(path)
    if matrix.get("experiment_id") != "scaffold_sota" or matrix.get("schema_version") != 2:
        raise ValueError("Execution requires a sealed scaffold_sota schema 2 matrix; compile a fresh plan")
    root = study_root(matrix["run_root"])
    path = owned_output(root, path)
    actual = {"manifest_hash": sha256_file(matrix["manifest"]),
              "registry_hash": sha256_file(REPO / "sota_repro/models.lock.yaml"), "code_hash": _code_hash()}
    if actual != matrix["sealed"]:
        raise ValueError("Matrix code, manifest or source registry changed; compile a new plan")
    for prior in matrix["priors"].values():
        if sha256_file(prior["path"]) != prior["sha256"]:
            raise ValueError("A matrix prior changed after planning")
    from .__main__ import parser
    seen = set()
    for job in matrix["jobs"]:
        command, expected = job["argv"], job["expected_identity"]
        if not job["id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in job["id"]):
            raise ValueError("Invalid matrix run ID")
        if job["id"] in seen or not set(job["depends_on"]).issubset(seen):
            raise ValueError("Matrix jobs must be unique and in dependency order")
        seen.add(job["id"])
        if digest_json(command) != job["argv_hash"] or command[1:4] != ["-m", "scaffold_sota", "train"]:
            raise ValueError("Matrix command was changed after planning")
        args = parser().parse_args(command[3:])
        if (Path(command[0]) != environment_python(root, expected["model"]) or Path(args.run_root).resolve() != root
                or args.model != expected["model"] or args.run_id != job["id"] or args.stage != expected["stage"]
                or args.condition != expected["condition"] or Path(args.manifest).resolve() != Path(matrix["manifest"])
                or args.config != job["config_path"] or args.feature_checkpoint != job["feature_checkpoint"]
                or args.initialize != job["initialize"] or args.cpu_smoke or args.options or args.source
                or args.seed is not None or args.updates is not None or args.resume):
            raise ValueError("Matrix command does not match its planned identity")
        if digest_json(read_json(owned_output(root, job["config_path"]))) != expected["cfg_hash"]:
            raise ValueError("Matrix config changed after planning")
        if (sha256_file(args.prior) if args.prior else None) != expected["prior_hash"]:
            raise ValueError("Matrix command prior differs from its planned identity")
    status, complete = [], set()
    result_path = path.with_name(path.stem + ".execution.json")
    for job in matrix["jobs"]:
        record_path = owned_output(root, root / "runs" / job["id"] / "run.json")
        if job["status"] == "awaiting_v3" or not set(job["depends_on"]).issubset(complete):
            status.append({"id": job["id"], "status": "dependency_pending"})
            continue
        if record_path.exists():
            record = _validate_record(job, record_path)
            if record.get("status") == "completed":
                _validate_record(job, record_path, completed=True)
                complete.add(job["id"])
                status.append({"id": job["id"], "status": "completed", "reused": True})
                continue
            if not resume:
                raise ValueError("Existing incomplete job requires run-matrix --resume")
            if not (record_path.parent / "latest.pt").is_file():
                raise ValueError("Incomplete job has no resume checkpoint: " + job["id"])
        command = list(job["argv"])
        if not Path(command[0]).is_file():
            raise FileNotFoundError("Install the study-owned native environment before execution: " + command[0])
        if record_path.exists():
            command += ["--resume"]
        result = subprocess.run(command, cwd=str(REPO), env=subprocess_environment(root))
        if result.returncode:
            status.append({"id": job["id"], "status": "failed", "exit_code": result.returncode})
            write_json(result_path, status)
            raise RuntimeError("Matrix stopped at failed job " + job["id"])
        if not record_path.is_file():
            raise RuntimeError("Job returned successfully without a run record: " + job["id"])
        _validate_record(job, record_path, completed=True)
        complete.add(job["id"])
        status.append({"id": job["id"], "status": "completed"})
        write_json(result_path, status)
    write_json(result_path, status)
    return status
