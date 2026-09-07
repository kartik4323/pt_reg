from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .bootstrap import source_dir, validate_source
from .materialize import clean_native_view, create_native_view
from .registry import load_registry
from .runs import create_run, environment_lock, finish_run, record_checkpoint_hashes
from .utils import copytree_or_link, flatten_command, write_json


def suite_root() -> Path:
    return Path(__file__).resolve().parent


def _visible_gpu(index: str, visible: str | None) -> str:
    if not index.isdigit():
        raise ValueError("DiffAssemble --gpu must be a nonnegative logical device index")
    if visible is None:
        return index
    if visible.strip() in {"", "-1"}:
        return visible  # Respect an explicitly disabled GPU allocation.
    devices = [value.strip() for value in visible.split(",")]
    if int(index) >= len(devices):
        raise ValueError(f"--gpu {index} is outside CUDA_VISIBLE_DEVICES={visible!r}")
    return devices[int(index)]


def _conda_env_exists(name: str, environment: dict) -> bool:
    try:
        payload = subprocess.check_output(
            ["conda", "env", "list", "--json"], env=environment, text=True
        )
        environments = json.loads(payload).get("envs", [])
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return False
    return any(Path(path).name == name for path in environments)


def _skip_existing_conda_create(commands: list[list[str]], env_name: str, environment: dict) -> list[list[str]]:
    """Let interrupted multi-command setup recipes resume in an existing env."""
    if not _conda_env_exists(env_name, environment):
        return commands
    creation_prefixes = (("conda", "create"), ("conda", "env", "create"))
    retained = [command for command in commands if not any(tuple(command[:len(prefix)]) == prefix
                                                            for prefix in creation_prefixes)]
    if len(retained) != len(commands):
        print(f"Conda environment {env_name!r} already exists; resuming after its creation step.", flush=True)
    return retained


def _run_logged(commands, cwd: Path, environment: dict, log_path: Path) -> None:
    """Tee native diagnostics to the terminal and the persistent run log."""
    print(f"Native log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as output:
        for command in commands:
            header = "$ " + " ".join(command) + "\n"
            print(header, end="", flush=True)
            output.write(header)
            output.flush()
            try:
                with subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                      errors="replace", bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        output.write(line)
                        output.flush()
                    code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            except (OSError, subprocess.CalledProcessError) as exc:
                message = f"Command failed: {exc}\nFull native output: {log_path}\n"
                output.write(message)
                print(message, file=sys.stderr, flush=True)
                raise


def _native_worktree(source: Path, run: Path) -> Path:
    """Copy source for a run so native scripts cannot write to the pinned checkout."""
    worktree = run / "native_source"
    shutil.copytree(source, worktree, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    return worktree


def _native_build_source(source: Path, model_dir: Path) -> Path:
    """Keep editable installs and extension builds outside the pinned source clone."""
    build_source = model_dir / "native_build"
    if not build_source.exists():
        shutil.copytree(source, build_source, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    return build_source


def _replace_with_link(target: Path, source: Path, worktree: Path) -> None:
    """Link selected data into the disposable source copy only."""
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        from .utils import safe_rmtree
        safe_rmtree(target, worktree)
    copytree_or_link(source, target, link=True)


def _attach_model_data(model: str, worktree: Path, native_view: Path | None) -> None:
    if native_view is None:
        return
    breaking_bad = native_view / "breaking_bad"
    if model in {"jigsaw", "ccs"} and breaking_bad.exists():
        _replace_with_link(worktree / "data" / "breaking_bad", breaking_bad, worktree)
    elif model == "diffassemble" and breaking_bad.exists():
        _replace_with_link(worktree / "datasets" / "breaking-bad", breaking_bad, worktree)
    elif model == "puzzlefusion_pp" and breaking_bad.exists():
        _replace_with_link(worktree / "data" / "breaking_bad", breaking_bad, worktree)
    elif model == "gpat" and (native_view / "partnet").exists():
        _replace_with_link(worktree / "dataset" / "partnet", native_view / "partnet", worktree)
    if model in {"pmtr", "cmnet"} and (native_view / "data_lists").exists():
        list_dir = worktree / "data" / "data_list"
        list_dir.mkdir(parents=True, exist_ok=True)
        for selected in (native_view / "data_lists").glob("*.txt"):
            shutil.copy2(selected, list_dir / selected.name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute one immutable official-model run")
    parser.add_argument("--model", required=True)
    parser.add_argument("--action", required=True, choices=["setup", "smoke", "train", "test"])
    parser.add_argument("--data-root", default=os.environ.get("SOTA_DATA_ROOT", "./sota_data"))
    parser.add_argument("--run-root", default=os.environ.get("SOTA_RUN_ROOT", "./sota_runs"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--verifier-checkpoint", default=None,
                        help="PuzzleFusion++ verifier checkpoint; required for its full auto-agglomerative test.")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--category", default="chair")
    parser.add_argument("--track", default=None,
                        help="Common protocol track, e.g. breaking_bad_everyday, breaking_bad_artifact, or partnet_gpat.")
    parser.add_argument("--native-data", default=None,
                        help="Model-native prepared data path, such as a GARF HDF5 cache, stored under SOTA_DATA_ROOT.")
    parser.add_argument("--scratch-root", default=os.environ.get("SOTA_SCRATCH_ROOT"))
    parser.add_argument("--keep-native-view", action="store_true",
                        help="Keep the disposable per-model native view after the run for diagnosis.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry = load_registry()
    spec = registry[args.model]
    if args.track:
        required_track = "breaking_bad" if args.track.startswith("breaking_bad_") else "partnet" if args.track == "partnet_gpat" else args.track
        if required_track not in spec.data.get("tracks", []):
            raise ValueError(f"{spec.name} is not registered for track {args.track}")
    if spec.name == "diffassemble" and args.track not in {None, "breaking_bad_everyday"}:
        raise ValueError("DiffAssemble's audited native entry supports everyday only; artifact evaluation is not wired up.")
    if spec.name == "diffassemble" and args.action == "test" and not args.checkpoint:
        raise ValueError("DiffAssemble test requires --checkpoint /absolute/path/model.ckpt")
    source = source_dir(suite_root(), spec.name)
    validate_source(spec, source)
    model_dir = suite_root() / "models" / spec.name
    generated = model_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    environment_file = generated / "common_v1.env"
    environment_file.write_text(
        f"SOTA_DATA_ROOT={Path(args.data_root).resolve()}\nSOTA_MODEL={spec.name}\nSOTA_SEED={args.seed}\n",
        encoding="utf-8",
    )
    template = spec.data.get("config_template")
    config = Path(args.config).resolve() if args.config else (source / template if template else environment_file)
    env_name = str(spec.data["environment"]["name"])
    track = args.track or ("partnet_gpat" if spec.data.get("tracks") == ["partnet"] else "breaking_bad_everyday")
    data_category = "artifact" if track == "breaking_bad_artifact" else "everyday"
    variables = {
        "source": str(source), "data": str(Path(args.data_root).resolve()), "config": str(config),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else "", "resume": args.resume or "",
        "verifier_checkpoint": str(Path(args.verifier_checkpoint).resolve()) if args.verifier_checkpoint else "",
        "gpu": str(args.gpu), "category": args.category, "track": track, "data_category": data_category,
        "env": env_name, "run_name": f"sota_{spec.name}", "model_dir": str(model_dir),
    }
    commands = spec.data["commands"].get(args.action)
    if not commands:
        raise ValueError(f"No {args.action} command registered for {spec.name}")
    if args.action == "setup":
        setup_source = source if args.dry_run else _native_build_source(source, model_dir)
        variables["source"] = str(setup_source)
        if not args.config and template:
            variables["config"] = str(setup_source / template)
        run = create_run(spec, source, Path(args.run_root).resolve(), args.action, args.seed, None, track)
        variables["run"] = str(run)
        record_checkpoint_hashes(run, checkpoint=args.checkpoint, verifier_checkpoint=args.verifier_checkpoint, resume=args.resume)
        resolved_commands = [flatten_command(raw, variables) for raw in commands]
        write_json(run / "command.json", {"commands": resolved_commands, "cwd": str(setup_source), "native_source": str(setup_source)})
        if args.dry_run:
            for command in resolved_commands:
                print("DRY RUN:", " ".join(command))
            finish_run(run, "dry_run")
            return 0
        setup_environment = os.environ.copy()
        setup_environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        if spec.data["environment"]["manager"] == "conda":
            original_command_count = len(resolved_commands)
            resolved_commands = _skip_existing_conda_create(resolved_commands, env_name, setup_environment)
            write_json(run / "command.json", {"commands": resolved_commands, "cwd": str(setup_source),
                                               "native_source": str(setup_source),
                                               "resumed_existing_environment": len(resolved_commands) < original_command_count})
        try:
            _run_logged(resolved_commands, setup_source, setup_environment, run / "native.stdout.log")
        except (OSError, subprocess.CalledProcessError) as exc:
            finish_run(run, "failed", returncode=getattr(exc, "returncode", 1))
            raise
        else:
            finish_run(run, "complete", environment_lock=environment_lock(spec, setup_source))
        return 0
    if spec.name == "puzzlefusion_pp" and args.action == "test" and not args.verifier_checkpoint:
        raise ValueError("PuzzleFusion++ test requires --verifier-checkpoint in addition to --checkpoint (denoiser).")
    data_root = Path(args.data_root).resolve()
    manifest = data_root / "common_v1_manifest.json"
    if spec.name == "diffassemble":
        if not manifest.is_file() and not args.dry_run:
            raise FileNotFoundError(f"Missing retained-data manifest: {manifest}. Set --data-root to the materialized corpus.")
        selected_gpu = _visible_gpu(args.gpu, os.environ.get("CUDA_VISIBLE_DEVICES"))
    run = create_run(spec, source, Path(args.run_root).resolve(), args.action, args.seed, manifest, track)
    record_checkpoint_hashes(run, checkpoint=args.checkpoint, verifier_checkpoint=args.verifier_checkpoint, resume=args.resume)
    scratch_root = Path(args.scratch_root).resolve() if args.scratch_root else run.parent / "native_views"
    native_view = create_native_view(spec.name, data_root, scratch_root) if data_root.exists() else None
    native_source = _native_worktree(source, run)
    if spec.name == "diffassemble":
        from .models.diffassemble.apply_native import apply_native
        apply_native(native_source, run / "native-patches.json")
    _attach_model_data(spec.name, native_source, native_view)
    if native_view:
        variables["data"] = str(native_view)
    if args.native_data:
        variables["data"] = str(Path(args.native_data).resolve())
    variables["source"] = str(native_source)
    if not args.config and template:
        variables["config"] = str(native_source / template)
    variables["run_name"] = run.name
    variables["run"] = str(run)
    resolved_commands = [flatten_command(raw, variables) for raw in commands]
    resume_args = spec.data.get("resume_args", {}).get(args.action, []) if args.resume else []
    if resume_args:
        resolved_commands = [command + flatten_command(resume_args, variables) for command in resolved_commands]
    # Reuse the environment created by `setup` for uv models. Pointing uv at the
    # disposable source copy would resolve and store another large environment
    # inside every smoke/train/test run.
    uv_project = _native_build_source(source, model_dir) if spec.data["environment"]["manager"] == "uv" else None
    wrapped_commands = [(["uv", "run", "--project", str(uv_project), *command] if uv_project else ["conda", "run", "--no-capture-output", "-n", env_name, *command]) for command in resolved_commands]
    write_json(run / "command.json", {"commands": wrapped_commands, "cwd": str(native_source), "native_source": str(native_source), "native_view": str(native_view) if native_view else None})
    if args.dry_run:
        for command in wrapped_commands:
            print("DRY RUN:", " ".join(command))
        finish_run(run, "dry_run")
        if native_view and not args.keep_native_view:
            clean_native_view(spec.name, scratch_root)
        return 0
    try:
        native_environment = os.environ.copy()
        native_environment.update({
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "SOTA_DATA_ROOT": str(data_root),
            "SOTA_RUN_ROOT": str(run.parent),
            "SOTA_RUN_DIR": str(run),
            "SOTA_SEED": str(args.seed),
        })
        if spec.name == "diffassemble":
            native_environment.pop("PYTHONPATH", None)
            native_environment.pop("PYTHONHOME", None)
            native_environment.update(PYTHONNOUSERSITE="1", MPLBACKEND="Agg", WANDB_MODE="offline",
                                      XDG_CACHE_HOME=str(run / "cache"), HF_HOME=str(run / "cache/huggingface"))
            # --gpu is a logical index within any scheduler-assigned devices.
            native_environment["CUDA_VISIBLE_DEVICES"] = selected_gpu
            print(f"DiffAssemble CUDA_VISIBLE_DEVICES={selected_gpu!r}", flush=True)
        _run_logged(wrapped_commands, native_source, native_environment, run / "native.stdout.log")
    except (OSError, subprocess.CalledProcessError) as exc:
        finish_run(run, "failed", returncode=getattr(exc, "returncode", 1))
        raise
    else:
        finish_run(run, "complete")
    finally:
        if native_view and not args.keep_native_view:
            clean_native_view(spec.name, scratch_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
