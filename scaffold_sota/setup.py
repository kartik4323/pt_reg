"""Install native dependencies only into study-owned environments and sources."""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .io import REPO, RunLock, owned_output, study_root, write_json, sha256_file
from .registry import materialize_source, model_spec
from .resources import ResourceGuard, directory_bytes


def environment_python(root, model):
    prefix = study_root(root) / "envs" / model
    return prefix / (("Scripts/python.exe" if model == "garf" else "python.exe") if os.name == "nt" else "bin/python")


def subprocess_environment(root):
    root = study_root(root)
    env = dict(os.environ)
    for key, relative in {"CONDA_PKGS_DIRS": "package_cache/conda", "PIP_CACHE_DIR": "package_cache/pip",
                          "UV_CACHE_DIR": "package_cache/uv", "UV_PYTHON_INSTALL_DIR": "python",
                          "TORCH_EXTENSIONS_DIR": "extensions", "HF_HOME": "package_cache/huggingface",
                          "WANDB_DIR": "logs/wandb", "MPLCONFIGDIR": "package_cache/matplotlib",
                          "TMPDIR": "scratch/tmp", "TEMP": "scratch/tmp", "TMP": "scratch/tmp"}.items():
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONUSERBASE", None)
    env.update(WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               PYTHONPATH=str(REPO), CONDA_ALWAYS_YES="true", CONDA_CHANNEL_PRIORITY="flexible")
    return env


def _cuda_compiler(env, prefix, release):
    """Choose the native compiler matching the pinned Torch CUDA ABI."""
    candidates = [Path(env[key]) for key in ("CUDA_HOME", "CUDA_PATH") if env.get(key)]
    found = shutil.which("nvcc", path=env.get("PATH"))
    if found:
        candidates.append(Path(found).resolve().parent.parent)
    candidates += [Path(prefix), Path("/usr/local/cuda-" + release), Path("/usr/local/cuda")]
    for candidate in candidates:
        compiler = candidate / "bin/nvcc"
        if compiler.is_file():
            version = subprocess.check_output([str(compiler), "--version"], env=env, text=True)
            if re.search(r"release\s+" + re.escape(release) + r"(?:,|\s)", version):
                env.update(CUDA_HOME=str(candidate), CUDA_PATH=str(candidate),
                           PATH=str(candidate / "bin") + os.pathsep + env.get("PATH", ""),
                           LD_LIBRARY_PATH=str(candidate / "lib64") + os.pathsep + env.get("LD_LIBRARY_PATH", ""))
                return {"path": str(candidate), "version": version.strip()}
    raise RuntimeError("Native extensions require CUDA %s nvcc (the runtime alone has no compiler). Set CUDA_HOME to an installed matching toolkit and rerun setup." % release)


def _stop_owned_child(child):
    # Each Linux setup child starts its own session, including pip/compiler children.
    if child.poll() is not None:
        return
    try:
        if sys.platform == "linux":
            os.killpg(child.pid, signal.SIGTERM)
        else:
            child.terminate()
    except ProcessLookupError:
        child.wait()
        return
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            if sys.platform == "linux":
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
        except ProcessLookupError:
            pass
        child.wait()


def setup_model(root, model, cfg, execute=False, source=None):
    import yaml
    root = study_root(root)
    if execute and sys.platform != "linux":
        raise RuntimeError("Native CUDA dependency installation is supported on the Linux VM only")
    if execute and model == "gpat":
        raise RuntimeError("GPAT's pinned Python 3.6 recipe is incompatible with this Python >=3.8 runner. Its real semantic backend needs a separately verified compatibility environment; automatic installation is not yet supported.")
    guard = ResourceGuard(root, cfg["resources"])
    with RunLock(root):
        guard.check()
        work = root / "environments" / model
        original = Path(source).expanduser().resolve() if source else REPO / "sota_repro/models" / model / "upstream"
        guard.check(additional_bytes=directory_bytes(original))
        source, info = materialize_source(model, root, work, original)
        spec = model_spec(model)
        prefix = owned_output(root, root / "envs" / model)
        python = environment_python(root, model)
        env = subprocess_environment(root)
        commands, cuda_steps, recipe_files = [], set(), []
        pip = [str(python), "-m", "pip"]
        if spec["environment"]["manager"] == "uv":
            env["UV_PROJECT_ENVIRONMENT"] = str(prefix)
            commands = [["uv", "sync", "--project", str(source), "--frozen", "--no-dev"],
                        ["uv", "sync", "--project", str(source), "--frozen", "--no-dev", "--extra", "post"]]
            # uv environments do not contain pip by default. The locked post
            # extras are prebuilt wheels; no compiler is needed for this profile.
            pip = ["uv", "pip", "--python", str(python)]
            recipe_files += [source / "pyproject.toml", source / "uv.lock"]
        elif model in ("jigsaw", "ccs", "diffassemble", "gpat"):
            env_file = spec["environment"].get("file")
            original_file = (REPO / "sota_repro" / env_file) if str(env_file).startswith("models/") else source / env_file
            document = yaml.safe_load(original_file.read_text(encoding="utf-8"))
            recipe_files.append(original_file)
            document.pop("name", None)
            document.pop("prefix", None)
            generated = work / "environment.yaml"
            generated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            commands.append(["conda", "env", "update" if prefix.exists() else "create", "--prefix", str(prefix), "--file", str(generated)])
            requirements = REPO / "sota_repro/models" / model / "requirements.repro.txt"
            if requirements.exists():
                command = pip + ["install", "-r", str(requirements)]
                constraints = requirements.with_name("constraints.repro.txt")
                if constraints.exists():
                    command += ["-c", str(constraints)]
                    recipe_files.append(constraints)
                resolved = requirements.with_name("constraints.resolved.txt")
                if resolved.exists():
                    recipe_files.append(resolved)
                commands.append(command)
                recipe_files.append(requirements)
        else:
            version = spec["environment"]["python"]
            if not prefix.exists():
                commands.append(["conda", "create", "--yes", "--prefix", str(prefix), "python=" + str(version)])
            if model == "puzzlefusion_pp":
                # Upstream requirements omits Torch and all three CUDA packages.
                constraints = work / "constraints.txt"
                constraints.write_text("torch==2.0.1\ntorchvision==0.15.2\ntorchaudio==2.0.2\nnumpy==1.23.5\nhuggingface-hub==0.20.3\ntorchmetrics==1.3.2\n", encoding="utf-8")
                commands += [pip + ["install", "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2", "--index-url", "https://download.pytorch.org/whl/cu118"],
                             pip + ["install", "-c", str(constraints), "-r", str(source / "requirements.txt"), "huggingface-hub==0.20.3", "torchmetrics==1.3.2", "fvcore", "iopath", "ninja"],
                             pip + ["install", "--no-deps", "torch-cluster==1.6.3+pt20cu118", "--only-binary=:all:", "--find-links", "https://data.pyg.org/whl/torch-2.0.1+cu118.html"]]
                cuda_steps.update((len(commands), len(commands) + 1))
                commands += [pip + ["install", "--no-build-isolation", "--no-deps", "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4"],
                             pip + ["install", "--no-build-isolation", "--no-deps", "chamferdist==1.0.3"]]
                recipe_files += [source / "requirements.txt", constraints]
            else:
                commands += [[str(python), "-m", "pip", "install", "torch==1.10.1+cu111", "torchvision==0.11.2+cu111",
                              "--extra-index-url", "https://download.pytorch.org/whl/cu111"],
                             [str(python), "-m", "pip", "install", "numpy<2", "pytorch-lightning==1.9.5", "einops", "trimesh", "open3d", "gtsam", "wandb"]]
        if model != "garf":
            commands.append(pip + ["install", "PyYAML", "scipy", "trimesh"])
        # Source-native editable extensions live only in this owned source copy.
        if model == "pmtr":
            cuda_steps.add(len(commands))
            commands.append(pip + ["install", "--no-build-isolation", "--no-deps", str(source)])
        if model in ("pmtr", "cmnet"):
            cuda_steps.update(range(len(commands), len(commands) + 3))
            commands += [[str(python), "-m", "pip", "install", "git+https://github.com/KinglittleQ/torch-batch-svd"],
                         [str(python), "-m", "pip", "install", "git+https://github.com/otaheri/chamfer_distance"],
                         [str(python), "-m", "pip", "install", "git+https://github.com/facebookresearch/pytorch3d.git@stable"]]
        if model == "cmnet":
            cuda_steps.add(len(commands))
            commands.append(pip + ["install", "--no-build-isolation", "--no-deps", str(source / "lib/pointops")])
        if model == "ccs":
            commands.append(pip + ["install", "--no-build-isolation", "--no-deps", "-e", str(source)])
            for relative in ("multi_part_assembly/utils/chamfer", "multi_part_assembly/models/modules/encoder/pointnet2/pointnet2_ops_lib"):
                cuda_steps.add(len(commands))
                commands.append(pip + ["install", "--no-build-isolation", "--no-deps", "-e", str(source / relative)])
        if model == "jigsaw":
            cuda_steps.add(len(commands))
            commands.append(pip + ["install", "--no-build-isolation", "--no-deps", "-e", str(source / "utils/chamfer")])
        if model == "diffassemble":
            repair = REPO / "sota_repro/models/diffassemble/repair_runtime.py"
            commands.append([str(python), str(repair), "--report", str(work / "runtime-repair.json")])
            recipe_files.append(repair)
        if model == "gpat":
            # Upstream has no root setup.py; native extension roots are separate.
            for relative in ("utils/chamfer", "utils/pointops"):
                commands.append(pip + ["install", "--no-build-isolation", "--no-deps", "-e", str(source / relative)])
        if model == "garf":
            # uv places --python after the subcommand, unlike python -m pip.
            commands += [["uv", "pip", "check", "--python", str(python)],
                         ["uv", "pip", "freeze", "--python", str(python)]]
        else:
            commands += [pip + ["check"], pip + ["freeze"]]
        plan = {"experiment_id": "scaffold_sota", "model": model, "source": info,
                "environment_prefix": str(prefix), "python": str(python), "commands": commands,
                "status": "unsupported_legacy_runtime" if model == "gpat" else "planned", "runtime_verified": False,
                "recipe_hashes": {str(p): sha256_file(p) for p in recipe_files},
                "cuda_build_steps": sorted(cuda_steps),
                "note": "Compatibility recipes require real GPU preflight; GPAT stock Python 3.6 cannot execute this runner." if model == "gpat" else "Installation is not GPU verification. Last step records the resolved package versions in its log."}
        write_json(work / "setup.json", plan)
        if not execute:
            return plan
        plan["status"] = "running"
        write_json(work / "setup.json", plan)
        try:
            for index, command in enumerate(commands):
                guard.check()
                if index in cuda_steps and not plan.get("cuda_compiler"):
                    release = "11.3" if model == "jigsaw" else spec["environment"]["cuda"]
                    plan["cuda_compiler"] = _cuda_compiler(env, prefix, release)
                    # This pilot targets the user's RTX A5000 (Ampere).
                    env["TORCH_CUDA_ARCH_LIST"] = "8.6"
                print("%s setup %d/%d: %s" % (model, index + 1, len(commands), command[0]), flush=True)
                with (work / ("setup-%02d.log" % index)).open("a", encoding="utf-8") as log:
                    child = subprocess.Popen(command, cwd=str(source), env=env, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=(sys.platform == "linux"))
                    try:
                        while child.poll() is None:
                            time.sleep(2)
                            guard.check()
                    except BaseException:
                        _stop_owned_child(child)
                        raise
                    if child.returncode:
                        raise RuntimeError("Dependency step failed (%s); inspect %s" % (child.returncode, log.name))
            plan["status"] = "installed_unverified"
            plan["note"] = "Run the native preflight on actual bottle data before training."
        except BaseException as exc:
            plan.update(status="failed", error=str(exc))
            raise
        finally:
            write_json(work / "setup.json", plan)
        return plan
