"""Bootstrap common CPU tools without importing the dependencies being installed.

This environment handles manifests, prior export and orchestration. Native model
training continues to use each model's separate environment.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .io import RunLock, digest_json, owned_output, read_json, study_root, write_json
from .resources import ResourceGuard
from .setup import _stop_owned_child, subprocess_environment


TOOLS_REQUIREMENTS = (
    "numpy==1.26.4", "scipy==1.13.1", "trimesh==4.6.8",
    "PyYAML==6.0.2", "rtree==1.4.1", "uv==0.9.26",
)
TORCH_REQUIREMENT = "torch==2.6.0"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# Run inside the new environment only. The current interpreter may have no pip,
# Torch, NumPy or YAML. Checking actual geometry catches missing libspatialindex.
SMOKE_SCRIPT = """import json, pathlib, sys, subprocess
import numpy, scipy, trimesh, yaml, rtree, torch
import scaffold_sota.data, scaffold_sota.priors, scaffold_sota.evaluation
from scaffold_sota.priors import architectures
assert sys.version_info[:2] == (3, 10), 'Tools environment must use Python 3.10'
assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(sys.argv[2]).resolve(), 'Wrong environment prefix'
assert torch.version.cuda is None, 'Tools must use the CPU-only Torch wheel'
assert not torch.cuda.is_available(), 'Tools must not initialize a GPU'
assert torch.from_numpy(numpy.zeros((2, 3), dtype=numpy.float32)).numpy().shape == (2, 3)
mesh = trimesh.creation.box()
assert numpy.isfinite(trimesh.proximity.signed_distance(mesh, numpy.array([[0., 0., 0.]]))).all()
result = {'status': 'passed', 'cpu_tools_verified': True, 'native_models_verified': False,
          'python': sys.version, 'prefix': sys.prefix,
          'versions': {module.__name__: module.__version__ for module in (numpy, scipy, trimesh, yaml, rtree, torch)}}
result['uv'] = subprocess.check_output([sys.executable, '-m', 'uv', '--version'], text=True).strip()
pathlib.Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\\n', encoding='utf-8')
print(json.dumps(result))
"""


def tools_python(root):
    """Return the explicit interpreter without activating or modifying any env."""
    return study_root(root) / "envs" / "tools" / ("python.exe" if os.name == "nt" else "bin/python")


def _conda_executable(execute):
    candidate = os.environ.get("CONDA_EXE")
    if candidate and Path(candidate).is_file():
        return str(Path(candidate).resolve())
    candidate = shutil.which("conda")
    if candidate:
        return candidate
    if execute:
        raise RuntimeError("Conda is required to create the isolated tools environment. Run from a shell where conda is available.")
    return "conda"


def _tools_environment(root):
    env = subprocess_environment(root)
    # Site/user pip configuration must not redirect an install into the active
    # environment or pull a CUDA Torch wheel through an inherited extra index.
    for key in tuple(env):
        if key.startswith("PIP_") and key not in ("PIP_CACHE_DIR",):
            env.pop(key)
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1",
               CONDA_ENVS_PATH=str(root / "envs"), CUDA_VISIBLE_DEVICES="",
               XDG_CACHE_HOME=str(root / "package_cache"),
               CONDA_AUTO_UPDATE_CONDA="false")
    return env


def _validate_owner(root, prefix, marker):
    expected = {"experiment_id": "scaffold_sota", "environment": "tools",
                "run_root": str(root), "environment_prefix": str(prefix)}
    # Do not permit an environment symlink, even to another owned directory.
    raw_prefix = root / "envs" / "tools"
    if raw_prefix.is_symlink():
        raise ValueError("Refusing a symlink for the tools environment prefix")
    if marker.exists():
        record = read_json(marker)
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("Tools environment ownership marker does not match this study")
    elif prefix.exists():
        raise ValueError("Refusing to modify an existing tools prefix without its study ownership marker: " + str(prefix))
    return expected


def setup_tools(root, cfg, execute=False):
    """Plan or install a study-owned common-tools environment on the Linux VM.

    Plans require only Python's standard library and do not claim an environment.
    A failed install leaves its owned prefix and per-step logs for a safe retry.
    """
    root = study_root(root)
    if execute and sys.platform != "linux":
        raise RuntimeError("Tools environment installation is supported on the Linux VM only")
    guard = ResourceGuard(root, cfg["resources"])
    with RunLock(root):
        guard.check()
        prefix = owned_output(root, root / "envs" / "tools")
        work = owned_output(root, root / "environments" / "tools")
        marker = work / "ownership.json"
        owner = _validate_owner(root, prefix, marker)
        python = tools_python(root)
        conda = _conda_executable(execute)
        env = _tools_environment(root)
        history = prefix / "conda-meta" / "history"
        commands = [[conda, "install" if history.is_file() else "create", "--yes",
                     "--prefix", str(prefix), "python=3.10", "pip=25.0.1"],
                    [str(python), "-m", "pip", "install", "--only-binary=:all:",
                     TORCH_REQUIREMENT, "--index-url", CPU_INDEX],
                    [str(python), "-m", "pip", "install", "--only-binary=:all:",
                     "--index-url", "https://pypi.org/simple"] + list(TOOLS_REQUIREMENTS),
                    [str(python), "-m", "pip", "check"],
                    [str(python), "-m", "pip", "freeze"],
                    [str(python), "-c", SMOKE_SCRIPT, str(work / "smoke.json"), str(prefix)]]
        plan = {"experiment_id": "scaffold_sota", "environment": "tools",
                "environment_prefix": str(prefix), "python": str(python),
                "commands": commands, "status": "planned", "cpu_tools_verified": False,
                "native_models_verified": False, "recipe_sha256": digest_json(commands),
                "note": "CPU tools support manifest checks, prior export and experiment orchestration; train each native model in its own environment.",
                "torch_install_reference": "https://pytorch.org/get-started/previous-versions/"}
        write_json(work / "setup.json", plan)
        if not execute:
            return plan
        # Persist ownership before the first child can create or change a prefix.
        if not marker.exists():
            write_json(marker, owner)
        write_json(work / "smoke.json", {"status": "pending", "cpu_tools_verified": False})
        plan["status"] = "running"
        write_json(work / "setup.json", plan)
        try:
            for index, command in enumerate(commands):
                guard.check()
                plan["active_step"] = index
                write_json(work / "setup.json", plan)
                log_path = work / ("setup-%02d.log" % index)
                print("tools setup %d/%d: %s" % (index + 1, len(commands), command[0]), flush=True)
                with log_path.open("a", encoding="utf-8") as log:
                    log.write("\nStarting tools setup step %d\n" % index)
                    log.flush()
                    child = subprocess.Popen(command, cwd=str(work), env=env, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        while child.poll() is None:
                            time.sleep(2)
                            guard.check()
                    except BaseException:
                        _stop_owned_child(child)
                        raise
                    if child.returncode:
                        raise RuntimeError("Tools dependency step failed (%s); inspect %s" % (child.returncode, log_path))
                    guard.check()
            smoke = read_json(work / "smoke.json")
            if smoke.get("status") != "passed" or smoke.get("cpu_tools_verified") is not True:
                raise RuntimeError("Tools import smoke check did not pass")
            plan.update(status="installed_cpu_tools_verified", cpu_tools_verified=True, smoke=smoke)
            plan.pop("active_step", None)
        except BaseException as exc:
            plan.update(status="failed", error=str(exc))
            raise
        finally:
            write_json(work / "setup.json", plan)
        return plan
