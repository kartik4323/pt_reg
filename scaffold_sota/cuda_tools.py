"""Optional private CUDA 11.3 build tools; never installs a GPU driver."""
from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path

from .bootstrap import _conda_executable
from .io import RunLock, owned_output, read_json, sha256_file, study_root, write_json
from .resources import ResourceGuard
from .setup import _cuda_compiler, _log_tail, _stop_owned_child, subprocess_environment


def setup_cuda(root, cfg, execute=False):
    root = study_root(root)
    if execute and (sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64")):
        raise RuntimeError("The pinned CUDA 11.3 toolchain requires Linux x86-64")
    guard = ResourceGuard(root, cfg["resources"])
    with RunLock(root):
        guard.check()
        raw_prefix = root / "toolchains/cuda-11.3"
        if raw_prefix.is_symlink():
            raise ValueError("CUDA toolchain prefix must not be a symlink")
        prefix = owned_output(root, raw_prefix)
        work = owned_output(root, root / "environments/cuda-11.3")
        marker = work / "ownership.json"
        identity = {"experiment_id": "scaffold_sota", "toolchain": "cuda-11.3", "prefix": str(prefix)}
        if marker.exists():
            if read_json(marker) != identity:
                raise ValueError("CUDA toolchain ownership mismatch")
        elif prefix.exists():
            raise ValueError("Refusing to change an existing CUDA prefix without a study ownership marker")
        conda = _conda_executable(execute)
        command = [conda, "install" if (prefix / "conda-meta/history").is_file() else "create",
                   "--yes", "--prefix", str(prefix), "--override-channels",
                   "-c", "nvidia/label/cuda-11.3.1", "-c", "nvidia/label/cuda-11.3.0",
                   "-c", "conda-forge", "cuda-toolkit=11.3.1", "gcc_linux-64=10", "gxx_linux-64=10", "ninja=1.11"]
        plan = {**identity, "status": "planned", "command": command, "compiler_verified": False,
                "gpu_verified": False, "installs_driver": False,
                "note": "Optional compiler-only dependency for Jigsaw/CCS. Reuse an existing matching CUDA_HOME if available.",
                "source": "https://anaconda.org/nvidia/cuda-toolkit/labels"}
        write_json(work / "setup.json", plan)
        if not execute:
            return plan
        if not marker.exists():
            write_json(marker, identity)
        env = subprocess_environment(root)
        plan["status"] = "running"
        write_json(work / "setup.json", plan)

        def run(command, name):
            guard.check()
            log_path = work / name
            print("CUDA tools: %s (log: %s)" % (command[0], log_path), flush=True)
            with log_path.open("a", encoding="utf-8") as log:
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
                    log.flush()
                    raise RuntimeError("CUDA build-tools step failed; inspect %s\n%s" % (log_path, _log_tail(log_path)))
            guard.check()

        try:
            run(command, "install.log")
            env["CUDA_HOME"] = str(prefix)
            compiler = _cuda_compiler(env, prefix, "11.3")
            if Path(compiler["path"]).resolve() != prefix or not compiler.get("host_cxx"):
                raise RuntimeError("Installed private CUDA/GCC toolchain is incomplete")
            probe = work / "compile_probe.cu"
            probe.write_text("#include <cuda_runtime.h>\n__global__ void kernel() {}\nint main() { return 0; }\n", encoding="utf-8")
            output = work / ("compile_probe_%d.bin" % time.time_ns())
            build = [str(prefix / "bin/nvcc"), "-ccbin", env["CXX"], "-std=c++14", "--cudart", "shared",
                     str(probe), "-o", str(output)]
            plan["compile_probe"] = build
            run(build, "compile.log")  # Compile/link only; no GPU workload or driver modification.
            if not output.is_file() or not output.stat().st_size:
                raise RuntimeError("CUDA compile probe produced no executable")
            plan.update(status="installed_compiler_verified", compiler_verified=True,
                        compiler=compiler, compile_probe_sha256=sha256_file(output), resources=guard.check())
        except BaseException as exc:
            plan.update(status="failed", error=str(exc))
            raise
        finally:
            write_json(work / "setup.json", plan)
        return plan
