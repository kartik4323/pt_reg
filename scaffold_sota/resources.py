"""Disk/device budgets without touching other experiments or jobs."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .io import study_root

GIB = 1024 ** 3


def directory_bytes(root):
    total = 0
    for current, directories, files in os.walk(str(root), followlinks=False):
        directories[:] = [name for name in directories if not Path(current, name).is_symlink()]
        for name in files:
            path = Path(current, name)
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                continue
    return total


def gpu_snapshot():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True, timeout=15)
        devices = []
        for line in result.stdout.splitlines():
            index, name, total, used = [x.strip() for x in line.split(",")]
            devices.append({"index": int(index), "name": name, "total_mib": int(total), "used_mib": int(used)})
        return devices
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


class ResourceGuard:
    def __init__(self, root, settings, device="cpu"):
        self.root = study_root(root)
        self.settings = settings
        self.device = str(device)
        self.peak_reserved = 0

    def check(self, additional_bytes=0):
        free = shutil.disk_usage(str(self.root)).free
        used = directory_bytes(self.root)
        if free - additional_bytes < self.settings["min_free_gib"] * GIB:
            raise RuntimeError("Disk reserve would fall below %.1f GiB" % self.settings["min_free_gib"])
        if used + additional_bytes > self.settings["cap_gib"] * GIB:
            raise RuntimeError("Independent study storage cap exceeded")
        result = {"study_bytes": used, "free_bytes": free}
        if self.device.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA device is unavailable; CPU does not certify A5000 feasibility")
            reserved = torch.cuda.max_memory_reserved(self.device)
            self.peak_reserved = max(self.peak_reserved, reserved)
            free_gpu, total_gpu = torch.cuda.mem_get_info(self.device)
            if reserved >= self.settings["max_reserved_gib"] * GIB:
                raise RuntimeError("PyTorch reserved-memory budget exceeded; reduce microbatch/chunking equally across conditions")
            if free_gpu < self.settings["min_gpu_free_gib"] * GIB:
                raise RuntimeError("Total device headroom is below reserve")
            result.update(peak_reserved_bytes=self.peak_reserved, gpu_free_bytes=free_gpu, gpu_total_bytes=total_gpu)
        return result

    def require_idle_device(self):
        if not self.device.startswith("cuda"):
            return
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        # CUDA-visible index may be remapped; physical process list is a safer
        # conservative gate than assuming nvidia-smi and torch indices coincide.
        try:
            result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, check=True, timeout=15)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("Cannot inspect GPU job occupancy") from exc
        others = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            pid, memory = [x.strip() for x in line.split(",")]
            if int(pid) != os.getpid():
                if not memory.isdigit() or int(memory) > self.settings["max_external_gpu_mib"]:
                    others.append(line)
        if others:
            raise RuntimeError("GPU is occupied by another job; retry after it finishes. " + "; ".join(others))
