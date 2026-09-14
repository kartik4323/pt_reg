"""Atomic study artifacts and narrowly scoped run ownership."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

from . import EXPERIMENT_ID

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent


def contained(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def digest_json(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".%s.tmp" % os.getpid())
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(jsonable(value), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def study_root(path, create=False):
    root = Path(path).expanduser().resolve()
    if contained(root, REPO) or root == Path(root.anchor):
        raise ValueError("Run root must be an independent directory outside the repository and filesystem root")
    if os.name != "nt" and contained(root, "/data"):
        raise ValueError("This study must not allocate on /data")
    marker = root / ".scaffold_sota.json"
    if marker.exists():
        if read_json(marker).get("experiment_id") != EXPERIMENT_ID:
            raise ValueError("Run root belongs to a different experiment")
    elif create:
        if root.exists() and any(root.iterdir()):
            raise ValueError("Refusing to claim a nonempty directory without the study marker")
        root.mkdir(parents=True, exist_ok=True)
        write_json(marker, {"experiment_id": EXPERIMENT_ID, "schema_version": 1,
                            "created_at": time.time()})
    else:
        raise ValueError("Initialize the independent run root with scaffold_sota init")
    return root


def owned_output(root, path):
    root = study_root(root)
    result = Path(path).expanduser().resolve()
    if not contained(result, root) or result == root:
        raise ValueError("All output must be inside the marked scaffold_sota run root")
    return result


class RunLock:
    def __init__(self, root):
        self.path = study_root(root) / ".active_job.lock"
        self.owned = False

    def __enter__(self):
        try:
            descriptor = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError("Another study job or stale lock exists: %s. Inspect it; use unlock only after the job exits." % self.path)
        with os.fdopen(descriptor, "w") as handle:
            json.dump({"pid": os.getpid(), "host": platform.node(), "created_at": time.time()}, handle)
        self.owned = True
        return self

    def __exit__(self, *args):
        if self.owned:
            self.path.unlink()


def _process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError("Invalid owning process ID")
    if os.name == "nt":
        # Windows os.kill(pid, 0) can terminate a process. Query its state using
        # a read-only handle instead; access denial never proves it has exited.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:  # ERROR_INVALID_PARAMETER: PID absent.
                return False
            raise RuntimeError("Cannot establish that the owning process has exited")
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise RuntimeError("Cannot query owning process state")
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        raise RuntimeError("Cannot establish that the owning process has exited")


def unlock(root):
    path = study_root(root) / ".active_job.lock"
    if not path.exists():
        return {"status": "not_locked"}
    record = read_json(path)
    if record["host"] != platform.node():
        raise ValueError("Lock belongs to another host; inspect that host first")
    if not _process_alive(int(record["pid"])):
        path.unlink()
        return {"status": "removed_stale_lock"}
    raise RuntimeError("Owning process is still alive; it will not be interrupted")
