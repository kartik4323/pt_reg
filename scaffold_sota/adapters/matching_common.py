"""Input-only contracts and native-import guards shared by Jigsaw and CCS."""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
import torch


def native_import_root(source_root, package_names):
    root = Path(source_root).resolve(strict=True)
    # Native source trees are read-only dependencies, including Python caches.
    sys.dont_write_bytecode = True
    for name in package_names:
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None)
        inside = False
        locations = [origin] if origin else list(getattr(module, "__path__", ()))
        if locations:
            try:
                for location in locations:
                    Path(location).resolve().relative_to(root)
                inside = True
            except ValueError:
                pass
        if module is not None and not inside:
            raise RuntimeError(f"Native package {name!r} already belongs to another source tree; use a fresh process")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_config_file(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load native configuration {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_cfg_defaults()


def input_parts(sample, device):
    """Pack valid fragments and return original indices and the packed anchor."""
    points = torch.as_tensor(sample["points"], device=device, dtype=torch.float32)
    valid = torch.as_tensor(sample["fragment_mask"], device=device, dtype=torch.bool)
    if points.ndim != 3 or points.shape[-1] != 3 or valid.shape != points.shape[:1]:
        raise ValueError("Expected unbatched points[F,N,3] and fragment_mask[F]")
    indices = valid.nonzero(as_tuple=False).flatten()
    if len(indices) not in (2, 3) or points.shape[1] < 3:
        raise ValueError("The bottle adapter requires two or three complete fragments")
    anchor_original = int(torch.as_tensor(sample["anchor_index"]).item())
    anchor_match = (indices == anchor_original).nonzero(as_tuple=False).flatten()
    if anchor_match.numel() != 1 or not torch.isfinite(points[indices]).all():
        raise ValueError("Invalid anchor or nonfinite fragment points")
    return points[indices], indices, int(anchor_match.item())


def anchor_relative(rotations, translations, anchor):
    """Convert local-to-global poses into the input anchor's coordinate frame."""
    r = torch.as_tensor(rotations)
    t = torch.as_tensor(translations, device=r.device, dtype=r.dtype)
    if r.ndim != 3 or r.shape[1:] != (3, 3) or t.shape != (r.shape[0], 3):
        raise ValueError("Invalid pose array shapes")
    inv = r[anchor].transpose(-1, -2)
    return inv.unsqueeze(0) @ r, (t - t[anchor]) @ inv.transpose(-1, -2)


def pose_result(rotations, translations, anchor, **diagnostics):
    r, t = anchor_relative(rotations, translations, anchor)
    if not torch.isfinite(r).all() or not torch.isfinite(t).all():
        return failed("nonfinite_native_pose", **diagnostics)
    identity = torch.eye(3, dtype=r.dtype, device=r.device)
    if not torch.allclose(r @ r.transpose(-1, -2), identity.expand_as(r), atol=1e-3, rtol=1e-3) or not torch.all(torch.linalg.det(r) > 0.99):
        return failed("invalid_native_rotation", **diagnostics)
    return {"status": "ok", "rotations": r.detach().cpu().numpy(),
            "translations": t.detach().cpu().numpy(), "diagnostics": diagnostics}


def failed(reason, **diagnostics):
    return {"status": "no_solution", "reason": reason,
            "rotations": None, "translations": None, "diagnostics": diagnostics}


@contextmanager
def inference_seed(seed, device):
    """Reproducible inference without altering the caller's training RNG stream."""
    py_state, np_state = random.getstate(), np.random.get_state()
    dev = torch.device(device)
    devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
