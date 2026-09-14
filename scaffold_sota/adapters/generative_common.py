"""Boundaries shared by the real generative backends (no substitute networks)."""
from __future__ import annotations

import contextlib
import importlib
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn


class NativeDependencyError(RuntimeError):
    pass


def option(options, name, default=None):
    return options.get(name, default) if isinstance(options, dict) else getattr(options, name, default)


def import_native(source_root, package, relative_file):
    root = Path(source_root).resolve()
    sys.dont_write_bytecode = True
    if not (root / relative_file).is_file():
        raise NativeDependencyError(f"Native source missing: {root / relative_file}")
    top = package.split('.')[0]
    loaded = sys.modules.get(top)
    if loaded is not None:
        locations = list(getattr(loaded, '__path__', []))
        if getattr(loaded, '__file__', None):
            locations.append(str(Path(loaded.__file__).parent))
        def inside(location):
            try:
                Path(location).resolve().relative_to(root)
                return True
            except ValueError:
                return False
        if locations and not any(inside(p) for p in locations):
            raise NativeDependencyError(f"{top} is already imported from another model; use a separate process")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module(package)
    except (ImportError, OSError) as exc:
        raise NativeDependencyError(
            f"Cannot import real native backend {package} from {root}: {exc}. "
            "Install and validate its pinned CUDA environment; no fallback architecture is used."
        ) from exc


def observation(sample, device):
    """Read only deployable observation keys; never inspect supervision here."""
    points = torch.as_tensor(sample['points'], dtype=torch.float32, device=device)
    mask = torch.as_tensor(sample['fragment_mask'], dtype=torch.bool, device=device)
    anchor = int(torch.as_tensor(sample['anchor_index']).item())
    if points.ndim != 3 or points.shape[-1] != 3 or mask.shape != points.shape[:1]:
        raise ValueError('Expected unbatched points[F,N,3] and fragment_mask[F]')
    if points.shape[1] < 3 or int(mask.sum()) < 2 or not 0 <= anchor < len(mask) or not mask[anchor]:
        raise ValueError('Need at least two nondegenerate fragments and an active input anchor')
    active = torch.where(mask)[0]
    if not torch.isfinite(points[active]).all():
        raise ValueError('Nonfinite input points')
    radii = points[active].abs().amax(dim=(1, 2))
    if (radii <= 1e-8).any():
        raise ValueError('Degenerate fragment scale')
    return points, mask, active, anchor


def labels(sample, points, active, anchor, transforms):
    rotations = torch.as_tensor(sample['rotations_gt'], dtype=points.dtype, device=points.device)
    translations = torch.as_tensor(sample['translations_gt'], dtype=points.dtype, device=points.device)
    if rotations.shape != (len(points), 3, 3) or translations.shape != (len(points), 3):
        raise ValueError('Pose supervision has the wrong shape')
    if not torch.isfinite(rotations[active]).all() or not torch.isfinite(translations[active]).all():
        raise ValueError('Nonfinite pose supervision')
    if not torch.allclose(rotations[anchor], torch.eye(3, device=points.device), atol=1e-5) or not torch.allclose(translations[anchor], torch.zeros(3, device=points.device), atol=1e-5):
        raise ValueError('Training labels must use the observed input anchor at identity')
    return transforms.matrix_to_quaternion(rotations[active]), translations[active]


def packed_parts(sample, device, transforms=None, supervised=False):
    points, mask, active, anchor = observation(sample, device)
    part_points = points[active]
    scales = part_points.abs().amax(dim=(1, 2)).unsqueeze(-1)
    ref = active == anchor
    batch = {
        'part_pcs': (part_points / scales[:, None, :]).unsqueeze(0),
        'part_scale': scales.unsqueeze(0),
        'part_valids': torch.ones((1, len(active)), dtype=torch.bool, device=points.device),
        'ref_part': ref.unsqueeze(0),
    }
    if supervised:
        q, t = labels(sample, points, active, anchor, transforms)
        batch.update(part_rots=q.unsqueeze(0), part_trans=t.unsqueeze(0))
    return batch, (points, mask, active, anchor)


@torch.no_grad()
def estimate_normals(points, neighbors=24, chunk_size=128):
    """Local PCA normals from observed XYZ only; radial orientation is approximate.

    This deterministic input adaptation is applied equally to every GARF arm.
    It does not use fracture labels, assembled geometry, or target mesh normals.
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError('Expected points[F,N,3]')
    k = min(max(3, int(neighbors)), points.shape[1])
    result = []
    for part in points.float():
        blocks = []
        for start in range(0, len(part), chunk_size):
            queries = part[start:start + chunk_size]
            indices = torch.cdist(queries, part).topk(k, largest=False).indices
            local = part[indices]
            centered = local - local.mean(dim=1, keepdim=True)
            covariance = centered.transpose(1, 2) @ centered / k
            normal = torch.linalg.eigh(covariance).eigenvectors[..., 0]
            radial = queries - part.mean(dim=0)
            sign = (normal * radial).sum(dim=-1)
            # Deterministic sign for zero radial projection, e.g. planar clouds.
            dominant = normal.gather(1, normal.abs().argmax(dim=1, keepdim=True)).squeeze(1)
            sign = torch.where(sign.abs() > 1e-10, sign, dominant)
            normal = normal * torch.where(sign < 0, -1.0, 1.0).unsqueeze(-1)
            blocks.append(normal)
        result.append(torch.cat(blocks))
    return torch.stack(result)


def pose_result(context, quaternions, translations, transforms, *, reanchor=True):
    points, mask, active, anchor = context
    q = quaternions.float()
    t = translations.float()
    if q.shape != (len(active), 4) or t.shape != (len(active), 3):
        return {'status': 'failed', 'reason': 'invalid_native_pose_shape'}
    if not torch.isfinite(q).all() or not torch.isfinite(t).all() or (q.norm(dim=-1) < 1e-8).any():
        return {'status': 'failed', 'reason': 'nonfinite_or_degenerate_native_pose'}
    rotations = transforms.quaternion_to_matrix(q / q.norm(dim=-1, keepdim=True))
    if reanchor:
        index = int(torch.where(active == anchor)[0][0])
        anchor_r = rotations[index].clone()
        anchor_t = t[index].clone()
        rotations = anchor_r.T.unsqueeze(0) @ rotations
        t = (t - anchor_t) @ anchor_r
    out_r = torch.eye(3, device=points.device).repeat(len(points), 1, 1)
    out_t = torch.zeros((len(points), 3), device=points.device)
    out_r[active], out_t[active] = rotations, t
    return {'status': 'ok', 'rotations': out_r, 'translations': out_t}


@contextlib.contextmanager
def prediction_rng(seed, device):
    np_state, py_state = np.random.get_state(), random.getstate()
    dev = torch.device(device)
    cuda_devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) % (2**32))
            random.seed(int(seed))
            yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def load_stage_weights(checkpoint, expected_model, expected_stage, prefix):
    path = Path(checkpoint).resolve()
    if not path.is_file():
        raise ValueError(f'Required trained feature checkpoint does not exist: {path}')
    # These are local runner checkpoints including optimizer/RNG metadata.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('model_name') != expected_model or payload.get('stage') != expected_stage:
        raise ValueError(f'Expected {expected_model}/{expected_stage} checkpoint; got {payload.get("model_name")}/{payload.get("stage")}')
    state = payload.get('model')
    if not isinstance(state, dict):
        raise ValueError('Feature checkpoint lacks model state dictionary')
    selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise ValueError(f'Feature checkpoint lacks required prefix {prefix}')
    return selected


class GenerativeBackend(nn.Module):
    conditioning_parameter_prefixes = ('conditioner.',)

    def configure_conditioning(self, options, hidden_dim, layers=None, output_module=None):
        self.condition = option(options, 'condition', 'native')
        if self.condition not in {'native', 'B0', 'B1', 'B2', 'B3', 'B4'}:
            raise ValueError(f'Unsupported condition: {self.condition}')
        self._active_prior = None
        self._condition_hooks = []
        if self.condition in {'native', 'B0'}:
            return
        from scaffold_sota.models.conditioning import ConditioningBlock
        self.conditioner = ConditioningBlock(hidden_dim, self.condition,
            token_dim=int(option(options, 'token_dim', 128)), heads=int(option(options, 'heads', 4)),
            num_tokens=int(option(options, 'num_tokens', 512)))
        # One shared zero-gated residual at the entry to native token mixing.
        # Hooks preserve all native state_dict names for baseline branching.
        if layers is not None:
            self._condition_hooks.append(layers[0].register_forward_pre_hook(self._condition_input, with_kwargs=True))
            # GARF's non-Flash path invokes layer.forward_sdpa directly, which
            # bypasses Module.__call__ and PyTorch forward hooks entirely.
            if hasattr(layers[0], 'forward_sdpa'):
                original_sdpa = layers[0].forward_sdpa
                def conditioned_sdpa(layer, *args, **kwargs):
                    args, kwargs = self._condition_input(layer, args, kwargs)
                    return original_sdpa(*args, **kwargs)
                layers[0].forward_sdpa = types.MethodType(conditioned_sdpa, layers[0])
        elif output_module is not None:
            self._condition_hooks.append(output_module.register_forward_hook(self._condition_output))

    def _condition_input(self, module, args, kwargs):
        if 'hidden_states' in kwargs:
            kwargs = dict(kwargs)
            kwargs['hidden_states'] = self.conditioner(kwargs['hidden_states'], self._active_prior)
        else:
            args = (self.conditioner(args[0], self._active_prior), *args[1:])
        return args, kwargs

    def _condition_output(self, module, args, output):
        return self.conditioner(output, self._active_prior)

    @contextlib.contextmanager
    def prior_context(self, prior):
        if self.condition not in {'native', 'B0', 'B1'} and prior is None:
            raise ValueError(f'{self.condition} requires a predicted scaffold record')
        if self._active_prior is not None:
            raise RuntimeError('Backend calls cannot be nested/concurrent')
        self._active_prior = prior
        try:
            yield
        finally:
            self._active_prior = None

    def initialize_from_native(self, state_dict):
        incompatible = self.load_state_dict(state_dict, strict=False)
        bad_missing = [key for key in incompatible.missing_keys if not key.startswith(self.conditioning_parameter_prefixes)]
        if bad_missing or incompatible.unexpected_keys:
            raise ValueError(f'Native initialization mismatch: missing={bad_missing}, unexpected={incompatible.unexpected_keys}')
        return incompatible
