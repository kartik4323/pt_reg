"""Continuous reference-frame fields for rigid assembly and labeled controls.

Queries, signed distances and gradients use the normalized reference frame.
Distances are negative inside. ``sample`` differentiates the distance it returns;
uncertainty is detached and maps to the existing solver's confidence convention.
No source geometry or ground-truth field is constructed by production inference.
"""
from __future__ import annotations

import copy
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import torch

from reassembly.prepare import signed_distance
from reassembly.solver import ScaffoldGrid


@runtime_checkable
class FieldSampler(Protocol):
    def sample(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return signed distance [N], confidence [N] and its XYZ derivative [N,3]."""
        ...


def _points(value):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("field queries must be finite [N,3] reference-frame coordinates")
    return points


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _confidence(sigma):
    return np.clip(.01 / (.01 + np.asarray(sigma)), .02, 1.)


def _clip(distance, gradient, truncation):
    if truncation is None:
        return distance, gradient
    gradient = gradient.copy()
    gradient[np.abs(distance) >= truncation] = 0
    return np.clip(distance, -truncation, truncation), gradient


class ContinuousNeuralField:
    """Evaluate a frozen private field copy; only query coordinates get gradients.

    Compatible with v2 and repaired models exposing ``field.context`` and
    ``field(encoded, queries, context=...)``. The encoder/context is computed once
    and detached. The caller's modes, parameters and existing gradients are never
    changed, including when this adapter runs under ``torch.no_grad()``.
    """
    diagnostic_only = False

    def __init__(self, model, encoded: dict, chunk_size: int = 2048,
                 truncation: float | None = None):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)
        self.truncation = float(truncation if truncation is not None else getattr(model.field, 'truncation', .1))
        if self.truncation <= 0:
            raise ValueError("truncation must be positive")
        # Clones also make inputs safe when constructed under inference_mode.
        with torch.inference_mode(False), torch.no_grad():
            self.module = copy.deepcopy(model.field).eval().float().requires_grad_(False)
            for parameter in self.module.parameters():
                parameter.grad = None
            self.encoded = {k: (v.detach().clone().float() if v.is_floating_point() else v.detach().clone())
                            if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k, v in encoded.items()}
            floating = next((v for v in self.encoded.values() if isinstance(v, torch.Tensor) and v.is_floating_point()), None)
            if floating is None:
                raise ValueError("encoded inputs must contain a floating-point tensor")
            if floating.ndim == 0 or floating.shape[0] != 1:
                raise ValueError("continuous assembly fields require one encoded object")
            self.device = floating.device
            self.module.to(self.device)
            self.context = self.module.context(self.encoded) if hasattr(self.module, 'context') else None

    def _predict(self, query):
        if self.context is None:
            return self.module(self.encoded, query)
        return self.module(self.encoded, query, context=self.context)

    def values(self, points, *, untruncated=False):
        points = _points(points)
        distances, sigmas = [], []
        with torch.inference_mode(False), torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
            for start in range(0, len(points), self.chunk_size):
                query = torch.tensor(points[start:start+self.chunk_size], device=self.device, dtype=torch.float32)[None]
                prediction = self._predict(query)
                distances.append(_numpy(prediction['distance']).reshape(-1))
                sigmas.append(np.exp(_numpy(prediction['log_scale']).reshape(-1)))
        distance = np.concatenate(distances).astype(np.float64) if distances else np.empty(0)
        sigma = np.concatenate(sigmas).astype(np.float64) if sigmas else np.empty(0)
        if not untruncated:
            distance = np.clip(distance, -self.truncation, self.truncation)
        return distance, sigma

    def raw_distance(self, points):
        """The network output before adapter clipping (its head may be bounded)."""
        return self.values(points, untruncated=True)[0]

    def raw_values(self, points):
        return self.values(points, untruncated=True)

    def sample(self, points):
        points = _points(points)
        distances, sigmas, gradients = [], [], []
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast(device_type=self.device.type, enabled=False):
            for start in range(0, len(points), self.chunk_size):
                query = torch.tensor(points[start:start+self.chunk_size], device=self.device, dtype=torch.float32)[None].requires_grad_(True)
                prediction = self._predict(query)
                value = prediction['distance']
                gradient = torch.autograd.grad(value.sum(), query, allow_unused=True)[0] if value.requires_grad else None
                distances.append(_numpy(value).reshape(-1))
                sigmas.append(np.exp(_numpy(prediction['log_scale']).reshape(-1)))
                gradients.append(np.zeros((query.shape[1], 3)) if gradient is None else _numpy(gradient)[0])
        distance = np.concatenate(distances).astype(np.float64) if distances else np.empty(0)
        sigma = np.concatenate(sigmas).astype(np.float64) if sigmas else np.empty(0)
        gradient = np.concatenate(gradients).astype(np.float64) if gradients else np.empty((0, 3))
        distance, gradient = _clip(distance, gradient, self.truncation)
        if not (np.isfinite(distance).all() and np.isfinite(sigma).all() and np.isfinite(gradient).all()):
            raise ValueError("continuous neural field returned non-finite values")
        return distance, _confidence(sigma), gradient


class ContinuousGTField:
    """Evaluation-only mesh SDF with nearest-surface distance gradients.

    A reference query maps to source coordinates as
    ``source = query * scale @ rotation + center``. Gradients map back as
    ``source_gradient @ rotation.T``; the SDF's scale division cancels the
    coordinate scale. Face normals define the derivative exactly on a surface;
    SDF derivatives at edges/medial-axis ties are inherently non-unique.
    """
    diagnostic_only = True

    def __init__(self, mesh, *, rotation=None, center=None, scale=1., truncation=.1,
                 uncertainty=.0025, chunk_size=512):
        self.mesh = mesh.copy()
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
        self.center = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
        self.scale, self.truncation = float(scale), float(truncation)
        self.uncertainty, self.chunk_size = float(uncertainty), int(chunk_size)
        if self.rotation.shape != (3, 3) or self.center.shape != (3,) or not np.isfinite(self.center).all():
            raise ValueError("GT field requires a proper rotation and XYZ center")
        if not np.allclose(self.rotation @ self.rotation.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(self.rotation), 1, atol=1e-6):
            raise ValueError("GT source rotation must be proper")
        if not (np.isfinite(self.scale) and self.scale > 0 and np.isfinite(self.truncation) and self.truncation > 0):
            raise ValueError("GT scale and truncation must be finite and positive")
        if self.chunk_size < 1 or not np.isfinite(self.uncertainty) or self.uncertainty < 0:
            raise ValueError("invalid GT field chunk size or uncertainty")
        if not self.mesh.is_watertight or not self.mesh.is_winding_consistent or self.mesh.volume <= 0:
            raise ValueError("GT field requires watertight consistently oriented positive-volume source geometry")

    @classmethod
    def from_sample(cls, sample, **kwargs):
        import trimesh
        with np.load(sample['source_mesh_path'], allow_pickle=False) as source:
            mesh = trimesh.Trimesh(source['vertices'], source['faces'], process=False)
        return cls(mesh, rotation=_numpy(sample['anchor_source_rotation']).reshape(3, 3),
                   center=_numpy(sample['anchor_source_centroid']).reshape(3),
                   scale=float(_numpy(sample['shared_scale'])), **kwargs)

    def values(self, points, *, untruncated=False):
        points = _points(points)
        source = (points*self.scale) @ self.rotation + self.center
        distance = signed_distance(self.mesh, source, chunk_size=self.chunk_size)/self.scale if len(source) else np.empty(0)
        if not untruncated:
            distance = np.clip(distance, -self.truncation, self.truncation)
        return distance, np.full(len(points), self.uncertainty)

    def raw_distance(self, points):
        return self.values(points, untruncated=True)[0]

    def raw_values(self, points):
        return self.values(points, untruncated=True)

    def sample(self, points):
        import trimesh
        points = _points(points)
        distance = self.raw_distance(points)
        source = (points*self.scale) @ self.rotation + self.center
        gradient = np.zeros_like(points)
        for start in range(0, len(points), self.chunk_size):
            query = source[start:start+self.chunk_size]
            nearest, unsigned, faces = trimesh.proximity.closest_point(self.mesh, query)
            local = query-nearest
            on_surface = unsigned <= 1e-10*self.scale
            local /= np.maximum(unsigned[:, None], 1e-15)
            local *= np.where(distance[start:start+len(query)] < 0, -1., 1.)[:, None]
            local[on_surface] = self.mesh.face_normals[faces[on_surface]]
            gradient[start:start+len(query)] = local @ self.rotation.T
        distance, gradient = _clip(distance, gradient, self.truncation)
        return distance, _confidence(np.full(len(points), self.uncertainty)), gradient


class ShiftedField:
    """A deliberately misleading diagnostic field; uncertainty is unchanged."""
    diagnostic_only = True

    def __init__(self, base: FieldSampler, translation=(.15, 0., 0.), offset=.035, truncation=None):
        self.base = base
        self.translation = np.asarray(translation, dtype=np.float64)
        self.offset = float(offset)
        self.truncation = float(truncation if truncation is not None else getattr(base, 'truncation', .1))
        if self.translation.shape != (3,) or not np.isfinite(self.translation).all() or not np.isfinite(self.offset) or self.truncation <= 0:
            raise ValueError("invalid diagnostic field perturbation")

    def sample(self, points):
        distance, confidence, gradient = self.base.sample(_points(points)-self.translation)
        distance, gradient = _clip(distance+self.offset, gradient, self.truncation)
        return distance, confidence, gradient


class ClippedGridField:
    """Interpolate untruncated node values, then clip value and derivative."""
    diagnostic_only = True

    def __init__(self, grid):
        self.grid = grid
        self.truncation = grid.truncation
        self.bounds = grid.bounds
        self.distance = grid.distance
        self.uncertainty = grid.uncertainty

    def sample(self, points):
        points = _points(points)
        distance, confidence, gradient = self.grid.sample(points)
        # Preserve the existing grid's explicit boundary residual outside bounds.
        inside = ((points >= self.bounds[0]) & (points <= self.bounds[1])).all(-1)
        distance[inside], gradient[inside] = _clip(distance[inside], gradient[inside], self.truncation)
        return distance, confidence, gradient

    def as_dict(self):
        return self.grid.as_dict()


def diagnostic_grid(field, resolution=32, extent=2.25, *, bounds=None, chunk_size=2048,
                    truncate_before_interpolation=True, progress: Callable[[int, int], None] | None = None):
    """Fresh field-node evaluation for diagnostics; never resample an older grid.

    ``progress(completed_nodes,total_nodes)`` runs before each bounded chunk and
    at completion, so callers can enforce deadlines and storage/memory checks.
    """
    resolution, chunk_size = int(resolution), int(chunk_size)
    bounds = np.asarray([[-extent]*3, [extent]*3] if bounds is None else bounds, dtype=np.float64)
    if resolution < 2 or chunk_size < 1 or bounds.shape != (2, 3) or not np.isfinite(bounds).all() or not np.all(bounds[1] > bounds[0]):
        raise ValueError("invalid diagnostic grid shape, bounds or chunk size")
    truncation = float(getattr(field, 'truncation', .1))
    total = resolution**3
    distances, sigmas = np.empty(total, np.float32), np.empty(total, np.float32)
    for start in range(0, total, chunk_size):
        if progress is not None:
            progress(start, total)
        index = np.arange(start, min(start+chunk_size, total))
        coordinates = np.stack(np.unravel_index(index, (resolution,)*3), -1)
        query = bounds[0] + coordinates/(resolution-1)*(bounds[1]-bounds[0])
        if hasattr(field, 'values'):
            value, sigma = field.values(query, untruncated=True)
        else:
            if not truncate_before_interpolation:
                raise ValueError("post-interpolation clipping requires a field exposing untruncated values")
            value, confidence, _ = field.sample(query)
            sigma = .01*(1/np.maximum(confidence, 1e-12)-1)
        distances[index] = np.clip(value, -truncation, truncation) if truncate_before_interpolation else value
        sigmas[index] = sigma
    if progress is not None:
        progress(total, total)
    grid = ScaffoldGrid(distances.reshape((resolution,)*3), sigmas.reshape((resolution,)*3), bounds, truncation)
    return grid if truncate_before_interpolation else ClippedGridField(grid)
