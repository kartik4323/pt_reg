"""GPAT-style dense matching and rigid pose extraction for Stage 3 assembly."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.token_encoder import (
    FragmentEncoding,
    _batched_fps_indices,
    _gather_points,
    build_token_encoder,
)


def _alignment_compute_dtype(*tensors: torch.Tensor) -> torch.dtype:
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _autocast_disabled(device_type: str):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=False)
    if device_type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def apply_transform(points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Apply x' = R x + t to row-vector point tensors."""
    return points @ rotation.transpose(-1, -2) + translation.unsqueeze(-2)


def apply_fragment_transforms(
    fragments: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
) -> torch.Tensor:
    """Apply per-fragment transforms to tensors shaped (B, F, N, 3)."""
    return torch.einsum("bfnd,bfcd->bfnc", fragments, rotations) + translations[:, :, None, :]


def kabsch_align(source: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Best-fit rigid transform from source to target.

    Both tensors have shape (B, N, 3), and correspondences are assumed to be
    ordered. Returns rotation and translation using row-vector convention:
    aligned = source @ R.T + t.
    """
    output_dtype = source.dtype
    compute_dtype = _alignment_compute_dtype(source, target)
    with _autocast_disabled(source.device.type):
        source_f = source.to(dtype=compute_dtype)
        target_f = target.to(dtype=compute_dtype)
        src_centroid = source_f.mean(dim=1, keepdim=True)
        tgt_centroid = target_f.mean(dim=1, keepdim=True)
        src_centered = source_f - src_centroid
        tgt_centered = target_f - tgt_centroid
        cov = src_centered.transpose(1, 2) @ tgt_centered

        u, _, vh = torch.linalg.svd(cov)
        rotation = vh.transpose(1, 2) @ u.transpose(1, 2)
        det = torch.det(rotation)
        needs_fix = det < 0
        if needs_fix.any():
            vh = vh.clone()
            vh[needs_fix, -1, :] *= -1
            rotation = vh.transpose(1, 2) @ u.transpose(1, 2)

        rotated_source_centroid = (
            src_centroid.squeeze(1).unsqueeze(1) @ rotation.transpose(1, 2)
        ).squeeze(1)
        translation = tgt_centroid.squeeze(1) - rotated_source_centroid
    return rotation.to(dtype=output_dtype), translation.to(dtype=output_dtype)


@torch.no_grad()
def differentiable_icp_initialization(
    fragments: torch.Tensor,
    reconstruction: torch.Tensor,
    fragment_mask: torch.Tensor,
    iterations: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ICP-style SE(3) initialization.

    Kept as an optional initialization path for inference/training. The learned
    Stage 3 model below does not regress pose; it estimates a residual pose by
    differentiable matching plus Kabsch.
    """
    batch_size, max_fragments, points_per_fragment, _ = fragments.shape
    rotations = torch.eye(3, device=fragments.device, dtype=fragments.dtype).repeat(
        batch_size, max_fragments, 1, 1
    )
    translations = torch.zeros(
        batch_size, max_fragments, 3, device=fragments.device, dtype=fragments.dtype
    )
    aligned = fragments.clone()

    for _ in range(iterations):
        flat_aligned = aligned.reshape(batch_size * max_fragments, points_per_fragment, 3)
        recon_expanded = reconstruction[:, None, :, :].expand(
            batch_size, max_fragments, reconstruction.shape[1], 3
        )
        flat_recon = recon_expanded.reshape(
            batch_size * max_fragments, reconstruction.shape[1], 3
        )
        dist = torch.cdist(flat_aligned, flat_recon)
        nearest = dist.argmin(dim=2)
        matched = torch.gather(
            flat_recon,
            dim=1,
            index=nearest.unsqueeze(-1).expand(-1, -1, 3),
        )
        step_rot, step_trans = kabsch_align(flat_aligned, matched)
        flat_next = apply_transform(flat_aligned, step_rot, step_trans)
        aligned = flat_next.reshape(batch_size, max_fragments, points_per_fragment, 3)

        step_rot = step_rot.reshape(batch_size, max_fragments, 3, 3)
        step_trans = step_trans.reshape(batch_size, max_fragments, 3)
        rotations = step_rot @ rotations
        translations = (
            (translations.unsqueeze(-2) @ step_rot.transpose(-1, -2)).squeeze(-2)
            + step_trans
        )

    eye = torch.eye(3, device=fragments.device, dtype=fragments.dtype)[None, None]
    rotations = torch.where(fragment_mask[:, :, None, None], rotations, eye)
    translations = torch.where(fragment_mask[:, :, None], translations, torch.zeros_like(translations))
    aligned = torch.where(fragment_mask[:, :, None, None], aligned, torch.zeros_like(aligned))
    return rotations, translations, aligned


def rotation_geodesic_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (radians) between rotations. For *reporting* only.

    ``acos`` has an unbounded derivative near ``|cos| -> 1`` (i.e. exactly as the
    prediction converges to the target), so this must not be used as a training
    objective. Use :func:`rotation_chordal_error` for gradients.
    """
    rot_delta = pred.transpose(-1, -2) @ target
    trace = rot_delta.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
    return torch.acos(cos_theta)


def rotation_chordal_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared chordal (Frobenius) distance ``||R_pred - R_gt||_F^2``.

    Smooth everywhere with a gradient that *vanishes* at convergence, so it is
    safe to backprop through the differentiable Kabsch/SVD. Related to the
    geodesic angle by ``||R_pred - R_gt||_F^2 = 4 (1 - cos theta)``, so it is on
    a comparable scale to the (radian) geodesic term it replaces. Returns a
    per-rotation tensor with the trailing (3, 3) dims reduced.
    """
    return ((pred - target) ** 2).sum(dim=(-2, -1))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Zhou et al. (2019) 6D continuity representation -> SO(3) via Gram-Schmidt.

    ``d6``: (..., 6). Returns (..., 3, 3) rotation matrices (orthonormal rows,
    det = +1). Continuous and differentiable, unlike quaternion/Euler
    parametrizations (which have double-cover / gimbal discontinuities that are
    poor regression targets). The output is always a proper rotation, so no
    reflection failure mode and no renormalization step is needed.
    """
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def _knn_indices(points: torch.Tensor, k: int) -> torch.Tensor:
    """Return k nearest-neighbor indices per point, excluding self when possible."""
    num_points = points.shape[1]
    if num_points <= 1:
        return torch.zeros(points.shape[0], num_points, 1, dtype=torch.long, device=points.device)
    k_eff = min(k + 1, num_points)
    dist = torch.cdist(points, points, p=2)
    return dist.topk(k=k_eff, dim=-1, largest=False).indices[..., 1:]


def _gather_neighbors(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # Gather k neighbors per point without materializing the (B, N, N, dim)
    # intermediate that a broadcast-expand + gather would create. On a 4096-point
    # object with dim=256 that intermediate (and its backward grad) is ~17 GB at
    # batch 1; the flat index_select below produces the (B, N, k, dim) result
    # directly (~tens of MB).
    batch_size, num_points, dim = values.shape
    k = indices.shape[-1]
    offset = (torch.arange(batch_size, device=values.device) * num_points).view(batch_size, 1, 1)
    flat_index = (indices + offset).reshape(-1)
    gathered = values.reshape(batch_size * num_points, dim)[flat_index]
    return gathered.reshape(batch_size, num_points, k, dim)


class PoseSensitivePointEncoder(nn.Module):
    """
    Dense DGCNN/PointNet++-style point encoder for GPAT matching.

    This module preserves one feature vector per input point. It uses absolute
    coordinates, centered coordinates, radial cues, and local kNN EdgeConv
    aggregation, so the transformer receives dense geometry instead of a pooled
    global token.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_neighbors: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.input_proj = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 10, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.dim() != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape (B, N, 3)")

        centroid = points.mean(dim=1, keepdim=True)
        centered = points - centroid
        radius = torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
        base = torch.cat([points, centered, radius, points.square()], dim=-1)
        point_features = self.input_proj(base)

        indices = _knn_indices(centered.detach(), self.num_neighbors)
        neighbor_features = _gather_neighbors(point_features, indices)
        neighbor_points = _gather_neighbors(centered, indices)
        central_features = point_features[:, :, None, :].expand_as(neighbor_features)
        central_points = centered[:, :, None, :].expand_as(neighbor_points)
        edge_features = torch.cat(
            [
                central_features,
                neighbor_features - central_features,
                neighbor_points - central_points,
            ],
            dim=-1,
        )
        local_features = self.edge_mlp(edge_features).max(dim=2).values
        return self.out(torch.cat([point_features, local_features, base], dim=-1))


class GPATTransformerBlock(nn.Module):
    """One GPAT coarse-to-fine block with self attention and bidirectional cross attention."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fragment_self = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.object_self = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fragment_to_object = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.object_to_fragment = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fragment_norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(3)])
        self.object_norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(3)])
        self.fragment_ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.object_ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.fragment_ffn_norm = nn.LayerNorm(embedding_dim)
        self.object_ffn_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        fragment_features: torch.Tensor,
        object_features: torch.Tensor,
        fragment_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_fragments, points_per_fragment, dim = fragment_features.shape
        point_mask = fragment_mask[:, :, None].expand(batch_size, max_fragments, points_per_fragment)
        flat_point_mask = point_mask.reshape(batch_size, max_fragments * points_per_fragment)

        flat_fragments = fragment_features.reshape(
            batch_size * max_fragments,
            points_per_fragment,
            dim,
        )
        frag_self, _ = self.fragment_self(flat_fragments, flat_fragments, flat_fragments, need_weights=False)
        flat_fragments = self.fragment_norms[0](flat_fragments + frag_self)
        fragment_features = flat_fragments.reshape(batch_size, max_fragments, points_per_fragment, dim)
        fragment_features = torch.where(point_mask[..., None], fragment_features, torch.zeros_like(fragment_features))

        obj_self, _ = self.object_self(object_features, object_features, object_features, need_weights=False)
        object_features = self.object_norms[0](object_features + obj_self)

        flat_fragments = fragment_features.reshape(batch_size, max_fragments * points_per_fragment, dim)
        frag_cross, _ = self.fragment_to_object(
            flat_fragments,
            object_features,
            object_features,
            need_weights=False,
        )
        flat_fragments = self.fragment_norms[1](flat_fragments + frag_cross)
        flat_fragments = torch.where(flat_point_mask[..., None], flat_fragments, torch.zeros_like(flat_fragments))

        obj_cross, _ = self.object_to_fragment(
            object_features,
            flat_fragments,
            flat_fragments,
            key_padding_mask=~flat_point_mask,
            need_weights=False,
        )
        object_features = self.object_norms[1](object_features + obj_cross)

        flat_fragments = self.fragment_ffn_norm(flat_fragments + self.fragment_ffn(flat_fragments))
        flat_fragments = torch.where(flat_point_mask[..., None], flat_fragments, torch.zeros_like(flat_fragments))
        object_features = self.object_ffn_norm(object_features + self.object_ffn(object_features))

        return (
            flat_fragments.reshape(batch_size, max_fragments, points_per_fragment, dim),
            object_features,
        )


class GPATCoarseToFineTransformer(nn.Module):
    """Dense bidirectional transformer used before open-vocabulary matching."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GPATTransformerBlock(
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        fragment_features: torch.Tensor,
        object_features: torch.Tensor,
        fragment_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            fragment_features, object_features = layer(
                fragment_features,
                object_features,
                fragment_mask,
            )
        return fragment_features, object_features


class DenseMatchingHead(nn.Module):
    """Temperature-scaled dot-product matching plus Sinkhorn/dual-softmax transport."""

    def __init__(
        self,
        embedding_dim: int,
        sinkhorn_iterations: int = 8,
        initial_temperature: float = 0.07,
        transport: str = "sinkhorn",
    ) -> None:
        super().__init__()
        self.fragment_proj = nn.Linear(embedding_dim, embedding_dim)
        self.object_proj = nn.Linear(embedding_dim, embedding_dim)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(initial_temperature))))
        self.sinkhorn_iterations = sinkhorn_iterations
        self.transport = transport

    def forward(
        self,
        fragment_features: torch.Tensor,
        object_features: torch.Tensor,
        fragment_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frag = F.normalize(self.fragment_proj(fragment_features), dim=-1)
        obj = F.normalize(self.object_proj(object_features), dim=-1)
        temperature = self.log_temperature.exp().clamp(min=1.0e-3, max=1.0)
        logits = torch.einsum("bfnd,bmd->bfnm", frag, obj) / temperature

        if self.transport == "dual_softmax":
            row_prob = logits.softmax(dim=-1)
            col_prob = logits.softmax(dim=-2)
            assignment = row_prob * col_prob
            assignment = assignment / assignment.sum(dim=(-1, -2), keepdim=True).clamp(min=1.0e-8)
        else:
            assignment = self._sinkhorn(logits)

        assignment = torch.where(
            fragment_mask[:, :, None, None],
            assignment,
            torch.zeros_like(assignment),
        )
        logits = torch.where(
            fragment_mask[:, :, None, None],
            logits,
            torch.zeros_like(logits),
        )
        return logits, assignment

    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, max_fragments, points_per_fragment, target_points = logits.shape
        log_transport = logits.reshape(batch_size * max_fragments, points_per_fragment, target_points)
        log_mu = torch.full(
            (batch_size * max_fragments, points_per_fragment),
            1.0 / float(points_per_fragment),
            device=logits.device,
            dtype=logits.dtype,
        ).log()
        log_nu = torch.full(
            (batch_size * max_fragments, target_points),
            1.0 / float(target_points),
            device=logits.device,
            dtype=logits.dtype,
        ).log()
        for _ in range(self.sinkhorn_iterations):
            log_transport = log_transport + (
                log_mu - torch.logsumexp(log_transport, dim=2)
            ).unsqueeze(2)
            log_transport = log_transport + (
                log_nu - torch.logsumexp(log_transport, dim=1)
            ).unsqueeze(1)
        return log_transport.exp().reshape(batch_size, max_fragments, points_per_fragment, target_points)


class DifferentiableKabsch(nn.Module):
    """Weighted SVD rigid alignment from dense assignment matrices."""

    def __init__(self, eps: float = 1.0e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        fragments: torch.Tensor,
        target: torch.Tensor,
        assignment: torch.Tensor,
        fragment_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_fragments, _, _ = fragments.shape
        output_dtype = fragments.dtype
        compute_dtype = _alignment_compute_dtype(fragments, target, assignment)
        with _autocast_disabled(fragments.device.type):
            fragments_f = fragments.to(dtype=compute_dtype)
            target_f = target.to(dtype=compute_dtype)
            assignment_f = assignment.to(dtype=compute_dtype)

            total_weight = assignment_f.sum(dim=(-1, -2)).clamp(min=self.eps)
            source_mass = assignment_f.sum(dim=-1)
            target_mass = assignment_f.sum(dim=-2)

            source_centroid = (
                source_mass[..., None] * fragments_f
            ).sum(dim=2) / total_weight[..., None]
            target_centroid = (
                target_mass[..., None] * target_f[:, None]
            ).sum(dim=2) / total_weight[..., None]

            source_centered = fragments_f - source_centroid[:, :, None, :]
            target_centered = target_f[:, None, :, :] - target_centroid[:, :, None, :]
            covariance = torch.einsum(
                "bfnm,bfni,bfmj->bfij",
                assignment_f,
                source_centered,
                target_centered,
            )

            flat_cov = covariance.reshape(batch_size * max_fragments, 3, 3)
            u, _, vh = torch.linalg.svd(flat_cov)
            v = vh.transpose(-1, -2)
            det = torch.det(v @ u.transpose(-1, -2))
            correction = torch.ones(
                batch_size * max_fragments,
                3,
                device=fragments.device,
                dtype=compute_dtype,
            )
            correction[:, -1] = torch.where(det < 0.0, -1.0, 1.0)
            rotation = v @ torch.diag_embed(correction) @ u.transpose(-1, -2)
            rotation = rotation.reshape(batch_size, max_fragments, 3, 3)

            translation = target_centroid - (
                source_centroid.unsqueeze(-2) @ rotation.transpose(-1, -2)
            ).squeeze(-2)

        rotation = rotation.to(dtype=output_dtype)
        translation = translation.to(dtype=output_dtype)
        eye = torch.eye(3, device=fragments.device, dtype=output_dtype)[None, None]
        rotation = torch.where(fragment_mask[:, :, None, None], rotation, eye)
        translation = torch.where(fragment_mask[:, :, None], translation, torch.zeros_like(translation))
        return rotation, translation


@dataclass
class PoseEstimatorOutput:
    rotations: torch.Tensor
    translations: torch.Tensor
    aligned_fragments: torch.Tensor
    aligned_union: torch.Tensor
    fragment_features: torch.Tensor
    object_features: torch.Tensor
    refined_fragment_features: torch.Tensor
    refined_object_features: torch.Tensor
    matching_logits: torch.Tensor
    assignment_matrix: torch.Tensor
    target_logits: Optional[torch.Tensor] = None
    target_probabilities: Optional[torch.Tensor] = None
    fragment_superpoints: Optional[torch.Tensor] = None
    object_superpoints: Optional[torch.Tensor] = None

    @property
    def fragment_tokens(self) -> torch.Tensor:
        return self.fragment_features

    @property
    def object_tokens(self) -> torch.Tensor:
        return self.object_features

    @property
    def refined_fragment_tokens(self) -> torch.Tensor:
        return self.refined_fragment_features


class FragmentObjectPoseEstimator(nn.Module):
    """
    GPAT-style fragment-to-object registration network.

    Dense fragment and target point features are refined with alternating
    self/cross attention. A temperature-scaled matching head produces an
    optimal-transport assignment matrix, and DifferentiableKabsch converts the
    correspondences into rigid transforms.
    """

    def __init__(
        self,
        fragment_encoder: nn.Module,
        object_encoder: nn.Module,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        cross_attention_layers: int = 2,
        transformer_layers: int = 2,
        dropout: float = 0.0,
        freeze_encoders: bool = True,
        num_neighbors: int = 16,
        sinkhorn_iterations: int = 8,
        matching_temperature: float = 0.07,
        transport: str = "sinkhorn",
    ) -> None:
        super().__init__()
        self.fragment_encoder = fragment_encoder
        self.object_encoder = object_encoder
        self.embedding_dim = embedding_dim
        self.freeze_encoders = freeze_encoders

        self.fragment_pose_encoder = PoseSensitivePointEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_neighbors=num_neighbors,
            dropout=dropout,
        )
        self.object_pose_encoder = PoseSensitivePointEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_neighbors=num_neighbors,
            dropout=dropout,
        )
        self.fragment_point_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        self.object_point_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        total_layers = max(int(cross_attention_layers), int(transformer_layers), 1)
        self.gpat_transformer = GPATCoarseToFineTransformer(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=total_layers,
            dropout=dropout,
        )
        self.matching_head = DenseMatchingHead(
            embedding_dim=embedding_dim,
            sinkhorn_iterations=sinkhorn_iterations,
            initial_temperature=matching_temperature,
            transport=transport,
        )
        self.kabsch = DifferentiableKabsch()

        if freeze_encoders:
            self.freeze_pretrained()

    def freeze_pretrained(self) -> None:
        for module in (self.fragment_encoder, self.object_encoder):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_pretrained(self) -> None:
        for module in (self.fragment_encoder, self.object_encoder):
            for param in module.parameters():
                param.requires_grad = True

    def forward(
        self,
        fragments: torch.Tensor,
        reconstructed_object: torch.Tensor,
        fragment_mask: torch.Tensor,
        initial_rotations: Optional[torch.Tensor] = None,
        initial_translations: Optional[torch.Tensor] = None,
    ) -> PoseEstimatorOutput:
        if fragments.dim() != 4 or fragments.shape[-1] != 3:
            raise ValueError("fragments must have shape (B, F, N, 3)")
        if reconstructed_object.dim() != 3 or reconstructed_object.shape[-1] != 3:
            raise ValueError("reconstructed_object must have shape (B, M, 3)")

        batch_size, max_fragments, points_per_fragment, _ = fragments.shape
        if initial_rotations is None:
            initial_rotations = torch.eye(
                3,
                device=fragments.device,
                dtype=fragments.dtype,
            )[None, None].expand(batch_size, max_fragments, 3, 3)
        if initial_translations is None:
            initial_translations = torch.zeros(
                batch_size,
                max_fragments,
                3,
                device=fragments.device,
                dtype=fragments.dtype,
            )

        initialized_fragments = apply_fragment_transforms(
            fragments,
            initial_rotations,
            initial_translations,
        )
        initialized_fragments = torch.where(
            fragment_mask[:, :, None, None],
            initialized_fragments,
            torch.zeros_like(initialized_fragments),
        )

        flat_fragments = initialized_fragments.reshape(batch_size * max_fragments, points_per_fragment, 3)
        dense_fragment_pretrained = self._encode_fragments(flat_fragments).point_features.reshape(
            batch_size,
            max_fragments,
            points_per_fragment,
            self.embedding_dim,
        )
        dense_object_pretrained = self._encode_object(reconstructed_object).point_features

        dense_fragment_pose = self.fragment_pose_encoder(flat_fragments).reshape(
            batch_size,
            max_fragments,
            points_per_fragment,
            self.embedding_dim,
        )
        dense_object_pose = self.object_pose_encoder(reconstructed_object)

        fragment_features = self.fragment_point_fusion(
            torch.cat(
                [
                    dense_fragment_pretrained,
                    dense_fragment_pose,
                    initialized_fragments,
                ],
                dim=-1,
            )
        )
        object_features = self.object_point_fusion(
            torch.cat(
                [
                    dense_object_pretrained,
                    dense_object_pose,
                    reconstructed_object,
                ],
                dim=-1,
            )
        )
        fragment_features = torch.where(
            fragment_mask[:, :, None, None],
            fragment_features,
            torch.zeros_like(fragment_features),
        )

        refined_fragments, refined_object = self.gpat_transformer(
            fragment_features,
            object_features,
            fragment_mask,
        )
        matching_logits, assignment = self.matching_head(
            refined_fragments,
            refined_object,
            fragment_mask,
        )
        residual_rotations, residual_translations = self.kabsch(
            initialized_fragments,
            reconstructed_object,
            assignment,
            fragment_mask,
        )

        rotations = residual_rotations @ initial_rotations
        translations = (
            (initial_translations.unsqueeze(-2) @ residual_rotations.transpose(-1, -2)).squeeze(-2)
            + residual_translations
        )

        eye = torch.eye(3, device=fragments.device, dtype=fragments.dtype)[None, None]
        rotations = torch.where(fragment_mask[:, :, None, None], rotations, eye)
        translations = torch.where(fragment_mask[:, :, None], translations, torch.zeros_like(translations))
        aligned = apply_fragment_transforms(fragments, rotations, translations)
        aligned = torch.where(fragment_mask[:, :, None, None], aligned, torch.zeros_like(aligned))
        aligned_union = aligned.reshape(batch_size, max_fragments * points_per_fragment, 3)

        return PoseEstimatorOutput(
            rotations=rotations,
            translations=translations,
            aligned_fragments=aligned,
            aligned_union=aligned_union,
            fragment_features=fragment_features,
            object_features=object_features,
            refined_fragment_features=refined_fragments,
            refined_object_features=refined_object,
            matching_logits=matching_logits,
            assignment_matrix=assignment,
        )

    def _encode_fragments(self, flat_fragments: torch.Tensor) -> FragmentEncoding:
        if self.freeze_encoders:
            with torch.no_grad():
                return self.fragment_encoder(flat_fragments)
        return self.fragment_encoder(flat_fragments)

    def _encode_object(self, reconstructed_object: torch.Tensor) -> FragmentEncoding:
        if self.freeze_encoders:
            with torch.no_grad():
                return self.object_encoder(reconstructed_object)
        return self.object_encoder(reconstructed_object)


def masked_chamfer_distance(
    source: torch.Tensor,
    target: torch.Tensor,
    source_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chamfer distance from a masked source point set to an unmasked target."""
    dist = torch.cdist(source, target, p=2)
    if source_mask is None:
        source_mask = torch.ones(source.shape[:2], dtype=torch.bool, device=source.device)
    weights = source_mask.to(source.dtype)
    source_to_target = dist.min(dim=2).values
    loss_source = (source_to_target * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    masked_dist = dist.masked_fill(~source_mask[:, :, None], float("inf"))
    target_to_source = masked_dist.min(dim=1).values
    target_to_source = torch.where(
        torch.isfinite(target_to_source),
        target_to_source,
        torch.zeros_like(target_to_source),
    )
    loss_target = target_to_source.mean(dim=1)
    return (loss_source + loss_target).mean(), loss_source.mean(), loss_target.mean()


def overlap_loss(
    aligned_fragments: torch.Tensor,
    fragment_mask: torch.Tensor,
    threshold: float = 0.03,
) -> torch.Tensor:
    """Penalize different fragments occupying nearly identical regions."""
    _, max_fragments, _, _ = aligned_fragments.shape
    losses = []
    for i in range(max_fragments):
        for j in range(i + 1, max_fragments):
            valid = fragment_mask[:, i] & fragment_mask[:, j]
            if not valid.any():
                continue
            dist = torch.cdist(aligned_fragments[valid, i], aligned_fragments[valid, j])
            nearest_i = dist.min(dim=2).values
            nearest_j = dist.min(dim=1).values
            penalty = 0.5 * (
                F.relu(threshold - nearest_i).mean(dim=1)
                + F.relu(threshold - nearest_j).mean(dim=1)
            )
            losses.append(penalty)
    if not losses:
        return aligned_fragments.sum() * 0.0
    return torch.cat(losses, dim=0).mean()


class PoseAssemblyLoss(nn.Module):
    """GPAT loss: dense matching supervision plus geometry refinement terms."""

    def __init__(
        self,
        lambda_matching: float = 2.0,
        lambda_recon: float = 1.0,
        lambda_coverage: float = 1.0,
        lambda_gt: float = 0.0,
        lambda_overlap: float = 0.1,
        lambda_pose: float = 0.0,
        overlap_threshold: float = 0.03,
        matching_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_matching = lambda_matching
        self.lambda_recon = lambda_recon
        self.lambda_coverage = lambda_coverage
        self.lambda_gt = lambda_gt
        self.lambda_overlap = lambda_overlap
        self.lambda_pose = lambda_pose
        self.overlap_threshold = overlap_threshold
        self.matching_temperature = matching_temperature

    def forward(
        self,
        output: PoseEstimatorOutput,
        fragments: torch.Tensor,
        fragment_mask: torch.Tensor,
        reconstructed_object: torch.Tensor,
        ground_truth_object: Optional[torch.Tensor] = None,
        target_fragments: Optional[torch.Tensor] = None,
        gt_rotations: Optional[torch.Tensor] = None,
        gt_translations: Optional[torch.Tensor] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, max_fragments, points_per_fragment, _ = fragments.shape
        union_mask = fragment_mask[:, :, None].expand(batch_size, max_fragments, points_per_fragment)
        union_mask = union_mask.reshape(batch_size, max_fragments * points_per_fragment)

        loss_recon_cd, recon_fit, recon_coverage = masked_chamfer_distance(
            output.aligned_union,
            reconstructed_object,
            union_mask,
        )
        if ground_truth_object is not None:
            loss_gt, _, _ = masked_chamfer_distance(output.aligned_union, ground_truth_object, union_mask)
        else:
            loss_gt = output.aligned_union.sum() * 0.0

        matching_loss, matching_acc = self._matching_loss(
            output=output,
            fragments=fragments,
            fragment_mask=fragment_mask,
            reconstructed_object=reconstructed_object,
            target_fragments=target_fragments,
            gt_rotations=gt_rotations,
            gt_translations=gt_translations,
        )
        overlap = overlap_loss(output.aligned_fragments, fragment_mask, self.overlap_threshold)

        if gt_rotations is not None and gt_translations is not None:
            pose_loss = pose_supervised_loss(
                output.rotations,
                output.translations,
                gt_rotations,
                gt_translations,
                fragment_mask,
            )
        else:
            zero = output.aligned_union.sum() * 0.0
            pose_loss = {
                "loss": zero,
                "loss_rotation": zero.detach(),
                "loss_translation": zero.detach(),
            }

        loss_weights = {
            "lambda_matching": self.lambda_matching,
            "lambda_recon": self.lambda_recon,
            "lambda_coverage": self.lambda_coverage,
            "lambda_gt": self.lambda_gt,
            "lambda_overlap": self.lambda_overlap,
            "lambda_pose": self.lambda_pose,
        }
        if weights is not None:
            loss_weights.update(weights)

        total = (
            loss_weights["lambda_matching"] * matching_loss
            + loss_weights["lambda_recon"] * recon_fit
            + loss_weights["lambda_coverage"] * recon_coverage
            + loss_weights["lambda_gt"] * loss_gt
            + loss_weights["lambda_overlap"] * overlap
            + loss_weights["lambda_pose"] * pose_loss["loss"]
        )

        diagnostics = self._pose_diagnostics(
            output,
            gt_rotations,
            gt_translations,
            fragment_mask,
        )
        return {
            "loss": total,
            "loss_matching": matching_loss.detach(),
            "matching_accuracy": matching_acc.detach(),
            "loss_recon_cd": loss_recon_cd.detach(),
            "loss_recon_fit": recon_fit.detach(),
            "loss_coverage": recon_coverage.detach(),
            "loss_gt_cd": loss_gt.detach(),
            "loss_overlap": overlap.detach(),
            "loss_pose": pose_loss["loss"].detach(),
            "loss_rotation": pose_loss["loss_rotation"],
            "loss_translation": pose_loss["loss_translation"],
            **diagnostics,
        }

    def _matching_loss(
        self,
        output: PoseEstimatorOutput,
        fragments: torch.Tensor,
        fragment_mask: torch.Tensor,
        reconstructed_object: torch.Tensor,
        target_fragments: Optional[torch.Tensor],
        gt_rotations: Optional[torch.Tensor],
        gt_translations: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if target_fragments is None:
            if gt_rotations is None or gt_translations is None:
                return output.matching_logits.sum() * 0.0, output.matching_logits.sum() * 0.0
            target_fragments = apply_fragment_transforms(fragments, gt_rotations, gt_translations)

        batch_size, max_fragments, points_per_fragment, _ = target_fragments.shape
        flat_target_fragments = target_fragments.reshape(
            batch_size * max_fragments,
            points_per_fragment,
            3,
        )
        expanded_target = reconstructed_object[:, None].expand(
            batch_size,
            max_fragments,
            reconstructed_object.shape[1],
            3,
        ).reshape(batch_size * max_fragments, reconstructed_object.shape[1], 3)

        with torch.no_grad():
            labels = torch.cdist(flat_target_fragments, expanded_target, p=2).argmin(dim=-1)

        logits = output.matching_logits.reshape(
            batch_size * max_fragments,
            points_per_fragment,
            reconstructed_object.shape[1],
        )
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]) / self.matching_temperature,
            labels.reshape(-1),
            reduction="none",
        ).reshape(batch_size, max_fragments, points_per_fragment)
        point_mask = fragment_mask[:, :, None].expand_as(ce)
        weights = point_mask.to(ce.dtype)
        loss = (ce * weights).sum() / weights.sum().clamp(min=1.0)

        pred = output.assignment_matrix.argmax(dim=-1)
        accuracy = ((pred == labels.reshape(batch_size, max_fragments, points_per_fragment)) & point_mask).to(ce.dtype)
        accuracy = accuracy.sum() / weights.sum().clamp(min=1.0)
        return loss, accuracy

    @staticmethod
    def _pose_diagnostics(
        output: PoseEstimatorOutput,
        gt_rotations: Optional[torch.Tensor],
        gt_translations: Optional[torch.Tensor],
        fragment_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if gt_rotations is None or gt_translations is None:
            zero = output.aligned_union.sum() * 0.0
            return {
                "rotation_error_deg": zero.detach(),
                "translation_error": zero.detach(),
            }

        rot_error = rotation_geodesic_error(output.rotations, gt_rotations)
        trans_error = torch.linalg.vector_norm(output.translations - gt_translations, dim=-1)
        mask = fragment_mask.to(rot_error.dtype)
        denom = mask.sum().clamp(min=1.0)
        rotation_error_deg = (rot_error * mask).sum() / denom * 180.0 / torch.pi
        translation_error = (trans_error * mask).sum() / denom
        return {
            "rotation_error_deg": rotation_error_deg.detach(),
            "translation_error": translation_error.detach(),
        }


class TargetSegmentationHead(nn.Module):
    """Predict a part label for every target point, plus a background class."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        initial_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.part_proj = nn.Linear(embedding_dim, embedding_dim)
        self.object_proj = nn.Linear(embedding_dim, embedding_dim)
        self.background_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(initial_temperature))))

    def forward(
        self,
        fragment_features: torch.Tensor,
        object_features: torch.Tensor,
        fragment_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        part_features = fragment_features.mean(dim=2)
        part_features = F.normalize(self.part_proj(part_features), dim=-1)
        object_features = F.normalize(self.object_proj(object_features), dim=-1)
        temperature = self.log_temperature.exp().clamp(min=1.0e-3, max=1.0)

        part_logits = torch.einsum("bmd,bfd->bmf", object_features, part_features) / temperature
        part_logits = part_logits.masked_fill(~fragment_mask[:, None, :], -1.0e4)
        background_logits = self.background_head(object_features)
        logits = torch.cat([part_logits, background_logits], dim=-1)
        return logits, logits.softmax(dim=-1)


class TargetSegmentationPoseEstimator(nn.Module):
    """
    GPAT replacement for Stage 3.

    It predicts a target-point segmentation over the available fragments and
    converts the predicted target regions into SE(3) poses with weighted Kabsch.
    """

    def __init__(
        self,
        fragment_encoder: nn.Module,
        object_encoder: nn.Module,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        cross_attention_layers: int = 2,
        transformer_layers: int = 2,
        dropout: float = 0.0,
        freeze_encoders: bool = True,
        num_neighbors: int = 16,
        matching_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.fragment_encoder = fragment_encoder
        self.object_encoder = object_encoder
        self.embedding_dim = embedding_dim
        self.freeze_encoders = freeze_encoders

        self.fragment_pose_encoder = PoseSensitivePointEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_neighbors=num_neighbors,
            dropout=dropout,
        )
        self.object_pose_encoder = PoseSensitivePointEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_neighbors=num_neighbors,
            dropout=dropout,
        )
        self.fragment_point_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        self.object_point_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        total_layers = max(int(cross_attention_layers), int(transformer_layers), 1)
        self.gpat_transformer = GPATCoarseToFineTransformer(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=total_layers,
            dropout=dropout,
        )
        self.segmentation_head = TargetSegmentationHead(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            initial_temperature=matching_temperature,
        )
        self.fragment_match_proj = nn.Linear(embedding_dim, embedding_dim)
        self.object_match_proj = nn.Linear(embedding_dim, embedding_dim)
        self.log_matching_temperature = nn.Parameter(torch.log(torch.tensor(float(matching_temperature))))
        self.kabsch = DifferentiableKabsch()

        if freeze_encoders:
            self.freeze_pretrained()

    def freeze_pretrained(self) -> None:
        for module in (self.fragment_encoder, self.object_encoder):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_pretrained(self) -> None:
        for module in (self.fragment_encoder, self.object_encoder):
            for param in module.parameters():
                param.requires_grad = True

    def forward(
        self,
        fragments: torch.Tensor,
        target_object: torch.Tensor,
        fragment_mask: torch.Tensor,
        initial_rotations: Optional[torch.Tensor] = None,
        initial_translations: Optional[torch.Tensor] = None,
    ) -> PoseEstimatorOutput:
        del initial_rotations, initial_translations
        if fragments.dim() != 4 or fragments.shape[-1] != 3:
            raise ValueError("fragments must have shape (B, F, N, 3)")
        if target_object.dim() != 3 or target_object.shape[-1] != 3:
            raise ValueError("target_object must have shape (B, M, 3)")

        batch_size, max_fragments, points_per_fragment, _ = fragments.shape
        flat_fragments = fragments.reshape(batch_size * max_fragments, points_per_fragment, 3)
        dense_fragment_pretrained = self._encode_fragments(flat_fragments).point_features.reshape(
            batch_size,
            max_fragments,
            points_per_fragment,
            self.embedding_dim,
        )
        dense_object_pretrained = self._encode_object(target_object).point_features
        dense_fragment_pose = self.fragment_pose_encoder(flat_fragments).reshape(
            batch_size,
            max_fragments,
            points_per_fragment,
            self.embedding_dim,
        )
        dense_object_pose = self.object_pose_encoder(target_object)

        fragment_features = self.fragment_point_fusion(
            torch.cat([dense_fragment_pretrained, dense_fragment_pose, fragments], dim=-1)
        )
        object_features = self.object_point_fusion(
            torch.cat([dense_object_pretrained, dense_object_pose, target_object], dim=-1)
        )
        fragment_features = torch.where(
            fragment_mask[:, :, None, None],
            fragment_features,
            torch.zeros_like(fragment_features),
        )

        refined_fragments, refined_object = self.gpat_transformer(
            fragment_features,
            object_features,
            fragment_mask,
        )
        target_logits, target_probabilities = self.segmentation_head(
            refined_fragments,
            refined_object,
            fragment_mask,
        )
        assignment = self._segmentation_guided_assignment(
            refined_fragments,
            refined_object,
            target_probabilities[..., :max_fragments],
            fragment_mask,
        )
        rotations, translations = self.kabsch(
            fragments,
            target_object,
            assignment,
            fragment_mask,
        )
        aligned = apply_fragment_transforms(fragments, rotations, translations)
        aligned = torch.where(fragment_mask[:, :, None, None], aligned, torch.zeros_like(aligned))
        aligned_union = aligned.reshape(batch_size, max_fragments * points_per_fragment, 3)

        return PoseEstimatorOutput(
            rotations=rotations,
            translations=translations,
            aligned_fragments=aligned,
            aligned_union=aligned_union,
            fragment_features=fragment_features,
            object_features=object_features,
            refined_fragment_features=refined_fragments,
            refined_object_features=refined_object,
            matching_logits=target_logits,
            assignment_matrix=assignment,
            target_logits=target_logits,
            target_probabilities=target_probabilities,
        )

    def _segmentation_guided_assignment(
        self,
        fragment_features: torch.Tensor,
        object_features: torch.Tensor,
        target_part_probabilities: torch.Tensor,
        fragment_mask: torch.Tensor,
    ) -> torch.Tensor:
        frag = F.normalize(self.fragment_match_proj(fragment_features), dim=-1)
        obj = F.normalize(self.object_match_proj(object_features), dim=-1)
        temperature = self.log_matching_temperature.exp().clamp(min=1.0e-3, max=1.0)
        logits = torch.einsum("bfnd,bmd->bfnm", frag, obj) / temperature
        logits = logits + target_part_probabilities.transpose(1, 2)[:, :, None, :].clamp(min=1.0e-8).log()
        logits = logits.masked_fill(~fragment_mask[:, :, None, None], -1.0e4)
        assignment = logits.softmax(dim=-1)
        return torch.where(
            fragment_mask[:, :, None, None],
            assignment,
            torch.zeros_like(assignment),
        )

    def _encode_fragments(self, flat_fragments: torch.Tensor) -> FragmentEncoding:
        if self.freeze_encoders:
            with torch.no_grad():
                return self.fragment_encoder(flat_fragments)
        return self.fragment_encoder(flat_fragments)

    def _encode_object(self, target_object: torch.Tensor) -> FragmentEncoding:
        if self.freeze_encoders:
            with torch.no_grad():
                return self.object_encoder(target_object)
        return self.object_encoder(target_object)


class DirectRegressionPoseEstimator(nn.Module):
    """Regress each fragment's SE(3) pose directly, bypassing correspondence + Kabsch.

    Shares the front-end of ``TargetSegmentationPoseEstimator`` (frozen SE(3)-invariant
    encoders + pose-sensitive encoders + fusion + GPAT cross-attention). Instead of a
    per-point assignment, it pools each fragment's cross-attended features into a single
    descriptor and an MLP head emits a 6D-continuity rotation and a placed-centroid. This
    needs only a global orientation/position signal (far better conditioned than
    point-identity on feature-poor fracture surfaces) and removes the differentiable-SVD.
    """

    def __init__(
        self,
        fragment_encoder: nn.Module,
        object_encoder: nn.Module,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        cross_attention_layers: int = 2,
        transformer_layers: int = 2,
        dropout: float = 0.0,
        freeze_encoders: bool = True,
        num_neighbors: int = 16,
        head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.fragment_encoder = fragment_encoder
        self.object_encoder = object_encoder
        self.embedding_dim = embedding_dim
        self.freeze_encoders = freeze_encoders

        # --- front-end shared with TargetSegmentationPoseEstimator ---
        self.fragment_pose_encoder = PoseSensitivePointEncoder(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim,
            num_neighbors=num_neighbors, dropout=dropout,
        )
        self.object_pose_encoder = PoseSensitivePointEncoder(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim,
            num_neighbors=num_neighbors, dropout=dropout,
        )
        self.fragment_point_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.SiLU(),
        )
        self.object_point_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.SiLU(),
        )
        total_layers = max(int(cross_attention_layers), int(transformer_layers), 1)
        self.gpat_transformer = GPATCoarseToFineTransformer(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=total_layers, dropout=dropout,
        )

        # --- regression head ---
        self.frag_query = nn.Parameter(torch.randn(embedding_dim) * 0.02)
        self.frag_score = nn.Linear(embedding_dim, embedding_dim)
        self.head_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3 + 6, head_hidden_dim), nn.LayerNorm(head_hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(head_hidden_dim, head_hidden_dim), nn.LayerNorm(head_hidden_dim), nn.SiLU(),
        )
        self.rot_head = nn.Linear(head_hidden_dim, 6)
        self.centroid_head = nn.Linear(head_hidden_dim, 3)
        # Identity rotation + centroid-at-origin at init (benign, non-exploding start).
        nn.init.zeros_(self.rot_head.weight)
        self.rot_head.bias.data = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        nn.init.zeros_(self.centroid_head.weight)
        nn.init.zeros_(self.centroid_head.bias)

        if freeze_encoders:
            self.freeze_pretrained()

    def freeze_pretrained(self) -> None:
        for module in (self.fragment_encoder, self.object_encoder):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_pretrained(self) -> None:
        for module in (self.fragment_encoder, self.object_encoder):
            for param in module.parameters():
                param.requires_grad = True

    def _encode_fragments(self, flat_fragments: torch.Tensor) -> FragmentEncoding:
        if self.freeze_encoders:
            with torch.no_grad():
                return self.fragment_encoder(flat_fragments)
        return self.fragment_encoder(flat_fragments)

    def _encode_object(self, target_object: torch.Tensor) -> FragmentEncoding:
        if self.freeze_encoders:
            with torch.no_grad():
                return self.object_encoder(target_object)
        return self.object_encoder(target_object)

    def forward(
        self,
        fragments: torch.Tensor,
        target_object: torch.Tensor,
        fragment_mask: torch.Tensor,
        initial_rotations: Optional[torch.Tensor] = None,
        initial_translations: Optional[torch.Tensor] = None,
    ) -> PoseEstimatorOutput:
        del initial_rotations, initial_translations
        if fragments.dim() != 4 or fragments.shape[-1] != 3:
            raise ValueError("fragments must have shape (B, F, N, 3)")
        if target_object.dim() != 3 or target_object.shape[-1] != 3:
            raise ValueError("target_object must have shape (B, M, 3)")

        batch_size, max_fragments, points_per_fragment, _ = fragments.shape
        flat = fragments.reshape(batch_size * max_fragments, points_per_fragment, 3)
        dense_frag_pretrained = self._encode_fragments(flat).point_features.reshape(
            batch_size, max_fragments, points_per_fragment, self.embedding_dim
        )
        dense_obj_pretrained = self._encode_object(target_object).point_features
        dense_frag_pose = self.fragment_pose_encoder(flat).reshape(
            batch_size, max_fragments, points_per_fragment, self.embedding_dim
        )
        dense_obj_pose = self.object_pose_encoder(target_object)

        fragment_features = self.fragment_point_fusion(
            torch.cat([dense_frag_pretrained, dense_frag_pose, fragments], dim=-1)
        )
        object_features = self.object_point_fusion(
            torch.cat([dense_obj_pretrained, dense_obj_pose, target_object], dim=-1)
        )
        fragment_features = torch.where(
            fragment_mask[:, :, None, None], fragment_features, torch.zeros_like(fragment_features)
        )
        refined_fragments, refined_object = self.gpat_transformer(
            fragment_features, object_features, fragment_mask
        )

        # --- pool each fragment to one descriptor ---
        frag_mean = refined_fragments.mean(dim=2)
        scores = torch.einsum("bfnd,d->bfn", self.frag_score(refined_fragments), self.frag_query)
        attn = scores.softmax(dim=2)
        frag_attn = torch.einsum("bfn,bfnd->bfd", attn, refined_fragments)
        obj_ctx = refined_object.mean(dim=1, keepdim=True).expand(batch_size, max_fragments, self.embedding_dim)
        c_in = fragments.mean(dim=2)
        obj_centroid = target_object.mean(dim=1, keepdim=True).expand(batch_size, max_fragments, 3)

        h = self.head_mlp(torch.cat([frag_mean, frag_attn, obj_ctx, c_in, obj_centroid], dim=-1))
        rotations = rotation_6d_to_matrix(self.rot_head(h))
        c_pred = self.centroid_head(h)
        # placed_centroid = R @ c_in + t  ==  c_pred  =>  t = c_pred - R @ c_in
        translations = c_pred - torch.einsum("bfcd,bfd->bfc", rotations, c_in)

        eye = torch.eye(3, device=fragments.device, dtype=fragments.dtype)[None, None]
        rotations = torch.where(fragment_mask[:, :, None, None], rotations, eye)
        translations = torch.where(fragment_mask[:, :, None], translations, torch.zeros_like(translations))
        aligned = apply_fragment_transforms(fragments, rotations, translations)
        aligned = torch.where(fragment_mask[:, :, None, None], aligned, torch.zeros_like(aligned))
        aligned_union = aligned.reshape(batch_size, max_fragments * points_per_fragment, 3)

        placeholder = fragments.new_zeros(batch_size, max_fragments, points_per_fragment, 1)
        return PoseEstimatorOutput(
            rotations=rotations,
            translations=translations,
            aligned_fragments=aligned,
            aligned_union=aligned_union,
            fragment_features=fragment_features,
            object_features=object_features,
            refined_fragment_features=refined_fragments,
            refined_object_features=refined_object,
            matching_logits=placeholder,
            assignment_matrix=placeholder,
        )


def target_segmentation_labels(
    target_object: torch.Tensor,
    target_fragments: torch.Tensor,
    fragment_mask: torch.Tensor,
    outlier_threshold: Optional[float] = None,
) -> torch.Tensor:
    """Project fragment IDs onto a target point cloud by nearest canonical fragment."""
    batch_size, max_fragments, points_per_fragment, _ = target_fragments.shape
    target_points = target_object.shape[1]
    flat_fragments = target_fragments.reshape(batch_size * max_fragments, points_per_fragment, 3)
    expanded_target = target_object[:, None].expand(
        batch_size,
        max_fragments,
        target_points,
        3,
    ).reshape(batch_size * max_fragments, target_points, 3)

    dist = torch.cdist(expanded_target, flat_fragments, p=2).min(dim=2).values
    dist = dist.reshape(batch_size, max_fragments, target_points).transpose(1, 2)
    dist = dist.masked_fill(~fragment_mask[:, None, :], float("inf"))
    nearest_dist, labels = dist.min(dim=-1)
    background = torch.full_like(labels, max_fragments)
    labels = torch.where(torch.isfinite(nearest_dist), labels, background)
    if outlier_threshold is not None and outlier_threshold > 0:
        labels = torch.where(nearest_dist <= float(outlier_threshold), labels, background)
    return labels


class TargetSegmentationPoseLoss(nn.Module):
    """Stage 3 target-segmentation loss with pose/geometry diagnostics."""

    def __init__(
        self,
        lambda_segmentation: float = 2.0,
        lambda_recon: float = 1.0,
        lambda_coverage: float = 1.0,
        lambda_gt: float = 0.0,
        lambda_overlap: float = 0.1,
        lambda_pose: float = 0.0,
        lambda_correspondence: float = 0.0,
        lambda_align: float = 0.0,
        overlap_threshold: float = 0.03,
        segmentation_outlier_threshold: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.lambda_segmentation = lambda_segmentation
        self.lambda_recon = lambda_recon
        self.lambda_coverage = lambda_coverage
        self.lambda_gt = lambda_gt
        self.lambda_overlap = lambda_overlap
        self.lambda_pose = lambda_pose
        self.lambda_correspondence = lambda_correspondence
        self.lambda_align = lambda_align
        self.overlap_threshold = overlap_threshold
        self.segmentation_outlier_threshold = segmentation_outlier_threshold

    def forward(
        self,
        output: PoseEstimatorOutput,
        fragments: torch.Tensor,
        fragment_mask: torch.Tensor,
        reconstructed_object: torch.Tensor,
        ground_truth_object: Optional[torch.Tensor] = None,
        target_fragments: Optional[torch.Tensor] = None,
        gt_rotations: Optional[torch.Tensor] = None,
        gt_translations: Optional[torch.Tensor] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        if output.target_logits is None:
            raise ValueError("TargetSegmentationPoseLoss requires output.target_logits")
        if target_fragments is None:
            if gt_rotations is None or gt_translations is None:
                raise ValueError("target_fragments or ground-truth transforms are required")
            target_fragments = apply_fragment_transforms(fragments, gt_rotations, gt_translations)

        labels = target_segmentation_labels(
            reconstructed_object,
            target_fragments,
            fragment_mask,
            outlier_threshold=self.segmentation_outlier_threshold,
        )
        segmentation_loss = F.cross_entropy(
            output.target_logits.reshape(-1, output.target_logits.shape[-1]),
            labels.reshape(-1),
        )
        pred_labels = output.target_logits.argmax(dim=-1)
        segmentation_accuracy = (pred_labels == labels).to(output.target_logits.dtype).mean()

        batch_size, max_fragments, points_per_fragment, _ = fragments.shape
        union_mask = fragment_mask[:, :, None].expand(batch_size, max_fragments, points_per_fragment)
        union_mask = union_mask.reshape(batch_size, max_fragments * points_per_fragment)
        loss_recon_cd, recon_fit, recon_coverage = masked_chamfer_distance(
            output.aligned_union,
            reconstructed_object,
            union_mask,
        )
        if ground_truth_object is not None:
            loss_gt, _, _ = masked_chamfer_distance(output.aligned_union, ground_truth_object, union_mask)
        else:
            loss_gt = output.aligned_union.sum() * 0.0
        overlap = overlap_loss(output.aligned_fragments, fragment_mask, self.overlap_threshold)

        if gt_rotations is not None and gt_translations is not None:
            pose_loss = pose_supervised_loss(
                output.rotations,
                output.translations,
                gt_rotations,
                gt_translations,
                fragment_mask,
            )
        else:
            zero = output.aligned_union.sum() * 0.0
            pose_loss = {
                "loss": zero,
                "loss_rotation": zero.detach(),
                "loss_translation": zero.detach(),
            }

        correspondence_loss, correspondence_accuracy = dense_correspondence_loss(
            output.assignment_matrix,
            fragments,
            reconstructed_object,
            fragment_mask,
            gt_rotations,
            gt_translations,
        )
        align_loss = aligned_point_loss(
            output.assignment_matrix,
            fragments,
            reconstructed_object,
            fragment_mask,
            gt_rotations,
            gt_translations,
        )

        loss_weights = {
            "lambda_segmentation": self.lambda_segmentation,
            "lambda_matching": self.lambda_segmentation,
            "lambda_recon": self.lambda_recon,
            "lambda_coverage": self.lambda_coverage,
            "lambda_gt": self.lambda_gt,
            "lambda_overlap": self.lambda_overlap,
            "lambda_pose": self.lambda_pose,
            "lambda_correspondence": self.lambda_correspondence,
            "lambda_align": self.lambda_align,
        }
        if weights is not None:
            loss_weights.update(weights)
            if "lambda_matching" in weights and "lambda_segmentation" not in weights:
                loss_weights["lambda_segmentation"] = weights["lambda_matching"]

        total = (
            loss_weights["lambda_segmentation"] * segmentation_loss
            + loss_weights["lambda_recon"] * recon_fit
            + loss_weights["lambda_coverage"] * recon_coverage
            + loss_weights["lambda_gt"] * loss_gt
            + loss_weights["lambda_overlap"] * overlap
            + loss_weights["lambda_pose"] * pose_loss["loss"]
            + loss_weights["lambda_correspondence"] * correspondence_loss
            + loss_weights["lambda_align"] * align_loss
        )
        diagnostics = PoseAssemblyLoss._pose_diagnostics(
            output,
            gt_rotations,
            gt_translations,
            fragment_mask,
        )
        return {
            "loss": total,
            "loss_segmentation": segmentation_loss.detach(),
            "loss_matching": segmentation_loss.detach(),
            "segmentation_accuracy": segmentation_accuracy.detach(),
            "matching_accuracy": segmentation_accuracy.detach(),
            "loss_recon_cd": loss_recon_cd.detach(),
            "loss_recon_fit": recon_fit.detach(),
            "loss_coverage": recon_coverage.detach(),
            "loss_gt_cd": loss_gt.detach(),
            "loss_overlap": overlap.detach(),
            "loss_pose": pose_loss["loss"].detach(),
            "loss_correspondence": correspondence_loss.detach(),
            "correspondence_accuracy": correspondence_accuracy,
            "loss_align": align_loss.detach(),
            "loss_rotation": pose_loss["loss_rotation"],
            "loss_translation": pose_loss["loss_translation"],
            **diagnostics,
        }


def dense_correspondence_loss(
    assignment: torch.Tensor,
    fragments: torch.Tensor,
    target_object: torch.Tensor,
    fragment_mask: torch.Tensor,
    gt_rotations: torch.Tensor,
    gt_translations: torch.Tensor,
    eps: float = 1.0e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Direct supervision for the soft point-correspondence Kabsch consumes.

    The ground-truth correspondence of a fragment point is the target-object point
    nearest to that fragment point once it is placed by the GT pose. We push the
    assignment (already a softmax over the M target points) toward that index with
    a masked NLL. This is what makes the correspondence sharp enough for Kabsch to
    recover *rotation*; without it the matching is only trained indirectly and
    rotation collapses to ~random.

    ``assignment`` has shape (B, F, N, M) of probabilities. Returns (nll_loss,
    accuracy) where accuracy is the fraction of valid fragment points whose argmax
    assignment hits the GT-nearest target point.
    """
    if assignment is None or gt_rotations is None or gt_translations is None:
        zero = fragments.sum() * 0.0
        return zero, zero.detach()

    batch_size, max_fragments, points_per_fragment, num_target = assignment.shape
    placed = apply_fragment_transforms(fragments, gt_rotations, gt_translations)
    flat_placed = placed.reshape(batch_size * max_fragments, points_per_fragment, 3)
    expanded_target = target_object[:, None].expand(
        batch_size, max_fragments, target_object.shape[1], 3
    ).reshape(batch_size * max_fragments, target_object.shape[1], 3)
    with torch.no_grad():
        labels = torch.cdist(flat_placed, expanded_target, p=2).argmin(dim=-1)
    labels = labels.reshape(batch_size, max_fragments, points_per_fragment)

    log_prob = assignment.clamp(min=eps).log()
    nll = -log_prob.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, F, N)
    mask = fragment_mask[:, :, None].expand_as(nll).to(nll.dtype)
    denom = mask.sum().clamp(min=1.0)
    loss = (nll * mask).sum() / denom

    with torch.no_grad():
        correct = (assignment.argmax(dim=-1) == labels).to(nll.dtype)
        accuracy = (correct * mask).sum() / denom
    return loss, accuracy.detach()


def aligned_point_loss(
    assignment: torch.Tensor,
    fragments: torch.Tensor,
    target_object: torch.Tensor,
    fragment_mask: torch.Tensor,
    gt_rotations: torch.Tensor,
    gt_translations: torch.Tensor,
) -> torch.Tensor:
    """Smooth, well-posed geometric supervision for the soft correspondence.

    The soft-corresponded position of a fragment point is the assignment-weighted
    average of object points, ``(assignment @ object)``. Its ground truth is the
    fragment point placed by the GT pose. Unlike the hard nearest-point NLL, this
    does not require identifying the exact object point (ill-posed on smooth
    surfaces where many points are near-equidistant) -- it only requires the
    weighted-average position to be correct, which is exactly what the downstream
    Kabsch consumes. This is the term that actually makes the correspondence
    geometrically meaningful for rotation.
    """
    if assignment is None or gt_rotations is None or gt_translations is None:
        return fragments.sum() * 0.0
    corresponded = torch.einsum("bfnm,bmc->bfnc", assignment, target_object)
    gt_placed = apply_fragment_transforms(fragments, gt_rotations, gt_translations)
    err = ((corresponded - gt_placed) ** 2).sum(dim=-1)  # (B, F, N)
    mask = fragment_mask[:, :, None].expand_as(err).to(err.dtype)
    return (err * mask).sum() / mask.sum().clamp(min=1.0)


def pose_supervised_loss(
    pred_rotations: torch.Tensor,
    pred_translations: torch.Tensor,
    gt_rotations: torch.Tensor,
    gt_translations: torch.Tensor,
    fragment_mask: torch.Tensor,
) -> dict:
    """Active Stage 3 pose objective (weighted by ``loss.stage3.lambda_pose``).

    Backprop uses the smooth chordal rotation term plus a translation L2; the
    geodesic angle is computed detached purely for the human-readable
    ``rotation_error_deg`` metric.
    """
    mask = fragment_mask.to(pred_rotations.dtype)
    denom = mask.sum().clamp(min=1.0)

    # --- differentiable training terms ---
    chordal = rotation_chordal_error(pred_rotations, gt_rotations)
    trans_error = torch.linalg.vector_norm(pred_translations - gt_translations, dim=-1)
    loss_rot = (chordal * mask).sum() / denom
    loss_trans = (trans_error * mask).sum() / denom

    # --- detached reporting term (interpretable degrees) ---
    geodesic_rad = rotation_geodesic_error(pred_rotations.detach(), gt_rotations)
    rotation_error_deg = (geodesic_rad * mask).sum() / denom * 180.0 / torch.pi

    return {
        "loss": loss_rot + loss_trans,
        "loss_rotation": loss_rot.detach(),
        "loss_translation": loss_trans.detach(),
        "rotation_error_deg": rotation_error_deg.detach(),
        "translation_error": loss_trans.detach(),
    }


class DirectRegressionPoseLoss(nn.Module):
    """Loss for ``DirectRegressionPoseEstimator``: chordal+translation pose regression
    plus an auxiliary Chamfer placing the union onto the target object."""

    def __init__(
        self,
        lambda_pose: float = 8.0,
        lambda_recon: float = 1.0,
        lambda_coverage: float = 0.5,
        lambda_gt: float = 0.0,
        lambda_overlap: float = 0.0,
        overlap_threshold: float = 0.03,
    ) -> None:
        super().__init__()
        self.lambda_pose = lambda_pose
        self.lambda_recon = lambda_recon
        self.lambda_coverage = lambda_coverage
        self.lambda_gt = lambda_gt
        self.lambda_overlap = lambda_overlap
        self.overlap_threshold = overlap_threshold

    def forward(
        self,
        output: PoseEstimatorOutput,
        fragments: torch.Tensor,
        fragment_mask: torch.Tensor,
        reconstructed_object: torch.Tensor,
        ground_truth_object: Optional[torch.Tensor] = None,
        target_fragments: Optional[torch.Tensor] = None,
        gt_rotations: Optional[torch.Tensor] = None,
        gt_translations: Optional[torch.Tensor] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        if gt_rotations is not None and gt_translations is not None:
            pose_loss = pose_supervised_loss(
                output.rotations, output.translations, gt_rotations, gt_translations, fragment_mask
            )
        else:
            zero = output.aligned_union.sum() * 0.0
            pose_loss = {
                "loss": zero,
                "loss_rotation": zero.detach(),
                "loss_translation": zero.detach(),
                "rotation_error_deg": zero.detach(),
                "translation_error": zero.detach(),
            }

        batch_size, max_fragments, points_per_fragment, _ = fragments.shape
        union_mask = fragment_mask[:, :, None].expand(
            batch_size, max_fragments, points_per_fragment
        ).reshape(batch_size, max_fragments * points_per_fragment)
        loss_recon_cd, recon_fit, recon_coverage = masked_chamfer_distance(
            output.aligned_union, reconstructed_object, union_mask
        )
        if ground_truth_object is not None:
            loss_gt, _, _ = masked_chamfer_distance(output.aligned_union, ground_truth_object, union_mask)
        else:
            loss_gt = output.aligned_union.sum() * 0.0
        overlap = overlap_loss(output.aligned_fragments, fragment_mask, self.overlap_threshold)

        loss_weights = {
            "lambda_pose": self.lambda_pose,
            "lambda_recon": self.lambda_recon,
            "lambda_coverage": self.lambda_coverage,
            "lambda_gt": self.lambda_gt,
            "lambda_overlap": self.lambda_overlap,
        }
        if weights is not None:
            loss_weights.update({k: v for k, v in weights.items() if k in loss_weights})

        total = (
            loss_weights["lambda_pose"] * pose_loss["loss"]
            + loss_weights["lambda_recon"] * recon_fit
            + loss_weights["lambda_coverage"] * recon_coverage
            + loss_weights["lambda_gt"] * loss_gt
            + loss_weights["lambda_overlap"] * overlap
        )

        zero = (output.aligned_union.sum() * 0.0).detach()
        return {
            "loss": total,
            "loss_pose": pose_loss["loss"].detach(),
            "loss_rotation": pose_loss["loss_rotation"],
            "loss_translation": pose_loss["loss_translation"],
            "rotation_error_deg": pose_loss["rotation_error_deg"],
            "translation_error": pose_loss["translation_error"],
            "loss_recon_cd": loss_recon_cd.detach(),
            "loss_recon_fit": recon_fit.detach(),
            "loss_coverage": recon_coverage.detach(),
            "loss_gt_cd": loss_gt.detach(),
            "loss_overlap": overlap.detach(),
            # placeholders so the shared Stage-3 log line / history render cleanly
            "loss_matching": zero,
            "matching_accuracy": zero,
            "loss_correspondence": zero,
            "correspondence_accuracy": zero,
            "loss_align": zero,
        }


# ============================================================================
# GeoTransformer Stage-3 matcher (geometric-attention correspondence + Kabsch)
# ----------------------------------------------------------------------------
# Coarse-path v1: rotation-invariant geometric self-attention over superpoints
# + coarse Gaussian-correlation matching -> soft superpoint assignment ->
# DifferentiableKabsch. (Fine patch-level OT refinement is a planned follow-up.)
# ============================================================================


def _sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of a scalar tensor -> (..., dim). No parameters."""
    half = dim // 2
    freq = 1.0 / (10000.0 ** (torch.arange(half, device=values.device, dtype=torch.float32) * 2.0 / dim))
    ang = values.to(torch.float32).unsqueeze(-1) * freq
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def _gather_rows(coords: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """coords (B,n,3), idx (B,m,k) -> (B,m,k,3). Small tensors (superpoint scale)."""
    b, n, c = coords.shape
    m, k = idx.shape[1], idx.shape[2]
    expanded = coords[:, None, :, :].expand(b, m, n, c)
    return torch.gather(expanded, 2, idx.unsqueeze(-1).expand(b, m, k, c))


class GeometricStructureEmbedding(nn.Module):
    """GeoTransformer relative-position map: pairwise distances + triplet angles."""

    def __init__(self, hidden_dim: int, sigma_d: float, sigma_a: float, angle_k: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.angle_k = angle_k
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        b, n, _ = coords.shape
        with torch.no_grad():
            dist = torch.cdist(coords, coords)                       # (B,n,n)
            d_emb = _sinusoidal_embedding(dist / self.sigma_d, self.hidden_dim)
            k = min(self.angle_k + 1, n)
            knn_idx = dist.topk(k=k, dim=-1, largest=False).indices[..., 1:]  # (B,n,k-1)
            kk = knn_idx.shape[-1]
            if kk == 0:
                a_emb = torch.zeros_like(d_emb)
            else:
                knn_pts = _gather_rows(coords, knn_idx)              # (B,n,kk,3)
                v_ref = knn_pts - coords.unsqueeze(2)                # (B,n,kk,3)
                v_ij = coords.unsqueeze(1) - coords.unsqueeze(2)     # (B,n,n,3) = P_j - P_i
                vij = v_ij.unsqueeze(3).expand(b, n, n, kk, 3)
                vrf = v_ref.unsqueeze(2).expand(b, n, n, kk, 3)
                cross = torch.linalg.cross(vij, vrf, dim=-1)
                angle = torch.atan2(torch.linalg.vector_norm(cross, dim=-1), (vij * vrf).sum(-1))
                a_ind = angle * (180.0 / (self.sigma_a * torch.pi))
                a_emb = _sinusoidal_embedding(a_ind, self.hidden_dim).max(dim=3).values  # (B,n,n,H)
        return self.proj_d(d_emb) + self.proj_a(a_emb)


class GeometricSelfAttention(nn.Module):
    """Self-attention with a relative-position bias (GeoTransformer RPE form)."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.h = num_heads
        self.d = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.p = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rpe: torch.Tensor) -> torch.Tensor:
        b, n, hd = x.shape
        q = self.q(x).view(b, n, self.h, self.d).transpose(1, 2)          # (B,h,n,d)
        k = self.k(x).view(b, n, self.h, self.d).transpose(1, 2)
        v = self.v(x).view(b, n, self.h, self.d).transpose(1, 2)
        p = self.p(rpe).view(b, n, n, self.h, self.d).permute(0, 3, 1, 2, 4)  # (B,h,n,n,d)
        attn_e = torch.einsum("bhic,bhjc->bhij", q, k)
        attn_p = torch.einsum("bhic,bhijc->bhij", q, p)
        attn = ((attn_e + attn_p) / (self.d ** 0.5)).softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhij,bhjc->bhic", attn, v).transpose(1, 2).reshape(b, n, hd)
        return self.out(out)


def _ffn(hidden_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim * 2, hidden_dim), nn.Dropout(dropout),
    )


class GeometricTransformerBlock(nn.Module):
    """One block: geometric self-attn (frag & obj) + bidirectional cross-attn."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_frag = GeometricSelfAttention(hidden_dim, num_heads, dropout)
        self.self_obj = GeometricSelfAttention(hidden_dim, num_heads, dropout)
        self.cross_f2o = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_o2f = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(6)])
        self.ffn_frag = _ffn(hidden_dim, dropout)
        self.ffn_obj = _ffn(hidden_dim, dropout)
        self.ffn_norm_frag = nn.LayerNorm(hidden_dim)
        self.ffn_norm_obj = nn.LayerNorm(hidden_dim)

    def forward(self, frag, obj, rpe_frag, rpe_obj, batch, num_frag, key_mask):
        # frag (B*F, Sf, H), obj (B, So, H), rpe_frag (B*F,Sf,Sf,H), rpe_obj (B,So,So,H)
        so = obj.shape[1]
        frag = frag + self.self_frag(self.norms[0](frag), rpe_frag)
        obj = obj + self.self_obj(self.norms[1](obj), rpe_obj)

        obj_exp = obj.unsqueeze(1).expand(batch, num_frag, so, obj.shape[-1]).reshape(batch * num_frag, so, obj.shape[-1])
        f_q = self.norms[2](frag)
        frag = frag + self.cross_f2o(f_q, obj_exp, obj_exp, need_weights=False)[0]

        frag_keys = frag.reshape(batch, num_frag * frag.shape[1], frag.shape[-1])
        o_q = self.norms[3](obj)
        obj = obj + self.cross_o2f(o_q, frag_keys, frag_keys, key_padding_mask=~key_mask, need_weights=False)[0]

        frag = frag + self.ffn_frag(self.ffn_norm_frag(frag))
        obj = obj + self.ffn_obj(self.ffn_norm_obj(obj))
        return frag, obj


class SuperpointDescriptor(nn.Module):
    """Invariant superpoint descriptor: projected frozen features + trainable local invariants."""

    def __init__(self, encoder_dim: int, hidden_dim: int, local_k: int = 8) -> None:
        super().__init__()
        self.local_k = local_k
        self.proj_frozen = nn.Linear(encoder_dim, hidden_dim)
        self.proj_local = nn.Linear(7, hidden_dim)  # 4 kNN dist stats + 3 covariance eigvals

    def local_invariants(self, coords: torch.Tensor) -> torch.Tensor:
        b, n, _ = coords.shape
        with torch.no_grad():
            dist = torch.cdist(coords, coords)
            k = min(self.local_k + 1, n)
            knn_d, knn_i = dist.topk(k=k, dim=-1, largest=False)
            nd = knn_d[..., 1:]                                    # (B,n,k-1)
            if nd.shape[-1] == 0:
                nd = torch.zeros(b, n, 1, device=coords.device)
            stats = torch.stack(
                [nd.min(-1).values, nd.mean(-1), nd.std(-1, unbiased=False), nd.max(-1).values], dim=-1
            )                                                      # (B,n,4)
            neigh = _gather_rows(coords, knn_i[..., 1:] if knn_i.shape[-1] > 1 else knn_i)  # (B,n,kk,3)
            centered = neigh - neigh.mean(2, keepdim=True)
            cov = centered.transpose(-1, -2) @ centered / max(centered.shape[2], 1)
            eig = torch.linalg.eigvalsh(cov)                       # (B,n,3) ascending, >=0
            eig = eig / (eig.sum(-1, keepdim=True) + 1e-8)
            feats = torch.cat([stats, eig], dim=-1)                # (B,n,7)
        return feats

    def forward(self, frozen_feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return self.proj_frozen(frozen_feats) + self.proj_local(self.local_invariants(coords))


class GeoTransformerPoseEstimator(nn.Module):
    """Geometric-attention superpoint matcher -> weighted Kabsch (per fragment)."""

    def __init__(
        self,
        fragment_encoder: nn.Module,
        object_encoder: nn.Module,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_blocks: int = 3,
        num_fragment_superpoints: int = 128,
        num_object_superpoints: int = 256,
        sigma_d: float = 0.15,
        sigma_a: float = 15.0,
        angle_k: int = 3,
        dropout: float = 0.0,
        freeze_encoders: bool = True,
    ) -> None:
        super().__init__()
        self.fragment_encoder = fragment_encoder
        self.object_encoder = object_encoder
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.freeze_encoders = freeze_encoders
        self.num_fragment_superpoints = num_fragment_superpoints
        self.num_object_superpoints = num_object_superpoints

        self.descriptor = SuperpointDescriptor(embedding_dim, hidden_dim)
        self.geo_embed = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k)
        self.blocks = nn.ModuleList(
            [GeometricTransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.frag_out = nn.Linear(hidden_dim, hidden_dim)
        self.obj_out = nn.Linear(hidden_dim, hidden_dim)
        self.kabsch = DifferentiableKabsch()
        if freeze_encoders:
            self.freeze_pretrained()

    def freeze_pretrained(self) -> None:
        for module in (self.fragment_encoder, self.object_encoder):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_pretrained(self) -> None:
        for module in (self.fragment_encoder, self.object_encoder):
            for param in module.parameters():
                param.requires_grad = True

    def _encode(self, encoder: nn.Module, points: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoders:
            with torch.no_grad():
                return encoder(points).point_features
        return encoder(points).point_features

    def _superpoints(self, points: torch.Tensor, feats: torch.Tensor, num_sp: int):
        n = points.shape[1]
        sp = min(num_sp, n)
        idx = _batched_fps_indices(points.detach(), sp)            # (B, sp)
        sp_xyz = _gather_points(points, idx)                       # (B, sp, 3)
        sp_feat = _gather_points(feats, idx)                       # (B, sp, C)
        return sp_xyz, sp_feat

    def forward(
        self,
        fragments: torch.Tensor,
        target_object: torch.Tensor,
        fragment_mask: torch.Tensor,
        initial_rotations: Optional[torch.Tensor] = None,
        initial_translations: Optional[torch.Tensor] = None,
    ) -> PoseEstimatorOutput:
        del initial_rotations, initial_translations
        if fragments.dim() != 4 or fragments.shape[-1] != 3:
            raise ValueError("fragments must have shape (B, F, N, 3)")
        if target_object.dim() != 3 or target_object.shape[-1] != 3:
            raise ValueError("target_object must have shape (B, M, 3)")

        b, num_frag, n_pts, _ = fragments.shape
        flat = fragments.reshape(b * num_frag, n_pts, 3)
        frag_pf = self._encode(self.fragment_encoder, flat)                    # (B*F, N, C)
        sp_frag_xyz, sp_frag_feat = self._superpoints(flat, frag_pf, self.num_fragment_superpoints)
        obj_pf = self._encode(self.object_encoder, target_object)              # (B, M, C)
        sp_obj_xyz, sp_obj_feat = self._superpoints(target_object, obj_pf, self.num_object_superpoints)

        desc_frag = self.descriptor(sp_frag_feat, sp_frag_xyz)                 # (B*F, Sf, H)
        desc_obj = self.descriptor(sp_obj_feat, sp_obj_xyz)                    # (B, So, H)

        rpe_frag = self.geo_embed(sp_frag_xyz)                                 # (B*F, Sf, Sf, H)
        rpe_obj = self.geo_embed(sp_obj_xyz)                                   # (B, So, So, H)

        sf = sp_frag_xyz.shape[1]
        key_mask = fragment_mask[:, :, None].expand(b, num_frag, sf).reshape(b, num_frag * sf)
        frag = desc_frag
        obj = desc_obj
        for block in self.blocks:
            frag, obj = block(frag, obj, rpe_frag, rpe_obj, b, num_frag, key_mask)
        frag = self.frag_out(frag).reshape(b, num_frag, sf, self.hidden_dim)
        obj = self.obj_out(obj)                                                # (B, So, H)

        # Coarse Gaussian-correlation matching + dual normalization -> soft assignment.
        fn = F.normalize(frag, dim=-1)
        on = F.normalize(obj, dim=-1)
        sim = torch.einsum("bfsh,boh->bfso", fn, on)                           # cosine in [-1,1]
        score = torch.exp(2.0 * (sim - 1.0))                                   # ~exp(-||.||^2)
        row = score / score.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        col = score / score.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        assignment = row * col                                                 # (B,F,Sf,So)
        assignment = torch.where(fragment_mask[:, :, None, None], assignment, torch.zeros_like(assignment))

        sp_frag_xyz_bf = sp_frag_xyz.reshape(b, num_frag, sf, 3)
        rotations, translations = self.kabsch(sp_frag_xyz_bf, sp_obj_xyz, assignment, fragment_mask)

        eye = torch.eye(3, device=fragments.device, dtype=fragments.dtype)[None, None]
        rotations = torch.where(fragment_mask[:, :, None, None], rotations, eye)
        translations = torch.where(fragment_mask[:, :, None], translations, torch.zeros_like(translations))
        aligned = apply_fragment_transforms(fragments, rotations, translations)
        aligned = torch.where(fragment_mask[:, :, None, None], aligned, torch.zeros_like(aligned))
        aligned_union = aligned.reshape(b, num_frag * n_pts, 3)

        return PoseEstimatorOutput(
            rotations=rotations,
            translations=translations,
            aligned_fragments=aligned,
            aligned_union=aligned_union,
            fragment_features=frag,
            object_features=obj,
            refined_fragment_features=frag,
            refined_object_features=obj,
            matching_logits=assignment,
            assignment_matrix=assignment,
            fragment_superpoints=sp_frag_xyz_bf,
            object_superpoints=sp_obj_xyz,
        )


def fragment_extent(fragments: torch.Tensor) -> torch.Tensor:
    """Rotation/translation-invariant per-fragment size proxy (radius). (B,F)."""
    centroid = fragments.mean(dim=2, keepdim=True)
    return torch.linalg.vector_norm(fragments - centroid, dim=-1).max(dim=-1).values


class GeoTransformerPoseLoss(nn.Module):
    """Coarse circle loss + aligned-point + chordal + chamfer, with soft size down-weight."""

    def __init__(
        self,
        lambda_matching: float = 2.0,
        lambda_align: float = 5.0,
        lambda_pose: float = 0.5,
        lambda_recon: float = 1.0,
        lambda_coverage: float = 1.0,
        lambda_gt: float = 0.0,
        lambda_overlap: float = 0.1,
        matching_radius: float = 0.04,
        pos_margin: float = 0.1,
        neg_margin: float = 1.4,
        log_scale: float = 24.0,
        overlap_threshold: float = 0.03,
        min_fragment_extent: float = 0.0,
        fragment_size_softness: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_matching = lambda_matching
        self.lambda_align = lambda_align
        self.lambda_pose = lambda_pose
        self.lambda_recon = lambda_recon
        self.lambda_coverage = lambda_coverage
        self.lambda_gt = lambda_gt
        self.lambda_overlap = lambda_overlap
        self.matching_radius = matching_radius
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.log_scale = log_scale
        self.overlap_threshold = overlap_threshold
        self.min_fragment_extent = min_fragment_extent
        self.fragment_size_softness = fragment_size_softness

    def _frag_weight(self, fragments: torch.Tensor, fragment_mask: torch.Tensor) -> torch.Tensor:
        mask = fragment_mask.to(fragments.dtype)
        if self.min_fragment_extent <= 0.0:
            return mask
        extent = fragment_extent(fragments)
        if self.fragment_size_softness > 0.0:
            gate = torch.sigmoid((extent - self.min_fragment_extent) / self.fragment_size_softness)
        else:
            gate = (extent >= self.min_fragment_extent).to(fragments.dtype)
        return mask * gate

    def _circle_loss(self, output, sp_frag_canon, sp_obj, fragment_mask, frag_weight):
        feat_f = F.normalize(output.fragment_features, dim=-1)      # (B,F,Sf,H)
        feat_o = F.normalize(output.object_features, dim=-1)        # (B,So,H)
        feat_dist = torch.sqrt((2.0 - 2.0 * torch.einsum("bfsh,boh->bfso", feat_f, feat_o)).clamp(min=1e-8))
        with torch.no_grad():
            coord_dist = torch.cdist(sp_frag_canon, sp_obj[:, None].expand(-1, sp_frag_canon.shape[1], -1, -1))
            positive = coord_dist < self.matching_radius                       # (B,F,Sf,So)
        pos_term = torch.where(positive, feat_dist, torch.zeros_like(feat_dist))
        neg_term = torch.where(~positive, feat_dist, torch.full_like(feat_dist, 1e4))
        has_pos = positive.any(dim=-1)                                         # (B,F,Sf)
        # row-wise circle-style loss over object superpoints
        w_pos = F.relu(pos_term - self.pos_margin).detach()
        w_neg = F.relu(self.neg_margin - neg_term).detach()
        l_pos = torch.logsumexp(self.log_scale * (pos_term - self.pos_margin) * w_pos + (~positive) * (-1e4), dim=-1)
        l_neg = torch.logsumexp(self.log_scale * (self.neg_margin - neg_term) * w_neg + positive * (-1e4), dim=-1)
        row_loss = F.softplus(l_pos + l_neg) / self.log_scale                   # (B,F,Sf)
        row_w = has_pos.to(feat_dist.dtype) * frag_weight[:, :, None]
        return (row_loss * row_w).sum() / row_w.sum().clamp(min=1.0)

    def forward(
        self,
        output: PoseEstimatorOutput,
        fragments: torch.Tensor,
        fragment_mask: torch.Tensor,
        reconstructed_object: torch.Tensor,
        ground_truth_object: Optional[torch.Tensor] = None,
        target_fragments: Optional[torch.Tensor] = None,
        gt_rotations: Optional[torch.Tensor] = None,
        gt_translations: Optional[torch.Tensor] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        frag_weight = self._frag_weight(fragments, fragment_mask)

        # Superpoint tensors carried on the output; place fragment superpoints by GT pose.
        sp_frag = output.fragment_superpoints                       # (B,F,Sf,3)
        sp_obj = output.object_superpoints                          # (B,So,3)
        if gt_rotations is not None and gt_translations is not None:
            sp_frag_canon = apply_fragment_transforms(sp_frag, gt_rotations, gt_translations)
            coarse = self._circle_loss(output, sp_frag_canon, sp_obj, fragment_mask, frag_weight)
            # Row-normalize the (dual-softmax, sub-stochastic) assignment so the
            # soft-corresponded position is a proper convex combination of object
            # points; without this the weighted mean shrinks to the origin and the
            # aligned-point loss is biased (stuck) rather than a geometric driver.
            assign_norm = output.assignment_matrix / output.assignment_matrix.sum(
                dim=-1, keepdim=True
            ).clamp(min=1e-8)
            align = aligned_point_loss(
                assign_norm, sp_frag, sp_obj, fragment_mask, gt_rotations, gt_translations
            )
            pose_loss = pose_supervised_loss(
                output.rotations, output.translations, gt_rotations, gt_translations, fragment_mask
            )
        else:
            zero = output.aligned_union.sum() * 0.0
            coarse = zero
            align = zero
            pose_loss = {"loss": zero, "loss_rotation": zero.detach(),
                         "loss_translation": zero.detach(), "rotation_error_deg": zero.detach(),
                         "translation_error": zero.detach()}

        b, num_frag, n_pts, _ = fragments.shape
        union_mask = fragment_mask[:, :, None].expand(b, num_frag, n_pts).reshape(b, num_frag * n_pts)
        loss_recon_cd, recon_fit, recon_coverage = masked_chamfer_distance(
            output.aligned_union, reconstructed_object, union_mask
        )
        if ground_truth_object is not None:
            loss_gt, _, _ = masked_chamfer_distance(output.aligned_union, ground_truth_object, union_mask)
        else:
            loss_gt = output.aligned_union.sum() * 0.0
        overlap = overlap_loss(output.aligned_fragments, fragment_mask, self.overlap_threshold)

        loss_weights = {
            "lambda_matching": self.lambda_matching,
            "lambda_align": self.lambda_align,
            "lambda_pose": self.lambda_pose,
            "lambda_recon": self.lambda_recon,
            "lambda_coverage": self.lambda_coverage,
            "lambda_gt": self.lambda_gt,
            "lambda_overlap": self.lambda_overlap,
        }
        if weights is not None:
            loss_weights.update({k: v for k, v in weights.items() if k in loss_weights})

        total = (
            loss_weights["lambda_matching"] * coarse
            + loss_weights["lambda_align"] * align
            + loss_weights["lambda_pose"] * pose_loss["loss"]
            + loss_weights["lambda_recon"] * recon_fit
            + loss_weights["lambda_coverage"] * recon_coverage
            + loss_weights["lambda_gt"] * loss_gt
            + loss_weights["lambda_overlap"] * overlap
        )
        zero = (output.aligned_union.sum() * 0.0).detach()
        return {
            "loss": total,
            "loss_matching": coarse.detach() if torch.is_tensor(coarse) else zero,
            "loss_recon_cd": loss_recon_cd.detach(),
            "loss_align": align.detach() if torch.is_tensor(align) else zero,
            "loss_pose": pose_loss["loss"].detach(),
            "loss_recon_fit": recon_fit.detach(),
            "loss_coverage": recon_coverage.detach(),
            "loss_gt_cd": loss_gt.detach(),
            "loss_overlap": overlap.detach(),
            "loss_rotation": pose_loss["loss_rotation"],
            "loss_translation": pose_loss["loss_translation"],
            "rotation_error_deg": pose_loss["rotation_error_deg"],
            "translation_error": pose_loss["translation_error"],
            "frac_fragments_weighted": (frag_weight.sum() / fragment_mask.to(frag_weight.dtype).sum().clamp(min=1.0)).detach(),
            # correspondence-metric placeholders for the shared log line
            "matching_accuracy": zero,
            "loss_correspondence": zero,
            "correspondence_accuracy": zero,
        }


def build_pose_estimator(cfg: dict, freeze_encoders: Optional[bool] = None) -> nn.Module:
    model_cfg = cfg.get("model", {})
    enc_cfg = model_cfg.get("encoder", {})
    pose_cfg = model_cfg.get("pose", {})
    if freeze_encoders is None:
        freeze_encoders = pose_cfg.get("freeze_encoders", True)

    fragment_encoder = build_token_encoder(enc_cfg)
    object_encoder = build_token_encoder(enc_cfg)
    architecture = pose_cfg.get("architecture", "target_segmentation")
    if architecture == "geotransformer":
        return GeoTransformerPoseEstimator(
            fragment_encoder=fragment_encoder,
            object_encoder=object_encoder,
            embedding_dim=enc_cfg.get("embedding_dim", 128),
            hidden_dim=pose_cfg.get("hidden_dim", 256),
            num_heads=pose_cfg.get("num_heads", model_cfg.get("assembly", {}).get("transformer_heads", 4)),
            num_blocks=pose_cfg.get("num_blocks", 3),
            num_fragment_superpoints=pose_cfg.get("num_fragment_superpoints", 128),
            num_object_superpoints=pose_cfg.get("num_object_superpoints", 256),
            sigma_d=pose_cfg.get("sigma_d", 0.15),
            sigma_a=pose_cfg.get("sigma_a", 15.0),
            angle_k=pose_cfg.get("angle_k", 3),
            dropout=pose_cfg.get("dropout", enc_cfg.get("dropout", 0.0)),
            freeze_encoders=freeze_encoders,
        )
    if architecture == "direct_regression":
        return DirectRegressionPoseEstimator(
            fragment_encoder=fragment_encoder,
            object_encoder=object_encoder,
            embedding_dim=enc_cfg.get("embedding_dim", 128),
            hidden_dim=pose_cfg.get("hidden_dim", 256),
            num_heads=pose_cfg.get("num_heads", model_cfg.get("assembly", {}).get("transformer_heads", 4)),
            cross_attention_layers=pose_cfg.get("cross_attention_layers", 2),
            transformer_layers=pose_cfg.get("transformer_layers", 2),
            dropout=pose_cfg.get("dropout", enc_cfg.get("dropout", 0.0)),
            freeze_encoders=freeze_encoders,
            num_neighbors=pose_cfg.get("num_neighbors", enc_cfg.get("num_neighbors", 16)),
            head_hidden_dim=pose_cfg.get("head_hidden_dim", pose_cfg.get("hidden_dim", 256)),
        )
    if architecture == "target_segmentation":
        return TargetSegmentationPoseEstimator(
            fragment_encoder=fragment_encoder,
            object_encoder=object_encoder,
            embedding_dim=enc_cfg.get("embedding_dim", 128),
            hidden_dim=pose_cfg.get("hidden_dim", 256),
            num_heads=pose_cfg.get("num_heads", model_cfg.get("assembly", {}).get("transformer_heads", 4)),
            cross_attention_layers=pose_cfg.get("cross_attention_layers", 2),
            transformer_layers=pose_cfg.get("transformer_layers", 2),
            dropout=pose_cfg.get("dropout", enc_cfg.get("dropout", 0.0)),
            freeze_encoders=freeze_encoders,
            num_neighbors=pose_cfg.get("num_neighbors", enc_cfg.get("num_neighbors", 16)),
            matching_temperature=pose_cfg.get("matching_temperature", 0.07),
        )
    if architecture != "dense_matching":
        raise ValueError(f"Unknown Stage 3 pose architecture: {architecture}")
    return FragmentObjectPoseEstimator(
        fragment_encoder=fragment_encoder,
        object_encoder=object_encoder,
        embedding_dim=enc_cfg.get("embedding_dim", 128),
        hidden_dim=pose_cfg.get("hidden_dim", 256),
        num_heads=pose_cfg.get("num_heads", model_cfg.get("assembly", {}).get("transformer_heads", 4)),
        cross_attention_layers=pose_cfg.get("cross_attention_layers", 2),
        transformer_layers=pose_cfg.get("transformer_layers", 2),
        dropout=pose_cfg.get("dropout", enc_cfg.get("dropout", 0.0)),
        freeze_encoders=freeze_encoders,
        num_neighbors=pose_cfg.get("num_neighbors", enc_cfg.get("num_neighbors", 16)),
        sinkhorn_iterations=pose_cfg.get("sinkhorn_iterations", 8),
        matching_temperature=pose_cfg.get("matching_temperature", 0.07),
        transport=pose_cfg.get("transport", "sinkhorn"),
    )


def build_pose_loss(cfg: dict) -> nn.Module:
    loss_cfg = cfg.get("loss", {}).get("stage3", {})
    pose_cfg = cfg.get("model", {}).get("pose", {})
    architecture = pose_cfg.get("architecture", "target_segmentation")
    if architecture == "geotransformer":
        return GeoTransformerPoseLoss(
            lambda_matching=loss_cfg.get("lambda_matching", 2.0),
            lambda_align=loss_cfg.get("lambda_align", 5.0),
            lambda_pose=loss_cfg.get("lambda_pose", 0.5),
            lambda_recon=loss_cfg.get("lambda_recon", 1.0),
            lambda_coverage=loss_cfg.get("lambda_coverage", 1.0),
            lambda_gt=loss_cfg.get("lambda_gt", 0.0),
            lambda_overlap=loss_cfg.get("lambda_overlap", 0.1),
            matching_radius=loss_cfg.get("matching_radius", 0.04),
            pos_margin=loss_cfg.get("circle_pos_margin", 0.1),
            neg_margin=loss_cfg.get("circle_neg_margin", 1.4),
            log_scale=loss_cfg.get("circle_log_scale", 24.0),
            overlap_threshold=loss_cfg.get("overlap_threshold", 0.03),
            min_fragment_extent=loss_cfg.get("min_fragment_extent", 0.0),
            fragment_size_softness=loss_cfg.get("fragment_size_softness", 0.0),
        )
    if architecture == "direct_regression":
        return DirectRegressionPoseLoss(
            lambda_pose=loss_cfg.get("lambda_pose", 8.0),
            lambda_recon=loss_cfg.get("lambda_recon", 1.0),
            lambda_coverage=loss_cfg.get("lambda_coverage", 0.5),
            lambda_gt=loss_cfg.get("lambda_gt", 0.0),
            lambda_overlap=loss_cfg.get("lambda_overlap", 0.0),
            overlap_threshold=loss_cfg.get("overlap_threshold", 0.03),
        )
    if architecture == "target_segmentation":
        return TargetSegmentationPoseLoss(
            lambda_segmentation=loss_cfg.get(
                "lambda_segmentation",
                loss_cfg.get("lambda_matching", 2.0),
            ),
            lambda_recon=loss_cfg.get("lambda_recon", 1.0),
            lambda_coverage=loss_cfg.get("lambda_coverage", 1.0),
            lambda_gt=loss_cfg.get("lambda_gt", 0.0),
            lambda_overlap=loss_cfg.get("lambda_overlap", 0.1),
            lambda_pose=loss_cfg.get("lambda_pose", 0.0),
            lambda_correspondence=loss_cfg.get("lambda_correspondence", 0.0),
            lambda_align=loss_cfg.get("lambda_align", 0.0),
            overlap_threshold=loss_cfg.get("overlap_threshold", 0.03),
            segmentation_outlier_threshold=loss_cfg.get("segmentation_outlier_threshold"),
        )
    return PoseAssemblyLoss(
        lambda_matching=loss_cfg.get("lambda_matching", 2.0),
        lambda_recon=loss_cfg.get("lambda_recon", 1.0),
        lambda_coverage=loss_cfg.get("lambda_coverage", 1.0),
        lambda_gt=loss_cfg.get("lambda_gt", 0.0),
        lambda_overlap=loss_cfg.get("lambda_overlap", 0.1),
        lambda_pose=loss_cfg.get("lambda_pose", 0.0),
        overlap_threshold=loss_cfg.get("overlap_threshold", 0.03),
        matching_temperature=loss_cfg.get("matching_temperature", 1.0),
    )
