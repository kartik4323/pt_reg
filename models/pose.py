"""GPAT-style dense matching and rigid pose extraction for Stage 3 assembly."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.token_encoder import FragmentEncoding, build_token_encoder


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


def _knn_indices(points: torch.Tensor, k: int) -> torch.Tensor:
    """Return k nearest-neighbor indices per point, excluding self when possible."""
    num_points = points.shape[1]
    if num_points <= 1:
        return torch.zeros(points.shape[0], num_points, 1, dtype=torch.long, device=points.device)
    k_eff = min(k + 1, num_points)
    dist = torch.cdist(points, points, p=2)
    return dist.topk(k=k_eff, dim=-1, largest=False).indices[..., 1:]


def _gather_neighbors(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_size, num_points, dim = values.shape
    k = indices.shape[-1]
    expanded = values[:, None].expand(batch_size, num_points, num_points, dim)
    gather_index = indices.unsqueeze(-1).expand(batch_size, num_points, k, dim)
    return expanded.gather(dim=2, index=gather_index)


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

        loss_weights = {
            "lambda_segmentation": self.lambda_segmentation,
            "lambda_matching": self.lambda_segmentation,
            "lambda_recon": self.lambda_recon,
            "lambda_coverage": self.lambda_coverage,
            "lambda_gt": self.lambda_gt,
            "lambda_overlap": self.lambda_overlap,
            "lambda_pose": self.lambda_pose,
            "lambda_correspondence": self.lambda_correspondence,
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


def build_pose_estimator(cfg: dict, freeze_encoders: Optional[bool] = None) -> nn.Module:
    model_cfg = cfg.get("model", {})
    enc_cfg = model_cfg.get("encoder", {})
    pose_cfg = model_cfg.get("pose", {})
    if freeze_encoders is None:
        freeze_encoders = pose_cfg.get("freeze_encoders", True)

    fragment_encoder = build_token_encoder(enc_cfg)
    object_encoder = build_token_encoder(enc_cfg)
    architecture = pose_cfg.get("architecture", "target_segmentation")
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
    if pose_cfg.get("architecture", "target_segmentation") == "target_segmentation":
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
