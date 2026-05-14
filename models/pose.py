"""Post-hoc SE(3) alignment utilities for reconstructed point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_transform(points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Apply x' = R x + t to row-vector point tensors."""
    return points @ rotation.transpose(-1, -2) + translation.unsqueeze(-2)


def kabsch_align(source: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Best-fit rigid transform from source to target.

    Both tensors have shape (B, N, 3), and correspondences are assumed to be
    ordered. Returns rotation and translation using row-vector convention:
    aligned = source @ R.T + t.
    """
    src_centroid = source.mean(dim=1, keepdim=True)
    tgt_centroid = target.mean(dim=1, keepdim=True)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid
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
    return rotation, translation


@torch.no_grad()
def differentiable_icp_initialization(
    fragments: torch.Tensor,
    reconstruction: torch.Tensor,
    fragment_mask: torch.Tensor,
    iterations: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ICP-style SE(3) estimation.

    The function uses differentiable PyTorch ops internally, but is decorated
    with no_grad because it is intended as a post-hoc estimator for this
    pipeline. Remove no_grad if you need gradients through ICP.
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

    rotations = torch.where(fragment_mask[:, :, None, None], rotations, torch.eye(3, device=fragments.device)[None, None])
    translations = torch.where(fragment_mask[:, :, None], translations, torch.zeros_like(translations))
    return rotations, translations, aligned


def rotation_6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


@dataclass
class PoseHeadOutput:
    rotations: torch.Tensor
    translations: torch.Tensor


class LearnedPoseHead(nn.Module):
    """
    Lightweight pose refinement head initialized from ICP outputs.

    It predicts a residual SE(3) transform per fragment from fragment and
    reconstruction summaries. The head can be trained after Stage 2 with the
    reconstruction model frozen.
    """

    def __init__(self, feature_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.fragment_summary = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.reconstruction_summary = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 9),
        )

    def forward(
        self,
        fragments: torch.Tensor,
        reconstruction: torch.Tensor,
        fragment_mask: torch.Tensor,
    ) -> PoseHeadOutput:
        batch_size, max_fragments, _, _ = fragments.shape
        frag_summary = self.fragment_summary(fragments.mean(dim=2))
        recon_summary = self.reconstruction_summary(reconstruction.mean(dim=1))
        recon_summary = recon_summary[:, None, :].expand(batch_size, max_fragments, -1)
        pred = self.head(torch.cat([frag_summary, recon_summary], dim=-1))
        rotations = rotation_6d_to_matrix(pred[..., :6])
        translations = pred[..., 6:9]
        rotations = torch.where(
            fragment_mask[:, :, None, None],
            rotations,
            torch.eye(3, device=fragments.device, dtype=fragments.dtype)[None, None],
        )
        translations = torch.where(fragment_mask[:, :, None], translations, torch.zeros_like(translations))
        return PoseHeadOutput(rotations=rotations, translations=translations)


def pose_supervised_loss(
    pred_rotations: torch.Tensor,
    pred_translations: torch.Tensor,
    gt_rotations: torch.Tensor,
    gt_translations: torch.Tensor,
    fragment_mask: torch.Tensor,
) -> dict:
    rot_delta = pred_rotations.transpose(-1, -2) @ gt_rotations
    trace = rot_delta.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
    rot_error = torch.acos(cos_theta)
    trans_error = torch.linalg.vector_norm(pred_translations - gt_translations, dim=-1)
    mask = fragment_mask.to(rot_error.dtype)
    loss_rot = (rot_error * mask).sum() / mask.sum().clamp(min=1.0)
    loss_trans = (trans_error * mask).sum() / mask.sum().clamp(min=1.0)
    return {
        "loss": loss_rot + loss_trans,
        "loss_rotation": loss_rot.detach(),
        "loss_translation": loss_trans.detach(),
    }
