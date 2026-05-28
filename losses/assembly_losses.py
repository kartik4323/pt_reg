"""Losses and metrics for Stage 2 assembly reconstruction."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def chamfer_terms(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dist = torch.cdist(pred, target, p=2)
    pred_to_target = dist.min(dim=2).values
    target_to_pred = dist.min(dim=1).values
    return pred_to_target, target_to_pred


def chamfer_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pred_to_target, target_to_pred = chamfer_terms(pred, target)
    loss_a = pred_to_target.mean(dim=1)
    if target_weights is None:
        loss_b = target_to_pred.mean(dim=1)
    else:
        weights = target_weights.to(target_to_pred.dtype).clamp(min=1.0e-4)
        loss_b = (target_to_pred * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
    return (loss_a + loss_b).mean()


def boundary_weights(
    target_boundary: torch.Tensor,
    boundary_weight: float,
) -> torch.Tensor:
    return torch.where(
        target_boundary > 0.5,
        torch.full_like(target_boundary, boundary_weight),
        torch.ones_like(target_boundary),
    )


def fragment_coverage_loss(
    pred: torch.Tensor,
    canonical_fragments: torch.Tensor,
    fragment_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, max_fragments, points_per_fragment, _ = canonical_fragments.shape
    pred_expanded = pred[:, None, :, :].expand(
        batch_size, max_fragments, pred.shape[1], 3
    )
    frag = canonical_fragments
    dist = torch.cdist(
        frag.reshape(batch_size * max_fragments, points_per_fragment, 3),
        pred_expanded.reshape(batch_size * max_fragments, pred.shape[1], 3),
        p=2,
    )
    nearest = dist.min(dim=2).values
    hausdorff = nearest.max(dim=1).values.reshape(batch_size, max_fragments)
    masked = hausdorff * fragment_mask.to(hausdorff.dtype)
    return masked.sum() / fragment_mask.sum().clamp(min=1).to(masked.dtype)


def compatibility_preservation_loss(
    current_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    fragment_mask: torch.Tensor,
) -> torch.Tensor:
    pair_mask = fragment_mask[:, :, None] & fragment_mask[:, None, :]
    eye = torch.eye(fragment_mask.shape[1], dtype=torch.bool, device=fragment_mask.device)[None, :, :]
    pair_mask = pair_mask & ~eye
    if not pair_mask.any():
        return current_scores.sum() * 0.0
    return F.mse_loss(current_scores[pair_mask], reference_scores[pair_mask])


def fscore_at_tau(pred: torch.Tensor, target: torch.Tensor, tau: float = 0.01) -> torch.Tensor:
    pred_to_target, target_to_pred = chamfer_terms(pred, target)
    precision = (pred_to_target < tau).to(pred.dtype).mean(dim=1)
    recall = (target_to_pred < tau).to(pred.dtype).mean(dim=1)
    return (2.0 * precision * recall / (precision + recall + 1.0e-8)).mean()


class AssemblyLoss(nn.Module):
    """Combined Stage 2 loss from the proposal."""

    def __init__(
        self,
        lambda_boundary: float = 2.0,
        lambda_coverage: float = 1.0,
        lambda_dropout: float = 0.5,
        lambda_consistency: float = 0.3,
        lambda_compat: float = 0.1,
        boundary_weight: float = 2.0,
        dropout_margin: float = 0.01,
    ) -> None:
        super().__init__()
        self.lambda_boundary = lambda_boundary
        self.lambda_coverage = lambda_coverage
        self.lambda_dropout = lambda_dropout
        self.lambda_consistency = lambda_consistency
        self.lambda_compat = lambda_compat
        self.boundary_weight = boundary_weight
        self.dropout_margin = dropout_margin

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_boundary: torch.Tensor,
        canonical_fragments: torch.Tensor,
        fragment_mask: torch.Tensor,
        current_scores: torch.Tensor,
        reference_scores: Optional[torch.Tensor] = None,
        subset_pred: Optional[torch.Tensor] = None,
        dropout_pred: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        weighted_cd = chamfer_distance(
            pred,
            target,
            target_weights=boundary_weights(target_boundary, self.boundary_weight),
        )
        plain_cd = chamfer_distance(pred, target)
        cov = fragment_coverage_loss(pred, canonical_fragments, fragment_mask)

        if subset_pred is not None:
            consistency = chamfer_distance(pred.detach(), subset_pred)
        else:
            consistency = pred.sum() * 0.0

        if dropout_pred is not None:
            drop_cd = chamfer_distance(dropout_pred, target)
            dropout_penalty = F.relu(plain_cd.detach() + self.dropout_margin - drop_cd)
        else:
            dropout_penalty = pred.sum() * 0.0

        if reference_scores is not None:
            compat = compatibility_preservation_loss(
                current_scores,
                reference_scores.detach(),
                fragment_mask,
            )
        else:
            compat = pred.sum() * 0.0

        total = (
            plain_cd
            + self.lambda_boundary * weighted_cd
            + self.lambda_coverage * cov
            + self.lambda_dropout * dropout_penalty
            + self.lambda_consistency * consistency
            + self.lambda_compat * compat
        )

        return {
            "loss": total,
            "loss_cd": plain_cd.detach(),
            "loss_boundary_cd": weighted_cd.detach(),
            "loss_coverage": cov.detach(),
            "loss_dropout": dropout_penalty.detach(),
            "loss_consistency": consistency.detach(),
            "loss_compat": compat.detach(),
            "fscore_tau": fscore_at_tau(pred.detach(), target.detach(), tau=0.05),
        }


def build_assembly_loss(cfg: dict) -> AssemblyLoss:
    loss_cfg = cfg.get("loss", {}).get("stage2", {})
    return AssemblyLoss(
        lambda_boundary=loss_cfg.get("lambda_boundary", 2.0),
        lambda_coverage=loss_cfg.get("lambda_coverage", 1.0),
        lambda_dropout=loss_cfg.get("lambda_dropout", 0.5),
        lambda_consistency=loss_cfg.get("lambda_consistency", 0.3),
        lambda_compat=loss_cfg.get("lambda_compat", 0.1),
        boundary_weight=loss_cfg.get("boundary_weight", 2.0),
        dropout_margin=loss_cfg.get("dropout_margin", 0.01),
    )
