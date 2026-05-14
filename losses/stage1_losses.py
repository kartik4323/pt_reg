"""Losses for Stage 1 compatibility pretraining."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.compatibility_three import Stage1Output
from models.token_encoder import FragmentEncoding


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


def _masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(features.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (features * weights).sum(dim=1) / denom


def boundary_alignment_loss(
    enc_a: FragmentEncoding,
    mask_a: torch.Tensor,
    enc_b: FragmentEncoding,
    mask_b: torch.Tensor,
) -> torch.Tensor:
    """Direct-match boundary feature agreement."""
    valid = (mask_a.sum(dim=1) > 0) & (mask_b.sum(dim=1) > 0)
    if not valid.any():
        return enc_a.embedding.sum() * 0.0

    feat_a = _masked_mean(enc_a.point_features[valid], mask_a[valid])
    feat_b = _masked_mean(enc_b.point_features[valid], mask_b[valid])
    return (1.0 - F.cosine_similarity(feat_a, feat_b, dim=-1)).mean()


class Stage1CompatibilityLoss(nn.Module):
    """
    Multi-margin triplet loss plus optional boundary and class supervision.

    Distances are ordered as:
        d(anchor, direct) < d(anchor, semantic) < d(anchor, negative)
    """

    def __init__(
        self,
        margin_direct_semantic: float = 0.5,
        margin_semantic_negative: float = 0.25,
        lambda_boundary: float = 0.2,
        lambda_classification: float = 0.5,
    ) -> None:
        super().__init__()
        self.margin_direct_semantic = margin_direct_semantic
        self.margin_semantic_negative = margin_semantic_negative
        self.lambda_boundary = lambda_boundary
        self.lambda_classification = lambda_classification

    def forward(
        self,
        output: Stage1Output,
        anchor_boundary: torch.Tensor,
        direct_boundary: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_anchor = output.anchor.embedding
        d_direct = _cosine_distance(z_anchor, output.direct.embedding)
        d_semantic = _cosine_distance(z_anchor, output.semantic.embedding)
        d_negative = _cosine_distance(z_anchor, output.negative.embedding)

        loss_direct_semantic = F.relu(
            d_direct - d_semantic + self.margin_direct_semantic
        ).mean()
        loss_semantic_negative = F.relu(
            d_semantic - d_negative + self.margin_semantic_negative
        ).mean()
        loss_triplet = loss_direct_semantic + loss_semantic_negative

        boundary = boundary_alignment_loss(
            output.anchor,
            anchor_boundary,
            output.direct,
            direct_boundary,
        )

        logits = torch.cat(
            [
                output.direct_pair.logits,
                output.semantic_pair.logits,
                output.negative_pair.logits,
            ],
            dim=0,
        )
        labels = torch.cat(
            [
                torch.zeros_like(d_direct, dtype=torch.long),
                torch.ones_like(d_semantic, dtype=torch.long),
                torch.full_like(d_negative, 2, dtype=torch.long),
            ],
            dim=0,
        )
        loss_class = F.cross_entropy(logits, labels)

        total = (
            loss_triplet
            + self.lambda_boundary * boundary
            + self.lambda_classification * loss_class
        )

        return {
            "loss": total,
            "loss_triplet": loss_triplet.detach(),
            "loss_boundary": boundary.detach(),
            "loss_class": loss_class.detach(),
            "dist_direct": d_direct.mean().detach(),
            "dist_semantic": d_semantic.mean().detach(),
            "dist_negative": d_negative.mean().detach(),
        }


def build_stage1_loss(cfg: dict) -> Stage1CompatibilityLoss:
    loss_cfg = cfg.get("loss", {}).get("stage1", {})
    return Stage1CompatibilityLoss(
        margin_direct_semantic=loss_cfg.get("margin_direct_semantic", 0.5),
        margin_semantic_negative=loss_cfg.get("margin_semantic_negative", 0.25),
        lambda_boundary=loss_cfg.get("lambda_boundary", 0.2),
        lambda_classification=loss_cfg.get("lambda_classification", 0.5),
    )
