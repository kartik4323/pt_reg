"""Losses for Stage 1 binary compatibility pretraining."""

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


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE over paired fragment embeddings."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        if z_a.shape[0] < 2:
            return z_a.sum() * 0.0

        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        logits = z_a @ z_b.T / self.temperature
        labels = torch.arange(z_a.shape[0], device=z_a.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_ab + loss_ba)


class Stage1CompatibilityLoss(nn.Module):
    """BCE fit supervision plus InfoNCE over positive pairs."""

    def __init__(
        self,
        lambda_contrast: float = 0.5,
        temperature: float = 0.07,
        pos_weight: float = 1.0,
        lambda_boundary: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.lambda_boundary = lambda_boundary
        self.infonce = InfoNCELoss(temperature=temperature)
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(
        self,
        output: Stage1Output,
        labels: torch.Tensor,
        boundary_a: torch.Tensor | None = None,
        boundary_b: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        labels = labels.to(output.pair.logits.dtype)
        loss_bce = F.binary_cross_entropy_with_logits(
            output.pair.logits,
            labels,
            pos_weight=self.pos_weight.to(output.pair.logits.device),
        )

        positive = labels > 0.5
        loss_contrast = self.infonce(
            output.frag_a.embedding[positive],
            output.frag_b.embedding[positive],
        )

        if self.lambda_boundary > 0 and boundary_a is not None and boundary_b is not None:
            boundary = boundary_alignment_loss(
                output.frag_a,
                boundary_a,
                output.frag_b,
                boundary_b,
            )
        else:
            boundary = output.pair.logits.sum() * 0.0

        total = loss_bce + self.lambda_contrast * loss_contrast + self.lambda_boundary * boundary

        return {
            "loss": total,
            "loss_bce": loss_bce.detach(),
            "loss_contrast": loss_contrast.detach(),
            "loss_boundary": boundary.detach(),
            "positive_rate": labels.mean().detach(),
            "score_positive": output.pair.score[positive].mean().detach()
            if positive.any()
            else output.pair.score.sum().detach() * 0.0,
            "score_negative": output.pair.score[~positive].mean().detach()
            if (~positive).any()
            else output.pair.score.sum().detach() * 0.0,
        }


def build_stage1_loss(cfg: dict) -> Stage1CompatibilityLoss:
    loss_cfg = cfg.get("loss", {}).get("stage1", {})
    return Stage1CompatibilityLoss(
        lambda_contrast=loss_cfg.get("lambda_contrast", 0.5),
        temperature=loss_cfg.get("temperature", 0.07),
        pos_weight=loss_cfg.get("pos_weight", 1.0),
        lambda_boundary=loss_cfg.get("lambda_boundary", 0.0),
    )
