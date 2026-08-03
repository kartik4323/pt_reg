"""Stage-2 losses for the coarse occupancy prior.

Replaces the point-cloud ``AssemblyLoss`` when ``model.assembly.decoder ==
"occupancy"``. The prior is deliberately low-fidelity -- its job is to be
*coarsely* right and to know where it is unreliable -- so the objective is:

* **BCE** on occupancy at sampled query points (the standard implicit-field loss),
  optionally class-balanced because most of a unit cube is empty;
* **soft IoU** (differentiable Dice-like term), which optimizes the metric we
  actually report and is far less sensitive to the empty/occupied imbalance;
* **confidence calibration** -- the head must predict its own correctness, which is
  what lets Stage 3 down-weight unreliable regions;
* **compatibility preservation**, reused verbatim from the point-cloud loss, so the
  Stage-1 pairwise scores are not destroyed while Stage 2 trains;
* **dropout / consistency hinges**, ported in structure from the point-cloud loss
  with occupancy IoU in place of Chamfer: removing a fragment should make the
  prediction *worse*, and predicting from a subset should stay consistent.

Query sampling lives here too (:func:`sample_occupancy_queries`) because the label
for a query is a lookup into the ground-truth grid.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.assembly_losses import compatibility_preservation_loss


def occupancy_at(grid: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
    """Look up a (B,R,R,R) occupancy grid at (B,Q,3) coords in [-1,1] -> (B,Q).

    Nearest-cell lookup: the grid is the ground truth, so no interpolation.
    """
    res = grid.shape[-1]
    idx = torch.floor((query_xyz + 1.0) / (2.0 / res)).long().clamp(0, res - 1)
    flat = grid.reshape(grid.shape[0], -1)
    lin = (idx[..., 0] * res + idx[..., 1]) * res + idx[..., 2]
    return torch.gather(flat, 1, lin)


def sample_occupancy_queries(
    grid: torch.Tensor,
    num_queries: int = 4096,
    surface_ratio: float = 0.5,
    surface_jitter: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample query points + their GT occupancy from a (B,R,R,R) grid.

    A mix of uniform-in-volume and near-surface queries: uniform alone wastes most
    samples deep inside empty space, which makes the boundary (the part that
    matters) badly under-constrained.
    Returns ``(query_xyz (B,Q,3), labels (B,Q) in {0,1})``.
    """
    batch, res = grid.shape[0], grid.shape[-1]
    device, dtype = grid.device, grid.dtype
    pitch = 2.0 / res
    n_surface = int(num_queries * surface_ratio)
    n_uniform = num_queries - n_surface

    queries = [torch.rand(batch, n_uniform, 3, device=device, dtype=dtype) * 2.0 - 1.0]

    if n_surface > 0:
        occ = grid > 0.5
        # boundary cells: occupied with an empty 6-neighbour
        empty_neighbour = torch.zeros_like(occ)
        for axis in (1, 2, 3):
            for shift in (-1, 1):
                rolled = torch.roll(occ, shifts=shift, dims=axis)
                sl = [slice(None)] * 4
                sl[axis] = 0 if shift == 1 else res - 1
                rolled[tuple(sl)] = False
                empty_neighbour |= ~rolled
        surface = (occ & empty_neighbour).reshape(batch, -1)

        lin = torch.arange(res, device=device, dtype=dtype)
        centers = torch.stack(
            torch.meshgrid((lin + 0.5) / res * 2 - 1, (lin + 0.5) / res * 2 - 1,
                           (lin + 0.5) / res * 2 - 1, indexing="ij"),
            dim=-1,
        ).reshape(-1, 3)

        picked = torch.empty(batch, n_surface, 3, device=device, dtype=dtype)
        for b in range(batch):
            idx = torch.nonzero(surface[b], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:                      # degenerate: fall back to uniform
                picked[b] = torch.rand(n_surface, 3, device=device, dtype=dtype) * 2 - 1
                continue
            sel = idx[torch.randint(0, idx.numel(), (n_surface,), device=device)]
            jitter = (torch.rand(n_surface, 3, device=device, dtype=dtype) - 0.5) * pitch * surface_jitter
            picked[b] = (centers[sel] + jitter).clamp(-1.0, 1.0)
        queries.append(picked)

    query_xyz = torch.cat(queries, dim=1)
    labels = (occupancy_at(grid, query_xyz) > 0.5).to(dtype)
    return query_xyz, labels


def soft_iou_loss(logits: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - soft IoU between sigmoid(logits) and binary labels. Differentiable."""
    prob = torch.sigmoid(logits)
    inter = (prob * labels).sum(dim=-1)
    union = (prob + labels - prob * labels).sum(dim=-1)
    return (1.0 - (inter + eps) / (union + eps)).mean()


def hard_iou(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Reported (non-differentiable) IoU at threshold 0."""
    pred = (logits > 0).to(labels.dtype)
    inter = (pred * labels).sum(dim=-1)
    union = ((pred + labels) > 0).to(labels.dtype).sum(dim=-1)
    return ((inter + 1e-6) / (union + 1e-6)).mean()


class OccupancyAssemblyLoss(nn.Module):
    """Stage-2 objective for the coarse occupancy prior."""

    def __init__(
        self,
        lambda_iou: float = 1.0,
        lambda_confidence: float = 0.1,
        lambda_compat: float = 0.1,
        lambda_dropout: float = 0.5,
        lambda_consistency: float = 0.3,
        dropout_margin: float = 0.01,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.lambda_iou = lambda_iou
        self.lambda_confidence = lambda_confidence
        self.lambda_compat = lambda_compat
        self.lambda_dropout = lambda_dropout
        self.lambda_consistency = lambda_consistency
        self.dropout_margin = dropout_margin
        self.pos_weight = pos_weight

    def forward(
        self,
        occupancy_logits: torch.Tensor,
        occupancy_labels: torch.Tensor,
        occupancy_confidence: Optional[torch.Tensor] = None,
        current_scores: Optional[torch.Tensor] = None,
        reference_scores: Optional[torch.Tensor] = None,
        fragment_mask: Optional[torch.Tensor] = None,
        subset_logits: Optional[torch.Tensor] = None,
        dropout_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pw = None
        if self.pos_weight is not None:
            pw = torch.tensor(float(self.pos_weight), device=occupancy_logits.device)
        bce = F.binary_cross_entropy_with_logits(occupancy_logits, occupancy_labels, pos_weight=pw)
        iou_loss = soft_iou_loss(occupancy_logits, occupancy_labels)

        # Confidence must predict its own correctness: target = 1 - |p - label|.
        if occupancy_confidence is not None:
            with torch.no_grad():
                target = 1.0 - (torch.sigmoid(occupancy_logits) - occupancy_labels).abs()
            conf_loss = F.binary_cross_entropy_with_logits(occupancy_confidence, target)
        else:
            conf_loss = occupancy_logits.sum() * 0.0

        if reference_scores is not None and current_scores is not None and fragment_mask is not None:
            compat = compatibility_preservation_loss(
                current_scores, reference_scores.detach(), fragment_mask
            )
        else:
            compat = occupancy_logits.sum() * 0.0

        # Consistency: a prediction from a fragment SUBSET should not contradict the
        # full prediction (structure ported from the point-cloud loss).
        if subset_logits is not None:
            consistency = F.binary_cross_entropy_with_logits(
                subset_logits, torch.sigmoid(occupancy_logits).detach()
            )
        else:
            consistency = occupancy_logits.sum() * 0.0

        # Dropout hinge: dropping a fragment must make the prediction WORSE. Penalize
        # when the dropped-input IoU is not worse than the full IoU by a margin.
        if dropout_logits is not None:
            full_iou = hard_iou(occupancy_logits.detach(), occupancy_labels)
            drop_iou_soft = 1.0 - soft_iou_loss(dropout_logits, occupancy_labels)
            dropout_penalty = F.relu(drop_iou_soft - (full_iou - self.dropout_margin))
        else:
            dropout_penalty = occupancy_logits.sum() * 0.0

        total = (
            bce
            + self.lambda_iou * iou_loss
            + self.lambda_confidence * conf_loss
            + self.lambda_compat * compat
            + self.lambda_consistency * consistency
            + self.lambda_dropout * dropout_penalty
        )

        with torch.no_grad():
            iou = hard_iou(occupancy_logits, occupancy_labels)
            acc = ((occupancy_logits > 0).to(occupancy_labels.dtype) == occupancy_labels).to(
                occupancy_labels.dtype
            ).mean()
            occupied_frac = occupancy_labels.mean()

        return {
            "loss": total,
            "occupancy_bce": bce.detach(),
            "occupancy_iou": iou,
            "occupancy_accuracy": acc,
            "occupancy_target_frac": occupied_frac,
            "loss_confidence": conf_loss.detach(),
            "loss_compat": compat.detach(),
            "loss_consistency": consistency.detach(),
            "loss_dropout": dropout_penalty.detach(),
            # Aliases so the shared Stage-2 log line and history render unchanged for
            # both decoder types (it hard-indexes loss_cd / loss_coverage).
            "loss_cd": bce.detach(),
            "loss_coverage": iou_loss.detach(),
        }


def build_occupancy_loss(cfg: dict) -> OccupancyAssemblyLoss:
    loss_cfg = cfg.get("loss", {}).get("stage2", {})
    return OccupancyAssemblyLoss(
        lambda_iou=loss_cfg.get("lambda_iou", 1.0),
        lambda_confidence=loss_cfg.get("lambda_confidence", 0.1),
        lambda_compat=loss_cfg.get("lambda_compat", 0.1),
        lambda_dropout=loss_cfg.get("lambda_dropout", 0.5),
        lambda_consistency=loss_cfg.get("lambda_consistency", 0.3),
        dropout_margin=loss_cfg.get("dropout_margin", 0.01),
        pos_weight=loss_cfg.get("occupancy_pos_weight"),
    )
