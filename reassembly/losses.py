"""Supervision for geometry, reference-frame fields, and partial contacts.

No complete-fragment dropout, canonical pose regression, negative-pair boundary
alignment, or confidence-weighted reconstruction objective is used here.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from .model import ReassemblyModel, configure_stage


def masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    weight = valid.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1)


def segmentation_loss(logits: Tensor, labels: Tensor, fragment_mask: Tensor) -> Tensor:
    valid = fragment_mask[..., None].expand_as(logits) & (labels >= 0)
    positive = valid & (labels > 0.5)
    negative = valid & ~positive
    # Balance the two surface classes without assigning enormous pos_weights
    # when a sampled fragment contains very few fracture points.
    raw = F.binary_cross_entropy_with_logits(logits.float(), labels.float().clamp(0, 1), reduction="none")
    count = positive.any().to(raw.dtype) + negative.any().to(raw.dtype)
    return (masked_mean(raw, positive) + masked_mean(raw, negative)) / count.clamp_min(1)


def contact_targets(pair: dict[str, Any], batch: dict[str, Tensor], radius: float) -> Tensor:
    i, j = pair["i"], pair["j"]
    source_indices, target_indices = pair["source_indices"], pair["target_indices"]
    source = batch["canonical_points"][:, i].gather(1, source_indices[..., None].expand(-1, -1, 3))
    target = batch["canonical_points"][:, j].gather(1, target_indices[..., None].expand(-1, -1, 3))
    source_interface = batch["interface_ids"][:, i].gather(1, source_indices)
    target_interface = batch["interface_ids"][:, j].gather(1, target_indices)
    same_interface = (source_interface[:, :, None] == target_interface[:, None, :]) & (source_interface[:, :, None] >= 0)
    near = torch.cdist(source.float(), target.float()) <= radius
    return same_interface & near & pair["valid"][:, None, None]


def directional_contact_loss(probabilities: Tensor, positives: Tensor, valid: Tensor) -> Tensor:
    """Any nearby point on the *same* interface is a correct local match.

    Positive and unmatched rows are balanced, so numerous exterior points do
    not make an all-dustbin model appear successful.
    """
    probabilities = probabilities.float()
    has_match = positives.any(-1)
    mass = (probabilities[..., :-1] * positives.to(probabilities.dtype)).sum(-1)
    matched_loss = -mass.clamp_min(1e-8).log()
    unmatched_loss = -probabilities[..., -1].clamp_min(1e-8).log()
    valid_rows = valid[:, None].expand_as(has_match)
    positive_rows = has_match & valid_rows
    negative_rows = ~has_match & valid_rows
    categories = positive_rows.any().to(mass.dtype) + negative_rows.any().to(mass.dtype)
    return (masked_mean(matched_loss, positive_rows) + masked_mean(unmatched_loss, negative_rows)) / categories.clamp_min(1)


def correspondence_loss(pairs: list[dict[str, Any]], batch: dict[str, Tensor],
                        radius: float) -> tuple[Tensor, Tensor]:
    losses, positive_counts = [], []
    for pair in pairs:
        positives = contact_targets(pair, batch, radius)
        source_loss = directional_contact_loss(pair["source_prob"], positives, pair["valid"])
        target_loss = directional_contact_loss(pair["target_prob"], positives.transpose(-1, -2), pair["valid"])
        losses.append((source_loss + target_loss) * 0.5)
        positive_counts.append(positives.sum())
    # A 2-3 fragment input always produces at least one pair. Invalid padded
    # pairs contribute no loss and are omitted from the average.
    valid_pairs = torch.stack([pair["valid"].any() for pair in pairs])
    return masked_mean(torch.stack(losses), valid_pairs), torch.stack(positive_counts).sum()


def transformed_view(points: Tensor) -> Tensor:
    """Independent Haar-SO(3) views retaining exact point identities."""
    random = torch.randn((*points.shape[:2], 3, 3), device=points.device, dtype=torch.float32)
    rotation, upper = torch.linalg.qr(random)
    signs = upper.diagonal(dim1=-2, dim2=-1).sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    rotation = rotation * signs[..., None, :]
    determinant = torch.linalg.det(rotation)
    orientation = torch.ones_like(signs)
    orientation[..., -1] = determinant
    rotation = rotation * orientation[..., None, :]
    return torch.einsum("bfnd,bfcd->bfnc", points.float(), rotation)


def view_consistency(model: ReassemblyModel, encoded: dict[str, Tensor],
                     batch: dict[str, Tensor]) -> Tensor:
    view = batch.get("points_view2")
    if view is None:
        view = transformed_view(batch["points"])
    other = model.encode(view, batch["fragment_mask"], batch["anchor_index"])
    # Both descriptors are evaluated at the very same sampled point IDs, even
    # though FPS tokens and neighborhoods may differ after the transformation.
    distance = 1 - (encoded["descriptor"].float() * other["descriptor"].float()).sum(-1)
    valid = batch["fragment_mask"].bool()[..., None].expand_as(distance)
    return masked_mean(distance, valid)


def field_losses(distance: Tensor, log_scale: Tensor, target: Tensor,
                 truncation: float) -> dict[str, Tensor]:
    error = (distance.float() - target.float().clamp(-truncation, truncation)).abs()
    # Calibration learns the bounded expected error without weakening the
    # unweighted TSDF loss. Detachment prevents the calibration term from
    # incentivizing a more uncertain / less accurate reconstruction.
    calibration = (error.detach() * torch.exp(-log_scale.float()) + log_scale.float()).mean()
    return {"sdf_l1": error.mean(), "sdf_calibration": calibration,
            "uncertainty_mae": (log_scale.float().exp() - error.detach()).abs().mean()}


def compute_losses(model: ReassemblyModel, batch: dict[str, Tensor], stage: int,
                   cfg: dict[str, Any]) -> dict[str, Tensor]:
    """Return a differentiable ``loss`` and detached scalar diagnostics.

    The caller configures stages before constructing the optimizer, and calls
    ``configure_stage`` after each ``model.train()``. ``loss`` is never detached.
    """
    if stage not in (1, 2, 3):
        raise ValueError("stage must be 1, 2, or 3")
    config = cfg.get("train", {})
    weights = cfg.get("loss", {})
    encoded = model.encode(batch["points"], batch["fragment_mask"], batch["anchor_index"])
    use_scaffold = stage == 3 and config.get("condition", "predicted") != "contact_only"
    pairs = model.match(encoded, use_scaffold=use_scaffold)
    matching, positive_count = correspondence_loss(pairs, batch, float(config.get("contact_radius", 0.05)))
    total = float(weights.get("matching", config.get("matching_weight", 1.0))) * matching
    diagnostics = {"matching": matching, "positive_contacts": positive_count}
    if stage in (1, 2):
        segmentation = segmentation_loss(encoded["fracture_logits"], batch["fracture_labels"], batch["fragment_mask"].bool())
        consistency_weight = float(weights.get("view_consistency", config.get("consistency_weight", 0.1)))
        consistency = view_consistency(model, encoded, batch) if consistency_weight else total * 0
        geometry_weight = float(weights.get("geometry_retention", config.get("geometry_retention_weight", 0.25))) if stage == 2 else 1.0
        total = geometry_weight * (total + float(weights.get("segmentation", config.get("segmentation_weight", 1.0))) * segmentation + consistency_weight * consistency)
        diagnostics.update(segmentation=segmentation, consistency=consistency)
    if stage == 2:
        prediction = model.scaffold(encoded, batch["sdf_queries"])
        losses = field_losses(prediction["distance"], prediction["log_scale"], batch["sdf_values"], model.field.truncation)
        total = total + float(weights.get("sdf", config.get("sdf_weight", 1.0))) * losses["sdf_l1"]
        total = total + float(weights.get("calibration", config.get("calibration_weight", 0.01))) * losses["sdf_calibration"]
        diagnostics.update(losses)
    return {"loss": total, **{name: value.detach() for name, value in diagnostics.items()}}


__all__ = ["compute_losses", "configure_stage", "contact_targets", "field_losses"]
