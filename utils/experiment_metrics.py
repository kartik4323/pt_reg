"""Deterministic evaluation metrics used by the remote experiment runner."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.classical_registration import register_fragment
from models.pose import apply_transform, rotation_geodesic_error


def symmetric_chamfer(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample symmetric mean Euclidean Chamfer distance."""
    distance = torch.cdist(source, target)
    return 0.5 * (distance.min(dim=2).values.mean(dim=1) + distance.min(dim=1).values.mean(dim=1))


@torch.no_grad()
def equivalent_part_pose_metrics(
    fragments: torch.Tensor,
    predicted_rotations: torch.Tensor,
    predicted_translations: torch.Tensor,
    expected_rotations: torch.Tensor,
    expected_translations: torch.Tensor,
    equivalence_classes: torch.Tensor,
    fragment_mask: torch.Tensor,
    *,
    threshold: float = 0.01,
    comparison_points: int = 128,
) -> Dict[str, object]:
    """Equivalent-part-aware pose scoring without an optional SciPy dependency.

    We match predictions to canonical parts only within their documented
    equivalence class, greedily by a fixed low-resolution Chamfer cost.  This
    respects interchangeable legs/arms while keeping the metric deterministic
    and bounded for the 20-part protocol.
    """
    pred_parts = torch.einsum("bfnd,bfcd->bfnc", fragments, predicted_rotations) + predicted_translations[:, :, None, :]
    target_parts = torch.einsum("bfnd,bfcd->bfnc", fragments, expected_rotations) + expected_translations[:, :, None, :]
    if pred_parts.shape[2] > comparison_points:
        indices = torch.linspace(0, pred_parts.shape[2] - 1, comparison_points, device=pred_parts.device).long()
        pred_parts = pred_parts[:, :, indices]
        target_parts = target_parts[:, :, indices]

    all_part_distances, all_rotation_degrees, all_translation_errors = [], [], []
    successes = []
    for batch_idx in range(pred_parts.shape[0]):
        valid = torch.where(fragment_mask[batch_idx])[0].tolist()
        assignments = []
        for eq_class in sorted({int(equivalence_classes[batch_idx, part]) for part in valid}):
            group = [part for part in valid if int(equivalence_classes[batch_idx, part]) == eq_class]
            # A unique greedy assignment is adequate here because equivalent
            # parts are usually small sets; tie-breaking is part index order.
            candidates = []
            for pred_idx in group:
                for target_idx in group:
                    cost = float(symmetric_chamfer(
                        pred_parts[batch_idx, pred_idx : pred_idx + 1],
                        target_parts[batch_idx, target_idx : target_idx + 1],
                    )[0].cpu())
                    candidates.append((cost, pred_idx, target_idx))
            used_pred, used_target = set(), set()
            for cost, pred_idx, target_idx in sorted(candidates):
                if pred_idx not in used_pred and target_idx not in used_target:
                    assignments.append((cost, pred_idx, target_idx))
                    used_pred.add(pred_idx)
                    used_target.add(target_idx)
        sample_distances = []
        for distance, pred_idx, target_idx in assignments:
            sample_distances.append(distance)
            all_part_distances.append(distance)
            rotation = rotation_geodesic_error(
                predicted_rotations[batch_idx, pred_idx : pred_idx + 1],
                expected_rotations[batch_idx, target_idx : target_idx + 1],
            )[0]
            all_rotation_degrees.append(float(rotation.cpu()) * 180.0 / torch.pi)
            all_translation_errors.append(float(torch.linalg.vector_norm(
                predicted_translations[batch_idx, pred_idx] - expected_translations[batch_idx, target_idx]
            ).cpu()))
        successes.append(bool(sample_distances) and all(value <= threshold for value in sample_distances))
    return {
        "part_chamfers": all_part_distances,
        "rotation_degrees": all_rotation_degrees,
        "translation_errors": all_translation_errors,
        "part_accuracy_at_0.01": float(sum(value <= threshold for value in all_part_distances) / max(1, len(all_part_distances))),
        "assembly_success_rate": float(sum(successes) / max(1, len(successes))),
    }


@torch.no_grad()
def evaluate_stage2(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    *,
    batch_size: int = 1,
    max_samples: int = 0,
    registration_method: str = "auto",
    prediction_dir: str | Path | None = None,
) -> Dict[str, float]:
    """Evaluate reconstruction and learned contact graph on a fixed dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    chamfer, aligned, fit, coverage = [], [], [], []
    contact_hits = contact_total = 0
    per_sample = []
    output_path = Path(prediction_dir) if prediction_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
    for batch in loader:
        tensors = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        output = model(tensors["fragments"], tensors["fragment_mask"])
        raw = symmetric_chamfer(output.point_cloud, tensors["target"])
        chamfer.extend(raw.detach().cpu().tolist())
        distance = torch.cdist(output.point_cloud, tensors["target"])
        fit.extend(distance.min(dim=2).values.mean(dim=1).detach().cpu().tolist())
        coverage.extend(distance.min(dim=1).values.mean(dim=1).detach().cpu().tolist())
        object_ids = batch.get("object_id", [str(len(per_sample) + idx) for idx in range(len(raw))])
        for local_idx, (predicted, target, scores, adjacency, mask) in enumerate(zip(
            output.point_cloud, tensors["target"], output.compatibility_scores,
            tensors["adjacency"], tensors["fragment_mask"],
        )):
            raw_value = float(raw[local_idx].detach().cpu())
            point_dist = torch.cdist(predicted[None], target[None])[0]
            fit_value = float(point_dist.min(dim=1).values.mean().detach().cpu())
            coverage_value = float(point_dist.min(dim=0).values.mean().detach().cpu())
            valid = mask[:, None] & mask[None, :]
            valid.fill_diagonal_(False)
            contact_value = None
            if valid.any():
                correct = int(((scores >= 0.5) == adjacency.bool())[valid].sum().item())
                total = int(valid.sum().item())
                contact_hits += correct
                contact_total += total
                contact_value = correct / total
            object_id = str(object_ids[local_idx])
            # The fallback multi-start ICP samples random rotations/points. Its
            # seed is part of sample identity so validation selection remains
            # reproducible across process restarts.
            torch.manual_seed(int(hashlib.sha256(object_id.encode("utf-8")).hexdigest()[:8], 16))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(hashlib.sha256(object_id.encode("utf-8")).hexdigest()[:8], 16))
            try:
                rotation, translation = register_fragment(predicted, target, method=registration_method)
                placed = apply_transform(
                    predicted[None], rotation.to(device)[None], translation.to(device)[None]
                )[0]
                aligned_value = float(symmetric_chamfer(placed[None], target[None])[0].cpu())
                aligned.append(aligned_value)
            except Exception:
                aligned_value = float("inf")
                aligned.append(aligned_value)
            per_sample.append(
                {
                    "object_id": object_id,
                    "reconstruction_chamfer": raw_value,
                    "fit": fit_value,
                    "coverage": coverage_value,
                    "aligned_chamfer": aligned_value,
                    "alignment_residual": aligned_value,
                    "contact_accuracy": contact_value,
                }
            )
            if output_path:
                filename = hashlib.sha256(object_id.encode("utf-8")).hexdigest()[:16] + ".npz"
                np.savez_compressed(
                    output_path / filename,
                    object_id=np.asarray(object_id),
                    reconstruction=predicted.detach().cpu().numpy(),
                    target=target.detach().cpu().numpy(),
                    compatibility_scores=scores.detach().cpu().numpy(),
                    predicted_adjacency=(scores >= 0.5).detach().cpu().numpy(),
                )
        if max_samples and len(chamfer) >= max_samples:
            break
    finite_aligned = [value for value in aligned if value != float("inf")]
    return {
        "num_samples": len(chamfer),
        "reconstruction_chamfer": float(sum(chamfer) / max(1, len(chamfer))),
        "reconstruction_fit": float(sum(fit) / max(1, len(fit))),
        "reconstruction_coverage": float(sum(coverage) / max(1, len(coverage))),
        "aligned_reconstruction_chamfer": (
            float(sum(finite_aligned) / len(finite_aligned)) if finite_aligned else float("inf")
        ),
        "registration_failure_count": len(aligned) - len(finite_aligned),
        "contact_accuracy": float(contact_hits / max(1, contact_total)),
        "per_sample": per_sample,
    }
