"""Deterministic target-fidelity perturbations for the Stage-3 test suite."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader

from data.assembly_dataset import build_object_dataset
from models.pose import apply_transform, build_pose_estimator
from models.classical_registration import register_fragment
from training.two_stage_trainer import _move_to_device, _subsample_points
from utils.experiment_metrics import equivalent_part_pose_metrics, symmetric_chamfer


NOISE_LEVELS = {
    "jitter": [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04],
    "dropout": [0.0, 0.1, 0.25, 0.5, 0.75],
    "local_hole": [0.0, 0.1, 0.2, 0.3, 0.4],
    "outliers": [0.0, 0.01, 0.05, 0.1],
    "global_se3_control": [1.0],
}


def _rotation(generator: torch.Generator, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    q = torch.randn(4, generator=generator, device=device, dtype=dtype)
    q = q / q.norm().clamp(min=1.0e-8)
    w, x, y, z = q
    return torch.stack(
        [
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w))),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w))),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
        ]
    )


def _resample_selected(points: torch.Tensor, indices: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    kept = points[indices]
    if len(kept) == 0:
        kept = points[:1]
    if len(kept) >= count:
        return kept[:count]
    repeat = torch.randint(len(kept), (count - len(kept),), generator=generator, device=points.device)
    return torch.cat((kept, kept[repeat]), dim=0)


def perturb_target(
    target: torch.Tensor,
    mode: str,
    level: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target plus known GT->perturbed SE(3) label composition."""
    if target.shape[0] != 1:
        raise ValueError("Noise suite is deliberately batch-size 1 for deterministic perturbation identity")
    generator = torch.Generator(device=target.device).manual_seed(seed)
    points = target[0]
    diagonal = torch.linalg.vector_norm(points.max(dim=0).values - points.min(dim=0).values).clamp(min=1.0e-6)
    identity = torch.eye(3, device=target.device, dtype=target.dtype)[None]
    zero = torch.zeros(1, 3, device=target.device, dtype=target.dtype)
    if mode == "jitter":
        return points.add(torch.randn(points.shape, generator=generator, device=points.device, dtype=points.dtype) * (level * diagonal))[None], identity, zero
    if mode == "dropout":
        keep = torch.rand(len(points), generator=generator, device=points.device) >= level
        return _resample_selected(points, torch.where(keep)[0], len(points), generator)[None], identity, zero
    if mode == "local_hole":
        centre = points[torch.randint(len(points), (1,), generator=generator, device=points.device)[0]]
        keep = torch.linalg.vector_norm(points - centre, dim=-1) > level * diagonal
        return _resample_selected(points, torch.where(keep)[0], len(points), generator)[None], identity, zero
    if mode == "outliers":
        noisy = points.clone()
        count = int(round(level * len(points)))
        if count:
            selected = torch.randperm(len(points), generator=generator, device=points.device)[:count]
            lower = points.min(dim=0).values - 0.1 * diagonal
            upper = points.max(dim=0).values + 0.1 * diagonal
            noisy[selected] = lower + torch.rand((count, 3), generator=generator, device=points.device, dtype=points.dtype) * (upper - lower)
        return noisy[None], identity, zero
    if mode == "global_se3_control":
        rotation = _rotation(generator, device=target.device, dtype=target.dtype)[None]
        translation = torch.randn((1, 3), generator=generator, device=target.device, dtype=target.dtype) * (0.25 * diagonal)
        return apply_transform(target, rotation, translation), rotation, translation
    raise ValueError(f"Unknown noise mode {mode}")


def _mean_ci(values: Iterable[float]) -> Dict[str, float]:
    tensor = torch.as_tensor(list(values), dtype=torch.float64)
    if len(tensor) == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "count": 0}
    mean = float(tensor.mean())
    ci = 0.0 if len(tensor) < 2 else float(1.96 * tensor.std(unbiased=True) / math.sqrt(len(tensor)))
    return {"mean": mean, "ci95": ci, "count": int(len(tensor))}


@torch.no_grad()
def evaluate_target_noise(
    cfg: dict,
    stage3_checkpoint: str,
    device: torch.device,
    *,
    repetitions: int = 5,
    max_samples: int = 0,
) -> Dict[str, object]:
    """Evaluate a learned Stage-3 checkpoint on fixed GT-target perturbations."""
    pose_model = build_pose_estimator(cfg).to(device)
    state = torch.load(stage3_checkpoint, map_location=device)
    pose_model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    pose_model.eval()
    requested = max_samples or cfg.get("evaluation", {}).get("max_samples", 0)
    dataset = build_object_dataset(cfg, "test", requested or None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    feature_count = int(cfg.get("stage3", {}).get("target_feature_points", 512))
    rows = []
    for sample_idx, batch in enumerate(loader):
        batch = _move_to_device(batch, device)
        object_id = str(batch.get("object_id", [sample_idx])[0])
        for mode, levels in NOISE_LEVELS.items():
            for level in levels:
                for repeat in range(repetitions):
                    seed = 1_003_001 * sample_idx + 10_007 * repeat + int(level * 1_000_000) + sum(map(ord, mode))
                    noisy_target, target_rotation, target_translation = perturb_target(batch["target"], mode, level, seed)
                    expected_rotation = target_rotation[:, None] @ batch["align_rotations"]
                    expected_translation = torch.einsum(
                        "bfd,bcd->bfc", batch["align_translations"], target_rotation
                    ) + target_translation[:, None, :]
                    prediction = pose_model(
                        batch["fragments"],
                        _subsample_points(noisy_target, feature_count),
                        batch["fragment_mask"],
                    )
                    aligned = prediction.aligned_union
                    target_chamfer = float(symmetric_chamfer(aligned, noisy_target)[0].cpu())
                    metric = equivalent_part_pose_metrics(
                        batch["fragments"], prediction.rotations, prediction.translations,
                        expected_rotation, expected_translation,
                        batch["equivalence_classes"], batch["fragment_mask"],
                    )
                    try:
                        torch.manual_seed(seed)
                        reg_rotation, reg_translation = register_fragment(noisy_target[0], batch["target"][0], method=cfg.get("stage3", {}).get("global_registration", {}).get("method", "auto"))
                        post_alignment = float(symmetric_chamfer(
                            apply_transform(noisy_target, reg_rotation.to(device)[None], reg_translation.to(device)[None]),
                            batch["target"],
                        )[0].cpu())
                    except Exception:
                        post_alignment = float("inf")
                    rows.append(
                        {
                            "object_id": object_id,
                            "mode": mode,
                            "level": level,
                            "seed": seed,
                            "target_chamfer": target_chamfer,
                            "post_alignment_chamfer": post_alignment,
                            "part_accuracy_at_0.01": metric["part_accuracy_at_0.01"],
                            "assembly_success_rate": metric["assembly_success_rate"],
                        }
                    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["level"])].append(row)
    curves = []
    for (mode, level), observations in sorted(grouped.items()):
        curves.append(
            {
                "mode": mode,
                "level": level,
                "post_alignment_chamfer": _mean_ci(row["post_alignment_chamfer"] for row in observations if math.isfinite(row["post_alignment_chamfer"])),
                "target_chamfer": _mean_ci(row["target_chamfer"] for row in observations),
                "part_accuracy_at_0.01": _mean_ci(row["part_accuracy_at_0.01"] for row in observations),
                "assembly_success_rate": _mean_ci(row["assembly_success_rate"] for row in observations),
            }
        )
    return {"repetitions": repetitions, "num_test_samples": len(dataset), "curves": curves, "observations": rows}
