"""CUDA memory preflight for the PartNet Stage-3 protocol."""

from __future__ import annotations

from typing import Dict

import torch

from models.assembly import build_assembly_model
from models.pose import build_pose_estimator


VRAM_POLICY_BYTES = 22 * 1024**3


def profile_stage3_worst_case(cfg: dict, device: torch.device) -> Dict[str, object]:
    """Profile one backward pass with the permitted 20-part input shape.

    This deliberately uses random geometry: it measures allocation behaviour,
    not model quality, and can run before PartNet is prepared.  Stage 2 is
    frozen in the real Stage-3 protocol, so it is executed under no-grad here.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"ran": False, "reason": "CUDA is required for the VRAM preflight"}

    stage3 = cfg.get("stage3", {})
    data = cfg.get("data", {})
    fragment = cfg.get("fragment", {})
    max_parts = int(fragment.get("max_fragments", 20))
    part_points = int(data.get("num_points_per_fragment", 1000))
    feature_points = int(stage3.get("target_feature_points", 512))
    geometry_points = int(stage3.get("geometry_loss_points", 2048))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    reconstruction = build_assembly_model(cfg).to(device).eval()
    pose = build_pose_estimator(cfg).to(device).train()
    fragments = torch.randn(1, max_parts, part_points, 3, device=device)
    mask = torch.ones(1, max_parts, dtype=torch.bool, device=device)
    target = torch.randn(1, max(feature_points, 1), 3, device=device)
    geometry_target = torch.randn(1, max(geometry_points, 1), 3, device=device)
    with torch.no_grad():
        # Includes the fixed 5,000-point reconstruction allocation.
        reconstruction(fragments, mask)
    with torch.amp.autocast("cuda", enabled=True):
        output = pose(fragments, target, mask)
        # Include the capped geometry-distance allocation too: this is the
        # largest loss-side tensor in the actual Stage-3 objective.
        geometry_distance = torch.cdist(output.aligned_union.float(), geometry_target.float())
        loss = output.aligned_union.float().square().mean() + geometry_distance.min(dim=2).values.mean()
    loss.backward()
    torch.cuda.synchronize(device)
    result = {
        "ran": True,
        "max_parts": max_parts,
        "fragment_points": part_points,
        "target_feature_points": feature_points,
        "geometry_loss_points": geometry_points,
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "policy_bytes": VRAM_POLICY_BYTES,
    }
    del output, loss, geometry_distance, geometry_target, target, fragments, mask, pose, reconstruction
    torch.cuda.empty_cache()
    return result


def apply_stage3_memory_policy(cfg: dict, profile: Dict[str, object]) -> bool:
    """Apply the specified low-memory fallback and return whether it was used."""
    if not profile.get("ran") or int(profile.get("max_reserved_bytes", 0)) <= VRAM_POLICY_BYTES:
        return False
    stage3 = cfg.setdefault("stage3", {})
    stage3["target_feature_points"] = 256
    stage3["geometry_loss_points"] = 1024
    return True
