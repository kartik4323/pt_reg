"""Fresh XYZ-only geometry, uncertain shape field, and partial contact matching.

All coordinates are fragment-local, except field queries, which are expressed in
the observed reference frame. Cross-fragment attention has no xyz position bias.
The module uses ordinary PyTorch operations and has no custom CUDA dependency.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def gather_points(values: Tensor, indices: Tensor) -> Tensor:
    """Gather arbitrary point-index dimensions without materializing repeats."""
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch.reshape((-1,) + (1,) * (indices.ndim - 1)), indices]


@torch.no_grad()
def farthest_point_indices(xyz: Tensor, count: int) -> Tensor:
    """Deterministic FPS; ties use the first input index."""
    count = min(count, xyz.shape[1])
    points = xyz.float()
    selected = torch.empty((len(points), count), dtype=torch.long, device=points.device)
    nearest = torch.full(points.shape[:2], torch.inf, device=points.device)
    current = (points - points.mean(1, keepdim=True)).square().sum(-1).argmax(-1)
    batch = torch.arange(len(points), device=points.device)
    for index in range(count):
        selected[:, index] = current
        distance = (points - points[batch, current, None]).square().sum(-1)
        nearest = torch.minimum(nearest, distance)
        current = nearest.argmax(-1)
    return selected


def mlp(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.LayerNorm(out_channels),
                         nn.GELU(), nn.Linear(out_channels, out_channels), nn.GELU())


@torch.no_grad()
def contact_point_indices(xyz: Tensor, fracture_logits: Tensor, count: int) -> Tensor:
    """Mix global coverage with predicted-contact coverage, using XYZ only.

    Half the budget is ordinary FPS. The rest covers unselected predicted
    fracture points before falling back to the remaining surface. This avoids
    losing a small interface at the encoder's 64-token context bottleneck.
    No fracture supervision is consulted by this discrete selector.
    """
    count = min(count, xyz.shape[1])
    base = max(1, count // 2)
    indices = farthest_point_indices(xyz, base)
    selected = torch.zeros(xyz.shape[:2], dtype=torch.bool, device=xyz.device)
    selected.scatter_(1, indices, True)
    nearest = torch.cdist(xyz.float(), gather_points(xyz, indices).float()).square().amin(-1)
    contact = fracture_logits.float() >= 0
    result = [indices]
    rows = torch.arange(len(xyz), device=xyz.device)
    for _ in range(count - base):
        available = contact & ~selected
        eligible = torch.where(available.any(-1, keepdim=True), available, ~selected)
        current = nearest.masked_fill(~eligible, -1).argmax(-1)
        result.append(current[:, None])
        selected[rows, current] = True
        distance = (xyz.float() - xyz[rows, current, None].float()).square().sum(-1)
        nearest = torch.minimum(nearest, distance)
    return torch.cat(result, -1)


class LocalAggregation(nn.Module):
    def __init__(self, input_dim: int, dim: int, count: int, neighbors: int):
        super().__init__()
        self.count, self.neighbors = count, neighbors
        self.message = mlp(input_dim + 4, dim)

    def forward(self, xyz: Tensor, features: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        indices = farthest_point_indices(xyz, self.count)
        centers = gather_points(xyz, indices)
        # cdist is explicitly fp32: CPU and CUDA AMP have different support.
        distances = torch.cdist(centers.float(), xyz.float())
        neighbors = distances.topk(min(self.neighbors, xyz.shape[1]), largest=False).indices
        relative = gather_points(xyz, neighbors) - centers[:, :, None, :]
        geometry = torch.cat((relative, relative.float().norm(dim=-1, keepdim=True)), -1)
        if features is not None:
            geometry = torch.cat((geometry, gather_points(features, neighbors)), -1)
        aggregated = self.message(geometry).amax(2)
        return centers, aggregated, indices


def interpolate(xyz: Tensor, support_xyz: Tensor, features: Tensor) -> Tensor:
    distance, indices = torch.cdist(xyz.float(), support_xyz.float()).topk(
        min(3, support_xyz.shape[1]), largest=False)
    inverse = distance.clamp_min(1e-8).reciprocal()
    weights = inverse / inverse.sum(-1, keepdim=True)
    return (gather_points(features, indices) * weights[..., None]).sum(-2)


class GeometryEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        dim = int(config.get("dim", 128))
        counts = config.get("sample_counts", [256, 128, 64])
        if len(counts) != 3 or any(int(n) < 1 for n in counts):
            raise ValueError("model.sample_counts must contain three positive counts")
        neighbors = int(config.get("neighbors", 16))
        self.levels = nn.ModuleList([
            LocalAggregation(0 if level == 0 else dim, dim, int(count), neighbors)
            for level, count in enumerate(counts)])
        self.propagation = mlp(dim * 3, dim)
        self.fracture_head = nn.Linear(dim, 1)
        self.descriptor_head = nn.Linear(dim, dim)

    def forward(self, points: Tensor) -> dict[str, Tensor]:
        batch, fragments, npoints, _ = points.shape
        xyz = points.reshape(batch * fragments, npoints, 3)
        features = None
        original_indices = torch.arange(npoints, device=points.device).expand(len(xyz), -1)
        hierarchy = []
        for level in self.levels:
            xyz, features, indices = level(xyz, features)
            original_indices = original_indices.gather(1, indices)
            hierarchy.append((xyz, features))
        full_features = self.propagation(torch.cat([
            interpolate(points.reshape(-1, npoints, 3), coords, feats)
            for coords, feats in hierarchy], -1))
        descriptors = F.normalize(self.descriptor_head(full_features).float(), dim=-1)
        return {
            "point_xyz": points,
            "point_features": full_features.reshape(batch, fragments, npoints, -1),
            "descriptor": descriptors.reshape(batch, fragments, npoints, -1),
            "fracture_logits": self.fracture_head(full_features).reshape(batch, fragments, npoints),
            "token_features": features.reshape(batch, fragments, -1, features.shape[-1]),
            "token_xyz": xyz.reshape(batch, fragments, -1, 3),
            "token_indices": original_indices.reshape(batch, fragments, -1),
        }


class ShapeField(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        dim = int(config.get("dim", 128))
        heads = int(config.get("heads", 4))
        self.truncation = float(config.get("truncation", 0.1))
        self.log_min = float(config.get("log_scale_min", -6.0))
        self.log_max = float(config.get("log_scale_max", -2.0))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.context_attention = nn.TransformerEncoder(
            layer, int(config.get("attention_layers", 2)), enable_nested_tensor=False)
        self.fragment_context = nn.Linear(dim, dim)
        self.query = mlp(dim + 3, dim)
        self.key = nn.Linear(dim, dim)
        self.relative_bias = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))
        self.decoder = nn.Sequential(nn.Linear(dim * 2 + 4, dim), nn.GELU(), nn.Linear(dim, 2))

    def context(self, encoded: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        tokens = encoded["token_features"]
        batch, fragments, count, dim = tokens.shape
        # A pooled descriptor tags each token with its fragment's geometry.
        # This retains grouping without an arbitrary fragment-index embedding.
        tokens = tokens + self.fragment_context(tokens.mean(2))[:, :, None]
        padding = ~encoded["fragment_mask"].bool()[:, :, None].expand(-1, -1, count)
        attended = self.context_attention(tokens.reshape(batch, fragments * count, dim),
                                          src_key_padding_mask=padding.reshape(batch, -1))
        # Spatial relationships are used only within the observed reference.
        attended = attended.reshape(batch, fragments, count, dim)
        valid = (~padding).to(attended.dtype)[..., None]
        global_context = (attended * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)
        rows = torch.arange(batch, device=tokens.device)
        anchor = encoded["anchor_index"].long()
        return global_context, attended[rows, anchor], encoded["token_xyz"][rows, anchor]

    def forward(self, encoded: dict[str, Tensor], queries: Tensor,
                context: tuple[Tensor, Tensor, Tensor] | None = None) -> dict[str, Tensor]:
        global_context, anchor_features, anchor_xyz = context if context is not None else self.context(encoded)
        query_features = self.query(torch.cat((queries, global_context[:, None].expand(-1, queries.shape[1], -1)), -1))
        relative = queries[:, :, None] - anchor_xyz[:, None]
        relative_geometry = torch.cat((relative, relative.float().norm(dim=-1, keepdim=True)), -1)
        logits = torch.einsum("bqd,bkd->bqk", query_features, self.key(anchor_features))
        logits = logits / math.sqrt(anchor_features.shape[-1]) + self.relative_bias(relative_geometry).squeeze(-1)
        attention = logits.float().softmax(-1).to(anchor_features.dtype)
        attended = torch.einsum("bqk,bkd->bqd", attention, anchor_features)
        geometry = (attention[..., None] * relative_geometry).sum(-2)
        prediction = self.decoder(torch.cat((query_features, attended, geometry), -1))
        return {
            "distance": prediction[..., 0].tanh() * self.truncation,
            "log_scale": self.log_min + (self.log_max - self.log_min) * prediction[..., 1].sigmoid(),
        }


class PartialContactMatcher(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.contact_points = int(config.get("contact_points", 256))
        dim = int(config.get("dim", 128))
        self.descriptor = mlp(dim * 2, dim)
        self.condition = mlp(5, dim)
        self.condition_gate = nn.Linear(dim, dim * 2)
        self.dustbin = nn.Parameter(torch.tensor(1.0))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.1)))

    def forward(self, encoded: dict[str, Tensor], prior: Tensor | None) -> list[dict[str, Any]]:
        xyz = encoded["point_xyz"]
        batch, fragments, count, _ = xyz.shape
        indices = contact_point_indices(xyz.reshape(-1, count, 3),
                                        encoded["fracture_logits"].reshape(-1, count),
                                        self.contact_points).reshape(batch, fragments, -1)
        local_xyz = xyz.gather(2, indices[..., None].expand(-1, -1, -1, 3))
        point_desc = encoded["descriptor"].gather(2, indices[..., None].expand(-1, -1, -1, encoded["descriptor"].shape[-1]))
        point_features = encoded["point_features"].gather(2, indices[..., None].expand(-1, -1, -1, point_desc.shape[-1]))
        features = self.descriptor(torch.cat((point_desc, point_features), -1))
        if prior is not None:
            context = self.condition(prior).mean(1)
            gate, shift = self.condition_gate(context).chunk(2, dim=-1)
            features = features * (1 + gate[:, None, None].tanh()) + shift[:, None, None]
        # A cast after sigmoid/log or the dot product is too late: AMP may
        # overflow their intermediate backward values before GradScaler unscales.
        # Keep this small token matching block in fp32; learned MLPs still use AMP.
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = F.normalize(features.float(), dim=-1)
            fracture = encoded["fracture_logits"].float().sigmoid().gather(2, indices)
            temperature = self.log_temperature.float().exp().clamp(0.02, 1.0)
            pairs = []
            for i in range(features.shape[1]):
                for j in range(i + 1, features.shape[1]):
                    logits = torch.einsum("bkd,bld->bkl", features[:, i], features[:, j]) / temperature
                    # Explicit dustbins keep unmatched mass. Neither direction is
                    # renormalized after discarding its dustbin column.
                    source_logits = logits + fracture[:, j, None].clamp_min(1e-6).log()
                    target_logits = logits.transpose(-1, -2) + fracture[:, i, None].clamp_min(1e-6).log()
                    source = torch.cat((source_logits, self.dustbin.expand(*source_logits.shape[:-1], 1)), -1).softmax(-1)
                    target = torch.cat((target_logits, self.dustbin.expand(*target_logits.shape[:-1], 1)), -1).softmax(-1)
                    weights = source[..., :-1] * target[..., :-1].transpose(-1, -2)
                    weights = weights * fracture[:, i, :, None] * fracture[:, j, None, :]
                    valid = encoded["fragment_mask"][:, i].bool() & encoded["fragment_mask"][:, j].bool()
                    weights = weights * valid[:, None, None]
                    pairs.append({"i": i, "j": j, "source_xyz": local_xyz[:, i],
                                  "target_xyz": local_xyz[:, j],
                                  "source_indices": indices[:, i], "target_indices": indices[:, j],
                                  "weights": weights, "source_prob": source, "target_prob": target,
                                  "source_matchability": (1 - source[..., -1]).clamp(0, 1) * fracture[:, i],
                                  "target_matchability": (1 - target[..., -1]).clamp(0, 1) * fracture[:, j],
                                  "valid": valid})
        return pairs


class ReassemblyModel(nn.Module):
    """Independent v2 model; never discovers or loads any checkpoint."""
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        config = cfg.get("model", {})
        self.encoder = GeometryEncoder(config)
        self.field = ShapeField(config)
        self.matcher = PartialContactMatcher(config)
        extent = float(config.get("field_extent", 1.5))
        axis = torch.linspace(-extent, extent, 4)
        self.register_buffer("conditioning_queries", torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3))

    def encode(self, points: Tensor, fragment_mask: Tensor, anchor_index: Tensor) -> dict[str, Tensor]:
        if points.ndim != 4 or points.shape[-1] != 3 or points.shape[1] not in (2, 3):
            raise ValueError("points must have shape (B, 2 or 3, N, 3)")
        if fragment_mask.shape != points.shape[:2] or anchor_index.shape != points.shape[:1]:
            raise ValueError("fragment_mask and anchor_index must match the batch")
        if not torch.isfinite(points).all() or points.shape[2] < 3:
            raise ValueError("points must be finite and contain at least three points per fragment")
        if (fragment_mask.sum(-1) < 2).any():
            raise ValueError("each input must contain two or three complete fragments")
        if ((anchor_index < 0) | (anchor_index >= points.shape[1])).any():
            raise ValueError("anchor_index is outside the fragment set")
        if not fragment_mask.gather(1, anchor_index.long()[:, None]).all():
            raise ValueError("anchor_index must identify a valid fragment")
        output = self.encoder(points)
        output.update(fragment_mask=fragment_mask.bool(), anchor_index=anchor_index.long())
        return output

    def scaffold(self, encoded: dict[str, Tensor], query: Tensor) -> dict[str, Tensor]:
        return self.field(encoded, query)

    def match(self, encoded: dict[str, Tensor], use_scaffold: bool = True,
              prior_override: Tensor | None = None) -> list[dict[str, Any]]:
        prior = None
        if use_scaffold and prior_override is not None:
            if prior_override.ndim != 3 or prior_override.shape[-1] != 5 or len(prior_override) != len(encoded["anchor_index"]):
                raise ValueError("prior_override must be (B, Q, 5): xyz, distance, log_scale")
            prior = prior_override
        elif use_scaffold:
            queries = self.conditioning_queries[None].expand(len(encoded["anchor_index"]), -1, -1)
            rows = torch.arange(len(queries), device=queries.device)
            anchor_queries = encoded["token_xyz"][rows, encoded["anchor_index"]]
            # Include observed reference locations: a small global lattice
            # alone can miss a thin object's entire truncation band.
            queries = torch.cat((queries, anchor_queries), dim=1)
            prediction = self.scaffold(encoded, queries)
            prior = torch.cat((queries, prediction["distance"][..., None], prediction["log_scale"][..., None]), -1)
        return self.matcher(encoded, prior)

    def forward(self, points: Tensor, fragment_mask: Tensor, anchor_index: Tensor,
                query: Tensor | None = None) -> dict[str, Any]:
        encoded = self.encode(points, fragment_mask, anchor_index)
        if query is not None:
            return {"encoded": encoded, "scaffold": self.scaffold(encoded, query)}
        return encoded


def configure_stage(model: ReassemblyModel, stage: int) -> None:
    """Set trainable modules *and* modes after model.train(), on every stage."""
    if stage not in (1, 2, 3):
        raise ValueError("stage must be 1 (geometry), 2 (scaffold), or 3 (matching)")
    active = {1: (model.encoder, model.matcher),
              2: (model.encoder, model.field, model.matcher),
              3: (model.matcher,)}[stage]
    for module in (model.encoder, model.field, model.matcher):
        enabled = module in active
        module.requires_grad_(enabled)
        module.train(enabled and model.training)
