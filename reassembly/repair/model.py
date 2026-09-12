"""Isolated repair model: local rigid geometry, contextual contacts, separate field.

Only within-fragment distances/angles enter geometric attention. Before a pose
exists, cross attention compares features and never subtracts fragment frames.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from reassembly.model import (GeometryEncoder, PartialContactMatcher, ReassemblyModel,
                              ShapeField, contact_point_indices, farthest_point_indices,
                              gather_points, mlp)


class InvariantAggregation(nn.Module):
    """Six distance/cosine statistics, with no coordinate-axis components."""

    def __init__(self, input_dim: int, dim: int, count: int, neighbors: int):
        super().__init__()
        self.count, self.neighbors = count, neighbors
        self.message = mlp(input_dim + 6, dim)

    def forward(self, xyz: Tensor, features: Tensor | None):
        indices = farthest_point_indices(xyz, self.count)
        centers = gather_points(xyz, indices)
        distances = torch.cdist(centers.float(), xyz.float(), compute_mode='donot_use_mm_for_euclid_dist')
        neighbors = distances.topk(min(self.neighbors, xyz.shape[1]), largest=False).indices
        surrounding = gather_points(xyz, neighbors).float()
        offset = surrounding - centers.float()[:, :, None]
        radial = centers.float() - xyz.float().mean(1, keepdim=True)
        local_axis = offset.mean(2)
        radius = offset.norm(dim=-1, keepdim=True)
        radial_radius = radial.norm(dim=-1, keepdim=True)[:, :, None].expand_as(radius)
        neighbor_radius = (surrounding - xyz.float().mean(1)[:, None, None]).norm(dim=-1, keepdim=True)
        radial_cos = F.cosine_similarity(offset, radial[:, :, None], dim=-1, eps=1e-6)[..., None]
        local_cos = F.cosine_similarity(offset, local_axis[:, :, None], dim=-1, eps=1e-6)[..., None]
        local_radius = local_axis.norm(dim=-1, keepdim=True)[:, :, None].expand_as(radius)
        geometry = torch.cat((radius, radial_radius, neighbor_radius, radial_cos, local_cos, local_radius), -1)
        if features is not None:
            geometry = torch.cat((geometry, gather_points(features, neighbors)), -1)
        return centers, self.message(geometry).amax(2), indices


class InvariantEncoder(GeometryEncoder):
    def __init__(self, config: dict):
        super().__init__(config)
        dim = int(config.get('dim', 128))
        self.levels = nn.ModuleList([
            InvariantAggregation(0 if i == 0 else dim, dim, int(count), int(config.get('neighbors', 16)))
            for i, count in enumerate(config.get('sample_counts', [256, 128, 64]))])

    def forward(self, points: Tensor):
        batch, fragments, count, _ = points.shape
        original = points.reshape(-1, count, 3)
        xyz, features = original, None
        indices = torch.arange(count, device=points.device).expand(len(original), -1)
        hierarchy = []
        for level in self.levels:
            xyz, features, selected = level(xyz, features)
            indices = indices.gather(1, selected)
            hierarchy.append((xyz, features))
        interpolated = []
        for coordinates, values in hierarchy:
            # Direct Euclidean differences preserve exact zero at observed
            # support points; matrix-product cdist can introduce cancellation
            # errors that inverse-distance interpolation magnifies on rotation.
            distance, nearest = torch.cdist(original.float(), coordinates.float(),
                compute_mode='donot_use_mm_for_euclid_dist').topk(min(3, len(coordinates[0])), largest=False)
            inverse = distance.clamp_min(1e-8).reciprocal()
            weights = inverse / inverse.sum(-1, keepdim=True)
            interpolated.append((gather_points(values, nearest) * weights[..., None]).sum(-2))
        full = self.propagation(torch.cat(interpolated, -1))
        descriptor = F.normalize(self.descriptor_head(full).float(), dim=-1)
        return dict(point_xyz=points, point_features=full.reshape(batch, fragments, count, -1),
            descriptor=descriptor.reshape(batch, fragments, count, -1),
            fracture_logits=self.fracture_head(full).reshape(batch, fragments, count),
            token_features=features.reshape(batch, fragments, -1, features.shape[-1]),
            token_xyz=xyz.reshape(batch, fragments, -1, 3), token_indices=indices.reshape(batch, fragments, -1))


class ContactAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.geometry_bias = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, heads))
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def self_context(self, features: Tensor, xyz: Tensor):
        radial = xyz.float() - xyz.float().mean(1, keepdim=True)
        unit = F.normalize(radial, dim=-1, eps=1e-6)
        angle_cosine = (unit[:, :, None] * unit[:, None, :]).sum(-1)
        geometry = torch.stack((torch.cdist(xyz.float(), xyz.float(),
            compute_mode='donot_use_mm_for_euclid_dist'), angle_cosine), -1)
        bias = self.geometry_bias(geometry).permute(0, 3, 1, 2).flatten(0, 1)
        normalized = self.self_norm(features)
        attended = self.self_attention(normalized, normalized, normalized,
                                       attn_mask=bias.to(normalized.dtype), need_weights=False)[0]
        return features + attended

    def forward(self, source: Tensor, target: Tensor, source_xyz: Tensor, target_xyz: Tensor):
        source, target = self.self_context(source, source_xyz), self.self_context(target, target_xyz)
        a, b = self.cross_norm(source), self.cross_norm(target)
        # Simultaneous feature-only cross attention preserves pair interchange.
        source = source + self.cross_attention(a, b, b, need_weights=False)[0]
        target = target + self.cross_attention(b, a, a, need_weights=False)[0]
        return (source + self.feedforward(self.output_norm(source)),
                target + self.feedforward(self.output_norm(target)))


class RepairContactMatcher(PartialContactMatcher):
    """Unchanged dustbin/weight semantics, with exposed consumed embeddings."""

    def __init__(self, config: dict, revised: bool):
        super().__init__(config)
        self.blocks = nn.ModuleList([
            ContactAttentionBlock(int(config.get('dim', 128)), int(config.get('heads', 4)))
            for _ in range(2 if revised else 0)])

    def forward(self, encoded: dict, prior=None):
        if prior is not None:
            raise ValueError('Repair contact matching is independent of scaffold predictions')
        xyz = encoded['point_xyz']
        batch, fragments, count, _ = xyz.shape
        indices = contact_point_indices(xyz.reshape(-1, count, 3),
            encoded['fracture_logits'].reshape(-1, count), self.contact_points).reshape(batch, fragments, -1)
        local_xyz = xyz.gather(2, indices[..., None].expand(-1, -1, -1, 3))
        descriptor = encoded['descriptor'].gather(2, indices[..., None].expand(-1, -1, -1, encoded['descriptor'].shape[-1]))
        point_features = encoded['point_features'].gather(2, indices[..., None].expand(-1, -1, -1, descriptor.shape[-1]))
        features = self.descriptor(torch.cat((descriptor, point_features), -1))
        with torch.autocast(device_type=features.device.type, enabled=False):
            fracture = encoded['fracture_logits'].float().sigmoid().gather(2, indices)
            temperature = self.log_temperature.float().exp().clamp(.02, 1.)
        pairs = []
        for i in range(fragments):
            for j in range(i + 1, fragments):
                a, b = features[:, i], features[:, j]
                for block in self.blocks:
                    a, b = block(a, b, local_xyz[:, i], local_xyz[:, j])
                with torch.autocast(device_type=features.device.type, enabled=False):
                    a, b = F.normalize(a.float(), dim=-1), F.normalize(b.float(), dim=-1)
                    logits = torch.einsum('bkd,bld->bkl', a, b) / temperature
                    source_logits = logits + fracture[:, j, None].clamp_min(1e-6).log()
                    target_logits = logits.transpose(-1, -2) + fracture[:, i, None].clamp_min(1e-6).log()
                    source = torch.cat((source_logits, self.dustbin.expand(*source_logits.shape[:-1], 1)), -1).softmax(-1)
                    target = torch.cat((target_logits, self.dustbin.expand(*target_logits.shape[:-1], 1)), -1).softmax(-1)
                    valid = encoded['fragment_mask'][:, i].bool() & encoded['fragment_mask'][:, j].bool()
                    weights = source[..., :-1] * target[..., :-1].transpose(-1, -2)
                    weights = weights * fracture[:, i, :, None] * fracture[:, j, None, :] * valid[:, None, None]
                    pairs.append(dict(i=i, j=j, source_xyz=local_xyz[:, i], target_xyz=local_xyz[:, j],
                        source_indices=indices[:, i], target_indices=indices[:, j], weights=weights,
                        source_prob=source, target_prob=target, source_embedding=a, target_embedding=b,
                        source_matchability=(1 - source[..., -1]).clamp(0, 1) * fracture[:, i],
                        target_matchability=(1 - target[..., -1]).clamp(0, 1) * fracture[:, j], valid=valid))
        return pairs


class AdaptedShapeField(ShapeField):
    """A field-owned adapter cannot change the frozen contact representation."""

    def __init__(self, config: dict, *, preserve_adapter_rng: bool = False):
        super().__init__(config)
        dim = int(config.get('dim', 128))
        if preserve_adapter_rng:
            # Models are initialized on CPU before being moved to the selected
            # device. The extra scaffold adapter must not shift the fresh
            # legacy matcher's initialization stream in the causal control.
            with torch.random.fork_rng(devices=[]):
                self.adapter = mlp(dim, dim)
        else:
            self.adapter = mlp(dim, dim)

    def context(self, encoded: dict):
        adapted = dict(encoded)
        adapted['token_features'] = encoded['token_features'] + self.adapter(encoded['token_features'])
        return super().context(adapted)


class RepairModel(ReassemblyModel):
    def __init__(self, cfg: dict):
        nn.Module.__init__(self)
        self.cfg = cfg
        repair = cfg.get('repair', {})
        variant = repair.get('geometry_variant', 'revised')
        if variant not in ('existing', 'revised'):
            raise ValueError('repair.geometry_variant must be existing or revised')
        if repair.get('view_supervision', 'resampled_contrastive') not in ('existing', 'resampled_contrastive'):
            raise ValueError('repair.view_supervision must be existing or resampled_contrastive')
        config = cfg.get('model', {})
        if variant == 'existing':
            # Preserve v2's encoder -> field -> matcher RNG allocation order.
            # No weights are read or copied from a checkpoint.
            self.encoder = GeometryEncoder(config)
            self.field = AdaptedShapeField(config, preserve_adapter_rng=True)
            self.matcher = RepairContactMatcher(config, revised=False)
        else:
            self.encoder = InvariantEncoder(config)
            self.matcher = RepairContactMatcher(config, revised=True)
            self.field = AdaptedShapeField(config)

    @property
    def field_adapter(self):
        return self.field.adapter

    def match(self, encoded: dict, use_scaffold: bool = False, prior_override=None):
        if prior_override is not None:
            raise ValueError('Scaffolds score/refine repair candidates; they do not override contact matching')
        # Retain the solver call signature; the boolean does not alter contacts.
        return self.matcher(encoded, None)


def configure_stage(model: RepairModel, stage: int):
    if stage not in (1, 2, 3):
        raise ValueError('Repair stages are 1 contact learning, 2 scaffold learning, 3 evaluation')
    for name, module in (('encoder', model.encoder), ('matcher', model.matcher), ('field', model.field)):
        enabled = (stage == 1 and name in ('encoder', 'matcher')) or (stage == 2 and name == 'field')
        module.requires_grad_(enabled)
        module.train(enabled and model.training)


__all__ = ['RepairModel', 'configure_stage']
