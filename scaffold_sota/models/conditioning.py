"""Matched, zero-gated scaffold branches for native recipient features.

This module never constructs an assembly model or evaluates ground truth. Query
coordinates must come from the study's fixed input-only sampling policy.
"""
from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import nn


class ConditioningBlock(nn.Module):
    """Residual cross-attention preserving the native feature shape.

    ``B2`` pools position-aware field encodings *before* exposing context to the
    recipient. Repeated pooled tokens have no local positional embedding. B1
    ignores all prior values, including validity, and uses a learned null bank.
    The same parameters exist in B1--B4 to make parameter counts comparable.
    """

    def __init__(self, dim: int, condition: str, token_dim: int = 128,
                 heads: int = 4, num_tokens: int = 512):
        super().__init__()
        if condition not in {"B1", "B2", "B3", "B4"}:
            raise ValueError("ConditioningBlock requires B1, B2, B3 or B4")
        if min(dim, token_dim, heads, num_tokens) < 1 or token_dim % heads:
            raise ValueError("Positive dimensions and token_dim divisible by heads required")
        self.condition, self.num_tokens = condition, int(num_tokens)
        self.field_encoder = nn.Sequential(nn.Linear(5, token_dim), nn.GELU(),
                                           nn.Linear(token_dim, token_dim))
        self.null_tokens = nn.Parameter(torch.empty(num_tokens, token_dim))
        nn.init.normal_(self.null_tokens, std=0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.query_projection = nn.Linear(dim, token_dim)
        self.reference_embedding = nn.Parameter(torch.zeros(token_dim))
        self.attention = nn.MultiheadAttention(token_dim, heads, dropout=0.0,
                                               batch_first=True)
        self.output_projection = nn.Linear(token_dim, dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def encode_prior(self, prior: Optional[Mapping], features: torch.Tensor):
        """Return context and padding mask; exposed for intervention tests."""
        batch = features.shape[0]
        if self.condition == "B1":
            return self.null_tokens.unsqueeze(0).expand(batch, -1, -1), None
        if prior is None:
            raise ValueError(f"{self.condition} requires a predicted prior")
        required = {"query_xyz", "distance", "valid"}
        if self.condition == "B4":
            required.add("log_scale")
        missing = required.difference(prior)
        if missing:
            raise ValueError(f"Missing prior fields: {sorted(missing)}")

        def tensor(key, dtype=None):
            return torch.as_tensor(prior[key], device=features.device,
                                   dtype=dtype or features.dtype).detach()

        xyz, distance = tensor("query_xyz"), tensor("distance")
        valid = tensor("valid", torch.bool)
        if xyz.ndim == 2:
            xyz, distance, valid = xyz.unsqueeze(0), distance.unsqueeze(0), valid.unsqueeze(0)
        if xyz.ndim != 3 or xyz.shape[-1] != 3 or distance.shape != xyz.shape[:2] or valid.shape != distance.shape:
            raise ValueError("Prior must have query_xyz[B,Q,3], distance[B,Q], valid[B,Q]")
        if xyz.shape[1] != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} fixed query tokens, got {xyz.shape[1]}")
        if xyz.shape[0] == 1 and batch != 1:
            xyz, distance, valid = xyz.expand(batch, -1, -1), distance.expand(batch, -1), valid.expand(batch, -1)
        if xyz.shape[0] != batch or not valid.any(dim=1).all():
            raise ValueError("Prior batch mismatch or no valid prior tokens")
        sigma = torch.zeros_like(distance)
        if self.condition == "B4":
            sigma = tensor("log_scale")
            if sigma.ndim == 1:
                sigma = sigma.unsqueeze(0)
            if sigma.shape[0] == 1 and batch != 1:
                sigma = sigma.expand(batch, -1)
            if sigma.shape != distance.shape:
                raise ValueError("log_scale shape must match distance")
        raw = torch.cat((xyz, distance.unsqueeze(-1), sigma.unsqueeze(-1)), dim=-1)
        if not torch.isfinite(raw[valid]).all():
            raise ValueError("Valid prior tokens contain nonfinite values")
        raw = torch.where(valid.unsqueeze(-1), raw, torch.zeros_like(raw))
        encoded = self.field_encoder(raw)
        if self.condition == "B2":
            weight = valid.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * weight).sum(1, keepdim=True) / weight.sum(1, keepdim=True)
            encoded = pooled.expand(-1, self.num_tokens, -1)
            # No support mask is exposed after pooling: support is not spatial context.
            return encoded, None
        return encoded, ~valid

    def forward(self, features: torch.Tensor, prior: Optional[Mapping] = None,
                reference_mask: Optional[torch.Tensor] = None):
        squeezed = features.ndim == 2
        x = features.unsqueeze(0) if squeezed else features
        if x.ndim != 3:
            raise ValueError("Recipient features must have shape [B,N,D] or [N,D]")
        context, padding = self.encode_prior(prior, x)
        query = self.query_projection(self.query_norm(x))
        if reference_mask is not None:
            mask = torch.as_tensor(reference_mask, dtype=x.dtype, device=x.device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != x.shape[:2]:
                raise ValueError("reference_mask shape must match recipient tokens")
            query = query + mask.unsqueeze(-1) * self.reference_embedding
        attended, _ = self.attention(query, context, context,
                                     key_padding_mask=padding, need_weights=False)
        output = x + torch.tanh(self.gate) * self.output_projection(attended)
        return output.squeeze(0) if squeezed else output


ScaffoldConditioning = ConditioningBlock
