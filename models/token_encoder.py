"""
Token-preserving point-cloud encoder.

The research design calls for a fragment encoder that produces both a pooled
geometry embedding and a set of spatial tokens. This implementation is a
lightweight PointNet-style encoder with farthest-point token selection. It is
dependency-light and therefore useful for CPU smoke tests; it can be replaced
by an SE(3)-equivariant backbone later while keeping the same output contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class FragmentEncoding:
    embedding: torch.Tensor
    tokens: torch.Tensor
    token_xyz: torch.Tensor
    point_features: torch.Tensor


def _batched_fps_indices(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Farthest point sampling indices for batched point clouds."""
    batch_size, num_points, _ = points.shape
    if num_points <= 0:
        raise ValueError("points must contain at least one point")

    if num_samples <= num_points:
        out_count = num_samples
    else:
        out_count = num_points

    centroids = torch.zeros(batch_size, out_count, dtype=torch.long, device=points.device)
    distance = torch.full((batch_size, num_points), float("inf"), device=points.device)
    farthest = torch.linalg.vector_norm(
        points - points.mean(dim=1, keepdim=True), dim=-1
    ).max(dim=1).indices
    batch_indices = torch.arange(batch_size, device=points.device)

    for i in range(out_count):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest].view(batch_size, 1, 3)
        dist = ((points - centroid) ** 2).sum(dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(dim=1).indices

    if num_samples > num_points:
        repeats = num_samples - num_points
        pad = centroids[:, :1].expand(batch_size, repeats)
        centroids = torch.cat([centroids, pad], dim=1)

    return centroids


def _gather_points(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    index = indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    return values.gather(dim=1, index=index)


class TokenPointNetEncoder(nn.Module):
    """
    Encode a point-cloud fragment into a pooled embedding and K spatial tokens.

    Parameters
    ----------
    embedding_dim:
        Dimension of pooled and token features.
    num_tokens:
        Number of spatial tokens retained per fragment.
    hidden_dim:
        Width of the per-point MLP.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_tokens: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens

        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        self.token_refine = nn.Sequential(
            nn.Linear(embedding_dim + 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.summary = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, points: torch.Tensor) -> FragmentEncoding:
        """
        Parameters
        ----------
        points:
            Tensor of shape (B, N, 3).
        """
        if points.dim() != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape (B, N, 3)")

        point_features = self.point_mlp(points)
        token_indices = _batched_fps_indices(points.detach(), self.num_tokens)
        token_xyz = _gather_points(points, token_indices)
        token_features = _gather_points(point_features, token_indices)
        tokens = self.token_refine(torch.cat([token_features, token_xyz], dim=-1))
        embedding = self.summary(tokens.mean(dim=1))
        return FragmentEncoding(
            embedding=embedding,
            tokens=tokens,
            token_xyz=token_xyz,
            point_features=point_features,
        )

    def encode_embedding(self, points: torch.Tensor) -> torch.Tensor:
        return self(points).embedding


def build_token_encoder(cfg: Optional[dict] = None) -> TokenPointNetEncoder:
    cfg = cfg or {}
    return TokenPointNetEncoder(
        embedding_dim=cfg.get("embedding_dim", 128),
        num_tokens=cfg.get("num_tokens", 16),
        hidden_dim=cfg.get("hidden_dim", 128),
        dropout=cfg.get("dropout", 0.0),
    )
