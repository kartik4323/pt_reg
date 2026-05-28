"""Token-preserving point-cloud encoders.

The assembly pipeline needs two outputs per fragment: a pooled geometry
embedding and a set of spatial tokens. The default encoder below is an
SE(3)-safe scalar encoder:

* learned scalar features are invariant to global translation and rotation;
* `token_xyz` is centered relative geometry, so it is translation-invariant and
  rotation-equivariant.

This is lighter than a full e3nn tensor-field network, but it fixes the key
failure mode of raw-XYZ PointNet: arbitrary fragment pose no longer changes the
compatibility embedding.
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


class SE3InvariantTokenEncoder(nn.Module):
    """
    Token encoder whose learned features are SE(3)-invariant scalars.

    The encoder builds per-point descriptors from centered radial distances and
    local kNN distance statistics. Since these quantities depend only on pairwise
    distances, global rotations and translations do not change the learned
    embeddings or token features. The sampled token coordinates are returned as
    centered vectors, which rotate equivariantly with the input.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_tokens: int = 16,
        hidden_dim: int = 128,
        num_neighbors: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.num_neighbors = num_neighbors
        self.scalar_dim = 8

        self.point_mlp = nn.Sequential(
            nn.Linear(self.scalar_dim, hidden_dim),
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
            nn.Linear(embedding_dim + self.scalar_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        self.summary = nn.Sequential(
            nn.Linear(embedding_dim * 2 + self.scalar_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, points: torch.Tensor) -> FragmentEncoding:
        if points.dim() != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape (B, N, 3)")

        centered = points - points.mean(dim=1, keepdim=True)
        scalar_features = self._scalar_features(centered)
        point_features = self.point_mlp(scalar_features)

        token_indices = _batched_fps_indices(centered.detach(), self.num_tokens)
        token_xyz = _gather_points(centered, token_indices)
        token_point_features = _gather_points(point_features, token_indices)
        token_scalars = _gather_points(scalar_features, token_indices)
        tokens = self.token_refine(torch.cat([token_point_features, token_scalars], dim=-1))

        token_mean = tokens.mean(dim=1)
        token_max = tokens.max(dim=1).values
        scalar_summary = scalar_features.mean(dim=1)
        embedding = self.summary(torch.cat([token_mean, token_max, scalar_summary], dim=-1))

        return FragmentEncoding(
            embedding=embedding,
            tokens=tokens,
            token_xyz=token_xyz,
            point_features=point_features,
        )

    def encode_embedding(self, points: torch.Tensor) -> torch.Tensor:
        return self(points).embedding

    def _scalar_features(self, centered: torch.Tensor) -> torch.Tensor:
        radius = torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
        radius_sq = radius.square()
        local_min, local_mean, local_std, local_max = self._local_distance_stats(centered)
        global_mean = radius.mean(dim=1, keepdim=True).expand_as(radius)
        global_std = radius.std(dim=1, keepdim=True, unbiased=False).expand_as(radius)
        return torch.cat(
            [
                radius,
                radius_sq,
                local_min,
                local_mean,
                local_std,
                local_max,
                global_mean,
                global_std,
            ],
            dim=-1,
        )

    def _local_distance_stats(
        self,
        centered: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_points, _ = centered.shape
        if num_points <= 1:
            zeros = torch.zeros(batch_size, num_points, 1, device=centered.device, dtype=centered.dtype)
            return zeros, zeros, zeros, zeros

        k = min(self.num_neighbors + 1, num_points)
        dist = torch.cdist(centered, centered, p=2)
        nearest = dist.topk(k=k, dim=-1, largest=False).values[..., 1:]
        if nearest.shape[-1] == 0:
            zeros = torch.zeros(batch_size, num_points, 1, device=centered.device, dtype=centered.dtype)
            return zeros, zeros, zeros, zeros

        local_min = nearest.min(dim=-1, keepdim=True).values
        local_mean = nearest.mean(dim=-1, keepdim=True)
        local_std = nearest.std(dim=-1, keepdim=True, unbiased=False)
        local_max = nearest.max(dim=-1, keepdim=True).values
        return local_min, local_mean, local_std, local_max


class SE3TransformerTokenEncoder(nn.Module):
    """
    Optional graph SE(3) encoder backed by torch-geometric radius graphs and
    e3nn spherical harmonics.

    This backend is loaded only when selected in config so the default install
    remains lightweight. It returns the same FragmentEncoding contract as the
    other encoders.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_tokens: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 3,
        max_radius: float = 0.3,
        num_basis: int = 8,
        lmax: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        try:
            import e3nn.o3 as o3  # noqa: F401
            from torch_geometric.nn import radius_graph  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "encoder.type='se3_transformer' requires optional dependencies "
                "e3nn and torch-geometric. Install them before selecting this backend."
            ) from exc

        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.max_radius = max_radius
        self.num_basis = num_basis
        self.lmax = lmax
        self.scalar_dim = 4 + num_basis + (lmax + 1)

        layers = []
        prev = self.scalar_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(prev, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden_dim
        layers.extend(
            [
                nn.Linear(prev, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.SiLU(),
            ]
        )
        self.point_mlp = nn.Sequential(*layers)
        self.token_refine = nn.Sequential(
            nn.Linear(embedding_dim + self.scalar_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        self.summary = nn.Sequential(
            nn.Linear(embedding_dim * 2 + self.scalar_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, points: torch.Tensor) -> FragmentEncoding:
        if points.dim() != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape (B, N, 3)")

        centered = points - points.mean(dim=1, keepdim=True)
        scalar_features = self._graph_scalar_features(centered)
        point_features = self.point_mlp(scalar_features)

        token_indices = _batched_fps_indices(centered.detach(), self.num_tokens)
        token_xyz = _gather_points(centered, token_indices)
        token_point_features = _gather_points(point_features, token_indices)
        token_scalars = _gather_points(scalar_features, token_indices)
        tokens = self.token_refine(torch.cat([token_point_features, token_scalars], dim=-1))

        token_mean = tokens.mean(dim=1)
        token_max = tokens.max(dim=1).values
        scalar_summary = scalar_features.mean(dim=1)
        embedding = self.summary(torch.cat([token_mean, token_max, scalar_summary], dim=-1))
        return FragmentEncoding(
            embedding=embedding,
            tokens=tokens,
            token_xyz=token_xyz,
            point_features=point_features,
        )

    def encode_embedding(self, points: torch.Tensor) -> torch.Tensor:
        return self(points).embedding

    def _graph_scalar_features(self, centered: torch.Tensor) -> torch.Tensor:
        import e3nn.o3 as o3
        from torch_geometric.nn import radius_graph

        batch_size, num_points, _ = centered.shape
        flat = centered.reshape(batch_size * num_points, 3)
        batch = torch.arange(batch_size, device=centered.device).repeat_interleave(num_points)
        edge_index = radius_graph(flat, r=self.max_radius, batch=batch, loop=False)

        radius = torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
        radius_sq = radius.square()
        graph_features = torch.zeros(
            batch_size * num_points,
            self.num_basis + self.lmax + 1,
            device=centered.device,
            dtype=centered.dtype,
        )
        counts = torch.zeros(batch_size * num_points, 1, device=centered.device, dtype=centered.dtype)

        if edge_index.numel() > 0:
            src, dst = edge_index
            edge_vec = flat[src] - flat[dst]
            dist = torch.linalg.vector_norm(edge_vec, dim=-1)
            rbf = self._gaussian_rbf(dist)
            sh = o3.spherical_harmonics(
                list(range(self.lmax + 1)),
                edge_vec,
                normalize=True,
                normalization="component",
            )
            sh_parts = []
            offset = 0
            for degree in range(self.lmax + 1):
                width = 2 * degree + 1
                sh_parts.append(sh[:, offset : offset + width].square().sum(dim=-1, keepdim=True).sqrt())
                offset += width
            edge_features = torch.cat([rbf, *sh_parts], dim=-1)
            graph_features.index_add_(0, dst, edge_features)
            ones = torch.ones(dst.shape[0], 1, device=centered.device, dtype=centered.dtype)
            counts.index_add_(0, dst, ones)

        graph_features = graph_features / counts.clamp(min=1.0)
        graph_features = graph_features.reshape(batch_size, num_points, -1)
        return torch.cat([radius, radius_sq, radius.mean(dim=1, keepdim=True).expand_as(radius), radius.std(dim=1, keepdim=True, unbiased=False).expand_as(radius), graph_features], dim=-1)

    def _gaussian_rbf(self, dist: torch.Tensor) -> torch.Tensor:
        centers = torch.linspace(
            0.0,
            self.max_radius,
            self.num_basis,
            device=dist.device,
            dtype=dist.dtype,
        )
        width = self.max_radius / max(self.num_basis, 1)
        return torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / (2 * width ** 2))


def build_token_encoder(cfg: Optional[dict] = None) -> nn.Module:
    cfg = cfg or {}
    enc_type = cfg.get("type", "se3_invariant")
    common = {
        "embedding_dim": cfg.get("embedding_dim", 128),
        "num_tokens": cfg.get("num_tokens", 16),
        "hidden_dim": cfg.get("hidden_dim", 128),
        "dropout": cfg.get("dropout", 0.0),
    }
    if enc_type in {"se3_invariant", "se3_equivariant", "se3"}:
        return SE3InvariantTokenEncoder(
            **common,
            num_neighbors=cfg.get("num_neighbors", 16),
        )
    if enc_type in {"se3_transformer", "e3nn"}:
        return SE3TransformerTokenEncoder(
            **common,
            num_layers=cfg.get("num_layers", 3),
            max_radius=cfg.get("max_radius", 0.3),
            num_basis=cfg.get("num_basis", 8),
            lmax=cfg.get("lmax", 2),
        )
    if enc_type in {"token_pointnet", "pointnet"}:
        return TokenPointNetEncoder(**common)
    raise ValueError(f"Unknown encoder type: {enc_type}")
