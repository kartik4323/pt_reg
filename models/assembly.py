"""Graph-based fragment assembly model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.compatibility_three import CompatibilityScorer
from models.token_encoder import TokenPointNetEncoder


@dataclass
class AssemblyOutput:
    point_cloud: torch.Tensor
    node_features: torch.Tensor
    refined_features: torch.Tensor
    compatibility_scores: torch.Tensor
    edge_features: torch.Tensor
    fragment_embeddings: torch.Tensor
    spatial_tokens: torch.Tensor
    memory_padding_mask: torch.Tensor


class CompatibilityGNNLayer(nn.Module):
    """Compatibility-weighted residual message passing layer."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.attn = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.update = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        scores: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_nodes, node_dim = nodes.shape
        zi = nodes[:, :, None, :].expand(batch_size, num_nodes, num_nodes, node_dim)
        zj = nodes[:, None, :, :].expand(batch_size, num_nodes, num_nodes, node_dim)
        pair_feat = torch.cat([zi, zj, edges], dim=-1)

        messages = self.message(pair_feat)
        attn_logits = self.attn(pair_feat).squeeze(-1)

        valid_j = node_mask[:, None, :].expand_as(scores)
        not_self = ~torch.eye(num_nodes, dtype=torch.bool, device=nodes.device)[None, :, :]
        valid_edges = valid_j & not_self

        weighted_logits = attn_logits + torch.log(scores.clamp(min=1e-4))
        weighted_logits = weighted_logits.masked_fill(~valid_edges, -1e4)
        weights = torch.softmax(weighted_logits, dim=-1)
        weights = weights * valid_edges.to(weights.dtype)
        aggregated = (weights.unsqueeze(-1) * messages).sum(dim=2)

        update = self.update(torch.cat([nodes, aggregated], dim=-1))
        updated = self.norm(nodes + update)
        return torch.where(node_mask.unsqueeze(-1), updated, nodes)


class CompatibilityGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CompatibilityGNNLayer(node_dim, edge_dim, hidden_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        scores: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            nodes = layer(nodes, edges, scores, node_mask)
        return nodes


class CrossAttentionPointDecoder(nn.Module):
    """Learnable point queries attending to global and local fragment tokens."""

    def __init__(
        self,
        dim: int,
        num_output_points: int = 1024,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_output_points = num_output_points
        self.query_tokens = nn.Parameter(torch.randn(num_output_points, dim) * 0.02)
        self.base_points = nn.Parameter(torch.randn(num_output_points, 3) * 0.15)
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.offset_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 3),
        )

    def forward(
        self,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = memory.shape[0]
        query = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        for attn, norm in zip(self.layers, self.norms):
            attended, _ = attn(
                query=query,
                key=memory,
                value=memory,
                key_padding_mask=memory_padding_mask,
                need_weights=False,
            )
            query = norm(query + attended)

        offsets = self.offset_head(query)
        return self.base_points.unsqueeze(0) + offsets


class FragmentAssemblyModel(nn.Module):
    """
    Stage 2 model:
    token encoder -> compatibility graph -> GNN -> refinement transformer ->
    cross-attention point decoder.
    """

    def __init__(
        self,
        encoder: TokenPointNetEncoder,
        compatibility: CompatibilityScorer,
        num_tokens: int = 16,
        embedding_dim: int = 128,
        edge_dim: int = 128,
        gnn_hidden_dim: int = 256,
        gnn_layers: int = 2,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        decoder_layers: int = 2,
        output_points: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.compatibility = compatibility
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.edge_dim = edge_dim

        self.gnn = CompatibilityGNN(
            node_dim=embedding_dim,
            edge_dim=edge_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=transformer_heads,
            dim_feedforward=gnn_hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.refinement = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_layers,
        )
        self.decoder = CrossAttentionPointDecoder(
            dim=embedding_dim,
            num_output_points=output_points,
            num_heads=transformer_heads,
            num_layers=decoder_layers,
            dropout=dropout,
        )

    def freeze_pretrained(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.compatibility.parameters():
            param.requires_grad = False

    def unfreeze_pretrained(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.compatibility.parameters():
            param.requires_grad = True

    def forward(
        self,
        fragments: torch.Tensor,
        fragment_mask: torch.Tensor,
    ) -> AssemblyOutput:
        if fragments.dim() != 4:
            raise ValueError("fragments must have shape (B, F, N, 3)")

        batch_size, max_fragments, points_per_fragment, _ = fragments.shape
        flat = fragments.reshape(batch_size * max_fragments, points_per_fragment, 3)
        embeddings, spatial_tokens = self._encode_fragment_set(
            flat, batch_size, max_fragments
        )

        scores, edges = self._pairwise_compatibility(embeddings, fragment_mask)
        node_features = self.gnn(embeddings, edges, scores, fragment_mask)
        refined = self.refinement(
            node_features,
            src_key_padding_mask=~fragment_mask,
        )
        refined = torch.where(fragment_mask.unsqueeze(-1), refined, node_features)

        memory = torch.cat(
            [refined, spatial_tokens.reshape(batch_size, max_fragments * self.num_tokens, -1)],
            dim=1,
        )
        token_mask = fragment_mask[:, :, None].expand(
            batch_size, max_fragments, self.num_tokens
        ).reshape(batch_size, max_fragments * self.num_tokens)
        memory_valid = torch.cat([fragment_mask, token_mask], dim=1)
        memory_padding_mask = ~memory_valid
        point_cloud = self.decoder(memory, memory_padding_mask)

        return AssemblyOutput(
            point_cloud=point_cloud,
            node_features=node_features,
            refined_features=refined,
            compatibility_scores=scores,
            edge_features=edges,
            fragment_embeddings=embeddings,
            spatial_tokens=spatial_tokens,
            memory_padding_mask=memory_padding_mask,
        )

    def _pairwise_compatibility(
        self,
        embeddings: torch.Tensor,
        fragment_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_fragments, dim = embeddings.shape
        zi = embeddings[:, :, None, :].expand(batch_size, max_fragments, max_fragments, dim)
        zj = embeddings[:, None, :, :].expand(batch_size, max_fragments, max_fragments, dim)
        flat_i = zi.reshape(batch_size * max_fragments * max_fragments, dim)
        flat_j = zj.reshape(batch_size * max_fragments * max_fragments, dim)
        pair = self.compatibility(flat_i, flat_j)

        scores = pair.score.reshape(batch_size, max_fragments, max_fragments)
        edges = pair.edge_embedding.reshape(
            batch_size, max_fragments, max_fragments, self.edge_dim
        )

        pair_mask = fragment_mask[:, :, None] & fragment_mask[:, None, :]
        eye = torch.eye(max_fragments, dtype=torch.bool, device=embeddings.device)[None, :, :]
        pair_mask = pair_mask & ~eye
        scores = torch.where(pair_mask, scores, torch.zeros_like(scores))
        edges = torch.where(pair_mask.unsqueeze(-1), edges, torch.zeros_like(edges))
        return scores, edges

    def _encode_fragment_set(
        self,
        flat_fragments: torch.Tensor,
        batch_size: int,
        max_fragments: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder(flat_fragments)
        embeddings = enc.embedding.reshape(batch_size, max_fragments, -1)
        spatial_tokens = enc.tokens.reshape(
            batch_size, max_fragments, self.num_tokens, -1
        )
        return embeddings, spatial_tokens

    @torch.no_grad()
    def compatibility_scores_only(
        self,
        fragments: torch.Tensor,
        fragment_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_fragments, points_per_fragment, _ = fragments.shape
        flat = fragments.reshape(batch_size * max_fragments, points_per_fragment, 3)
        embeddings, _ = self._encode_fragment_set(flat, batch_size, max_fragments)
        scores, _ = self._pairwise_compatibility(embeddings, fragment_mask)
        return scores


def build_assembly_model(cfg: dict) -> FragmentAssemblyModel:
    from models.compatibility_three import CompatibilityScorer
    from models.token_encoder import build_token_encoder

    model_cfg = cfg.get("model", {})
    enc_cfg = model_cfg.get("encoder", {})
    compat_cfg = model_cfg.get("compatibility", {})
    assembly_cfg = model_cfg.get("assembly", {})

    encoder = build_token_encoder(enc_cfg)
    compatibility = CompatibilityScorer(
        embedding_dim=enc_cfg.get("embedding_dim", 128),
        hidden_dims=compat_cfg.get("hidden_dims", [256, 128]),
        edge_dim=compat_cfg.get("edge_dim", 128),
        dropout=compat_cfg.get("dropout", 0.0),
    )

    return FragmentAssemblyModel(
        encoder=encoder,
        compatibility=compatibility,
        num_tokens=enc_cfg.get("num_tokens", 16),
        embedding_dim=enc_cfg.get("embedding_dim", 128),
        edge_dim=compat_cfg.get("edge_dim", 128),
        gnn_hidden_dim=assembly_cfg.get("gnn_hidden_dim", 256),
        gnn_layers=assembly_cfg.get("gnn_layers", 2),
        transformer_layers=assembly_cfg.get("transformer_layers", 1),
        transformer_heads=assembly_cfg.get("transformer_heads", 4),
        decoder_layers=assembly_cfg.get("decoder_layers", 2),
        output_points=assembly_cfg.get("output_points", 1024),
        dropout=assembly_cfg.get("dropout", 0.0),
    )
