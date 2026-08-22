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
    interaction_features: torch.Tensor
    fused_features: torch.Tensor
    refined_features: torch.Tensor
    compatibility_scores: torch.Tensor
    edge_features: torch.Tensor
    fragment_embeddings: torch.Tensor
    spatial_tokens: torch.Tensor
    memory_padding_mask: torch.Tensor
    # Occupancy head (``decoder: "occupancy"``). ``occupancy_logits`` is evaluated at
    # ``query_xyz`` during training (a random subset, so cost is independent of
    # resolution) or on a full grid at inference, in which case ``occupancy_grid`` is
    # populated and ``point_cloud`` is sampled from its surface.
    occupancy_logits: Optional[torch.Tensor] = None
    occupancy_confidence: Optional[torch.Tensor] = None
    query_xyz: Optional[torch.Tensor] = None
    occupancy_grid: Optional[torch.Tensor] = None


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


def _fourier_encode(coords: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """(..., 3) -> (..., 3 + 6*num_freqs) sinusoidal positional encoding."""
    feats = [coords]
    for i in range(num_freqs):
        freq = float(2 ** i) * torch.pi
        feats.append(torch.sin(coords * freq))
        feats.append(torch.cos(coords * freq))
    return torch.cat(feats, dim=-1)


class ImplicitOccupancyDecoder(nn.Module):
    """Coordinate-queried occupancy decoder: (query xyz, fragment memory) -> occupancy.

    Chosen over a 3D-deconv head for three reasons:

    * **Resolution is a runtime argument.** One trained model can be queried at
      8^3 ... 64^3, so the target-fidelity ablation costs nothing extra.
    * **Training cost is independent of resolution** -- we evaluate a random subset
      of query points per step (the standard implicit-field recipe), instead of
      materializing a full grid. A full 32^3 grid would need a
      (B*heads, 32768, memory) attention tensor, which does not fit.
    * It reuses the existing cross-attention-to-``memory`` pattern, so the whole
      Stage-2 trunk is untouched.

    Emits an occupancy logit and a **confidence** logit per query; the confidence is
    what lets Stage 3 ignore regions where the coarse prior is unreliable.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        num_freqs: int = 6,
    ) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        self.query_proj = nn.Sequential(
            nn.Linear(3 + 6 * num_freqs, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
                )
                for _ in range(num_layers)
            ]
        )
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.occupancy_head = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim // 2), nn.SiLU(), nn.Linear(dim // 2, 1)
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.SiLU(), nn.Linear(dim // 2, 1)
        )

    def forward(
        self,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        query_xyz: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """memory (B,S,D), mask (B,S), query_xyz (B,Q,3) -> occupancy/confidence (B,Q)."""
        query = self.query_proj(_fourier_encode(query_xyz, self.num_freqs))
        for attn, norm, ffn, ffn_norm in zip(self.layers, self.norms, self.ffns, self.ffn_norms):
            attended, _ = attn(
                query=query,
                key=memory,
                value=memory,
                key_padding_mask=memory_padding_mask,
                need_weights=False,
            )
            query = norm(query + attended)
            query = ffn_norm(query + ffn(query))
        return self.occupancy_head(query).squeeze(-1), self.confidence_head(query).squeeze(-1)


def grid_query_points(
    resolution: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Voxel-centre coordinates of an R^3 grid over [-1,1]^3 -> (R^3, 3), C-order."""
    lin = (torch.arange(resolution, device=device, dtype=dtype) + 0.5) / resolution * 2.0 - 1.0
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)


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
        decoder: str = "points",
        occupancy_resolution: int = 32,
        occupancy_num_freqs: int = 6,
        disable_compatibility: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.compatibility = compatibility
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.edge_dim = edge_dim
        # An explicit architecture ablation.  Keeping the modules in the state
        # dict makes checkpoints load-compatible, while the forward path below
        # proves that no compatibility score, edge embedding, or GNN message is
        # consumed when this switch is enabled.
        self.disable_compatibility = bool(disable_compatibility)
        self.interaction_fusion = nn.Sequential(
            nn.Linear(embedding_dim + edge_dim, gnn_hidden_dim),
            nn.LayerNorm(gnn_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )

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
        self.decoder_type = decoder
        if decoder == "occupancy":
            self.occupancy_resolution = occupancy_resolution
            self.decoder = ImplicitOccupancyDecoder(
                dim=embedding_dim,
                num_heads=transformer_heads,
                num_layers=decoder_layers,
                dropout=dropout,
                num_freqs=occupancy_num_freqs,
            )
        elif decoder == "points":
            self.occupancy_resolution = occupancy_resolution
            self.decoder = CrossAttentionPointDecoder(
                dim=embedding_dim,
                num_output_points=output_points,
                num_heads=transformer_heads,
                num_layers=decoder_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown Stage-2 decoder: {decoder!r} (expected 'points' or 'occupancy')")
        self.output_points = output_points

    @torch.no_grad()
    def predict_occupancy_grid(
        self,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        resolution: Optional[int] = None,
        chunk: int = 8192,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the implicit field on a full R^3 grid, chunked. -> (B,R,R,R) x2.

        Resolution is a free parameter of a *trained* model, which is what makes the
        target-fidelity sweep cheap.
        """
        if self.decoder_type != "occupancy":
            raise RuntimeError("predict_occupancy_grid requires decoder='occupancy'")
        res = int(resolution or self.occupancy_resolution)
        batch = memory.shape[0]
        queries = grid_query_points(res, memory.device, memory.dtype)      # (R^3, 3)
        occ_parts, conf_parts = [], []
        for start in range(0, queries.shape[0], chunk):
            block = queries[start : start + chunk].unsqueeze(0).expand(batch, -1, -1)
            occ, conf = self.decoder(memory, memory_padding_mask, block)
            occ_parts.append(occ)
            conf_parts.append(conf)
        occ = torch.cat(occ_parts, dim=1).reshape(batch, res, res, res)
        conf = torch.cat(conf_parts, dim=1).reshape(batch, res, res, res)
        return occ, conf

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
        query_xyz: Optional[torch.Tensor] = None,
        occupancy_resolution: Optional[int] = None,
    ) -> AssemblyOutput:
        """``query_xyz`` (B,Q,3) evaluates the implicit field at those points only --
        that is the training path, and its cost is independent of grid resolution.
        With ``query_xyz=None`` and the occupancy decoder, a full grid is evaluated and
        ``point_cloud`` is sampled from the predicted surface so that every existing
        point-cloud consumer keeps working."""
        if fragments.dim() != 4:
            raise ValueError("fragments must have shape (B, F, N, 3)")

        batch_size, max_fragments, points_per_fragment, _ = fragments.shape
        flat = fragments.reshape(batch_size * max_fragments, points_per_fragment, 3)
        embeddings, spatial_tokens = self._encode_fragment_set(
            flat, batch_size, max_fragments
        )

        if self.disable_compatibility:
            scores = embeddings.new_zeros(batch_size, max_fragments, max_fragments)
            edges = embeddings.new_zeros(batch_size, max_fragments, max_fragments, self.edge_dim)
            interaction_features = embeddings.new_zeros(batch_size, max_fragments, self.edge_dim)
            # Do not pass zero tensors through trainable compatibility fusion or
            # GNN layers: that would still let their biases carry graph-path
            # information in the supposedly disabled condition.
            fused_features = embeddings
            node_features = embeddings
        else:
            scores, edges = self._pairwise_compatibility(embeddings, fragment_mask)
            interaction_features = self._aggregate_interactions(edges, scores, fragment_mask)
            fused_features = self.interaction_fusion(
                torch.cat([embeddings, interaction_features], dim=-1)
            )
            fused_features = torch.where(fragment_mask.unsqueeze(-1), fused_features, embeddings)
            node_features = self.gnn(fused_features, edges, scores, fragment_mask)
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

        occupancy_logits = occupancy_confidence = occupancy_grid = None
        if self.decoder_type == "points":
            point_cloud = self.decoder(memory, memory_padding_mask)
        elif query_xyz is not None:
            # Training path: evaluate only the sampled queries.
            occupancy_logits, occupancy_confidence = self.decoder(
                memory, memory_padding_mask, query_xyz
            )
            # A real point cloud would need the full grid; keep this cheap and let the
            # loss/consumers use the occupancy fields during training.
            point_cloud = fragments.new_zeros(batch_size, 1, 3)
        else:
            # Inference path: full grid, then sample the predicted surface.
            res = int(occupancy_resolution or self.occupancy_resolution)
            grid_logits, grid_conf = self.predict_occupancy_grid(
                memory, memory_padding_mask, resolution=res
            )
            occupancy_grid = grid_logits
            occupancy_confidence = grid_conf
            occupancy_logits = grid_logits.reshape(batch_size, -1)
            query_xyz = grid_query_points(res, fragments.device, fragments.dtype).unsqueeze(0).expand(
                batch_size, -1, -1
            )
            point_cloud = self._sample_surface_points(grid_logits, self.output_points)

        return AssemblyOutput(
            point_cloud=point_cloud,
            node_features=node_features,
            interaction_features=interaction_features,
            fused_features=fused_features,
            refined_features=refined,
            compatibility_scores=scores,
            edge_features=edges,
            fragment_embeddings=embeddings,
            spatial_tokens=spatial_tokens,
            memory_padding_mask=memory_padding_mask,
            occupancy_logits=occupancy_logits,
            occupancy_confidence=occupancy_confidence,
            query_xyz=query_xyz,
            occupancy_grid=occupancy_grid,
        )

    @staticmethod
    def _sample_surface_points(grid_logits: torch.Tensor, num_points: int) -> torch.Tensor:
        """Sample points from the BOUNDARY of a predicted occupancy grid -> (B,P,3).

        The boundary, not the filled interior: interior samples lie on no real
        surface, so registration / FPFH / chamfer against a surface point cloud
        would be meaningless.
        """
        batch, res = grid_logits.shape[0], grid_logits.shape[1]
        occ = (grid_logits > 0)
        # boundary = occupied with at least one empty 6-neighbour
        empty_neighbour = torch.zeros_like(occ)
        for axis in (1, 2, 3):
            for shift in (-1, 1):
                rolled = torch.roll(occ, shifts=shift, dims=axis)
                idx = [slice(None)] * 4
                idx[axis] = 0 if shift == 1 else res - 1
                rolled[tuple(idx)] = False
                empty_neighbour |= ~rolled
        surface = occ & empty_neighbour

        pitch = 2.0 / res
        centers = grid_query_points(res, grid_logits.device, grid_logits.dtype)   # (R^3,3)
        flat = surface.reshape(batch, -1)
        out = grid_logits.new_zeros(batch, num_points, 3)
        for b in range(batch):
            idx = torch.nonzero(flat[b], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                idx = torch.nonzero(occ[b].reshape(-1), as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            pick = idx[torch.randint(0, idx.numel(), (num_points,), device=idx.device)]
            jitter = (torch.rand(num_points, 3, device=out.device, dtype=out.dtype) - 0.5) * pitch
            out[b] = centers[pick] + jitter
        return out

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

    def _aggregate_interactions(
        self,
        edges: torch.Tensor,
        scores: torch.Tensor,
        fragment_mask: torch.Tensor,
    ) -> torch.Tensor:
        pair_mask = fragment_mask[:, :, None] & fragment_mask[:, None, :]
        eye = torch.eye(fragment_mask.shape[1], dtype=torch.bool, device=fragment_mask.device)[None, :, :]
        pair_mask = pair_mask & ~eye
        weights = torch.where(pair_mask, scores, torch.zeros_like(scores))
        denom = weights.sum(dim=2, keepdim=True).clamp(min=1.0e-6)
        aggregated = (edges * weights.unsqueeze(-1)).sum(dim=2) / denom
        return torch.where(fragment_mask.unsqueeze(-1), aggregated, torch.zeros_like(aggregated))

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
        decoder=assembly_cfg.get("decoder", "points"),
        occupancy_resolution=assembly_cfg.get("occupancy_resolution", 32),
        occupancy_num_freqs=assembly_cfg.get("occupancy_num_freqs", 6),
        disable_compatibility=assembly_cfg.get("disable_compatibility", False),
    )
