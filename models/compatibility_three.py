"""
Three-class compatibility module and Stage 1 wrapper.

Class convention:
0 = Direct match
1 = Semantic match
2 = Negative match
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from models.token_encoder import FragmentEncoding, TokenPointNetEncoder


@dataclass
class PairCompatibilityOutput:
    logits: torch.Tensor
    score: torch.Tensor
    edge_embedding: torch.Tensor


class CompatibilityScorer(nn.Module):
    """Pairwise scorer over fragment geometry embeddings."""

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dims: Optional[list[int]] = None,
        edge_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        in_dim = embedding_dim * 4

        layers = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.LayerNorm(hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden

        self.backbone = nn.Sequential(*layers)
        self.edge_head = nn.Sequential(
            nn.Linear(prev, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(edge_dim, 3)
        self.edge_dim = edge_dim

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> PairCompatibilityOutput:
        feat = torch.cat([z_a, z_b, (z_a - z_b).abs(), z_a * z_b], dim=-1)
        hidden = self.backbone(feat)
        edge = self.edge_head(hidden)
        logits = self.classifier(edge)
        probs = logits.softmax(dim=-1)
        direct_score = probs[..., 0]
        return PairCompatibilityOutput(logits=logits, score=direct_score, edge_embedding=edge)


@dataclass
class Stage1Output:
    anchor: FragmentEncoding
    direct: FragmentEncoding
    semantic: FragmentEncoding
    negative: FragmentEncoding
    direct_pair: PairCompatibilityOutput
    semantic_pair: PairCompatibilityOutput
    negative_pair: PairCompatibilityOutput


class Stage1CompatibilityModel(nn.Module):
    """Encoder plus three-class compatibility scorer for triplet pretraining."""

    def __init__(
        self,
        encoder: TokenPointNetEncoder,
        scorer: CompatibilityScorer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.scorer = scorer

    def encode(self, points: torch.Tensor) -> FragmentEncoding:
        return self.encoder(points)

    def score_pair(self, frag_a: torch.Tensor, frag_b: torch.Tensor) -> PairCompatibilityOutput:
        enc_a = self.encoder(frag_a)
        enc_b = self.encoder(frag_b)
        return self.scorer(enc_a.embedding, enc_b.embedding)

    def forward(
        self,
        anchor: torch.Tensor,
        direct: torch.Tensor,
        semantic: torch.Tensor,
        negative: torch.Tensor,
    ) -> Stage1Output:
        enc_anchor = self.encoder(anchor)
        enc_direct = self.encoder(direct)
        enc_semantic = self.encoder(semantic)
        enc_negative = self.encoder(negative)

        direct_pair = self.scorer(enc_anchor.embedding, enc_direct.embedding)
        semantic_pair = self.scorer(enc_anchor.embedding, enc_semantic.embedding)
        negative_pair = self.scorer(enc_anchor.embedding, enc_negative.embedding)

        return Stage1Output(
            anchor=enc_anchor,
            direct=enc_direct,
            semantic=enc_semantic,
            negative=enc_negative,
            direct_pair=direct_pair,
            semantic_pair=semantic_pair,
            negative_pair=negative_pair,
        )


def build_stage1_model(cfg: dict) -> Stage1CompatibilityModel:
    from models.token_encoder import build_token_encoder

    model_cfg = cfg.get("model", {})
    enc_cfg = model_cfg.get("encoder", {})
    compat_cfg = model_cfg.get("compatibility", {})

    encoder = build_token_encoder(enc_cfg)
    scorer = CompatibilityScorer(
        embedding_dim=enc_cfg.get("embedding_dim", 128),
        hidden_dims=compat_cfg.get("hidden_dims", [256, 128]),
        edge_dim=compat_cfg.get("edge_dim", 128),
        dropout=compat_cfg.get("dropout", 0.0),
    )
    return Stage1CompatibilityModel(encoder, scorer)


def load_stage1_weights(
    model: Stage1CompatibilityModel,
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.encoder.load_state_dict(checkpoint["encoder"], strict=strict)
    model.scorer.load_state_dict(checkpoint["compatibility"], strict=strict)
    return checkpoint
