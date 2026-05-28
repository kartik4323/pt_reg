"""Binary compatibility module and Stage 1 wrapper."""

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
        self.classifier = nn.Linear(edge_dim, 1)
        self.edge_dim = edge_dim

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> PairCompatibilityOutput:
        feat = torch.cat([z_a, z_b, (z_a - z_b).abs(), z_a * z_b], dim=-1)
        hidden = self.backbone(feat)
        edge = self.edge_head(hidden)
        logits = self.classifier(edge).squeeze(-1)
        score = torch.sigmoid(logits)
        return PairCompatibilityOutput(logits=logits, score=score, edge_embedding=edge)


@dataclass
class Stage1Output:
    frag_a: FragmentEncoding
    frag_b: FragmentEncoding
    pair: PairCompatibilityOutput


class Stage1CompatibilityModel(nn.Module):
    """Encoder plus binary compatibility scorer for pair pretraining."""

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
        frag_a: torch.Tensor,
        frag_b: torch.Tensor,
    ) -> Stage1Output:
        enc_a = self.encoder(frag_a)
        enc_b = self.encoder(frag_b)
        pair = self.scorer(enc_a.embedding, enc_b.embedding)

        return Stage1Output(
            frag_a=enc_a,
            frag_b=enc_b,
            pair=pair,
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
