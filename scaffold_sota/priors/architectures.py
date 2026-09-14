"""State-compatible Stage 2 fields for both Torch 1.10 and modern runtimes.

Only constructor compatibility differs from the read-only original field. Its
attention, field equations and state-dict names are preserved exactly.
"""
from __future__ import annotations

import inspect

from torch import nn
from reassembly.model import ShapeField as OriginalShapeField, mlp


class ShapeField(OriginalShapeField):
    def __init__(self, config):
        nn.Module.__init__(self)
        dim = int(config.get("dim", 128))
        heads = int(config.get("heads", 4))
        self.truncation = float(config.get("truncation", .1))
        self.log_min = float(config.get("log_scale_min", -6.))
        self.log_max = float(config.get("log_scale_max", -2.))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0., batch_first=True, norm_first=True)
        extra = {"enable_nested_tensor": False} if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoder).parameters else {}
        self.context_attention = nn.TransformerEncoder(layer, int(config.get("attention_layers", 2)), **extra)
        self.fragment_context = nn.Linear(dim, dim)
        self.query = mlp(dim + 3, dim)
        self.key = nn.Linear(dim, dim)
        self.relative_bias = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))
        self.decoder = nn.Sequential(nn.Linear(dim * 2 + 4, dim), nn.GELU(), nn.Linear(dim, 2))


class AdaptedShapeField(ShapeField):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = mlp(int(config.get("dim", 128)), int(config.get("dim", 128)))

    def context(self, encoded):
        adapted = dict(encoded)
        adapted["token_features"] = encoded["token_features"] + self.adapter(encoded["token_features"])
        return super().context(adapted)
