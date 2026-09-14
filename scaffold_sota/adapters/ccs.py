"""Native CCS WXTransformer adapter, retaining its shared-memory correlator.

Native output slots remain at the configured 20-part padding budget so top-10
attention retains its original valid tensor dimensions for two/three-part data.
Only valid fragments enter the encoder and leave the study's pose interface.
"""
from __future__ import annotations

import torch
from torch import nn

from ..models.conditioning import ConditioningBlock
from .matching_common import (inference_seed, input_parts, load_config_file,
                              native_import_root, pose_result)


class CCSBackend(nn.Module):
    conditioning_parameter_prefixes = ("conditioning.",)

    def __init__(self, native, options, rotation_class):
        super().__init__()
        self.native, self.options = native, dict(options)
        self._rotation_class = rotation_class
        self.condition = self.options.get("condition", "native")
        if self.condition not in {"native", "B0", "B1", "B2", "B3", "B4"}:
            raise ValueError(f"Unknown condition {self.condition}")
        self.conditioning = None
        if self.condition not in {"native", "B0"}:
            self.conditioning = ConditioningBlock(
                native.pc_feat_dim, self.condition,
                token_dim=int(options.get("token_dim", 128)),
                heads=int(options.get("heads", 4)),
                num_tokens=int(options.get("num_tokens", 512)))
        self.native_config = {
            "architecture": "WXTransformer",
            "encoder": str(native.cfg.model.encoder),
            "native_max_parts": int(native.max_num_part),
            "topk": int(native.cfg.model.topk),
            "noise_dim": int(native.cfg.loss.noise_dim),
            "sample_iter": 1,
            "prediction_alignment": "input_anchor_relative",
            "memory_policy": "reset per independent object",
        }

    @property
    def device(self):
        return next(self.native.parameters()).device

    def _data(self, sample):
        points, indices, anchor = input_parts(sample, self.device)
        count, size, _ = points.shape
        budget = self.native.max_num_part
        if count > budget:
            raise ValueError("Native CCS padding budget is smaller than fragment count")
        padded = torch.zeros((1, budget, size, 3), device=self.device, dtype=points.dtype)
        padded[:, :count] = points
        valids = torch.zeros((1, budget), device=self.device)
        valids[:, :count] = 1
        data = {"part_pcs": padded, "part_valids": valids,
                "part_label": padded.new_zeros((1, budget, 0)),
                "instance_label": padded.new_zeros((1, budget, 0)),
                "part_curs": None}
        return data, indices, anchor

    def _forward(self, data, anchor, prior):
        # The upstream correlator keeps transient memory on attention modules.
        # Independent objects must never inherit a previous object's graph/state.
        for module in self.native.corr_module.modules():
            if hasattr(module, "relational_memory") and hasattr(module, "memory"):
                module.memory = None
        feats = self.native._extract_part_feats(data["part_pcs"], data["part_valids"])
        if self.conditioning is not None:
            reference = torch.zeros_like(data["part_valids"])
            reference[:, anchor] = 1
            feats = self.conditioning(feats, prior, reference)
            # Do not add features to native padded slots: its internal memory
            # implementation does not consistently propagate key-padding masks.
            feats = feats * data["part_valids"].unsqueeze(-1)
        corr = self.native.corr_module(feats, data["part_valids"] == 1)
        cached = torch.cat((corr, data["part_label"], data["instance_label"]), dim=-1)
        # Execute the native pose head through its supported cached-feature path.
        return self.native({**data, "pre_pose_feats": cached})

    def loss(self, sample, prior=None):
        data, indices, anchor = self._data(sample)
        count, budget = indices.numel(), self.native.max_num_part
        rotations = torch.eye(3, device=self.device).reshape(1, 1, 3, 3).repeat(1, budget, 1, 1)
        translations = torch.zeros((1, budget, 3), device=self.device)
        rotations[:, :count] = torch.as_tensor(sample["rotations_gt"], device=self.device)[indices]
        translations[:, :count] = torch.as_tensor(sample["translations_gt"], device=self.device)[indices]
        data["part_rot"] = self._rotation_class(rotations, rot_type="rmat").convert(self.native.rot_type)
        data["part_trans"] = translations
        prediction = self._forward(data, anchor, prior)
        losses, _ = self.native._calc_loss(prediction, data, C=self.native.cfg.loss.collision_loss_C)
        result = {key: value.mean() for key, value in losses.items() if key.endswith("_loss")}
        total = sum(value * float(getattr(self.native.cfg.loss, f"{key}_w"))
                    for key, value in result.items())
        if not isinstance(total, torch.Tensor) or not torch.isfinite(total):
            raise RuntimeError("Native CCS produced a missing or nonfinite loss")
        result["loss"] = total
        return result

    @torch.no_grad()
    def predict(self, sample, prior=None, seed=0):
        previous_training = self.training
        self.eval()
        try:
            with inference_seed(seed, self.device):
                data, indices, anchor = self._data(sample)
                prediction = self._forward(data, anchor, prior)
                count = indices.numel()
                rotations = prediction["rot"].to_rmat()[0, :count]
                translations = prediction["trans"][0, :count]
                return pose_result(rotations, translations, anchor, native_samples=1)
        finally:
            self.train(previous_training)


def build(source_root, options=None, device="cpu"):
    """Build the pinned shared-memory CCS network; never substitute a toy model."""
    options = dict(options or {})
    root = native_import_root(source_root, ("multi_part_assembly",))
    config_path = root / "configs/wx_transformer/wx_transformer/topk/everyday/top10-everyday.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"Not a CCS native source tree: {root}")
    config = load_config_file(config_path, "scaffold_sota_ccs_native_config")
    config.data.max_num_part = int(options.get("native_max_parts", 20))
    if config.data.max_num_part < max(3, config.model.topk):
        raise ValueError("CCS padded slots must accommodate native top-k attention")
    config.loss.noise_dim = 0
    config.loss.sample_iter = 1
    from multi_part_assembly.models.wx_transformer.network import WXTransformer
    from multi_part_assembly.utils.rotation import Rotation3D
    native = WXTransformer(config)
    return CCSBackend(native, options, Rotation3D).to(device)
