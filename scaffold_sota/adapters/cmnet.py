"""Native CMNet with an explicitly invariant pooled scaffold context arm.

Spatial B3/B4 are intentionally not advertised: they require a separately
designed pose-aware pathway. B2 summarizes field distance against radius from
the anchor origin; arbitrary XYZ is never concatenated to equivariant features.
"""
from __future__ import annotations

import itertools
import torch
from torch import nn

from ..models.conditioning import ConditioningBlock
from .matching_common import failed, inference_seed, input_parts, native_import_root
from .matching_pairwise import load_translation_solver, native_pose_graph, supervised_pair

SUPPORTED_CONDITIONS = ("native", "B0", "B1", "B2")


def invariant_radial_prior(prior):
    if prior is None:
        return None
    xyz = torch.as_tensor(prior["query_xyz"])
    radial = torch.zeros_like(xyz)
    radial[..., 0] = torch.linalg.vector_norm(xyz, dim=-1)
    return {**prior, "query_xyz": radial}


class CMNetBackend(nn.Module):
    conditioning_parameter_prefixes = ("conditioning.",)
    supported_conditions = SUPPORTED_CONDITIONS

    def __init__(self, native, options, translation_solver):
        super().__init__()
        self.native, self.options = native, dict(options)
        self.condition = options.get("condition", "native")
        if self.condition not in SUPPORTED_CONDITIONS:
            raise ValueError("CMNet supports native/B0/B1/B2 only; spatial B3/B4 need a pose-aware invariant design")
        self.conditioning = None
        self._active_prior = None
        self._translation_solver = translation_solver
        if self.condition not in {"native", "B0"}:
            self.conditioning = ConditioningBlock(1023, self.condition,
                token_dim=int(options.get("token_dim", 128)), heads=int(options.get("heads", 4)),
                num_tokens=int(options.get("num_tokens", 512)))
            self._conditioning_hook = self.native.c_attn.register_forward_pre_hook(self._condition_attention)
        self.native_config = {"architecture": "CMNet", "context": "invariant_radial_field_pool",
                              "supported_conditions": list(SUPPORTED_CONDITIONS),
                              "graph_policy": "native_highest_score_edge_per_fragment",
                              "graph_initialization": "predicted_anchor_pairs",
                              "training_pair_policy": "native_random_source_max_overlap_target"}

    @property
    def device(self):
        return next(self.native.parameters()).device

    def _condition_attention(self, module, args):
        features = args[0].transpose(1, 2)
        prior = None if self.condition == "B1" else invariant_radial_prior(self._active_prior)
        conditioned = self.conditioning(features, prior)
        return (conditioned.transpose(1, 2),) + args[1:]

    def _pair(self, points, i, j, prior, mode):
        self._active_prior = prior
        try:
            return self.native({"pcd_t": [points[i:i + 1], points[j:j + 1]]}, mode=mode)
        finally:
            self._active_prior = None

    def loss(self, sample, prior=None):
        points, indices, _ = input_parts(sample, self.device)
        i, j, canonical = supervised_pair(sample, indices, points, .018, self.training,
                                           all_correspondences=True)
        output = self._pair(points, i, j, prior, "train")
        src, trg = canonical[i], canonical[j]
        correspondence = (torch.cdist(src, trg) < .018).nonzero(as_tuple=False)
        rotations = torch.as_tensor(sample["rotations_gt"], device=self.device)[indices]
        # Native gt_rotat maps canonical centered points INTO each local frame.
        gt_rotat = [rotations[i:i + 1].transpose(-1, -2), rotations[j:j + 1].transpose(-1, -2)]
        losses = {
            "shp_loss": self.native.shape_loss(src, trg, output["src_shape_feats"].transpose(-2, -1),
                                               output["trg_shape_feats"].transpose(-2, -1), correspondence),
            "occ_loss": self.native.occupancy_loss(src, trg, output["src_occ_feats"].transpose(-2, -1),
                                                   -output["trg_occ_feats"].transpose(-2, -1), correspondence),
            "mat_loss": self.native.matching_loss(output["matching_scores"], src, trg),
            "ori_loss": self.native.orientation_loss(output["src_ori"], output["trg_ori"], correspondence, gt_rotat),
        }
        losses["loss"] = sum(losses[key] * getattr(self.native, f"{name}_loss_weight")
                             for key, name in (("shp_loss", "shp"), ("occ_loss", "occ"),
                                               ("mat_loss", "mat"), ("ori_loss", "ori")))
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError("Nonfinite native CMNet loss")
        return losses

    @torch.no_grad()
    def predict(self, sample, prior=None, seed=0):
        was_training = self.training
        self.eval()
        try:
            with inference_seed(seed, self.device):
                points, _, anchor = input_parts(sample, self.device)
                if points.shape[1] ** 2 < 128:
                    return failed("too_few_points_for_native_top128_matching")
                outputs = {pair: self._pair(points, *pair, prior, "test")
                           for pair in itertools.permutations(range(len(points)), 2)}
                return native_pose_graph(outputs, len(points), anchor, "cmnet", self._translation_solver)
        finally:
            self.train(was_training)


def build(source_root, options=None, device="cpu"):
    options = dict(options or {})
    if options.get("condition", "native") not in SUPPORTED_CONDITIONS:
        raise ValueError("CMNet spatial B3/B4 are unsupported; use invariant pooled B2")
    root = native_import_root(source_root, ("model", "common", "data"))
    from model.cmnet import CMNet
    native = CMNet(lr=float(options.get("lr", .001)), visualize=False)
    solver = load_translation_solver(root / "multi_part_assembly.py", "scaffold_sota_cmnet_native_graph")
    return CMNetBackend(native, options, solver).to(device)
