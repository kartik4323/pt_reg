"""Native PMTR plus scaffold conditioning at coarse superpoint features.

Training calls the native objective/forward. Prediction reproduces only its
input-derived computation, omitting the upstream forward's unconditional GT
node-correspondence bookkeeping. No Stage 3 assembly solver is imported.
"""
from __future__ import annotations

import itertools
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

from ..models.conditioning import ConditioningBlock
from .matching_common import failed, inference_seed, input_parts, native_import_root
from .matching_pairwise import load_translation_solver, native_pose_graph, supervised_pair

SUPPORTED_CONDITIONS = ("native", "B0", "B1", "B2", "B3", "B4")


class PMTRBackend(nn.Module):
    conditioning_parameter_prefixes = ("conditioning.",)
    supported_conditions = SUPPORTED_CONDITIONS

    def __init__(self, native, options, subsampling, node_ops, translation_solver):
        super().__init__()
        self.native, self.options = native, dict(options)
        self.condition = options.get("condition", "native")
        if self.condition not in SUPPORTED_CONDITIONS:
            raise ValueError(f"Unsupported PMTR condition {self.condition}")
        self._subsampling, self._node_ops = subsampling, node_ops
        self._translation_solver = translation_solver
        self._active_prior, self._pair_reference = None, (False, False)
        # Upstream uses local_rank only to enter a Lightning logging/profiler
        # block. The standalone runner owns logging, so disable that block.
        self.native._trainer = SimpleNamespace(local_rank=1)
        self.native.evaluate = False
        self.conditioning = None
        if self.condition not in {"native", "B0"}:
            self.conditioning = ConditioningBlock(512, self.condition,
                token_dim=int(options.get("token_dim", 128)), heads=int(options.get("heads", 4)),
                num_tokens=int(options.get("num_tokens", 512)))
            self._conditioning_hook = self.native.coarse_matcher.register_forward_pre_hook(self._condition_coarse)
        self.native_config = {
            "architecture": "PMTR", "fine_matcher": options.get("fine_matcher", "pmt"),
            "cpconv_radius": float(options.get("cpconv_radius", .05)),
            "subsampling_radius": float(options.get("subsampling_radius", .01)),
            "conditioning_site": "coarse_matcher input superpoint features",
            "graph_policy": "native_all_ordered_pairs", "graph_initialization": "predicted_anchor_pairs",
            "training_pair_policy": "native_random_source_max_overlap_target",
            "native_logging": "disabled; study runner owns logging",
        }

    @property
    def device(self):
        return next(self.native.parameters()).device

    def _condition_coarse(self, module, args):
        src_pos, trg_pos, src_feat, trg_feat = args
        sr = torch.full(src_feat.shape[:2], float(self._pair_reference[0]), device=src_feat.device)
        tr = torch.full(trg_feat.shape[:2], float(self._pair_reference[1]), device=trg_feat.device)
        return (src_pos, trg_pos, self.conditioning(src_feat, self._active_prior, sr),
                self.conditioning(trg_feat, self._active_prior, tr))

    def _pair_data(self, points, i, j):
        if points.shape[1] < self.native.npts_per_node:
            raise ValueError(f"Native PMTR requires at least {self.native.npts_per_node} sampled points per fragment")
        pair = [points[i:i + 1], points[j:j + 1]]
        cpu_points = torch.cat([pair[0][0], pair[1][0]]).detach().cpu().contiguous()
        lengths = torch.tensor([pair[0].shape[1], pair[1].shape[1]], dtype=torch.long)
        sampled = self._subsampling(cpu_points, lengths, 3,
                                     float(self.options.get("subsampling_radius", .01)),
                                     .125, [35, 32, 34])
        data = {"pcd_t": pair, "n_frac": 2}
        for name, levels in zip(("points", "lengths", "neighbors", "subsampling", "upsampling"), sampled):
            data[f"{name}_ext_t"] = {"0-1": [level.to(self.device).unsqueeze(0) for level in levels]}
        return data

    def _predict_pair(self, data):
        """Native forward operations without GT targets, loss or metric calls."""
        partition, select = self._node_ops
        src_pts, src_nds, src_len, src_nd_len, trg_pts, trg_nds, _, _ = self.native._prepare_input(data)
        _, src_mask, src_idx, src_knn_mask = partition(src_pts, src_nds, self.native.npts_per_node)
        _, trg_mask, trg_idx, trg_knn_mask = partition(trg_pts, trg_nds, self.native.npts_per_node)
        src_pad = torch.cat([src_pts, torch.zeros_like(src_pts[:1])])
        trg_pad = torch.cat([trg_pts, torch.zeros_like(trg_pts[:1])])
        src_knn_pts, trg_knn_pts = select(src_pad, src_idx, 0), select(trg_pad, trg_idx, 0)
        features = self.native.backbone(torch.ones_like(torch.cat([src_pts, trg_pts])[:, :1]), data)
        src_fine, trg_fine = features[0][:src_len], features[0][src_len:]
        src_coarse, trg_coarse = features[-1][:src_nd_len], features[-1][src_nd_len:]
        src_coarse, trg_coarse = self.native.coarse_matcher(src_nds.unsqueeze(0), trg_nds.unsqueeze(0),
                                                          src_coarse.unsqueeze(0), trg_coarse.unsqueeze(0))
        src_coarse, trg_coarse = F.normalize(src_coarse.squeeze(0), dim=1), F.normalize(trg_coarse.squeeze(0), dim=1)
        src_corr, trg_corr, scores = self.native.coarse_matching(src_coarse, trg_coarse, src_mask, trg_mask)
        if src_corr.numel() == 0:
            raise ValueError("Native PMTR produced no coarse correspondences")
        output = {"node_corr_scores": scores}
        collection = self.native._node_match_collection(src_corr, trg_corr, output, src_idx, trg_idx,
            src_knn_mask, trg_knn_mask, src_knn_pts, trg_knn_pts, src_fine, trg_fine, scores)
        transform = collection[-1]
        inverse = transform[:3, :3].inverse()
        output["estimated_rotat"] = inverse
        output["estimated_trans"] = inverse @ transform[:3, 3]
        return output

    def loss(self, sample, prior=None):
        points, indices, anchor = input_parts(sample, self.device)
        i, j, canonical = supervised_pair(sample, indices, points, .015, self.training)
        data = self._pair_data(points, i, j)
        r = torch.as_tensor(sample["rotations_gt"], device=self.device)[indices]
        t = torch.as_tensor(sample["translations_gt"], device=self.device)[indices]
        data.update(pcd=[canonical[i:i + 1], canonical[j:j + 1]],
                    gt_rotat_inv=[r[i:i + 1], r[j:j + 1]],
                    gt_trans_inv=[-t[i:i + 1], -t[j:j + 1]],
                    gt_rotat=[r[i:i + 1].transpose(-1, -2), r[j:j + 1].transpose(-1, -2)],
                    gt_trans=[t[i:i + 1], t[j:j + 1]])
        relative_r = r[j].transpose(-1, -2) @ r[i]
        relative_t = -(r[j].transpose(-1, -2) @ (t[i] - t[j]))
        data["relative_trsfm"] = {"0-1": [relative_r.unsqueeze(0), relative_t.unsqueeze(0)]}
        self._active_prior, self._pair_reference = prior, (i == anchor, j == anchor)
        try:
            # This mode retains native supervised coarse-target sampling.
            _, losses = self.native.forward_pass(data, mode="train", optimizer_idx=-1)
        finally:
            self._active_prior = None
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError("Nonfinite native PMTR loss")
        return {key: value.mean() for key, value in losses.items()
                if isinstance(value, torch.Tensor) and (key == "loss" or key.endswith("loss"))}

    @torch.no_grad()
    def predict(self, sample, prior=None, seed=0):
        was_training = self.training
        self.eval()
        try:
            with inference_seed(seed, self.device):
                points, _, anchor = input_parts(sample, self.device)
                outputs = {}
                for i, j in itertools.permutations(range(len(points)), 2):
                    self._active_prior, self._pair_reference = prior, (i == anchor, j == anchor)
                    try:
                        outputs[(i, j)] = self._predict_pair(self._pair_data(points, i, j))
                    except (ValueError, RuntimeError) as exc:
                        if "out of memory" in str(exc).lower():
                            raise
                        return failed("native_pair_registration_failed", pair=[i, j], error=str(exc))
                    finally:
                        self._active_prior = None
                return native_pose_graph(outputs, len(points), anchor, "pmtr", self._translation_solver)
        finally:
            self.train(was_training)


def build(source_root, options=None, device="cpu"):
    options = dict(options or {})
    root = native_import_root(source_root, ("model", "common", "data"))
    from model.pmtr import PMTR
    from common.sampling import subsampling
    from common.node import point_to_node_partition, index_select
    native = PMTR(options.get("fine_matcher", "pmt"), float(options.get("cpconv_radius", .05)),
                  float(options.get("lr", .001)), evaluate=False)
    solver = load_translation_solver(root / "test_mpa.py", "scaffold_sota_pmtr_native_graph")
    return PMTRBackend(native, options, subsampling, (point_to_node_partition, index_select), solver).to(device)
