"""Native Jigsaw adapter with a study-owned scaffold residual.

The backbone, attention, classifier, affinity, Sinkhorn, matching loss, RANSAC
and global alignment are the pinned upstream implementations. This module only
adapts data, inserts conditioning, and removes GT-dependent inference framing.
Requires Jigsaw's native CUDA environment for full execution.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from ..models.conditioning import ConditioningBlock
from .matching_common import (failed, inference_seed, input_parts,
                              native_import_root, pose_result)


def critical_layout(labels, n_pcs):
    """Native critical-point layout without any target coordinate access."""
    labels = labels.to(torch.long)
    indices = torch.zeros_like(labels)
    counts = torch.zeros_like(n_pcs)
    for b in range(labels.shape[0]):
        start = 0
        for part in range(n_pcs.shape[1]):
            size = int(n_pcs[b, part])
            selected = labels[b, start:start + size].nonzero(as_tuple=False).flatten()
            counts[b, part] = selected.numel()
            indices[b, start:start + selected.numel()] = selected
            start += size
    return indices, counts


def supported_match_graph(perm, counts):
    """Reject missing/underconstrained native output rather than identity-fill it.

    Upstream uses translation-only fallback for pieces lacking three matches.
    The study's rigid-pose export requires a connected graph of >=3-match edges;
    this deliberately stricter export policy is identical across conditions.
    """
    counts = np.asarray(counts, dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    adjacency = [set() for _ in counts]
    for i in range(len(counts)):
        for j in range(i + 1, len(counts)):
            forward = perm[offsets[i]:offsets[i + 1], offsets[j]:offsets[j + 1]].sum()
            backward = perm[offsets[j]:offsets[j + 1], offsets[i]:offsets[i + 1]].sum()
            if max(forward, backward) >= 3:
                adjacency[i].add(j)
                adjacency[j].add(i)
    seen, frontier = {0}, [0]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node].difference(seen):
            seen.add(neighbor)
            frontier.append(neighbor)
    return len(seen) == len(counts)


class JigsawBackend(nn.Module):
    conditioning_parameter_prefixes = ("conditioning.",)

    def __init__(self, native, options, estimate_global_transform):
        super().__init__()
        self.native = native
        self.options = dict(options)
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
        # Native forward checks only trainer.testing. Critical labels are supplied
        # explicitly below, so its hidden test-label and GT fallback are bypassed.
        self.native._trainer = SimpleNamespace(testing=False)
        self._estimate_global_transform = estimate_global_transform
        self.native_config = {
            "architecture": "JointSegmentationAlignmentModel",
            "encoder": str(native.cfg.MODEL.ENCODER),
            "matching_loss_weight": float(native.w_mat_loss),
            "rigid_loss_weight": float(native.w_rig_loss),
            "prediction_alignment": "input_anchor_relative; align_pivot=False",
            "unsupported_pose_policy": "fail_if_3_match_graph_disconnected",
            "training_schedule": "explicit_joint_loss_weights; native epoch callbacks not invoked",
        }

    @property
    def device(self):
        return next(self.native.parameters()).device

    def _features(self, points, anchor, prior):
        count, size, _ = points.shape
        flat = points.reshape(1, count * size, 3)
        lengths = torch.full((count,), size, dtype=torch.long, device=points.device)
        feats = self.native._extract_part_feats(flat, lengths)
        for name, layer in self.native.tf_layers:
            if name == "self":
                feats = layer(flat.reshape(-1, 3).contiguous(),
                              feats.reshape(-1, self.native.pc_feat_dim), lengths)
                feats = feats.reshape(1, count * size, -1).contiguous()
            else:
                feats = layer(feats)
        if self.conditioning is not None:
            reference = torch.zeros((1, count, size), device=points.device)
            reference[:, anchor] = 1
            feats = self.conditioning(feats, prior, reference.reshape(1, -1))
        return flat, feats

    def _data(self, sample, prior, supervised):
        points, indices, anchor = input_parts(sample, self.device)
        count, size, _ = points.shape
        flat, feats = self._features(points, anchor, prior)
        n_pcs = torch.full((1, count), size, device=self.device, dtype=torch.long)
        if supervised:
            labels = torch.as_tensor(sample["fracture_labels"], device=self.device)[indices]
            labels = (labels > 0).long().reshape(1, -1)
        else:
            logits = self.native.pc_classifier(feats.transpose(1, 2))
            if self.native.pc_cls_method == "binary":
                labels = (logits.squeeze(1).sigmoid() > .5).long()
            else:
                labels = logits.argmax(dim=1)
        critical_idx, critical_count = critical_layout(labels, n_pcs)
        data = {
            "part_pcs": flat, "part_feats": feats,
            "part_valids": torch.ones((1, count), device=self.device),
            "n_pcs": n_pcs, "critical_label": labels,
            "critical_pcs_idx": critical_idx, "n_critical_pcs": critical_count,
        }
        if supervised:
            data["gt_pcs"] = torch.as_tensor(sample["canonical_points"], device=self.device,
                                             dtype=points.dtype)[indices].reshape(1, -1, 3)
        return data, anchor

    def loss(self, sample, prior=None):
        data, _ = self._data(sample, prior, supervised=True)
        if int(data["n_critical_pcs"].sum()) < 2:
            raise ValueError("Insufficient sampled fracture labels for Jigsaw matching loss")
        output = self.native(data)
        losses = self.native._loss_function(data, output)
        result = {key: value.mean() for key, value in losses.items()
                  if isinstance(value, torch.Tensor) and key.endswith("loss")}
        if "loss" not in result or not torch.isfinite(result["loss"]):
            raise RuntimeError("Native Jigsaw produced a missing or nonfinite loss")
        return result

    @torch.no_grad()
    def predict(self, sample, prior=None, seed=0):
        previous_training = self.training
        self.eval()
        try:
            with inference_seed(seed, self.device):
                data, anchor = self._data(sample, prior, supervised=False)
                counts = data["n_critical_pcs"].detach().cpu().numpy()
                if np.any(counts < 3):
                    return failed("insufficient_predicted_fracture_points", critical_counts=counts[0].tolist())
                output = self.native(data)
                perm = output["perm_mat"].detach().cpu().numpy()
                if not supported_match_graph(perm[0], counts[0]):
                    return failed("underconstrained_match_graph", critical_counts=counts[0].tolist())
                # Seed Open3D's RANSAC RNG as well as Python/NumPy/Torch.
                import open3d as o3d
                if hasattr(o3d.utility, "random"):
                    o3d.utility.random.seed(int(seed))
                try:
                    pred = self._estimate_global_transform(
                        perm, data["part_pcs"].detach().cpu().numpy(),
                        np.asarray([counts.shape[1]], dtype=np.int64),
                        data["n_pcs"].detach().cpu().numpy(), counts,
                        data["critical_pcs_idx"].detach().cpu().numpy(),
                        None, None, align_pivot=False)
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                    return failed("native_registration_failed", error=str(exc))
                return pose_result(pred["rot"][0], pred["trans"][0], anchor,
                                   critical_counts=counts[0].tolist())
        finally:
            self.train(previous_training)


def build(source_root, options=None, device="cpu"):
    """Build the real pinned Jigsaw model; missing native dependencies are errors."""
    options = dict(options or {})
    root = native_import_root(source_root, ("model", "utils", "dataset"))
    expected = root / "model/jigsaw/joint_seg_align_model.py"
    if not expected.is_file():
        raise FileNotFoundError(f"Not a Jigsaw native source tree: {root}")
    from utils.config import cfg, cfg_from_file
    cfg_from_file(str(root / "experiments/jigsaw_250e_cosine.yaml"))
    config = deepcopy(cfg)
    config.STATS = ""  # Native constructor must not write statistics elsewhere.
    config.DATA.MAX_NUM_PART = 3
    config.MODEL.ENCODER = options.get("encoder", config.MODEL.ENCODER)
    config.MODEL.LOSS.w_mat_loss = float(options.get("matching_loss_weight", 1.0))
    config.MODEL.LOSS.w_rig_loss = float(options.get("rigid_loss_weight", 0.0))
    from model.jigsaw.joint_seg_align_model import JointSegmentationAlignmentModel
    from utils.estimate_transform import estimate_global_transform
    native = JointSegmentationAlignmentModel(config)
    return JigsawBackend(native, options, estimate_global_transform).to(device)
