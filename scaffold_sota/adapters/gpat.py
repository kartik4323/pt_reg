"""Native GPAT for the separate PartNet surface-target replacement track.

Field-query tokens are intentionally insufficient. A study-owned surface export
and semantic dataset adapter must supply this backend's documented contract.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .generative_common import (import_native, observation, option, pose_result,
    prediction_rng)


TARGET_CONDITIONS = {'native', 'B0', 'target_clean', 'target_predicted',
                     'target_degraded', 'target_wrong', 'target_generic'}


def surface_digest(points):
    array = np.asarray(torch.as_tensor(points).detach().cpu(), dtype='<f4', order='C')
    return hashlib.sha256(array.tobytes()).hexdigest()


def surface_input(sample, prior, device, target_points=5000, part_points=1000):
    if sample.get('dataset_track') != 'partnet':
        raise ValueError('GPAT requires an explicit dataset_track=partnet semantic sample, not the bottle StudyDataset')
    if prior is None or 'surface_points' not in prior:
        raise ValueError('GPAT requires prior.surface_points exported from a surface; signed-distance query tokens are not target points')
    target = torch.as_tensor(prior['surface_points'], dtype=torch.float32, device=device)
    if target.shape != (target_points, 3) or not torch.isfinite(target).all():
        raise ValueError(f'GPAT target requires exactly {target_points} finite surface points')
    if (target - target.mean(dim=0)).norm(dim=-1).max() <= 1e-8:
        raise ValueError('GPAT target surface is degenerate')
    context = observation(sample, device)
    points, mask, active, anchor = context
    if points.shape[1] != part_points:
        raise ValueError(f'GPAT requires {part_points} observed points per part')
    if not torch.equal(active, torch.arange(len(active), device=active.device)):
        raise ValueError('GPAT native pose fitting requires contiguous active parts followed by padding')
    return target, context


class GPATBackend(nn.Module):
    model_name = 'gpat'
    stage = 'assembly'
    variant = 'partnet_target_replacement'
    conditioning_parameter_prefixes = ()

    def __init__(self, source_root, options, device):
        super().__init__()
        if option(options, 'dataset_track') != 'partnet':
            raise ValueError('GPAT is a separate PartNet surface-target track; set dataset_track=partnet with its semantic adapter')
        if option(options, 'stage', 'assembly') != 'assembly':
            raise ValueError('GPAT supports assembly/segmentation training, not field pretraining')
        self.condition = option(options, 'condition', 'native')
        if self.condition not in TARGET_CONDITIONS:
            raise ValueError('GPAT uses named target-quality conditions, not B1-B4 additional-field branches')
        self.max_num_part = int(option(options, 'max_parts', 20))
        self.target_points = int(option(options, 'target_points', 5000))
        if self.max_num_part < 2 or self.target_points != 5000:
            raise ValueError('The native GPAT protocol requires at least two part slots and 5000 target points')
        self.use_dense_target = bool(option(options, 'use_dense_target', False))
        self.optimize = bool(option(options, 'optimize', False))
        # Guard native generic namespaces against another model in this process.
        import_native(source_root, 'utils.utils', 'utils/utils.py')
        module = import_native(source_root, 'learning.assembler', 'learning/assembler.py')
        training = import_native(source_root, 'learning.gpat.run', 'learning/gpat/run.py')
        from pytorch3d import transforms
        self.transforms = transforms
        self.training_implementation = training.Segmenter
        self.assembler = module.Assembler(torch.device(device),
            SimpleNamespace(max_num_part=self.max_num_part, opt=self.optimize))
        self.native = self.assembler.segmenter
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def initialize_from_native(self, state_dict):
        return self.load_state_dict(state_dict, strict=True)

    def _input(self, sample, prior):
        target, context = surface_input(sample, prior, self.device, self.target_points)
        points, mask, active, _ = context
        if len(points) != self.max_num_part:
            raise ValueError(f'GPAT sample must have {self.max_num_part} padded part slots')
        return target.unsqueeze(0), points.unsqueeze(0), (~mask).float().unsqueeze(0), context

    def match_gt(self, gt_seg, pred_seg, eq_classes):
        return self.training_implementation.match_gt(self, gt_seg, pred_seg, eq_classes)

    def loss(self, sample, prior=None):
        target, parts, masks, context = self._input(sample, prior)
        # Labels must refer to these exact sampled target points. This prevents
        # inadvertently retaining labels for the clean target after replacement.
        if sample.get('target_surface_sha256') != surface_digest(target[0]):
            raise ValueError('GPAT segmentation supervision must identify the exact replacement surface digest')
        segmentation = torch.as_tensor(sample['target_segmentation'], dtype=torch.long, device=self.device)
        equivalences = torch.as_tensor(sample['equivalence_classes'], dtype=torch.long, device=self.device)
        count = len(context[2])
        if segmentation.shape != (self.target_points,) or equivalences.shape != (self.max_num_part,):
            raise ValueError('GPAT semantic supervision has incompatible shapes')
        if (segmentation < 0).any() or (segmentation >= count).any():
            raise ValueError('GPAT target segmentation labels must reference active part slots')
        probability, indices = self.native(parts, target, masks)
        # Preserve the released objective and native equivalence reassignment,
        # including its probability-valued cross entropy convention.
        loss, accuracy, recall, miou, _, _ = self.training_implementation.calc_loss(
            self, segmentation.unsqueeze(0), indices, probability, equivalences.unsqueeze(0))
        return {'loss': loss, 'segmentation_accuracy': accuracy.mean()}

    @torch.no_grad()
    def predict(self, sample, prior=None, seed=0):
        was_training = self.training
        self.eval()
        try:
            with prediction_rng(seed, self.device):
                target, parts, masks, context = self._input(sample, prior)
                count = len(context[2])
                _, segmentation = self.assembler.get_seg(target, parts, masks)
                if self.use_dense_target:
                    if 'surface_points_dense' not in prior:
                        raise ValueError('Dense GPAT fitting requires replacement surface_points_dense, never a clean-target fallback')
                    dense = torch.as_tensor(prior['surface_points_dense'], dtype=torch.float32, device=self.device)
                    if dense.shape != (100000, 3) or not torch.isfinite(dense).all():
                        raise ValueError('Dense GPAT target requires 100000 finite surface points')
                    # Derive every dense-to-sparse map from this pair of supplied
                    # targets; ignore external/stale target_100k_label indices.
                    nearest = torch.cat([torch.cdist(chunk, target[0]).argmin(dim=1)
                                         for chunk in dense.split(128)], dim=0)
                else:
                    dense = target[0]
                    nearest = torch.zeros(self.target_points, dtype=torch.long, device=self.device)
                # Use native fitting. All geometry comes from the replacement
                # target (or explicit native clean-target condition) above.
                correspondence = self.assembler.get_seg_pc(segmentation[0], parts[0], target[0],
                    dense, nearest, count)
                # Released get_seg_pc omits pc_i.shape[0] == N and leaves that
                # slot uninitialized. Copy the exact segment in that boundary
                # case; all other native fitting behavior remains intact.
                fitting_points = dense if nearest.sum() != 0 else target[0]
                fitting_seg = segmentation[0, nearest] if nearest.sum() != 0 else segmentation[0]
                for i in range(count):
                    selected = fitting_points[fitting_seg == i]
                    if len(selected) == parts.shape[2]:
                        correspondence[i] = selected
                poses = self.assembler.get_poses(target[0], correspondence, parts[0], count)
                result = pose_result(context, poses[:count, 3:], poses[:count, :3],
                    self.transforms, reanchor=False)
                result.update(variant=self.variant, coordinate_frame='provided_target',
                    empty_target_segments=int(sum(not (segmentation[0] == i).any() for i in range(count))),
                    use_dense_target=self.use_dense_target, native_cma=self.optimize)
                return result
        finally:
            self.train(was_training)


def build(source_root, options, device):
    return GPATBackend(source_root, options, device)
