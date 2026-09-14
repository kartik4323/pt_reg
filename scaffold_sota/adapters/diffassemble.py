"""The released DiffAssemble 3D double-diffusion implementation.

The import-only namespace loader avoids upstream's absent backbone_vist module;
all instantiated networks, sampling equations and losses come from native code.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch

from .generative_common import (GenerativeBackend, NativeDependencyError, labels,
    observation, option, pose_result, prediction_rng)


def native_module(source_root):
    sys.dont_write_bytecode = True
    root = Path(source_root).resolve() / 'puzzle_diff' / 'model'
    implementation = root / 'spatial_diffusion_3d_test_double_diffusion.py'
    if not implementation.is_file():
        raise NativeDependencyError(f'Missing actual released DiffAssemble implementation: {implementation}')
    # Equivalent to the suite's import-only 3d-backbone-imports.patch, without
    # changing upstream files or importing its unrelated incomplete 2D models.
    name = '_scaffold_sota_native_diffassemble'
    for suffix, path in (('', root), ('.backbones', root / 'backbones')):
        module_name = name + suffix
        if module_name in sys.modules:
            if list(sys.modules[module_name].__path__) != [str(path)]:
                raise NativeDependencyError('DiffAssemble source changed within one process')
        else:
            package = types.ModuleType(module_name)
            package.__path__ = [str(path)]
            package.__package__ = module_name
            sys.modules[module_name] = package
    try:
        backbone = importlib.import_module(name + '.backbones.efficient_gat_3d')
        sys.modules[name + '.backbones'].Eff_GAT_3d = backbone.Eff_GAT_3d
        return importlib.import_module(name + '.spatial_diffusion_3d_test_double_diffusion')
    except (ImportError, OSError) as exc:
        raise NativeDependencyError(f'DiffAssemble requires its pinned legacy CUDA environment: {exc}') from exc


class DiffAssembleBackend(GenerativeBackend):
    model_name = 'diffassemble'

    def __init__(self, source_root, options, device):
        super().__init__()
        self.stage = option(options, 'stage', 'assembly')
        if self.stage != 'assembly':
            raise ValueError('DiffAssemble has only the assembly training stage')
        self.max_parts = int(option(options, 'max_parts', 3))
        if self.max_parts != 3:
            raise ValueError('This bottle adapter uses exactly three padded fragment slots')
        steps = int(option(options, 'diffusion_steps', 300))
        ratio = int(option(options, 'inference_ratio', 10))
        if steps < 2 or ratio < 1 or ratio > steps:
            raise ValueError('Invalid DiffAssemble diffusion schedule')
        if option(options, 'use_6dof', False):
            raise ValueError('This adapter preserves the default quaternion/translation path; 6D variant not supported')
        module = native_module(source_root)
        from pytorch3d import transforms
        self.transforms = transforms
        self.native = module.GNN_Diffusion(steps=steps, inference_ratio=ratio, sampling='DDIM',
            classifier_free_prob=0.0, classifier_free_w=0.2,
            noise_weight=float(option(options, 'noise_weight', 0.0)),
            model_mean_type=module.ModelMeanType.START_X,
            freeze_backbone=False, visual_pretrained=False, n_layers=4,
            loss_type='all', backbone='vn_dgcnn', max_epochs=500,
            max_num_part=self.max_parts, use_6dof=False, architecture='transformer')
        self.configure_conditioning(options, self.native.model.gnn_feat_dim,
            output_module=self.native.model.mlp)
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def _batch(self, sample):
        context = observation(sample, self.device)
        points, mask, active, _ = context
        if len(points) != self.max_parts or points.shape[1] != 1000:
            raise ValueError('Pinned DiffAssemble loss requires points[3,1000,3]; use the same observation resolution across arms')
        count = len(active)
        nodes = torch.arange(count, device=self.device)
        edges = torch.stack((nodes.repeat_interleave(count), nodes.repeat(count)))
        batch_ids = torch.zeros(count, dtype=torch.long, device=self.device)
        return points[active], edges, batch_ids, context

    def loss(self, sample, prior=None):
        pcd, edges, batch_ids, context = self._batch(sample)
        points, mask, active, anchor = context
        q, t = labels(sample, points, active, anchor, self.transforms)
        target = torch.cat((q, t), dim=-1)
        times = torch.randint(0, self.native.steps, (1,), device=self.device).expand(len(active))
        with self.prior_context(prior):
            losses = self.native.p_losses(target, times, loss_type='all', cond=pcd,
                edge_index=edges, batch=batch_ids, n_batch=1, valids=mask.unsqueeze(0))
        return {'loss': sum(losses.values()), **losses}

    @torch.no_grad()
    def predict(self, sample, prior=None, seed=0):
        was_training = self.training
        self.eval()
        try:
            with prediction_rng(seed, self.device), self.prior_context(prior):
                pcd, edges, batch_ids, context = self._batch(sample)
                samples, _ = self.native.p_sample_loop((len(pcd), 7), pcd, edges, batch_ids)
                poses = samples[-1]
                result = pose_result(context, poses[:, :4], poses[:, 4:7], self.transforms)
                result['native_noise_weight'] = self.native.noise_weight
                return result
        finally:
            self.train(was_training)


def build(source_root, options, device):
    return DiffAssembleBackend(source_root, options, device)
