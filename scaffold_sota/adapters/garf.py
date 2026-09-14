"""Real GARF fracture pretraining and SE(3) flow, adapted to input anchor frames.

Requires the pinned GARF CUDA environment. No target-derived geometry enters
prediction. XYZ-only observations use identical estimated normals in all arms.
"""
from __future__ import annotations

from functools import partial

import torch

from .generative_common import (GenerativeBackend, estimate_normals, import_native,
    labels, load_stage_weights, observation, option, pose_result, prediction_rng)


def make_batch(sample, device, *, supervised=False, transforms=None, normal_neighbors=24):
    points, mask, active, anchor = observation(sample, device)
    observed = points[active]
    scale = observed.abs().amax(dim=(1, 2)).unsqueeze(-1)
    if 'normals' in sample:
        all_normals = torch.as_tensor(sample['normals'], dtype=points.dtype, device=points.device)
        if all_normals.shape != points.shape or not torch.isfinite(all_normals[active]).all():
            raise ValueError('Observed normals must have shape points[F,N,3] and be finite')
        normals = all_normals[active]
        if (normals.norm(dim=-1) < 1e-8).any():
            raise ValueError('Observed normals contain zero vectors')
        normals = torch.nn.functional.normalize(normals, dim=-1)
    else:
        normals = estimate_normals(observed, normal_neighbors)
    count, n, _ = observed.shape
    batch = {
        'pointclouds': (observed / scale[:, None]).reshape(1, count*n, 3),
        'pointclouds_normals': normals.reshape(1, count*n, 3),
        'points_per_part': torch.full((1, count), n, dtype=torch.long, device=points.device),
        'num_parts': torch.tensor([count], device=points.device),
        'scale': scale.unsqueeze(0),
        'ref_part': (active == anchor).unsqueeze(0),
        # Required native bookkeeping. Features/poses never use a GT graph.
        'graph': torch.zeros((1, count, count), dtype=torch.bool, device=points.device),
    }
    if supervised:
        if transforms is not None:
            q, t = labels(sample, points, active, anchor, transforms)
            batch.update(quaternions=q.unsqueeze(0), translations=t.unsqueeze(0))
        fracture = torch.as_tensor(sample['fracture_labels'], device=points.device)
        if fracture.shape != points.shape[:2] or not torch.isfinite(fracture[active]).all():
            raise ValueError('Fracture labels must align with observed input points')
        if not ((fracture[active] == 0) | (fracture[active] == 1)).all():
            raise ValueError('GARF fracture supervision must be binary')
        batch['fracture_surface_gt'] = fracture[active].reshape(1, count*n).long()
    return batch, (points, mask, active, anchor)


class GARFBackend(GenerativeBackend):
    model_name = 'garf'
    requires_amp = True

    def __init__(self, source_root, options, device):
        super().__init__()
        self.stage = option(options, 'stage', 'assembly')
        if self.stage not in {'pretrain', 'assembly'}:
            raise ValueError('GARF supports stage=pretrain or assembly')
        self.normal_neighbors = int(option(options, 'normal_neighbors', 24))
        self.inference_steps = int(option(options, 'inference_steps', 20))
        if self.inference_steps < 1:
            raise ValueError('inference_steps must be positive')
        self.condition = option(options, 'condition', 'native')
        if self.stage == 'pretrain' and self.condition not in {'native', 'B0'}:
            raise ValueError('GARF fracture pretraining is prior-independent; use condition=native')
        feature_path = option(options, 'feature_checkpoint')
        feature_state = None
        if self.stage == 'assembly':
            if not feature_path:
                raise ValueError('GARF assembly requires feature_checkpoint from a completed GARF pretrain stage')
            feature_state = load_stage_weights(feature_path, 'garf', 'pretrain', 'feature_extractor.')
        module = import_native(source_root, 'assembly.models.pretraining.frac_seg', 'assembly/models/pretraining/frac_seg.py')
        ptv3 = import_native(source_root, 'assembly.backbones.pointtransformerv3', 'assembly/backbones/pointtransformerv3/model.py')
        from pytorch3d import transforms
        self.transforms = transforms
        # Exactly the upstream ptv3.yaml architecture. Memory switches are
        # explicit runtime choices; native feature widths/depths are unchanged.
        encoder = ptv3.PointTransformerV3(
            stride=(2, 2, 2), enc_depths=(2, 2, 6, 2), enc_num_head=(2, 4, 8, 16),
            enc_patch_size=(1024,)*4, enc_channels=(32, 64, 128, 256),
            dec_depths=(2, 2, 2), dec_num_head=(4, 8, 16), dec_patch_size=(1024,)*3,
            dec_channels=(64, 128, 128), enable_flash=bool(option(options, 'encoder_flash', True)),
        )
        feature_extractor = module.FracSeg(pc_feat_dim=64, encoder=encoder,
            optimizer=partial(torch.optim.AdamW, lr=1e-4, weight_decay=1e-5), grid_size=0.02)
        if self.stage == 'pretrain':
            self.feature_extractor = feature_extractor
            self.configure_conditioning(options, 512)
        else:
            feature_extractor.load_state_dict(feature_state, strict=True)
            native = import_native(source_root, 'assembly.models.denoiser', 'assembly/models/denoiser/__init__.py')
            denoiser = native.DenoiserTransformer(in_dim=64, out_dim=7, embed_dim=512,
                num_layers=6, num_heads=8, dropout_rate=0.2, trans_out_dim=3, rot_out_dim=3,
                use_flash_attn=bool(option(options, 'denoiser_flash', False)))
            self.native = native.DenoiserFlowMatching(
                feature_extractor=feature_extractor, feature_extractor_ckpt=None,
                denoiser=denoiser,
                noise_scheduler=native.SE3FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
                val_noise_scheduler=native.SE3FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
                optimizer=partial(torch.optim.AdamW, lr=2e-4),
                inference_config={'num_inference_steps': self.inference_steps})
            self.configure_conditioning(options, 512, layers=self.native.denoiser.transformer_layers)
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def train(self, mode=True):
        super().train(mode)
        if self.stage == 'assembly':
            self.native.feature_extractor.eval()
        return self

    def loss(self, sample, prior=None):
        batch, _ = make_batch(sample, self.device, supervised=True,
            transforms=self.transforms if self.stage == 'assembly' else None,
            normal_neighbors=self.normal_neighbors)
        if self.stage == 'pretrain':
            if prior is not None:
                raise ValueError('Do not provide a prior during GARF fracture pretraining')
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
                output = self.feature_extractor(batch)
                loss, metrics = self.feature_extractor.criteria(batch, output)
            return {'loss': loss, **metrics}
        with self.prior_context(prior), torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
            output = self.native(batch)
            losses, selected = self.native._loss(batch, output)
            return {'loss': sum(losses[key] for key in selected), **losses}

    @torch.no_grad()
    def predict(self, sample, prior=None, seed=0):
        if self.stage != 'assembly':
            return {'status': 'unsupported_stage', 'reason': 'GARF fracture pretraining does not predict poses'}
        was_training = self.training
        self.eval()
        try:
            with prediction_rng(seed, self.device), self.prior_context(prior):
                batch, context = make_batch(sample, self.device, normal_neighbors=self.normal_neighbors)
                count = int(batch['num_parts'][0])
                valid = batch['points_per_part'] != 0
                ref = batch['ref_part'][valid]
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
                    latent = self.native._extract_features(batch)
                pose = torch.randn((count, 7), device=self.device)
                pose[:, 3:] = torch.nn.functional.normalize(torch.randn((count, 4), device=self.device), dim=-1)
                identity = torch.tensor([0., 0., 0., 1., 0., 0., 0.], device=self.device)
                pose[ref] = identity
                scheduler = self.native.val_noise_scheduler
                scheduler.set_timesteps(self.inference_steps)
                for timestep in scheduler.timesteps:
                    times = timestep.reshape(1).repeat(count).to(self.device)
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
                        pred = self.native.denoiser(x=pose, timesteps=times, latent=latent,
                            part_valids=valid, scale=batch['scale'][valid], ref_part=ref)['pred']
                    pose = scheduler.step(pred.float(), timestep, pose.float()).prev_sample
                    pose[ref] = identity
                return pose_result(context, pose[:, 3:], pose[:, :3], self.transforms)
        finally:
            self.train(was_training)


def build(source_root, options, device):
    return GARFBackend(source_root, options, device)
