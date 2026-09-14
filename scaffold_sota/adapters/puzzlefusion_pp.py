"""Native PuzzleFusion++ VQVAE and denoiser-only experimental backend.

This deliberately does not impersonate full verify/agglomerate inference. The
native reconstruction loss fixes the 25 x 40 output to 1000 points per fragment.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .generative_common import (GenerativeBackend, import_native, load_stage_weights,
    option, packed_parts, pose_result, prediction_rng)


class PuzzleFusionDenoiserBackend(GenerativeBackend):
    model_name = 'puzzlefusion_pp'
    variant = 'denoiser_only'

    def __init__(self, source_root, options, device):
        super().__init__()
        self.stage = option(options, 'stage', 'assembly')
        if self.stage not in {'pretrain', 'assembly'}:
            raise ValueError('PF++ supports VQVAE pretrain or denoiser-only assembly')
        if option(options, 'variant', 'denoiser_only') != 'denoiser_only':
            raise ValueError('Full PF++ verifier/agglomeration is not implemented by this backend')
        self.condition = option(options, 'condition', 'native')
        if self.stage == 'pretrain' and self.condition not in {'native', 'B0'}:
            raise ValueError('VQVAE pretraining is prior-independent')
        feature_state = None
        if self.stage == 'assembly':
            checkpoint = option(options, 'feature_checkpoint')
            if not checkpoint:
                raise ValueError('PF++ denoiser requires feature_checkpoint from completed PF++ VQVAE pretrain')
            feature_state = load_stage_weights(checkpoint, self.model_name, 'pretrain', 'autoencoder.')
        # Validate the generic upstream utils namespace as well: Jigsaw also
        # uses this name, so models must run in independent native processes.
        import_native(source_root, 'utils.pn2_utils', 'utils/pn2_utils.py')
        from omegaconf import OmegaConf
        from pytorch3d import transforms
        self.transforms = transforms
        root = Path(source_root)
        cfg = OmegaConf.merge(
            OmegaConf.load(root / 'config/denoiser/model.yaml'),
            OmegaConf.load(root / 'config/denoiser/encoder.yaml'))
        self.inference_steps = int(option(options, 'inference_steps', 20))
        if not 1 <= self.inference_steps <= 1000:
            raise ValueError('PF++ inference_steps must be between 1 and 1000')
        cfg.model.num_inference_steps = self.inference_steps
        cfg.model.multiple_ref_parts = False
        if self.stage == 'pretrain':
            module = import_native(source_root,
                'puzzlefusion_plusplus.vqvae.model.modules.vq_vae',
                'puzzlefusion_plusplus/vqvae/model/modules/vq_vae.py')
            self.autoencoder = module.VQVAE(cfg)
            self.configure_conditioning(options, 512)
        else:
            module = import_native(source_root,
                'puzzlefusion_plusplus.denoiser.model.denoiser',
                'puzzlefusion_plusplus/denoiser/model/denoiser.py')
            self.native = module.Denoiser(cfg)
            self.native.encoder.load_state_dict(feature_state, strict=True)
            self.native.encoder.requires_grad_(False)
            self.configure_conditioning(options, 512, layers=self.native.denoiser.transformer_layers)
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def _batch(self, sample, supervised=False):
        batch, context = packed_parts(sample, self.device, self.transforms, supervised)
        if batch['part_pcs'].shape[2] != 1000:
            raise ValueError('Pinned PF++ requires exactly 1000 observed points per fragment; configure the shared dataset accordingly')
        return batch, context

    def loss(self, sample, prior=None):
        batch, _ = self._batch(sample, supervised=self.stage == 'assembly')
        if self.stage == 'pretrain':
            if prior is not None:
                raise ValueError('VQVAE pretraining does not accept a scaffold')
            data = {'part_pcs': batch['part_pcs'].squeeze(0)}
            output = self.autoencoder(data)
            losses = self.autoencoder.loss(data, output)
        else:
            with self.prior_context(prior):
                output = self.native(batch)
                losses = self.native._loss(batch, output)
        return {'loss': sum(losses.values()), **losses}

    @torch.no_grad()
    def predict(self, sample, prior=None, seed=0):
        if self.stage != 'assembly':
            return {'status': 'unsupported_stage', 'reason': 'PF++ VQVAE does not predict rigid poses'}
        was_training = self.training
        self.eval()
        try:
            with prediction_rng(seed, self.device), self.prior_context(prior):
                batch, context = self._batch(sample)
                count = batch['part_pcs'].shape[1]
                pose = torch.randn((1, count, 7), device=self.device)
                ref = batch['ref_part']
                identity = torch.tensor([0., 0., 0., 1., 0., 0., 0.], device=self.device)
                pose[ref] = identity
                self.native.noise_scheduler.set_timesteps(self.inference_steps)
                for timestep in self.native.noise_scheduler.timesteps:
                    times = timestep.reshape(1).to(self.device)
                    latent, xyz = self.native._extract_features(batch['part_pcs'], batch['part_valids'], pose)
                    noise = self.native.denoiser(pose, times, latent, xyz,
                        batch['part_valids'], batch['part_scale'], ref)
                    pose = self.native.noise_scheduler.step(noise, timestep, pose).prev_sample
                    pose[ref] = identity
                output = pose_result(context, pose[0, :, 3:], pose[0, :, :3], self.transforms)
                output['variant'] = self.variant
                return output
        finally:
            self.train(was_training)


def build(source_root, options, device):
    return PuzzleFusionDenoiserBackend(source_root, options, device)
