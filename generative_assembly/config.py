from __future__ import annotations
import copy
from .storage import read

DEFAULT = {
    'schema_version': 1, 'smoke': False, 'seed': 4101, 'points': 512, 'evaluation_points': 2048,
    'pixels': 512, 'splat_radius': 2, 'canvas_extent': 1.5,
    'image_models': ['sd15_depth', 'qwen'], 'input_types': ['F', 'A'], 'image_seeds': [11, 23, 37, 51],
    'reconstruction_top_k': 2, 'reconstruct_all_for_E5': True,
    'primary_model': 'sd15_depth', 'primary_input': 'A', 'primary_policy': 'gated',
    'prompt': 'The image shows broken pieces from one rigid object. Create one plausible intact object containing the surviving original exterior. Retain the reference camera, scale and location of surviving features. Extend the missing body beyond the fragment boundary. Remove exposed break faces where they become internal. Neutral gray material, white background, one object, no labels.',
    'category_prompt': False,
    'images': {'python': None, 'device': 'cuda', 'dtype': 'float16', 'cpu_offload': True,
               'steps': 30, 'qwen_steps': 40, 'guidance': 7.5, 'control_strength': 0.5, 'qwen_cfg': 4.0,
               'sd15_depth': 'stable-diffusion-v1-5/stable-diffusion-inpainting',
               'sd15': 'stable-diffusion-v1-5/stable-diffusion-inpainting',
               'controlnet': 'lllyasviel/control_v11f1p_sd15_depth',
               'qwen': 'Qwen/Qwen-Image-Edit-2509',
               'sdxl': 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1', 'revisions': {}},
    'reconstruction': {'backend': 'instantmesh', 'python': None, 'repo': None, 'commit': None,
                       'config': 'configs/instant-mesh-large.yaml', 'steps': 75, 'seed': 42,
                       'revisions': {}, 'timeout_seconds': 3600},
    'solver': {'candidates': 32, 'patches': 32, 'contact_fraction': 0.08, 'contact_cap': 0.15,
               'template_weight': 0.3, 'refine_evaluations': 25, 'alignment_starts': 8,
               'alignment_iterations': 8, 'scales': [1.0, 1.5, 2.0],
               'gate_exterior': 0.06, 'gate_contact_ratio': 1.1},
    'training': {'updates': 2000, 'seeds': [101, 202, 303], 'lr': 0.001, 'hidden': 96,
                 'checkpoint_every': 100, 'device': 'cuda', 'consistency_weight': 0.2},
    'robustness': {'noise': [0.0025, 0.005, 0.01], 'dropout': [0.25, 0.5],
                   'repeats': 3, 'missing_piece': True, 'erosion_fraction': 0.1},
    'evaluation': {'threshold': 0.01, 'bootstrap': 2000, 'success_gain': 0.05, 'damage_max': 0.05},
    'oracles': True, 'resources': {'minimum_free_gib': 10, 'worker_timeout_seconds': 3600}
}


def merge(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = v
    return base


def load(path):
    cfg = merge(copy.deepcopy(DEFAULT), read(path))
    if cfg['primary_model'] not in cfg['image_models'] or cfg['primary_input'] not in cfg['input_types']:
        raise ValueError('Primary image model/input must be in experiment conditions')
    if cfg['primary_policy'] not in ('always', 'weak', 'gated'):
        raise ValueError('Unknown primary policy')
    for name in ('points', 'pixels', 'evaluation_points'):
        if int(cfg[name]) < 16:
            raise ValueError(f'{name} must be >=16')
    if not cfg['image_seeds'] or cfg['reconstruction_top_k'] < 1:
        raise ValueError('Positive image/hypothesis budgets required')
    if cfg['solver']['candidates'] < 1 or cfg['solver']['alignment_starts'] < 1:
        raise ValueError('Positive solver budgets required')
    if cfg['smoke']:
        cfg['reconstruction']['backend'] = 'smoke'
    elif cfg['reconstruction']['backend'] not in ('instantmesh', 'command'):
        raise ValueError('Real runs require instantmesh or an explicit reconstruction command')
    return cfg
