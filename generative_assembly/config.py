from __future__ import annotations
import copy
from .storage import read

DEFAULT = {
    'schema_version': 1, 'smoke': False, 'seed': 4101, 'points': 512, 'evaluation_points': 2048,
    'pixels': 512, 'splat_radius': 2, 'canvas_extent': 1.5,
    # Compare imagination models in experiment:
    # 'sd15_depth' : SD 1.5 with ControlNet depth conditioning
    # 'sd15'       : SD 1.5 pure inpainting without depth blocker
    # 'gemini'     : Google Gemini multimodal API
    # 'sdxl'       : SDXL inpainting
    'image_models': ['sd15_depth', 'sd15'], 'input_types': ['F', 'A'], 'image_seeds': [11, 23, 37, 51],
    'reconstruction_top_k': 3, 'reconstruct_all_for_E5': True,
    'prior_count': 3, 'prior_diversity_distance': 0.025,
    'prior_max_heldout_error': 0.08, 'prior_min_growth': 0.05,
    'skip_invalid_e1': True,
    'primary_model': 'sd15', 'primary_input': 'A', 'primary_policy': 'gated',

    # ── Prompt variants (FIX #7) ──────────────────────────────────────────────
    # prompt_variant selects which prompt text to use at runtime.
    # 'original'  : the original 96-word prompt (baseline)
    # 'short'     : short, concrete, visually descriptive (recommended fix)
    # 'category'  : short prompt + explicit category name appended
    'prompt_variant': 'short',

    # Original prompt kept for reference / ablation
    'prompt_original': (
        'The image shows broken pieces from one rigid object. Create one plausible intact '
        'object containing the surviving original exterior. Retain the reference camera, '
        'scale and location of surviving features. Extend the missing body beyond the '
        'fragment boundary. Remove exposed break faces where they become internal. '
        'Neutral gray material, white background, one object, no labels.'
    ),
    # Short prompt — pure white background without 'studio lighting' (prevents grey studio vignettes)
    'prompt_short': (
        'A smooth intact object, complete and undamaged, neutral gray CAD surface, '
        'isolated on seamless solid white background, single object, centered, no shadows.'
    ),
    # We keep category_prompt for backward compat; prompt_variant='category' uses it automatically
    'category_prompt': True,

    # ── Mask type (FIX #5) ────────────────────────────────────────────────────
    # 'inverted'  : mask covers background → SD paints the missing body around the fragment
    # 'exterior'/'original': preserve estimated exterior, regenerate fracture/background
    # 'none': allow the generator to revise every pixel
    # Both are saved; this selects which one is used as the active mask.png for E1.
    'mask_type': 'exterior',

    # ── Surface rendering (FIX #1) ────────────────────────────────────────────
    # 'surface'   : Poisson mesh + Phong shading (requires open3d; falls back if unavailable)
    # 'splat'     : original point-splat renderer
    'render_mode': 'surface',

    # ── Tight canvas crop (FIX #6) ────────────────────────────────────────────
    # Fraction of the canvas the fragment should occupy after cropping.
    # 0.0 = no crop (original); 0.65 = fragment fills ~65% of the frame.
    'canvas_fill_target': 0.0,

    # ── Multi-view rendering for E0 input (FIX #2 / Fix B) ───────────────────
    # Number of camera viewpoints to render and pass to E1.
    # 1 = original single canonical view; 4 = structured azimuths every 90°.
    # Each view produces its own set of E1 generations; best one is selected.
    'n_render_views': 1,

    # ── Background stripping (Fix A) ─────────────────────────────────────────
    # Historical configs may supply bg_obj_threshold here; load() migrates it
    # to images.bg_obj_threshold. Main arms disable destructive whitening.

    # ── Multi-view InstantMesh grid (Fix C) ───────────────────────────────────
    # Only the official single-image InstantMesh path is supported. Independent
    # edits are not geometrically consistent canonical multi-view observations.
    'n_instantmesh_views': 1,

    'images': {'python': None, 'device': 'cuda', 'dtype': 'float16', 'cpu_offload': True,
               'bg_obj_threshold': 0,
               'steps': 30, 'qwen_steps': 40, 'guidance': 7.5, 'control_strength': 0.5, 'qwen_cfg': 4.0,
               'sd15_depth': 'stable-diffusion-v1-5/stable-diffusion-inpainting',
               'sd15': 'stable-diffusion-v1-5/stable-diffusion-inpainting',
               'controlnet': 'lllyasviel/control_v11f1p_sd15_depth',
               'qwen': 'Qwen/Qwen-Image-Edit-2509',
               'sdxl': 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1', 'revisions': {}},
    'reconstruction': {'backend': 'instantmesh', 'python': None, 'repo': None, 'commit': None,
                       'config': 'configs/instant-mesh-large.yaml', 'steps': 75, 'seed': 42,
                       'revisions': {}, 'timeout_seconds': 3600,
                       # FIX #8: Reject templates with degenerate shape before feeding to E3.
                       # PCA second-largest/thinnest bbox dimension ratio; permits long bottles.
                       'template_max_aspect_ratio': 6.0,
                       # flatness_max: max fraction of points within a thin slab (thickness < 5% of bbox).
                       # Above this, the template is considered degenerate.
                       'template_max_flatness': 0.85},
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
    supplied = read(path)
    cfg = merge(copy.deepcopy(DEFAULT), supplied)
    # Accept historical top-level configuration, but give the worker setting priority.
    if 'bg_obj_threshold' in supplied and 'bg_obj_threshold' not in supplied.get('images', {}):
        cfg['images']['bg_obj_threshold'] = supplied['bg_obj_threshold']
    if cfg['mask_type'] not in ('original', 'inverted', 'exterior', 'none'):
        raise ValueError('Unknown mask_type')
    if cfg['prior_count'] < 1:
        raise ValueError('prior_count must be positive')
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
