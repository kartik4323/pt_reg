"""Prepare controlled study configurations; inference remains in the standard CLI."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
from . import config
from .storage import write

PROFILES = {
    # Each adjacent SD comparison changes exactly one factor.
    'legacy_cleanup': dict(prompt_variant='short', mask_type='inverted', cleanup=140),
    'clean': dict(prompt_variant='short', mask_type='inverted', cleanup=0),
    'completion': dict(prompt_variant='original', mask_type='inverted', cleanup=0),
    'exterior': dict(prompt_variant='original', mask_type='exterior', cleanup=0),
    'unprotected': dict(prompt_variant='original', mask_type='none', cleanup=0),
    'surface': dict(prompt_variant='original', mask_type='exterior', cleanup=0, render_mode='surface'),
    # Instruction editor comparison holds render and prompt fixed. The editor has no edit mask.
    'qwen': dict(prompt_variant='original', mask_type='exterior', cleanup=0, models=['qwen']),
    'qwen_depth': dict(prompt_variant='original', mask_type='exterior', cleanup=0, models=['qwen'], qwen_use_depth=True),
    'gemini': dict(prompt_variant='original', mask_type='exterior', cleanup=0, models=['gemini']),
}


def prepare(base, output, profiles, images_python, reconstruction_python):
    base = config.load(base)
    for name in profiles:
        if name not in PROFILES:
            raise ValueError(f'Unknown profile {name}; choices: {list(PROFILES)}')
        factors = PROFILES[name]
        cfg = copy.deepcopy(base)
        cfg.update(experiment_profile=name, prompt_variant=factors['prompt_variant'],
            mask_type=factors['mask_type'], canvas_fill_target=0.0, n_render_views=1,
            n_instantmesh_views=1, image_models=factors.get('models', ['sd15', 'sd15_depth']),
            input_types=['F', 'A'], image_seeds=[11, 23, 37, 51], category_prompt=True,
            reconstruct_all_for_E5=True, reconstruction_top_k=3, prior_count=3,
            skip_invalid_e1=True, oracles=True, render_mode=factors.get('render_mode', 'splat'),
            prompt_original='The image shows broken fragments of one rigid object. Complete the '
                'object by extending its missing body beyond the fragment boundary. Preserve '
                'the camera, scale, location and surviving exterior features. Continue surfaces '
                'smoothly through the exposed break; remove fracture faces that become internal. '
                'Produce one plausible complete undamaged object, neutral gray material, pure '
                'white background, no shadows, no text, no additional objects.')
        cfg['primary_model'] = cfg['image_models'][0]
        cfg['images'].update(python=images_python, bg_obj_threshold=factors['cleanup'])
        cfg['images']['qwen_use_depth'] = factors.get('qwen_use_depth', False)
        cfg.pop('bg_obj_threshold', None)
        cfg['reconstruction']['python'] = reconstruction_python
        write(Path(output) / f'{name}.json', cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--profiles', nargs='+', default=['legacy_cleanup', 'clean', 'completion', 'exterior'])
    p.add_argument('--images-python', required=True)
    p.add_argument('--reconstruction-python', required=True)
    a = p.parse_args()
    prepare(a.base, a.out, a.profiles, a.images_python, a.reconstruction_python)


if __name__ == '__main__':
    main()
