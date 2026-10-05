"""Isolated real model workers. Smoke substitutes are explicitly branded."""
from __future__ import annotations
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image
from .storage import digest, read, write


def launch(request, directory, python=None, timeout=3600):
    directory = Path(directory).resolve()
    write(directory / 'request.json', request)
    if request.get('smoke'):
        (directory/'worker.log').write_text('Explicit in-process SMOKE substitute; no model inference.\n',encoding='utf-8')
        worker(directory/'request.json')
        return read(directory/'backend.json')
    env = os.environ.copy()
    if python:
        env['PATH'] = os.path.dirname(python) + os.pathsep + env.get('PATH', '')
    package_parent = str(Path(__file__).resolve().parent.parent)
    env['PYTHONPATH'] = package_parent + os.pathsep + env.get('PYTHONPATH', '')
    with (directory / 'worker.log').open('w', encoding='utf-8') as log:
        subprocess.run([python or sys.executable, '-m', 'generative_assembly.worker', str(directory / 'request.json')],
                       stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=timeout)
    return read(directory / 'backend.json')


def lock_models(config):
    """Resolve floating model references once, before study identity is created."""
    from huggingface_hub import HfApi
    api = HfApi()
    repos = {config['images'][m] for m in config['image_models']}
    if 'sd15_depth' in config['image_models']: repos.add(config['images']['controlnet'])
    config['images']['revisions'] = {r: api.model_info(r, revision=config['images']['revisions'].get(r, 'main')).sha for r in sorted(repos)}
    config['reconstruction']['revisions'] = {r: api.model_info(r, revision=config['reconstruction']['revisions'].get(r, 'main')).sha
                                          for r in ('sudo-ai/zero123plus-v1.2', 'TencentARC/InstantMesh')}
    if config['reconstruction']['backend'] == 'instantmesh':
        repo = Path(config['reconstruction']['repo']).resolve()
        config['reconstruction']['repo'] = str(repo)
        config['reconstruction']['commit'] = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
    return config


def run_gemini_image(req, out):
    """Generate intact object using Google Gemini multimodal API."""
    import os, io, base64, requests
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        env_file = Path(__file__).resolve().parent.parent / '.env'
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith('GEMINI_API_KEY='):
                    api_key = line.split('=', 1)[1].strip()
    if not api_key:
        raise ValueError('GEMINI_API_KEY not found in environment or .env file')

    with open(req['image'], 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    model_name = req.get('config', {}).get('gemini_model', 'gemini-2.5-flash-image')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    prompt = (
        f"{req['prompt']} Look at the provided 2D rendering of the broken fragment. "
        "Complete and render the full intact 3D object on a seamless pure white background (#FFFFFF) with neutral gray CAD shading."
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/png", "data": img_b64}}
            ]
        }]
    }
    resp = requests.post(url, json=payload, timeout=60)
    if resp.status_code == 200:
        data = resp.json()
        candidates = data.get('candidates', [])
        for c in candidates:
            for p in c.get('content', {}).get('parts', []):
                if 'inlineData' in p and p['inlineData'].get('data'):
                    raw_bytes = base64.b64decode(p['inlineData']['data'])
                    img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
                    img.save(out / 'image.png')
                    return dict(kind='gemini_api_image_generation', model=model_name, status='success')
    err_msg = resp.text[:300]
    if 'RESOURCE_EXHAUSTED' in err_msg or resp.status_code == 429:
        raise RuntimeError(f"Gemini API Quota Exceeded (429): Google AI Studio requires billing enabled for image generation models. Details: {err_msg}")
    raise RuntimeError(f"Gemini API error ({resp.status_code}): {err_msg}")


def run_image(req, out):
    cfg, model = req['config'], req['model']
    if req['smoke']:
        shutil.copy2(req['image'], out / 'image.png')
        return dict(kind='smoke_copy_NOT_image_completion', model=model, smoke=True)
    if model == 'gemini':
        return run_gemini_image(req, out)
    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, StableDiffusionInpaintPipeline, AutoPipelineForInpainting
    revisions = cfg['revisions']
    repo = cfg[model]
    if not revisions.get(repo) or len(revisions[repo]) != 40:
        raise ValueError('Run lock-models first; exact image checkpoint revisions required')
    dtype = getattr(torch, 'bfloat16' if model == 'qwen' else cfg['dtype'])
    kwargs = dict(torch_dtype=dtype, revision=revisions[repo], safety_checker=None)
    if model == 'sd15_depth':
        control_id = cfg['controlnet']
        control = ControlNetModel.from_pretrained(control_id, revision=revisions[control_id], torch_dtype=dtype)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(repo, controlnet=control, **kwargs)
    elif model == 'sd15':
        pipe = StableDiffusionInpaintPipeline.from_pretrained(repo, **kwargs)
    elif model == 'sdxl':
        pipe = AutoPipelineForInpainting.from_pretrained(repo, **kwargs)
    elif model == 'qwen':
        from diffusers import QwenImageEditPlusPipeline
        kwargs.pop('safety_checker', None) # Qwen pipeline doesn't accept safety_checker
        pipe = QwenImageEditPlusPipeline.from_pretrained(repo, **kwargs)
    else:
        raise ValueError(f'Unsupported image model {model}')
    if cfg['cpu_offload'] and cfg['device'].startswith('cuda'):
        pipe.enable_model_cpu_offload()
    else: pipe.to(cfg['device'])
    image = Image.open(req['image']).convert('RGB')
    params = dict(prompt=req['prompt'], generator=torch.Generator(device='cpu').manual_seed(req['seed']),
                  num_inference_steps=cfg['qwen_steps'] if model == 'qwen' else cfg['steps'])
    if model == 'qwen':
        params.update(image=[image, Image.open(req['control']).convert('RGB')], true_cfg_scale=cfg['qwen_cfg'], guidance_scale=1.0,
                      negative_prompt=' ', prompt=req['prompt']+' The second image is a depth rendering of the same input, not another object.')
    else:
        neg_prompt = 'vignette, shadow, dark background, grey background, floor, table, gradient, noisy, blurry, border, frame, studio background'
        params.update(image=image, mask_image=Image.open(req['mask']).convert('L'), width=image.width, height=image.height,
                      guidance_scale=cfg['guidance'], negative_prompt=neg_prompt)
        if model == 'sd15_depth':
            params.update(control_image=Image.open(req['control']).convert('RGB'), controlnet_conditioning_scale=cfg['control_strength'])
    with torch.inference_mode():
        result = pipe(**params)
    if getattr(result, 'nsfw_content_detected', None) and any(result.nsfw_content_detected):
        raise RuntimeError('Model returned a filtered image; record a generation failure')
    output_image = result.images[0]
    # ── FIX A: Mask-aware background stripping ───────────────────────────────
    # Strip any background pixel brighter than the rendered object (~90-150 grey)
    # to pure 255,255,255. Also clamp the outer border so InstantMesh gets clean edges.
    bg_obj_threshold = cfg.get('bg_obj_threshold', 140)
    if bg_obj_threshold > 0:
        import numpy as np
        arr = np.asarray(output_image).copy()
        if req.get('mask'):
            mask_arr = np.asarray(Image.open(req['mask']).convert('L').resize(
                (arr.shape[1], arr.shape[0]), Image.NEAREST))
            is_background = mask_arr > 127
            bg_to_strip = is_background & (arr.mean(axis=2) >= bg_obj_threshold)
            arr[bg_to_strip] = 255
        else:
            arr[np.all(arr >= bg_obj_threshold, axis=2)] = 255
        # Ensure canvas border (outer 4 pixels) is pure white
        arr[:4, :] = 255; arr[-4:, :] = 255; arr[:, :4] = 255; arr[:, -4:] = 255
        output_image = Image.fromarray(arr)
    # ─────────────────────────────────────────────────────────────────────────
    output_image.save(out / 'image.png')
    return dict(kind='pretrained_image_generation', model=repo, revision=revisions[repo], revisions=revisions, dtype=str(dtype),
                dimensions=list(output_image.size), diffusers=importlib.metadata.version('diffusers'), torch=torch.__version__,
                bg_obj_threshold=bg_obj_threshold)


def run_reconstruction(req, out):
    cfg = req['config']
    if req['smoke']:
        # Geometry substitute tests orchestration only. It cannot be exported as evidence.
        import trimesh
        mask = np.min(np.asarray(Image.open(req['image']).convert('RGB')), axis=2) < 235
        y, x = np.nonzero(mask)
        dims = [max(0.1, np.ptp(x)/mask.shape[1]) if len(x) else 0.1,
                max(0.1, np.ptp(y)/mask.shape[0]) if len(y) else 0.1, 0.3]
        trimesh.creation.box(extents=dims).export(out / 'mesh.obj')
        return dict(kind='smoke_box_NOT_3D_reconstruction', smoke=True)
    if cfg['backend'] == 'command':
        # User-supplied argv adapter: no shell, no fabricated fallback output.
        argv = [str(s).replace('{image}', req['image']).replace('{output}', str(out / 'mesh.obj')) for s in cfg['command']]
        subprocess.run(argv, check=True, timeout=cfg['timeout_seconds'])
        if not (out / 'mesh.obj').is_file(): raise RuntimeError('Custom backend must write {output}')
        return dict(kind='external_3d_command', argv=argv, declared_provenance=cfg.get('provenance'))
    repo = Path(cfg['repo']).resolve()
    commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
    if not cfg['commit'] or commit != cfg['commit']:
        raise ValueError('InstantMesh commit differs from model lock')
    dirty = subprocess.check_output(['git','-C',str(repo),'diff','--name-only','HEAD'], text=True).strip()
    if dirty: raise ValueError('InstantMesh tracked source has uncommitted changes')

    # ── Fix C: Multi-view grid packing ───────────────────────────────────────
    # When multiple SD completions exist from different camera angles, pack them
    # into a 2×3 grid image that InstantMesh accepts as Zero123++ bypass input.
    # This gives the LRM real geometrically-diverse views instead of 6 hallucinated
    # views from a single image, dramatically improving 3D reconstruction quality.
    input_image_path = req['image']
    multiview_images = req.get('multiview_images', [])
    grid_used = False
    if len(multiview_images) >= 4:
        try:
            import numpy as np
            views = [Image.open(p).convert('RGB') for p in multiview_images[:6]]
            # Pad to 6 views by repeating the last one if fewer than 6 provided
            while len(views) < 6:
                views.append(views[-1])
            w, h = views[0].size
            # Build 2 rows × 3 cols grid (Zero123++ canonical layout)
            grid = Image.new('RGB', (w * 3, h * 2), (255, 255, 255))
            for idx, v in enumerate(views):
                row, col = divmod(idx, 3)
                grid.paste(v, (col * w, row * h))
            grid_path = out / 'multiview_grid.png'
            grid.save(str(grid_path))
            input_image_path = str(grid_path)
            grid_used = True
        except Exception as e:
            # Graceful fallback: if grid assembly fails, use single image
            grid_used = False
    # ─────────────────────────────────────────────────────────────────────────

    # Pin upstream weight resolution without changing its source checkout.
    import huggingface_hub
    import diffusers
    original_download = huggingface_hub.hf_hub_download
    downloaded = []
    def download(*args, **kwargs):
        model = kwargs.get('repo_id', args[0] if args else None)
        if model in cfg['revisions']: kwargs['revision'] = cfg['revisions'][model]
        path = original_download(*args, **kwargs)
        downloaded.append(dict(path=str(path), sha256=digest(path)))
        return path
    huggingface_hub.hf_hub_download = download
    original_pretrained = diffusers.DiffusionPipeline.from_pretrained
    def pretrained(model, *args, **kwargs):
        if model not in cfg['revisions']: raise ValueError(f'Unpinned model: {model}')
        kwargs['revision'] = cfg['revisions'][model]
        return original_pretrained(model, *args, **kwargs)
    diffusers.DiffusionPipeline.from_pretrained = staticmethod(pretrained)
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    sys.argv = [str(repo/'run.py'), cfg['config'], input_image_path, '--output_path', str(out/'upstream'),
                '--diffusion_steps', str(cfg['steps']), '--seed', str(cfg['seed']), '--no_rembg']
    runpy.run_path(str(repo/'run.py'), run_name='__main__')
    meshes = list((out/'upstream').glob('*/meshes/*.obj'))
    if len(meshes) != 1: raise RuntimeError('Expected exactly one InstantMesh output mesh')
    shutil.copy2(meshes[0], out/'mesh.obj')
    return dict(kind='InstantMesh', commit=commit, run_py_sha256=digest(repo/'run.py'), revisions=cfg['revisions'],
                downloaded=downloaded, foreground_resize=False, diffusers=importlib.metadata.version('diffusers'),
                multiview_grid=grid_used, n_views_in_grid=len(multiview_images) if grid_used else 1)


def worker(path):
    req, out = read(path), Path(path).resolve().parent
    start = time.time()
    result = run_image(req, out) if req['kind'] == 'image' else run_reconstruction(req, out)
    result['seconds'] = time.time()-start
    try:
        import torch
        result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        result['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
    except ImportError: pass
    write(out/'backend.json', result)
