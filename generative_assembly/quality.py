"""Input-only object mattes and explicit image-to-reconstruction transforms."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label
from .storage import digest, write


def score_alpha(alpha, observed, protected, cfg):
    alpha = np.asarray(alpha, dtype=np.uint8)
    size = observed.shape
    resized=alpha.shape != size
    if resized:
        alpha = np.asarray(Image.fromarray(alpha).resize((size[1], size[0]), Image.Resampling.NEAREST))
    mask = alpha >= cfg.get('threshold', 128)
    occupancy = float(mask.mean())
    retention = float(mask[protected].mean()) if protected.any() else None
    border = float(np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1])).mean())
    components, _ = label(mask)
    counts = np.bincount(components.ravel())[1:]
    dominant = float(counts.max() / max(1, mask.sum())) if len(counts) else 0.
    reasons = []
    if not cfg.get('min_occupancy', .01) <= occupancy <= cfg.get('max_occupancy', .85): reasons.append('occupancy')
    if retention is None: reasons.append('protected_support_unavailable')
    elif retention < cfg.get('min_retention', .95): reasons.append('protected_region_loss')
    if border > cfg.get('max_border', .02): reasons.append('clipped_or_background')
    if dominant < cfg.get('min_dominant_fraction', .95): reasons.append('multiple_or_disconnected_objects')
    added = mask & ~observed
    growth = float(added.sum() / max(1, observed.sum()))
    return dict(valid_foreground=not reasons, rejection_reasons=reasons,
        foreground_fraction=occupancy, observed_mask_retention=retention,
        crop_border_fraction=border, dominant_component_fraction=dominant,
        significant_components=int(sum(counts >= max(8, mask.size*.001))),
        growth_relative_to_input=growth, added_foreground_fraction=float(added.mean()),
        selection_score=len(reasons)*10 + border + (1-dominant) +
                        (1-(retention if retention is not None else .5)) + max(0., .05-growth),
        selector='object_matte_input_only', camera_preservation_verified=False,
        camera_correspondence='unverified_requires_review', scoring_resized=resized)


def save_matte(image, alpha, out, provenance):
    out = Path(out)
    image = image.convert('RGB')
    alpha = Image.fromarray(np.asarray(alpha, np.uint8), 'L')
    if alpha.size != image.size: raise ValueError('Matte dimensions differ from generated image')
    rgba = image.copy(); rgba.putalpha(alpha)
    white = Image.new('RGB', image.size, 'white'); white.paste(image, mask=alpha)
    alpha.save(out/'alpha.png'); rgba.save(out/'image_rgba.png'); white.save(out/'image_white.png')
    white.save(out/'image.png')
    write(out/'matte.json', provenance)


def segment(req, out):
    cfg = req['config']; weights = Path(cfg['weights'] or '')
    if weights.name!='isnet-general-use.onnx': raise ValueError('Pinned matte filename must match the rembg model actually loaded')
    if not weights.is_file() or digest(weights) != cfg.get('sha256'):
        raise ValueError('Missing/changed pinned matte weights; run lock-matte first')
    if cfg['model'] != 'isnet-general-use': raise ValueError('Only explicitly selected isnet-general-use is supported')
    import os
    # rembg 2.0.67 expects the ONNX file in U2NET_HOME. No remote backend/default model.
    os.environ['U2NET_HOME'] = str(weights.parent)
    from rembg import new_session, remove
    session = new_session(cfg['model'], providers=['CPUExecutionProvider'])
    image = Image.open(req['image']).convert('RGB')
    result = remove(image, session=session, only_mask=True)
    alpha = np.asarray(result.convert('L'), np.uint8)
    if digest(weights) != cfg['sha256']: raise ValueError('Matte checkpoint changed during session initialization')
    provenance = dict(model=cfg['model'], sha256=cfg['sha256'], provider='CPUExecutionProvider',
                      source='predicted_not_ground_truth', dimensions=list(image.size))
    save_matte(image, alpha, out, provenance)
    return provenance


def normalize_foreground(path, destination, fill=.85):
    with Image.open(path) as source: image=source.copy()
    if image.mode != 'RGBA': raise ValueError('Validated reconstruction input requires RGBA')
    arr = np.asarray(image); support = arr[:, :, 3] >= 128
    y, x = np.nonzero(support)
    if not len(x): raise ValueError('Empty reconstruction matte')
    x0, x1, y0, y1 = int(x.min()), int(x.max()+1), int(y.min()), int(y.max()+1)
    side = max(x1-x0, y1-y0)
    canvas = int(np.ceil(side/fill))
    ox, oy = (canvas-(x1-x0))//2, (canvas-(y1-y0))//2
    output = Image.new('RGBA', (canvas, canvas), (255,255,255,0))
    output.paste(image.crop((x0,y0,x1,y1)), (ox,oy))
    # Explicit constant-size input to Zero123++; alpha is retained, not re-estimated.
    output.resize((512,512), Image.Resampling.LANCZOS).save(destination)
    transform = dict(crop_box=[x0,y0,x1,y1], padded_size=canvas, offset=[ox,oy],
        output_size=512, fill=fill, source_dimensions=list(image.size),
        affine_source_to_output=[[512/canvas,0,(ox-x0)*512/canvas],
                                 [0,512/canvas,(oy-y0)*512/canvas],[0,0,1]],
        separate_from_e0_camera=True)
    write(Path(destination).with_suffix('.json'), transform)
    return transform


def shape_metrics(predicted, complete, observed, threshold=.02):
    """Evaluator-only missing-surface scores; never used by deployment selection."""
    from scipy.spatial import cKDTree
    d = cKDTree(observed).query(complete)[0]
    missing = complete[d > threshold]
    if not len(missing): return dict(missing_surface_available=False)
    additions = predicted[cKDTree(observed).query(predicted)[0] > threshold]
    distances = cKDTree(predicted).query(missing)[0]
    precision = float((cKDTree(missing).query(additions)[0] < threshold).mean()) if len(additions) else 0.
    recall = float((distances < threshold).mean())
    return dict(missing_surface_available=True, missing_surface_distance=float(distances.mean()),
        missing_surface_precision=precision, missing_surface_recall=recall,
        missing_surface_fscore=2*precision*recall/max(precision+recall,1e-12),
        missing_surface_threshold=threshold, missing_surface_points=len(missing))


def evaluator_observed(case,ref,kind):
    from .geometry import apply
    pieces=[case['anchor']] if kind=='F' else list(range(len(case['original'])))
    normalized=[(p[np.linspace(0,len(p)-1,min(8192,len(p))).astype(int)]-case['centers'][i])/case['scale']
                for i,p in enumerate(case['original'])]
    return np.concatenate([normalized[i] if kind=='F' else apply(normalized[i],ref['poses'][i]) for i in pieces])
