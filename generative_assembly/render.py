"""Deterministic CPU orthographic point-splat rendering with camera/depth provenance."""
import numpy as np
from scipy.ndimage import binary_dilation
from PIL import Image
from .geometry import frame


def camera_for(points, pixels, extent, radius=2):
    candidates = [frame(n) for n in np.concatenate((np.eye(3), -np.eye(3)))]
    scores = [int(render(points, None, None, dict(basis=B.tolist(), pixels=pixels, extent=extent), radius)['valid'].sum()) for B in candidates]
    return dict(basis=candidates[int(np.argmax(scores))].tolist(), pixels=pixels, extent=extent,
                projection='orthographic', view_scores=scores, selected_from='reference_fragment_only')


def render(points, normals, exterior, camera, radius=2):
    size, extent = camera['pixels'], camera['extent']
    B = np.asarray(camera['basis'])
    q = np.asarray(points) @ B
    u = np.rint((q[:, 0]/(2*extent)+0.5)*(size-1)).astype(int)
    v = np.rint((0.5-q[:, 1]/(2*extent))*(size-1)).astype(int)
    depth = np.full(size*size, -np.inf)
    ids = np.full(size*size, -1, dtype=int)
    order = np.argsort(q[:, 2])
    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            if dx*dx+dy*dy > radius*radius: continue
            x, y = u[order]+dx, v[order]+dy
            good = (x >= 0) & (x < size) & (y >= 0) & (y < size)
            idx, pi = y[good]*size+x[good], order[good]
            # Keep the last occurrence per pixel, i.e. the nearest point.
            _, rev = np.unique(idx[::-1], return_index=True)
            take = len(idx)-1-rev; idx, pi = idx[take], pi[take]
            update = q[pi, 2] > depth[idx]
            depth[idx[update]], ids[idx[update]] = q[pi[update], 2], pi[update]
    valid = ids >= 0
    rgb = np.full((size*size, 3), 255, dtype=np.uint8)
    shade = np.full(len(points), 150.) if normals is None else 90 + 90*np.abs(np.asarray(normals) @ B[:, 2])
    rgb[valid] = shade[ids[valid], None].astype(np.uint8)
    protected = np.zeros(size*size, bool)
    if exterior is not None: protected[valid] = np.asarray(exterior)[ids[valid]] >= 0.6
    result = dict(rgb=rgb.reshape(size,size,3), valid=valid.reshape(size,size),
                  depth=np.where(valid, depth, 0).reshape(size,size), ids=ids.reshape(size,size),
                  protected=protected.reshape(size,size))
    # A validity mask accompanies depth; zero is not interpreted as free space.
    control = np.zeros(size*size, np.uint8)
    if valid.any():
        z = depth[valid]
        control[valid] = (40+200*(z-z.min())/max(z.max()-z.min(), 1e-6)).astype(np.uint8)
    result['control'] = np.repeat(control.reshape(size,size,1),3,axis=2)
    return result


def save(directory, rendered):
    Image.fromarray(rendered['rgb']).save(directory / 'image.png')
    Image.fromarray(rendered['control']).save(directory / 'control.png')
    Image.fromarray((~rendered['protected']).astype(np.uint8)*255).save(directory / 'mask.png')
    np.savez_compressed(directory / 'render.npz', **{k:v for k,v in rendered.items() if k not in ('rgb','control')})


def foreground(image):
    a = np.asarray(image.convert('RGB'), float)
    return np.min(a, axis=2) < 235


def image_score(path, input_render):
    image = Image.open(path).convert('RGB')
    size = input_render['valid'].shape[0]
    if image.size != (size,size):
        # Size changes are logged, and this transform is used for scoring only.
        resized = True; image = image.resize((size,size))
    else: resized = False
    mask = foreground(image)
    p = input_render['protected']
    retention = float(mask[p].mean()) if p.any() else None
    valid = bool(mask.mean() > 0.01 and mask.mean() < 0.95)
    score = (1-(retention if retention is not None else 0.5)) + (0 if valid else 10)
    return dict(selection_score=score, observed_mask_retention=retention, valid_foreground=valid,
                scoring_resized=resized, foreground_fraction=float(mask.mean()),
                crop_border_fraction=float(np.concatenate((mask[0],mask[-1],mask[:,0],mask[:,-1])).mean()),
                selector='input_mask_retention_then_fixed_seed_tie_break', camera_preservation_verified=False)
