"""Deterministic CPU orthographic point-splat rendering with camera/depth provenance.

Changes vs original:
  - render_surface(): explicitly diagnostic Poisson surface; never silently falls back.
  - render_surfel(): bounded tangent support on original observed point samples.
  - tight_crop(): crops the canvas to the fragment silhouette with a small border margin,
    so the subject fills the frame instead of floating in a large white void.
  - inverted_mask(): returns a mask covering the BACKGROUND (region SD should fill in
    to complete the object), rather than the fragment itself.
  - camera_for() now accepts a n_views argument; when n_views > 1 it returns multiple
    camera bases (structured azimuths) rather than a single canonical view.
"""
import numpy as np
from scipy.ndimage import binary_dilation, label
from PIL import Image
from .geometry import frame
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# ── Camera ─────────────────────────────────────────────────────────────────────

def camera_for(points, pixels, extent, radius=2, n_views=1):
    """Return one or more camera dicts.

    n_views=1  → original behaviour: pick the best axis-aligned canonical view.
    n_views>1  → return n_views evenly-spaced azimuths around the object
                 (structured viewpoints; best for multi-view E1 rendering).
    """
    if n_views == 1:
        candidates = [frame(n) for n in np.concatenate((np.eye(3), -np.eye(3)))]
        scores = [int(render(points, None, None,
                             dict(basis=B.tolist(), pixels=pixels, extent=extent), radius)['valid'].sum())
                  for B in candidates]
        return dict(basis=candidates[int(np.argmax(scores))].tolist(),
                    pixels=pixels, extent=extent,
                    projection='orthographic', view_scores=scores,
                    selected_from='reference_fragment_only')

    # Multi-view: n_views equally-spaced rotations around the actual camera up axis,
    # starting from the canonical best view so the first camera is always comparable.
    base_cam = camera_for(points, pixels, extent, radius, n_views=1)
    B0 = np.asarray(base_cam['basis'])
    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c, s = np.cos(theta), np.sin(theta)
        # Rotate around the world-up axis (column 1 of B0)
        Rz = Rotation.from_rotvec(B0[:,1]*theta).as_matrix()
        B = Rz @ B0
        cameras.append(dict(basis=B.tolist(), pixels=pixels, extent=extent,
                            projection='orthographic', azimuth_index=i,
                            azimuth_deg=round(np.degrees(theta), 1)))
    return cameras


# ── Core point-splat renderer ──────────────────────────────────────────────────

def render(points, normals, exterior, camera, radius=2):
    size, extent = camera['pixels'], camera['extent']
    B = np.asarray(camera['basis'])
    q = (np.asarray(points)-np.asarray(camera.get('center',[0,0,0]))) @ B
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
    control = np.zeros(size*size, np.uint8)
    if valid.any():
        z = depth[valid]
        control[valid] = (40+200*(z-z.min())/max(z.max()-z.min(), 1e-6)).astype(np.uint8)
    result['control'] = np.repeat(control.reshape(size,size,1),3,axis=2)
    return result


def frame_observed(camera, points, fill):
    if not fill: return dict(camera)
    B = np.asarray(camera['basis']); q = np.asarray(points)@B
    center_q = (q.min(0)+q.max(0))/2
    extent = max(float(np.ptp(q[:,:2],axis=0).max())/(2*fill),1e-6)
    return dict(camera, center=(center_q@B.T).tolist(), extent=extent,
                framing_fill=fill, image_transform='input_derived_camera', projection='orthographic')


def render_surfel(points, normals, exterior, camera, radius=2):
    """Bounded tangent disks, not Poisson completion; ids reference observed samples."""
    points = np.asarray(points,float); normals = np.asarray(normals,float)
    if normals.shape != points.shape or not np.isfinite(normals).all(): raise ValueError('Invalid surfel normals')
    B = np.asarray(camera['basis']); size = camera['pixels']; extent = camera['extent']
    q = (points-np.asarray(camera.get('center',[0,0,0])))@B; ns = normals@B
    spacing = cKDTree(points).query(points,k=min(3,len(points)))[0][:,-1]
    cap = max(float(np.quantile(spacing,.9)),extent/(size-1))
    support = np.clip(spacing*1.2,extent/(size-1)*.6,cap*1.2)
    pixel_scale = (size-1)/(2*extent)
    depth = np.full((size,size),-np.inf); ids = np.full((size,size),-1,dtype=int)
    for i in np.argsort(q[:,2],kind='stable'):
        u = (q[i,0]/(2*extent)+.5)*(size-1); v = (.5-q[i,1]/(2*extent))*(size-1)
        pr = min(12,max(1,int(np.ceil(support[i]*pixel_scale))))
        x0,x1 = max(0,int(np.floor(u))-pr),min(size,int(np.ceil(u))+pr+1)
        y0,y1 = max(0,int(np.floor(v))-pr),min(size,int(np.ceil(v))+pr+1)
        if x0>=x1 or y0>=y1: continue
        ys,xs = np.mgrid[y0:y1,x0:x1]
        dx=(xs-u)/pixel_scale; dy=-(ys-v)/pixel_scale
        nz=ns[i,2]
        dz=-(ns[i,0]*dx+ns[i,1]*dy)/(np.copysign(max(abs(nz),.1),nz))
        good=dx*dx+dy*dy+dz*dz <= support[i]**2
        z=q[i,2]+dz; take=good & (z>depth[y0:y1,x0:x1])
        depth[y0:y1,x0:x1][take]=z[take]; ids[y0:y1,x0:x1][take]=i
    valid=ids>=0; rgb=np.full((size,size,3),255,np.uint8)
    shade=125+40*np.abs(ns[:,2]); rgb[valid]=shade[ids[valid],None].astype(np.uint8)
    protected=valid & (np.asarray(exterior)[np.maximum(ids,0)]>=.6) if exterior is not None else np.zeros_like(valid)
    control=np.zeros((size,size),np.uint8)
    if valid.any():
        z=depth[valid]; control[valid]=(40+200*(z-z.min())/max(np.ptp(z),1e-6)).astype(np.uint8)
    return dict(rgb=rgb,valid=valid,depth=np.where(valid,depth,0),ids=ids,protected=protected,
        control=np.repeat(control[:,:,None],3,axis=2),renderer_actual='surfel',
        support_radius_max=float(support.max()), observed_samples=len(points))


# ── Surface-mesh renderer (FIX #1) ─────────────────────────────────────────────

def render_surface(points, normals, exterior, camera, radius=2):
    """Attempt Poisson surface reconstruction → Phong-shaded mesh render.

    Fails explicitly if open3d is unavailable or the surface cannot be reconstructed.
    The result dict is identical to render() so callers are interchangeable.
    """
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points, float))
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, float))
        else:
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(k=10)

        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)
        mesh = mesh.simplify_quadric_decimation(4000)

        if len(np.asarray(mesh.triangles)) < 10:
            raise ValueError('Poisson diagnostic mesh too sparse')

        # Sample dense point cloud from mesh surface for rendering
        o3d.utility.random.seed(4101)
        sampled = mesh.sample_points_uniformly(number_of_points=max(4096, len(points)))
        pts_s = np.asarray(sampled.points, float)
        mesh.compute_vertex_normals()
        # Per-point normal via nearest vertex
        from scipy.spatial import cKDTree
        _, idx = cKDTree(np.asarray(mesh.vertices)).query(pts_s)
        ns_s = np.asarray(mesh.vertex_normals)[idx]

        # Re-use exterior labels by proximity to original points
        if exterior is not None:
            _, orig_idx = cKDTree(np.asarray(points)).query(pts_s)
            ext_s = np.asarray(exterior)[orig_idx]
        else:
            ext_s = None

        result = render(pts_s, ns_s, ext_s, camera, radius)
        result.update(renderer_actual='poisson_dense_splats', generated_surface_not_observed=True)
        return result

    except Exception as exc:
        raise RuntimeError(f'Explicit Poisson diagnostic renderer failed: {exc}') from exc


# ── Tight crop (FIX #6) ────────────────────────────────────────────────────────

def tight_crop(rendered, border_fraction=0.08, target_fill=0.65):
    """Crop the rendered image so the fragment fills ~65% of the canvas.

    Returns a new rendered dict with all arrays cropped and a new 'crop_box'
    key storing (y0, y1, x0, x1) of the crop in the original canvas.
    The mask, control, and depth arrays are all cropped consistently.
    """
    valid = rendered['valid']
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    if not rows.any():
        return rendered

    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    size = valid.shape[0]

    # Compute desired canvas size so the object fills target_fill of the frame
    obj_h = r1 - r0 + 1
    obj_w = c1 - c0 + 1
    pad = int(max(obj_h, obj_w) * border_fraction + 0.5)
    crop_size = int(max(obj_h, obj_w) / target_fill + 2 * pad)
    cy = (r0 + r1) // 2
    cx = (c0 + c1) // 2
    half = crop_size // 2

    y0 = max(0, cy - half); y1 = min(size, cy + half)
    x0 = max(0, cx - half); x1 = min(size, cx + half)

    def _crop2d(arr):
        if arr.ndim == 2:
            return arr[y0:y1, x0:x1]
        return arr[y0:y1, x0:x1]

    def _crop3d(arr):
        return arr[y0:y1, x0:x1]

    cropped = dict(
        rgb=_crop3d(rendered['rgb']),
        valid=_crop2d(rendered['valid']),
        depth=_crop2d(rendered['depth']),
        ids=_crop2d(rendered['ids']),
        protected=_crop2d(rendered['protected']),
        control=_crop3d(rendered['control']),
        crop_box=(int(y0), int(y1), int(x0), int(x1)),
        original_size=size,
    )
    # Resize all arrays back to original canvas size using PIL for SD compatibility
    target = (size, size)
    for key in ('rgb', 'control'):
        img = Image.fromarray(cropped[key].astype(np.uint8))
        cropped[key] = np.array(img.resize(target, Image.LANCZOS))
    for key in ('valid', 'protected'):
        img = Image.fromarray(cropped[key].astype(np.uint8) * 255)
        cropped[key] = np.array(img.resize(target, Image.NEAREST)) > 127
    for key in ('depth', 'ids'):
        img = Image.fromarray(cropped[key].astype(np.float32) if key == 'depth'
                              else cropped[key].astype(np.int32))
        cropped[key] = np.array(img.resize(target, Image.NEAREST))

    return cropped


def crop_render(rendered, box):
    """Apply an already measured input crop to an evaluator render."""
    y0, y1, x0, x1 = map(int, box)
    size = rendered['valid'].shape[0]
    result = dict(rendered)
    for key in ('rgb', 'control', 'valid', 'protected', 'depth', 'ids'):
        arr = rendered[key][y0:y1, x0:x1]
        if arr.dtype == bool:
            result[key] = np.asarray(Image.fromarray(arr.astype(np.uint8) * 255).resize(
                (size, size), Image.NEAREST)) > 127
        else:
            if key == 'ids': arr = arr.astype(np.int32)
            if key == 'depth': arr = arr.astype(np.float32)
            result[key] = np.asarray(Image.fromarray(arr).resize((size, size),
                Image.LANCZOS if key in ('rgb', 'control') else Image.NEAREST))
    result.update(crop_box=tuple(map(int, box)), original_size=size)
    return result


# ── Inverted mask (FIX #5) ──────────────────────────────────────────────────────

def inverted_mask(rendered):
    """Return a mask that covers the BACKGROUND (area to be inpainted by SD).

    Diffusers convention: 255 = regenerate, 0 = preserve.
    This legacy arm protects the whole fragment and a four-pixel surrounding rim.
    """
    # Dilate the fragment silhouette slightly so SD blends cleanly at edges
    fragment_mask = binary_dilation(rendered['valid'], iterations=4)
    # Background is white (editable); fragment is black (protected).
    return (~fragment_mask).astype(np.uint8) * 255


# ── Save helpers ────────────────────────────────────────────────────────────────

def save(directory, rendered, use_inverted_mask=False, mask_type=None):
    """Save E0 rendering artifacts.

    use_inverted_mask=True  → save inverted mask (background to fill in).
    use_inverted_mask=False → preserve estimated exterior, edit remaining pixels.
    Both are saved so experiments can compare them.
    """
    Image.fromarray(rendered['rgb']).save(directory / 'image.png')
    Image.fromarray(rendered['control']).save(directory / 'control.png')
    Image.fromarray(rendered['valid'].astype(np.uint8)*255).save(directory/'observed_mask.png')
    Image.fromarray(rendered['protected'].astype(np.uint8)*255).save(directory/'exterior_mask.png')
    # Always save both mask variants so E1 arms can select
    orig_mask = (~rendered['protected']).astype(np.uint8) * 255
    inv_mask  = inverted_mask(rendered)
    Image.fromarray(orig_mask).save(directory / 'mask_original.png')
    Image.fromarray(inv_mask).save(directory / 'mask_inverted.png')
    # Default mask used by downstream code: choose which one to use
    masks = {'original': orig_mask, 'exterior': orig_mask, 'inverted': inv_mask,
             'none': np.full_like(orig_mask, 255)}
    active = mask_type or ('inverted' if use_inverted_mask else 'original')
    Image.fromarray(masks[active]).save(directory / 'mask.png')
    np.savez_compressed(directory / 'render.npz',
                        **{k: v for k, v in rendered.items() if k not in ('rgb', 'control')})


def foreground(image):
    a = np.asarray(image.convert('RGB'), float)
    return np.min(a, axis=2) < 235


def image_score(path, input_render):
    image = Image.open(path).convert('RGB')
    size = input_render['valid'].shape[0]
    if image.size != (size, size):
        resized = True; image = image.resize((size, size))
    else:
        resized = False
    mask = foreground(image)
    p = input_render['protected']
    retention = float(mask[p].mean()) if p.any() else None
    valid = bool(mask.mean() > 0.01 and mask.mean() < 0.95)
    observed = input_render['valid'].astype(bool)
    added = mask & ~observed
    growth = float(added.sum() / max(1, observed.sum()))
    components, count = label(mask)
    sizes = np.bincount(components.ravel())[1:]
    significant_components = int((sizes >= max(8, mask.size * 0.001)).sum())
    border = float(np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1])).mean())
    # Input-only heuristic: retention alone must not reward an unchanged fragment.
    # Growth is diagnostic evidence, not proof of correct geometry.
    score = (1-(retention if retention is not None else 0.5)) + (0 if valid else 10)
    score += 0.25 * max(0., 1. - growth / 0.1) + border + 0.1 * max(0, significant_components - 1)
    return dict(selection_score=score, observed_mask_retention=retention, valid_foreground=valid,
                scoring_resized=resized, foreground_fraction=float(mask.mean()),
                crop_border_fraction=border, added_foreground_fraction=float(added.mean()),
                growth_relative_to_input=growth, significant_components=significant_components,
                selector='retention_growth_components_border_then_fixed_seed',
                camera_preservation_verified=False)
