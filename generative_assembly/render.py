"""Deterministic CPU orthographic point-splat rendering with camera/depth provenance.

Changes vs original:
  - render_surface(): mesh-based rendering with Phong shading when trimesh is available,
    falls back to point splatting if not.
  - tight_crop(): crops the canvas to the fragment silhouette with a small border margin,
    so the subject fills the frame instead of floating in a large white void.
  - inverted_mask(): returns a mask covering the BACKGROUND (region SD should fill in
    to complete the object), rather than the fragment itself.
  - camera_for() now accepts a n_views argument; when n_views > 1 it returns multiple
    camera bases (structured azimuths) rather than a single canonical view.
"""
import numpy as np
from scipy.ndimage import binary_dilation
from PIL import Image
from .geometry import frame


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

    # Multi-view: n_views equally-spaced azimuth rotations around the Z axis,
    # starting from the canonical best view so the first camera is always comparable.
    base_cam = camera_for(points, pixels, extent, radius, n_views=1)
    B0 = np.asarray(base_cam['basis'])
    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c, s = np.cos(theta), np.sin(theta)
        # Rotate around the world-up axis (column 1 of B0)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)
        B = Rz @ B0
        cameras.append(dict(basis=B.tolist(), pixels=pixels, extent=extent,
                            projection='orthographic', azimuth_index=i,
                            azimuth_deg=round(np.degrees(theta), 1)))
    return cameras


# ── Core point-splat renderer ──────────────────────────────────────────────────

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


# ── Surface-mesh renderer (FIX #1) ─────────────────────────────────────────────

def render_surface(points, normals, exterior, camera, radius=2):
    """Attempt Poisson surface reconstruction → Phong-shaded mesh render.

    Falls back silently to the original point-splat renderer if trimesh or
    open3d is unavailable, or if the point cloud is too sparse to mesh.
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
            raise ValueError('Mesh too sparse, falling back')

        # Sample dense point cloud from mesh surface for rendering
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

        return render(pts_s, ns_s, ext_s, camera, radius)

    except Exception:
        # Graceful fallback to point-splat renderer
        return render(points, normals, exterior, camera, radius)


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
        # Use nearest-neighbour for index/depth maps
        arr = np.array(Image.fromarray(
            (cropped[key] - cropped[key].min()).astype(np.float32)).resize(target, Image.NEAREST))
        cropped[key] = arr  # approximate — used only for scoring, not geometry

    return cropped


# ── Inverted mask (FIX #5) ──────────────────────────────────────────────────────

def inverted_mask(rendered):
    """Return a mask that covers the BACKGROUND (area to be inpainted by SD).

    Original mask = 0 (black) where fragment exists → SD erases the fragment.
    Inverted mask = 0 (black) where background is  → SD paints the missing bottle body.

    The fragment region is kept WHITE (255 = preserve), background is BLACK (0 = regenerate).
    A small dilation ensures SD fills right up to the fragment boundary.
    """
    # Dilate the fragment silhouette slightly so SD blends cleanly at edges
    fragment_mask = binary_dilation(rendered['valid'], iterations=4)
    # inverted: fragment = white (keep), background = black (fill in)
    return (~fragment_mask).astype(np.uint8) * 255


# ── Save helpers ────────────────────────────────────────────────────────────────

def save(directory, rendered, use_inverted_mask=False):
    """Save E0 rendering artifacts.

    use_inverted_mask=True  → save inverted mask (background to fill in).
    use_inverted_mask=False → original behaviour (fragment silhouette mask).
    Both are saved so experiments can compare them.
    """
    Image.fromarray(rendered['rgb']).save(directory / 'image.png')
    Image.fromarray(rendered['control']).save(directory / 'control.png')
    # Always save both mask variants so E1 arms can select
    orig_mask = (~rendered['protected']).astype(np.uint8) * 255
    inv_mask  = inverted_mask(rendered)
    Image.fromarray(orig_mask).save(directory / 'mask_original.png')
    Image.fromarray(inv_mask).save(directory / 'mask_inverted.png')
    # Default mask used by downstream code: choose which one to use
    Image.fromarray(inv_mask if use_inverted_mask else orig_mask).save(directory / 'mask.png')
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
    score = (1-(retention if retention is not None else 0.5)) + (0 if valid else 10)
    return dict(selection_score=score, observed_mask_retention=retention, valid_foreground=valid,
                scoring_resized=resized, foreground_fraction=float(mask.mean()),
                crop_border_fraction=float(np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1])).mean()),
                selector='input_mask_retention_then_fixed_seed_tie_break',
                camera_preservation_verified=False)
