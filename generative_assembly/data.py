"""Public fragment inputs and evaluator-only references are physically separate."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation
from .storage import checked, digest, read, write


def seed_for(*items):
    return int(hashlib.sha256('|'.join(map(str, items)).encode()).hexdigest()[:8], 16)


def sample_mesh(mesh, count, rng):
    area = np.asarray(mesh.area_faces)
    if area.sum() <= 0:
        raise ValueError('Empty mesh surface')
    f = rng.choice(len(area), count, p=area / area.sum())
    uv = rng.random((count, 2))
    uv[uv.sum(1) > 1] = 1 - uv[uv.sum(1) > 1]
    tri = np.asarray(mesh.triangles)[f]
    points = tri[:, 0] + uv[:, :1] * (tri[:, 1] - tri[:, 0]) + uv[:, 1:] * (tri[:, 2] - tri[:, 0])
    return points, np.asarray(mesh.face_normals)[f]


def xyz(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) < 16 or not np.isfinite(a).all():
        raise ValueError('Each fragment requires >=16 finite XYZ points')
    return a


def inventory(path):
    doc = read(path)
    if doc.get('schema_version') != 1 or not doc.get('cases'):
        raise ValueError('Expected schema_version=1 dataset with nonempty cases')
    ids, sources, hashes = set(), {}, {}
    for c in doc['cases']:
        if c['id'] in ids or c['split'] not in ('train', 'dev', 'test', 'real'):
            raise ValueError('Duplicate case ID or invalid split')
        ids.add(c['id'])
        for mapping, key in ((sources, c['source_id']), (hashes, c.get('source_hash', c['source_id']))):
            if key in mapping and mapping[key] != c['split']:
                raise ValueError('Source or identical source geometry crosses splits')
            mapping[key] = c['split']
        asset = checked(Path(path).parent, c['path'])
        if digest(asset) != c['sha256']:
            raise ValueError(f'Input hash mismatch: {asset}')
        with np.load(asset, allow_pickle=False) as f:
            expected = {f'points_{i}' for i in range(c['pieces'])}
            allowed = expected | {f'normals_{i}' for i in range(c['pieces'])}
            if not expected.issubset(f.files) or set(f.files) - allowed:
                raise ValueError('Public NPZ permits only points_i and optional input normals_i; labels belong in evaluator_only')
            if c['pieces'] < 2:
                raise ValueError('At least two fragments required')
            for key in expected:
                xyz(f[key])
    return doc


def load_case(dataset_path, record, cfg, seed=None):
    """Never opens evaluator data; returns normalization and original point indices."""
    from .geometry import normals_features
    rng = np.random.default_rng(seed_for(cfg['seed'] if seed is None else seed, record['id']))
    path = checked(Path(dataset_path).parent, record['path'])
    if digest(path) != record['sha256']:
        raise ValueError('Changed input asset')
    with np.load(path, allow_pickle=False) as f:
        original = [xyz(f[f'points_{i}']) for i in range(record['pieces'])]
        centers = np.stack([p.mean(0) for p in original])
        # Fixed index sample depends only on immutable observed data, never GT.
        diameters = [float(pdist(p[np.linspace(0, len(p)-1, min(len(p), 1024)).astype(int)]).max()) for p in original]
        scale = max(diameters)
        if scale <= 1e-10:
            raise ValueError('Degenerate fragments')
        anchor = int(np.argmax(diameters))
        indices = [rng.choice(len(p), min(cfg['points'], len(p)), replace=False) for p in original]
        pts, normals, exterior = [], [], []
        for i, p in enumerate(original):
            q = (p[indices[i]] - centers[i]) / scale
            n, e = normals_features(q)
            if f'normals_{i}' in f.files:
                n = np.asarray(f[f'normals_{i}'], float)[indices[i]]
                if n.shape != q.shape or not np.isfinite(n).all() or (np.linalg.norm(n, axis=1) < 1e-8).any():
                    raise ValueError('Invalid observed normals')
                n /= np.linalg.norm(n, axis=1, keepdims=True)
            pts.append(q); normals.append(n); exterior.append(e)
    return dict(record=record, points=pts, normals=normals, exterior=exterior, original=original,
                indices=indices, centers=centers, scale=scale, anchor=anchor,
                exterior_method='fixed_local_curvature_heuristic_not_true_fracture_labels')


def reference(dataset_path, case):
    base = Path(dataset_path).parent
    index_path = base / 'evaluator_only' / 'index.json'
    if not index_path.exists():
        return None
    entry = read(index_path).get(case['record']['id'])
    if entry is None:
        return None
    path = checked(base, entry['path'])
    if digest(path) != entry['sha256']:
        raise ValueError('Changed evaluator reference')
    with np.load(path, allow_pickle=False) as f:
        world = np.asarray(f['world_transforms'], float)
        if world.shape != (len(case['points']), 4, 4):
            raise ValueError('Reference pose count mismatch')
        inverse = np.linalg.inv(world[case['anchor']])
        poses = inverse[None] @ world
        target = poses.copy()
        target[:, :3, 3] = (np.einsum('fij,fj->fi', poses[:, :3, :3], case['centers']) + poses[:, :3, 3] - case['centers'][case['anchor']]) / case['scale']
        result = dict(poses=target, original_poses=poses)
        if 'complete' in f.files:
            q = np.asarray(f['complete']) @ inverse[:3, :3].T + inverse[:3, 3]
            result['complete'] = (q - case['centers'][case['anchor']]) / case['scale']
        result['exterior'] = [np.asarray(f[f'exterior_{i}'])[idx].astype(bool) if f'exterior_{i}' in f.files else None for i, idx in enumerate(case['indices'])]
    return result


def save_case(root, source_id, case_id, split, category, points, normals=None, world=None, complete=None, exterior=None, source_hash=None):
    root = Path(root)
    slug = hashlib.sha256(case_id.encode()).hexdigest()[:20]
    p = root / 'inputs' / f'{slug}.npz'
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {f'points_{i}': xyz(q).astype(np.float32) for i, q in enumerate(points)}
    if normals is not None:
        payload.update({f'normals_{i}': np.asarray(n, np.float32) for i, n in enumerate(normals)})
    np.savez_compressed(p, **payload)
    entry = dict(id=case_id, source_id=source_id, split=split, category=category, pieces=len(points),
                 path=str(p.relative_to(root)).replace('\\', '/'), sha256=digest(p), source_hash=source_hash or source_id)
    ref = None
    if world is not None:
        path = root / 'evaluator_only' / f'{slug}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        values = dict(world_transforms=np.asarray(world))
        if complete is not None:
            values['complete'] = np.asarray(complete)
        if exterior is not None:
            values.update({f'exterior_{i}': np.asarray(e) for i, e in enumerate(exterior)})
        np.savez_compressed(path, **values)
        ref = dict(path=str(path.relative_to(root)).replace('\\', '/'), sha256=digest(path))
    return entry, ref


def _finish(root, rows, refs, provenance):
    write(Path(root) / 'dataset.json', dict(schema_version=1, cases=rows, provenance=provenance))
    write(Path(root) / 'evaluator_only' / 'index.json', refs)
    inventory(Path(root) / 'dataset.json')


def demo(root, count=8, seed=42):
    """Small box partitions for infrastructure checks, NOT a fracture benchmark."""
    import trimesh
    root = Path(root)
    if (root / 'dataset.json').exists():
        raise ValueError('Dataset already exists; choose a new output directory')
    if count < 6:
        raise ValueError('Demo needs >=6 sources for train/dev/test')
    rng = np.random.default_rng(seed)
    rows, refs = [], {}
    for j in range(count):
        dims = rng.uniform(0.7, 1.4, 3)
        shape = trimesh.creation.box(extents=dims)
        full, _ = sample_mesh(shape, 4096, rng)
        points, ns, ts, ext = [], [], [], []
        cuts = [-dims[0]/2, rng.uniform(-0.1, 0.1), dims[0]/2]
        for i in range(2):
            d = dims.copy(); d[0] = cuts[i+1] - cuts[i]
            mesh = trimesh.creation.box(extents=d)
            mesh.apply_translation([(cuts[i]+cuts[i+1])/2, 0, 0])
            p, n = sample_mesh(mesh, 1024, rng)
            e = np.any(np.isclose(np.abs(p), dims/2, atol=1e-6), axis=1)
            r, t = Rotation.random(random_state=rng).as_matrix(), rng.uniform(-2, 2, 3)
            points.append(p @ r.T + t); ns.append(n @ r.T)
            T = np.eye(4); T[:3, :3] = r.T; T[:3, 3] = -r.T @ t
            ts.append(T); ext.append(e)
        split = 'train' if j < count-4 else ('dev' if j < count-2 else 'test')
        c, ref = save_case(root, f'demo-{j}', f'demo-{j}-p2', split, 'box_smoke', points, ns, ts, full, ext)
        rows.append(c); refs[c['id']] = ref
    _finish(root, rows, refs, dict(kind='smoke_box_partitions_not_research_evidence', seed=seed))


def import_v2(manifest, root, seed=42):
    """Export existing prepared v2 reservoirs; supervision is never public."""
    src, root = Path(manifest), Path(root)
    if (root / 'dataset.json').exists():
        raise ValueError('Refusing to overwrite a dataset')
    doc = read(src)
    if doc.get('schema_version') != 2:
        raise ValueError('import-v2 requires the original schema_version=2 prepared manifest')
    sources = {s['source_id']: s for s in doc['sources']}
    rows, refs = [], {}
    for c in doc['patterns']:
        source = sources[c['source_id']]
        path, sp = checked(src.parent, c['path']), checked(src.parent, source['path'])
        if digest(path) != c['sha256'] or digest(sp) != source['sha256']:
            raise ValueError('Original prepared asset hash mismatch')
        rng = np.random.default_rng(seed_for(seed, c['pattern_id']))
        with np.load(path, allow_pickle=False) as f, np.load(sp, allow_pickle=False) as s:
            points, transforms = [], []
            for p in f['points']:
                r, t = Rotation.random(random_state=rng).as_matrix(), rng.uniform(-1, 1, 3)
                points.append(p @ r.T + t)
                T = np.eye(4); T[:3, :3] = r.T; T[:3, 3] = -r.T @ t; transforms.append(T)
            ext = [x == 0 for x in f['fracture_labels']] if 'fracture_labels' in f.files else None
            split = {'val': 'dev', 'validation': 'dev', 'cut_holdout': 'test'}.get(c['split'], c['split'])
            row, ref = save_case(root, c['source_id'], c['pattern_id'], split, c.get('category', 'bottle'),
                                 points, world=transforms, complete=s['target_points'], exterior=ext, source_hash=source['sha256'])
            row.update(original_split=c['split'],cut_family=c.get('cut_family'),difficulty_band=c.get('band'))
        rows.append(row); refs[row['id']] = ref
    _finish(root, rows, refs, dict(kind='prepared_v2_export', manifest_sha256=digest(src), source_fingerprint=doc.get('fingerprint'), seed=seed,
                                 warning='Previously inspected test sources are not newly blind'))


def import_spec(spec, root, seed=42):
    """JSON-described original scans or canonical mesh fragments (e.g. Breaking Bad)."""
    import trimesh
    spec, root = Path(spec), Path(root)
    if (root / 'dataset.json').exists():
        raise ValueError('Refusing to overwrite a dataset')
    rows, refs = [], {}
    for c in read(spec)['cases']:
        rng = np.random.default_rng(seed_for(seed, c['id']))
        points, normals, world = [], [], []
        for item in c['fragments']:
            path = (spec.parent / item).resolve()
            if path.suffix == '.npy':
                p = xyz(np.load(path, allow_pickle=False)); n = None
            else:
                mesh = trimesh.load(path, process=False)
                if isinstance(mesh,trimesh.Scene): mesh=mesh.to_geometry()
                if len(getattr(mesh,'faces',[])):
                    p, n = sample_mesh(mesh, c.get('reservoir_points', 4096), rng)
                else:
                    p, n = xyz(mesh.vertices), None
            if c.get('canonical_fragments', False):
                r, t = Rotation.random(random_state=rng).as_matrix(), rng.uniform(-1, 1, 3)
                p = p @ r.T + t
                if n is not None: n = n @ r.T
                T = np.eye(4); T[:3, :3] = r.T; T[:3, 3] = -r.T @ t; world.append(T)
            points.append(p); normals.append(n)
        complete = None
        if c.get('complete_mesh'):
            mesh = trimesh.load((spec.parent / c['complete_mesh']).resolve(), force='mesh', process=False)
            complete, _ = sample_mesh(mesh, 8192, rng)
        if not world and c.get('world_transforms'):
            world = np.load((spec.parent / c['world_transforms']).resolve(), allow_pickle=False)
        row, ref = save_case(root, c['source_id'], c['id'], c['split'], c.get('category', 'unknown'), points,
                             normals if all(n is not None for n in normals) else None,
                             world if len(world) else None, complete, source_hash=c.get('source_hash'))
        rows.append(row)
        if ref: refs[row['id']] = ref
    _finish(root, rows, refs, dict(kind='scan_or_canonical_mesh_spec', spec_sha256=digest(spec), seed=seed))
