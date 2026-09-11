"""Controlled reservoir resampling/reposing, isolated from the production loader."""
import copy
from collections import defaultdict, deque

import numpy as np
import torch

from reassembly.data import _seed, random_rotation


def choose_subset(records, limit=12):
    """Round robin sources, then available band/piece groups, independent of predictions."""
    buckets = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        buckets[record['source_id']][(record['band'], record['pieces'])].append(index)
    queues = {}
    for source_ordinal,source in enumerate(sorted(buckets,key=lambda s:_seed(42,s,'diagnostic'))):
        groups=buckets[source]
        groups = {key: deque(sorted(value, key=lambda i: _seed(42, records[i]['pattern_id'], 'diagnostic')))
                  for key, value in groups.items()}
        ordered = []
        keys=sorted(groups)
        offset=source_ordinal%len(keys); keys=keys[offset:]+keys[:offset]
        while any(groups.values()):
            for key in keys:
                if groups[key]:
                    ordered.append(groups[key].popleft())
        queues[source] = deque(ordered)
    result = []
    while len(result) < min(limit, len(records)):
        for source in sorted(queues, key=lambda s: _seed(42, s, 'diagnostic')):
            if queues[source] and len(result) < limit:
                result.append(queues[source].popleft())
    return result


def variant(dataset, index, mode='unchanged', seed=0):
    base = dataset[index]
    if mode == 'unchanged':
        return base
    rng_new = np.random.default_rng(_seed(seed, base['pattern_id'], mode))
    result = copy.deepcopy(base)
    n = base['points'].shape[1]
    if mode == 'permutation':
        for fragment in range(int(base['fragment_mask'].sum())):
            indices = rng_new.permutation(n)
            for key in ('points', 'points_view2', 'canonical_points', 'original_points',
                        'fracture_labels', 'interface_ids'):
                result[key][fragment] = base[key][fragment, indices]
        return result
    if mode not in ('identity_rebuild', 'rotation', 'translation', 'samples', 'samples_poses'):
        raise ValueError(mode)
    record = dataset.records[index]
    rng = np.random.default_rng(_seed(dataset.seed, record['pattern_id'], 'fixed'))
    data = dataset.cfg['data']
    with np.load(dataset.root / record['path'], allow_pickle=False) as archive, \
         np.load(base['source_mesh_path'], allow_pickle=False) as source:
        reservoir = archive['points']
        count = len(reservoir)
        original_indices = [rng.choice(reservoir.shape[1], n, replace=False) for _ in range(count)]
        old_r = np.stack([random_rotation(rng) for _ in range(count)])
        old_t = rng.uniform(-data.get('translation_range', .7), data.get('translation_range', .7), (count, 3))
        noise = rng.normal(scale=data.get('hard_noise_std', .001), size=(count, n, 3)) if record['band'] == 'hard' else np.zeros((count, n, 3))
        q = data['sdf_queries']
        near = np.flatnonzero(source['sdf_near_mask']); far = np.flatnonzero(~source['sdf_near_mask'])
        qi = np.concatenate([rng.choice(near, q//2, replace=len(near)<q//2),
                             rng.choice(far, q-q//2, replace=len(far)<q-q//2)])
        rng.shuffle(qi)
        view_r = np.stack([random_rotation(rng) for _ in range(3)])
        indices = ([rng_new.choice(reservoir.shape[1], n, replace=False) for _ in range(count)]
                   if mode in ('samples', 'samples_poses') else original_indices)
        rotation = np.stack([random_rotation(rng_new) for _ in range(count)]) if mode in ('rotation', 'samples_poses') else old_r
        shift = rng_new.uniform(-data.get('translation_range', .7), data.get('translation_range', .7), (count, 3)) if mode in ('translation', 'samples_poses') else old_t
        if mode in ('rotation', 'samples_poses'):
            # Rotate existing observation noise with the fragment: a rotation-only
            # experiment must not introduce a second noise realization.
            noise = np.einsum('fni,fij,fkj->fnk', noise, old_r, rotation)
        src = np.stack([reservoir[i, ids] for i, ids in enumerate(indices)]).astype(np.float64)
        original = np.einsum('fni,fji->fnj', src, rotation) + shift[:, None] + noise
        centers = original.mean(1)
        centered = original - centers[:, None]
        scale = np.linalg.norm(centered, axis=-1).max(1).sum()
        anchor = int(np.argmax(np.sqrt(np.mean(np.sum(centered**2, axis=-1), axis=-1))))
        center = (centers[anchor] - shift[anchor]) @ rotation[anchor]
        anchor_r = rotation[anchor]
        canonical = np.einsum('fni,ji->fnj', src-center, anchor_r) / scale
        gt_r = anchor_r[None] @ rotation.transpose(0, 2, 1)
        gt_t = (np.einsum('fi,fij->fj', centers-shift, rotation)-center) @ anchor_r.T / scale
        def padded(values, shape, fill=0):
            out = np.full(shape, fill, dtype=values.dtype)
            out[:count] = values
            return torch.from_numpy(out)
        result['points'] = padded((centered/scale).astype('float32'), (3, n, 3))
        result['original_points'] = padded(original.astype('float32'), (3, n, 3))
        result['centroids'] = padded(centers.astype('float32'), (3, 3))
        result['canonical_points'] = padded(canonical.astype('float32'), (3, n, 3))
        result['fracture_labels'] = padded(np.stack([archive['fracture_labels'][i, ids] for i, ids in enumerate(indices)]).astype('float32'), (3, n))
        result['interface_ids'] = padded(np.stack([archive['interface_ids'][i, ids] for i, ids in enumerate(indices)]).astype('int64'), (3, n), -1)
        result['rotations_gt'] = torch.eye(3).repeat(3, 1, 1)
        result['rotations_gt'][:count] = torch.from_numpy(gt_r.astype('float32'))
        result['translations_gt'] = padded(gt_t.astype('float32'), (3, 3))
        result['rotations_gt'][anchor] = torch.eye(3); result['translations_gt'][anchor] = 0
        result['anchor_index'] = torch.tensor(anchor)
        result['shared_scale'] = torch.tensor(scale, dtype=torch.float32)
        result['anchor_source_rotation'] = torch.from_numpy(anchor_r.astype('float32'))
        result['anchor_source_centroid'] = torch.from_numpy(center.astype('float32'))
        result['sdf_queries'] = torch.from_numpy(((source['sdf_queries'][qi]-center) @ anchor_r.T/scale).astype('float32'))
        result['sdf_values'] = torch.from_numpy((source['sdf_values'][qi]/scale).astype('float32'))
        result['sdf_near_mask'] = torch.from_numpy(source['sdf_near_mask'][qi].astype(bool))
        result['target_points'] = torch.from_numpy(((source['target_points']-center) @ anchor_r.T/scale).astype('float32'))
        result['points_view2'] = torch.from_numpy(np.einsum('fni,fji->fnj', result['points'].numpy(), view_r).astype('float32'))
        result['view2_rotations'] = torch.from_numpy(view_r.astype('float32'))
    return result


def check_adapter(dataset, index):
    base, rebuilt = dataset[index], variant(dataset, index, 'identity_rebuild')
    errors = {}
    for key, value in base.items():
        if isinstance(value, torch.Tensor):
            other = rebuilt[key]
            if value.dtype in (torch.bool, torch.int64, torch.int32):
                if not torch.equal(value, other):
                    raise AssertionError(f'Adapter identity mismatch: {key}')
            else:
                error = float((value-other).abs().max())
                errors[key] = error
                torch.testing.assert_close(value, other, atol=1e-6, rtol=1e-6)
    return errors
