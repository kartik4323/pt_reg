"""Bounded, supervision-only contact ceiling audit; never a learned-model result.

Run as ``python -m reassembly.contact_audit --help`` from the repo.
This uses independently sampled prepared points. Ground-truth interface IDs
and coordinates generate ideal probabilities solely to diagnose the matching
representation/solver; this is not an inference path or an acceptance gate.
"""
import argparse
import json
from pathlib import Path

import torch

from reassembly.config import load_config
from reassembly.data import FractureDataset
from reassembly.evaluation import aggregate, assembly_metrics
from reassembly.losses import contact_geometry, local_contact_distribution
from reassembly.model import contact_point_indices, farthest_point_indices
from reassembly.resources import make_guard, write_json
from reassembly.solver import solve_from_matches
from reassembly.training import collate_samples


def audit(manifest, cfg):
    dataset = FractureDataset(manifest, 'train', cfg, fixed=True, limit=16)
    results = {}
    for mode in ('legacy64_uniform', 'coverage256_uniform', 'contact256_localized'):
        rows = []
        for sample in dataset:
            count = int(sample['fragment_mask'].sum())
            points = sample['points'][:count]
            if mode == 'contact256_localized':
                # Explicit ORACLE segmentation, not a prediction.
                indices = contact_point_indices(points, 2 * sample['fracture_labels'][:count] - 1, 256)
            elif mode == 'legacy64_uniform':
                xyz = points
                indices = torch.arange(points.shape[1]).expand(count, -1)
                for number in cfg['model']['sample_counts']:
                    subset = farthest_point_indices(xyz, number)
                    indices = indices.gather(1, subset)
                    xyz = points.gather(1, indices[..., None].expand(-1, -1, 3))
            else:
                indices = farthest_point_indices(points, 256)
            batch = collate_samples([sample], 'cpu')
            matches = []
            for i in range(count):
                for j in range(i + 1, count):
                    pair = dict(i=i, j=j, source_indices=indices[i:i+1], target_indices=indices[j:j+1],
                                valid=torch.ones(1, dtype=torch.bool))
                    positive, distance = contact_geometry(pair, batch, cfg['train']['contact_radius'])
                    if mode == 'contact256_localized':
                        source = local_contact_distribution(positive, distance, cfg['train']['contact_sigma'])
                        target = local_contact_distribution(positive.transpose(-1, -2), distance.transpose(-1, -2),
                                                            cfg['train']['contact_sigma'])
                    else:
                        source = positive.float() / positive.sum(-1, keepdim=True).clamp_min(1)
                        transposed = positive.transpose(-1, -2)
                        target = transposed.float() / transposed.sum(-1, keepdim=True).clamp_min(1)
                    matches.append(dict(i=i, j=j, source_xyz=points[i, indices[i]].numpy(),
                                        target_xyz=points[j, indices[j]].numpy(),
                                        source_matchability=positive.any(-1)[0].float().numpy(),
                                        target_matchability=positive.any(-2)[0].float().numpy(),
                                        weights=(source * target.transpose(-1, -2))[0].numpy()))
            result = solve_from_matches(matches, count, int(sample['anchor_index']), cfg=cfg)
            row = assembly_metrics(sample, result, cfg['solver']['success_threshold'])
            row.update(pattern_id=sample['pattern_id'], diagnostics=result['diagnostics'])
            rows.append(row)
        results[mode] = {'summary': aggregate(rows), 'samples': rows}
    return {'kind': 'oracle_contact_ceiling_audit', 'dataset_fingerprint': dataset.fingerprint,
            'note': 'Ground-truth diagnostic, no learned weights, no scaffold, no pilot acceptance claim.',
            'conditions': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--managed-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg['resources']['managed_root'] = args.managed_root
    guard = make_guard(cfg, Path(args.output).resolve().parent)
    guard.check()
    torch.set_num_threads(4)
    report = audit(args.manifest, cfg)
    write_json(args.output, report, guard)
    print(json.dumps({name: value['summary'] for name, value in report['conditions'].items()}, indent=2))
