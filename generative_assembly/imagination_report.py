"""Portable E0/E1/E2 report. References are accessed only in this evaluator."""
from __future__ import annotations
import argparse
import base64
import csv
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing
from scipy.spatial import cKDTree
from . import data, render
from .storage import read, write


def discover(root):
    root = Path(root)
    return [root] if (root / 'study.json').exists() else sorted(p.parent for p in root.glob('*/study.json'))


def collect(root, dataset=None):
    rows = []
    for run in discover(root):
        cfg = read(run / 'study.json')['identity']['config']
        # Explicit dataset override makes downloaded artifacts portable.
        ds = Path(dataset) if dataset else Path(read(run / 'experiment.json')['dataset']) if (run / 'experiment.json').exists() else None
        cases = {c['id']: c for c in read(ds)['cases']} if ds and ds.exists() else {}
        jobs = [read(p) for p in sorted((run / 'jobs').glob('*/result.json'))]
        prepared = {j['case_id']: j for j in jobs if j['stage'] == 'E0' and j['status'] == 'complete'}
        cache = {}
        for job in jobs:
            if job['stage'] not in ('E1', 'E1_ORACLE', 'E2', 'E4'):
                continue
            out = job.get('output', {})
            directory = run / 'jobs' / job['job_id']
            row = dict(profile=cfg.get('experiment_profile', run.name), case_id=job['case_id'],
                stage=job['stage'], arm=job['arm'], status=job['status'], job_id=job['job_id'],
                model=out.get('model'), input_type=out.get('input_type'), seed=out.get('seed'),
                oracle=job.get('oracle', False), smoke=job.get('smoke', False), seconds=job.get('seconds'),
                error=job.get('error'), artifact_dir=str(directory), normalized_distance_units=True,
                reference_available=False)
            if job['status'] != 'complete':
                rows.append(row); continue
            if out.get('not_applicable'):
                row.update(status='not_applicable', reason=out['not_applicable'])
                rows.append(row); continue
            if job['case_id'] in cases:
                if job['case_id'] not in cache:
                    case = data.load_case(ds, cases[job['case_id']], cfg)
                    cache[job['case_id']] = (case, data.reference(ds, case))
                case, ref = cache[job['case_id']]
            else:
                case, ref = None, None
            row['reference_available'] = ref is not None and 'complete' in ref
            if job['stage'] in ('E1', 'E1_ORACLE'):
                row.update(out.get('selection', {}))
                if out.get('raw_selection'):
                    row.update({'raw_' + k: v for k, v in out['raw_selection'].items()})
                    row['cleanup_removed_foreground_fraction'] = out['raw_selection']['foreground_fraction'] - out['selection']['foreground_fraction']
                if row['reference_available']:
                    parent = prepared.get(job['case_id'])
                    if parent:
                        kind = out.get('input_type', 'F')
                        input_dir = run / 'jobs' / parent['job_id'] / kind
                        camera_file = input_dir / 'camera.json'
                        row['camera_provenance_valid'] = camera_file.exists() or cfg.get('n_render_views', 1) == 1
                        camera = read(camera_file if camera_file.exists() else run / 'jobs' / parent['job_id'] / 'camera.json')
                        if row['camera_provenance_valid']:
                            truth = render.render(ref['complete'], None, None, camera, cfg['splat_radius'])
                            if camera.get('crop_box') is not None:
                                truth = render.crop_render(truth, camera['crop_box'])
                            truth = binary_closing(truth['valid'], iterations=2)
                            predicted = render.foreground(Image.open(directory / 'image.png').convert('RGB'))
                            predicted = binary_closing(predicted, iterations=2)
                            row['silhouette_iou'] = float((truth & predicted).sum() / max(1, (truth | predicted).sum()))
                            row['silhouette_metric'] = 'point_splat_proxy_2px_closing'
                            if (input_dir / 'render.npz').exists():
                                with np.load(input_dir / 'render.npz', allow_pickle=False) as f:
                                    observed = binary_closing(f['valid'], iterations=2)
                                missing, added = truth & ~observed, predicted & ~observed
                                row['missing_region_precision'] = float((missing & added).sum() / max(1, added.sum()))
                                row['missing_region_recall'] = float((missing & added).sum() / max(1, missing.sum()))
            elif job['stage'] == 'E2':
                fit = out.get('alignment', {})
                shape = directory / 'shape.npz'
                row.update(template_rejected=bool(out.get('template_rejected') or fit.get('rejected')),
                    rejection_reason=fit.get('rejected'), heldout_error=fit.get('heldout_error'), fit_error=fit.get('fit_error'))
                if shape.exists():
                    with np.load(shape, allow_pickle=False) as f:
                        valid_alignment = not row['template_rejected'] and 'aligned' in f.files and fit.get('heldout_error') is not None
                        row['valid_alignment'] = valid_alignment
                        if valid_alignment and row['reference_available']:
                            points, truth = f['aligned'], ref['complete']
                            a, b = cKDTree(truth).query(points)[0], cKDTree(points).query(truth)[0]
                            row.update(chamfer=float((a.mean() + b.mean()) / 2),
                                precision_002=float((a < .02).mean()), recall_002=float((b < .02).mean()))
                            pr, re = row['precision_002'], row['recall_002']
                            row['f1_002'] = 2 * pr * re / max(1e-12, pr + re)
                # Never score a rejected/raw template in the normalized reference frame.
            elif job['stage'] == 'E4' and ref is not None:
                from .evaluate import pose_metrics
                with np.load(directory / 'poses.npz', allow_pickle=False) as f:
                    row.update(pose_metrics(case, ref, f['normalized'], cfg))
            rows.append(row)
    return rows


def gallery(rows, case_id, stage='E1', max_images=48):
    """Embed actual bytes so executed notebooks work away from the inference host."""
    tiles = []
    from html import escape
    for row in rows:
        if row['case_id'] != case_id or row['stage'] not in (stage, 'E1_ORACLE') or row['status'] != 'complete':
            continue
        directory = Path(row['artifact_dir'])
        for name in ('image_raw.png', 'image.png'):
            path = directory / name
            if not path.exists(): continue
            image = Image.open(path).convert('RGB'); image.thumbnail((220, 220))
            buffer = io.BytesIO(); image.save(buffer, format='PNG')
            encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
            label = escape(f"{row['profile']} | {row['arm']} | {name}")
            tiles.append(f'<div style="display:inline-block;margin:8px;width:230px"><p>{label}</p><img src="data:image/png;base64,{encoded}"/></div>')
        if len(tiles) >= max_images: break
    return ''.join(tiles)


def input_gallery(root, case_id):
    """Show the actual RGB/depth/edit mask for each controlled profile."""
    from html import escape
    tiles = []
    for run in discover(root):
        for path in sorted((run / 'jobs').glob('*/result.json')):
            job = read(path)
            if job['stage'] != 'E0' or job['case_id'] != case_id or job['status'] != 'complete':
                continue
            for directory in sorted(p for p in path.parent.iterdir() if p.is_dir()):
                for name in ('image.png', 'control.png', 'mask.png'):
                    image_path = directory / name
                    if not image_path.exists(): continue
                    image = Image.open(image_path).convert('RGB'); image.thumbnail((180, 180))
                    buffer = io.BytesIO(); image.save(buffer, format='PNG')
                    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
                    title = escape(f'{run.name} | {directory.name} | {name}')
                    tiles.append(f'<div style="display:inline-block;margin:8px;width:190px"><p>{title}</p><img src="data:image/png;base64,{encoded}"/></div>')
    return ''.join(tiles)


def export(root, dataset=None):
    rows = collect(root, dataset)
    output = Path(root) / 'analysis'
    output.mkdir(exist_ok=True, parents=True)
    write(output / 'metrics.json', rows)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--dataset')
    args = p.parse_args()
    rows = export(args.root, args.dataset)
    print(f'Exported {len(rows)} rows to {Path(args.root) / "analysis"}')
    print(f"Failed jobs: {sum(r['status'] == 'failed' for r in rows)}; "
          f"rejected templates: {sum(bool(r.get('template_rejected')) for r in rows)}")
