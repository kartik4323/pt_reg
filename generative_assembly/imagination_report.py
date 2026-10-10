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
    cached=Path(root)/'analysis'/'metrics.json'
    runs=discover(root)
    if dataset is None and cached.exists() and runs and all(
        not (r/'experiment.json').exists() or not Path(read(r/'experiment.json')['dataset']).exists() for r in runs):
        rows=read(cached)
        profiles={read(r/'experiment.json').get('profile',r.name) if (r/'experiment.json').exists() else
                  read(r/'study.json')['identity']['config'].get('experiment_profile',r.name):r for r in runs}
        for row in rows:
            if row['profile'] in profiles: row['artifact_dir']=str(profiles[row['profile']]/'jobs'/row['job_id'])
            row['cached_evaluator_metrics']=True
        return rows
    rows = []
    for run in runs:
        cfg = read(run / 'study.json')['identity']['config']
        manifest=read(run/'experiment.json') if (run/'experiment.json').exists() else {}
        # Explicit dataset override makes downloaded artifacts portable.
        ds = Path(dataset) if dataset else Path(read(run / 'experiment.json')['dataset']) if (run / 'experiment.json').exists() else None
        cases = {c['id']: c for c in read(ds)['cases']} if ds and ds.exists() else {}
        jobs = [read(p) for p in sorted((run / 'jobs').glob('*/result.json'))]
        prepared = {j['case_id']: j for j in jobs if j['stage'] == 'E0' and j['status'] == 'complete'}
        cache = {}
        for job in jobs:
            if job['stage'] not in ('E0','E1', 'E1_ORACLE', 'E2', 'E3','E4'):
                continue
            out = job.get('output', {})
            directory = run / 'jobs' / job['job_id']
            row = dict(profile=manifest.get('profile',cfg.get('experiment_profile', run.name)), case_id=job['case_id'],
                source_id=job.get('source_id'), split=job.get('split'),
                stage=job['stage'], arm=job['arm'], status=job['status'], job_id=job['job_id'],
                model=out.get('model'), input_type=out.get('input_type'), seed=out.get('seed'),
                oracle=job.get('oracle', False), smoke=job.get('smoke', False), seconds=job.get('seconds'),
                error=job.get('error'), artifact_dir=str(directory), normalized_distance_units=True,
                reference_available=False)
            row['failure_kind']=job.get('failure_kind')
            if job['status'] != 'complete':
                rows.append(row); continue
            if out.get('not_applicable'):
                row.update(status='not_applicable', reason=out['not_applicable'])
                rows.append(row); continue
            if job['stage'] in ('E0','E3'):
                row.update(renderer_requested=out.get('render_mode'),diagnostics=out.get('diagnostics'))
                if job['stage']=='E0':
                    renderers=[read(p) for p in directory.glob('*/renderer.json')]
                    row['renderer_actual']=[v['actual'] for v in renderers]
                    row['render_coverage']=[v['coverage'] for v in renderers]
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
                    if out.get('selection',{}).get('foreground_fraction') is not None:
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
                            alpha=directory/'alpha.png'
                            predicted=np.asarray(Image.open(alpha).resize((cfg['pixels'],cfg['pixels']),Image.Resampling.NEAREST))>=128 if alpha.exists() else render.foreground(Image.open(directory / 'image.png').convert('RGB'))
                            row['foreground_metric']='object_matte' if alpha.exists() else 'legacy_brightness_proxy'
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
                row.update(shape_diagnostics=out.get('shape_diagnostics'),watertight=out.get('watertight'),
                           components=out.get('components'),reconstruction_input_mode=out.get('input_mode'))
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
                            from .quality import shape_metrics,evaluator_observed
                            from . import geometry as g
                            kind=out.get('input_type','F').split('_v')[0]
                            observed=evaluator_observed(case,ref,kind)
                            row.update(shape_metrics(points,truth,observed))
                # Never score a rejected/raw template in the normalized reference frame.
            elif job['stage'] == 'E4' and ref is not None:
                from .evaluate import pose_metrics
                with np.load(directory / 'poses.npz', allow_pickle=False) as f:
                    row.update(pose_metrics(case, ref, f['normalized'], cfg))
                row['abstained']=out.get('diagnostics',{}).get('abstained')
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
        for name in ('image_raw.png','alpha.png','image_white.png', 'image.png'):
            path = directory / name
            if not path.exists(): continue
            image = Image.open(path).convert('RGB'); image.thumbnail((220, 220))
            buffer = io.BytesIO(); image.save(buffer, format='PNG')
            encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
            label = escape(f"{row['profile']} | {row['arm']} | {name}")
            tiles.append(f'<div style="display:inline-block;margin:8px;width:230px"><p>{label}</p><img src="data:image/png;base64,{encoded}"/></div>')
        if len(tiles) >= max_images: break
    return ''.join(tiles)


def review_sheet(rows,case_id):
    """Blinded input-only review, with no GT image or evaluator outcome."""
    from html import escape
    tiles=gallery([r for r in rows if not r.get('oracle')],case_id,max_images=1000)
    return '<p>Review: bottle identity; full outline; same view/location; texture corruption. '+\
           'Foreground retention is NOT camera verification. Mark uncertain cases.</p>'+tiles


def study_diagnostics(root):
    funnel=[]; priors=[]; resources=[]
    for run in discover(root):
        for p in (run/'jobs').glob('*/result.json'):
            r=read(p); output=r.get('output',{})
            funnel.append(dict(profile=run.name,stage=r['stage'],case_id=r['case_id'],
                source_id=r['source_id'],status=r['status'],failure_kind=r.get('failure_kind'),
                valid_e1=output.get('selection',{}).get('valid_foreground'),
                template_rejected=output.get('template_rejected')))
        for p in (run/'priors').glob('*.json'):
            for arm,value in read(p).items(): priors.append(dict(profile=run.name,case_id=p.stem,arm=arm,**value))
        for p in (run/'jobs').glob('*/gpu_preflight.json'): resources.append(dict(profile=run.name,**read(p)))
    budget=read(Path(root)/'budget.json') if (Path(root)/'budget.json').exists() else None
    return dict(funnel=funnel,priors=priors,resources=resources,budget=budget)


def prepare_review(root):
    """Create blank, non-overwriting review forms keyed by selected E2 candidate."""
    for run in discover(root):
        jobs={read(p)['job_id']:read(p) for p in (run/'jobs').glob('*/result.json')}
        review=read(run/'review.json') if (run/'review.json').exists() else {}
        for path in (run/'priors').glob('*.json'):
            for arm,entry in read(path).items():
                if arm.startswith('raw__'): continue
                for jid in entry['job_ids']:
                    review.setdefault(jid,dict(bottle_identity=None,full_outline=None,same_view=None,
                        surface_quality=None,image_job=jobs[jid]['output'].get('image_job'),
                        case_id=path.stem,arm=arm,reviewer_note='Blinded image/geometry review; never use GT scores'))
        write(run/'review.json',review)
        from html import escape
        blocks=['<!doctype html><meta charset="utf-8"><title>Blinded prior review</title>',
            '<h1>Input-only review: no reference images or evaluator scores</h1>',
            '<p>Review bottle identity, complete outline, same camera/location and surface quality. '
            'Use review.json; null is pending. Review this page before inspecting evaluator results.</p>']
        for jid,entry in review.items():
            image_job=entry.get('image_job'); image=jobs.get(image_job)
            if not image or image.get('oracle'): continue
            blocks.append('<h2>'+escape(jid+' | '+entry['case_id']+' | '+entry['arm'])+'</h2>')
            blocks.append('<h3>Observed inputs (not reference geometry)</h3>'+input_gallery(run,entry['case_id']))
            blocks.append(gallery([dict(profile=run.name,case_id=entry['case_id'],stage='E1',status='complete',
                arm=image['arm'],artifact_dir=str(run/'jobs'/image_job))],entry['case_id']))
        (run/'blinded_review.html').write_text(''.join(blocks),encoding='utf-8')
    return 'Fill boolean review fields in each profile/review.json; null is pending, not approval.'


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
