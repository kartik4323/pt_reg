"""Failure-inclusive, source-paired evidence and architecture reports."""
import collections
import base64
import io
import csv
import html
import json
from pathlib import Path
import numpy as np
from generative_assembly import data, evaluate as evaluator
from generative_assembly.storage import read, write


def paired(rows, stage, arm, baseline_stage='E4', baseline_arm='B0__refine', factor=None):
    selected = [r for r in rows if r['stage'] == stage and r['arm'] == arm and not r.get('oracle')]
    controls = {r['case_id']: r for r in rows if r['stage'] == baseline_stage and r['arm'] == baseline_arm and not r.get('oracle')}
    if factor:
        selected = [r for r in selected if r.get('factors', {}).get('factor') == factor[0] and
                    r.get('factors', {}).get('value') == factor[1]]
    a, b = collections.defaultdict(list), collections.defaultdict(list)
    damaged, eligible = 0, 0
    for row in selected:
        base = controls.get(row['case_id'])
        m, n = row.get('metrics') or {}, (base or {}).get('metrics') or {}
        if 'success' not in m or 'success' not in n:
            continue
        a[row['source_id']].append(float(m['success']))
        b[row['source_id']].append(float(n['success']))
        if n['success']:
            eligible += 1; damaged += int(not m['success'])
    delta = evaluator.bootstrap_delta(a, b, count=2000)
    return {'stage': stage, 'arm': arm, 'baseline_stage': baseline_stage, 'baseline': baseline_arm,
            'paired_cases': sum(map(len, a.values())), **delta,
            'damage_rate': damaged/eligible if eligible else None, 'damage_denominator': eligible}


def gates(rows, cfg):
    arm = f'B2__{cfg["primary_model"]}__{cfg["primary_input"]}__refine'
    primary = paired(rows, 'E4', arm)
    K = len(cfg['image_seeds'])
    gated = paired(rows, 'E5', f'gated__K{K}')
    always = paired(rows, 'E5', f'always__K{K}')
    decisions = [r for r in rows if r['stage'] == 'E5' and r['arm'] == f'gated__K{K}' and not r.get('oracle')]
    by_source = collections.defaultdict(list)
    complete_by_source=collections.defaultdict(list)
    for row in decisions:
        diag=row.get('input_only_diagnostics') or {}
        by_source[row['source_id']].append(bool(diag.get('template_accepted')))
        complete_by_source[row['source_id']].append(bool(diag.get('template_accepted') and diag.get('budget_complete')))
    coverage = [float(np.mean(values)) for values in by_source.values()]
    complete_coverage=[float(np.mean(values)) for values in complete_by_source.values()]
    def useful(comparison):
        return comparison['sources'] >= 2 and comparison['difference'] is not None and comparison['difference'] >= cfg['evaluation']['success_gain'] and \
            comparison['ci95'][0] > 0 and comparison['damage_rate'] is not None and comparison['damage_rate'] <= cfg['evaluation']['damage_max']
    e4_pass = useful(primary)
    damage_reduced = always['damage_rate'] is not None and gated['damage_rate'] is not None and \
        always['damage_rate'] > 0 and gated['damage_rate'] <= .5*always['damage_rate']
    e5_pass = bool(useful(gated) and damage_reduced and complete_coverage and np.mean(complete_coverage) > 0)
    return {'smoke': cfg['smoke'], 'E4_pass': bool(e4_pass) and not cfg['smoke'],
            'E5_pass': e5_pass and not cfg['smoke'], 'E4': primary, 'E5_gated': gated, 'E5_always': always,
            'coverage': float(np.mean(coverage)) if coverage else None,
            'accepted_complete_budget_coverage':float(np.mean(complete_coverage)) if complete_coverage else None,
            'reason': 'SMOKE_NOT_RESEARCH_EVIDENCE' if cfg['smoke'] else
                'pass' if e4_pass and e5_pass else 'not_passed_or_inconclusive'}


def evaluate(store, split):
    # First use the existing evaluator for geometric metrics and image review.
    evaluator.evaluate(store, split)
    out = store.root/'evaluation'/split
    rows = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines() if line]
    all_records={c['id']: c for c in read(store.dataset)['cases']}
    records = {cid:c for cid,c in all_records.items() if c['split'] == split}
    references = read(Path(store.dataset).parent/'evaluator_only'/'index.json') if (Path(store.dataset).parent/'evaluator_only'/'index.json').exists() else {}
    jobs = {r['job_id']: r for r in store.jobs(split=split)}
    latest = {}
    for row in rows:
        row['started'] = jobs[row['job_id']]['started']
        key=(row['stage'],row['case_id'],row['arm'])
        if key not in latest or row['started'] > latest[key]['started']:
            latest[key]=row
    superseded=[row for row in rows if latest[(row['stage'],row['case_id'],row['arm'])]['job_id'] != row['job_id']]
    write(out/'superseded_outcomes.json',superseded)
    rows=list(latest.values())
    for row in rows:
        base = row['base_case_id']
        job = jobs[row['job_id']]
        # Never convert absent real-scan reference accuracy to a zero or success.
        if base not in references:
            row['metrics'] = None
            if row['status'] not in ('failed', 'not_applicable'):
                row['status'] = 'unscored_no_reference'
            continue
        if row['stage'] in ('E3', 'E4', 'E5', 'E6', 'E7') and row['status'] in ('failed', 'not_applicable'):
            # Unavailable diagnostic categories are marked separately and do not
            # become fabricated deployments. A requested deployable arm fails.
            if not row['oracle'] and '__unavailable' not in row['arm']:
                row['metrics'] = {'success': False, 'failure': job.get('error', job.get('output', {}).get('not_applicable')),
                                  'threshold_success':{t:False for t in ('0.005','0.01','0.02')}}
    (out/'metrics.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows), encoding='utf-8')
    # Rebuild summaries after fixing denominators; retain original image review.
    summary = summarize(rows, store.config)
    quality_path=store.root/'evaluation'/'train'/'pseudo_label_quality.json'
    if quality_path.exists():
        quality=read(quality_path)['labels']
        lookup={r['case_id']:r for r in quality}
        accepted=[r['case_id'] for r in quality if r['accepted']]
        def label_error(ids):
            by_source=collections.defaultdict(list)
            for cid in ids:
                item=lookup.get(cid,{})
                value=(item.get('metrics') or {}).get('max_part_chamfer')
                if value is not None:
                    by_source[all_records[cid]['source_id'] if cid in all_records else cid].append(value)
            return float(np.mean([np.mean(v) for v in by_source.values()])) if by_source else None
        quality_comparisons=[]
        for trained in store.jobs('E6_TRAIN'):
            if trained['status']=='complete' and trained['arm'].startswith('random_count__'):
                pool=read(store.artifact(trained,'training.json'))['pool']
                filtered_error,random_error=label_error(accepted),label_error(pool)
                quality_comparisons.append({'arm':trained['arm'],'filtered_count':len(accepted),'random_count':len(pool),
                    'count_matched':len(accepted)==len(pool),'filtered_error':filtered_error,'random_error':random_error,
                    'filtered_error_lower':filtered_error is not None and random_error is not None and filtered_error<random_error})
        summary['pseudo_label_quality_vs_random_count']=quality_comparisons
        quality_pass=bool(quality_comparisons) and all(r['count_matched'] and r['filtered_error_lower'] for r in quality_comparisons)
        summary['E6_gate']='SMOKE_NOT_RESEARCH_EVIDENCE' if store.config['smoke'] else (
            'pass' if summary['E6_geometry_gain_gate']=='pass' and quality_pass else 'not_passed_or_inconclusive')
    summary.update(split=split, reference_use='evaluation_only', E4_primary_gate=summary['gates']['reason'],
                   limitations=['contact/collision are proxies', 'erosion is point-removal proxy', 'raw rotations are not symmetry-reduced'])
    write(out/'summary.json', summary)
    table(out/'per_source.csv', source_rows(rows))
    failures = [{'stage': r['stage'], 'case_id': r['case_id'], 'arm': r['arm'], 'status': r['status'],
                 'artifact_dir': str(store.root/'jobs'/r['job_id'])} for r in rows
                if r['status'] != 'complete' or (r.get('metrics') or {}).get('success') is False]
    table(out/'failures.csv', failures)
    tiles = []
    from PIL import Image
    for failed in failures:
        directory=Path(failed['artifact_dir'])
        record=jobs[directory.name]
        image=directory/'image.png'
        template=record.get('output',{}).get('template_job')
        if template in jobs:
            upstream=jobs[template].get('output',{}).get('image_job')
            if upstream:
                image=store.root/'jobs'/upstream/'image.png'
        tile='<p>'+html.escape(str(failed))+'</p>'
        if image.is_file():
            with Image.open(image) as picture:
                picture=picture.convert('RGB'); picture.thumbnail((240,240))
                buffer=io.BytesIO(); picture.save(buffer,format='PNG')
                tile+='<img src="data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()+'">'
        tiles.append(tile)
    (out/'failure_gallery.html').write_text('<h1>Failure and unscored outputs</h1>'+''.join(tiles), encoding='utf-8')
    report(out/'REPORT.md', summary)
    return summary


def source_rows(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        factor = row.get('factors', {})
        key = (row['stage'], row['arm'], row['source_id'], factor.get('factor'), str(factor.get('value')), row['oracle'])
        groups[key].append(row)
    result = []
    for (stage, arm, source, factor, value, oracle), group in sorted(groups.items(), key=str):
        successes = [float(r['metrics']['success']) for r in group if stage in ('E3','E4','E5','E6','E7') and
                     r.get('metrics') and 'success' in r['metrics']]
        item={'stage': stage, 'arm': arm, 'source_id': source, 'factor': factor, 'value': value,
                       'oracle': oracle, 'requested': len(group), 'scored': len(successes),
                       'success': float(np.mean(successes)) if successes else None,
                       'failed': sum(r['status'] == 'failed' for r in group)}
        for name in ('max_part_chamfer','rotation_degrees','translation_error','shape_chamfer','silhouette_iou'):
            values=[float(np.mean(r['metrics'][name])) for r in group if r.get('metrics') and
                    r['metrics'].get(name) is not None and np.size(r['metrics'][name])]
            item['mean_'+name]=float(np.mean(values)) if values else None
        for threshold in ('0.005','0.01','0.02'):
            values=[float(r['metrics']['threshold_success'][threshold]) for r in group if r.get('metrics') and
                    threshold in r['metrics'].get('threshold_success',{})]
            item['success_'+threshold]=float(np.mean(values)) if values else None
        result.append(item)
    return result


def summarize(rows, cfg):
    groups = collections.defaultdict(list)
    for row in source_rows(rows):
        groups[(row['stage'], row['arm'], row['factor'], row['value'], row['oracle'])].append(row)
    summaries = []
    for (stage, arm, factor, value, oracle), group in sorted(groups.items(), key=str):
        success = [r['success'] for r in group if r['success'] is not None]
        summaries.append({'stage': stage, 'arm': arm, 'factor': factor, 'value': value, 'oracle': oracle,
                          'sources': len(group), 'requested': sum(r['requested'] for r in group),
                          'failed': sum(r['failed'] for r in group), 'success_rate': float(np.mean(success)) if success else None})
    arms = sorted({(r['stage'], r['arm']) for r in rows if not r['oracle'] and r['stage'] in ('E4', 'E5', 'E6')})
    comparisons = [paired(rows, stage, arm) for stage, arm in arms]
    primary = f'B2__{cfg["primary_model"]}__{cfg["primary_input"]}__refine'
    # Oracle controls stay labelled; they are evaluator diagnostics only.
    targeted = []
    for control in [f'B1__raw__{cfg["primary_input"]}__refine', 'B3__refine', 'B4__refine', 'B5__refine', 'B6__refine']:
        if control in ('B3__refine', 'B4__refine', 'B6__refine'):
            copy_rows = [dict(r, oracle=False) if r['arm'] == control else r for r in rows]
            item = paired(copy_rows, 'E4', primary, 'E4', control)
            item['oracle_control'] = True
        else:
            item = paired(rows, 'E4', primary, 'E4', control)
        targeted.append(item)
    for K in (1, 2, 4):
        targeted.append(paired(rows, 'E5', f'gated__K{K}', 'E5', f'always__K{K}'))
    for seed in cfg['training']['seeds']:
        for control in ('geometry', 'all', 'random_count'):
            targeted.append(paired(rows, 'E6', f'filtered__seed{seed}', 'E6', f'{control}__seed{seed}'))
        targeted.append(paired(rows, 'E6', f'filtered__seed{seed}', 'E5', f'gated__K{len(cfg["image_seeds"])}'))
    student_checks=[paired(rows,'E6',f'filtered__seed{seed}','E6',f'geometry__seed{seed}') for seed in cfg['training']['seeds']]
    student_pass=bool(student_checks) and all(c['difference'] is not None and c['sources']>=2 and
        c['difference']>=cfg['evaluation']['success_gain'] and c['ci95'][0]>0 for c in student_checks)
    robust = []
    for factor, value in sorted({(r.get('factors', {}).get('factor'), r.get('factors', {}).get('value')) for r in rows if r['stage'] == 'E7'}, key=str):
        for arm in sorted({r['arm'] for r in rows if r['stage'] == 'E7' and r['arm'] != 'B0' and '__unavailable' not in r['arm']}):
            robust.append(dict(paired(rows, 'E7', arm, 'E7', 'B0', (factor, value)), factor=factor, value=value))
    return {'smoke': cfg['smoke'], 'groups': summaries, 'paired_comparisons': comparisons,
            'targeted_comparisons': targeted, 'robustness_comparisons': robust, 'gates': gates(rows, cfg),
            'E6_geometry_gain_gate': 'SMOKE_NOT_RESEARCH_EVIDENCE' if cfg['smoke'] else
                'pass' if student_pass else 'not_measured_or_inconclusive'}


def table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def report(path, summary):
    text = ['# Architecture study evidence', '', 'SMOKE: software validation only.' if summary['smoke'] else 'Exploratory development evidence.',
            '', '| Stage | Arm | Sources | Requested | Failed | Success |', '|---|---|---:|---:|---:|---:|']
    for row in summary['groups']:
        rate = 'unscored' if row['success_rate'] is None else f'{row["success_rate"]:.3f}'
        text.append(f'| {row["stage"]} | {row["arm"]} | {row["sources"]} | {row["requested"]} | {row["failed"]} | {rate} |')
    text += ['', 'E4/E5 gate: '+summary['gates']['reason'], '',
             'B4 → B6 → B2 localizes solver, reconstruction and completion losses. Compare B2 with raw/wrong/compute controls before attributing gains.',
             'Compare rerank/refine damage and K/coverage before increasing guidance. Reject-all gates do not qualify.',
             'Students must beat geometry-only learners and the frozen gated teacher; loss curves are not accuracy evidence.',
             'See per_source.csv, metrics.jsonl and failure_gallery.html; unavailable categories and unscored scans remain explicit.']
    path.write_text('\n'.join(text)+'\n', encoding='utf-8')


def aggregate(root, plan):
    # Each condition is aggregated separately; copied cached jobs and shards
    # never become independent sources or get pooled across architectures.
    output = Path(root)/'analysis'
    conditions = collections.defaultdict(list)
    for task in plan['tasks']:
        for path in (Path(root)/'runs'/task['id']/'evaluation').glob('*/metrics.jsonl'):
            for line in path.read_text().splitlines():
                if line:
                    row = json.loads(line)
                    row['split'] = path.parent.name
                    conditions[(task['condition'], path.parent.name)].append(row)
    all_rows, summaries = [], {}
    for (condition, split), records in conditions.items():
        unique = {}
        for row in records:
            key = (row['job_id'], row['case_id'], row['stage'], row['arm'])
            if key in unique and unique[key] != row:
                raise ValueError('Conflicting results in shard aggregation')
            unique[key] = row
        rows = list(unique.values())
        cfg = next(t['config'] for t in plan['tasks'] if t['condition'] == condition)
        cfg = dict(cfg, training=dict(cfg['training'], seeds=plan['base_config']['training']['seeds']))
        summary = summarize(rows, cfg)
        quality_checks={}
        for task in plan['tasks']:
            if task['condition']!=condition:
                continue
            path=Path(root)/'runs'/task['id']/'evaluation'/split/'summary.json'
            if path.exists():
                for check in read(path).get('pseudo_label_quality_vs_random_count',[]):
                    quality_checks[check['arm']]=check
        if quality_checks:
            summary['pseudo_label_quality_vs_random_count']=list(quality_checks.values())
            quality_pass=all(r['count_matched'] and r['filtered_error_lower'] for r in quality_checks.values())
            summary['E6_gate']='SMOKE_NOT_RESEARCH_EVIDENCE' if cfg['smoke'] else (
                'pass' if summary['E6_geometry_gain_gate']=='pass' and quality_pass else 'not_passed_or_inconclusive')
        name = condition+'__'+split
        summaries[name] = summary
        destination = output/name
        write(destination/'summary.json', summary)
        table(destination/'per_source.csv', source_rows(rows))
        (destination/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        report(destination/'REPORT.md', summary)
        all_rows.extend(dict(row, condition=condition) for row in rows)
    write(output/'summary.json', summaries)
    table(output/'per_case.csv', [dict(condition=r['condition'], **{k: v for k, v in r.items() if k not in ('condition','metrics')},
                                     metrics=json.dumps(r.get('metrics'))) for r in all_rows])
    base = conditions.get(('baseline', 'dev'), [])
    base_unique = {r['job_id']: r for r in base}
    gate = gates(list(base_unique.values()), plan['base_config'])
    write(output/'development_gate.json', gate)
    return gate


def development_gate(root,plan):
    rows={}
    for task in plan['tasks']:
        if task['condition']!='baseline':
            continue
        path=Path(root)/'runs'/task['id']/'evaluation'/'dev'/'metrics.jsonl'
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line:
                row=json.loads(line)
                key=(row['stage'],row['case_id'],row['arm'])
                if key not in rows or row.get('started',0)>rows[key].get('started',0):
                    rows[key]=row
    decision=gates(list(rows.values()),plan['base_config'])
    write(Path(root)/'analysis'/'development_gate.json',decision)
    return decision
