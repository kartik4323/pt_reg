"""Append-only diagnostic outputs, bounded resource checks, and evidence reporting."""
import hashlib
import json
import os
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from reassembly.resources import ResourceGuard, jsonable, tree_bytes


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''): h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def job_id(job):
    return hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:20]


class Limits:
    def __init__(self,root,output,deadline):
        self.output=Path(output); self.deadline=deadline
        self.guard=ResourceGuard([root],40,50)
    def check(self):
        if time.time()>=self.deadline: raise TimeoutError('Diagnostic deadline reached')
        used=tree_bytes(self.output)
        archive=self.output/'diagnostic_bundle.tar.gz'
        self.guard.check(additional_bytes=max(0,2*1024**3-used))
        if used-(archive.stat().st_size if archive.exists() else 0)>900*1024**2:
            raise RuntimeError('Raw diagnostics reached 900 MiB; reserve remaining 2 GiB budget for bundle/finalization')


def compare(reference, actual, path='root'):
    """Compare every stored scalar/status, excluding timing/resource bookkeeping."""
    import numpy as np
    differences=[]
    if isinstance(reference,dict):
        for key,value in reference.items():
            if key in ('elapsed_seconds','seconds','checkpoint','config'): continue
            if key not in actual: differences.append(path+'.'+key+': missing')
            else: differences.extend(compare(value,actual[key],path+'.'+key))
    elif isinstance(reference,list):
        if not isinstance(actual,list) or len(reference)!=len(actual): differences.append(path+': length mismatch')
        else:
            for i,(a,b) in enumerate(zip(reference,actual)): differences.extend(compare(a,b,f'{path}[{i}]'))
    elif isinstance(reference,bool) or reference is None or isinstance(reference,str):
        if reference!=actual: differences.append(f'{path}: {reference!r} != {actual!r}')
    elif isinstance(reference,int):
        if reference!=actual: differences.append(f'{path}: count {reference!r} != {actual!r}')
    elif isinstance(reference,float):
        if actual is None or not np.isclose(reference,actual,atol=1e-5,rtol=1e-4,equal_nan=False):
            differences.append(f'{path}: {reference!r} != {actual!r}')
    return differences


def finalize(output, reason=None):
    output=Path(output)
    manifest=read(output/'experiments.json') if (output/'experiments.json').exists() else {'jobs':[]}
    jobs=manifest['jobs']; rows=[]; remaining=[]
    for job in jobs:
        file=output/'results'/f"{job['id']}.json"
        if file.exists(): rows.append(read(file))
        else: remaining.append(job)
    errors=[r for r in rows if r.get('error')]
    inventory=read(output/'inventory.json') if (output/'inventory.json').exists() else {}
    changes=[]; checkpoint_verification={}
    for name,entry in inventory.get('checkpoints',{}).items():
        file=Path(entry['path'])
        current=sha256(file) if file.exists() else None
        checkpoint_verification[name]={'before':entry['sha256'],'after':current,'unchanged':current==entry['sha256']}
        if current!=entry['sha256']: changes.append(name)
    production_changes=[]
    for name,entry in inventory.get('production_files',{}).items():
        file=Path(entry['path'])
        if not file.exists() or sha256(file)!=entry['sha256']: production_changes.append(name)
    failures=[r for r in rows if r.get('replay_differences')]
    input_changes=[]
    for name,entry in inventory.get('files',{}).items():
        if name.endswith('.pt'): continue
        file=Path(entry['path'])
        if not file.exists() or sha256(file)!=entry['sha256']: input_changes.append(name)
    complete=bool(jobs) and not remaining and not errors and not failures and not changes and not production_changes and not input_changes and reason is None
    summary={'kind':'diagnosis_only','status':'complete' if complete else 'partial_or_blocked',
             'reason':reason,'completed_jobs':len(rows),'planned_jobs':len(jobs),'unrun_jobs':remaining,
             'errors':[{'job':r['job'],'error':r['error']} for r in errors],
             'checkpoint_changes':changes,'production_file_changes':production_changes,
             'checkpoint_verification':checkpoint_verification,'input_changes':input_changes,
             'replay_mismatches':len(failures),'optimizer_updates':0,'production_acceptance':False}
    if not jobs:
        summary['unrun_sections']=['Artifact verification/replay','Full training/validation evaluation',
            'Stage representations/segmentation','Pose/sample controls','Oracle and threshold interventions','Scaffold and gradient probes']
    groups=defaultdict(list)
    for row in rows:
        if 'metrics' in row:
            j=row['job']
            name='/'.join(str(j.get(k,'')) for k in ('kind','checkpoint','cohort','condition','operation','variant','seed','weight','mass','shuffled'))
            groups[name].append(row)
    summary['observed']={name:{'examples':len(items),'source_objects':len({r.get('source_id') for r in items}),
        'geometric_successes':sum(r['metrics']['success'] for r in items),
        'statuses':dict(Counter(r['metrics']['status'] for r in items)),
        'by_source':{source:{'patterns':len({r['pattern_id'] for r in items if r.get('source_id')==source}),
            'successes':sum(r['metrics']['success'] for r in items if r.get('source_id')==source)} for source in sorted({r.get('source_id','unknown') for r in items})},
        'by_band':{band:{'count':sum(r.get('band')==band for r in items),'successes':sum(r['metrics']['success'] for r in items if r.get('band')==band)} for band in ('easy','intermediate','hard')}} for name,items in groups.items()}
    failure_counts=defaultdict(Counter)
    for row in rows:
        traces=row.get('filter_traces',{}) or row.get('solver_trace',{}).get('pairs',{}) or {k:v['support_trace'] for k,v in row.get('measurements',{}).get('pairs',{}).items()}
        key='/'.join(str(row['job'].get(k,'')) for k in ('kind','checkpoint','cohort','operation'))
        for trace in traces.values(): failure_counts[key][trace['first_failure'] or 'pair_candidates_exist']+=1
    summary['first_pair_failure_counts']={k:dict(v) for k,v in failure_counts.items()}
    measurements=defaultdict(list)
    for row in rows:
        if row['job']['kind']=='measurement' and 'measurements' in row: measurements[row['job']['checkpoint']].append(row)
    def mean(values):
        values=[v for v in values if v is not None]
        return sum(values)/len(values) if values else None
    summary['representation_and_segmentation']={}
    for checkpoint,items in measurements.items():
        fragments=[f['predicted'] for r in items for f in r['measurements']['fragments']]
        summary['representation_and_segmentation'][checkpoint]={
            'patterns':len(items),'sources':len({r['source_id'] for r in items}),
            'mean_fragment_balanced_accuracy':mean([f['balanced_accuracy'] for f in fragments]),
            'mean_fragment_iou':mean([f['iou'] for f in fragments]),
            'mean_predicted_fracture_fraction':mean([f['predicted_fraction'] for f in fragments]),
            'mean_true_fracture_fraction':mean([f['true_fraction'] for f in fragments]),
            'descriptor_effective_rank':mean([r['measurements']['representations']['descriptor']['effective_rank'] for r in items]),
            'descriptor_variance':mean([r['measurements']['representations']['descriptor']['mean_channel_variance'] for r in items]),
            'matcher_effective_rank':mean([r['measurements']['representations']['consumed_matcher_features']['effective_rank'] for r in items]),
            'sdf_l1':mean([r.get('reconstruction',{}).get('sdf_l1') for r in items]),
            'aggregation':'Unweighted means across the fixed subset; descriptive only, not independent-source significance estimates.'}
    summary['field_observations']=[{'pattern_id':r['pattern_id'],'source_id':r['source_id'],'field':r['job']['field'],**r['field_metrics']} for r in rows if 'field_metrics' in r]
    summary['nonfinite_gradient_probes']=[r['job'] for r in rows if 'gradients' in r and any(not x['finite'] for x in r['gradients']['objectives'].values())]
    summary['supported_explanations']=[]; summary['paired_interventions']=[]
    baseline={(r['job']['checkpoint'],r['job']['index'],r['job']['operation']):r for r in rows if r['job']['kind']=='intervention'}
    interventions=defaultdict(list)
    for row in rows:
        job=row['job']
        if job['kind']=='intervention' and job['operation']!='baseline' and 'metrics' in row:
            control='fixed_matches_no_field' if job['operation'].startswith('fixed_matches_') else 'conditioning_predicted' if job['operation'].startswith('conditioning_') else 'baseline'
            base=baseline.get((job['checkpoint'],job['index'],control))
            if base and 'metrics' in base: interventions[job['operation']].append((base,row))
    for operation,pairs in interventions.items():
        helped=[r['pattern_id'] for b,r in pairs if r['metrics']['success'] and not b['metrics']['success']]
        harmed=[r['pattern_id'] for b,r in pairs if b['metrics']['success'] and not r['metrics']['success']]
        unchanged=[r['pattern_id'] for b,r in pairs if b['metrics']['success']==r['metrics']['success']]
        support_changed=[{'pattern_id':r['pattern_id'],'before':{k:v['first_failure'] for k,v in b.get('filter_traces',{}).items()},
             'after':{k:v['first_failure'] for k,v in r.get('filter_traces',{}).items()}} for b,r in pairs
             if {k:v['first_failure'] for k,v in b.get('filter_traces',{}).items()}!={k:v['first_failure'] for k,v in r.get('filter_traces',{}).items()}]
        effect={'intervention':operation,'paired_examples':len(pairs),
            'improved_geometry':helped,'worsened_geometry':harmed,'unchanged_geometry':unchanged,
            'changed_first_pair_failures':support_changed,
            'supporting_evidence':{'geometric_changes':helped+harmed,'filter_boundary_changes':[r['pattern_id'] for r in support_changed]},
            'counterevidence':{'unchanged_geometry':unchanged,'worsened_geometry':harmed},
            'limitations':'Restricted deterministic validation subset. An effect supports sensitivity to this intervention, not a unique historical training cause. Oracle/threshold outputs are not production performance.'}
        summary['paired_interventions'].append(effect)
        if helped or harmed or support_changed: summary['supported_explanations'].append(effect)
    summary['unresolved']=['Gradient/representation associations do not prove historical causes.',
        'Intervention failure can leave several explanations indistinguishable.',
        'No architecture or threshold change is recommended automatically.']
    if remaining: summary['unresolved'].append('Coverage incomplete: inspect unrun_jobs before concluding.')
    write(output/'summary.json',summary)
    lines=['# Diagnostic evidence report','',f"Status: **{summary['status']}**. Completed {len(rows)}/{len(jobs)} jobs.",
           '', 'No optimizer updates. This report cannot authorize production training or change its acceptance gate.',
           '', '## Observed', '', '| Probe group | Examples | Sources | Geometric successes | Status counts |',
           '|---|---:|---:|---:|---|']
    for name,value in summary['observed'].items():
        lines.append(f"| {name} | {value['examples']} | {value['source_objects']} | {value['geometric_successes']} | {value['statuses']} |")
    lines+=['','## Representation and segmentation measurements','',
        '| Checkpoint | Patterns / sources | Mean fragment balanced accuracy | Descriptor rank | Matcher rank | SDF L1 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,v in summary['representation_and_segmentation'].items():
        lines.append(f"| {name} | {v['patterns']} / {v['sources']} | {v['mean_fragment_balanced_accuracy']:.5f} | {v['descriptor_effective_rank']:.3f} | {v['matcher_effective_rank']:.3f} | {v['sdf_l1']} |")
    lines+=['','Rank and consistency alone are not pass criteria. Field errors, baseline comparisons, calibration and gradient probes are recorded in summary.json and individual result files.']
    lines+=['','## Supported intervention effects','']
    for item in summary['paired_interventions']:
        lines.append(f"- {item['intervention']}: improved {len(item['improved_geometry'])}, worsened {len(item['worsened_geometry'])}, unchanged {len(item['unchanged_geometry'])} of {item['paired_examples']}. Exact affected pattern IDs are in summary.json.")
    lines+=['','These effects do not uniquely identify historical training causes. Individual results include the fixed inputs, raw measurements, and the intervention label.',
            '', '## Unresolved and limitations','']+['- '+x for x in summary['unresolved']]
    lines += [f'- Replay mismatches: {len(failures)}. Checkpoint changes: {changes}. Production file changes: {production_changes}.',
              '- See inventory.json, experiments.json, and results/ for evidence; null measurements indicate undefined/unavailable values, not successful zero errors.',
              '- Source-level counts prevent treating multiple fractures of one object as independent evidence.']
    if reason: lines+=['',f'Execution stopped: {reason}']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    try: render(output,rows)
    except Exception as error:
        write(output/'visualization_error.json',{'error':str(error)})
        summary['status']='partial_or_blocked'; summary['unresolved'].append('Visualization generation failed: '+str(error))
        write(output/'summary.json',summary)
        with open(output/'REPORT.md','a',encoding='utf-8') as report: report.write('\nVisualization generation failed; coverage is partial. See visualization_error.json.\n')
    archive=output/'diagnostic_bundle.tar.gz'
    temporary=output/'diagnostic_bundle.tar.gz.tmp'
    if archive.exists(): archive.unlink()  # Derived package only; incremental evidence remains intact.
    with tarfile.open(temporary,'w:gz') as bundle:
        for file in sorted(output.rglob('*')):
            if file.is_file() and file not in (archive,temporary) and file.suffix!='.tmp':
                bundle.add(file,arcname=str(file.relative_to(output)),recursive=False)
    os.replace(temporary,archive)
    if tree_bytes(output)>2*1024**3: raise RuntimeError('Diagnostic bundle exceeded 2 GiB')
    return summary


def render(output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curves=read(output/'training_curves.json') if (output/'training_curves.json').exists() else {}
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for name,points in curves.items():
        stage=int(name.split('/')[-1][1:])-1
        if points: axes[stage].plot([p['step'] for p in points],[p['validation'] for p in points],label=name)
    for i,ax in enumerate(axes):
        ax.set_title(f'Stage {i+1} logged validation'); ax.set_xlabel('Optimizer update'); ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(output/'training_curves.png',dpi=140); plt.close(fig)
    selected=[r for r in rows if r['job']['kind']=='measurement' and 'measurements' in r]
    if selected:
        counts=np_sum_hist(selected)
        fig,ax=plt.subplots(figsize=(6,4)); ax.bar([.05+.1*i for i in range(10)],counts,width=.085)
        ax.set(xlabel='Predicted fracture probability',ylabel='Points',title='Diagnostic subset (multiple checkpoints)')
        fig.tight_layout(); fig.savefig(output/'fracture_histogram.png',dpi=140); plt.close(fig)


def np_sum_hist(rows):
    return [sum(f['predicted']['probability_histogram'][i] for row in rows for f in row['measurements']['fragments']) for i in range(10)]
