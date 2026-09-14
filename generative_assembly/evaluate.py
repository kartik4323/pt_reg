"""Reference-dependent evaluation; not imported by model workers or training."""
from __future__ import annotations
import collections
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion, distance_transform_edt, binary_closing
from PIL import Image
from . import data, geometry as g, render
from .storage import read, write, digest


def chamfer(a,b):
    return float((cKDTree(a).query(b)[0].mean()+cKDTree(b).query(a)[0].mean())/2)


def pose_metrics(case, ref, poses, cfg, retained=None):
    retained=retained or list(range(len(case['original'])))
    target=ref['poses'][retained]
    g.check_poses(poses,len(retained))
    errors=[]
    for j,i in enumerate(retained):
        p=case['original'][i]
        p=p[np.linspace(0,len(p)-1,min(len(p),cfg['evaluation_points'])).astype(int)]
        p=(p-case['centers'][i])/case['scale']
        errors.append(chamfer(g.apply(p,poses[j]),g.apply(p,target[j])))
    relative=poses[:,:3,:3]@target[:,:3,:3].transpose(0,2,1)
    angles=np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))
    translations=np.linalg.norm(poses[:,:3,3]-target[:,:3,3],axis=1)
    moving=[j for j,i in enumerate(retained) if i!=case['anchor']]
    return dict(success=bool(max(errors)<=cfg['evaluation']['threshold']),part_chamfer=errors,max_part_chamfer=max(errors),
                rotation_degrees=angles[moving].tolist(),translation_error=translations[moving].tolist(),
                threshold_success={str(t):bool(max(errors)<=t) for t in [0.005,0.01,0.02]},
                symmetry_note='part_surface_distance_is_primary; raw_rotation_is_not_symmetry_reduced')


def bootstrap_delta(a,b,seed=42,count=2000):
    sources=sorted(set(a)&set(b))
    if not sources: return dict(sources=0,difference=None,ci95=None)
    delta=np.array([np.mean(a[s])-np.mean(b[s]) for s in sources])
    rng=np.random.default_rng(seed)
    draws=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(count)])
    return dict(sources=len(sources),difference=float(delta.mean()),ci95=np.quantile(draws,[0.025,0.975]).tolist())


def evaluate(store,split):
    cfg=store.config
    reference_index=Path(store.dataset).parent/'evaluator_only'/'index.json'
    reference_hash=digest(reference_index) if reference_index.exists() else None
    identity_path=store.root/'evaluation'/'reference_identity.json'
    identity=dict(index_sha256=reference_hash)
    if identity_path.exists() and read(identity_path)!=identity:
        raise ValueError('Evaluator reference identity changed; use a new study root')
    write(identity_path,identity)
    records={c['id']:c for c in read(store.dataset)['cases'] if c['split']==split}
    cache={}; rows=[]
    for job in store.jobs(split=split):
        if job['stage'] not in ('E1','E2','E3','E4','E5','E6','E7'): continue
        base=job.get('output',{}).get('base_case_id',job.get('extra',{}).get('base_case',job['case_id']))
        if base not in records: continue
        if base not in cache:
            case=data.load_case(store.dataset,records[base],cfg); cache[base]=(case,data.reference(store.dataset,case))
        case,ref=cache[base]
        row=dict(job_id=job['job_id'],case_id=job['case_id'],base_case_id=base,source_id=job['source_id'],stage=job['stage'],arm=job['arm'],
                 oracle=job['oracle'],smoke=job['smoke'],status=job['status'],seconds=job['seconds'],factors=job.get('extra',{}),metrics=None)
        if job['status']=='failed':
            row['metrics']=dict(success=False,failure=job['error'])
        elif job['output'].get('not_applicable'):
            row['status']='not_applicable'; row['reason']=job['output']['not_applicable']
        elif ref is None:
            row['status']='unscored_no_reference'; row['input_only_diagnostics']=job['output'].get('diagnostics')
        elif job['stage'] in ('E3','E4','E5','E6','E7'):
            with np.load(store.artifact(job,'poses.npz'),allow_pickle=False) as f: poses=f['normalized']
            row['metrics']=pose_metrics(case,ref,poses,cfg,job['output'].get('retained_ids'))
            row['input_only_diagnostics']=job['output']['diagnostics']
            if job['stage']=='E3':
                with np.load(store.artifact(job,'starts.npz'),allow_pickle=False) as f: starts=f['poses']
                before=[pose_metrics(case,ref,p,cfg) for p in starts]
                row['metrics'].update(candidate_recall=any(m['success'] for m in before),best_candidate_error=min(m['max_part_chamfer'] for m in before),
                                      initially_successful=before[0]['success'] if len(before)==1 else None)
        elif job['stage']=='E2' and 'complete' in ref:
            with np.load(store.artifact(job,'shape.npz'),allow_pickle=False) as f: p=f['aligned']
            q=ref['complete']; a=cKDTree(p).query(q)[0]; b=cKDTree(q).query(p)[0]
            precision=float(np.mean(b<0.01)); recall=float(np.mean(a<0.01))
            row['metrics']=dict(shape_chamfer=chamfer(p,q),fscore_001=2*precision*recall/max(precision+recall,1e-12),
                                observed_heldout_error=job['output']['alignment']['heldout_error'],valid_surface=True,
                                alignment='input_only_not_oracle_aligned',thickness_and_cavity_metrics=None)
        elif job['stage']=='E1' and 'complete' in ref:
            parents=store.find('E0',base,'prepare')
            camera=read(store.artifact(parents[0],'camera.json'))
            truth=render.render(ref['complete'],None,None,camera,cfg['splat_radius'])['valid']
            image=Image.open(store.artifact(job,'image.png')).convert('RGB').resize((cfg['pixels'],cfg['pixels']))
            predicted=render.foreground(image)
            # Explicit point-splat silhouette proxy. No filled convex hull/cavity replacement.
            truth=binary_closing(truth,iterations=2); predicted=binary_closing(predicted,iterations=2)
            intersection=(truth&predicted).sum(); union=(truth|predicted).sum()
            drift=None; missing_precision=None; missing_recall=None
            exterior=ref['exterior'][case['anchor']]
            if exterior is not None and exterior.any():
                known=render.render(case['points'][case['anchor']][exterior],None,None,camera,cfg['splat_radius'])['valid']
                surviving=known & (truth & ~binary_erosion(truth))
                boundary=predicted & ~binary_erosion(predicted)
                if surviving.any() and boundary.any():
                    drift=float(distance_transform_edt(~boundary)[surviving].mean()/cfg['pixels'])
                missing=truth & ~known; added=predicted & ~known
                missing_precision=float((missing & added).sum()/max(added.sum(),1))
                missing_recall=float((missing & added).sum()/max(missing.sum(),1))
            row['metrics']=dict(silhouette_iou=float(intersection/max(union,1)),silhouette_kind='point_splat_proxy_2px_closing',
                                exterior_contour_drift=drift,missing_region_precision=missing_precision,missing_region_recall=missing_recall,viewpoint_verified=None,
                                note='camera changes and surviving exterior need the saved blinded-review sheet')
        rows.append(row)
    out=store.root/'evaluation'/split
    out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.jsonl').write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
    groups=collections.defaultdict(list)
    for r in rows: groups[(r['stage'],r['arm'])].append(r)
    summaries=[]
    for (stage,arm),items in sorted(groups.items()):
        scored=[r for r in items if r['metrics'] is not None]
        success=[r for r in scored if 'success' in r['metrics']]
        by_source=collections.defaultdict(list)
        for r in success: by_source[r['source_id']].append(float(r['metrics']['success']))
        scalar_values=collections.defaultdict(lambda:collections.defaultdict(list))
        for r in scored:
            for name,value in r['metrics'].items():
                if isinstance(value,(float,int)) and not isinstance(value,bool):
                    scalar_values[name][r['source_id']].append(value)
        means={name:float(np.mean([np.mean(v) for v in source.values()])) for name,source in scalar_values.items()}
        summaries.append(dict(stage=stage,arm=arm,requested=len(items),failed=sum(r['status']=='failed' for r in items),
                              scored=len(scored),sources=len({r['source_id'] for r in items}),
                              scalar_source_macro_means=means,
                              success_rate=float(np.mean([np.mean(x) for x in by_source.values()])) if by_source else None))
    baseline={r['case_id']:r for r in rows if r['stage']=='E4' and r['arm']=='B0__refine' and r['metrics'] and 'success' in r['metrics']}
    comparisons=[]
    for (stage,arm),items in sorted(groups.items()):
        if stage not in ('E4','E5','E6'): continue
        a,b=collections.defaultdict(list),collections.defaultdict(list); damage=eligible=0
        for row in items:
            base=baseline.get(row['case_id']); m=row['metrics']
            if not base or not m or 'success' not in m: continue
            a[row['source_id']].append(float(m['success'])); b[row['source_id']].append(float(base['metrics']['success']))
            if base['metrics']['success']: eligible+=1; damage+=int(not m['success'])
        if not a: continue
        delta=bootstrap_delta(a,b,count=cfg['evaluation']['bootstrap'])
        comparisons.append(dict(stage=stage,arm=arm,baseline='B0__refine',**delta,damage_rate=damage/eligible if eligible else None,
                                damage_denominator=eligible))
    targeted=[]
    def compare(stage,arm,baseline_stage,baseline_arm):
        controls={r['case_id']:r for r in groups.get((baseline_stage,baseline_arm),[])}
        a,b=collections.defaultdict(list),collections.defaultdict(list)
        for r in groups.get((stage,arm),[]):
            c=controls.get(r['case_id']); m=r['metrics']; n=c['metrics'] if c else None
            if not m or not n or 'success' not in m or 'success' not in n: continue
            a[r['source_id']].append(float(m['success'])); b[r['source_id']].append(float(n['success']))
        if a:
            targeted.append(dict(stage=stage,arm=arm,baseline_stage=baseline_stage,baseline=baseline_arm,
                                 paired_cases=sum(map(len,a.values())),**bootstrap_delta(a,b,count=cfg['evaluation']['bootstrap'])))
    primary_arm=f'B2__{cfg["primary_model"]}__{cfg["primary_input"]}__refine'
    for control in [f'B1__raw__{cfg["primary_input"]}__refine','B3__refine','B4__refine','B5__refine','B6__refine']:
        compare('E4',primary_arm,'E4',control)
    for K in sorted(set([1,2,4,len(cfg['image_seeds'])])):
        compare('E5',f'gated__K{K}','E5',f'always__K{K}')
    for seed in cfg['training']['seeds']:
        for control in ['geometry','all','random_count']:
            compare('E6',f'filtered__seed{seed}','E6',f'{control}__seed{seed}')
        compare('E6',f'filtered__seed{seed}','E5',f'always__K{len(cfg["image_seeds"])}')
    robust=[]
    conditions=sorted({(r['factors'].get('factor','unknown'),str(r['factors'].get('value'))) for r in rows if r['stage']=='E7'})
    for factor,value in conditions:
        condition=[r for r in rows if r['stage']=='E7' and r['factors'].get('factor','unknown')==factor and str(r['factors'].get('value'))==value]
        controls={r['case_id']:r for r in condition if r['arm']=='B0'}
        a,b=collections.defaultdict(list),collections.defaultdict(list)
        for r in condition:
            if r['arm']!='B2': continue
            c=controls.get(r['case_id']); m=r['metrics']; n=c['metrics'] if c else None
            if not m or not n or 'success' not in m or 'success' not in n: continue
            a[r['source_id']].append(float(m['success'])); b[r['source_id']].append(float(n['success']))
        robust.append(dict(factor=factor,value=value,arm='B2',baseline='B0',**bootstrap_delta(a,b,count=cfg['evaluation']['bootstrap'])))
    label_path=store.root/'pseudo_labels'/'train.json'
    if split=='train' and label_path.exists():
        quality=[]
        for label in read(label_path)['labels']:
            if label['case_id'] not in cache: continue
            c,r=cache[label['case_id']]
            quality.append(dict(case_id=label['case_id'],source_id=label['source_id'],accepted=label['accepted'],
                                metrics=pose_metrics(c,r,np.asarray(label['poses']),cfg) if r is not None else None))
        write(out/'pseudo_label_quality.json',dict(evaluation_only=True,selection_uses_reference=False,labels=quality))
    primary=next((c for c in comparisons if c['stage']=='E4' and c['arm']==primary_arm),None)
    gate='not_measured'
    if cfg['smoke']: gate='SMOKE_NOT_RESEARCH_EVIDENCE'
    elif primary and primary['ci95'] and primary['sources']>=2:
        gate='pass' if primary['difference']>=cfg['evaluation']['success_gain'] and primary['ci95'][0]>0 and primary['damage_rate'] is not None and primary['damage_rate']<=cfg['evaluation']['damage_max'] else 'not_passed_or_inconclusive'
    summary=dict(split=split,smoke=cfg['smoke'],groups=summaries,paired_comparisons=comparisons,targeted_comparisons=targeted,
                 robustness_comparisons=robust,evaluator_index_sha256=reference_hash,E4_primary_gate=gate,
                 reference_use='evaluation_only',E1_gate='requires_manual_camera_and_exterior_review',
                 E2_gate='inspect_paired_shape_and_exterior_metrics',E6_gate='compare_student_to_geometry_learner_and_teacher; B0 comparison is supplementary',
                 limitations=['collision is an oriented-surface proxy','exterior mask is a curvature heuristic','raw rotation metrics are not symmetry reduced'])
    write(out/'summary.json',summary)
    # Review rows are requests for human judgements, never fake automatic labels.
    previous_review={r['job_id']:r for r in read(out/'image_review.json')} if (out/'image_review.json').exists() else {}
    review=[previous_review.get(r['job_id'],dict(job_id=r['job_id'],source_id=r['source_id'],image=str(store.root/'jobs'/r['job_id']/'image.png'),
                 single_object=None,same_camera=None,preserved_exterior=None,notes='')) for r in rows if r['stage']=='E1']
    write(out/'image_review.json',review)
    text=['# Experiment results',f'Split: {split}. E4 gate: **{gate}**.',
          'Smoke rows test software only. Oracle results are privileged controls. See metrics.jsonl for denominators and references.',
          '| Stage | Arm | Sources | Jobs | Failed | Source-macro success |','|---|---|---:|---:|---:|---:|']
    for s in summaries:
        rate='unscored' if s['success_rate'] is None else f'{s["success_rate"]:.3f}'
        text.append(f'| {s["stage"]} | {s["arm"]} | {s["sources"]} | {s["requested"]} | {s["failed"]} | {rate} |')
    text+=['','See paired_comparisons and targeted_comparisons in summary.json. Training loss or a complete job is not a scientific pass.']
    (out/'REPORT.md').write_text('\n\n'.join(text[:3])+'\n\n'+'\n'.join(text[3:])+'\n',encoding='utf-8')
    return summary
