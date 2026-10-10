"""Source-paired evaluation gates. This module is NEVER imported by inference."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from .storage import read, write
from .evaluate import bootstrap_delta


def assess(run, dataset, split, case_ids):
    run=Path(run); cfg=read(run/'study.json')['identity']['config']
    cases={c['id']:c for c in read(dataset)['cases'] if c['split']==split and c['id'] in case_ids}
    evaluation=run/'evaluation'/split/'metrics.jsonl'
    rows=[__import__('json').loads(s) for s in evaluation.read_text().splitlines()] if evaluation.exists() else []
    e2={r['job_id']:r for r in rows if r['stage']=='E2'}
    model,kind=cfg['primary_model'],cfg['primary_input']
    raw={r['case_id']:r for r in rows if r['stage']=='E2' and r['arm']==f'raw__{kind}'}
    deploy={r['case_id']:r for r in rows if r['stage']=='E4' and r['arm']=='B2__deploy'}
    baseline={r['case_id']:r for r in rows if r['stage']=='E4' and r['arm']=='B0__refine'}
    reviews=read(run/'review.json') if (run/'review.json').exists() else {}
    covered=0; reviewed=0; pairs=0; public=True; shape=defaultdict(list); fscores=defaultdict(list)
    success=defaultdict(list); base_success=defaultdict(list); damage=[]
    for cid,case in cases.items():
        prior_file=run/'priors'/f'{cid}.json'
        priors=read(prior_file).get(f'{model}__{kind}',{}) if prior_file.exists() else {}
        ids=priors.get('job_ids',[])
        public=public and all(j in e2 and not e2[j].get('oracle') and not e2[j].get('smoke') for j in ids)
        if len(ids)==3:
            covered+=1
            fields=('bottle_identity','full_outline','same_view','surface_quality')
            if all(all(reviews.get(j,{}).get(f) is True for f in fields) for j in ids): reviewed+=1
        a=e2.get(ids[0],{}).get('metrics') if ids else None
        b=raw.get(cid,{}).get('metrics')
        if a and b and a.get('missing_surface_available') and b.get('missing_surface_available'):
            pairs+=1
            shape[case['source_id']].append((a['missing_surface_distance'],b['missing_surface_distance']))
            fscores[case['source_id']].append((a['missing_surface_fscore'],b['missing_surface_fscore']))
        ma=deploy.get(cid,{}).get('metrics') or {}; mb=baseline.get(cid,{}).get('metrics') or {}
        # Missing/failed jobs count as unsuccessful, not removed from the experiment.
        success[case['source_id']].append(float(ma.get('success',False)))
        base_success[case['source_id']].append(float(mb.get('success',False)))
        if mb.get('success'): damage.append(not ma.get('success',False))
    def mean_pair(values):
        return np.mean([np.mean(v,axis=0) for v in values.values()],axis=0) if values else [None,None]
    new_dist,old_dist=mean_pair(shape); new_f,old_f=mean_pair(fscores)
    gain=float((old_dist-new_dist)/old_dist) if old_dist is not None and old_dist>0 else None
    delta=bootstrap_delta(success,base_success,count=cfg['evaluation']['bootstrap'])
    damage_rate=float(np.mean(damage)) if damage else None
    n=len(cases); source_count=len({c['source_id'] for c in cases.values()})
    checks=dict(source_count=source_count>=(20 if split=='test' else 10),
        three_prior_coverage=n>0 and covered/n>=.8,
        reviewed_three_prior_coverage=n>0 and reviewed/n>=.8,
        shape_pair_coverage=n>0 and pairs/n>=.8,
        missing_distance=gain is not None and gain>=.1,
        missing_fscore=new_f is not None and new_f>=old_f,
        assembly_gain=delta['difference'] is not None and delta['difference']>=.05,
        assembly_ci=delta['ci95'] is not None and delta['ci95'][0]>0,
        damage=damage_rate is not None and damage_rate<=.05,
        non_smoke=not cfg['smoke'], references_available=n>0 and all(
            (raw.get(c,{}).get('metrics') or {}).get('missing_surface_available',False) for c in cases))
    checks['public_hypotheses_only']=public
    checks={key:bool(value) for key,value in checks.items()}
    result=dict(split=split,eligible_cases=n,sources=source_count,three_prior_cases=covered,
        reviewed_three_prior_cases=reviewed,shape_paired_cases=pairs,
        missing_distance_relative_gain=gain,missing_fscore_delta=float(new_f-old_f) if new_f is not None else None,
        assembly=delta,damage_rate=damage_rate,damage_denominator=len(damage),checks=checks,
        approved=split=='test' and all(checks.values()),
        status='passed' if all(checks.values()) else 'preliminary_or_failed_gates',
        ranking='fixed_input_selected_rank1_not_best_GT_candidate')
    write(run/'evaluation'/split/'promotion.json',result)
    return result
