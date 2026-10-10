"""Bounded v3 phases. Identity, source splits and budget survive VM resumptions."""
import argparse
import json
import os
import subprocess
import sys
import zipfile
import shutil
import numpy as np
from pathlib import Path
from . import config,data
from .storage import read,write,digest,Store
from .resources import BudgetExhausted

REPRESENTATIONS={
    'legacy_splat': dict(render_mode='splat',framing_fill=0.,mask_type='inverted'),
    'dense': dict(render_mode='surfel',framing_fill=0.,mask_type='inverted'),
    'frame35': dict(render_mode='surfel',framing_fill=.35,mask_type='inverted'),
    'frame50': dict(render_mode='surfel',framing_fill=.5,mask_type='inverted'),
    'exterior35': dict(render_mode='surfel',framing_fill=.35,mask_type='exterior'),
    'depth35': dict(render_mode='surfel',framing_fill=.35,mask_type='inverted',model='sd15_depth'),
}


def estimate_phase(root, specs, bucket):
    """Conservative pilot-based admission before a phase; worker deadlines still enforce the cap."""
    path=Path(root)/'budget.json'
    if not path.exists(): return
    ledger=read(path); estimate=0.; unknown=[]
    for name,cfg,records,stages in specs:
        jobs=[read(p) for p in (Path(root)/name/'jobs').glob('*/result.json')]
        for kind,stage,model,multiplier in (
            ('image','E1',cfg['primary_model'],len(cfg['input_types'])*len(cfg['image_seeds'])),
            ('reconstruction','E2','instantmesh',len(cfg['input_types'])*(len(cfg['image_seeds'])+1)+
                int(cfg['oracles'])*(1+int(cfg.get('oracle_full_frame_control',False))))):
            if stage not in stages: continue
            cached=sum(r['status']=='complete' and r['stage']==stage and r['case_id'] in {c['id'] for c in records}
                       and r.get('output',{}).get('model')!='raw' for r in jobs) if stage=='E1' else sum(
                           r['status']=='complete' and r['stage']==stage and r['case_id'] in {c['id'] for c in records} for r in jobs)
            history=[h['seconds'] for h in ledger['history'] if h['kind']==kind and h.get('model')==model]
            if not history: unknown.append(dict(kind=kind,model=model)); continue
            estimate+=max(0,len(records)*multiplier-cached)*max(history)*1.5
    remaining=ledger['limits'][bucket]-ledger['used'].get(bucket,0)-sum(
        r['seconds'] for r in ledger['reservations'].values() if r['bucket']==bucket)
    report=dict(estimated_seconds=estimate,remaining_seconds=remaining,bucket=bucket,unknown_runtime=unknown,
                conservative_all_candidates=True,admitted=remaining>=estimate)
    write(Path(root)/'phase_admission.json',report)
    if not report['admitted']: raise BudgetExhausted('Phase estimate exceeds remaining budget; retained at a resumable phase boundary')


def stress_dataset(dataset, records, output, condition, cfg):
    """Perturb full public input, not just its assembly subsample; evaluator files remain separate."""
    output=Path(output); path=output/'dataset.json'
    if path.exists(): data.inventory(path); return path
    source=Path(dataset).parent; rows=[]; refs={}
    index=read(source/'evaluator_only'/'index.json') if (source/'evaluator_only'/'index.json').exists() else {}
    for rec in records:
        case=data.load_case(dataset,rec,cfg); rng=np.random.default_rng(data.seed_for(rec['id'],condition,'stress_v3'))
        points=[]
        for p in case['original']:
            if condition=='noise': p=p+rng.normal(0,.005*case['scale'],p.shape)
            elif condition=='dropout': p=p[np.sort(rng.choice(len(p),max(16,int(.75*len(p))),replace=False))]
            else: raise ValueError('Unknown stress condition')
            points.append(p)
        entry,_=data.save_case(output,rec['source_id'],rec['id'],'test',rec['category'],points,source_hash=rec.get('source_hash'))
        rows.append(entry)
        if rec['id'] in index:
            old=index[rec['id']]; target=output/old['path']; target.parent.mkdir(parents=True,exist_ok=True)
            # Perturbation changes point counts. Fracture labels are not valid and are not copied.
            with np.load(source/old['path'],allow_pickle=False) as f:
                np.savez_compressed(target,**{k:f[k] for k in ('world_transforms','complete') if k in f.files})
            refs[rec['id']]=dict(path=old['path'],sha256=digest(target))
    write(output/'evaluator_only'/'index.json',refs)
    write(path,dict(schema_version=1,cases=rows,provenance=dict(condition=condition,
        noise_anchor_diameter=.005 if condition=='noise' else 0,dropout=.25 if condition=='dropout' else 0,
        public_dataset_sha256=digest(dataset),evaluator_only=True)))
    return path


def matte_lock(directory):
    directory=Path(directory).resolve(); directory.mkdir(parents=True,exist_ok=True)
    os.environ['U2NET_HOME']=str(directory)
    from rembg import new_session
    new_session('isnet-general-use',providers=['CPUExecutionProvider'])
    path=directory/'isnet-general-use.onnx'
    if not path.is_file(): raise RuntimeError('Expected pinned rembg 2.0.67 ONNX layout')
    lock=dict(model='isnet-general-use',weights=str(path),sha256=digest(path),provider='CPUExecutionProvider')
    if (directory/'lock.json').exists() and read(directory/'lock.json')!=lock:
        raise RuntimeError('Matte checkpoint changed; preserve old lock and use a new model directory')
    write(directory/'lock.json',lock); return lock


def build_config(base,name,matte,images_python,mesh_python,matte_python,budget,bucket,representation='frame35'):
    cfg=config.load(base)
    factors=REPRESENTATIONS[representation]
    model=factors.get('model','sd15')
    if name.startswith(('pilot_','models_','validation_')): model=name.split('_')[-1]
    cfg.update(experiment_profile=name,prompt_variant='bottle',canvas_fill_target=0.,
        render_points=8192,n_render_views=1,n_instantmesh_views=1,
        image_models=[model],primary_model=model,primary_input='F',input_types=['F','A'],image_seeds=[11,23,37,51],
        skip_invalid_e1=True,reconstruct_all_for_E5=True,prior_count=3,oracles=True,oracle_full_frame_control=True,
        **{k:v for k,v in factors.items() if k!='model'})
    cfg['images'].update(python=images_python,bg_obj_threshold=0,resolution=1024 if model in ('sdxl','qwen') else 512)
    if model=='qwen': cfg['images']['qwen_offload']='sequential'
    cfg['matte'].update(enabled=True,python=matte_python,**{k:matte[k] for k in ('model','weights','sha256')})
    cfg['reconstruction'].update(python=mesh_python,input_mode='rgba')
    cfg['reconstruction']['template_min_component_area_fraction']=.9
    cfg['solver'].update(continuous_scale=True,scale_bounds=[.5,6.])
    cfg['resources'].update(gpu_preflight=True,budget_path=str(budget),budget_bucket=bucket)
    if name.startswith('pilot_'): cfg['image_seeds']=[11]
    return cfg


def execute(args,name,cfg,split,records,stages):
    run=args.root/name; configs=args.root/'configs'; configs.mkdir(exist_ok=True)
    source=configs/f'{name}.json'; locked=configs/f'{name}-locked.json'
    if source.exists() and read(source)!=cfg: raise ValueError('Phase config changed; use a new RUN_GROUP')
    write(source,cfg)
    if not locked.exists():
        subprocess.run([args.images_python,'-m','generative_assembly','lock-models','--config',str(source),'--out',str(locked)],check=True)
    run.mkdir(exist_ok=True)
    metadata=dict(dataset=str(args.dataset.resolve()),profile=name,split=split,case_ids=[r['id'] for r in records])
    if (run/'experiment.json').exists() and read(run/'experiment.json')!=metadata:
        raise ValueError('Source cohort changed; use a new RUN_GROUP')
    write(run/'experiment.json',metadata)
    argv=[args.images_python,'-m','generative_assembly','run','--config',str(locked),
          '--dataset',str(args.dataset),'--root',str(run),'--split',split,'--case-ids',*[r['id'] for r in records],
          '--stages',*stages,'--allow-failures']
    if args.retry_failed: argv.append('--retry-failed')
    with (args.root/f'{name}-{args.phase}.log').open('a',encoding='utf-8') as log:
        result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'Phase stopped; inspect {name}-{args.phase}.log. Resume same group/config explicitly.')
    return run


def collect_zip(root,output):
    root=Path(root).resolve(); output=Path(output).resolve()
    if output.exists(): raise ValueError('ZIP destination exists; choose a new file')
    if output.is_relative_to(root): raise ValueError('ZIP must be outside run group')
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for p in root.rglob('*'):
            if not p.is_file() or p.suffix in ('.ckpt','.pt','.pth','.onnx','.bin','.safetensors'): continue
            if p.name=='running.lock': continue
            # Include configuration, portable metrics, notebook, priors and exact failed-job diagnostics.
            rejected=False
            if p.name in ('mesh.obj','shape.npz') and (p.parent/'result.json').exists():
                rejected=read(p.parent/'result.json').get('output',{}).get('template_rejected',False)
            if (rejected or 'jobs' not in p.parts or p.name in ('result.json','error.txt','worker.log','request.json',
                'backend.json','alignment.json','gpu_preflight.json','matte.json','camera.json','shape.npz',
                'poses.npz','render.npz','point_provenance.npz') or
                p.name.startswith(('image','alpha','renderer','reconstruction_input','mask','control','observed_mask','exterior_mask'))):
                z.write(p,str(p.relative_to(root)))
    return str(output)


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('lock-matte'); q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('collect'); q.add_argument('--root',type=Path,required=True); q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('phase')
    q.add_argument('--phase',required=True,choices=['pilot','representation','models','reconstruction','validate','freeze','test','stress','promote'])
    for key in ('root','dataset','base','matte_lock'): q.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    for key in ('images_python','mesh_python','matte_python'): q.add_argument('--'+key.replace('_','-'),required=True)
    q.add_argument('--representation',choices=REPRESENTATIONS)
    q.add_argument('--finalists',nargs='*',default=['models_sd15','models_sdxl'])
    q.add_argument('--stages',nargs='+'); q.add_argument('--retry-failed',action='store_true')
    q.add_argument('--profiles',nargs='+')
    a=p.parse_args()
    if a.command=='lock-matte': print(json.dumps(matte_lock(a.out),indent=2)); return
    if a.command=='collect': print(collect_zip(a.root,a.out)); return
    try: run_phase(a)
    finally:
        if a.root.exists():
            from .imagination_report import export,prepare_review
            export(a.root)
            print(prepare_review(a.root))


def run_phase(a):
    a.root=a.root.resolve(); a.root.mkdir(parents=True,exist_ok=True)
    ds=data.inventory(a.dataset); matte=read(a.matte_lock)
    chosen=read(a.root/'representation.json')['profile'] if (a.root/'representation.json').exists() else None
    if chosen and a.representation and chosen!=a.representation:
        raise ValueError('Representation already locked by model screening; use a new configuration identity')
    a.representation=a.representation or chosen or 'frame35'
    settings=dict(dataset_sha256=digest(a.dataset),base_sha256=digest(a.base),matte_sha256=matte['sha256'],
                  scope='bottles_pretrained_image_priors_only',budget_hours=24)
    if (a.root/'cycle.json').exists() and read(a.root/'cycle.json')!=settings:
        raise ValueError('Cycle inputs changed; use a new RUN_GROUP')
    write(a.root/'cycle.json',settings)
    if (a.root/'test_started.json').exists() and a.phase not in ('test','stress','promote'):
        raise ValueError('Test cohort exposed: development/refreeze forbidden. Existing test may only resume unchanged.')
    counts={s:len({c['source_id'] for c in ds['cases'] if c['split']==s}) for s in ('dev','test')}
    bucket='validation' if a.phase in ('validate','freeze','test','stress','promote') else 'screen'
    n={'pilot':1,'representation':3,'models':3,'reconstruction':3,'validate':10,'freeze':10,'test':20,'stress':20,'promote':20}[a.phase]
    split='test' if a.phase in ('test','stress','promote') else 'dev'
    if counts[split]<n: raise ValueError(f'{a.phase} requires {n} independent {split} sources; found {counts[split]}. Results remain preliminary.')
    records=data.balanced_cases(ds['cases'],split,n)
    names=[]; stages=a.stages or (['E0','E1'] if a.phase in ('representation','models') else
                               ['E0','E1','E2'] if a.phase=='pilot' else ['E0','E1','E2','E4'])
    if a.phase=='pilot': names=['pilot_sd15','pilot_sdxl','pilot_qwen']
    if a.phase=='representation': names=list(REPRESENTATIONS)
    if a.phase=='models': names=['models_sd15','models_sdxl','models_qwen']
    if a.phase=='models' and not chosen:
        write(a.root/'representation.json',dict(profile=a.representation,selection_basis='explicit_development_screen_only'))
    if a.phase=='validate':
        if len(a.finalists)>2 or not a.finalists: raise ValueError('Advance one or two model finalists')
        if any(n not in ('models_sd15','models_sdxl','models_qwen') for n in a.finalists): raise ValueError('Finalists must be screened model profile names')
        names=['validation_'+x.split('_')[-1] for x in a.finalists]
    if a.profiles:
        if not names or not set(a.profiles).issubset(names): raise ValueError('PROFILES must belong to this phase')
        names=[n for n in names if n in a.profiles]
    if names:
        specs=[(name,build_config(a.base,name,matte,a.images_python,a.mesh_python,a.matte_python,
            a.root/'budget.json',bucket,name if name in REPRESENTATIONS else a.representation),records,stages) for name in names]
        if a.phase!='pilot': estimate_phase(a.root,specs,bucket)
        for name in names:
            if name=='models_qwen' and (a.root/'pilot_qwen').exists():
                failures=[read(p) for p in (a.root/'pilot_qwen'/'jobs').glob('*/result.json')]
                failures=[r for r in failures if r['stage']=='E1' and r.get('failure_kind')=='unsupported_capability']
                if failures:
                    write(a.root/'qwen_skipped.json',dict(reason='unsupported_capability',pilot_jobs=[r['job_id'] for r in failures]))
                    continue
            representation=name if name in REPRESENTATIONS else a.representation
            cfg=build_config(a.base,name,matte,a.images_python,a.mesh_python,a.matte_python,a.root/'budget.json',bucket,representation)
            execute(a,name,cfg,split,records,stages)
    if a.phase=='reconstruction':
        names=a.finalists
        specs=[(name,read(a.root/'configs'/f'{name}.json'),records,a.stages or ['E2']) for name in names]
        estimate_phase(a.root,specs,'screen')
        for name in names:
            run=a.root/name
            if not (run/'study.json').exists(): raise ValueError('Run models phase before reconstruction')
            cfg=read(a.root/'configs'/f'{name}.json')
            execute(a,name,cfg,split,records,a.stages or ['E2'])
            store=Store(run,read(a.root/'configs'/f'{name}-locked.json'),a.dataset,a.retry_failed)
            from .experiments import _reconstruct
            # Fixed seed-11 paired raw/alpha input ablation, never selected as deployable priors.
            for rec in records:
                case=data.load_case(a.dataset,rec,store.config)
                for row in store.find('E1',rec['id']):
                    if row['output'].get('seed')==11 and row['output'].get('selection',{}).get('valid_foreground'):
                        _reconstruct(store,case,row,input_mode='raw_rgb')
    if a.phase in ('validate','freeze','test','stress','promote'):
        from .evaluate import evaluate
        from .promotion import assess
        from .export import freeze,bundle
        if a.phase in ('validate','freeze'):
            names=names or sorted(r.name for r in a.root.glob('validation_*') if r.is_dir())
            if not names: raise ValueError('Run validate phase before freeze')
            results=[]
            for name in names:
                store=Store(a.root/name,read(a.root/'configs'/f'{name}-locked.json'),a.dataset,a.retry_failed)
                evaluate(store,'dev'); result=assess(store.root,a.dataset,'dev',[r['id'] for r in records])
                results.append((name,result))
            if a.phase=='freeze':
                selected=sorted(results,key=lambda item:(-float(item[1]['assembly']['difference'] or 0),
                    -float(item[1]['missing_distance_relative_gain'] or 0),item[0]))[0][0]
                store=Store(a.root/selected,read(a.root/'configs'/f'{selected}-locked.json'),a.dataset)
                freeze(store); write(a.root/'finalist.json',dict(profile=selected,selection_basis='development_only',results=results))
        else:
            selected=read(a.root/'finalist.json')['profile']; run=a.root/selected
            store=Store(run,read(a.root/'configs'/f'{selected}-locked.json'),a.dataset,a.retry_failed)
            from .export import require_frozen
            require_frozen(store)
            if a.phase=='test':
                marker=dict(profile=selected,case_ids=[r['id'] for r in records],identity=store.identity)
                if (a.root/'test_started.json').exists() and read(a.root/'test_started.json')!=marker:
                    raise ValueError('Frozen test identity/cohort changed')
                estimate_phase(a.root,[(selected,store.config,records,a.stages or ['E0','E1','E2','E4'])],'validation')
                write(a.root/'test_started.json',marker)
                # Same frozen store/config; test cohort does not replace the dev experiment manifest.
                from .experiments import run_stage
                for stage in a.stages or ['E0','E1','E2','E4']:
                    for rec in records: run_stage(store,stage,data.load_case(a.dataset,rec,store.config),True); store.index()
            if a.phase=='stress':
                if not (a.root/'test_started.json').exists(): raise ValueError('Run frozen test before optional stress')
                for condition in ('noise','dropout'):
                    name=f'stress_{condition}'
                    estimate_phase(a.root,[(name,store.config,records,['E0','E1','E2','E4'])],'validation')
                    dataset=stress_dataset(a.dataset,records,a.root/name/'dataset',condition,store.config)
                    stressed=Store(a.root/name,store.config,dataset,a.retry_failed); freeze(stressed)
                    write(stressed.root/'experiment.json',dict(dataset=str(dataset),profile=name,
                        condition=condition,parent_frozen_identity=store.identity,case_ids=[r['id'] for r in records]))
                    from .experiments import run_stage
                    for stage in a.stages or ['E0','E1','E2','E4']:
                        for rec in data.inventory(dataset)['cases']: run_stage(stressed,stage,data.load_case(dataset,rec,stressed.config),True); stressed.index()
                    evaluate(stressed,'test'); assess(stressed.root,dataset,'test',[r['id'] for r in records])
                return
            evaluate(store,'test'); result=assess(run,a.dataset,'test',[r['id'] for r in records])
            print(json.dumps(result,indent=2))
            if a.phase=='promote':
                if not result['approved']: raise RuntimeError('Promotion gates failed/pending; no approved pipeline exported')
                output=a.root.parent/(a.root.name+'-approved')
                bundle(store,output,'pipeline')
                write(output/'approval.json',dict(approved=True,evidence=result,finalist=selected))
                hashes=read(output/'SHA256SUMS.json'); hashes['approval.json']=digest(output/'approval.json')
                write(output/'SHA256SUMS.json',hashes)


if __name__=='__main__': main()
