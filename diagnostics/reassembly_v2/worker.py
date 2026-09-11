"""Ordered diagnosis jobs using explicitly selected, immutable checkpoints."""
import copy
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import OrderedDict, Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from reassembly import solver
from reassembly.checkpoints import load_checkpoint
from reassembly.data import FractureDataset, verify_manifest, make_oracle_field
from reassembly.evaluation import assembly_metrics, aggregate
from reassembly.losses import correspondence_loss, segmentation_loss
from reassembly.model import ReassemblyModel
from reassembly.resources import seed_all
from reassembly.geometry import so3_exp
from reassembly.training import collate_samples
from reassembly.validation import run_correctness_checks

from . import probes
from .runtime import Limits, compare, job_id, read, sha256, write
from .variants import variant, choose_subset, check_adapter
from .contracts import sample_contract, matching_contract


REPO=Path(__file__).resolve().parents[2]
SEEDS=(4101,4102,4103)


def inventory(work, output, limits, device):
    limits.check()
    manifest=work/'prepared/manifest.json'
    integrity=verify_manifest(manifest)
    write(output/'prepared_manifest.json',read(manifest))
    if integrity['dataset_fingerprint']!='b7f2b84a20964276c894300a7ece4d8f97f232903bace0f14131535594462043':
        raise RuntimeError('This diagnosis is bound to the recorded VM dataset fingerprint; do not substitute local prepared data')
    entries={}; checkpoints={}; curves={}
    state={'dataset_integrity':integrity,'files':entries,'checkpoints':checkpoints,
           'production_files':{f.name:{'path':str(f),'sha256':sha256(f)} for f in (REPO/'reassembly').glob('*.py')},
           'diagnostic_files':{f.name:{'path':str(f),'sha256':sha256(f)} for f in Path(__file__).parent.glob('*.py')}}
    write(output/'inventory.json',state)
    for run in ('overfit','pilot/predicted','pilot/contact_only'):
        for stage in (1,2,3):
            directory=work/run/f's{stage}'
            for name in ('config.resolved.json','run.json','training_report.json','history.jsonl','best.pt','latest.pt'):
                file=directory/name; relative=file.relative_to(work).as_posix()
                if not file.is_file(): raise FileNotFoundError(f'Required diagnostic input missing: {file}')
                entry={'path':str(file),'sha256':sha256(file),'bytes':file.stat().st_size}
                entries[relative]=entry
                if name.endswith('.pt'):
                    state=load_checkpoint(file,dataset_fingerprint=integrity['dataset_fingerprint'],stage=stage,purpose='overfit' if run=='overfit' else 'pilot')
                    report=read(directory/'training_report.json')
                    if report.get('run_id')!=state.get('run_id') or report.get('dataset_fingerprint')!=state['dataset_fingerprint']:
                        raise RuntimeError(f'Checkpoint and training report lineage mismatch: {relative}')
                    expected_condition='contact_only' if run.endswith('contact_only') else 'predicted'
                    if state['cfg']['train'].get('condition','predicted')!=expected_condition:
                        raise RuntimeError(f'Checkpoint condition mismatch: {relative}')
                    checkpoints[relative]={**entry,**{k:state.get(k) for k in ('architecture','stage','step','purpose','run_id','training_lineage','model_fingerprint','dataset_fingerprint')},
                        'config':state['cfg'],'random_state_keys':sorted(state['random_state']),
                        'optimizer_state_present':bool(state['optimizer']),'scaler_state':state.get('scaler')}
                    # Keep hashes available even if a later input is missing.
                    partial=read(output/'inventory.json'); partial.update(files=entries,checkpoints=checkpoints)
                    write(output/'inventory.json',partial)
            curves[f'{run}/s{stage}']=[]
            for line in (directory/'history.jsonl').read_text().splitlines():
                r=json.loads(line); metric=r.get('validation',{})
                if metric:
                    curves[f'{run}/s{stage}'].append({'step':r['update'],'validation':metric.get('sdf_l1') if stage==2 else metric['loss']})
            limits.check()
    sources=list((REPO/'reassembly').glob('*.py'))
    version=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    reports=[work/'overfit/eval/evaluation.json']+[work/f'eval/{s}/{c}/evaluation.json' for s in ('test','cut_holdout') for c in ('contact_only','predicted','gt','perturbed')]
    for file in [manifest,work/'preflight/preflight.json',*reports]:
        if not file.is_file(): raise FileNotFoundError(file)
        entries[file.relative_to(work).as_posix()]={'path':str(file),'sha256':sha256(file),'bytes':file.stat().st_size}
    previous=read(output/'inventory.json')
    state={'dataset_integrity':integrity,'files':entries,'checkpoints':checkpoints,
           'diagnostic_files':previous['diagnostic_files'],
           'production_files':{f.name:{'path':str(f),'sha256':sha256(f)} for f in sources},
           'git_revision':version,'python':platform.python_version(),'platform':platform.platform(),
           'tracked_git_status':subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True),
           'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
           'libraries':{name:importlib.metadata.version(name) for name in ('torch','numpy','trimesh','scipy','manifold3d')},
           'precision':'FP32 inference; no AMP overrides','seed':42,'variant_seeds':SEEDS,
           'torch_runtime':{'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
               'matmul_precision':torch.get_float32_matmul_precision(),'cuda_tf32':torch.backends.cuda.matmul.allow_tf32,
               'cudnn_tf32':torch.backends.cudnn.allow_tf32,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},
           'note':'Artifact paths are explicit. No weights, configuration, or architecture are migrated.'}
    write(output/'inventory.json',state); write(output/'training_curves.json',curves)
    correctness=run_correctness_checks(); write(output/'correctness.json',correctness)
    if not correctness['passed']: raise RuntimeError('Production geometry/solver correctness checks failed')
    return state


def build_jobs(work, cfg):
    datasets={name:FractureDataset(work/'prepared/manifest.json','train' if name=='overfit' else name,cfg,fixed=True,limit=16 if name=='overfit' else None)
              for name in ('overfit','train','val','test','cut_holdout')}
    subset=choose_subset(datasets['val'].records)
    if len(subset)!=12 or len(datasets['overfit'])!=16: raise RuntimeError('Expected 12 validation diagnostics and 16 overfit patterns')
    expected_counts={'train':418,'val':48,'test':69,'cut_holdout':11}
    if any(len(datasets[name])!=count for name,count in expected_counts.items()):
        raise RuntimeError(f'Dataset coverage differs from recorded VM run: { {k:len(v) for k,v in datasets.items()} }')
    jobs=[]
    def add(kind,cohort,checkpoint,indices,**extra):
        for index in indices:
            j=dict(kind=kind,cohort=cohort,checkpoint=checkpoint,index=index,**extra); j['id']=job_id(j); jobs.append(j)
    for cohort,condition in [('overfit','predicted')]+[(s,c) for s in ('test','cut_holdout') for c in ('contact_only','predicted','gt','perturbed')]:
        checkpoint='overfit/s3/best.pt' if cohort=='overfit' else f"pilot/{'contact_only' if condition=='contact_only' else 'predicted'}/s3/best.pt"
        add('replay',cohort,checkpoint,range(len(datasets[cohort])),condition=condition)
    for condition in ('predicted','contact_only'):
        for cohort in ('train','val'):
            add('baseline',cohort,f'pilot/{condition}/s3/best.pt',range(len(datasets[cohort])),condition=condition)
    for run in ('overfit','pilot/predicted','pilot/contact_only'):
        for stage in (1,2,3):
            for name in ('best','latest'):
                add('measurement','val',f'{run}/s{stage}/{name}.pt',subset,stage=stage,
                    condition='contact_only' if run.endswith('contact_only') or stage<3 else 'predicted')
    for cohort,checkpoint,ids in [('overfit','overfit/s3/best.pt',range(16)),('val','pilot/predicted/s3/best.pt',subset)]:
        add('sensitivity',cohort,checkpoint,ids,variant='unchanged',seed=0)
        for mode in ('permutation','rotation','translation','samples','samples_poses'):
            for seed in SEEDS: add('sensitivity',cohort,checkpoint,ids,variant=mode,seed=seed)
    operations=['baseline','oracle_selection','oracle_gating','oracle_selection_gating','oracle_matches','oracle_selection_matches',
                'fixed_matches_no_field','fixed_matches_predicted_field','fixed_matches_gt_field',
                'conditioning_predicted','conditioning_gt','conditioning_perturbed','conditioning_disabled']
    for operation in operations:
        add('intervention','val','pilot/predicted/s3/best.pt',subset,operation=operation)
    for weight in (0,1e-6,1e-5,1e-4,1e-3):
        for mass in (0,.005,.05):
            for shuffled in (False,True):
                add('threshold','val','pilot/predicted/s3/best.pt',subset,weight=weight,mass=mass,shuffled=shuffled)
    for kind in ('field','pose_field'):
        for field in ('predicted','gt'):
            add(kind,'val','pilot/predicted/s3/best.pt',subset,field=field)
    for run in ('overfit','pilot/predicted','pilot/contact_only'):
        for stage in (1,2,3): add('gradient','val',f'{run}/s{stage}/best.pt',subset[:3],stage=stage)
    return {'version':1,'jobs':jobs,'variant_seeds':SEEDS,'validation_subset':[datasets['val'].records[i] for i in subset],
            'cohorts':{k:[r['pattern_id'] for r in ds.records] for k,ds in datasets.items()},
            'thresholds_are_diagnostic_only':True,'batch_size':1,'atol':1e-5,'rtol':1e-4},datasets


class Engine:
    def __init__(self,work,output,limits,device,datasets):
        self.work,self.output,self.limits,self.device,self.datasets=work,output,limits,device,datasets
        self.models=OrderedDict(); self.fields=OrderedDict(); self.metadata=OrderedDict(); self.median=None
    def model(self,name):
        if name in self.models:
            self.models.move_to_end(name); return self.models[name]
        self.models.clear()
        if self.device.type=='cuda': torch.cuda.empty_cache()
        state=load_checkpoint(self.work/name,dataset_fingerprint=self.datasets['val'].fingerprint)
        model=ReassemblyModel(state['cfg']).to(self.device); model.load_state_dict(state['model']); model.eval()
        seed_all(state['cfg']['train']['seed'])
        self.models[name]=(model,state['cfg']); return model,state['cfg']
    def field(self,model,encoded,cfg,sample,kind):
        key=(sample['pattern_id'],kind)
        if key not in self.fields:
            self.limits.check()
            with torch.no_grad(): self.fields[key]=probes.grid(model,encoded,cfg,sample,kind)
            if len(self.fields)>40: self.fields.popitem(last=False)
        return self.fields[key]
    def evaluate(self,model,sample,cfg,condition):
        self.last_calibration=[]
        batch=collate_samples([sample],self.device)
        override=make_oracle_field(sample) if condition=='gt' else None
        trace={'pairs':{},'trees':[],'refinements':[]}
        original_solve,original_tree,original_refine=solver.solve_from_matches,solver._compose_tree,solver._refine
        def traced_solve(matches,*args,**kwargs):
            trace['pairs']={f"{p['i']}-{p['j']}":probes.trace_pair(p,cfg) for p in matches}
            return original_solve(matches,*args,**kwargs)
        def traced_tree(edges,*args,**kwargs):
            value=original_tree(edges,*args,**kwargs); trace['trees'].append({'edges':edges,'connected':value is not None}); return value
        def traced_refine(poses,anchor,contacts,points,weights,field,options):
            value,accepted=original_refine(poses,anchor,contacts,points,weights,field,options)
            trace['refinements'].append({'accepted_steps':accepted,
                'before':solver._score(poses,contacts,points,weights,field,options)[1],
                'after':solver._score(value,contacts,points,weights,field,options)[1]})
            return value,accepted
        with patch.object(solver,'solve_from_matches',traced_solve),patch.object(solver,'_compose_tree',traced_tree),patch.object(solver,'_refine',traced_refine):
            result=solver.solve_assembly(model,batch,cfg,condition,field_override=override)
        self.last_trace=trace
        row=assembly_metrics(sample,result,cfg['solver']['success_threshold'])
        row.update(pattern_id=sample['pattern_id'],source_id=sample['source_id'],band=sample['band'],cut_family=sample['cut_family'],
                   pieces=int(sample['fragment_mask'].sum()),reason=result.get('reason'),diagnostics=result['diagnostics'])
        with torch.no_grad():
            encoded=model.encode(batch['points'],batch['fragment_mask'],batch['anchor_index'])
            pairs=model.match(encoded,use_scaffold=condition!='contact_only'); details={}
            loss,positive=correspondence_loss(pairs,batch,cfg['train'].get('contact_radius',.05),
                localization_sigma=cfg['train'].get('contact_sigma',.01),localization_weight=cfg['loss'].get('matching_localization',1),diagnostics=details)
            row.update(predicted_model_matching_loss=float(loss),positive_contact_targets=int(positive),
                fracture_segmentation_loss=float(segmentation_loss(encoded['fracture_logits'],batch['fracture_labels'],batch['fragment_mask'])),
                **{k:float(v) for k,v in details.items()})
            if condition!='contact_only':
                field=model.scaffold(encoded,batch['sdf_queries']); tau=cfg['model']['truncation']
                errors=(field['distance'].clamp(-tau,tau)-batch['sdf_values'].clamp(-tau,tau)).abs()
                row.update(sdf_l1=float(errors.mean()),uncertainty_mae=float((errors-field['log_scale'].exp()).abs().mean()))
                sigma=field['log_scale'].exp()
                for lower,upper in ((0,.005),(.005,.02),(.02,1)):
                    mask=(sigma>=lower)&(sigma<upper)
                    if mask.any(): self.last_calibration.append({'bin':[lower,upper],'count':int(mask.sum()),'error':float(errors[mask].mean()),'sigma':float(sigma[mask].mean())})
        return row,result
    def execute(self,job):
        model,cfg=self.model(job['checkpoint']); ds=self.datasets[job['cohort']]
        sample=variant(ds,job['index'],job.get('variant','unchanged'),job.get('seed',0))
        if hasattr(ds,'root'):
            identity=sample['pattern_id']
            if identity not in self.metadata:
                with np.load(ds.root/ds.records[job['index']]['path'],allow_pickle=False) as archive:
                    self.metadata[identity]=json.loads(str(archive['metadata']))
                if len(self.metadata)>512: self.metadata.popitem(last=False)
            sample['_diagnostic_metadata']=self.metadata[identity]
        row={'job':job,'pattern_id':sample['pattern_id'],'source_id':sample['source_id'],'band':sample['band'],
             'pieces':int(sample['fragment_mask'].sum()),'geometry_metadata':sample.get('_diagnostic_metadata')}
        kind=job['kind']; condition=job.get('condition','predicted')
        if kind=='sensitivity': row['sample_contract']=sample_contract(sample)
        if kind in ('replay','baseline','sensitivity'):
            if kind=='replay':
                path=self.work/('overfit/eval/evaluation.json' if job['cohort']=='overfit' else f"eval/{job['cohort']}/{condition}/evaluation.json")
                report=read(path); state=load_checkpoint(self.work/job['checkpoint'])
                if len(report['samples'])!=len(ds): raise RuntimeError('Replay example count differs from recorded report')
                if Path(report['checkpoint']).resolve()!= (self.work/job['checkpoint']).resolve():
                    raise RuntimeError('Recorded checkpoint path differs; resolve artifact relocation explicitly before replay')
                if report['dataset_fingerprint']!=ds.fingerprint or report['training_lineage']!=state.get('training_lineage',{}):
                    raise RuntimeError('Replay fingerprint or checkpoint lineage mismatch')
                if report['trained_updates']!=state['step']: raise RuntimeError('Replay checkpoint step mismatch')
                if report['config']['data']!=ds.cfg['data'] or report['config']['model']!=cfg['model']:
                    raise RuntimeError('Recorded input sampling/model configuration differs')
                cfg=report['config']
            metrics,result=self.evaluate(model,sample,cfg,condition); row['metrics']=metrics
            row['solver_trace']=self.last_trace
            row['calibration_bins']=self.last_calibration
            if kind=='replay':
                expected=report['samples'][job['index']]
                row['replay_differences']=compare(expected,metrics)
                if job['index']==len(ds)-1:
                    prior=[read(self.output/'results'/f"{job_id({**{k:v for k,v in job.items() if k!='id'},'index':i})}.json") for i in range(job['index'])]
                    rows=[r['metrics'] for r in prior]+[metrics]
                    totals={'summary':aggregate(rows),'by_band':{b:aggregate([r for r in rows if r['band']==b]) for b in ('easy','intermediate','hard')},
                        'by_pieces':{str(n):aggregate([r for r in rows if r['pieces']==n]) for n in (2,3)},
                        'predicted_sdf_l1':float(np.mean([r['sdf_l1'] for r in rows])) if condition!='contact_only' else None,
                        'calibration_bins':[b for r in prior+[row] for b in r['calibration_bins']],
                        'fixed_fit_passed':bool(job['cohort']=='overfit' and condition=='predicted' and len(rows)==16 and all(r['success'] and r['status']=='ok' for r in rows)),
                        'failure_breakdown':{'no_solution':sum(r['failed'] for r in rows),
                            'geometrically_inaccurate_solution':sum(not r['failed'] and not r['success'] for r in rows),
                            'geometrically_correct_low_confidence':sum(r['success'] and r['status']!='ok' for r in rows),
                            'accepted':sum(r['success'] and r['status']=='ok' for r in rows)}}
                    row['replay_differences']+=compare({k:report[k] for k in totals},totals,'aggregate')
            if kind=='baseline': row['measurements']=probes.measure(model,sample,cfg,self.device,condition)[0]
            return row
        if kind=='gradient':
            row['gradients']=probes.gradient_probe(model,sample,cfg,self.device,job['stage']); return row
        batch=collate_samples([sample],self.device)
        with torch.no_grad():
            if kind=='measurement':
                measurements,encoded,pairs=probes.measure(model,sample,cfg,self.device,condition,deep=True)
                row['measurements']=measurements
                if job.get('stage',3)>=2:
                    prediction=model.scaffold(encoded,batch['sdf_queries']); tau=cfg['model']['truncation']
                    error=(prediction['distance']-batch['sdf_values'].clamp(-tau,tau)).abs()[0]
                    row['reconstruction']={'sdf_l1':float(error.mean()),'uncertainty_mae':float((error-prediction['log_scale'][0].exp()).abs().mean()),
                        'near_l1':float(error[batch['sdf_near_mask'][0]].mean()),'surrounding_l1':float(error[~batch['sdf_near_mask'][0]].mean())}
                if job['checkpoint']=='pilot/predicted/s3/best.pt': self.save_tensors(job,sample,encoded,pairs)
                return row
            encoded,pairs=probes.predictions(model,batch)
            matches=probes.numpy_matches(pairs)
            if kind=='threshold':
                changed=copy.deepcopy(cfg); changed['solver'].update(min_correspondence_weight=job['weight'],min_pair_mass=job['mass'])
                if job['shuffled']:
                    rng=np.random.default_rng(4101)
                    matches=copy.deepcopy(matches)
                    for match in matches:
                        order=rng.permutation(len(match['target_xyz']))
                        match['weights']=match['weights'][:,order]
                        match['target_matchability']=match['target_matchability'][order]
                row['metrics']=probes.solve_matches(matches,sample,changed,encoded)
                traces={f"{m['i']}-{m['j']}":probes.trace_pair(m,changed) for m in matches}
                false_edges=[]
                for match in matches:
                    a=set(sample['interface_ids'][match['i']].tolist())-{-1}; b=set(sample['interface_ids'][match['j']].tolist())-{-1}
                    qa=sample.get('_diagnostic_metadata',{}).get('qa',{})
                    true_contact=bool(qa['adjacency'][match['i']][match['j']]) if 'adjacency' in qa else bool(a&b)
                    if not true_contact and traces[f"{match['i']}-{match['j']}"]['first_failure'] is None: false_edges.append([match['i'],match['j']])
                row.update(filter_traces=traces,false_contact_edges=false_edges,
                    false_contact_definition='Non-contacting mesh pairs with at least one valid pose candidate; these edges need not be in the winning assembly.'); return row
            if kind=='intervention':
                operation=job['operation']; field=None
                if operation.startswith('oracle_'):
                    selection='oracle' if 'selection' in operation else 'predicted'
                    gating='oracle' if 'gating' in operation else 'predicted'
                    _,altered=probes.predictions(model,batch,selection=selection,gating=gating,encoded=encoded)
                    matches=probes.oracle_matches(altered,batch,cfg,encoded) if 'matches' in operation else probes.numpy_matches(altered)
                    row['selection_and_learned_probabilities']=probes.measure(model,sample,cfg,self.device,selection=selection,gating=gating,encoded=encoded)[0]
                if operation.startswith('conditioning_'):
                    which=operation.removeprefix('conditioning_')
                    if which in ('gt','perturbed'):
                        field=self.field(model,encoded,cfg,sample,which)
                        altered=model.match(encoded,prior_override=probes.override_prior(model,encoded,field))
                    else: altered=model.match(encoded,use_scaffold=which!='disabled')
                    row['probability_changes']={f"{a['i']}-{a['j']}":{key:probes.distribution(a[key]-b[key]) for key in ('source_prob','target_prob','weights')} for a,b in zip(altered,pairs) if bool(a['valid'][0])}
                    self.limits.check(); arrays={}
                    for a,b in zip(altered,pairs):
                        if not bool(a['valid'][0]): continue
                        for key in ('source_prob','target_prob','weights'):
                            arrays[f"{a['i']}_{a['j']}_{key}"]=probes.array(a[key])
                            arrays[f"{a['i']}_{a['j']}_{key}_delta"]=probes.array(a[key]-b[key])
                    directory=self.output/'tensors'; directory.mkdir(exist_ok=True)
                    relative=f"tensors/{job['id']}_conditioning.npz"
                    np.savez_compressed(self.output/relative,**arrays); row['probability_tensor_file']=relative
                    matches=probes.numpy_matches(altered)
                    # Conditioning intervention holds solver field fixed.
                    field=None
                if operation in ('fixed_matches_predicted_field','fixed_matches_gt_field'):
                    field=self.field(model,encoded,cfg,sample,'gt' if '_gt_' in operation else 'predicted')
                if operation.startswith('fixed_matches_'):
                    row['metrics'],row['fixed_initializations']=probes.refinement_control(matches,sample,cfg,encoded,field)
                else: row['metrics']=probes.solve_matches(matches,sample,cfg,encoded,field)
                row['filter_traces']={f"{m['i']}-{m['j']}":probes.trace_pair(m,cfg) for m in matches}
                row['diagnostic_only']=True; return row
            if kind=='field':
                row['field_metrics']=self.field_metrics(model,encoded,sample,batch,cfg,job['field'])
                return row
            if kind=='pose_field':
                row['pose_field']=self.pose_field(model,encoded,sample,cfg,job['field']); return row
        raise ValueError(f'Unknown job {kind}')
    def save_tensors(self,job,sample,encoded,pairs):
        self.limits.check()
        arrays={k:probes.array(encoded[k]) for k in ('point_xyz','descriptor','point_features','fracture_logits')}
        arrays.update(canonical_points=probes.array(sample['canonical_points']),interface_ids=probes.array(sample['interface_ids']))
        for p in pairs:
            if bool(p['valid'][0]):
                for key in ('source_indices','target_indices','source_prob','target_prob','weights'):
                    arrays[f"pair_{p['i']}_{p['j']}_{key}"]=probes.array(p[key])
        directory=self.output/'tensors'; directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/f"{job['id']}.npz",**arrays)
    def field_metrics(self,model,encoded,sample,batch,cfg,kind):
        field=self.field(model,encoded,cfg,sample,kind)
        query=probes.array(sample['sdf_queries']); target=probes.array(sample['sdf_values']).clip(-cfg['model']['truncation'],cfg['model']['truncation'])
        direct=model.scaffold(encoded,batch['sdf_queries']); learned=probes.array(direct['distance'])[0]
        sigma=np.exp(probes.array(direct['log_scale'])[0]); interpolated=field.sample(query)[0]
        exact=make_oracle_field(sample)(query).clip(-cfg['model']['truncation'],cfg['model']['truncation'])
        if self.median is None:
            values=[]
            for i in range(len(self.datasets['train'])):
                self.limits.check(); values.append(probes.array(self.datasets['train'][i]['sdf_values']).clip(-cfg['model']['truncation'],cfg['model']['truncation']))
            self.median=float(np.median(np.concatenate(values)))
        masks={'all':np.ones(len(target),bool),'near':probes.array(sample['sdf_near_mask']).astype(bool),'surrounding':~probes.array(sample['sdf_near_mask']).astype(bool),
               'inside':target<0,'outside':target>0}
        groups={}
        for name,mask in masks.items():
            groups[name]={'queries':int(mask.sum()),'learned_l1':float(np.abs(learned[mask]-target[mask]).mean()) if mask.any() else None,
                'zero_l1':float(np.abs(target[mask]).mean()) if mask.any() else None,
                'training_median_l1':float(np.abs(self.median-target[mask]).mean()) if mask.any() else None,
                'grid_l1':float(np.abs(interpolated[mask]-target[mask]).mean()) if mask.any() else None}
        nonzero=np.abs(target)>1e-6
        error=np.abs(learned-target)
        bins=[]
        for lo,hi in ((0,.005),(.005,.02),(.02,.05),(.05,1)):
            mask=(sigma>=lo)&(sigma<hi)
            bins.append({'bounds':[lo,hi],'count':int(mask.sum()),'mean_sigma':float(sigma[mask].mean()) if mask.any() else None,'mean_error':float(error[mask].mean()) if mask.any() else None})
        self.save_field_visual(sample,field,kind)
        return {'groups':groups,'training_median':self.median,'sign_error_rate':float(((learned[nonzero]>0)!=(target[nonzero]>0)).mean()) if nonzero.any() else None,
                'uncertainty_mae':float(np.abs(error-sigma).mean()),'calibration':bins,
                'grid_vs_direct_l1':float(np.abs(interpolated-(exact if kind=='gt' else learned)).mean()),
                'source_exact_vs_saved_target_l1':float(np.abs(exact-target).mean()),
                'outside_grid_queries':int(((query<field.bounds[0])|(query>field.bounds[1])).any(-1).sum())}
    def save_field_visual(self,sample,field,kind):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        directory=self.output/'visuals'; directory.mkdir(exist_ok=True)
        name=sample['pattern_id']+'_'+kind
        np.savez_compressed(directory/(name+'.npz'),**field.as_dict())
        fig=plt.figure(figsize=(11,3.5)); axes=[fig.add_subplot(131),fig.add_subplot(132),fig.add_subplot(133,projection='3d')]
        values=field.distance[:,:,field.distance.shape[2]//2]
        axes[0].imshow(values.T,origin='lower',extent=[field.bounds[0,0],field.bounds[1,0],field.bounds[0,1],field.bounds[1,1]],cmap='coolwarm',vmin=-field.truncation,vmax=field.truncation)
        axes[0].set_title(kind+' SDF central slice')
        points=probes.array(sample['target_points']); axes[1].scatter(points[:,0],points[:,1],s=1,alpha=.3)
        axes[1].set_title('Intact target: XY projection'); axes[1].axis('equal')
        sign=field.distance>=0; boundary=np.zeros_like(sign)
        for axis in range(3):
            left=[slice(None)]*3; right=left.copy(); left[axis]=slice(None,-1); right[axis]=slice(1,None)
            changed=sign[tuple(left)]!=sign[tuple(right)]; boundary[tuple(left)]|=changed; boundary[tuple(right)]|=changed
        nodes=np.argwhere(boundary)
        if len(nodes):
            nodes=nodes[np.linspace(0,len(nodes)-1,min(6000,len(nodes)),dtype=int)]
            xyz=field.bounds[0]+nodes/(np.array(field.distance.shape)-1)*(field.bounds[1]-field.bounds[0])
            axes[2].scatter(*xyz.T,s=2,alpha=.25)
        axes[2].set_title('Coarse zero-crossing grid nodes' if len(nodes) else 'No grid sign crossings')
        fig.tight_layout(); fig.savefig(directory/(name+'.png'),dpi=120); plt.close(fig)
    def pose_field(self,model,encoded,sample,cfg,kind):
        field=self.field(model,encoded,cfg,sample,kind); count=int(sample['fragment_mask'].sum()); anchor=int(sample['anchor_index'])
        base_r=probes.array(sample['rotations_gt'])[:count].astype('float64'); base_t=probes.array(sample['translations_gt'])[:count].astype('float64')
        outputs=[]
        for oracle_exterior in (False,True):
            points,weights=probes.exterior(sample,encoded,oracle_exterior)
            for degrees,translation in ((0,0),(5,.02),(15,.05)):
                rng=np.random.default_rng(4101); r=base_r.copy(); t=base_t.copy()
                for i in range(count):
                    if i==anchor: continue
                    axis=rng.normal(size=3); axis/=np.linalg.norm(axis)
                    update=so3_exp(axis*np.deg2rad(degrees)); offset=rng.normal(size=3); offset=offset/np.linalg.norm(offset)*translation
                    r[i]=update@r[i]; t[i]=update@t[i]+offset
                def metric(rotations,translations):
                    return assembly_metrics(sample,{'rotations':rotations,'translations':translations,'status':'diagnostic_pose','confidence':0},cfg['solver']['success_threshold'])
                before=metric(r,t)
                # Field-only probe: no learned or oracle contacts can hide the field's direction.
                poses,accepted=solver._refine((r,t),anchor,[],points,weights,field,cfg['solver'])
                outputs.append({'oracle_exterior':oracle_exterior,'degrees':degrees,'translation':translation,'before':before,
                                'after':metric(*poses),'accepted_steps':accepted,'note':'Field-only pose probe; not production assembly.'})
        return outputs


def run(work,output,device,deadline):
    limits=Limits(work.parent,output,deadline)
    torch.set_num_threads(4)
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('Requested CUDA is unavailable; no silent CPU fallback')
        total=torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.,(20*1024**3-64*1024**2)/total),device)
    inv=inventory(work,output,limits,device)
    cfg=read(work/'eval/test/predicted/evaluation.json')['config']
    manifest,datasets=build_jobs(work,cfg)
    write(output/'experiments.json',manifest)
    identity={}
    for cohort,indices in [('overfit',range(16)),('val',choose_subset(datasets['val'].records))]:
        for i in indices:
            limits.check(); identity[f'{cohort}/{i}']={'rebuild':check_adapter(datasets[cohort],i),'sample':sample_contract(datasets[cohort][i])}
    write(output/'adapter_identity.json',identity)
    engine=Engine(work,output,limits,device,datasets)
    # Representative profile uses real checkpoint operations without training.
    start=time.monotonic(); model,configuration=engine.model('pilot/predicted/s3/best.pt')
    sample=datasets['val'][0]
    engine.evaluate(model,sample,configuration,'predicted')
    predicted_seconds=time.monotonic()-start
    identity['matcher_unchanged']=matching_contract(model,sample,configuration,device)
    write(output/'adapter_identity.json',identity)
    limits.check(); start=time.monotonic(); engine.evaluate(model,sample,configuration,'gt'); gt_seconds=time.monotonic()-start
    start=time.monotonic(); probes.gradient_probe(model,sample,configuration,device,2); backward_seconds=time.monotonic()-start
    write(output/'profile.json',{'predicted_seconds':predicted_seconds,'gt_seconds':gt_seconds,'copied_stage2_backward_seconds':backward_seconds,'device':str(device),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
        'geometry_resolution':configuration['data']['points_per_fragment'],'batch_size':1})
    for job in manifest['jobs']:
        limits.check(); path=output/'results'/f"{job['id']}.json"
        if path.exists():
            previous=read(path)
            if previous.get('replay_differences') or previous.get('error'): raise RuntimeError('A previous job failed; inspect it before continuing')
            continue
        write(output/'progress.json',{'job':job,'started_at':time.time()})
        start=time.monotonic()
        try:
            result=engine.execute(job)
            result['seconds']=time.monotonic()-start
            if device.type=='cuda':
                result['peak_reserved_bytes']=torch.cuda.max_memory_reserved(device)
                if result['peak_reserved_bytes']>=20*1024**3: raise RuntimeError('20 GiB reserved memory limit reached')
            write(path,result)
            if result.get('replay_differences'): raise RuntimeError('Replay differs; downstream causal probes blocked. Inspect replay_differences.')
        except Exception as error:
            if not path.exists(): write(path,{'job':job,'error':f'{type(error).__name__}: {error}'})
            raise
        print(f"completed {job['kind']} {job['id']}",flush=True)
    return {'status':'complete'}
