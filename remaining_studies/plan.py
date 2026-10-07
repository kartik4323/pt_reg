"""Deterministic task DAG and OFAT condition construction."""
import copy
import re
import statistics
from pathlib import Path
from generative_assembly import config, data, storage
from .stages import corruption_factors


def suite_hash():
    return storage.fingerprint({p.name: storage.digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def make_plan(base_path, dataset, studies, gpus, cpu_workers=2, shards=None, excluded=(), sources=(), allow_shared_gpu=False):
    if cpu_workers < 1 or len(gpus) != len(set(gpus)) or set(gpus) & set(excluded):
        raise ValueError('Positive CPU workers and unique, non-excluded GPU IDs required')
    cfg = config.load(base_path)
    cfg['suite_allow_shared_gpu'] = bool(allow_shared_gpu)
    doc = data.inventory(dataset)
    identities, observations = {}, {}
    for case in doc['cases']:
        identity=case.get('source_hash',case['source_id'])
        if identity in identities and identities[identity] != case['source_id']:
            raise ValueError('Duplicate original geometry uses different source IDs; consolidate source aliases')
        identities[identity]=case['source_id']
        if case['sha256'] in observations and observations[case['sha256']] != case['split']:
            raise ValueError('Identical observations cross source splits')
        observations[case['sha256']]=case['split']
    if not cfg['smoke']:
        cfg.update(image_models=['sd15', 'sd15_depth'], input_types=['F', 'A'],
                   image_seeds=[11, 23, 37, 51], reconstruct_all_for_E5=True,
                   n_render_views=1, n_instantmesh_views=1, points=512)
        cfg['solver'].update(candidates=4096, patches=64, alignment_starts=32,
                             alignment_iterations=20, refine_evaluations=100)
        cfg['training'].update(updates=2000, seeds=[101, 202, 303])
        cfg['images']['device'] = 'cuda'
        cfg['training']['device'] = 'cuda'
        cfg['robustness'].update(noise=[0,.0025,.005,.01], dropout=[0,.25,.5],
                                 erosion_fraction=.1, repeats=3, missing_piece=True)
        cfg['evaluation']['bootstrap'] = 2000
    cfg.update(suite_code_sha256=suite_hash(), suite_selector='exterior')
    if cfg['primary_model'] not in cfg['image_models']:
        cfg['primary_model'] = 'sd15'
    if cfg['primary_input'] not in cfg['input_types']:
        cfg['primary_input'] = 'A'
    splits = {s: sorted({c['source_id'] for c in doc['cases'] if c['split'] == s})
              for s in ('train', 'dev', 'test', 'real')}
    if not splits['dev']:
        raise ValueError('No development sources in dataset')
    count = shards or max(1, len(gpus))
    if count < 1:
        raise ValueError('Shards must be positive')
    tasks = []
    def add(condition, kind, mode, split='dev', dependencies=(), conf=None, gate=False, shard_list=None):
        ids = []
        pool = splits[split]
        partitions = shard_list if shard_list is not None else [pool[i::count] for i in range(min(count, max(1, len(pool))))]
        for index, subset in enumerate(partitions):
            if not subset:
                continue
            tid = f'{condition}-{mode}-{index:03d}'
            deps = list(dependencies(index) if callable(dependencies) else dependencies)
            tasks.append({'id': tid, 'condition': condition, 'kind': kind, 'mode': mode,
                          'split': split, 'sources': subset, 'depends_on': deps,
                          'gate_required': gate, 'config': copy.deepcopy(conf or cfg)})
            ids.append(tid)
        return ids
    neural_needed = bool(set(studies) & {'E4', 'E5', 'E6'})
    teacher = add('baseline', 'gpu', 'generate') if neural_needed else []
    devbase = add('baseline', 'cpu', 'assembly_gate', dependencies=lambda i: [teacher[i]]) if neural_needed else []
    if 'E4' in studies:
        variants = [('weight', v) for v in (0,.03,.3,1.) if v != cfg['solver']['template_weight']]
        variants += [('refine', v) for v in (0,25,100) if v != cfg['solver']['refine_evaluations']]
        variants += [('align', v) for v in (8,32) if v != cfg['solver']['alignment_starts']]
        if cfg['smoke']:
            variants = [('weight', 0), ('refine', 0), ('align', 2)]
        keys = {'weight': 'template_weight', 'refine': 'refine_evaluations', 'align': 'alignment_starts'}
        for factor, value in variants:
            condition = f'e4_{factor}_{value}'
            variant = copy.deepcopy(cfg)
            variant['solver'][keys[factor]] = value
            add(condition, 'gpu' if factor == 'align' else 'cpu', 'assembly', conf=variant,
                dependencies=lambda i: [teacher[i]])
        add('baseline', 'exclusive', 'compute', dependencies=devbase)
    if 'E5' in studies:
        contact_cfg = dict(copy.deepcopy(cfg), suite_selector='exterior_contact')
        add('e5_exterior_contact', 'cpu', 'gate', conf=contact_cfg,
            dependencies=lambda i: [devbase[i]])
    trainteachers = []
    students = []
    if 'E6' in studies and splits['train']:
        trainteachers = add('training_teacher', 'gpu', 'train_teacher', 'train', dependencies=devbase, gate=True)
        for seed in cfg['training']['seeds']:
            student = copy.deepcopy(cfg)
            student['training']['seeds'] = [seed]
            students += add('students', 'gpu', f'train_seed_{seed}', dependencies=devbase+trainteachers,
                            conf=student, gate=True, shard_list=[splits['dev']+splits['train']])
    if 'E7' in studies:
        robust = copy.deepcopy(cfg)
        robust.update(image_models=[cfg['primary_model']], input_types=[cfg['primary_input']], oracles=False)
        points = [cfg['points']] if cfg['smoke'] else [512,256,2048]
        for budget in points:
            variant = copy.deepcopy(robust)
            variant.update(points=budget, suite_density_only=budget != cfg['points'])
            condition = f'e7_points_{budget}'
            cpu_ids = add(condition+'_cpu', 'cpu', 'robust_cpu', conf=variant)
            gpu_ids = add(condition, 'gpu', 'robust_generated', conf=variant,
                          dependencies=lambda i: [cpu_ids[i]])
            if students:
                add(condition+'_students', 'cpu', 'robust_students', conf=variant,
                    dependencies=lambda i: [gpu_ids[i]]+students, gate=True)
        if splits['real']:
            add('real_transfer', 'gpu', 'real', 'real', conf=robust)
    # Per-task cost upper bounds: E2 may reject inputs, so actual use can be lower.
    for task in tasks:
        cases = [c for c in doc['cases'] if c['source_id'] in task['sources'] and c['split'] == task['split']]
        condition_cfg = task['config']
        n = len(cases)
        factor_count = len(corruption_factors(condition_cfg))
        instances = sum(sum(kind != 'missing' or c['pieces'] >= 3 for kind, value in corruption_factors(condition_cfg))
                        for c in cases)*condition_cfg['robustness']['repeats'] if task['mode'].startswith('robust') else n
        generating = task['mode'] in ('generate', 'train_teacher', 'robust_generated', 'real') or (
            task['mode'] == 'assembly' and task['kind'] == 'gpu')
        models, inputs, seeds = len(condition_cfg['image_models']), len(condition_cfg['input_types']), len(condition_cfg['image_seeds'])
        edited = instances*models*inputs*seeds if generating else 0
        raw = instances*inputs if generating else 0
        oracle = instances if generating and condition_cfg['oracles'] and task['split'] != 'train' and not task['mode'].startswith('robust') else 0
        task['counts'] = {'cases': n, 'sources': len({c['source_id'] for c in cases}),
                          'instances': instances, 'edited_images_upper_bound': edited,
                          'reconstructions_upper_bound': edited+raw+oracle}
    unavailable = []
    categories = sorted({c['category'] for c in doc['cases']})
    if len(categories) < 2:
        unavailable.append('Wrong-category distractors unavailable: only one category')
    for pieces in (3,5):
        if not any(c['split'] == 'dev' and c['pieces'] == pieces for c in doc['cases']):
            unavailable.append(f'No development {pieces}-piece patterns')
    if 'E6' in studies and not splits['train']:
        unavailable.append('E6 unavailable: no training sources')
    if not splits['real']:
        unavailable.append('Real-scan transfer unavailable: no real records')
    roots = []
    for value in sources:
        path = Path(value).expanduser().resolve()
        if (path/'study.json').exists():
            roots.append(str(path))
        elif path.exists():
            roots += [str(p.parent) for p in path.rglob('study.json')]
        else:
            unavailable.append(f'Source run not currently available: {path}')
    missing_locks = []
    if not cfg['smoke']:
        repos = {cfg['images'][m] for m in cfg['image_models']}
        if 'sd15_depth' in cfg['image_models']:
            repos.add(cfg['images']['controlnet'])
        pinned=lambda value: bool(re.fullmatch(r'[0-9a-fA-F]{40}',str(value or '')))
        missing_locks += [repo for repo in sorted(repos) if not pinned(cfg['images']['revisions'].get(repo))]
        if cfg['reconstruction']['backend'] == 'instantmesh':
            missing_locks += [repo for repo in ('sudo-ai/zero123plus-v1.2','TencentARC/InstantMesh')
                              if not pinned(cfg['reconstruction']['revisions'].get(repo))]
            if not pinned(cfg['reconstruction'].get('commit')):
                missing_locks.append('InstantMesh code commit')
    timings={'E1':[],'E2':[]}
    for value in roots:
        path=Path(value)
        source=storage.read(path/'study.json')['identity']['config']
        image_match=({k:v for k,v in source['images'].items() if k!='python'} ==
                     {k:v for k,v in cfg['images'].items() if k!='python'} and source['pixels']==cfg['pixels'])
        recon_match=({k:v for k,v in source['reconstruction'].items() if k not in ('python','repo','timeout_seconds')} ==
                     {k:v for k,v in cfg['reconstruction'].items() if k not in ('python','repo','timeout_seconds')})
        for record in (path/'jobs').glob('*/result.json'):
            row=storage.read(record)
            if row.get('status')!='complete' or bool(row.get('smoke')) != cfg['smoke'] or not row.get('seconds'):
                continue
            if (record.parent/'running.lock').exists():
                continue
            if row['stage']=='E1' and image_match and row.get('output',{}).get('model') in cfg['image_models']:
                timings['E1'].append(row['seconds'])
            if row['stage']=='E2' and recon_match:
                timings['E2'].append(row['seconds'])
    estimates={'basis':'historical compatible worker jobs; work estimate, not parallel wall time',
               'hardware_compatibility':'must be checked on the target server',
               'cache_reuse':'upper bound assumes all requested neural assets need generation'}
    for stage,key in [('E1','edited_images_upper_bound'),('E2','reconstructions_upper_bound')]:
        estimates[stage]={'measured_jobs':len(timings[stage]),'median_seconds':statistics.median(timings[stage]) if len(timings[stage])>=3 else None}
        estimates[stage]['estimated_work_hours']=sum(t['counts'][key] for t in tasks)*estimates[stage]['median_seconds']/3600 if estimates[stage]['median_seconds'] is not None else None
    return {'schema_version': 1, 'base_config': cfg, 'dataset': str(Path(dataset).resolve()),
            'dataset_sha256': storage.digest(dataset), 'engine_code_sha256': storage.code_hash(),
            'suite_code_sha256': suite_hash(), 'studies': sorted(studies), 'gpu_ids': list(gpus),
            'excluded_gpus': list(excluded), 'allow_shared_gpu': bool(allow_shared_gpu),
            'cpu_workers': cpu_workers, 'source_runs': sorted(set(roots)),
            'source_counts': {s: len(v) for s,v in splits.items()}, 'categories': categories,
            'unavailable': unavailable, 'tasks': tasks,
            'model_locks_ready': not missing_locks, 'missing_model_locks': missing_locks,
            'runtime_estimate': 'unknown until representative real job measurements exist',
            'measured_runtime_estimates': estimates,
            'development_only': True, 'smoke': cfg['smoke']}
