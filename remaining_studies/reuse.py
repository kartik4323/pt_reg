"""Conservative, verified, dependency-complete imports into a new study root."""
from pathlib import Path
import shutil
from generative_assembly import storage, config


def signature(cfg, stage, derived=False):
    cfg = config.merge({}, cfg)
    # Only optimization/gating/training variations can share generated assets.
    common = {k: v for k, v in cfg.items() if k not in (
        'solver', 'training', 'evaluation', 'resources', 'robustness') and not k.startswith('suite_')}
    common['images'] = {k: v for k, v in cfg['images'].items() if k != 'python'}
    common['reconstruction'] = {k: v for k, v in cfg['reconstruction'].items()
                                if k not in ('python', 'repo', 'timeout_seconds')}
    solver = cfg['solver']
    # E0 candidates/render A depend on the proposer, but not final prior weighting.
    common['proposer'] = {k: v for k, v in solver.items() if k not in (
        'template_weight', 'refine_evaluations', 'gate_exterior', 'gate_contact_ratio',
        'alignment_starts', 'alignment_iterations', 'scales') and not k.startswith('compute_control_')}
    if stage in ('E2', 'E4', 'E5', 'E7'):
        common['alignment'] = {k: solver.get(k) for k in ('alignment_starts', 'alignment_iterations', 'scales', 'patches')}
    if stage in ('E4', 'E5', 'E7'):
        common['solver'] = solver
        common['suite_algorithm'] = cfg.get('suite_code_sha256')
    if stage == 'E5':
        common['selector'] = cfg.get('suite_selector', 'exterior')
    if derived or stage == 'E7':
        common['robustness'] = cfg['robustness']
        common['suite_corruption_mode'] = cfg.get('suite_corruption_mode')
    if stage == 'E6_TRAIN':
        # A checkpoint can be evaluated on different point counts; preserve its
        # source training configuration in the imported record and provenance.
        common = {'engine': storage.code_hash(), 'student': 'PointNet-v1',
                  'suite_algorithm': cfg.get('suite_code_sha256')}
    return storage.fingerprint(common)


def import_completed(store, sources, allowed_sources=None, stages=None):
    stages = set(stages or ['E0', 'E1', 'E1_ORACLE', 'E2'])
    destination = store.root
    current = {r['job_id']: r for r in store.jobs() if r['status'] == 'complete'}
    provenance_path = destination / 'reuse.json'
    provenance = storage.read(provenance_path) if provenance_path.exists() else {'imports': [], 'skipped': []}
    records = storage.read(store.dataset)['cases']
    public = {c['id']: c for c in records}
    eligible = set(allowed_sources) if allowed_sources is not None else {c['source_id'] for c in records}
    for source_path in sources:
        source = Path(source_path).expanduser().resolve()
        if source == destination or not (source / 'study.json').is_file():
            continue
        identity = storage.read(source / 'study.json')['identity']
        if identity['code_sha256'] != storage.code_hash():
            provenance['skipped'].append({'source': str(source), 'reason': 'engine_code_mismatch'})
            continue
        source_cfg = identity['config']
        pending = []
        for path in sorted((source / 'jobs').glob('*/result.json')):
            row = storage.read(path)
            if row['stage'] not in stages or row['status'] != 'complete':
                continue
            if row['stage'] != 'E6_TRAIN' and row['source_id'] not in eligible:
                continue
            if (path.parent / 'running.lock').exists():
                provenance['skipped'].append({'source': str(source), 'job_id': row['job_id'], 'reason': 'active_or_stale_lock'})
                continue
            rec = public.get(row['case_id'])
            derived = rec is None and row['stage'] != 'E6_TRAIN'
            if derived:
                matches = [c for c in records if c['source_id'] == row['source_id'] and
                           row['case_id'].startswith(c['id'] + '__') and c['sha256'] == row.get('input_sha256')]
                if len(matches) != 1:
                    continue
                rec = matches[0]
            if rec and (rec['sha256'] != row.get('input_sha256') or rec['split'] != row['split']):
                provenance['skipped'].append({'source': str(source), 'job_id': row['job_id'], 'reason': 'input_or_split_mismatch'})
                continue
            if row.get('oracle') and (row['split'] == 'train' or not store.config['oracles']):
                continue
            if bool(row.get('smoke')) != bool(store.config['smoke']):
                raise ValueError('Cannot mix smoke and research artifacts')
            if signature(source_cfg, row['stage'], derived) != signature(store.config, row['stage'], derived):
                provenance['skipped'].append({'source': str(source), 'job_id': row['job_id'], 'reason': 'stage_config_mismatch'})
                continue
            storage.verify_job(source, row)
            pending.append(row)
        while pending:
            progress = False
            for row in pending[:]:
                if not all(p in current for p in row.get('parents', [])):
                    continue
                if not row.get('oracle') and any(current[p].get('oracle') for p in row.get('parents', [])):
                    raise ValueError('Oracle ancestry cannot enter a public import')
                jid = row['job_id']
                if jid in current:
                    storage.verify_job(destination, current[jid])
                    if current[jid].get('artifacts') != row.get('artifacts'):
                        raise ValueError(f'Conflicting imported artifacts for {jid}')
                else:
                    target = destination / 'jobs' / jid
                    if target.exists():
                        raise ValueError(f'Import refuses to replace an existing job: {target}')
                    temporary = destination / 'jobs' / (jid + '.importing')
                    temporary.parent.mkdir(parents=True, exist_ok=True)
                    if temporary.exists():
                        raise ValueError(f'Interrupted import requires inspection: {temporary}')
                    # Copy only declared outputs and the immutable completed record.
                    temporary.mkdir()
                    for asset in row.get('artifacts', []):
                        original = storage.checked(source, asset['path'])
                        relative = original.relative_to(source / 'jobs' / jid)
                        output = storage.checked(temporary, relative)
                        output.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(original, output)
                    shutil.copy2(source / 'jobs' / jid / 'result.json', temporary / 'result.json')
                    temporary.rename(target)
                    storage.verify_job(destination, row)
                    current[jid] = row
                    provenance['imports'].append({'source': str(source), 'job_id': jid,
                                                   'source_identity': storage.fingerprint(identity)})
                pending.remove(row)
                progress = True
            if not progress:
                provenance['skipped'].extend({'source': str(source), 'job_id': r['job_id'], 'reason': 'incompatible_or_missing_parent'} for r in pending)
                break
    storage.write(provenance_path, provenance)
    return provenance


def import_students(store, sources):
    """Import a checkpoint with verified source ancestry, without relabelling its teacher jobs."""
    imported = []
    for value in sources:
        source = Path(value).resolve()
        if not (source/'study.json').exists():
            continue
        identity = storage.read(source/'study.json')['identity']
        if identity['code_sha256'] != storage.code_hash() or identity['dataset_sha256'] != storage.digest(store.dataset):
            raise ValueError('Student engine/dataset lineage mismatch')
        cfg = identity['config']
        if cfg.get('suite_code_sha256') != store.config['suite_code_sha256'] or cfg['smoke'] != store.config['smoke']:
            raise ValueError('Student suite/smoke lineage mismatch')
        lookup = {r['job_id']:r for r in [storage.read(p) for p in (source/'jobs').glob('*/result.json')]}
        for row in lookup.values():
            if row['stage'] != 'E6_TRAIN' or row['status'] != 'complete':
                continue
            queue, ancestry = [row], {}
            while queue:
                ancestor = queue.pop()
                if ancestor['job_id'] in ancestry:
                    continue
                if ancestor['oracle'] or ancestor['status'] != 'complete' or (source/'jobs'/ancestor['job_id']/'running.lock').exists():
                    raise ValueError('Invalid checkpoint ancestry')
                storage.verify_job(source, ancestor)
                ancestry[ancestor['job_id']] = ancestor
                queue += [lookup[jid] for jid in ancestor.get('parents', [])]
            def copy_checkpoint(directory, row=row, ancestry=ancestry):
                for name in ('last.pt','training.json','learning_curve.jsonl'):
                    path=source/'jobs'/row['job_id']/name
                    if path.is_file():
                        shutil.copy2(path,directory/name)
                storage.write(directory/'source_lineage.json', {'source':str(source),'identity':identity,'ancestors':list(ancestry.values())})
                return dict(row['output'], source_job=row['job_id'], source_run=str(source))
            record={'id':row['case_id'],'source_id':'training_pool','split':'train','sha256':storage.fingerprint(identity)}
            imported.append(store.run('E6_TRAIN',record,row['arm'],copy_checkpoint,
                extra={'checkpoint_source_job':row['job_id'],'source_identity':storage.fingerprint(identity)}))
    return imported
