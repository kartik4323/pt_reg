"""Frozen method recipe and portable evidence/pipeline bundles."""
from pathlib import Path
import shutil
from .storage import read, write, fingerprint, digest, verify_job, checked, is_relative_to


def freeze(store):
    path=store.root/'frozen_method.json'
    recipe=dict(schema_version=1,study_identity=store.identity,selection_basis='explicit_config_not_test_metrics',
                image_model=store.config['primary_model'],input_type=store.config['primary_input'],policy=store.config['primary_policy'],
                checkpoint_jobs=[r['job_id'] for r in store.jobs('E6_TRAIN') if r['status']=='complete' and not r['oracle']],
                artifact_contract='poses.npz contains normalized and original input-coordinate rigid matrices',
                production_approved=False,smoke=store.config['smoke'])
    if path.exists():
        if read(path)!=recipe: raise ValueError('Frozen method changed; use a new study root and untouched test set')
    else: write(path,recipe)
    return recipe


def require_frozen(store):
    path=store.root/'frozen_method.json'
    if not path.exists(): raise ValueError('Run freeze before locked test inference/evaluation')
    r=read(path)
    if r['study_identity']!=store.identity: raise ValueError('Frozen identity mismatch')
    current=[j['job_id'] for j in store.jobs('E6_TRAIN') if j['status']=='complete' and not j['oracle']]
    if current!=r['checkpoint_jobs']: raise ValueError('Training/checkpoint set changed after freeze')
    return r


def bundle(store,destination,kind='pipeline'):
    destination=Path(destination).resolve()
    if destination.exists(): raise ValueError('Bundle destination already exists; choose a new path')
    if is_relative_to(destination, store.root): raise ValueError('Export outside the run root to avoid recursive bundles')
    if kind=='pipeline' and store.config['smoke']:
        raise ValueError('Smoke artifacts cannot enter a pipeline bundle; use --kind research')
    recipe=require_frozen(store)
    destination.mkdir(parents=True)
    rows=store.index()
    selected=[]
    for row in rows:
        if row['status']!='complete' and kind!='research': continue
        if kind=='pipeline' and (row['oracle'] or row['smoke'] or row['stage']=='E3'): continue
        verify_job(store.root,row)
        selected.append(row)
        source=store.root/'jobs'/row['job_id']
        shutil.copytree(source,destination/'jobs'/row['job_id'])
    for name in ['study.json','frozen_method.json','status.json','doctor.json']:
        if (store.root/name).exists(): shutil.copy2(store.root/name,destination/name)
    for path in store.root.glob('*.freeze.txt'):
        shutil.copy2(path,destination/path.name)
    write(destination/'job_index.json',selected)
    # Public inputs are copied so the artifacts do not depend on the old VM path.
    ds=read(store.dataset); base=Path(store.dataset).parent
    for case in ds['cases']:
        p=checked(base,case['path']); target=checked(destination/'dataset',case['path'])
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
    write(destination/'dataset'/'dataset.json',ds)
    if kind=='research' and (base/'evaluator_only').exists(): shutil.copytree(base/'evaluator_only',destination/'dataset'/'evaluator_only')
    if (store.root/'evaluation').exists(): shutil.copytree(store.root/'evaluation',destination/'evaluation')
    if (store.root/'pseudo_labels').exists(): shutil.copytree(store.root/'pseudo_labels',destination/'pseudo_labels')
    shutil.copytree(Path(__file__).parent,destination/'source'/'generative_assembly',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    recipe=dict(recipe,bundle_kind=kind,exported_jobs=len(selected),pretrained_weights='download exact locked revisions; not duplicated in bundle',
                source_package='source/generative_assembly',dataset='dataset/dataset.json',
                runtime_paths='images.python and reconstruction.python/repo are environment bindings; rebind when deploying',
                outputs='jobs/<job_id>/{image.png,mesh.obj,shape.npz,poses.npz,last.pt}; see job_index.json',
                uses_ground_truth_at_inference=False,pipeline_integration_status='candidate_components_pending_review')
    write(destination/'pipeline_recipe.json',recipe)
    hashes={str(p.relative_to(destination)).replace('\\','/'):digest(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    write(destination/'SHA256SUMS.json',hashes)
    return dict(path=str(destination),jobs=len(selected),files=len(hashes),kind=kind)
