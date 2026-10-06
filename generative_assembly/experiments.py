"""E0–E7 execution. Only explicitly marked oracle branches access reference()."""
from __future__ import annotations
import copy
import json
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from . import data, geometry as g, render
from .backends import launch
from .storage import read, write


def prepared(store, case):
    found = store.find('E0', case['record']['id'], 'prepare')
    if not found: raise RuntimeError('E0 prepare must complete first')
    row = found[0]
    with np.load(store.artifact(row, 'candidates.npz'), allow_pickle=False) as f:
        bank = f['poses']
    return row, bank


def save_prediction(directory, case, poses, diagnostics, **extra):
    g.check_poses(poses, len(case['points']), case['anchor'])
    np.savez_compressed(directory/'poses.npz', normalized=poses, original=g.export_poses(case, poses))
    write(directory/'prediction.json', dict(diagnostics=diagnostics, coordinate_convention='y=x@R.T+t', anchor=case['anchor'], scale=case['scale'],
                                          centers=case['centers'].tolist(), **extra))
    return dict(diagnostics=diagnostics, **extra)


def _build_prompt(cfg, rec):
    """Select the active prompt text based on cfg['prompt_variant']."""
    variant = cfg.get('prompt_variant', 'original')
    if variant == 'original':
        base = cfg.get('prompt_original', cfg.get('prompt', ''))
    else:  # 'short' or 'category'
        base = cfg.get('prompt_short', cfg.get('prompt', ''))
    if cfg.get('category_prompt') and rec.get('category'):
        base = base.rstrip('.') + f'. The object is a {rec["category"]}.'
    return base


def _render_fragment(case, cfg, bank, kind):
    """Render a single fragment-set into a rendering dict.

    Applies render_mode (surface vs splat) and tight_crop based on config.
    """
    selected = [case['anchor']] if kind == 'F' else list(range(len(case['points'])))
    points = np.concatenate([case['points'][i] if kind == 'F' else g.apply(case['points'][i], bank[0, i]) for i in selected])
    ns = np.concatenate([case['normals'][i] if kind == 'F' else case['normals'][i] @ bank[0, i, :3, :3].T for i in selected])
    ext = np.concatenate([case['exterior'][i] for i in selected])
    return points, ns, ext


def e0(store, case):
    cfg, rec = store.config, case['record']
    render_mode = cfg.get('render_mode', 'splat')
    mask_type   = cfg.get('mask_type', 'original')
    fill_target = cfg.get('canvas_fill_target', 0.0)
    n_views     = cfg.get('n_render_views', 1)

    def prepare(out):
        started = time.time()
        bank, scores = g.candidates(case, cfg['solver'], data.seed_for(cfg['seed'], rec['id'], 'bank'))
        np.savez_compressed(out/'candidates.npz', poses=bank, scores=scores)

        # ── Camera(s) ────────────────────────────────────────────────────────
        anchor_pts = case['points'][case['anchor']]
        if n_views == 1:
            camera = render.camera_for(anchor_pts, cfg['pixels'], cfg['canvas_extent'], cfg['splat_radius'])
            cameras = [camera]
            camera_labels = ['v0']
        else:
            cameras = render.camera_for(anchor_pts, cfg['pixels'], cfg['canvas_extent'],
                                        cfg['splat_radius'], n_views=n_views)
            camera_labels = [f'v{i}' for i in range(n_views)]
            camera = cameras[0]  # keep canonical camera for coordinate checks

        write(out/'camera.json', camera)

        # ── Render each view × each input_type ───────────────────────────────
        for kind in cfg['input_types']:
            pts, ns, ext = _render_fragment(case, cfg, bank, kind)

            for cam, vlabel in zip(cameras, camera_labels):
                # Choose renderer
                if render_mode == 'surface':
                    rendered = render.render_surface(pts, ns, ext, cam, cfg['splat_radius'])
                else:
                    rendered = render.render(pts, ns, ext, cam, cfg['splat_radius'])

                # Optional tight crop (FIX #6)
                if fill_target > 0:
                    rendered = render.tight_crop(rendered, target_fill=fill_target)

                # Save with mask type (FIX #5)
                directory = out / f'{kind}_{vlabel}' if n_views > 1 else out / kind
                directory.mkdir(exist_ok=True)
                render.save(directory, rendered, mask_type=mask_type)
                write(directory / 'camera.json', dict(cam, crop_box=rendered.get('crop_box'),
                      image_transform='crop_then_resize' if fill_target > 0 else 'identity'))

        np.savez_compressed(out/'observed.npz', **{f'points_{i}':p for i,p in enumerate(case['points'])},
                            **{f'indices_{i}':p for i,p in enumerate(case['indices'])},
                            centers=case['centers'], scale=case['scale'], anchor=case['anchor'])

        # Coordinate integrity check (unchanged, uses canonical camera)
        exported = g.export_poses(case, bank[0])
        errors = [np.max(abs((g.apply(case['points'][i]*case['scale']+case['centers'][i],exported[i])-case['centers'][case['anchor']])/case['scale'] - g.apply(case['points'][i],bank[0,i]))) for i in range(len(exported))]
        p_anc = case['points'][case['anchor']]
        r = render.render(p_anc, None, None, camera, 0)
        y,x = np.nonzero(r['valid']); B=np.asarray(camera['basis']); extent=cfg['canvas_extent']; size=cfg['pixels']
        reconstructed = np.column_stack(((x/(size-1)-0.5)*2*extent,(0.5-y/(size-1))*2*extent,r['depth'][y,x])) @ B.T
        raster_error = float(np.linalg.norm(reconstructed-p_anc[r['ids'][y,x]],axis=1).max())
        tolerance = np.sqrt(2)*extent/(size-1)+1e-6
        if max(errors) > 1e-5 or raster_error > tolerance: raise RuntimeError('Coordinate integrity failed')
        if shutil.disk_usage(store.root).free < cfg['resources']['minimum_free_gib']*1024**3:
            raise RuntimeError('Insufficient free disk for configured experiment budget')
        write(out/'checks.json', dict(transform_error=max(errors), depth_error=raster_error, raster_tolerance=tolerance,
                                      reference_files_accessed=False, gpu_smoke='performed_by_first_real_E1_E2_jobs',
                                      render_mode=render_mode, mask_type=mask_type, n_views=n_views))
        return dict(anchor=case['anchor'], scale=case['scale'], candidates=len(bank),
                    geometry_seconds=time.time()-started,
                    exterior_method=case['exterior_method'],
                    normalization='input_only_shared_max_fragment_diameter',
                    render_mode=render_mode, mask_type=mask_type, n_views=n_views)
    return [store.run('E0', rec, 'prepare', prepare)]



def e1(store, case):
    cfg, rec = store.config, case['record']
    parent, _ = prepared(store, case)
    n_views = cfg.get('n_render_views', 1)
    rows = []

    # Build the list of (kind_label, kind_base) pairs to iterate.
    # If n_views=1: F, A  (original behaviour)
    # If n_views>1: F_v0, F_v1, ..., A_v0, A_v1, ...
    input_dirs = []
    for kind in cfg['input_types']:
        if n_views == 1:
            input_dirs.append((kind, kind))   # (arm_suffix, dir_name)
        else:
            for vi in range(n_views):
                input_dirs.append((f'{kind}_v{vi}', f'{kind}_v{vi}'))

    prompt = _build_prompt(cfg, rec)

    for kind_label, dir_name in input_dirs:
        image_path = store.artifact(parent, f'{dir_name}/image.png')
        try:
            with np.load(store.artifact(parent, f'{dir_name}/render.npz'), allow_pickle=False) as f:
                input_render = {k: f[k] for k in f.files}
        except Exception:
            continue

        def raw(out, image=image_path, kl=kind_label):
            shutil.copy2(image, out / 'image.png')
            return dict(model='raw', input_type=kl, seed=None)

        rows.append(store.run('E1', rec, f'raw__{kind_label}', raw, parents=[parent]))

        for model in cfg['image_models']:
            for seed in cfg['image_seeds']:
                def generate(out, model=model, kl=kind_label, dn=dir_name, seed=seed,
                             input_render=input_render, image=image_path):
                    request = dict(
                        kind='image', smoke=cfg['smoke'], model=model,
                        config=cfg['images'], image=str(image),
                        control=str(store.artifact(parent, f'{dn}/control.png')),
                        mask=str(store.artifact(parent, f'{dn}/mask.png')),
                        seed=seed, prompt=prompt, oracle=False,
                    )
                    launch(request, out, cfg['images']['python'], cfg['resources']['worker_timeout_seconds'])
                    return dict(model=model, input_type=kl, seed=seed,
                                selection=render.image_score(out / 'image.png', input_render),
                                raw_selection=render.image_score(out / 'image_raw.png', input_render)
                                if (out / 'image_raw.png').exists() else None,
                                mask_type=cfg['mask_type'], prompt=prompt)
                rows.append(store.run('E1', rec, f'{model}__{kind_label}__{seed}', generate, parents=[parent]))
    return rows



def _reconstruct(store, case, row, multiview_rows=None):
    """Run InstantMesh reconstruction on a single E1 image row.

    If multiview_rows is provided (list of E1 rows from different camera angles),
    their images are packed into a 2×3 grid and passed to InstantMesh as Fix C.
    """
    cfg = store.config
    def reconstruct(out):
        import trimesh
        image = store.artifact(row, 'image.png')

        # Fix C: Collect sibling view images for multi-view grid if configured
        multiview_images = []
        if multiview_rows:
            for mv_row in multiview_rows:
                try:
                    p = store.artifact(mv_row, 'image.png')
                    if Path(p).exists():
                        multiview_images.append(str(p))
                except Exception:
                    pass

        launch(dict(kind='reconstruction', smoke=cfg['smoke'], config=cfg['reconstruction'],
                    image=str(image), oracle=row['oracle'],
                    multiview_images=multiview_images),
               out, cfg['reconstruction']['python'], cfg['reconstruction']['timeout_seconds'])
        mesh = trimesh.load(out / 'mesh.obj', force='mesh', process=False)
        if len(mesh.faces) == 0: raise ValueError('Reconstructor returned no surface')
        points, _ = data.sample_mesh(mesh, max(1024, cfg['evaluation_points']),
                                     np.random.default_rng(data.seed_for(row['job_id'], 'surface')))

        # FIX #8: fit_template returns (None, rejection_dict) for degenerate shapes.
        solver_cfg = dict(cfg.get('reconstruction', {}), **cfg['solver'])
        aligned, fit = g.fit_template(points, case, solver_cfg, data.seed_for(case['record']['id'], 'align'))
        if aligned is None:
            np.savez_compressed(out / 'shape.npz', raw=points)
        else:
            np.savez_compressed(out / 'shape.npz', raw=points, aligned=aligned)
        write(out / 'alignment.json', fit)
        return dict(model=row['output']['model'], input_type=row['output']['input_type'],
                    seed=row['output'].get('seed'), image_job=row['job_id'],
                    alignment=fit, template_rejected='rejected' in fit,
                    watertight=bool(mesh.is_watertight), vertices=len(mesh.vertices), faces=len(mesh.faces),
                    n_multiview_images=len(multiview_images))
    return store.run('E2', case['record'], row['arm'], reconstruct, oracle=row['oracle'], parents=[row])



def e2(store, case):
    cfg = store.config
    rows = []
    image_rows = store.find('E1', case['record']['id'])
    n_mv = cfg.get('n_instantmesh_views', 1)

    for model in ['raw'] + cfg['image_models']:
        # Collect all base kinds (F, A) — strip view suffixes for grouping
        base_kinds = cfg['input_types']
        for base_kind in base_kinds:
            # Match rows for this model+base_kind, across all view variants (F, F_v0, F_v1, ...)
            group = [r for r in image_rows
                     if r['output']['model'] == model
                     and (r['output']['input_type'] == base_kind
                          or r['output']['input_type'].startswith(f'{base_kind}_v'))]
            if model != 'raw' and cfg.get('skip_invalid_e1', True):
                group = [r for r in group if r['output'].get('selection', {}).get('valid_foreground', False)]
            group.sort(key=lambda r: (r['output'].get('selection', {}).get('selection_score', 0),
                                      r['output'].get('seed') or 0))
            if not group:
                continue
            if not cfg['reconstruct_all_for_E5']:
                group = group[:cfg['reconstruction_top_k']]

            if n_mv > 1:
                raise ValueError('Independent completions are not calibrated consistent InstantMesh views; use n_instantmesh_views=1')
            rows.extend(_reconstruct(store, case, r) for r in group)

    if cfg['oracles'] and case['record']['split'] != 'train':
        ref = data.reference(store.dataset, case)
        if ref is not None and 'complete' in ref:
            parent, _ = prepared(store, case)
            camera = read(store.artifact(parent, 'camera.json'))
            def oracle_image(out):
                r = (render.render_surface if cfg.get('render_mode') == 'surface' else render.render)(
                    ref['complete'], None, None, camera, cfg['splat_radius'])
                # Oracle uses the same input-derived image transform, never its own crop.
                kind = cfg['input_types'][0] + ('_v0' if cfg.get('n_render_views', 1) > 1 else '')
                with np.load(store.artifact(parent, f'{kind}/render.npz')) as f:
                    if 'crop_box' in f.files:
                        r = render.crop_render(r, f['crop_box'])
                render.save(out, r)
                return dict(model='true_image', input_type='oracle', seed=None)
            r = store.run('E1_ORACLE', case['record'], 'true_image', oracle_image, oracle=True, parents=[parent])
            if r['status'] == 'complete':
                rows.append(_reconstruct(store, case, r))
    select_priors(store, case)
    if not rows: raise RuntimeError('No valid E1 outputs to reconstruct')
    return rows



def templates(store,case,model=None,kind=None):
    rows=store.find('E2',case['record']['id'])
    rows=[r for r in rows if not r['output'].get('template_rejected')
          and r['output'].get('alignment', {}).get('heldout_error') is not None
          and (model is None or r['output']['model']==model)
          and (kind is None or r['output']['input_type'].split('_v')[0]==kind)]
    result=[]
    for row in rows:
        with np.load(store.artifact(row,'shape.npz'),allow_pickle=False) as f: p=f['aligned']
        result.append((row,p))
    # Deployment selection uses held-out observed reference points, never GT.
    return sorted(result,key=lambda x:(x[0]['output']['alignment']['heldout_error'],x[0]['output'].get('seed') or 0))


def select_priors(store, case):
    """Keep up to three diverse, input-compatible hypotheses per model/input.

    Selection never reads evaluator references. A shortfall is reported, not padded.
    """
    from scipy.spatial import cKDTree
    groups = {}
    for model in ['raw'] + store.config['image_models']:
        for kind in store.config['input_types']:
            selected = []
            rejected = []
            for row, points in templates(store, case, model, kind):
                if row['output']['alignment']['heldout_error'] > store.config.get('prior_max_heldout_error', 0.08):
                    rejected.append(dict(job_id=row['job_id'], reason='heldout_error'))
                    continue
                if model != 'raw' and not store.config['smoke']:
                    image_rows = store.find('E1', case['record']['id'])
                    image_row = next((r for r in image_rows if r['job_id'] == row['output'].get('image_job')), None)
                    if not image_row or image_row['output'].get('selection', {}).get('growth_relative_to_input', 0) < store.config.get('prior_min_growth', 0.05):
                        rejected.append(dict(job_id=row['job_id'], reason='insufficient_foreground_growth'))
                        continue
                distances = [(cKDTree(p).query(points)[0].mean() +
                              cKDTree(points).query(p)[0].mean()) / 2 for _, p in selected]
                if distances and min(distances) < store.config.get('prior_diversity_distance', 0.025):
                    rejected.append(dict(job_id=row['job_id'], reason='duplicate_geometry'))
                    continue
                selected.append((row, points))
                if len(selected) == store.config.get('prior_count', 3):
                    break
            bundle = {}
            for rank, (row, points) in enumerate(selected):
                bundle[f'normalized_{rank+1}'] = points
                bundle[f'original_anchor_frame_{rank+1}'] = points * case['scale'] + case['centers'][case['anchor']]
            bundle_path = store.root / 'priors' / case['record']['id'] / f'{model}__{kind}.npz'
            if bundle:
                bundle_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(bundle_path, **bundle)
            groups[f'{model}__{kind}'] = dict(job_ids=[r['job_id'] for r, _ in selected],
                requested=store.config.get('prior_count', 3), available=len(selected),
                filtered_candidates=rejected,
                point_cloud_bundle=str(bundle_path.relative_to(store.root)) if bundle else None,
                selection='heldout_observed_error_then_aligned_shape_diversity')
    write(store.root / 'priors' / f"{case['record']['id']}.json", groups)
    return groups


def oracle_templates(store,case):
    ref=data.reference(store.dataset,case)
    if ref is None or 'complete' not in ref: return {},ref
    cfg=store.config['solver']
    true,_=g.fit_template(ref['complete'],case,cfg,data.seed_for(case['record']['id'],'align'))
    distorted=ref['complete']*np.array([1.4,0.7,1.0])
    distorted,_=g.fit_template(distorted,case,cfg,4101)
    result={'true_shape':true,'distorted_shape':distorted}
    pool=read(store.dataset)['cases']
    pool=sorted([c for c in pool if c['source_id']!=case['record']['source_id'] and c['split'] in ('train','dev') and c['category']==case['record']['category']],key=lambda c:c['id'])
    for c in pool:
        other=data.load_case(store.dataset,c,store.config)
        r=data.reference(store.dataset,other)
        if r is not None and 'complete' in r:
            result['wrong_shape'],_=g.fit_template(r['complete'],case,cfg,4101)
            break
    return result,ref


def e3(store,case):
    parent,bank=prepared(store,case); rows=[]
    if not store.config['oracles'] or case['record']['split']=='train':
        return [store.run('E3',case['record'],'not_applicable',lambda d:dict(not_applicable='oracles_disabled'))]
    shapes,ref=oracle_templates(store,case)
    if ref is None:
        return [store.run('E3',case['record'],'not_applicable',lambda d:dict(not_applicable='reference_unavailable'))]
    regimes={'global':bank}
    for degrees,translation in [(0,0),(5,0.02),(15,0.05)]:
        rng=np.random.default_rng(data.seed_for(case['record']['id'],degrees))
        p=ref['poses'].copy()
        for i in range(len(p)):
            if i==case['anchor']: continue
            axis=rng.normal(size=3); axis/=np.linalg.norm(axis)
            p[i,:3,:3]=Rotation.from_rotvec(axis*np.deg2rad(degrees)).as_matrix()@p[i,:3,:3]
            axis=rng.normal(size=3); axis/=np.linalg.norm(axis)
            p[i,:3,3]+=axis*translation
        regimes[f'perturb_{degrees}']=p[None]
    for name,template in {'none':None,**shapes}.items():
        for regime,b in regimes.items():
            def job(out,b=b,template=template,regime=regime,name=name):
                # If we have a shape template, regenerate candidates using the shape prior
                if regime == 'global' and template is not None:
                    b, _ = g.candidates(case, store.config['solver'], data.seed_for(case['record']['id'], 'bank', name), template=template)
                
                p,diag=g.solve(case,b,store.config['solver'],template,'always',True)
                np.savez_compressed(out/'starts.npz',poses=b)
                return save_prediction(out,case,p,diag,template=name,regime=regime,privilege='GT_start' if regime!='global' else 'template_only')
            rows.append(store.run('E3',case['record'],f'{name}__{regime}',job,oracle=True,parents=[parent]))
    return rows


def e4(store,case):
    cfg=store.config; parent,bank=prepared(store,case); rows=[]
    arms=[('B0',None,None,False)]
    for model,tag in [('raw','B1')]+[(m,'B2') for m in cfg['image_models']]:
        for kind in cfg['input_types']:
            found=templates(store,case,model,kind)
            chosen = select_priors(store, case)[f'{model}__{kind}']['job_ids']
            found = [x for x in found if x[0]['job_id'] in chosen]
            if found:
                for rank, (tr, points) in enumerate(found):
                    arms.append((f'{tag}__{model}__{kind}' + (f'__prior{rank+1}' if rank else ''), points, tr, False))
            else:
                def missing(d): return dict(not_applicable='No selected compatible complete hypothesis; see priors manifest')
                rows.append(store.run('E4',case['record'],f'{tag}__{model}__{kind}__refine',missing,parents=[parent]))
    if cfg['oracles'] and case['record']['split']!='train':
        shapes,_=oracle_templates(store,case)
        for k,tag in [('wrong_shape','B3'),('true_shape','B4')]:
            if k in shapes: arms.append((tag,shapes[k],None,True))
        oracle=templates(store,case,'true_image')
        if oracle: arms.append(('B6',oracle[0][1],oracle[0][0],True))
    for name,template,tr,oracle in arms:
        for refine in [False,True]:
            def job(out,template=template,tr=tr,refine=refine):
                p,diag=g.solve(case,bank,cfg['solver'],template,cfg['primary_policy'],refine,
                               tr['output']['alignment']['heldout_error'] if tr else None)
                return save_prediction(out,case,p,diag,template_job=tr['job_id'] if tr else None)
            rows.append(store.run('E4',case['record'],name+('__refine' if refine else '__rerank'),job,oracle=oracle,parents=[parent]+([tr] if tr else [])))
    # Compute-control is explicitly capped and reports whether it actually matched.
    def extra_compute(out):
        target=sum(r['seconds'] for r in store.find('E1',case['record']['id'])+store.find('E2',case['record']['id']))
        cap=cfg['solver'].get('compute_control_max_seconds',60)
        start=time.time(); banks=[bank]; count=0
        while time.time()-start < min(target,cap) and count < cfg['solver'].get('compute_control_max_batches',8):
            b,_=g.candidates(case,cfg['solver'],data.seed_for(case['record']['id'],'compute',count)); banks.append(b); count+=1
        p,diag=g.solve(case,np.concatenate(banks),cfg['solver'],None,'always',True)
        diag.update(target_seconds=target,search_seconds=time.time()-start,compute_matched=(time.time()-start)>=target,
                    search_candidates=sum(len(b) for b in banks),cap_seconds=cap)
        return save_prediction(out,case,p,diag)
    rows.append(store.run('E4',case['record'],'B5__refine',extra_compute,parents=[parent]))
    return rows


def e5(store,case):
    cfg=store.config; parent,bank=prepared(store,case); rows=[]
    found=templates(store,case,cfg['primary_model'],cfg['primary_input'])
    # Nested seed sets, then input-only selection within the permitted budget.
    by_seed={r['output']['seed']:(r,p) for r,p in found}
    budgets=sorted(set([1,2,4,len(cfg['image_seeds'])]))
    for K in budgets:
        if K>len(cfg['image_seeds']): continue
        eligible=[by_seed[s] for s in cfg['image_seeds'][:K] if s in by_seed]
        for policy in ['always','weak','gated']:
            def job(out,eligible=eligible,policy=policy,K=K):
                if not eligible: raise RuntimeError('No reconstructed hypotheses for this nested budget')
                tr,p=min(eligible,key=lambda x:x[0]['output']['alignment']['heldout_error'])
                T,diag=g.solve(case,bank,cfg['solver'],p,policy,True,tr['output']['alignment']['heldout_error'])
                diag.update(requested_hypotheses=K,available_hypotheses=len(eligible),budget_complete=len(eligible)==K)
                return save_prediction(out,case,T,diag,template_job=tr['job_id'],policy=policy)
            rows.append(store.run('E5',case['record'],f'{policy}__K{K}',job,parents=[parent]+[r for r,p in eligible]))
    return rows


def perturb_case(case,kind,value,seed):
    out=copy.deepcopy(case); rng=np.random.default_rng(seed)
    if kind=='missing' and len(out['points'])<=2: return None
    if kind=='missing':
        remove=next(i for i in range(len(out['points'])) if i!=out['anchor'])
        for key in ('points','normals','exterior','original','indices'): out[key].pop(remove)
        out['centers']=np.delete(out['centers'],remove,0)
        out['anchor']-=int(remove<out['anchor']); out['retained_ids']=[i for i in range(len(case['points'])) if i!=remove]
    else:
        for i,p in enumerate(out['points']):
            if kind=='noise': out['points'][i]=p+rng.normal(0,value,p.shape)
            elif kind in ('dropout','erosion'):
                order=rng.permutation(len(p)) if kind=='dropout' else np.argsort(out['exterior'][i])[::-1]
                keep=order[:max(16,int(len(p)*(1-value)))]
                for key in ('points','normals','exterior','indices'): out[key][i]=out[key][i][keep]
            elif kind=='rotation':
                R=Rotation.random(random_state=rng).as_matrix()
                out['points'][i]=p@R.T; out['normals'][i]=out['normals'][i]@R.T
                out.setdefault('augmentation_rotations',[]).append(R)
        if kind=='noise':
            for i,p in enumerate(out['points']): out['normals'][i],out['exterior'][i]=g.normals_features(p)
    return out


def e7(store,case):
    """Re-render, regenerate and reconstruct changed observations; never reuse clean priors."""
    cfg=store.config; rows=[]
    factors=[('noise',v) for v in cfg['robustness']['noise']]+[('dropout',v) for v in cfg['robustness']['dropout']]
    factors += [('erosion',cfg['robustness']['erosion_fraction'])]
    if cfg['robustness']['missing_piece']: factors += [('missing',1)]
    for repeat in range(cfg['robustness']['repeats']):
        factors_repeat=factors+[('rotation',repeat)]
        for kind,value in factors_repeat:
            seed=data.seed_for(case['record']['id'],kind,value,repeat)
            changed=perturb_case(case,kind,value,seed)
            if changed is None: continue
            changed['record']=dict(case['record'],id=f'{case["record"]["id"]}__{kind}_{value}_r{repeat}')
            # Changed-case ID participates in every artifact key and model seed.
            subrows=e0(store,changed)+e1(store,changed)+e2(store,changed)
            failures=[r for r in subrows if r['status']=='failed']
            parent,bank=prepared(store,changed)
            found=templates(store,changed,cfg['primary_model'],cfg['primary_input'])
            for arm in ('B0','B2'):
                def job(out,arm=arm,found=found,changed=changed,bank=bank,kind=kind,value=value,repeat=repeat,failures=failures):
                    tr,p=(found[0] if found and arm=='B2' else (None,None))
                    if arm=='B2' and not found: raise RuntimeError('Corrupted-input generation/reconstruction failed')
                    T,diag=g.solve(changed,bank,cfg['solver'],p,cfg['primary_policy'],True,
                                   tr['output']['alignment']['heldout_error'] if tr else None)
                    # Map augmented rotations back to the base observation convention for evaluation/export.
                    if 'augmentation_rotations' in changed:
                        R=changed['augmentation_rotations']; a=changed['anchor']
                        for i in range(len(T)):
                            T[i,:3,:3]=R[a].T@T[i,:3,:3]@R[i]
                            T[i,:3,3]=R[a].T@T[i,:3,3]
                    return save_prediction(out,changed,T,diag,base_case_id=case['record']['id'],factor=kind,value=value,repeat=repeat,
                                           retained_ids=changed.get('retained_ids',list(range(len(T)))),
                                           upstream_failures=len(failures),template_job=tr['job_id'] if tr else None,
                                           erosion_kind='heuristic_low_curvature_removal_not_physical_material_loss' if kind=='erosion' else None)
                rows.append(store.run('E7',changed['record'],arm,job,extra=dict(base_case=case['record']['id'],factor=kind,value=value,repeat=repeat),parents=[parent]+([found[0][0]] if found and arm=='B2' else [])))
    return rows


STAGES={'E0':e0,'E1':e1,'E2':e2,'E3':e3,'E4':e4,'E5':e5,'E7':e7}
