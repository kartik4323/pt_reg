"""Architecture interventions using the unchanged geometry/model engine."""
import copy
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial import cKDTree
from generative_assembly import data, geometry as g, experiments as ex, learning
from generative_assembly.storage import read, write, fingerprint, verify_job


def predict(store, case, stage, arm, bank, template=None, tr=None, policy='gated', refine=True,
            oracle=False, extra=None, diagnostics=None):
    parent, _ = ex.prepared(store, case)
    def job(directory):
        cfg = store.config['solver']
        poses, diag = g.solve(case, bank, cfg, template, policy,
                             refine and cfg['refine_evaluations'] > 0,
                             tr['output']['alignment']['heldout_error'] if tr else None)
        diag.update(diagnostics or {})
        return ex.save_prediction(directory, case, poses, diag,
                                  template_job=tr['job_id'] if tr else None,
                                  policy=policy, **(extra or {}))
    return store.run(stage, case['record'], arm, job, oracle=oracle,
                     extra=extra, parents=[parent] + ([tr] if tr else []))


def wrong_category(store, case):
    pool = sorted((r for r in read(store.dataset)['cases'] if r['split'] in ('train', 'dev')
                   and r['category'] != case['record']['category']
                   and r['source_id'] != case['record']['source_id']), key=lambda r: r['id'])
    for rec in pool:
        other = data.load_case(store.dataset, rec, store.config)
        ref = data.reference(store.dataset, other)
        if ref and 'complete' in ref:
            points, _ = g.fit_template(ref['complete'], case, store.config['solver'], 4101)
            if points is not None:
                return points
    return None


def e4(store, case, compute_only=False):
    cfg = store.config
    parent, bank = ex.prepared(store, case)
    if compute_only:
        def control(directory):
            relevant = [r for r in store.find('E1', case['record']['id']) + store.find('E2', case['record']['id'])
                        if r['output'].get('model') == cfg['primary_model'] and
                        r['output'].get('input_type', '').split('_v')[0] == cfg['primary_input']]
            target = sum(r['seconds'] for r in relevant)
            budget_complete = all(any(r['stage'] == stage and r['output'].get('seed') == seed for r in relevant)
                                  for stage in ('E1', 'E2') for seed in cfg['image_seeds'])
            start = time.monotonic()
            banks = [bank]
            cap = cfg['solver'].get('compute_control_max_seconds', 60)
            batches = cfg['solver'].get('compute_control_max_batches', 8)
            for index in range(batches):
                if time.monotonic() - start >= min(target, cap):
                    break
                additional, _ = g.candidates(case, cfg['solver'], data.seed_for(case['record']['id'], 'compute', index))
                banks.append(additional)
            poses, diag = g.solve(case, np.concatenate(banks), cfg['solver'], None, 'always',
                                 cfg['solver']['refine_evaluations'] > 0)
            elapsed = time.monotonic() - start
            diag.update(target_seconds=target, search_seconds=elapsed,
                        compute_matched=bool(not cfg.get('suite_allow_shared_gpu', False) and budget_complete and target > 0 and target <= cap and elapsed >= target),
                        shared_gpu_mode=cfg.get('suite_allow_shared_gpu', False),
                        timing_valid_for_isolated_comparison=not cfg.get('suite_allow_shared_gpu', False),
                        target_exceeds_cap=target>cap,
                        generation_budget_complete=budget_complete, search_candidates=sum(map(len, banks)),
                        cap_seconds=cap, isolated_scheduler=True, target_scope='primary_model_input_all_seeds_E1_E2')
            return ex.save_prediction(directory, case, poses, diag)
        return [store.run('E4', case['record'], 'B5__refine', control, parents=[parent])]
    rows = []
    def add(name, points=None, template_row=None, oracle=False, missing=None):
        for refine in (False, True):
            arm = name + ('__refine' if refine else '__rerank')
            if missing:
                rows.append(predict(store, case, 'E4', arm, bank, refine=refine, oracle=oracle,
                                    diagnostics={'fallback': 'B0', 'prior_unavailable': missing}))
            else:
                rows.append(predict(store, case, 'E4', arm, bank, points, template_row,
                                    cfg['primary_policy'], refine, oracle))
    add('B0')
    groups = ex.select_priors(store, case)
    for model, tag in [('raw', 'B1')] + [(m, 'B2') for m in cfg['image_models']]:
        for kind in cfg['input_types']:
            selected = groups[f'{model}__{kind}']['job_ids']
            found = {r['job_id']: (r, p) for r, p in ex.templates(store, case, model, kind)}
            name = f'{tag}__{model}__{kind}'
            if not selected:
                add(name, missing='no_selected_compatible_prior')
            for rank, jid in enumerate(selected):
                tr, points = found[jid]
                add(name + (f'__prior{rank+1}' if rank else ''), points, tr)
    if cfg['oracles'] and case['record']['split'] != 'train':
        shapes, _ = ex.oracle_templates(store, case)
        for key, arm in [('wrong_shape', 'B3'), ('true_shape', 'B4')]:
            if shapes.get(key) is not None:
                add(arm, shapes[key], oracle=True)
            else:
                rows.append(store.run('E4', case['record'], arm+'__unavailable',
                                      lambda d: {'not_applicable': 'reference_control_unavailable'}, oracle=True))
        cross = wrong_category(store, case)
        if cross is not None:
            add('B3_wrong_category', cross, oracle=True)
        else:
            rows.append(store.run('E4', case['record'], 'B3_wrong_category__unavailable',
                                  lambda d: {'not_applicable': 'no_other_category_in_allowed_pool'}, oracle=True))
        true_image = ex.templates(store, case, 'true_image')
        if true_image:
            add('B6', true_image[0][1], true_image[0][0], True)
        else:
            rows.append(store.run('E4', case['record'], 'B6__unavailable',
                                  lambda d: {'not_applicable': 'true_image_bridge_unavailable'}, oracle=True))
    return rows


def choose(case, bank, candidates, cfg, selector):
    if not candidates:
        return None
    if selector == 'exterior':
        return min(candidates, key=lambda x: (x[0]['output']['alignment']['heldout_error'], x[0]['job_id']))
    _, baseline = g.solve(case, bank, cfg, None, 'always', False)
    scored = []
    for row, points in candidates:
        poses, diag = g.solve(case, bank, cfg, points, 'always', False,
                             row['output']['alignment']['heldout_error'])
        exterior = row['output']['alignment']['heldout_error'] / max(cfg['gate_exterior'], 1e-6)
        contact = diag['contact_after'] / max(baseline['contact_after'], 1e-5)
        # Predetermined input-only criterion, never evaluator-calibrated.
        scored.append((exterior + contact, row['job_id'], row, points))
    best = min(scored, key=lambda x: x[:2])
    return best[2], best[3]


def e5(store, case):
    cfg = store.config
    _, bank = ex.prepared(store, case)
    found = ex.templates(store, case, cfg['primary_model'], cfg['primary_input'])
    rows = []
    # Preserve multi-view identity: one input-selected representative per seed,
    # rather than silently overwriting that seed's first reconstruction.
    by_seed = {}
    for row, points in found:
        by_seed.setdefault(row['output']['seed'], []).append((row, points))
    for K in (1, 2, 4):
        if K > len(cfg['image_seeds']):
            continue
        allowed = cfg['image_seeds'][:K]
        eligible = [choose(case, bank, by_seed[seed], cfg['solver'], cfg['suite_selector'])
                    for seed in allowed if seed in by_seed]
        selected = choose(case, bank, eligible, cfg['solver'], cfg['suite_selector'])
        disagreement = []
        for i, (_, p) in enumerate(eligible):
            for _, q in eligible[i+1:]:
                disagreement.append(float((cKDTree(p).query(q)[0].mean()+cKDTree(q).query(p)[0].mean())/2))
        for policy in ('always', 'weak', 'gated'):
            tr, points = selected if selected else (None, None)
            diag = {'requested_hypotheses': K, 'available_hypotheses': len(eligible),
                    'budget_complete': len(eligible) == K, 'selection_method': cfg['suite_selector'],
                    'mean_hypothesis_disagreement': float(np.mean(disagreement)) if disagreement else None,
                    'fallback': None if selected else 'B0', 'prior_unavailable': not bool(selected)}
            rows.append(predict(store, case, 'E5', f'{policy}__K{K}', bank, points, tr, policy,
                                diagnostics=diag))
    if cfg['oracles'] and case['record']['split'] != 'train':
        shapes, _ = ex.oracle_templates(store, case)
        controls = {'same_category': shapes.get('wrong_shape'), 'wrong_category': wrong_category(store, case)}
        # Generated-shape distortions are diagnostic controls, not pseudo-label candidates.
        if found:
            controls['aspect'] = found[0][1]*np.array([1.4, .7, 1.])
            controls['thickness'] = found[0][1]*np.array([1., 1., .5])
        for name, points in controls.items():
            if points is None:
                rows.append(store.run('E5', case['record'], f'stress_{name}__unavailable',
                                      lambda d: {'not_applicable': 'control_unavailable'}, oracle=True))
                continue
            for policy in ('always', 'weak', 'gated'):
                rows.append(predict(store, case, 'E5', f'stress_{name}__{policy}', bank, points,
                                    policy=policy, oracle=True, diagnostics={'stress_control': name}))
    return rows


def corruption_factors(cfg):
    result = [('clean', 0)]
    result += [('noise', v) for v in cfg['robustness']['noise'] if v > 0]
    result += [('dropout', v) for v in cfg['robustness']['dropout'] if v > 0]
    if cfg['robustness']['erosion_fraction'] > 0:
        result.append(('erosion', cfg['robustness']['erosion_fraction']))
    if cfg['robustness']['missing_piece']:
        result.append(('missing', 1))
    result.append(('rotation', 1))
    # Density is one factor; do not cross it with every corruption.
    return [('density', cfg['points'])] if cfg.get('suite_density_only') else result


def e7(store, case, mode='generated'):
    cfg = store.config
    # Retain the unaugmented observed frame for mapped-back rotation predictions
    # and evaluator-only previews, in addition to independently keyed corruptions.
    ex.e0(store, case)
    rows = []
    for repeat in range(cfg['robustness']['repeats']):
        for kind, value in corruption_factors(cfg):
            changed = copy.deepcopy(case) if kind in ('clean', 'density') else ex.perturb_case(
                case, kind, value, data.seed_for(case['record']['id'], kind, value, repeat))
            if changed is None:
                rows.append(store.run('E7', case['record'], f'missing__unavailable_r{repeat}',
                    lambda d: {'not_applicable': 'requires_at_least_three_input_pieces'},
                    extra={'base_case': case['record']['id'], 'factor': kind, 'value': value, 'repeat': repeat}))
                continue
            changed['record'] = dict(case['record'], id=f'{case["record"]["id"]}__{kind}_{value}_r{repeat}')
            ex.e0(store, changed)
            upstream = []
            if mode == 'generated':
                upstream = ex.e1(store, changed)
                try:
                    upstream += ex.e2(store, changed)
                except RuntimeError as exc:
                    upstream.append({'status': 'failed', 'error': str(exc)})
            parent, bank = ex.prepared(store, changed)
            found = ex.templates(store, changed, cfg['primary_model'], cfg['primary_input']) if mode != 'cpu' else []
            selected = choose(changed, bank, found, cfg['solver'], cfg['suite_selector'])
            arms = [('B0', None, None, 'always')]
            if mode == 'generated':
                arms += [(name, selected[1] if selected else None, selected[0] if selected else None, policy)
                         for name, policy in [('B2_always', 'always'), ('B2', 'gated')]]
            if mode == 'students':
                arms = []
                for trained in store.jobs('E6_TRAIN'):
                    if trained['status'] == 'complete':
                        arms.append(('student_'+trained['arm'], None, trained, 'student'))
            for arm, points, tr, policy in arms:
                def job(directory, arm=arm, points=points, tr=tr, policy=policy):
                    if policy == 'student':
                        import torch
                        state = torch.load(store.artifact(tr, 'last.pt'), map_location='cpu', weights_only=False)
                        if state.get('oracle') or state.get('smoke') != cfg['smoke']:
                            raise ValueError('Invalid student checkpoint lineage')
                        model = learning.make_model(state['hidden'])
                        model.load_state_dict(state['model']); model.eval()
                        with torch.inference_mode():
                            R, t = model([torch.tensor(p, dtype=torch.float32) for p in changed['points']], changed['anchor'])
                        poses = np.repeat(np.eye(4)[None], len(R), 0)
                        poses[:, :3, :3], poses[:, :3, 3] = R.numpy(), t.numpy()
                        diag = {'template_accepted': False, 'checkpoint_job': tr['job_id']}
                    else:
                        poses, diag = g.solve(changed, bank, cfg['solver'], points, policy,
                            cfg['solver']['refine_evaluations'] > 0,
                            tr['output']['alignment']['heldout_error'] if tr else None)
                    diag.update(upstream_failures=sum(r['status'] == 'failed' for r in upstream),
                                fallback='B0' if policy != 'student' and arm != 'B0' and not selected else None)
                    if 'augmentation_rotations' in changed:
                        R = changed['augmentation_rotations']; a = changed['anchor']
                        for i in range(len(poses)):
                            poses[i, :3, :3] = R[a].T @ poses[i, :3, :3] @ R[i]
                            poses[i, :3, 3] = R[a].T @ poses[i, :3, 3]
                    return ex.save_prediction(directory, changed, poses, diag,
                        base_case_id=case['record']['id'], factor=kind, value=value, repeat=repeat,
                        retained_ids=changed.get('retained_ids', list(range(len(poses)))),
                        template_job=tr['job_id'] if tr and policy != 'student' else None,
                        checkpoint_job=tr['job_id'] if policy == 'student' else None,
                        erosion_kind='heuristic_point_removal_proxy_not_physical_erosion' if kind == 'erosion' else None)
                extra = {'base_case': case['record']['id'], 'factor': kind, 'value': value, 'repeat': repeat}
                rows.append(store.run('E7', changed['record'], arm, job, extra=extra,
                                      parents=[parent]+([tr] if tr else [])))
    return rows


def pseudo_labels(store, records):
    K = len(store.config['image_seeds'])
    labels = []
    for rec in records:
        if rec['split'] != 'train':
            continue
        teachers = store.find('E5', rec['id'], f'gated__K{K}')
        if not teachers:
            continue
        teacher = max(teachers,key=lambda row:row['started'])
        if teacher['oracle']:
            raise ValueError('Oracle teacher rejected')
        # Audit the entire ancestor graph, not only the teacher's own flag.
        lookup = {r['job_id']: r for r in store.jobs()}
        queue = [teacher]
        seen = set()
        while queue:
            row = queue.pop()
            if row['job_id'] in seen:
                continue
            seen.add(row['job_id'])
            if row['oracle'] or row['status'] != 'complete':
                raise ValueError('Invalid pseudo-label ancestry')
            verify_job(store.root, row)
            queue += [lookup[jid] for jid in row.get('parents', [])]
        diag = teacher['output']['diagnostics']
        with np.load(store.artifact(teacher, 'poses.npz'), allow_pickle=False) as f:
            poses = f['normalized'].tolist()
        accepted = bool(diag['template_accepted'] and diag.get('budget_complete') and
                        diag['contact_after'] <= max(diag['contact_before']*store.config['solver']['gate_contact_ratio'], 1e-5))
        labels.append({'case_id': rec['id'], 'source_id': rec['source_id'], 'teacher_job': teacher['job_id'],
                       'gate_job': teacher['job_id'], 'oracle': False, 'smoke': teacher['smoke'],
                       'accepted': accepted, 'poses': poses})
    write(store.root/'pseudo_labels'/'train.json', {'schema_version': 1, 'split': 'train',
          'labels': labels, 'selection_uses_reference': False, 'teacher_policy': 'gated'})
    return labels


def train(store, records):
    recipe = {'config': {k: v for k, v in store.config.items() if k != 'training'},
              'teacher_policy': 'gated', 'selection_uses_reference': False, 'round': 1}
    frozen = store.root/'frozen_teacher.json'
    if frozen.exists() and read(frozen) != recipe:
        raise ValueError('Frozen teacher changed')
    write(frozen, recipe)
    # Scope the compatibility hook to this isolated worker process. The original
    # learner, PointNet, optimizer, and checkpoint format remain unchanged.
    with patch.object(learning, 'pseudo_labels', pseudo_labels):
        return learning.train(store, records)
