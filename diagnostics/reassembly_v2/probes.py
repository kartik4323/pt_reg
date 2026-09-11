"""Measurements and temporary, explicitly labeled diagnostic interventions."""
import copy
from collections import Counter
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from reassembly import solver
from reassembly.data import make_oracle_field
from reassembly.evaluation import assembly_metrics
from reassembly.geometry import so3_exp
from reassembly.losses import (contact_geometry, correspondence_loss, field_losses,
                               local_contact_distribution, segmentation_loss, view_consistency)
from reassembly.model import configure_stage
from reassembly.training import collate_samples


def array(value):
    if isinstance(value,torch.Tensor):
        value=value.detach().cpu()
        return (value.float() if value.dtype==torch.bfloat16 else value).numpy()
    return np.asarray(value)


def distribution(values):
    values = array(values).reshape(-1)
    finite = values[np.isfinite(values)]
    return {'count': len(values), 'nonfinite': int(len(values)-len(finite)),
            'quantiles': np.quantile(finite, [0, .1, .5, .9, 1]).tolist() if len(finite) else None,
            'mean': float(finite.mean()) if len(finite) else None}


def binary(probability, labels):
    p, y = array(probability), array(labels).astype(bool)
    prediction = p >= .5
    tp, fp = int((prediction & y).sum()), int((prediction & ~y).sum())
    fn, tn = int((~prediction & y).sum()), int((~prediction & ~y).sum())
    divide = lambda a, b: a/b if b else None
    recalls = [a/b for a, b in ((tp, tp+fn), (tn, tn+fp)) if b]
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': divide(tp, tp+fp), 'recall': divide(tp, tp+fn), 'iou': divide(tp, tp+fp+fn),
            'balanced_accuracy': float(np.mean(recalls)), 'accuracy': (tp+tn)/y.size,
            'predicted_fraction': float(prediction.mean()), 'true_fraction': float(y.mean()),
            'probability_histogram': np.histogram(p, np.linspace(0, 1, 11))[0].tolist(),
            'fracture_probability_histogram':np.histogram(p[y],np.linspace(0,1,11))[0].tolist(),
            'exterior_probability_histogram':np.histogram(p[~y],np.linspace(0,1,11))[0].tolist()}


def representation(features):
    x = array(features).reshape(-1, features.shape[-1])
    x = x[np.linspace(0, len(x)-1, min(256, len(x)), dtype=int)]
    centered = x-x.mean(0)
    singular = np.linalg.svd(centered, compute_uv=False)
    mass = singular/singular.sum() if singular.sum() else np.zeros_like(singular)
    norm = x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    similarity = norm @ norm.T
    return {'mean_channel_variance': float(centered.var(0).mean()),
            'effective_rank': float(np.exp(-(mass*np.log(np.maximum(mass, 1e-12))).sum())) if mass.sum() else 0,
            'cross_point_cosine': distribution(similarity[~np.eye(len(x), dtype=bool)])}


def predictions(model, batch, condition='predicted', selection='predicted', gating='predicted', encoded=None):
    """No source mutation: temporary selector override restores on exit."""
    import reassembly.model as model_module
    encoded = model.encode(batch['points'], batch['fragment_mask'], batch['anchor_index']) if encoded is None else encoded
    original_selector = model_module.contact_point_indices
    labels = batch['fracture_labels']
    selection_logits = encoded['fracture_logits'] if selection == 'predicted' else torch.where(labels > .5, 20., -20.)
    xyz = encoded['point_xyz']
    fixed = original_selector(xyz.reshape(-1, xyz.shape[-2], 3), selection_logits.reshape(-1, xyz.shape[-2]), model.matcher.contact_points)
    altered = dict(encoded)
    if gating == 'oracle':
        altered['fracture_logits'] = torch.where(labels > .5, 20., -20.)
    with patch.object(model_module, 'contact_point_indices', lambda *args, **kwargs: fixed):
        return encoded, model.match(altered, use_scaffold=condition != 'contact_only')


def numpy_matches(pairs):
    return [dict(i=p['i'], j=p['j'], **{k: array(p[k])[0] for k in
            ('source_xyz', 'target_xyz', 'weights', 'source_matchability', 'target_matchability')})
            for p in pairs if bool(p['valid'][0])]


def oracle_matches(pairs, batch, cfg, encoded):
    result = numpy_matches(pairs)
    active = [p for p in pairs if bool(p['valid'][0])]
    for out, pair in zip(result, active):
        positive, distance = contact_geometry(pair, batch, cfg['train']['contact_radius'])
        p = local_contact_distribution(positive, distance, cfg['train']['contact_sigma'])
        q = local_contact_distribution(positive.transpose(-1, -2), distance.transpose(-1, -2), cfg['train']['contact_sigma'])
        gate_a=encoded['fracture_logits'][0,pair['i'],pair['source_indices'][0]].sigmoid()
        gate_b=encoded['fracture_logits'][0,pair['j'],pair['target_indices'][0]].sigmoid()
        out.update(weights=array(p*q.transpose(-1, -2)*gate_a[None,:,None]*gate_b[None,None,:])[0],
                   source_matchability=array(positive.any(-1)*gate_a[None])[0], target_matchability=array(positive.any(-2)*gate_b[None])[0])
    return result


def trace_pair(pair, cfg):
    """Expose each support boundary; do not reuse zero sentinel as a measurement."""
    weights = np.asarray(pair['weights'],dtype=np.float64); n, m = weights.shape
    ij = np.unique(np.concatenate([np.stack([np.arange(n), weights.argmax(1)], -1),
                                   np.stack([weights.argmax(0), np.arange(m)], -1)]), axis=0)
    values = weights[ij[:, 0], ij[:, 1]]
    threshold = cfg['solver'].get('min_correspondence_weight', .001)
    kept = ij[values >= threshold]; mass = float(values[values >= threshold].sum())
    result = {'matrix_weight': distribution(weights), 'row_column_max_union': len(ij),
              'above_weight_threshold': len(kept), 'selected_mass': mass,
              'unique_source': len(np.unique(kept[:, 0])), 'unique_target': len(np.unique(kept[:, 1])),
              'weight_threshold': threshold, 'mass_threshold': cfg['solver'].get('min_pair_mass', .05)}
    if not np.isfinite(weights).all():
        result['first_failure'] = 'nonfinite_weights'; return result
    if len(kept) < 3:
        result['first_failure'] = 'fewer_than_three_thresholded_entries'; return result
    if mass < cfg['solver'].get('min_pair_mass', .05):
        result['first_failure'] = 'selected_mass_below_threshold'; return result
    selected = solver._correspondences(pair, cfg['solver'])
    calls = []
    original = solver.weighted_kabsch
    def record(*args, **kwargs):
        fit = original(*args, **kwargs)
        entry={'points': len(args[0]), 'valid': fit['valid'], 'reason': fit.get('reason'),'mass':fit.get('mass')}
        if fit['valid']:
            residual=np.linalg.norm(selected['source']@fit['rotation'].T+fit['translation']-selected['target'],axis=-1)
            inliers=residual<=cfg['solver'].get('inlier_threshold',.04)
            a=len(np.unique(selected['source_indices'][inliers])); b=len(np.unique(selected['target_indices'][inliers]))
            entry.update(fit_rms=fit['rms'],determinant=float(np.linalg.det(fit['rotation'])),
                full_support_inliers=int(inliers.sum()),inlier_source_endpoints=a,inlier_target_endpoints=b,
                next_endpoint_requirement_passed=a>=3 and b>=3)
        calls.append(entry)
        return fit
    with patch.object(solver, 'weighted_kabsch', record):
        candidates = solver._pair_candidates(selected, cfg['solver']) if selected else []
    result.update(kabsch_calls=calls, candidates=len(candidates),
                  source_singular_values=np.linalg.svd(selected['source']-selected['source'].mean(0), compute_uv=False).tolist() if selected else None,
                  target_singular_values=np.linalg.svd(selected['target']-selected['target'].mean(0), compute_uv=False).tolist() if selected else None,
                  first_failure=None if candidates else 'no_geometrically_valid_candidate')
    return result


def measure(model, sample, cfg, device, condition='predicted', deep=False, selection='predicted', gating='predicted', encoded=None):
    batch = collate_samples([sample], device)
    with torch.no_grad():
        encoded, pairs = predictions(model, batch, condition,selection,gating,encoded)
        output = {'fragments': [], 'pairs': {}}
        qa=sample.get('_diagnostic_metadata',{}).get('qa',{})
        consumed=None
        if deep:
            captured={}
            def take(name):
                def hook(module,args,result): captured[name]=result.detach()
                return hook
            hooks=[model.matcher.descriptor.register_forward_hook(take('features')),
                   model.matcher.condition_gate.register_forward_hook(take('conditioning'))]
            try: model.match(encoded,use_scaffold=condition!='contact_only')
            finally:
                for hook in hooks: hook.remove()
            consumed=captured['features']
            if 'conditioning' in captured:
                gate,shift=captured['conditioning'].chunk(2,-1)
                consumed=consumed*(1+gate[:,None,None].tanh())+shift[:,None,None]
            consumed=F.normalize(consumed.float(),dim=-1)
        count = int(sample['fragment_mask'].sum())
        for i in range(count):
            p = encoded['fracture_logits'][0, i].sigmoid(); y = batch['fracture_labels'][0, i]
            majority = torch.full_like(p, float(y.mean() >= .5))
            output['fragments'].append({'fragment': i, 'points': len(p),
                'source_volume_fraction':qa.get('volume_ratios',[None]*count)[i],
                'rms_radius': float(batch['points'][0, i].square().sum(-1).mean().sqrt()),
                'predicted': binary(p, y), 'constant_half': binary(torch.full_like(p, .5), y),
                'per_fragment_majority_oracle_baseline': binary(majority, y),
                'constant_half_bce': float(np.log(2))})
        for pair, match in zip([p for p in pairs if bool(p['valid'][0])], numpy_matches(pairs)):
            positive, distance = contact_geometry(pair, batch, cfg['train']['contact_radius'])
            i, j = pair['i'], pair['j']; positives = array(positive)[0].astype(bool)
            probs = array(pair['source_prob'])[0]; target_probs = array(pair['target_prob'])[0]
            ids = [set(sample['interface_ids'][k].numpy().tolist())-{-1} for k in (i,j)]
            if 'interface_areas' in qa: ids=[{int(x) for x in qa['interface_areas'][k]} for k in (i,j)]
            value = {'true_contact': bool(qa['adjacency'][i][j]) if 'adjacency' in qa else bool(ids[0] & ids[1]),
                     'contact_truth_source':'mesh_preparation_adjacency' if 'adjacency' in qa else 'sampled_interfaces_fixture_fallback',
                     'positive_target_pairs': int(positive.sum()),
                     'support_trace': trace_pair(match, cfg), 'directions': {}}
            for name, p, truth, dist in (('source', probs, positives, array(distance)[0]),
                                         ('target', target_probs, positives.T, array(distance)[0].T)):
                top = p[:, :-1].argmax(-1); rows = np.arange(len(p)); has = truth.any(-1)
                predicted_matched = p.argmax(-1) != p.shape[-1]-1
                good = truth[rows, top]
                value['directions'][name] = {'dustbin': distribution(p[:, -1]),
                    'dustbin_matchable':distribution(p[has,-1]),'dustbin_unmatched':distribution(p[~has,-1]),
                    'entropy': distribution(-(p*np.log(np.maximum(p, 1e-12))).sum(-1)),
                    'matched_rows': int(predicted_matched.sum()),
                    'top1_precision_among_predicted_matched': float(good[predicted_matched].mean()) if predicted_matched.any() else None,
                    'top1_recall_on_matchable_rows': float(good[has].mean()) if has.any() else None,
                    'top1_gt_distance': distribution(dist[rows, top]),
                    'recall_at_k': {str(k): float(np.take_along_axis(truth, np.argsort(-p[:, :-1], axis=-1)[:, :k], axis=-1).any(-1)[has].mean()) if has.any() else None for k in (1,5,10)}}
            product = probs[:, :-1] * target_probs[:, :-1].T
            value['weight_factors'] = {'directional_product': distribution(product),
                                       'source_directional':distribution(probs[:,:-1]),'target_directional':distribution(target_probs[:,:-1]),
                                       'source_fracture_gate':distribution(encoded['fracture_logits'][0,i,pair['source_indices'][0]].sigmoid()),
                                       'target_fracture_gate':distribution(encoded['fracture_logits'][0,j,pair['target_indices'][0]].sigmoid()),
                                       'final_gated_weight': distribution(match['weights']),
                                       'product_mass': float(product.sum()), 'gated_mass': float(match['weights'].sum())}
            coverage = {}
            for part, indices in ((i, pair['source_indices'][0]), (j, pair['target_indices'][0])):
                chosen_ids = sample['interface_ids'][part, indices.cpu()]
                chosen_xyz = array(batch['points'][0, part, indices])
                per_interface = {}
                for identity in sorted(ids[0] | ids[1]):
                    selected = chosen_xyz[array(chosen_ids)==identity]
                    all_points = array(sample['points'][part])[array(sample['interface_ids'][part])==identity]
                    singular = np.linalg.svd(selected-selected.mean(0), compute_uv=False) if len(selected) else np.zeros(3)
                    per_interface[str(identity)] = {'selected': len(selected), 'available': len(all_points),
                        'singular_values': singular.tolist(),
                        'coverage_nearest_distance': distribution(np.linalg.norm(all_points[:,None]-selected[None],axis=-1).min(-1)) if len(selected) and len(all_points) else None}
                coverage[str(part)] = per_interface
            value['selection'] = coverage
            if deep:
                a = encoded['descriptor'][0,i,pair['source_indices'][0]]
                b = encoded['descriptor'][0,j,pair['target_indices'][0]]
                sim = array(a @ b.T)
                value['descriptor_cosines'] = {'true_matches': distribution(sim[positives]), 'nonmatches': distribution(sim[~positives])}
                sim=array(consumed[0,i]@consumed[0,j].T)
                value['consumed_matcher_cosines']={'true_matches':distribution(sim[positives]),'nonmatches':distribution(sim[~positives])}
            output['pairs'][f'{i}-{j}'] = value
        if deep:
            output['representations'] = {name: representation(encoded[name][:,:count]) for name in ('descriptor', 'point_features', 'token_features')}
            output['representations']['consumed_matcher_features']=representation(consumed[:,:count])
            other = model.encode(batch['points_view2'], batch['fragment_mask'], batch['anchor_index'])
            output['same_point_transformed_cosine'] = distribution((encoded['descriptor'][:,:count]*other['descriptor'][:,:count]).sum(-1))
            output['same_point_point_features_cosine']=distribution(F.cosine_similarity(encoded['point_features'][:,:count],other['point_features'][:,:count],dim=-1))
    return output, encoded, pairs


def exterior(sample, encoded, oracle=False):
    count = int(sample['fragment_mask'].sum()); points=[]; weights=[]
    for i in range(count):
        ids = np.linspace(0, len(sample['points'][i])-1, min(256,len(sample['points'][i])), dtype=int)
        points.append(array(sample['points'][i])[ids])
        w = 1-array(sample['fracture_labels'][i]) if oracle else array(torch.sigmoid(-encoded['fracture_logits'][0,i]))
        weights.append(w[ids])
    return points, weights


def grid(model, encoded, cfg, sample, condition):
    if condition == 'none': return None
    value = solver._make_field(model, encoded, cfg, make_oracle_field(sample) if condition == 'gt' else None)
    if condition == 'perturbed':
        value = solver.ScaffoldGrid(np.clip(np.roll(value.distance,3,axis=0)+.035,-value.truncation,value.truncation),
                                     value.uncertainty.copy(),value.bounds.copy(),value.truncation)
    return value


def override_prior(model, encoded, field):
    anchor = int(encoded['anchor_index'][0])
    xyz = encoded['token_xyz'][:,anchor]
    query = torch.cat([model.conditioning_queries[None].to(xyz),xyz],1)
    distance, confidence, _ = field.sample(array(query)[0])
    sigma = np.maximum(.01*(1/confidence-1),np.exp(-6))
    return torch.cat([query, torch.as_tensor(distance,device=query.device,dtype=query.dtype)[None,:,None],
                      torch.as_tensor(np.log(sigma),device=query.device,dtype=query.dtype)[None,:,None]],-1)


def solve_matches(matches, sample, cfg, encoded, field=None):
    points, weights = exterior(sample, encoded)
    result = solver.solve_from_matches(matches,int(sample['fragment_mask'].sum()),int(sample['anchor_index']),points,weights,field,cfg)
    metrics = assembly_metrics(sample,result,cfg['solver']['success_threshold'])
    metrics.update(reason=result.get('reason'),diagnostics=result['diagnostics'])
    return metrics


def refinement_control(matches,sample,cfg,encoded,field):
    """Capture contact-only initial assemblies before refinement; replay fixed starts."""
    starts=[]; original=solver._refine
    def capture(poses,anchor,contacts,points,weights,unused_field,options):
        starts.append((copy.deepcopy(poses),anchor,contacts,points,weights,options))
        return original(poses,anchor,contacts,points,weights,unused_field,options)
    with patch.object(solver,'_refine',capture):
        baseline=solve_matches(matches,sample,cfg,encoded,None)
    rows=[]
    for poses,anchor,contacts,points,weights,options in starts:
        refined,accepted=original(poses,anchor,contacts,points,weights,field,options)
        def metric(value):
            return assembly_metrics(sample,dict(rotations=value[0],translations=value[1],status='diagnostic_fixed_initialization',confidence=0),cfg['solver']['success_threshold'])
        rows.append({'before':metric(poses),'after':metric(refined),'accepted_steps':accepted,
                     'initial_rotations':poses[0],'initial_translations':poses[1]})
    return (rows[0]['after'] if rows else baseline),rows


def gradient_probe(model, sample, cfg, device, stage):
    model = copy.deepcopy(model).train(); configure_stage(model,stage)
    batch = collate_samples([sample],device)
    encoded = model.encode(batch['points'],batch['fragment_mask'],batch['anchor_index'])
    parts = {}
    pairs = model.match(encoded, use_scaffold=stage==3 and cfg['train'].get('condition')!='contact_only')
    correspondence_loss(pairs,batch,cfg['train']['contact_radius'],localization_sigma=cfg['train']['contact_sigma'],diagnostics=parts)
    if stage < 3:
        parts['segmentation'] = segmentation_loss(encoded['fracture_logits'],batch['fracture_labels'],batch['fragment_mask'])
        parts['view_consistency'] = view_consistency(model,encoded,batch)
    if stage == 2:
        field = model.scaffold(encoded,batch['sdf_queries'])
        f = field_losses(field['distance'],field['log_scale'],batch['sdf_values'],cfg['model']['truncation'])
        parts.update(sdf=f['sdf_l1'],calibration=f['sdf_calibration'])
    parameters = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
    w=cfg['loss']; geometry=w.get('geometry_retention',.25) if stage==2 else 1.
    factors={'matching_mass':geometry*w.get('matching',1),
        'matching_localization':geometry*w.get('matching',1)*w.get('matching_localization',1),
        'segmentation':geometry*w.get('segmentation',1),'view_consistency':geometry*w.get('view_consistency',.1),
        'sdf':w.get('sdf',1),'calibration':w.get('calibration',.01)}
    vectors={}; result={}
    for name,loss in parts.items():
        gradients=torch.autograd.grad(loss,[p for _,p in parameters],allow_unused=True,retain_graph=True)
        vector=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for (_,p),g in zip(parameters,gradients)])
        vectors[name]=vector
        result[name]={'loss':float(loss.detach()),'gradient_norm':float(vector.norm()),'finite':bool(torch.isfinite(vector).all()),
                      'objective_weight':factors[name],'weighted_gradient_norm':float(vector.norm())*abs(factors[name]),
                      'module_norms':{module:float(torch.sqrt(sum((g.detach().square().sum() for (n,p),g in zip(parameters,gradients) if n.startswith(module+'.') and g is not None),torch.tensor(0.,device=device)))) for module in ('encoder','matcher','field')}}
    cosines={}
    for i,a in enumerate(vectors):
        for b in list(vectors)[i+1:]:
            denominator=vectors[a].norm()*vectors[b].norm()
            cosines[f'{a}/{b}']=float(vectors[a]@vectors[b]/denominator) if denominator>0 else None
    return {'objectives':result,'gradient_cosines':cosines,'optimizer_updates':0,
            'note':'Current-checkpoint gradients are associations, not evidence of past optimization dynamics.'}
