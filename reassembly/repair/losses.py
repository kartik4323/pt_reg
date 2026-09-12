"""Objectives and validation metrics for the isolated repair experiments."""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from reassembly.losses import (contact_geometry, correspondence_loss, field_losses,
                               masked_mean, segmentation_loss, view_consistency)


def contrastive_masks(distance: Tensor, valid_source: Tensor, valid_target: Tensor,
                      positive_radius: float = .05, negative_radius: float = .1,
                      same_surface: Tensor | None = None):
    if not 0 < positive_radius < negative_radius:
        raise ValueError('Contrastive radii must satisfy 0 < positive < negative')
    valid = valid_source[..., :, None].bool() & valid_target[..., None, :].bool()
    positive = valid & (distance <= positive_radius)
    if same_surface is not None:
        positive = positive & same_surface.bool()
    negative = valid & (distance > negative_radius)
    return positive, negative


def multipositive_contrastive(source: Tensor, target: Tensor, positive: Tensor,
                              negative: Tensor, temperature: float = .1):
    """Symmetric positive-set likelihood; ambiguous neighbors are excluded.

    Rows without a positive or a negative provide no discrimination supervision.
    Replacing their logits before logsumexp prevents masked -inf/NaN gradients.
    """
    if temperature <= 0:
        raise ValueError('Contrastive temperature must be positive')
    with torch.autocast(device_type=source.device.type, enabled=False):
        logits = F.normalize(source.float(), dim=-1) @ F.normalize(target.float(), dim=-1).transpose(-1, -2)
        logits = logits / temperature
        values, counts = [], []
        for score, pos, neg in ((logits, positive, negative),
                                (logits.transpose(-1, -2), positive.transpose(-1, -2), negative.transpose(-1, -2))):
            rows = pos.any(-1) & neg.any(-1)
            numerator = torch.where(rows[..., None], score.masked_fill(~pos, -torch.inf), torch.zeros_like(score))
            denominator = torch.where(rows[..., None], score.masked_fill(~(pos | neg), -torch.inf), torch.zeros_like(score))
            value = denominator.logsumexp(-1) - numerator.logsumexp(-1)
            values.append((value * rows).sum())
            counts.append(rows.sum())
        count = torch.stack(counts).sum()
        return torch.stack(values).sum() / count.clamp_min(1), count


def _selected(batch: dict, name: str, part: int, indices: Tensor):
    value = batch[name][:, part]
    return value.gather(1, indices[..., None].expand(-1, -1, value.shape[-1])) if value.ndim == 3 else value.gather(1, indices)


def resampled_view_contrastive(model, encoded: dict, pairs: list, batch: dict, cfg: dict):
    if not isinstance(batch.get('view2'), dict):
        raise ValueError('Resampled contrastive supervision requires an independent batch[view2] dictionary')
    other_batch = batch['view2']
    required = ('points', 'fragment_mask', 'anchor_index', 'canonical_points')
    if any(name not in other_batch for name in required):
        raise ValueError('view2 must include points, masks, reference, and common-frame canonical_points')
    if batch['points'].shape[:2] != other_batch['points'].shape[:2]:
        raise ValueError('Both views must retain the same batch and fragment slots')
    other_encoded = model.encode(other_batch['points'], other_batch['fragment_mask'], other_batch['anchor_index'])
    other_pairs = {(p['i'], p['j']): p for p in model.match(other_encoded, use_scaffold=False)}
    options = cfg.get('repair', {})
    terms, weights = [], []
    positive_pairs = batch['points'].new_zeros(())
    negative_pairs = batch['points'].new_zeros(())
    for pair in pairs:
        other = other_pairs[pair['i'], pair['j']]
        pair_valid = pair['valid'] & other['valid']
        for side, part in (('source', pair['i']), ('target', pair['j'])):
            a, b = pair[f'{side}_indices'], other[f'{side}_indices']
            xyz_a = _selected(batch, 'canonical_points', part, a)
            xyz_b = _selected(other_batch, 'canonical_points', part, b)
            valid_a = pair_valid[:, None].expand_as(a)
            valid_b = pair_valid[:, None].expand_as(b)
            if 'fracture_labels' in batch:
                valid_a = valid_a & (_selected(batch, 'fracture_labels', part, a) >= 0)
            if 'fracture_labels' in other_batch:
                valid_b = valid_b & (_selected(other_batch, 'fracture_labels', part, b) >= 0)
            same_surface = None
            if 'interface_ids' in batch and 'interface_ids' in other_batch:
                ia, ib = _selected(batch, 'interface_ids', part, a), _selected(other_batch, 'interface_ids', part, b)
                same_surface = ia[:, :, None] == ib[:, None, :]
            positive, negative = contrastive_masks(torch.cdist(xyz_a.float(), xyz_b.float()), valid_a, valid_b,
                float(options.get('view_positive_radius', .05)), float(options.get('view_negative_radius', .1)), same_surface)
            term, count = multipositive_contrastive(pair[f'{side}_embedding'], other[f'{side}_embedding'],
                positive, negative, float(options.get('view_temperature', .1)))
            terms.append(term * count)
            weights.append(count)
            positive_pairs = positive_pairs + positive.sum()
            negative_pairs = negative_pairs + negative.sum()
    count = torch.stack(weights).sum()
    loss = torch.stack(terms).sum() / count.clamp_min(1)
    return loss, dict(view_valid_rows=count, view_positive_pairs=positive_pairs, view_negative_pairs=negative_pairs)


@torch.no_grad()
def contact_metrics(encoded: dict, pairs: list, batch: dict, cfg: dict, include_rank: bool = False):
    valid = batch['fragment_mask'].bool()[..., None] & (batch['fracture_labels'] >= 0)
    truth = batch['fracture_labels'] > .5
    prediction = encoded['fracture_logits'] >= 0
    tp = (valid & prediction & truth).sum().float()
    tn = (valid & ~prediction & ~truth).sum().float()
    fp = (valid & prediction & ~truth).sum().float()
    fn = (valid & ~prediction & truth).sum().float()
    result = dict(segmentation_tp=tp, segmentation_tn=tn, segmentation_fp=fp, segmentation_fn=fn,
        segmentation_precision=tp/(tp+fp).clamp_min(1), segmentation_recall=tp/(tp+fn).clamp_min(1),
        segmentation_iou=tp/(tp+fp+fn).clamp_min(1),
        segmentation_balanced_accuracy=.5*(tp/(tp+fn).clamp_min(1)+tn/(tn+fp).clamp_min(1)),
        predicted_fracture_fraction=(valid & prediction).sum()/valid.sum().clamp_min(1))
    total = tp.new_zeros(())
    good = {k: tp.new_zeros(()) for k in (1, 5, 10)}
    uniform = {k: tp.new_zeros(()) for k in (1, 5, 10)}
    embeddings = []
    for pair in pairs:
        positive, _ = contact_geometry(pair, batch, float(cfg.get('train', {}).get('contact_radius', .05)))
        for side, truth_pair in (('source', positive), ('target', positive.transpose(-1, -2))):
            probability = pair[f'{side}_prob'][..., :-1]
            rows = truth_pair.any(-1) & pair['valid'][:, None]
            positives = truth_pair.sum(-1).float()
            total += rows.sum()
            for k in good:
                size = min(k, probability.shape[-1])
                chosen = probability.topk(size, dim=-1).indices
                good[k] += (truth_pair.gather(-1, chosen).any(-1) & rows).sum()
                # Exact uniform sampling without replacement for each row.
                no_hit = torch.ones_like(positives)
                for position in range(size):
                    no_hit *= ((probability.shape[-1] - positives - position) / (probability.shape[-1] - position)).clamp_min(0)
                uniform[k] += ((1 - no_hit) * rows).sum()
            embedding = pair[f'{side}_embedding'][pair['valid']].reshape(-1, pair[f'{side}_embedding'].shape[-1])
            if len(embedding):
                embeddings.append(embedding)
    result['retrieval_valid_rows'] = total
    for k in good:
        result[f'retrieval_top{k}'] = good[k] / total.clamp_min(1)
        result[f'retrieval_top{k}_hits'] = good[k]
        result[f'retrieval_uniform_top{k}'] = uniform[k] / total.clamp_min(1)
    result['matching_top1_recall'] = result['retrieval_top1']
    result['matching_top1_chance'] = result['retrieval_uniform_top1']
    if embeddings:
        features = torch.cat(embeddings).float()
        result['matcher_embedding_variance'] = features.var(0, unbiased=False).mean()
        sampled = features[torch.linspace(0, len(features)-1, min(256, len(features)), device=features.device).long()]
        result['matcher_embedding_cross_cosine'] = ((sampled @ sampled.T).sum() - sampled.square().sum()) / max(1, len(sampled)*(len(sampled)-1))
        if include_rank:
            singular = torch.linalg.svdvals(sampled - sampled.mean(0))
            mass = singular / singular.sum().clamp_min(1e-12)
            result['matcher_embedding_effective_rank'] = torch.where(singular.sum() > 0,
                (-(mass * mass.clamp_min(1e-12).log()).sum()).exp(), singular.new_zeros(()))
    return result


def compute_losses(model, batch: dict, stage: int, cfg: dict):
    if stage not in (1, 2):
        raise ValueError('Repair Stage 3 performs evaluation and has no training objective')
    weights, options = cfg.get('loss', {}), cfg.get('train', {})
    if stage == 2:
        # Explicitly detach even when called without configure_stage by a probe.
        with torch.no_grad():
            encoded = model.encode(batch['points'], batch['fragment_mask'], batch['anchor_index'])
        prediction = model.scaffold(encoded, batch['sdf_queries'])
        result = field_losses(prediction['distance'], prediction['log_scale'], batch['sdf_values'], model.field.truncation)
        groups = batch.get('sdf_query_group')
        if groups is None or groups.shape != batch['sdf_values'].shape or ((groups < 0) | (groups > 3)).any():
            raise ValueError('Stage 2 requires sdf_query_group with labels 0 surface, 1 inside, 2 outside, 3 space')
        error = (prediction['distance'].float() - batch['sdf_values'].float().clamp(-model.field.truncation, model.field.truncation)).abs()
        sigma = prediction['log_scale'].float().exp()
        calibration = error.detach() / sigma + prediction['log_scale'].float()
        for group, name in enumerate(('surface', 'inside', 'outside', 'space')):
            mask = groups == group
            result[f'sdf_{name}_l1'] = masked_mean(error, mask)
            result[f'sdf_{name}_count'] = mask.sum()
            result[f'sdf_{name}_mean_error'] = result[f'sdf_{name}_l1']
            result[f'sdf_{name}_mean_sigma'] = masked_mean(sigma, mask)
            result[f'sdf_{name}_uncertainty_mae'] = masked_mean((sigma - error.detach()).abs(), mask)
            result[f'sdf_{name}_calibration'] = masked_mean(calibration, mask)
        result['near_surface_sdf_l1'] = masked_mean(error, groups < 3)
        result['signed_near_surface_sdf_l1'] = masked_mean(error, (groups == 1) | (groups == 2))
        zero_error = batch['sdf_values'].float().clamp(-model.field.truncation, model.field.truncation).abs()
        result['sdf_zero_l1'] = zero_error.mean()
        result['near_surface_zero_l1'] = masked_mean(zero_error, groups < 3)
        result['signed_near_surface_zero_l1'] = masked_mean(zero_error, (groups == 1) | (groups == 2))
        sign_valid = (groups != 0) & (batch['sdf_values'] != 0)
        result['sdf_sign_error'] = masked_mean(((prediction['distance'] >= 0) != (batch['sdf_values'] >= 0)).float(), sign_valid)
        result['sdf_sign_count'] = sign_valid.sum()
        loss = float(weights.get('sdf', 1.)) * result['sdf_l1'] + float(weights.get('calibration', .01)) * result['sdf_calibration']
        return {'loss': loss, **{name: value.detach() for name, value in result.items()}}
    encoded = model.encode(batch['points'], batch['fragment_mask'], batch['anchor_index'])
    pairs = model.match(encoded, use_scaffold=False)
    result = {}
    matching, positives = correspondence_loss(pairs, batch, float(options.get('contact_radius', .05)),
        localization_sigma=float(options.get('contact_sigma', .01)),
        localization_weight=float(weights.get('matching_localization', 1.)), diagnostics=result)
    segmentation = segmentation_loss(encoded['fracture_logits'], batch['fracture_labels'], batch['fragment_mask'].bool())
    supervision = cfg.get('repair', {}).get('view_supervision', 'resampled_contrastive')
    if supervision == 'existing':
        consistency = view_consistency(model, encoded, batch)
    elif supervision == 'resampled_contrastive':
        consistency, measurements = resampled_view_contrastive(model, encoded, pairs, batch, cfg)
        result.update(measurements)
    else:
        raise ValueError('Unknown repair.view_supervision')
    result.update(matching=matching, positive_contacts=positives, segmentation=segmentation, consistency=consistency)
    result.update(contact_metrics(encoded, pairs, batch, cfg, include_rank=bool(cfg.get('repair', {}).get('rank_metrics', False))))
    loss = (float(weights.get('matching', 1.)) * matching + float(weights.get('segmentation', 1.)) * segmentation
            + float(weights.get('view_consistency', .1)) * consistency)
    return {'loss': loss, **{name: value.detach() for name, value in result.items()}}


__all__ = ['compute_losses', 'contact_metrics', 'contrastive_masks', 'multipositive_contrastive', 'resampled_view_contrastive']
