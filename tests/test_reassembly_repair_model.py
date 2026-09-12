"""Repair adapters/objectives are checked without training a dataset."""
import unittest

import torch

from reassembly.losses import transformed_view
from reassembly.model import ReassemblyModel
from reassembly.repair.model import RepairModel, configure_stage
from reassembly.repair.losses import (compute_losses, contact_metrics, contrastive_masks,
                                       multipositive_contrastive, resampled_view_contrastive)


def config(geometry='revised', supervision='resampled_contrastive'):
    return dict(model=dict(dim=16, sample_counts=[20, 12, 8], neighbors=6,
                           contact_points=24, heads=4, attention_layers=2, truncation=.1),
                train=dict(contact_radius=.05, contact_sigma=.01),
                repair=dict(geometry_variant=geometry, view_supervision=supervision))


def fixture():
    torch.manual_seed(174)
    canonical = torch.randn(2, 3, 40, 3) * .2
    canonical[:, 1] = canonical[:, 0] + torch.randn(2, 40, 3) * .001
    ids = torch.full((2, 3, 40), -1, dtype=torch.long)
    ids[:, :2, :25] = 0
    ids[0, 1:, 25:35] = 1
    mask = torch.tensor([[True, True, True], [True, True, False]])
    def observations(value):
        value = transformed_view(value)
        return value - value.mean(2, keepdim=True)
    points = observations(canonical)
    second = canonical + torch.randn_like(canonical) * .002
    permutation = torch.randperm(40)
    batch = dict(points=points, canonical_points=canonical, interface_ids=ids,
        fracture_labels=(ids >= 0).float(), fragment_mask=mask, anchor_index=torch.tensor([0, 1]),
        points_view2=transformed_view(points), sdf_queries=torch.randn(2, 32, 3) * .3,
        sdf_values=torch.randn(2, 32).clamp(-.1, .1), sdf_query_group=torch.arange(4).repeat(2, 8))
    batch['view2'] = dict(points=observations(second)[:, :, permutation],
        canonical_points=second[:, :, permutation], interface_ids=ids[:, :, permutation],
        fracture_labels=(ids[:, :, permutation] >= 0).float(), fragment_mask=mask,
        anchor_index=torch.tensor([0, 1]))
    return batch


class RepairModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_existing_fresh_seed_matches_legacy_states_and_probabilities(self):
        batch = fixture()
        for supervision in ('existing', 'resampled_contrastive'):
            with self.subTest(supervision=supervision):
                cfg = config('existing', supervision)
                torch.manual_seed(713)
                original = ReassemblyModel(cfg).eval()
                legacy_rng = torch.get_rng_state().clone()
                torch.manual_seed(713)
                repaired = RepairModel(cfg).eval()
                torch.testing.assert_close(torch.get_rng_state(), legacy_rng, atol=0, rtol=0)
                # A true fresh-initialization control: no state_dict copying.
                for name in ('encoder', 'matcher'):
                    first_state = getattr(original, name).state_dict()
                    second_state = getattr(repaired, name).state_dict()
                    self.assertEqual(set(first_state), set(second_state))
                    for key in first_state:
                        torch.testing.assert_close(first_state[key], second_state[key], atol=0, rtol=0)
                a = original.encode(batch['points'], batch['fragment_mask'], batch['anchor_index'])
                b = repaired.encode(batch['points'], batch['fragment_mask'], batch['anchor_index'])
                for first, second in zip(original.match(a, use_scaffold=False), repaired.match(b)):
                    for key in ('weights', 'source_prob', 'target_prob', 'source_indices', 'target_indices'):
                        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
                    self.assertTrue(second['source_embedding'].requires_grad)

    def test_revised_rotation_and_permutation_consistency(self):
        cfg, batch = config(), fixture()
        model = RepairModel(cfg).eval()
        original = model.encode(batch['points'], batch['fragment_mask'], batch['anchor_index'])
        permutation = torch.randperm(40)
        changed_points = transformed_view(batch['points'])[:, :, permutation]
        changed = model.encode(changed_points, batch['fragment_mask'], batch['anchor_index'])
        for key in ('descriptor', 'point_features', 'fracture_logits'):
            torch.testing.assert_close(original[key][:, :, permutation], changed[key], atol=2e-5, rtol=2e-4)
        for first, second in zip(model.match(original), model.match(changed)):
            torch.testing.assert_close(first['source_embedding'], second['source_embedding'], atol=3e-5, rtol=3e-4)
            torch.testing.assert_close(first['target_embedding'], second['target_embedding'], atol=3e-5, rtol=3e-4)
            torch.testing.assert_close(first['weights'], second['weights'], atol=1e-5, rtol=3e-4)

    def test_padding_cannot_change_valid_pair_or_field(self):
        cfg, batch = config(), fixture()
        model = RepairModel(cfg).eval()
        first = model.encode(batch['points'], batch['fragment_mask'], batch['anchor_index'])
        points = batch['points'].clone()
        points[1, 2] = torch.randn_like(points[1, 2]) * 10
        second = model.encode(points, batch['fragment_mask'], batch['anchor_index'])
        a, b = model.match(first), model.match(second)
        torch.testing.assert_close(a[0]['weights'][1], b[0]['weights'][1])
        for pair in b[1:]:
            self.assertEqual(pair['weights'][1].sum().item(), 0.)
        torch.testing.assert_close(model.scaffold(first, batch['sdf_queries'])['distance'][1],
                                   model.scaffold(second, batch['sdf_queries'])['distance'][1])

    def test_context_cache_applies_adapter_once_and_keeps_reference_coordinates(self):
        cfg, batch = config(), fixture()
        model = RepairModel(cfg).eval()
        encoded = model.encode(batch['points'], batch['fragment_mask'], batch['anchor_index'])
        saved = {k: v.clone() for k, v in encoded.items()}
        context = model.field.context(encoded)
        direct = model.scaffold(encoded, batch['sdf_queries'])
        cached = model.field(encoded, batch['sdf_queries'], context=context)
        torch.testing.assert_close(direct['distance'], cached['distance'])
        for key in encoded:
            torch.testing.assert_close(encoded[key], saved[key], atol=0, rtol=0)
        torch.testing.assert_close(context[2], encoded['token_xyz'][torch.arange(2), batch['anchor_index']])

    def test_contrastive_masks_ignore_gray_zone_invalid_points_and_surface_boundaries(self):
        distance = torch.tensor([[[.01, .04, .075, .1, .11, .2]]])
        source_valid, target_valid = torch.ones(1, 1, dtype=torch.bool), torch.tensor([[True, True, True, True, True, False]])
        same = torch.tensor([[[True, False, True, True, True, True]]])
        pos, neg = contrastive_masks(distance, source_valid, target_valid, same_surface=same)
        self.assertEqual(pos.tolist(), [[[True, False, False, False, False, False]]])
        self.assertEqual(neg.tolist(), [[[False, False, False, False, True, False]]])

    def test_contrastive_objective_discriminates_and_handles_empty_masks(self):
        a = torch.eye(4)[None].requires_grad_()
        b = torch.eye(4)[None].requires_grad_()
        positive = torch.eye(4, dtype=torch.bool)[None]
        negative = ~positive
        good, count = multipositive_contrastive(a, b, positive, negative)
        wrong, _ = multipositive_contrastive(a, b.roll(1, 1), positive, negative)
        self.assertLess(good.item(), wrong.item())
        self.assertEqual(count.item(), 8)
        wrong.backward()
        self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all())
        self.assertGreater(a.grad.abs().sum().item(), 0.)
        for which in ('positive', 'negative'):
            a.grad = None
            pos = torch.zeros_like(positive) if which == 'positive' else positive
            neg = torch.zeros_like(negative) if which == 'negative' else negative
            empty, count = multipositive_contrastive(a, b, pos, neg)
            empty.backward()
            self.assertEqual(count.item(), 0)
            self.assertEqual(empty.item(), 0.)
            self.assertTrue(torch.isfinite(a.grad).all())

    def test_both_geometries_and_both_supervisions_backpropagate(self):
        for geometry in ('existing', 'revised'):
            for supervision in ('existing', 'resampled_contrastive'):
                with self.subTest(geometry=geometry, supervision=supervision):
                    cfg, batch = config(geometry, supervision), fixture()
                    model = RepairModel(cfg).train()
                    configure_stage(model, 1)
                    losses = compute_losses(model, batch, 1, cfg)
                    self.assertTrue(torch.isfinite(losses['loss']))
                    losses['loss'].backward()
                    for module in (model.encoder, model.matcher):
                        gradients = [p.grad for p in module.parameters() if p.grad is not None]
                        self.assertTrue(gradients)
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)
                    self.assertTrue(all(p.grad is None for p in model.field.parameters()))
                    for key in ('matching_top1_recall', 'matching_top1_chance', 'matcher_embedding_variance'):
                        self.assertIn(key, losses)
                        self.assertFalse(losses[key].requires_grad)
                    if supervision == 'resampled_contrastive':
                        self.assertGreater(losses['view_valid_rows'].item(), 0)

    def test_contrastive_gradient_reaches_actual_contextual_embedding(self):
        cfg, batch = config(), fixture()
        model = RepairModel(cfg)
        encoded = model.encode(batch['points'], batch['fragment_mask'], batch['anchor_index'])
        pairs = model.match(encoded)
        pairs[0]['source_embedding'].retain_grad()
        loss, counts = resampled_view_contrastive(model, encoded, pairs, batch, cfg)
        loss.backward()
        self.assertGreater(counts['view_valid_rows'].item(), 0)
        self.assertGreater(pairs[0]['source_embedding'].grad.abs().sum().item(), 0.)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.matcher.blocks.parameters()))

    def test_stage_two_only_trains_field_and_reports_balanced_groups(self):
        cfg, batch = config(), fixture()
        model = RepairModel(cfg).train()
        configure_stage(model, 2)
        self.assertFalse(model.encoder.training or model.matcher.training)
        self.assertTrue(model.field.training and model.field_adapter.training)
        losses = compute_losses(model, batch, 2, cfg)
        losses['loss'].backward()
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.matcher.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.field_adapter.parameters() if p.grad is not None), 0.)
        for group in ('surface', 'inside', 'outside', 'space'):
            self.assertEqual(losses[f'sdf_{group}_count'].item(), 16)
        self.assertIn('near_surface_sdf_l1', losses)
        self.assertIn('signed_near_surface_sdf_l1', losses)
        configure_stage(model, 3)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertFalse(model.encoder.training or model.matcher.training or model.field.training)
        with self.assertRaisesRegex(ValueError, 'no training objective'):
            compute_losses(model, batch, 3, cfg)

    def test_metrics_include_actual_embedding_rank_and_missing_view_fails(self):
        cfg, batch = config(), fixture()
        model = RepairModel(cfg)
        encoded = model.encode(batch['points'], batch['fragment_mask'], batch['anchor_index'])
        metrics = contact_metrics(encoded, model.match(encoded), batch, cfg, include_rank=True)
        self.assertGreater(metrics['matcher_embedding_effective_rank'].item(), 1.)
        del batch['view2']
        with self.assertRaisesRegex(ValueError, 'independent'):
            compute_losses(model, batch, 1, cfg)

    def test_mixed_precision_and_zero_padding_have_finite_gradients(self):
        cfg, batch = config(), fixture()
        batch['points'][1, 2] = 0
        batch['view2']['points'][1, 2] = 0
        model = RepairModel(cfg).train()
        configure_stage(model, 1)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            losses = compute_losses(model, batch, 1, cfg)
        self.assertTrue(torch.isfinite(losses['loss']))
        losses['loss'].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_scaffold_flag_does_not_change_contact_probabilities(self):
        cfg, batch = config(), fixture()
        model = RepairModel(cfg).eval()
        encoded = model.encode(batch['points'], batch['fragment_mask'], batch['anchor_index'])
        for first, second in zip(model.match(encoded, use_scaffold=False), model.match(encoded, use_scaffold=True)):
            torch.testing.assert_close(first['weights'], second['weights'], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'do not override'):
            model.match(encoded, prior_override=torch.zeros(2, 5, 5))


if __name__ == '__main__':
    unittest.main()
