"""Continuous field derivatives and immutable contact-candidate contracts."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import trimesh

from reassembly.geometry import export_transforms, normalize_fragments, so3_exp
from reassembly.solver import solve_from_matches
from reassembly.repair.fields import (
    ContinuousGTField, ContinuousNeuralField, FieldSampler, ShiftedField, diagnostic_grid,
)
from reassembly.repair.assembly import build_candidates, build_candidates_from_matches, candidate_pose_metrics, solve_candidates


def finite_difference(field, points, epsilon=1e-5):
    derivatives = []
    for axis in range(3):
        offset = np.eye(3)[axis]*epsilon
        derivatives.append((field.sample(points+offset)[0]-field.sample(points-offset)[0])/(2*epsilon))
    return np.stack(derivatives, -1)


class LinearField:
    truncation = .1

    def values(self, points, untruncated=False):
        value = np.asarray(points)[:, 0]
        return (value if untruncated else value.clip(-self.truncation, self.truncation)), np.full(len(points), .01)

    def sample(self, points):
        value, _ = self.values(points)
        gradient = np.tile([1., 0., 0.], (len(points), 1))
        gradient[np.abs(np.asarray(points)[:, 0]) >= self.truncation] = 0
        return value, np.full(len(points), .5), gradient


class AnalyticModule(nn.Module):
    truncation = 1.

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.3, -.4, .6]))
        self.calls = []

    def context(self, encoded):
        return encoded['point_features'].mean()*0

    def forward(self, encoded, query, context=None):
        self.calls.append(query.shape[1])
        return dict(distance=query @ self.weight + (0 if context is None else context),
                    log_scale=torch.full(query.shape[:-1], -4., device=query.device))


class AnalyticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.field = AnalyticModule()


def contact_fixture(count=3):
    rng = np.random.default_rng(33)
    rotations = np.stack([so3_exp(rng.normal(size=3)) for _ in range(count)])
    translations = rng.normal(size=(count, 3))*.15
    matches, exterior = [], [[] for _ in range(count)]
    for i, j in [(0, 1), (1, 2)][:count-1]:
        world = rng.normal(size=(24, 3))*[.2, .15, .04]
        a, b = (world-translations[i]) @ rotations[i], (world-translations[j]) @ rotations[j]
        exterior[i].append(a)
        exterior[j].append(b)
        matches.append(dict(i=i, j=j, source_xyz=a, target_xyz=b, weights=np.eye(24)))
    return matches, [np.concatenate(points) for points in exterior]


class ContinuousFieldTests(unittest.TestCase):
    def test_neural_query_derivative_under_no_grad_does_not_mutate_model(self):
        model = AnalyticModel().train()
        model.field.weight.grad = torch.ones_like(model.field.weight)*7
        encoded = dict(point_features=torch.ones(1, 2, 4, 3, requires_grad=True))
        before = model.field.weight.detach().clone()
        with torch.no_grad():
            field = ContinuousNeuralField(model, encoded, chunk_size=2)
            points = np.array([[.1, .05, -.1], [.02, -.1, .04], [-.03, .1, .1]])
            value, confidence, gradient = field.sample(points)
        self.assertIsInstance(field, FieldSampler)
        np.testing.assert_allclose(value, points @ before.numpy(), atol=1e-7)
        np.testing.assert_allclose(gradient, np.tile(before.numpy(), (len(points), 1)), atol=1e-7)
        np.testing.assert_allclose(gradient, finite_difference(field, points, 1e-3), atol=2e-5)
        self.assertTrue(np.all((confidence > 0) & (confidence <= 1)))
        self.assertTrue(model.training)
        torch.testing.assert_close(model.field.weight, before)
        torch.testing.assert_close(model.field.weight.grad, torch.ones_like(before)*7)
        self.assertIsNone(encoded['point_features'].grad)
        self.assertFalse(any(parameter.requires_grad for parameter in field.module.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in field.module.parameters()))
        self.assertLessEqual(max(field.module.calls), 2)

    def test_neural_adapter_inside_inference_mode_and_empty_input(self):
        with torch.inference_mode():
            field = ContinuousNeuralField(AnalyticModel(), dict(point_features=torch.ones(1, 3)))
            _, _, gradient = field.sample(np.array([[.1, 0., 0.]]))
            empty = field.sample(np.empty((0, 3)))
        np.testing.assert_allclose(gradient, [[1.3, -.4, .6]], atol=1e-7)
        self.assertEqual(empty[0].shape, (0,))
        self.assertEqual(empty[2].shape, (0, 3))

    def test_neural_clipped_derivative_matches_returned_value(self):
        field = ContinuousNeuralField(AnalyticModel(), dict(point_features=torch.ones(1, 3)), truncation=.1)
        points = np.array([[.2, 0., 0.], [.02, 0., 0.]])
        value, _, gradient = field.sample(points)
        self.assertAlmostEqual(value[0], .1)
        self.assertGreater(field.raw_values(points)[0][0], .1)
        np.testing.assert_allclose(gradient, finite_difference(field, points, 1e-3), atol=2e-5)

    def test_exact_gt_sign_reference_transform_and_gradient(self):
        rotation, center, scale = so3_exp(np.array([.4, -.2, .7])), np.array([.2, -.3, .1]), 2.7
        field = ContinuousGTField(trimesh.creation.box(), rotation=rotation, center=center, scale=scale, truncation=.2)
        source = np.array([[.6, .1, .03], [.4, .1, .03], [-.7, .05, -.1]])
        query = (source-center) @ rotation.T / scale
        value, _, gradient = field.sample(query)
        np.testing.assert_allclose(value, np.array([.1, -.1, .2])/scale, atol=1e-7)
        np.testing.assert_allclose(gradient, np.array([[1., 0, 0], [1., 0, 0], [-1., 0, 0]]) @ rotation.T, atol=1e-7)
        np.testing.assert_allclose(gradient, finite_difference(field, query, 1e-4), atol=1e-4)
        self.assertTrue(field.diagnostic_only)

    def test_gt_surface_normal_and_truncation(self):
        field = ContinuousGTField(trimesh.creation.box(), truncation=.1)
        value, _, gradient = field.sample(np.array([[.5, .11, .02], [1., .11, .02]]))
        np.testing.assert_allclose(value, [0, .1], atol=1e-7)
        np.testing.assert_allclose(gradient, [[1, 0, 0], [0, 0, 0]], atol=1e-7)
        self.assertAlmostEqual(field.raw_distance(np.array([[1., .11, .02]]))[0], .5)

    def test_grid_before_after_clipping_are_distinct_and_have_correct_derivatives(self):
        progress = []
        field = LinearField()
        before = diagnostic_grid(field, resolution=4, extent=1., chunk_size=7, progress=lambda done, total: progress.append((done, total)))
        after = diagnostic_grid(field, resolution=4, extent=1., truncate_before_interpolation=False)
        query = np.array([[.02, .1, .03], [.3, .1, .03]])
        self.assertAlmostEqual(before.sample(query)[0][0], .006, places=7)
        self.assertAlmostEqual(after.sample(query)[0][0], .02, places=7)
        self.assertAlmostEqual(after.sample(query)[0][1], .1, places=7)
        for adapter in (before, after):
            np.testing.assert_allclose(adapter.sample(query)[2], finite_difference(adapter, query), atol=1e-7)
        self.assertEqual(progress[0], (0, 64))
        self.assertEqual(progress[-1], (64, 64))

    def test_field_perturbation_preserves_confidence_and_correct_gradient(self):
        field = ShiftedField(LinearField(), translation=(.01, .02, 0), offset=.035)
        points = np.array([[.02, .03, .04], [.3, 0., 0.]])
        value, confidence, gradient = field.sample(points)
        np.testing.assert_allclose(value, [.045, .1], atol=1e-7)
        np.testing.assert_allclose(confidence, [.5, .5])
        np.testing.assert_allclose(gradient, finite_difference(field, points), atol=1e-7)


class CandidateCacheTests(unittest.TestCase):
    def test_xyz_builder_preserves_modes_exposes_pairs_and_never_conditions_matching(self):
        matches, _ = contact_fixture(2)

        class FixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.field = nn.Linear(3, 1)
                self.match_calls = 0

            def encode(self, points, mask, anchor):
                return dict(point_xyz=points, fracture_logits=torch.zeros_like(points[..., 0]),
                            fragment_mask=mask, anchor_index=anchor)

            def match(self, encoded, use_scaffold):
                if use_scaffold:
                    raise AssertionError('field must not generate contact candidates')
                self.match_calls += 1
                return [dict(i=m['i'], j=m['j'], valid=torch.tensor([True]),
                             **{key: torch.tensor(m[key])[None] for key in ('source_xyz', 'target_xyz', 'weights')}) for m in matches]

        model = FixedModel().train()
        model.field.eval()
        batch = dict(points=torch.randn(1, 3, 32, 3), fragment_mask=torch.tensor([[True, True, False]]), anchor_index=torch.tensor([0]))
        cache = build_candidates(model, batch, {})
        self.assertEqual(model.match_calls, 1)
        self.assertTrue(model.training)
        self.assertFalse(model.field.training)
        self.assertEqual(len(cache.pairs), 1)
        self.assertIsInstance(cache.pairs[0]['weights'], torch.Tensor)
        self.assertEqual(len(cache.exterior_points), 2)
        np.testing.assert_allclose(cache.exterior_weights[0], .5)
        self.assertEqual(solve_candidates(cache, {})['status'], 'ok')

    def test_candidate_pose_coverage_is_reporting_only(self):
        points = np.random.default_rng(4).normal(size=(12, 3))*.1
        cache = build_candidates_from_matches([dict(i=0, j=1, source_xyz=points, target_xyz=points, weights=np.eye(12))], 2)
        sample = dict(points=torch.tensor(np.stack([points, points])), canonical_points=torch.tensor(np.stack([points, points])),
                      fragment_mask=torch.tensor([True, True]), rotations_gt=torch.eye(3).repeat(2, 1, 1),
                      translations_gt=torch.zeros(2, 3), anchor_index=torch.tensor(0))
        before = solve_candidates(cache, {})
        metrics = candidate_pose_metrics(cache, sample)
        self.assertTrue(metrics['candidate_oracle_success'])
        self.assertGreater(metrics['geometrically_correct_candidates'], 0)
        self.assertEqual(metrics['candidate_count'], len(cache.initial_poses))
        np.testing.assert_allclose(solve_candidates(cache, {})['rotations'], before['rotations'])

    def test_same_candidate_cache_matches_original_solver_for_each_field(self):
        for count in (2, 3):
            matches, exterior = contact_fixture(count)
            weights = [np.ones(len(points)) for points in exterior]
            for anchor in range(count):
                cache = build_candidates_from_matches(matches, count, anchor, exterior, weights, {})
                initial = [(r.copy(), t.copy()) for r, t in cache.poses]
                fingerprint = cache.fingerprint
                self.assertTrue(cache.hypotheses)
                for field in (None, LinearField(), ShiftedField(LinearField())):
                    expected = solve_from_matches(matches, count, anchor, exterior, weights, field, {})
                    with patch('reassembly.solver._pair_candidates', side_effect=AssertionError('candidate regeneration')):
                        actual = solve_candidates(cache, {}, field)
                    self.assertEqual(actual['status'], expected['status'])
                    self.assertAlmostEqual(actual['confidence'], expected['confidence'], places=12)
                    np.testing.assert_allclose(actual['rotations'], expected['rotations'], atol=1e-12)
                    np.testing.assert_allclose(actual['translations'], expected['translations'], atol=1e-12)
                    np.testing.assert_allclose(np.linalg.det(actual['rotations']), 1, atol=1e-10)
                    np.testing.assert_allclose(actual['rotations'][anchor], np.eye(3), atol=1e-12)
                    np.testing.assert_allclose(actual['translations'][anchor], 0, atol=1e-12)
                    self.assertEqual(actual['diagnostics']['candidate_fingerprint'], fingerprint)
                for (r, t), (old_r, old_t) in zip(cache.poses, initial):
                    np.testing.assert_array_equal(r, old_r)
                    np.testing.assert_array_equal(t, old_t)

    def test_degenerate_and_unmatched_return_explicit_failure(self):
        line = np.stack([np.arange(6)*.02, np.zeros(6), np.zeros(6)], -1)
        for source, weights in ((line, np.eye(6)), (np.random.default_rng(3).normal(size=(6, 3)), np.zeros((6, 6)))):
            cache = build_candidates_from_matches([dict(i=0, j=1, source_xyz=source, target_xyz=source, weights=weights)], 2)
            result = solve_candidates(cache, {})
            self.assertEqual(result['status'], 'failed')
            self.assertIsNone(result['rotations'])
            self.assertEqual(result['reason'], 'no_valid_connected_assembly')

    def test_changed_thresholds_cannot_reuse_cached_candidates(self):
        matches, _ = contact_fixture(2)
        cache = build_candidates_from_matches(matches, 2)
        with self.assertRaisesRegex(ValueError, 'thresholds changed'):
            solve_candidates(cache, {'solver': {'min_pair_mass': 0}})

    def test_original_coordinate_export_remains_rigid_and_reference_identity(self):
        rng = np.random.default_rng(9)
        target = rng.normal(size=(41, 3))
        r, t = so3_exp(np.array([.5, -.3, .1])), np.array([3., -4., 2.])
        original = [target, (target-t) @ r]
        norm = normalize_fragments(original, num_points=41)
        centers, scale = norm['centroids'], norm['scale']
        local = [(part-center)/scale for part, center in zip(original, centers)]
        cache = build_candidates_from_matches([dict(i=0, j=1, source_xyz=local[0], target_xyz=local[1], weights=np.eye(41))], 2, norm['anchor_index'])
        result = solve_candidates(cache, {})
        exported = export_transforms(result['rotations'], result['translations'], norm)
        np.testing.assert_allclose(exported['transforms'][norm['anchor_index']], np.eye(4), atol=1e-12)
        np.testing.assert_allclose(exported['aligned_fragments'][0], exported['aligned_fragments'][1], atol=1e-10)
        for before, after in zip(original, exported['aligned_fragments']):
            np.testing.assert_allclose(np.linalg.norm(before-before[0], axis=-1), np.linalg.norm(after-after[0], axis=-1), atol=1e-10)


if __name__ == '__main__':
    unittest.main()
