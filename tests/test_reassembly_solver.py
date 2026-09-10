"""Geometry and solver contracts, independent of trained model quality."""
import unittest

import numpy as np
import torch

from reassembly.geometry import (
    deterministic_fps, export_transforms, normalize_fragments, so3_exp, weighted_kabsch,
)
from reassembly.solver import ScaffoldGrid, solve_assembly, solve_from_matches


def fixture(count=3, seed=3):
    rng = np.random.default_rng(seed)
    rotations = np.stack([so3_exp(rng.normal(size=3)) for _ in range(count)])
    translations = rng.normal(size=(count, 3)) * 0.15
    interfaces, matches = {}, []
    for i, j in ((0, 1), (1, 2))[:count - 1]:
        world = rng.normal(size=(24, 3)) * np.array([0.2, 0.15, 0.04]) + [0.2 * i, 0, 0]
        local_i = (world - translations[i]) @ rotations[i]
        local_j = (world - translations[j]) @ rotations[j]
        interfaces[(i, j)] = world
        matches.append({"i": i, "j": j, "source_xyz": local_i,
                        "target_xyz": local_j, "weights": np.eye(len(world))})
    return rotations, translations, matches


class GeometryTests(unittest.TestCase):
    def test_oracle_transform_accuracy_and_proper_rotation(self):
        rng = np.random.default_rng(4)
        source = rng.normal(size=(41, 3))
        rotation, translation = so3_exp(np.array([0.7, -1.1, 0.9])), np.array([0.2, -0.4, 0.3])
        fit = weighted_kabsch(source, source @ rotation.T + translation, rng.uniform(0.1, 1, len(source)))
        self.assertTrue(fit["valid"])
        np.testing.assert_allclose(fit["rotation"], rotation, atol=1e-12)
        np.testing.assert_allclose(fit["translation"], translation, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(fit["rotation"]), 1)

    def test_planar_is_valid_collinear_is_not(self):
        planar = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 2, 0.]])
        rotation = so3_exp(np.array([0.3, 0.5, -1]))
        self.assertTrue(weighted_kabsch(planar, planar @ rotation.T)["valid"])
        line = np.stack([np.arange(8), np.zeros(8), np.zeros(8)], -1)
        fit = weighted_kabsch(line, line @ rotation.T)
        self.assertFalse(fit["valid"])
        self.assertIsNone(fit["rotation"])
        self.assertIn("collinear", fit["reason"])

    def test_mirror_fit_keeps_determinant_positive(self):
        points = np.random.default_rng(2).normal(size=(20, 3))
        reflected = points * [-1, 1, 1]
        fit = weighted_kabsch(points, reflected)
        self.assertTrue(fit["valid"])
        self.assertAlmostEqual(np.linalg.det(fit["rotation"]), 1)
        self.assertGreater(fit["rms"], 0.01)

    def test_dustbin_confidence_mass_not_discarded(self):
        points = np.random.default_rng(1).normal(size=(10, 3))
        self.assertTrue(weighted_kabsch(points, points, np.ones(10), min_mass=.05)["valid"])
        self.assertFalse(weighted_kabsch(points, points, np.full(10, 1e-8), min_mass=.05)["valid"])

    def test_shared_scale_anchor_and_input_permutation(self):
        rng = np.random.default_rng(5)
        fragments = [rng.normal(size=(51, 3)) * 0.4 + [1, 2, 3], rng.normal(size=(67, 3)) * 1.4 + [-3, 4, 8]]
        result = normalize_fragments(fragments, num_points=32)
        permuted = normalize_fragments([p[rng.permutation(len(p))] for p in fragments], num_points=32)
        self.assertEqual(result["anchor_index"], 1)
        self.assertEqual(result["points"].shape, (3, 32, 3))
        self.assertEqual(result["fragment_mask"].tolist(), [True, True, False])
        np.testing.assert_allclose(result["points"], permuted["points"], atol=1e-7)
        expected_scale = sum(np.linalg.norm(p - p.mean(0), axis=1).max() for p in fragments)
        self.assertAlmostEqual(result["scale"], expected_scale)
        scaled = normalize_fragments([p * 7 for p in fragments], num_points=32)
        np.testing.assert_allclose(result["points"], scaled["points"], atol=1e-7)
        self.assertAlmostEqual(scaled["scale"], result["scale"] * 7)

    def test_fps_padding_and_small_inputs(self):
        points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        indices = deterministic_fps(points, 8)
        self.assertEqual(len(indices), 8)
        self.assertEqual(len(set(indices[:3])), 3)
        with self.assertRaises(ValueError):
            normalize_fragments([points])
        with self.assertRaises(ValueError):
            normalize_fragments([points, np.zeros_like(points)])

    def test_export_original_coordinates_reference_identity(self):
        rng = np.random.default_rng(8)
        shared = rng.normal(size=(29, 3))
        original_rotation = so3_exp(np.array([0.2, -0.7, 0.4]))
        original_translation = np.array([4., -5., 2.])
        fragments = [shared, (shared - original_translation) @ original_rotation]
        normalization = normalize_fragments(fragments, num_points=29)
        # Equal RMS tie selects fragment zero despite independent rigid pose.
        self.assertEqual(normalization["anchor_index"], 0)
        local = [(p - center) / normalization["scale"] for p, center in zip(fragments, normalization["centroids"])]
        fit = weighted_kabsch(local[1], local[0])
        rotations = np.stack([np.eye(3), fit["rotation"]])
        translations = np.stack([np.zeros(3), fit["translation"]])
        exported = export_transforms(rotations, translations, normalization)
        np.testing.assert_array_equal(exported["transforms"][0], np.eye(4))
        np.testing.assert_allclose(exported["rotations"][1], original_rotation, atol=1e-12)
        np.testing.assert_allclose(exported["translations"][1], original_translation, atol=1e-12)
        np.testing.assert_allclose(exported["aligned_fragments"][1], shared, atol=1e-12)
        distances_before = np.linalg.norm(fragments[1][1:] - fragments[1][0], axis=1)
        distances_after = np.linalg.norm(exported["aligned_fragments"][1][1:] - exported["aligned_fragments"][1][0], axis=1)
        np.testing.assert_allclose(distances_before, distances_after, atol=1e-12)


class FieldTests(unittest.TestCase):
    def test_signed_linear_field_values_gradients_and_boundary(self):
        axes = np.linspace(-1, 1, 13)
        x, y, z = np.meshgrid(axes, axes, axes, indexing="ij")
        field = ScaffoldGrid(x + 2*y - z, np.full_like(x, .01), np.array([[-1]*3, [1]*3]))
        query = np.array([[.2, -.3, .4], [-1, .5, 1], [.1, 0, 0]])
        values, confidence, gradient = field.sample(query)
        np.testing.assert_allclose(values, query @ np.array([1, 2, -1]), atol=1e-12)
        np.testing.assert_allclose(gradient, np.tile([1, 2, -1], (3, 1)), atol=1e-12)
        np.testing.assert_allclose(confidence, .5)
        values, confidence, gradient = field.sample(np.array([[2., 0, 0]]))
        self.assertGreater(values[0], 1)
        self.assertEqual(confidence[0], .05)
        np.testing.assert_allclose(gradient[0], [1, 0, 0])

    def test_uncertainty_weakens_confidence(self):
        values = np.zeros((4, 4, 4))
        precise = ScaffoldGrid(values, np.full_like(values, .003), np.array([[-1]*3, [1]*3]))
        uncertain = ScaffoldGrid(values, np.full_like(values, .12), np.array([[-1]*3, [1]*3]))
        self.assertGreater(precise.sample(np.zeros((1, 3)))[1][0], uncertain.sample(np.zeros((1, 3)))[1][0])


class SolverTests(unittest.TestCase):
    def test_two_and_three_fragment_oracle_all_anchor_choices(self):
        for count in (2, 3):
            rotations, translations, matches = fixture(count)
            for anchor in range(count):
                result = solve_from_matches(matches, count, anchor)
                self.assertEqual(result["status"], "ok")
                expected_r = np.einsum("ij,fjk->fik", rotations[anchor].T, rotations)
                expected_t = (translations - translations[anchor]) @ rotations[anchor]
                np.testing.assert_allclose(result["rotations"], expected_r, atol=1e-8)
                np.testing.assert_allclose(result["translations"], expected_t, atol=1e-8)
                np.testing.assert_array_equal(result["rotations"][anchor], np.eye(3))
                np.testing.assert_array_equal(result["translations"][anchor], np.zeros(3))

    def test_all_three_spanning_trees_are_enumerated(self):
        rotations, translations, matches = fixture(3)
        world = np.random.default_rng(13).normal(size=(17, 3)) * .1
        matches.append({"i": 0, "j": 2, "source_xyz": (world-translations[0]) @ rotations[0],
                        "target_xyz": (world-translations[2]) @ rotations[2], "weights": np.eye(len(world))})
        result = solve_from_matches(matches, 3)
        self.assertEqual(result["diagnostics"]["hypotheses"], 3)

    def test_noncontacting_pair_and_point_permutations(self):
        _, _, matches = fixture(3)
        rng = np.random.default_rng(20)
        permuted = []
        for pair in matches:
            source_order = rng.permutation(len(pair["source_xyz"]))
            target_order = rng.permutation(len(pair["target_xyz"]))
            permuted.append({**pair, "source_xyz": pair["source_xyz"][source_order],
                             "target_xyz": pair["target_xyz"][target_order],
                             "weights": pair["weights"][source_order][:, target_order]})
        permuted.append({"i": 0, "j": 2, "source_xyz": rng.normal(size=(20, 3)),
                         "target_xyz": rng.normal(size=(16, 3)), "weights": np.zeros((20, 16))})
        baseline = solve_from_matches(matches, 3)
        result = solve_from_matches(permuted, 3)
        np.testing.assert_allclose(result["rotations"], baseline["rotations"], atol=1e-8)
        np.testing.assert_allclose(result["translations"], baseline["translations"], atol=1e-8)

    def test_explicit_failed_status_for_weak_and_disconnected_matches(self):
        _, _, matches = fixture(3)
        for case in (matches[:1], [{**pair, "weights": pair["weights"]*1e-8} for pair in matches], []):
            result = solve_from_matches(case, 3)
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["rotations"])
            self.assertIsNone(result["translations"])
            self.assertEqual(result["confidence"], 0)

    def test_weak_erroneous_edge_preserves_relative_confidence(self):
        rotations, translations, matches = fixture(3)
        baseline = solve_from_matches(matches, 3)
        world = np.random.default_rng(18).normal(size=(24, 3)) * .1
        matches.append({"i": 0, "j": 2,
                        "source_xyz": (world-translations[0]) @ rotations[0],
                        "target_xyz": (world-translations[2]) @ rotations[2] + [.2, -.1, 0],
                        "weights": np.eye(len(world)) * .01})
        result = solve_from_matches(matches, 3)
        self.assertLess(np.max(np.abs(result["translations"]-baseline["translations"])), .001)
        self.assertLess(np.max(np.abs(result["rotations"]-baseline["rotations"])), .005)
        detail = result["diagnostics"]
        self.assertAlmostEqual(detail["pair_contact_mass"][-1] / detail["pair_contact_mass"][0], .01)
        self.assertLessEqual(detail["score"], detail["pre_refinement_score"] + 1e-12)
        self.assertIn("pre_refinement_contact_rms", detail)
        self.assertIn("pre_refinement_prior_score", detail)

    def test_collinear_pair_fails_planar_pair_succeeds(self):
        rng = np.random.default_rng(4)
        for collinear in (True, False):
            source = rng.normal(size=(20, 3)) * .1
            source[:, 2] = 0
            if collinear:
                source[:, 1] = 0
            target = source @ so3_exp(np.array([.5, -.8, .2])).T + [.2, .4, 0]
            result = solve_from_matches([{"i": 0, "j": 1, "source_xyz": source,
                                          "target_xyz": target, "weights": np.eye(len(source))}], 2)
            self.assertEqual(result["status"], "failed" if collinear else "ok")

    def test_bad_prior_cannot_overrule_reliable_contacts(self):
        _, _, matches = fixture(2)
        baseline = solve_from_matches(matches, 2)
        # Deliberately wrong plane pulls all points in +x. Contact residuals
        # retain greater influence and the fixed reference never moves.
        axes = np.linspace(-2, 2, 32)
        x, _, _ = np.meshgrid(axes, axes, axes, indexing="ij")
        field = ScaffoldGrid(x-.5, np.full_like(x, .003), np.array([[-2]*3, [2]*3]))
        exterior = [matches[0]["source_xyz"], matches[0]["target_xyz"]]
        result = solve_from_matches(matches, 2, exterior_points=exterior, field=field)
        self.assertLess(np.linalg.norm(result["translations"]-baseline["translations"]), .02)
        self.assertLess(result["diagnostics"]["contact_rms"], .01)
        np.testing.assert_array_equal(result["rotations"][0], np.eye(3))

    def test_scaffold_not_applied_to_fracture_only_points(self):
        _, _, matches = fixture(2)
        baseline = solve_from_matches(matches, 2)
        axes = np.linspace(-1, 1, 8)
        x, _, _ = np.meshgrid(axes, axes, axes, indexing="ij")
        field = ScaffoldGrid(x+.5, np.zeros_like(x), np.array([[-1]*3, [1]*3]))
        exterior = [matches[0]["source_xyz"], matches[0]["target_xyz"]]
        result = solve_from_matches(matches, 2, exterior_points=exterior,
                                    exterior_weights=[np.zeros(len(p)) for p in exterior], field=field)
        np.testing.assert_allclose(result["translations"], baseline["translations"], atol=1e-10)
        self.assertEqual(result["diagnostics"]["prior_score"], 0)


class InferenceIsolationTests(unittest.TestCase):
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used_scaffold = []
            self.prior_overrides = []
            self.field_calls = 0
            _, _, self.matches = fixture(2)

        def encode(self, points, fragment_mask, anchor_index):
            return {"token_xyz": points, "fracture_logits": torch.zeros(points.shape[:-1]),
                    "anchor_index": anchor_index}

        def match(self, encoded, use_scaffold=True, prior_override=None):
            self.used_scaffold.append(use_scaffold)
            self.prior_overrides.append(prior_override)
            return [{**pair, "source_xyz": torch.tensor(pair["source_xyz"])[None],
                     "target_xyz": torch.tensor(pair["target_xyz"])[None],
                     "weights": torch.tensor(pair["weights"])[None], "valid": torch.tensor([True])}
                    for pair in self.matches]

        def scaffold(self, encoded, query):
            self.field_calls += 1
            return {"distance": query[..., 0], "log_scale": torch.full_like(query[..., 0], -5)}

    def test_contact_only_does_not_query_scaffold_or_read_labels(self):
        model = self.Stub()
        model.train()
        batch = {"points": torch.zeros(1, 3, 24, 3), "fragment_mask": torch.tensor([[True, True, False]]),
                 "anchor_index": torch.tensor([0]), "labels": object(), "gt_rotation": object()}
        result = solve_assembly(model, batch, {}, condition="contact_only")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(model.used_scaffold, [False])
        self.assertEqual(model.field_calls, 0)
        self.assertIsNone(result["scaffold"])
        self.assertTrue(model.training)

    def test_gt_cannot_silently_fall_back_or_enter_predicted_inference(self):
        model = self.Stub()
        batch = {"points": torch.zeros(1, 3, 24, 3), "fragment_mask": torch.tensor([[True, True, False]]),
                 "anchor_index": torch.tensor([0])}
        with self.assertRaisesRegex(ValueError, "explicit"):
            solve_assembly(model, batch, {}, condition="gt")
        with self.assertRaisesRegex(ValueError, "only"):
            solve_assembly(model, batch, {}, field_override=lambda q: q[:, 0])

    def test_gt_and_perturbed_change_matcher_conditioning(self):
        model = self.Stub()
        batch = {"points": torch.zeros(1, 3, 24, 3), "fragment_mask": torch.tensor([[True, True, False]]),
                 "anchor_index": torch.tensor([0])}
        cfg = {"solver": {"resolution": 8, "field_chunk": 512}}
        solve_assembly(model, batch, cfg, condition="gt", field_override=lambda q: np.full(len(q), -.04))
        self.assertEqual(model.field_calls, 0)
        np.testing.assert_allclose(model.prior_overrides[-1][..., 3].numpy(), -.04, atol=1e-7)
        solve_assembly(model, batch, cfg, condition="predicted")
        self.assertIsNone(model.prior_overrides[-1])
        solve_assembly(model, batch, cfg, condition="perturbed")
        self.assertIsNotNone(model.prior_overrides[-1])
        self.assertEqual(model.used_scaffold, [True, True, True])


if __name__ == "__main__":
    unittest.main()
