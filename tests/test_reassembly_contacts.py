"""Regressions for the failed fixed-example assembly, not only loss reduction."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from reassembly.checkpoints import load_checkpoint
from reassembly.losses import directional_contact_loss, localization_loss, local_contact_distribution
from reassembly.model import contact_point_indices, farthest_point_indices
from reassembly.solver import solve_from_matches


class ContactRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_small_contact_survives_context_bottleneck_and_permutation(self):
        torch.manual_seed(2)
        exterior = torch.nn.functional.normalize(torch.randn(1019, 3), dim=-1)
        contact = torch.randn(5, 3) * .001
        xyz = torch.cat([exterior, contact])[None]
        logits = torch.cat([torch.full((1019,), -10.), torch.full((5,), 10.)])[None]
        sparse = farthest_point_indices(xyz, 64)
        self.assertLess(int((sparse >= 1019).sum()), 3)
        selected = contact_point_indices(xyz, logits, 256)
        self.assertEqual(len(set(selected[0].tolist())), 256)
        self.assertEqual(int((selected >= 1019).sum()), 5)
        permutation = torch.randperm(1024)
        shuffled = contact_point_indices(xyz[:, permutation], logits[:, permutation], 256)
        torch.testing.assert_close(xyz[0, selected[0]], xyz[0, permutation[shuffled[0]]])

    def test_no_predicted_contact_falls_back_to_global_coverage(self):
        torch.manual_seed(3)
        xyz = torch.randn(2, 39, 3)
        selected = contact_point_indices(xyz, torch.full((2, 39), -10.), 24)
        torch.testing.assert_close(selected, farthest_point_indices(xyz, 24))

    def test_zero_membership_loss_does_not_hide_imprecise_matches(self):
        positive = torch.ones(1, 1, 2, dtype=torch.bool)
        distance = torch.tensor([[[.002, .045]]])
        valid = torch.tensor([True])
        good = torch.tensor([[[.999, .001, 0.]]])
        bad = torch.tensor([[[.001, .999, 0.]]], requires_grad=True)
        self.assertEqual(float(directional_contact_loss(good, positive, valid)), 0)
        self.assertEqual(float(directional_contact_loss(bad, positive, valid).detach()), 0)
        precise = localization_loss(good, positive, distance, valid, .01)
        imprecise = localization_loss(bad, positive, distance, valid, .01)
        self.assertGreater(float((imprecise - precise).detach()), 6)
        imprecise.backward()
        self.assertTrue(torch.isfinite(bad.grad).all())
        self.assertLess(float(bad.grad[0, 0, 0]), float(bad.grad[0, 0, 1]))

    def test_multi_positive_localization_accepts_sampling_uncertainty_and_empty_rows(self):
        positive = torch.tensor([[[True, True, False], [False, False, False]]])
        distance = torch.tensor([[[.007, .009, .001], [.001, .003, .002]]])
        target = local_contact_distribution(positive, distance, .01)
        self.assertGreater(float(target[0, 0, 1]), .3)
        self.assertEqual(float(target[0, 0, 2]), 0)
        self.assertEqual(float(target[0, 1].sum()), 0)
        probabilities = torch.cat([target, torch.tensor([[[0.], [1.]]])], -1).requires_grad_()
        loss = localization_loss(probabilities, positive, distance, torch.tensor([True]), .01)
        self.assertAlmostEqual(float(loss.detach()), 0, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(probabilities.grad).all())

    def test_exterior_count_does_not_dilute_small_contact_confidence(self):
        rng = np.random.default_rng(9)
        contact = rng.normal(size=(5, 3)) * .1
        results = []
        for extra in (0, 59, 251):
            points = np.concatenate([contact, rng.normal(size=(extra, 3))])
            weights = np.zeros((len(points), len(points)))
            weights[:5, :5] = np.eye(5) * .9
            pair = dict(i=0, j=1, source_xyz=points, target_xyz=points, weights=weights)
            results.append(solve_from_matches([pair], 2))
        self.assertTrue(all(r['status'] == 'ok' for r in results))
        np.testing.assert_allclose([r['confidence'] for r in results], .9, atol=1e-8)

    def test_underflowed_contact_mass_cannot_make_localization_negative(self):
        probabilities = torch.tensor([[[1e-20, 1e-20, 1.]]], requires_grad=True)
        loss = localization_loss(probabilities, torch.ones(1, 1, 2, dtype=torch.bool),
                                 torch.tensor([[[.001, .045]]]), torch.tensor([True]), .01)
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(probabilities.grad).all())

    def test_retained_dustbin_mass_and_weak_bridge_remain_low_confidence(self):
        points = np.random.default_rng(2).normal(size=(12, 3))
        pairs = []
        for i, j, mass in ((0, 1, .99), (1, 2, .03)):
            pairs.append(dict(i=i, j=j, source_xyz=points, target_xyz=points, weights=np.eye(12) * mass,
                              source_matchability=np.full(12, mass), target_matchability=np.full(12, mass)))
        result = solve_from_matches(pairs, 3)
        self.assertEqual(result['status'], 'low_confidence')
        self.assertLessEqual(result['confidence'], .03)

    def test_old_contact_architecture_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pt'
            torch.save({'schema_version': 2, 'architecture': 'coarse-scaffold-reassembly-v2'}, path)
            with self.assertRaisesRegex(ValueError, 'Legacy or external'):
                load_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
