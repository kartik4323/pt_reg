from __future__ import annotations

import unittest

import torch

from reassembly.losses import (compute_losses, contact_targets, directional_contact_loss,
                               field_losses, transformed_view)
from reassembly.model import ReassemblyModel, configure_stage


def tiny_config() -> dict:
    return {"model": {"dim": 16, "sample_counts": [16, 10, 6], "neighbors": 4,
                      "heads": 4, "attention_layers": 2, "truncation": 0.1},
            "train": {"contact_radius": 0.2}}


def batch_fixture() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    canonical = torch.randn(2, 3, 24, 3) * 0.12
    canonical[:, 1] = canonical[:, 0] + torch.randn(2, 24, 3) * 0.002
    points = transformed_view(canonical)
    points -= points.mean(2, keepdim=True)
    interfaces = torch.full((2, 3, 24), -1, dtype=torch.long)
    interfaces[:, :2, :12] = 0
    interfaces[0, 1:, 12:18] = 1
    mask = torch.tensor([[True, True, True], [True, True, False]])
    return {"points": points, "points_view2": transformed_view(points),
            "canonical_points": canonical, "interface_ids": interfaces,
            "fracture_labels": (interfaces >= 0).float(), "fragment_mask": mask,
            "anchor_index": torch.tensor([0, 1]),
            "sdf_queries": torch.randn(2, 32, 3) * 0.25,
            "sdf_values": torch.randn(2, 32).clamp(-0.1, 0.1)}


class ReassemblyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_shapes_masks_and_field_bounds(self):
        cfg, batch = tiny_config(), batch_fixture()
        model = ReassemblyModel(cfg).eval()
        encoded = model.encode(batch["points"], batch["fragment_mask"], batch["anchor_index"])
        self.assertEqual(encoded["token_features"].shape, (2, 3, 6, 16))
        self.assertEqual(encoded["descriptor"].shape, (2, 3, 24, 16))
        self.assertTrue(torch.allclose(encoded["descriptor"].norm(dim=-1), torch.ones(2, 3, 24), atol=1e-5))
        field = model.scaffold(encoded, batch["sdf_queries"])
        self.assertTrue((field["distance"].abs() <= 0.1).all())
        self.assertTrue(((field["log_scale"] >= -6) & (field["log_scale"] <= -2)).all())
        for pair in model.match(encoded):
            self.assertTrue((pair["weights"].sum(-1) <= 1).all())
            self.assertTrue((pair["weights"].sum(-2) <= 1).all())
            if pair["j"] == 2:
                self.assertEqual(pair["weights"][1].sum().item(), 0)
        changed = batch["points"].clone()
        changed[1, 2] = torch.randn_like(changed[1, 2]) * 100
        replacement = model.encode(changed, batch["fragment_mask"], batch["anchor_index"])
        field2 = model.scaffold(replacement, batch["sdf_queries"])
        torch.testing.assert_close(field["distance"][1], field2["distance"][1])

    def test_point_permutation_equivariance(self):
        cfg, batch = tiny_config(), batch_fixture()
        model = ReassemblyModel(cfg).eval()
        perm = torch.randperm(24)
        encoded = model.encode(batch["points"], batch["fragment_mask"], batch["anchor_index"])
        permuted = model.encode(batch["points"][:, :, perm], batch["fragment_mask"], batch["anchor_index"])
        torch.testing.assert_close(permuted["descriptor"], encoded["descriptor"][:, :, perm], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(model.scaffold(encoded, batch["sdf_queries"])["distance"],
                                   model.scaffold(permuted, batch["sdf_queries"])["distance"], atol=1e-6, rtol=1e-4)

    def test_all_stages_backward_and_freeze(self):
        cfg, batch = tiny_config(), batch_fixture()
        for stage in (1, 2, 3):
            with self.subTest(stage=stage):
                model = ReassemblyModel(cfg).train()
                configure_stage(model, stage)
                losses = compute_losses(model, batch, stage, cfg)
                self.assertTrue(torch.isfinite(losses["loss"]))
                losses["loss"].backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.matcher.parameters()))
                if stage == 3:
                    self.assertFalse(model.encoder.training)
                    self.assertFalse(model.field.training)
                    self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
                    self.assertTrue(all(p.grad is None for p in model.field.parameters()))
                else:
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()))
                if stage == 2:
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.field.parameters()))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_dustbin_removes_mass_instead_of_renormalizing(self):
        cfg, batch = tiny_config(), batch_fixture()
        model = ReassemblyModel(cfg).eval()
        encoded = model.encode(batch["points"], batch["fragment_mask"], batch["anchor_index"])
        baseline = model.match(encoded, use_scaffold=False)[0]
        with torch.no_grad():
            model.matcher.dustbin.fill_(30)
        unmatched = model.match(encoded, use_scaffold=False)[0]
        self.assertLess(unmatched["weights"].sum().item(), baseline["weights"].sum().item() * 1e-6)
        self.assertTrue((unmatched["source_prob"][..., -1] > 0.999).all())
        weak = dict(encoded, fracture_logits=torch.full_like(encoded["fracture_logits"], -20))
        self.assertLess(model.match(weak, use_scaffold=False)[0]["weights"].sum().item(), unmatched["weights"].sum().item())

    def test_prior_conditioning_uses_predicted_field_and_reference_geometry(self):
        cfg, batch = tiny_config(), batch_fixture()
        model = ReassemblyModel(cfg).eval()
        encoded = model.encode(batch["points"], batch["fragment_mask"], batch["anchor_index"])
        base = model.match(encoded, True)[0]["weights"]
        no_prior = model.match(encoded, False)[0]["weights"]
        with torch.no_grad():
            model.field.decoder[-1].bias.add_(torch.tensor([3.0, -2.0]))
        changed = model.match(encoded, True)[0]["weights"]
        self.assertGreater((base - changed).abs().max().item(), 1e-8)
        torch.testing.assert_close(no_prior, model.match(encoded, False)[0]["weights"])
        field = model.scaffold(encoded, batch["sdf_queries"])["distance"]
        shifted = dict(encoded, token_xyz=encoded["token_xyz"].clone())
        # Reference token positions affect queries; the other fragments' raw
        # local coordinates are deliberately never field position keys.
        shifted["token_xyz"][0, 1] += 10
        torch.testing.assert_close(field[0], model.scaffold(shifted, batch["sdf_queries"])["distance"][0])
        shifted["token_xyz"][0, 0] += 0.5
        self.assertGreater((field[0] - model.scaffold(shifted, batch["sdf_queries"])["distance"][0]).abs().max().item(), 1e-8)

    def test_multi_positive_targets_are_local_and_interface_specific(self):
        pair = {"i": 0, "j": 1, "source_indices": torch.tensor([[0, 1]]),
                "target_indices": torch.tensor([[0, 1, 2, 3]]), "valid": torch.tensor([True])}
        canonical = torch.zeros(1, 2, 4, 3)
        canonical[0, 1, :, 0] = torch.tensor([0.01, 0.02, 0.8, 0.01])
        labels = torch.tensor([[[0, -1, -1, -1], [0, 0, 0, 1]]])
        target = contact_targets(pair, {"canonical_points": canonical, "interface_ids": labels}, 0.05)
        self.assertEqual(target[0, 0].tolist(), [True, True, False, False])
        self.assertFalse(target[0, 1].any())
        probs = torch.tensor([[[0.4, 0.4, 0.05, 0.05, 0.1], [0.01, 0.01, 0.01, 0.01, 0.96]]], requires_grad=True)
        loss = directional_contact_loss(probs, target, pair["valid"])
        self.assertAlmostEqual(loss.item(), float((-torch.log(torch.tensor(0.8)) - torch.log(torch.tensor(0.96))) / 2), places=6)

    def test_uncertainty_cannot_reduce_distance_gradient(self):
        target = torch.tensor([[0.05, -0.06]])
        gradients = []
        for scale in (-6.0, -2.0):
            distance = torch.tensor([[0.0, 0.0]], requires_grad=True)
            log_scale = torch.full_like(distance, scale, requires_grad=True)
            losses = field_losses(distance, log_scale, target, 0.1)
            (losses["sdf_l1"] + 0.01 * losses["sdf_calibration"]).backward()
            gradients.append(distance.grad.clone())
            self.assertIsNotNone(log_scale.grad)
        torch.testing.assert_close(gradients[0], gradients[1])

    def test_contact_only_stage3_excludes_field(self):
        cfg, batch = tiny_config(), batch_fixture()
        cfg["train"]["condition"] = "contact_only"
        model = ReassemblyModel(cfg).train()
        configure_stage(model, 3)
        model.scaffold = lambda *args, **kwargs: self.fail("contact-only matcher queried the scaffold")
        compute_losses(model, batch, 3, cfg)["loss"].backward()

    def test_explicit_prior_ablation_bypasses_field(self):
        cfg, batch = tiny_config(), batch_fixture()
        model = ReassemblyModel(cfg).eval()
        encoded = model.encode(batch["points"], batch["fragment_mask"], batch["anchor_index"])
        model.scaffold = lambda *args, **kwargs: self.fail("explicit prior queried predicted scaffold")
        override = torch.randn(2, 17, 5)
        weights = model.match(encoded, True, prior_override=override)[0]["weights"]
        self.assertTrue(torch.isfinite(weights).all())


if __name__ == "__main__":
    unittest.main()
