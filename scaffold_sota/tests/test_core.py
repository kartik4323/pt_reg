"""Focused contracts: no label leakage, correct frames, failures and prior swaps."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scaffold_sota.data import StudyDataset, sanitize_input, observation_fingerprint
from scaffold_sota.geometry import fixed_query_bank, transform_points
from scaffold_sota.evaluation import evaluate_prediction, aggregate, compare_conditions
from scaffold_sota.evaluation.refinement import refine_prediction
from scaffold_sota.priors import FrozenPrior, export_prior, ContentControl, WrongPrior, GroundTruthPrior
from scaffold_sota.priors import GenericPrior, PerturbedPrior, CachedPrior, export_token_cache, token_fingerprint
from scaffold_sota.priors.diagnostics import perturb_oracle_prediction, evaluate_pose_stability
from scaffold_sota.priors.architectures import ShapeField as CompatibleField
from reassembly.model import GeometryEncoder, ShapeField
from reassembly.prepare import manifest_fingerprint


def sample_fixture():
    rng = np.random.default_rng(5)
    points = rng.normal(size=(3, 16, 3)).astype(np.float32)*.1
    points -= points.mean(axis=1, keepdims=True)
    points[2] = 0
    rotations = np.repeat(np.eye(3)[None], 3, axis=0).astype(np.float32)
    translations = np.array([[0, 0, 0], [.3, 0, 0], [0, 0, 0]], dtype=np.float32)
    result = dict(points=torch.from_numpy(points), fragment_mask=torch.tensor([True, True, False]),
                  anchor_index=torch.tensor(0), canonical_points=torch.from_numpy(transform_points(points, rotations, translations)),
                  rotations_gt=torch.from_numpy(rotations), translations_gt=torch.from_numpy(translations),
                  source_id="source-a", pattern_id="pattern-a", split="test")
    result["observation_id"] = observation_fingerprint(result)
    return result


def manifest_fixture(root):
    rng = np.random.default_rng(3)
    source_path, pattern_path = root/"source.npz", root/"pattern.npz"
    np.savez(source_path, sdf_near_mask=np.array([True, False]*16),
             sdf_queries=rng.normal(size=(32, 3)), sdf_values=np.linspace(-.1, .1, 32),
             target_points=rng.normal(size=(32, 3)))
    np.savez(pattern_path, points=rng.normal(size=(2, 32, 3)), fracture_labels=np.zeros((2, 32)),
             interface_ids=np.full((2, 32), -1))
    hashed = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    document = dict(schema_version=2, sdf_convention="negative_inside", seed=42, config={},
                    sources=[dict(source_id="source-a", path=source_path.name, sha256=hashed(source_path))],
                    patterns=[dict(source_id="source-a", pattern_id="pattern-a", path=pattern_path.name,
                                   sha256=hashed(pattern_path), split="train", pieces=2, band="hard", cut_family="random")])
    document["fingerprint"] = manifest_fingerprint(document)
    path = root/"manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class FakePrior:
    def __init__(self):
        self.queries = fixed_query_bank(32)
        self.provenance = {"kind": "predicted", "diagnostic_only": False}
    def sample_tokens(self, sample):
        return dict(query_xyz=self.queries.clone(), distance=(self.queries[:, 0]-.1).clamp(-.1, .1),
                    log_scale=torch.linspace(-6, -2, len(self.queries)), valid=torch.ones(len(self.queries), dtype=torch.bool),
                    provenance=self.provenance)
    def exterior_probabilities(self, sample):
        return torch.ones_like(sample["points"][..., 0])


class CoreContracts(unittest.TestCase):
    def test_observation_whitelist_does_not_touch_labels(self):
        sample = sample_fixture()
        class Poison:
            def __array__(self):
                raise AssertionError("Ground truth accessed at inference")
        for key in ("rotations_gt", "translations_gt", "canonical_points", "sdf_queries", "source_mesh_path"):
            sample[key] = Poison()
        self.assertEqual(set(sanitize_input(sample)), {"points", "fragment_mask", "anchor_index"})
        self.assertEqual(observation_fingerprint(sample), sample["observation_id"])

    def test_dataset_all_bands_fresh_deterministic_and_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = manifest_fixture(root)
            original = {path.name: path.read_bytes() for path in root.iterdir()}
            dataset = StudyDataset(manifest, "train", points_per_fragment=8, sdf_queries=8)
            first = dataset[0]
            self.assertEqual(first["band"], "hard")  # At step zero: no inherited easy-only curriculum.
            self.assertEqual(first["observation_id"], dataset[0]["observation_id"])
            dataset.set_step(1)
            self.assertNotEqual(first["observation_id"], dataset[0]["observation_id"])
            dataset.set_step(0)
            self.assertEqual(first["observation_id"], dataset[0]["observation_id"])
            for name, content in original.items():
                self.assertEqual((root/name).read_bytes(), content)
            with (root/"pattern.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                StudyDataset(manifest, "train", points_per_fragment=8, sdf_queries=8)

    def test_common_rigid_gauge_and_failure_denominators(self):
        sample = sample_fixture()
        r, t = sample["rotations_gt"].numpy(), sample["translations_gt"].numpy()
        angle = .7
        global_r = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0,0,1]])
        global_t = np.array([4., -2., 1.])
        prediction = dict(rotations=global_r[None]@r, translations=t@global_r.T+global_t)
        row = evaluate_prediction(sample, prediction)
        self.assertTrue(row["success"])
        self.assertLess(row["whole_chamfer"], 1e-6)
        failure_sample = {**sample, "pattern_id": "pattern-b"}
        failure = evaluate_prediction(failure_sample, dict(rotations=3, translations=4))
        self.assertTrue(failure["failed"])
        expected = [row, failure, {**row, "pattern_id": "missing"}]
        summary = aggregate([row, failure], expected)
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["missing_count"], 1)
        self.assertAlmostEqual(summary["success_rate"], 1/3)
        self.assertAlmostEqual(summary["failure_rate"], 2/3)

    def test_paired_comparison_requires_full_observations(self):
        sample = sample_fixture()
        good = evaluate_prediction(sample, dict(rotations=sample["rotations_gt"], translations=sample["translations_gt"]))
        bad = evaluate_prediction(sample, None)
        report = compare_conditions([good], [bad], bootstrap_samples=20)
        self.assertEqual(report["source_count"], 1)
        self.assertEqual(report["harm_rate_previously_successful"], 1)
        self.assertEqual(report["success_delta"]["mean"], -1)
        with self.assertRaisesRegex(ValueError, "identical"):
            compare_conditions([good], [])

    def test_source_macro_does_not_overweight_sources_with_more_cuts(self):
        baseline, treatment = [], []
        for source, pattern, before, after in [("a", "a1", True, True), ("a", "a2", False, True), ("b", "b1", False, False)]:
            base = dict(source_id=source, pattern_id=pattern, observation_id="fixed", success=before,
                        failed=False, part_accuracy=float(before), moving_part_accuracy=float(before))
            baseline.append(base)
            treatment.append({**base, "success": after, "part_accuracy": float(after), "moving_part_accuracy": float(after)})
        self.assertAlmostEqual(aggregate(baseline)["source_macro_success_rate"], .25)
        comparison = compare_conditions(baseline, treatment, bootstrap_samples=50)
        self.assertAlmostEqual(comparison["success_delta"]["mean"], 1/3)
        self.assertAlmostEqual(comparison["success_source_macro_delta"]["mean"], .25)

    def test_content_interventions_hold_support_and_uncertainty_mass(self):
        provider, sample = FakePrior(), sample_fixture()
        original = provider.sample_tokens(sample)
        shuffled = ContentControl(provider, "shuffled_uncertainty").sample_tokens(sample)
        torch.testing.assert_close(original["query_xyz"], shuffled["query_xyz"])
        torch.testing.assert_close(original["distance"], shuffled["distance"])
        torch.testing.assert_close(original["log_scale"].sort()[0], shuffled["log_scale"].sort()[0])
        with self.assertRaisesRegex(ValueError, "different source"):
            WrongPrior(provider, sample).sample_tokens(sample)
        with self.assertRaisesRegex(ValueError, "diagnostic_only"):
            GroundTruthPrior()

    def test_frozen_export_uses_no_solver_and_matches_original_field(self):
        torch.manual_seed(4)
        cfg = {"model": dict(dim=8, heads=2, sample_counts=[8,4,3], neighbors=3, attention_layers=1)}
        encoder, field = GeometryEncoder(cfg["model"]).eval(), ShapeField(cfg["model"]).eval()
        weights = {"encoder."+key: value for key, value in encoder.state_dict().items()}
        weights.update({"field."+key: value for key, value in field.state_dict().items()})
        weights["matcher.must_not_be_exported"] = torch.ones(1)
        state = dict(architecture="coarse-scaffold-reassembly-v2.1-local-contacts", schema_version=2,
                     stage=2, step=5, dataset_fingerprint="dataset", cfg=cfg, model=weights)
        with tempfile.TemporaryDirectory() as temporary:
            source, exported = Path(temporary)/"source.pt", Path(temporary)/"export.pt"
            torch.save(state, source)
            export_prior(source, exported)
            with patch("reassembly.model.ReassemblyModel", side_effect=AssertionError("Stage 3 model constructed")):
                prior = FrozenPrior(exported, query_count=32)
            sample = sample_fixture()
            tokens = prior.sample_tokens(sample)
            with torch.no_grad():
                encoded = encoder(sample["points"][None])
                encoded.update(fragment_mask=sample["fragment_mask"][None], anchor_index=sample["anchor_index"][None])
                expected = field(encoded, tokens["query_xyz"][None])
            torch.testing.assert_close(tokens["distance"], expected["distance"][0])
            prior.train()
            self.assertFalse(prior.training)
            self.assertFalse(any(parameter.requires_grad for parameter in prior.parameters()))
            query = prior.queries[:3].clone().requires_grad_()
            prior.field(sample, query)["distance"].sum().backward()
            self.assertIsNotNone(query.grad)
            self.assertFalse(any(parameter.grad is not None for parameter in prior.parameters()))
            state["stage"] = 3
            torch.save(state, source)
            historical = Path(temporary)/"historical-v2.pt"
            metadata = export_prior(source, historical)
            self.assertEqual(metadata["source_checkpoint_stage"], 3)
            self.assertEqual(set(torch.load(historical, weights_only=False)["weights"]), {"encoder", "field"})
            with self.assertRaisesRegex(ValueError, "verified training source"):
                FrozenPrior(historical, expected_fingerprint="dataset")
            manifest = manifest_fixture(Path(temporary))
            state["dataset_fingerprint"] = json.loads(manifest.read_text())["fingerprint"]
            torch.save(state, source)
            verified = Path(temporary)/"verified.pt"
            export_prior(source, verified, training_manifest=manifest)
            prior = FrozenPrior(verified, expected_fingerprint=state["dataset_fingerprint"])
            with self.assertRaisesRegex(ValueError, "used to train"):
                prior.sample_tokens(sample)

    def test_legacy_transformer_constructor_and_v3_field_state_compatibility(self):
        from reassembly.repair.model import InvariantEncoder, AdaptedShapeField
        from scaffold_sota.priors.architectures import AdaptedShapeField as CompatibleAdaptedField
        cfg = dict(dim=8, heads=2, sample_counts=[8,4,3], neighbors=3, attention_layers=1)
        original_constructor = torch.nn.TransformerEncoder
        def legacy_constructor(encoder_layer, num_layers, norm=None):
            return original_constructor(encoder_layer, num_layers, norm=norm, enable_nested_tensor=False)
        expected = AdaptedShapeField(cfg).eval()
        with patch("torch.nn.TransformerEncoder", legacy_constructor):
            compatible = CompatibleAdaptedField(cfg).eval()
        compatible.load_state_dict(expected.state_dict(), strict=True)
        sample = sample_fixture()
        with torch.no_grad():
            encoded = InvariantEncoder(cfg).eval()(sample["points"][None])
            encoded.update(fragment_mask=sample["fragment_mask"][None], anchor_index=sample["anchor_index"][None])
            queries = fixed_query_bank(16)[None]
            for key in ("distance", "log_scale"):
                torch.testing.assert_close(expected(encoded, queries)[key], compatible(encoded, queries)[key])

    def test_refinement_is_rigid_bounded_and_never_reads_labels(self):
        sample, provider = sample_fixture(), FakePrior()
        prediction = dict(rotations=sample["rotations_gt"].clone(), translations=sample["translations_gt"].clone(),
                          exterior_probabilities=torch.ones(3,16))
        clean = refine_prediction(sample, prediction, provider, steps=2, points_per_fragment=8)
        poisoned = {**sample, "canonical_points": None, "rotations_gt": None, "translations_gt": None,
                    "source_mesh_path": "must-not-be-read", "fracture_labels": None}
        replay = refine_prediction(poisoned, prediction, provider, steps=2, points_per_fragment=8)
        np.testing.assert_allclose(clean["rotations"], replay["rotations"])
        np.testing.assert_allclose(clean["translations"], replay["translations"])
        np.testing.assert_allclose(np.linalg.det(clean["rotations"]), 1, atol=1e-5)
        self.assertLessEqual(np.linalg.norm(clean["translations"][1]-.3*np.array([1,0,0])), .050001)
        np.testing.assert_allclose(clean["translations"][0], 0)
        skipped = refine_prediction(sample, {key:value for key,value in prediction.items() if key != "exterior_probabilities"}, provider)
        self.assertEqual(skipped["refinement"]["status"], "skipped")
        np.testing.assert_allclose(skipped["translations"], prediction["translations"])
        row = evaluate_prediction(sample, skipped)
        self.assertTrue(row["success"])
        self.assertEqual(aggregate([row])["refinement_skipped_count"], 1)

    def test_cache_binds_exact_points_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = StudyDataset(manifest_fixture(root), "train", points_per_fragment=8, sdf_queries=8, fixed=True)
            provider = FakePrior()
            provider.provenance.update(training_source_ids=["declared-source"], source_identity_status="unverified")
            export_token_cache(provider, dataset, root/"cache")
            with self.assertRaisesRegex(ValueError, "verified prior training source"):
                CachedPrior(root/"cache", expected_fingerprint=dataset.fingerprint)
            provider.provenance["source_identity_status"] = "verified_manifest_training_partition"
            export_token_cache(provider, dataset, root/"verified-cache")
            CachedPrior(root/"verified-cache", expected_fingerprint=dataset.fingerprint)
            cached = CachedPrior(root/"cache")
            sample = dataset[0]
            self.assertEqual(token_fingerprint(cached.sample_tokens(sample)), token_fingerprint(provider.sample_tokens(sample)))
            torch.testing.assert_close(cached.exterior_probabilities(sample), provider.exterior_probabilities(sample))
            altered = {**sample, "points": sample["points"]+.001}
            with self.assertRaisesRegex(ValueError, "observation hash"):
                cached.sample_tokens(altered)
            document = json.loads((root/"cache/cache.json").read_text())
            with (root/"cache"/document["records"][0]["path"]).open("ab") as stream:
                stream.write(b"modified")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                cached.sample_tokens(sample)

    def test_generic_is_streamed_and_stress_keeps_support(self):
        refs = []
        class Donors:
            split, fingerprint = "train", "manifest"
            def __len__(self):
                return 12
            def __getitem__(self, index):
                # A list-of-all-samples implementation trips this memory contract.
                if sum(reference() is not None for reference in refs) > 1:
                    raise AssertionError("Generic prior retained previous source observations")
                sample = sample_fixture()
                sample.update(split="train", source_id="source-%d" % index)
                refs.append(weakref.ref(sample["points"]))
                return sample
        generic = GenericPrior.from_dataset(FakePrior(), Donors())
        self.assertEqual(len(generic.provenance["training_source_ids"]), 12)
        provider, sample = FakePrior(), sample_fixture()
        original = provider.sample_tokens(sample)
        perturbed = PerturbedPrior(provider, "distance_noise", seed=99).sample_tokens(sample)
        replay = PerturbedPrior(provider, "distance_noise", seed=99).sample_tokens(sample)
        torch.testing.assert_close(perturbed["distance"], replay["distance"])
        torch.testing.assert_close(perturbed["query_xyz"], original["query_xyz"])
        torch.testing.assert_close(perturbed["log_scale"], original["log_scale"])
        self.assertNotEqual(token_fingerprint(perturbed), token_fingerprint(original))

    def test_exact_five_fifteen_degree_diagnostics_are_explicit(self):
        sample = sample_fixture()
        original_rotations = sample["rotations_gt"].clone()
        original_translations = sample["translations_gt"].clone()
        perturbations = {}
        for angle in (0, 5, 15):
            prediction = perturb_oracle_prediction(sample, angle, seed=4)
            row = evaluate_prediction(sample, prediction)
            self.assertAlmostEqual(row["rotation_deg"][0], angle, places=3)
            magnitude = {0: 0., 5: .02, 15: .05}[angle]
            self.assertAlmostEqual(row["translation_error"][0], magnitude, places=6)
            self.assertEqual(prediction["translation_perturbation"], magnitude)
            self.assertEqual(prediction["translation_units"], "normalized_anchor_frame")
            np.testing.assert_array_equal(prediction["rotations"][0], sample["rotations_gt"][0])
            np.testing.assert_array_equal(prediction["translations"][0], sample["translations_gt"][0])
            np.testing.assert_array_equal(prediction["translations"][2], sample["translations_gt"][2])
            repeated = perturb_oracle_prediction(sample, angle, seed=4)
            np.testing.assert_array_equal(repeated["translations"], prediction["translations"])
            perturbations[angle] = prediction["translations"][1]-sample["translations_gt"][1].numpy()
            self.assertTrue(prediction["diagnostic_only"])
        np.testing.assert_allclose(perturbations[5]/.02, perturbations[15]/.05, atol=1e-6)
        torch.testing.assert_close(original_rotations, sample["rotations_gt"])
        torch.testing.assert_close(original_translations, sample["translations_gt"])
        with self.assertRaisesRegex(ValueError, "explicit translation"):
            perturb_oracle_prediction(sample, 10.)
        custom = perturb_oracle_prediction(sample, 10., translation_magnitude=.03)
        self.assertAlmostEqual(evaluate_prediction(sample, custom)["translation_error"][0], .03, places=6)
        class Samples:
            fingerprint = "manifest"
            def __len__(self):
                return 1
            def __getitem__(self, index):
                return sample
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "diagnostic_only"):
                evaluate_pose_stability(FakePrior(), Samples(), Path(temporary)/"forbidden")
            report = evaluate_pose_stability(FakePrior(), Samples(), Path(temporary)/"diagnostic",
                                              diagnostic_only=True, conditions=("A1", "A2", "A3"), steps=1)
            self.assertTrue(report["oracle_pose_initialization"])
            self.assertEqual(len(report["groups"]), 9)
            self.assertEqual(report["translation_perturbations"], {"0deg": 0., "5deg": .02, "15deg": .05})
            self.assertEqual(report["groups"]["5deg_A3"]["translation_perturbation"], .02)


if __name__ == "__main__":
    unittest.main()
