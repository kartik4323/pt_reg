"""Integration checks for the supported CLI, XYZ inference, metrics, and gates."""
from __future__ import annotations

from collections import namedtuple
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reassembly.checkpoints import save_checkpoint
from reassembly.cli import parser, profile_signature, require_overfit, require_preflight
from reassembly.config import load_config, validate_config
from reassembly.evaluation import aggregate, assembly_metrics, chamfer, infer_fragments, pilot_report
from reassembly.model import ReassemblyModel, configure_stage
from reassembly.resources import GIB, ResourceGuard
from reassembly.training import synthetic_batch


def tiny_config():
    cfg = load_config()
    cfg["model"].update(dim=16, sample_counts=[12, 8, 4], neighbors=4)
    cfg["data"].update(points_per_fragment=20, sdf_queries=24)
    cfg["train"].update(max_updates=2, overfit_updates=2, batch_size=1, grad_accum_steps=1,
                        validation_samples=1, amp=False)
    cfg["solver"].update(resolution=4, field_chunk=16)
    return cfg


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = tiny_config()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def unmatched_checkpoint(self, condition="predicted"):
        cfg = copy.deepcopy(self.cfg)
        cfg["train"]["condition"] = condition
        model = ReassemblyModel(cfg)
        configure_stage(model, 3)
        with torch.no_grad():
            model.matcher.dustbin.fill_(40)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        path = self.root / f"unmatched-{condition}.pt"
        save_checkpoint(path, model, optimizer, cfg, "fixture", 3, 0, run_id="fixture-fresh")
        return path

    def test_xyz_variable_length_two_three_piece_failures_are_explicit(self):
        path = self.unmatched_checkpoint()
        rng = np.random.default_rng(41)
        fragments = [rng.normal(size=(23, 3)) * 0.2 + [1, 2, 3],
                     rng.normal(size=(37, 3)) * 0.1 + [-1, 0, 2],
                     rng.normal(size=(29, 3)) * 0.15 + [3, 1, 0]]
        originals = [p.copy() for p in fragments]
        for count in (2, 3):
            result = infer_fragments(fragments[:count], path, self.cfg, torch.device("cpu"))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["reason"], "no_valid_connected_assembly")
            self.assertIsNone(result["rotations"])
            self.assertIsNone(result["translations"])
            self.assertIsNone(result["transforms"])
            self.assertIsNone(result["aligned_fragments"])
            self.assertEqual(result["convention"], "x_aligned = x @ R.T + t")
            self.assertTrue(0 <= result["anchor_index"] < count)
            self.assertEqual(result["scaffold"]["distance"].shape, (4, 4, 4))
        for original, fragment in zip(originals, fragments):
            np.testing.assert_array_equal(original, fragment)

    def test_xyz_inference_rejects_oracle_legacy_and_condition_mismatch(self):
        fragments = [np.random.default_rng(i).normal(size=(20, 3)) for i in range(2)]
        path = self.unmatched_checkpoint()
        for condition in ("gt", "perturbed"):
            with self.assertRaisesRegex(ValueError, "never oracle"):
                infer_fragments(fragments, path, self.cfg, torch.device("cpu"), condition)
        with self.assertRaisesRegex(ValueError, "trained for predicted"):
            infer_fragments(fragments, path, self.cfg, torch.device("cpu"), "contact_only")
        contact = self.unmatched_checkpoint("contact_only")
        result = infer_fragments(fragments, contact, self.cfg, torch.device("cpu"), "contact_only")
        self.assertIsNone(result["scaffold"])
        legacy = self.root / "legacy.pt"
        torch.save({"model": {}}, legacy)
        with self.assertRaisesRegex(ValueError, "Legacy or external"):
            infer_fragments(fragments, legacy, self.cfg, torch.device("cpu"))

    def test_metrics_known_transforms_and_failed_denominator(self):
        batch = synthetic_batch(self.cfg, 1, torch.device("cpu"))
        sample = {key: value[0] for key, value in batch.items()}
        result = {"rotations": sample["rotations_gt"].numpy(),
                  "translations": sample["translations_gt"].numpy(),
                  "status": "ok", "confidence": 1.0,
                  "diagnostics": {"contact_rms": 0.0}}
        metrics = assembly_metrics(sample, result, threshold=1e-5)
        self.assertTrue(metrics["success"])
        self.assertLess(metrics["whole_chamfer"], 1e-6)
        np.testing.assert_allclose(metrics["rotation_deg"], 0, atol=1e-5)
        np.testing.assert_allclose(metrics["translation_error"], 0, atol=1e-5)
        self.assertAlmostEqual(chamfer([[0, 0, 0]], [[0.25, 0, 0]]), 0.25)
        failed = assembly_metrics(sample, {"rotations": None, "status": "failed", "confidence": 0.0}, 1e-5)
        summary = aggregate([metrics, failed])
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["failure_rate"], 0.5)
        self.assertAlmostEqual(summary["whole_chamfer"]["mean"], metrics["whole_chamfer"])
        self.assertIsNone(aggregate([])["success_rate"])

    def test_cli_parser_exposes_one_workflow_and_bounds_stage(self):
        cli = parser()
        args = cli.parse_args(["infer", "--checkpoint", "fresh.pt", "--fragment", "a.npy", "--fragment", "b.npy", "--device", "cpu"])
        self.assertEqual(args.command, "infer")
        self.assertEqual(args.fragment, [Path("a.npy"), Path("b.npy")])
        self.assertEqual(args.condition, "predicted")
        args = cli.parse_args(["train", "--manifest", "manifest.json", "--stage", "2", "--run-dir", "stage2", "--initialize-from", "stage1/best.pt", "--preflight-report", "preflight.json"])
        self.assertEqual(args.stage, 2)
        self.assertEqual(args.initialize_from, Path("stage1/best.pt"))
        for command in ("acquire", "prepare", "preflight", "train", "evaluate", "infer", "report"):
            self.assertIn(command, cli.format_help())

    def test_preflight_profile_is_bound_to_shapes_precision_and_dataset(self):
        manifest = self.write("manifest.json", {"fingerprint": "fixture"})
        report = {"kind": "preflight", "profile_signature": profile_signature(self.cfg),
                  "dataset_fingerprint": "fixture", "correctness_passed": True,
                  "status": "cpu_verified_cuda_unmeasured"}
        path = self.write("preflight.json", report)
        require_preflight(path, self.cfg, manifest, torch.device("cpu"))
        changed = copy.deepcopy(self.cfg)
        changed["data"]["points_per_fragment"] += 1
        with self.assertRaisesRegex(ValueError, "configuration"):
            require_preflight(path, changed, manifest, torch.device("cpu"))
        wrong = self.write("other-manifest.json", {"fingerprint": "unrelated"})
        with self.assertRaisesRegex(ValueError, "Prepared-data"):
            require_preflight(path, self.cfg, wrong, torch.device("cpu"))
        report.update(status="passed", attempts=[{"gpu_name": "RTX A5000", "max_reserved_bytes": 20 * GIB}])
        path = self.write("cuda.json", report)
        with patch("torch.cuda.get_device_name", return_value="RTX A5000"):
            with self.assertRaisesRegex(ValueError, "strict VRAM"):
                require_preflight(path, self.cfg, manifest, torch.device("cuda"))

    def test_overfit_report_must_pass_on_the_same_dataset(self):
        manifest = self.write("manifest.json", {"fingerprint": "fixture"})
        report = {"kind": "evaluation", "purpose": "overfit", "fixed_fit_passed": True,
                  "dataset_fingerprint": "fixture", "config": self.cfg, "condition": "predicted"}
        path = self.write("overfit.json", report)
        require_overfit(path, manifest)
        report["dataset_fingerprint"] = "unrelated"
        path = self.write("wrong-overfit.json", report)
        with self.assertRaisesRegex(ValueError, "matching 16-pattern"):
            require_overfit(path, manifest)
        report.update(dataset_fingerprint="fixture", condition="gt")
        with self.assertRaisesRegex(ValueError, "matching 16-pattern"):
            require_overfit(self.write("oracle-overfit.json", report), manifest, self.cfg)
        report["condition"] = "predicted"
        report["config"] = copy.deepcopy(self.cfg)
        report["config"]["model"]["dim"] = 32
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            require_overfit(self.write("different-model.json", report), manifest, self.cfg)

    def comparison_reports(self):
        lineage = {str(stage): {"updates": 2000, "checkpoint_step": 100,
                               "batch_size": 2, "grad_accum_steps": 4,
                               "seed": 42, "condition": "predicted"} for stage in (1, 2, 3)}
        predicted = {"kind": "evaluation", "purpose": "pilot", "condition": "predicted",
                     "dataset_fingerprint": "fixture", "split": "test", "config": copy.deepcopy(self.cfg),
                     "checkpoint": "fresh-predicted.pt", "training_lineage": lineage,
                     "samples": [{"pattern_id": "test-0"}],
                     "summary": {"success_rate": .5, "failure_rate": 0., "whole_chamfer": {"mean": .01}}}
        baseline = copy.deepcopy(predicted)
        baseline.update(condition="contact_only", checkpoint="fresh-contact-only.pt")
        baseline["summary"].update(success_rate=0., failure_rate=.5, whole_chamfer={"mean": .02})
        for stage in baseline["training_lineage"].values():
            stage.update(condition="contact_only", checkpoint_step=200)
        fixed = {"kind": "evaluation", "purpose": "overfit", "condition": "predicted",
                 "dataset_fingerprint": "fixture", "fixed_fit_passed": True,
                 "config": copy.deepcopy(self.cfg)}
        profile = {"kind": "preflight", "status": "passed", "correctness_passed": True,
                   "dataset_fingerprint": "fixture", "profile_signature": profile_signature(self.cfg),
                   "attempts": [{"gpu_name": "RTX A5000", "max_reserved_bytes": 19 * GIB}]}
        preparation = {"kind": "preparation", "learning_ready": True, "dataset_fingerprint": "fixture"}
        gt, perturbed = copy.deepcopy(predicted), copy.deepcopy(predicted)
        gt["condition"], perturbed["condition"] = "gt", "perturbed"
        return [predicted, baseline, fixed, profile, preparation, gt, perturbed]

    def compile_report(self, reports, name):
        paths = [self.write(f"{name}-{index}.json", report) for index, report in enumerate(reports)]
        return pilot_report(paths, self.root / f"{name}-summary.json")

    def test_pilot_acceptance_requires_matched_budgets_and_dataset(self):
        reports = self.comparison_reports()
        good = self.compile_report(reports, "matched")
        self.assertTrue(good["comparisons_are_matched"])
        self.assertTrue(good["scaffold_benefit"])
        self.assertTrue(good["advance_eligible"])
        for mismatch in ("dataset", "budget", "samples", "solver"):
            changed = copy.deepcopy(reports)
            if mismatch == "dataset":
                changed[1]["dataset_fingerprint"] = "unrelated"
            elif mismatch == "budget":
                changed[1]["training_lineage"]["1"]["updates"] = 1000
            elif mismatch == "samples":
                changed[1]["samples"][0]["pattern_id"] = "other-test"
            else:
                changed[1]["config"]["solver"]["refinement_iterations"] = 1
            bad = self.compile_report(changed, mismatch)
            self.assertFalse(bad["comparisons_are_matched"])
            self.assertFalse(bad["advance_eligible"])
        missing_geometry = self.compile_report([r for r in reports if r["kind"] != "preparation"], "no-geometry")
        self.assertFalse(missing_geometry["advance_eligible"])
        missing_control = self.compile_report([r for r in reports if r.get("condition") != "gt"], "no-gt")
        self.assertFalse(missing_control["advance_eligible"])
        over_limit = copy.deepcopy(reports)
        over_limit[3]["attempts"][-1]["max_reserved_bytes"] = 20 * GIB
        self.assertFalse(self.compile_report(over_limit, "over-vram")["advance_eligible"])
        unchanged = copy.deepcopy(reports)
        unchanged[0]["summary"]["success_rate"] = 0.
        self.assertFalse(self.compile_report(unchanged, "no-benefit")["scaffold_benefit"])

    def test_resource_guard_deduplicates_roots_and_enforces_cap_and_free_space(self):
        folder = self.root / "managed"
        child = folder / "runs"
        child.mkdir(parents=True)
        (child / "artifact.bin").write_bytes(b"1234567890")
        usage = namedtuple("usage", "total used free")
        with patch("reassembly.resources.shutil.disk_usage", return_value=usage(100 * GIB, 10 * GIB, 90 * GIB)):
            guard = ResourceGuard([folder, child], cap_gib=1, min_free_gib=50)
            self.assertEqual(guard.check()["managed_bytes"], 10)
            self.assertEqual(len(guard.roots), 1)
            with self.assertRaisesRegex(RuntimeError, "storage cap"):
                guard.check(additional_bytes=GIB)
        with patch("reassembly.resources.shutil.disk_usage", return_value=usage(100 * GIB, 51 * GIB, 49 * GIB)):
            with self.assertRaisesRegex(RuntimeError, "Insufficient free space"):
                guard.check()

    def test_config_rejects_legacy_categories_and_fragment_counts(self):
        for section, key, value in ((None, "version", 1), ("data", "category", "chair"),
                                     ("data", "max_fragments", 4), ("solver", "prior_weight", .8),
                                     ("train", "max_updates", 2001)):
            cfg = copy.deepcopy(self.cfg)
            (cfg if section is None else cfg[section])[key] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
