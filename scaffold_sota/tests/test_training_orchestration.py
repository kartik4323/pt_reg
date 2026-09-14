"""CPU integration checks for orchestration, using test-only native boundaries.

These fixtures exercise the real runner, checkpoint I/O, RNG restoration,
sanitizer and metrics. They are deliberately not native execution evidence.
"""
from __future__ import annotations

import contextlib
import copy
import importlib
import io
import random
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from scaffold_sota import training
from scaffold_sota.config import load_config
from scaffold_sota.io import digest_json, read_json, read_jsonl, study_root

preflight_module = importlib.import_module("scaffold_sota.preflight")


class _TestDataset:
    fingerprint = "test-only-dataset-fingerprint"
    records = [{"pieces": 2}, {"pieces": 3}]

    def __init__(self, split, interruption=None):
        self.split = split
        self.interruption = interruption

    def __len__(self):
        return len(self.records)

    def set_step(self, step):
        if self.interruption and self.interruption.get("step") == step:
            self.interruption.clear()
            raise KeyboardInterrupt("test-only interruption after a durable checkpoint")

    def __getitem__(self, index):
        points = torch.arange(3 * 16 * 3, dtype=torch.float32).reshape(3, 16, 3) / 200
        mask = torch.tensor([True, True, index == 1])
        return {"points": points, "fragment_mask": mask, "anchor_index": torch.tensor(0),
                "canonical_points": points.clone(), "rotations_gt": torch.eye(3).repeat(3, 1, 1),
                "translations_gt": torch.zeros(3, 3), "fracture_labels": torch.zeros(3, 16),
                "interface_ids": torch.full((3, 16), -1), "source_id": "source-%d" % index,
                "pattern_id": "%s-pattern-%d" % (self.split, index),
                "observation_id": "%s-observation-%d" % (self.split, index),
                "dataset_fingerprint": self.fingerprint, "split": self.split}


class _TestBackend(torch.nn.Module):
    conditioning_parameter_prefixes = ("conditioning.",)

    def __init__(self, options, calls, failure=None):
        super().__init__()
        self.native = torch.nn.Linear(1, 1)
        self.calls, self.failure = calls, failure
        if options["condition"] not in ("native", "B0"):
            self.conditioning = torch.nn.Linear(1, 1)

    def loss(self, sample, prior=None):
        # All three RNGs affect the optimizer trajectory, making resume checks
        # sensitive to forgotten RNG state and validation-induced RNG drift.
        self.calls.append(("loss", tuple(sorted(sample))))
        target = random.random() + float(np.random.random()) + torch.rand(())
        prediction = self.native(sample["points"].mean().reshape(1, 1)).squeeze()
        if hasattr(self, "conditioning"):
            prediction = prediction + self.conditioning(torch.ones(1, 1)).squeeze()
        return {"loss": (prediction - target).square()}

    def predict(self, sample, prior=None, seed=0):
        assert set(sample) == {"points", "fragment_mask", "anchor_index"}, "Ground truth crossed prediction boundary"
        assert not torch.is_grad_enabled(), "Prediction must run without gradients"
        self.calls.append(("predict", tuple(sorted(sample))))
        random.random(), np.random.random(), torch.rand(())
        if self.failure == "missing":
            return {"status": "no_solution", "reason": "test-only disconnected matches",
                    "rotations": None, "translations": None}
        if self.failure == "malformed":
            return {"status": "ok", "rotations": torch.zeros(3, 3, 3),
                    "translations": torch.zeros(3, 3)}
        count = int(sample["fragment_mask"].sum())
        return {"status": "ok", "rotations": torch.eye(3).repeat(count, 1, 1),
                "translations": torch.zeros(count, 3)}


class _TestGuard:
    def __init__(self, *args, **kwargs):
        pass

    def check(self, **kwargs):
        return {"test_only": True}

    def require_idle_device(self):
        pass


class TrainingOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="scaffold_sota_runner_tests_")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.manifest = self.base / "test-only-manifest.json"
        self.manifest.write_text('{"test_only": true}\n', encoding="utf-8")
        self.source = self.base / "test-only-native-source"
        self.source.mkdir()
        self.cfg = load_config()
        self.cfg["data"].update(points_per_fragment=16, sdf_queries=8)
        self.cfg["prior"]["query_count"] = 8
        self.cfg["train"].update(updates=3, grad_accum_steps=1, validation_interval=1,
                                  checkpoint_interval=1, log_interval=1, learning_rate=.01,
                                  betas=[.8, .95], eps=1e-6)
        self.calls = []

    def root(self, name):
        return study_root(self.base / name, create=True)

    @contextlib.contextmanager
    def seams(self, module=training, *, interruption=None, failure=None):
        def dataset(manifest, split, cfg, **kwargs):
            return _TestDataset(split, interruption if split == "train" else None)

        def materialize(model, root, output, source):
            path = Path(output) / "test-only-native-copy"
            path.mkdir(parents=True, exist_ok=True)
            return path, {}

        def backend(model, source, options, device):
            return _TestBackend(options, self.calls, failure).to(device)

        info = {"path": str(self.source), "revision": "test-only-pinned-revision",
                "registry_sha256": "test-only-registry-hash"}
        with contextlib.ExitStack() as stack:
            for name, value in (("ResourceGuard", _TestGuard), ("source_info", lambda *a: info),
                                ("materialize_source", materialize), ("build_backend", backend),
                                ("make_dataset", dataset)):
                stack.enter_context(mock.patch.object(module, name, value))
            # Concurrent edits by other agents must not change test fixture identity.
            stack.enter_context(mock.patch.object(training, "code_hash", lambda: "test-only-code-hash"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(warnings.catch_warnings())
            warnings.simplefilter("ignore", FutureWarning)
            yield

    def run_training(self, root, **kwargs):
        return training.train(root=root, model="test-only", manifest=self.manifest,
                              cfg=self.cfg, device="cpu", cpu_smoke=True,
                              run_id=kwargs.pop("run_id", "native_run"), **kwargs)

    def checkpoint(self, root, run_id="native_run"):
        return training.load_study_checkpoint(root / "runs" / run_id / "latest.pt")

    def assert_state_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_state_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_resume_preserves_rng_optimizer_steps_and_prediction_boundary(self):
        continuous, interrupted = self.root("continuous"), self.root("interrupted")
        with self.seams():
            self.run_training(continuous)
        with self.seams(interruption={"step": 1}):
            with self.assertRaisesRegex(KeyboardInterrupt, "test-only interruption"):
                self.run_training(interrupted)
        self.assertEqual(self.checkpoint(interrupted)["step"], 1)
        self.assertEqual(read_json(interrupted / "runs/native_run/run.json")["status"], "interrupted")
        self.assertFalse((interrupted / ".active_job.lock").exists())
        with self.seams():
            self.run_training(interrupted, resume=True)
        uninterrupted_state, resumed_state = self.checkpoint(continuous), self.checkpoint(interrupted)
        self.assertEqual(resumed_state["step"], 3)
        for key in ("model", "optimizer", "rng", "scaler"):
            self.assert_state_equal(uninterrupted_state[key], resumed_state[key])
        self.assertEqual([row["step"] for row in read_jsonl(interrupted / "runs/native_run/history.jsonl")], [1, 2, 3])
        self.assertEqual(read_json(interrupted / "runs/native_run/run.json")["status"], "completed")
        self.assertFalse((interrupted / ".active_job.lock").exists())
        self.assertTrue(any(name == "predict" for name, keys in self.calls))
        self.assertTrue(any(name == "loss" and "rotations_gt" in keys for name, keys in self.calls))
        group = resumed_state["optimizer"]["param_groups"][0]
        self.assertEqual(tuple(group["betas"]), (.8, .95))
        self.assertEqual(group["eps"], 1e-6)

    def test_resume_rejects_changed_configuration_before_build(self):
        root = self.root("resume_identity")
        with self.seams(interruption={"step": 1}):
            with self.assertRaises(KeyboardInterrupt):
                self.run_training(root)
        self.cfg["train"]["learning_rate"] *= 2
        self.calls.clear()
        with self.seams():
            with self.assertRaisesRegex(ValueError, "Resume identity mismatch"):
                self.run_training(root, resume=True)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.checkpoint(root)["step"], 1)
        self.assertFalse((root / ".active_job.lock").exists())

    def test_native_initialization_requires_complete_matched_lineage(self):
        root = self.root("native_lineage")
        with self.seams():
            self.run_training(root)
        path = root / "runs/native_run/latest.pt"
        state = self.checkpoint(root)
        identity = dict(state["identity"], condition="B1", run_id="adapted_run")
        backend = _TestBackend({"condition": "B1"}, [])
        training.initialize_native(backend, path, identity)
        for key, value in state["model"].items():
            self.assertTrue(torch.equal(backend.state_dict()[key], value))
        protected = ("model", "stage", "manifest_hash", "dataset_fingerprint", "source_revision", "seed",
                     "registry_hash", "code_hash", "feature_hash", "cfg_hash", "native_options_hash", "cpu_smoke")
        for key in protected:
            with self.subTest(key=key):
                changed = dict(identity)
                changed[key] = "changed-test-value"
                with self.assertRaisesRegex(ValueError, key):
                    training.initialize_native(backend, path, changed)
                changed.pop(key)
                with self.assertRaisesRegex(ValueError, key):
                    training.initialize_native(backend, path, changed)
        poisoned = copy.deepcopy(state)
        poisoned["model"].pop("native.weight")
        bad_path = root / "missing_native.pt"
        torch.save(poisoned, bad_path)
        with self.assertRaisesRegex(ValueError, "does not match backend"):
            training.initialize_native(backend, bad_path, identity)
        poisoned = copy.deepcopy(state)
        poisoned["identity"]["condition"] = "B1"
        poisoned["identity_hash"] = digest_json(poisoned["identity"])
        torch.save(poisoned, bad_path)
        with self.assertRaisesRegex(ValueError, "native baseline"):
            training.initialize_native(backend, bad_path, identity)

    def test_matched_branch_runs_from_native_checkpoint(self):
        root = self.root("matched_branch")
        with self.seams():
            self.run_training(root)
            self.run_training(root, condition="B1", run_id="B1_run",
                              initialize=root / "runs/native_run/latest.pt")
        parent, child = self.checkpoint(root), self.checkpoint(root, "B1_run")
        self.assertEqual(child["step"], 3)
        self.assertEqual(child["identity"]["native_options_hash"], parent["identity"]["native_options_hash"])
        self.assertEqual(child["identity"]["cfg_hash"], parent["identity"]["cfg_hash"])
        self.assertIsNotNone(child["identity"]["native_parent_hash"])
        self.assertTrue(any(key.startswith("conditioning.") for key in child["model"]))

    def test_standalone_preflight_checks_both_piece_counts_and_no_gt(self):
        root = self.root("passing_preflight")
        with self.seams(preflight_module):
            result = preflight_module.preflight(root, "test-only", self.manifest, self.cfg, device="cpu")
        self.assertEqual(result["status"], "execution_checks_passed")
        self.assertEqual([row["pieces"] for row in result["examples"]], [3, 2])
        self.assertFalse(result["learning_acceptance"])
        self.assertFalse(result["cuda_verified"])
        self.assertEqual(read_json(result["report_path"])["status"], result["status"])
        self.assertEqual(sum(name == "predict" for name, keys in self.calls), 2)
        self.assertFalse((root / ".active_job.lock").exists())

    def test_standalone_preflight_records_missing_and_malformed_pose_rejections(self):
        for failure, message in (("missing", "no poses"), ("malformed", "proper rigid")):
            with self.subTest(failure=failure):
                root = self.root("reject_" + failure)
                with self.seams(preflight_module, failure=failure):
                    with self.assertRaisesRegex((RuntimeError, ValueError), message):
                        preflight_module.preflight(root, "test-only", self.manifest, self.cfg, device="cpu")
                reports = list((root / "preflight").glob("*/preflight.json"))
                self.assertEqual(len(reports), 1)
                report = read_json(reports[0])
                self.assertEqual(report["status"], "failed")
                self.assertIn(message, report["error"])
                self.assertFalse(report["cuda_verified"])
                self.assertFalse(report["learning_acceptance"])
                self.assertFalse((root / ".active_job.lock").exists())

    def test_warm_training_preflight_rejects_malformed_rigid_poses(self):
        root = self.root("reject_training")
        with self.seams(failure="malformed"):
            with self.assertRaisesRegex(ValueError, "proper rigid"):
                self.run_training(root)
        self.assertEqual(read_json(root / "runs/native_run/run.json")["status"], "failed")
        self.assertFalse((root / "runs/native_run/latest.pt").exists())
        self.assertFalse((root / ".active_job.lock").exists())

    def test_warm_training_allows_explicit_untrained_matcher_failure(self):
        root = self.root("untrained_matcher")
        with self.seams(failure="missing"):
            record = self.run_training(root)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(self.checkpoint(root)["step"], 3)
        preflight = read_json(root / "runs/native_run/preflight.json")
        self.assertEqual(preflight["prediction_status"], "no_solution")
        self.assertFalse(preflight["learning_acceptance"])
        for row in read_jsonl(root / "runs/native_run/validation.jsonl"):
            self.assertEqual(row["selection_score"], 0)
            self.assertEqual(row["metrics"]["failure_rate"], 1)

    def test_allowing_structured_failure_does_not_accept_partial_or_ok_failures(self):
        sample = _TestDataset("train")[0]
        bad = [None, {}, {"status": "ok", "reason": "missing"},
               {"status": "no_solution"},
               {"status": "no_solution", "reason": "partial", "rotations": torch.eye(3).repeat(2, 1, 1)}]
        for prediction in bad:
            with self.subTest(prediction=str(prediction)):
                with self.assertRaisesRegex(RuntimeError, "no poses"):
                    training.validate_prediction_contract(sample, prediction, allow_structured_failure=True)

    def test_candidate_bank_is_fixed_not_gt_selected_and_marks_duplicates(self):
        dataset = _TestDataset("val")
        backend = _TestBackend({"condition": "native"}, self.calls)
        output = self.base / "candidates.jsonl"
        summary, _ = training.validation(backend, dataset, None, "cpu", self.cfg,
                                         output=output, candidates=3)
        self.assertEqual(summary["degenerate_bank_count"], 2)
        self.assertEqual(summary["candidate_recall_at_k_diagnostic"], 1)
        for row in read_jsonl(output):
            self.assertEqual(row["unique_valid_candidates"], 1)
            self.assertEqual(row["native_selection_index"], 0)
            self.assertEqual(row["prediction"], row["candidates"][0])
            self.assertEqual(len(set(row["candidate_seeds"])), 3)
            self.assertEqual(row["points_per_fragment"], 16)

    def test_candidate_failures_remain_in_denominator(self):
        dataset = _TestDataset("val")
        backend = _TestBackend({"condition": "native"}, self.calls, failure="missing")
        summary, _ = training.validation(backend, dataset, None, "cpu", self.cfg, candidates=3)
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["failure_rate"], 1)
        self.assertEqual(summary["candidate_recall_at_k_diagnostic"], 0)

    def test_unlock_inspection_does_not_signal_live_process(self):
        import os
        from scaffold_sota.io import RunLock, unlock, _process_alive
        root = self.root("unlock_live")
        with mock.patch.object(os, "kill", side_effect=AssertionError("Must not signal Windows process")) if os.name == "nt" else contextlib.nullcontext():
            self.assertTrue(_process_alive(os.getpid()))
            with RunLock(root):
                with self.assertRaisesRegex(RuntimeError, "still alive"):
                    unlock(root)


if __name__ == "__main__":
    unittest.main()
