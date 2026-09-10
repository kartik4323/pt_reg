"""Real tiny-model training lifecycle, with only source data replaced by fixtures."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from reassembly.checkpoints import load_checkpoint
from reassembly.config import load_config
from reassembly.model import ReassemblyModel
from reassembly.resources import GIB
from reassembly.training import (_check_cuda_memory, _validate, preflight,
                                  synthetic_batch, train_stage)


def tiny_config():
    cfg = load_config()
    cfg["model"].update(dim=16, sample_counts=[12, 8, 4], neighbors=4)
    cfg["data"].update(points_per_fragment=20, sdf_queries=24, min_sources=1)
    cfg["loss"]["view_consistency"] = 0
    cfg["train"].update(batch_size=1, grad_accum_steps=1, max_updates=1,
                        overfit_updates=1, validation_interval=1, validation_samples=2,
                        amp=False)
    cfg["solver"].update(resolution=4, field_chunk=16)
    return cfg


class FixtureDataset:
    def __init__(self, manifest, split, cfg, stage=1, fixed=False, limit=None):
        document = json.loads(Path(manifest).read_text())
        self.records = [p for p in document["patterns"] if p["split"] == split]
        if limit is not None:
            self.records = self.records[:limit]
        self.fixed = fixed
        with torch.random.fork_rng():
            torch.manual_seed(197)
            self.batch = synthetic_batch(cfg, 1, torch.device("cpu"))
        self.step = 0

    def __len__(self):
        return len(self.records)

    def set_step(self, step):
        self.step = step

    def __getitem__(self, index):
        return {key: value[0].clone() for key, value in self.batch.items()}


class TrainingLifecycleTests(unittest.TestCase):
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
        self.manifest = self.root / "manifest.json"
        self.write_manifest()
        self.dataset_patch = patch("reassembly.data.FractureDataset", FixtureDataset)
        self.dataset_patch.start()

    def tearDown(self):
        self.dataset_patch.stop()
        self.temp.cleanup()

    def write_manifest(self, count=16):
        self.manifest.write_text(json.dumps({"fingerprint": "synthetic-lifecycle-fixture",
            "patterns": [{"source_id": "train-source", "pattern_id": f"train-{i}", "split": "train"}
                         for i in range(count)] + [{"source_id": "val-source", "pattern_id": "val-0", "split": "val"}]}))

    def run_stage(self, name, stage=1, **kwargs):
        return train_stage(self.cfg, self.manifest, self.root / name, stage, "cpu", overfit=True, **kwargs)

    def test_fresh_three_stage_lifecycle_and_purpose_isolation(self):
        first = self.run_stage("stage1")
        path1 = Path(first["best_checkpoint"])
        second = self.run_stage("stage2", stage=2, initialize_from=path1)
        third = self.run_stage("stage3", stage=3, initialize_from=second["best_checkpoint"])
        final = load_checkpoint(third["best_checkpoint"], stage=3, purpose="overfit")
        self.assertEqual(final["step"], 1)
        self.assertTrue(final["run_id"])
        self.assertIn("random_state", final)
        self.assertEqual(set(final["training_lineage"]), {"1", "2", "3"})
        self.assertTrue(all(entry["updates"] == 1 and entry["batch_size"] == 1 for entry in final["training_lineage"].values()))
        self.assertEqual(final["cfg"]["run_lineage"], final["training_lineage"])
        self.assertTrue(all(torch.isfinite(tensor).all() for tensor in final["model"].values()))
        with self.assertRaisesRegex(ValueError, "purpose"):
            load_checkpoint(path1, purpose="pilot")
        with self.assertRaisesRegex(ValueError, "stage-2"):
            self.run_stage("badstage", stage=3, initialize_from=path1)
        with self.assertRaisesRegex(ValueError, "fresh random"):
            self.run_stage("badinit", initialize_from=path1)
        with self.assertRaisesRegex(FileExistsError, "fresh run"):
            self.run_stage("stage1")

    def test_explicit_same_run_resume_and_random_state_reproducibility(self):
        self.run_stage("resumed")
        original_run_id = load_checkpoint(self.root / "resumed/latest.pt")["run_id"]
        with self.assertRaisesRegex(ValueError, "already reached"):
            self.run_stage("resumed", resume=self.root / "resumed/latest.pt")
        self.cfg["train"]["overfit_updates"] = 2
        with self.assertRaisesRegex(ValueError, "existing run directory"):
            self.run_stage("unrelated", resume=self.root / "resumed/latest.pt")
        self.assertFalse((self.root / "unrelated").exists())
        self.run_stage("resumed", resume=self.root / "resumed/latest.pt")
        self.run_stage("uninterrupted")
        resumed = load_checkpoint(self.root / "resumed/latest.pt")
        uninterrupted = load_checkpoint(self.root / "uninterrupted/latest.pt")
        self.assertEqual(resumed["run_id"], original_run_id)
        self.assertEqual(resumed["step"], 2)
        self.assertEqual(resumed["training_lineage"]["1"]["updates"], 2)
        for key in resumed["model"]:
            torch.testing.assert_close(resumed["model"][key], uninterrupted["model"][key], rtol=0, atol=0)

    def test_forged_run_lineage_legacy_and_incomplete_checkpoints_rejected(self):
        self.run_stage("first")
        self.run_stage("other")
        checkpoint = load_checkpoint(self.root / "first/latest.pt")
        torch.save(checkpoint, self.root / "other/latest.pt")
        self.cfg["train"]["overfit_updates"] = 2
        with self.assertRaisesRegex(ValueError, "different run"):
            self.run_stage("other", resume=self.root / "other/latest.pt")
        legacy = self.root / "legacy.pt"
        torch.save({"model": checkpoint["model"]}, legacy)
        with self.assertRaisesRegex(ValueError, "Legacy or external"):
            load_checkpoint(legacy)
        del checkpoint["random_state"]
        torch.save(checkpoint, legacy)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            load_checkpoint(legacy)

    def test_exact_overfit_pattern_count_and_bounded_updates(self):
        self.write_manifest(15)
        with self.assertRaisesRegex(ValueError, "16 distinct"):
            self.run_stage("few")
        self.assertFalse((self.root / "few").exists())
        self.write_manifest()
        self.cfg["train"]["overfit_patterns"] = 8
        with self.assertRaisesRegex(ValueError, "exactly 16"):
            self.run_stage("small")
        self.cfg["train"]["overfit_patterns"] = 16
        for budget in (0, 2001):
            self.cfg["train"]["overfit_updates"] = budget
            with self.assertRaisesRegex(ValueError, "1..2000"):
                self.run_stage(f"budget-{budget}")

    def test_nonfinite_validation_cannot_save_a_checkpoint(self):
        model = ReassemblyModel(self.cfg)
        dataset = FixtureDataset(self.manifest, "val", self.cfg)
        with patch("reassembly.losses.compute_losses", return_value={"loss": torch.tensor(0.), "sdf_l1": torch.tensor(float("nan"))}):
            with self.assertRaisesRegex(RuntimeError, "validation metrics"):
                _validate(model, dataset, 2, self.cfg, torch.device("cpu"))
        with patch("reassembly.training._validate", return_value={"loss": float("nan")}):
            with self.assertRaisesRegex(RuntimeError, "non-finite validation"):
                self.run_stage("badvalidation")
        self.assertFalse((self.root / "badvalidation/latest.pt").exists())

    def test_cpu_preflight_runs_three_real_optimizer_steps(self):
        report, resolved = preflight(self.cfg, "cpu")
        self.assertEqual(report["status"], "cpu_verified_cuda_unmeasured")
        self.assertEqual([r["stage"] for r in report["attempts"][0]["stages"]], [1, 2, 3])
        self.assertIsNone(report["attempts"][0]["max_reserved_bytes"])
        self.assertEqual(resolved["data"]["points_per_fragment"], self.cfg["data"]["points_per_fragment"])

    def test_runtime_memory_limit_is_strict(self):
        with patch("torch.cuda.max_memory_reserved", return_value=20 * GIB):
            with self.assertRaisesRegex(RuntimeError, "memory cap exceeded"):
                _check_cuda_memory(torch.device("cuda"), self.cfg)
        with patch("torch.cuda.max_memory_reserved", return_value=19 * GIB):
            _check_cuda_memory(torch.device("cuda"), self.cfg)

    def test_interrupted_prior_stage_cannot_initialize_the_next_stage(self):
        self.cfg["train"]["overfit_updates"] = 2
        with patch("reassembly.training._validate", side_effect=[{"loss": 1.0}, RuntimeError("fixture interruption")]):
            with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                self.run_stage("interrupted")
        report = json.loads((self.root / "interrupted/training_report.json").read_text())
        self.assertEqual(report["status"], "running")
        self.assertTrue((self.root / "interrupted/best.pt").exists())
        with self.assertRaisesRegex(ValueError, "completed prior run"):
            self.run_stage("blocked-stage2", stage=2, initialize_from=self.root / "interrupted/best.pt")
        self.assertFalse((self.root / "blocked-stage2").exists())

    def test_resume_cannot_change_optimizer_data_loss_or_solver(self):
        self.run_stage("immutable")
        self.cfg["train"]["overfit_updates"] = 2
        original = copy.deepcopy(self.cfg)
        changes = [("train", "learning_rate", .25), ("train", "batch_size", 2),
                   ("train", "amp", True), ("data", "points_per_fragment", 21),
                   ("loss", "segmentation", 2.), ("solver", "prior_weight", .1)]
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                self.cfg = copy.deepcopy(original)
                self.cfg[section][key] = value
                with self.assertRaisesRegex(ValueError, "Cannot change"):
                    self.run_stage("immutable", resume=self.root / "immutable/latest.pt")
        self.cfg = original

    def test_completed_resume_retains_best_weights_and_updates_budget_metadata(self):
        with patch("reassembly.training._validate", return_value={"loss": 1.0}):
            self.run_stage("oldbest")
        old_best = load_checkpoint(self.root / "oldbest/best.pt")
        self.cfg["train"]["overfit_updates"] = 2
        with patch("reassembly.training._validate", return_value={"loss": 2.0}):
            self.run_stage("oldbest", resume=self.root / "oldbest/latest.pt")
        updated_best = load_checkpoint(self.root / "oldbest/best.pt")
        self.assertEqual(updated_best["step"], 1)
        self.assertEqual(updated_best["cfg"]["train"]["overfit_updates"], 2)
        self.assertEqual(updated_best["training_lineage"]["1"]["updates"], 2)
        for key in old_best["model"]:
            torch.testing.assert_close(updated_best["model"][key], old_best["model"][key], rtol=0, atol=0)
        # Best can be an early selected checkpoint from a fully completed run.
        self.run_stage("next-stage", stage=2, initialize_from=self.root / "oldbest/best.pt")


if __name__ == "__main__":
    unittest.main()
