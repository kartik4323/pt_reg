"""Actual prepared-fixture evaluation, persistent evidence, and resume contracts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from test_reassembly_repair_data import write_prepared, tiny_config
from reassembly.repair.model import RepairModel, configure_stage
from reassembly.repair.checkpoints import save_checkpoint
from reassembly.repair.evaluation import evaluate
from reassembly.repair.reporting import read_rows, paired_benefit
from reassembly.training import _optimizer, _scaler
from reassembly.resources import write_json


class RepairEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = write_prepared(self.root / "prepared")
        self.cfg = tiny_config()
        self.cfg["train"].update(amp=False)
        self.cfg["model"].update(dim=8, sample_counts=[8, 4, 2], contact_points=8, neighbors=4)
        self.cfg["data"].update(points_per_fragment=20, sdf_queries=16)
        self.fingerprint = json.loads(self.manifest.read_text())["fingerprint"]
        self.checkpoint = self.root / "run" / "best.pt"
        self.save(stage=1)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, stage):
        torch.manual_seed(817)
        model = RepairModel(self.cfg)
        configure_stage(model, stage)
        save_checkpoint(self.checkpoint, model, _optimizer(model, self.cfg), _scaler(False), self.cfg,
            self.fingerprint, stage=stage, step=1, purpose="experiment", run_id="evaluation-fixture",
            metrics={"best_score": [0.]}, lineage={})
        write_json(self.checkpoint.parent / "config.resolved.json", self.cfg)

    def run_evaluation(self, output, **kwargs):
        return evaluate(self.checkpoint, self.manifest, output, "cpu", seeds=(4101,),
                        conditions=("contact_only",), **kwargs)

    def test_real_evaluation_and_completed_resume_skip_computation(self):
        output = self.root / "evaluation"
        self.run_evaluation(output)
        first = (output / "examples.jsonl").read_bytes()
        with patch("reassembly.repair.evaluation.sample_results", side_effect=AssertionError("already complete")):
            self.run_evaluation(output, resume=True)
        self.assertEqual(first, (output / "examples.jsonl").read_bytes())
        report, rows = read_rows(output / "evaluation.json")
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(list((output / "tensors").glob("*.npz"))), 3)
        self.assertTrue(all("candidate_fingerprint" in r["candidate_coverage"] for r in rows))

    def test_partial_resume_preserves_completed_rows_and_rejects_changed_checkpoint(self):
        output = self.root / "partial"
        self.run_evaluation(output)
        rows = (output / "examples.jsonl").read_bytes().splitlines(keepends=True)
        (output / "examples.jsonl").write_bytes(rows[0] + b'{"incomplete":')
        (output / "evaluation.json").unlink()
        self.run_evaluation(output, resume=True)
        _, completed = read_rows(output / "evaluation.json")
        self.assertEqual(len(completed), 3)
        self.assertTrue((output / "resume_recovery.json").exists())
        with self.checkpoint.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_evaluation(output, resume=True)

    def test_stage2_balanced_metrics_and_identical_candidates_for_all_conditions(self):
        self.save(stage=2)
        output = self.root / "paired"
        evaluate(self.checkpoint, self.manifest, output, "cpu", seeds=(4101,))
        _, rows = read_rows(output / "evaluation.json")
        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertIn("near_surface_sdf_l1", row["field_metrics"])
            self.assertIn("sdf_inside_uncertainty_mae", row["field_metrics"])
        comparison = paired_benefit(rows)
        self.assertEqual(comparison["paired_count"], 3)


if __name__ == "__main__":
    unittest.main()
