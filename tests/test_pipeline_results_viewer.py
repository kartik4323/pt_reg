"""Read-only notebook regressions; no experiment package hash changes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pipeline_results_viewer import Catalog, contained, outcome_table, point_panels, sample_points


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "group" / "condition" / "shard0"
        self.run.mkdir(parents=True)
        (self.run / "study.json").write_text(json.dumps({"identity": {"config": {"smoke": False}}}))

    def job(self, jid, **kwargs):
        directory = self.run / "jobs" / jid
        directory.mkdir(parents=True)
        value = dict(job_id=jid, stage="E4", arm="B0__refine", case_id=jid,
                     source_id=jid, split="dev", status="complete", oracle=False,
                     smoke=False, seconds=1, output={})
        value.update(kwargs)
        (directory / "result.json").write_text(json.dumps(value))
        return directory

    def metrics(self, rows):
        directory = self.run / "evaluation" / "dev"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_nested_discovery_deduplicates_explicit_and_parent_paths(self):
        self.job("a")
        catalog = Catalog([self.root, self.run])
        self.assertEqual(len(catalog.runs), 1)
        self.assertEqual(len(catalog.frame), 1)

    def test_failure_stays_in_denominator_and_repeats_average_within_source(self):
        self.job("a1", source_id="a")
        self.job("a2", source_id="a")
        self.job("b", source_id="b", status="failed", error="generation failed")
        self.metrics([{"job_id": "a1", "metrics": {"success": True}},
                      {"job_id": "a2", "metrics": {"success": True}}])
        result = outcome_table(Catalog([self.run]).frame).iloc[0]
        self.assertEqual(result.source_macro_success, .5)
        self.assertEqual(result.scored_or_failed, 3)
        self.assertEqual(result.failed, 1)

    def test_unevaluated_and_not_applicable_are_not_fake_successes(self):
        self.job("not_ready")
        self.job("no_prior", output={"not_applicable": "no compatible hypothesis"})
        result = outcome_table(Catalog([self.run]).frame).iloc[0]
        self.assertEqual(result.scored_or_failed, 1)
        self.assertEqual(result.source_macro_success, 0)

    def test_oracle_smoke_defaults_and_lock_liveness_unknown(self):
        self.job("oracle", oracle=True)
        self.job("smoke", smoke=True)
        self.job("a")
        pending = self.run / "jobs" / "pending"
        pending.mkdir()
        (pending / "running.lock").write_text('{"pid": 999}')
        catalog = Catalog([self.run])
        visible = catalog.select()
        self.assertEqual(set(visible.job_id), {"a", "pending"})
        self.assertEqual(catalog.row(self.run, "pending").status, "locked")
        self.assertEqual(catalog.row(self.run, "pending").stage, "UNKNOWN")

    def test_retry_lock_masks_old_completed_metrics(self):
        directory = self.job("a")
        self.metrics([{"job_id": "a", "metrics": {"success": True}}])
        (directory / "running.lock").write_text('{"pid": 999}')
        catalog = Catalog([self.run])
        result = outcome_table(catalog.frame).iloc[0]
        self.assertEqual(result.locked, 1)
        self.assertEqual(result.scored_or_failed, 0)

    def test_partial_metric_line_retains_good_rows_and_warns(self):
        self.job("a")
        self.metrics([{"job_id": "a", "metrics": {"success": True}}])
        with (self.run / "evaluation" / "dev" / "metrics.jsonl").open("a") as stream:
            stream.write('{"job_id":')
        catalog = Catalog([self.run])
        self.assertTrue(catalog.warnings)
        self.assertEqual(catalog.row(self.run, "a")["metric.success"], True)

    def test_integrity_detects_mutation_and_blocks_path_escape(self):
        asset = b"original surface"
        self.job("a", artifacts=[{"path": "jobs/a/shape.npz", "sha256": hashlib.sha256(asset).hexdigest()}])
        path = self.run / "jobs" / "a" / "shape.npz"
        path.write_bytes(asset)
        catalog = Catalog([self.run])
        self.assertEqual(catalog.integrity(self.run, "a").integrity.iloc[0], "ok")
        path.write_bytes(b"modified")
        self.assertEqual(catalog.integrity(self.run, "a").integrity.iloc[0], "HASH_MISMATCH")
        with self.assertRaises(ValueError):
            contained(self.run, "../outside")

    def preparation(self, case_id, jid, points):
        directory = self.job(jid, stage="E0", case_id=case_id, arm="prepare")
        np.savez(directory / "observed.npz", points_0=points, points_1=points+2,
                 centers=np.zeros((2, 3)), scale=1., anchor=0)
        np.savez(directory / "candidates.npz", poses=np.eye(4)[None, None].repeat(2, axis=1))

    def test_rotation_preview_uses_base_points_and_never_reads_reference_by_default(self):
        points = np.arange(48).reshape(16, 3).astype(float)
        self.preparation("base", "prep_base", points)
        self.preparation("changed", "prep_changed", points+100)
        output = {"base_case_id": "base", "factor": "rotation", "retained_ids": [0, 1]}
        directory = self.job("prediction", stage="E7", case_id="changed", output=output)
        np.savez(directory / "poses.npz", normalized=np.eye(4)[None].repeat(2, axis=0))
        catalog = Catalog([self.run])
        with patch.object(catalog, "dataset", side_effect=AssertionError("reference access")):
            panels = point_panels(catalog, catalog.row(self.run, "prediction"))
        np.testing.assert_array_equal(panels[1][1][0][1], points)
        self.assertIn("mapped back", panels[1][0])

    def test_reading_does_not_change_artifacts_and_empty_roots_work(self):
        self.job("a")
        before = {str(p): p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        catalog = Catalog([self.run])
        catalog.refresh()
        after = {str(p): p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        empty = Catalog([self.root / "does-not-exist"])
        self.assertTrue(empty.frame.empty)
        self.assertTrue(outcome_table(empty.frame).empty)

    def test_cloud_sampling_is_deterministic_and_nonmutating(self):
        points = np.arange(300).reshape(100, 3).astype(float)
        original = points.copy()
        first = sample_points(points, 10)
        np.testing.assert_array_equal(first, sample_points(points, 10))
        np.testing.assert_array_equal(points, original)
        self.assertEqual(len(first), 10)


if __name__ == "__main__":
    unittest.main()
