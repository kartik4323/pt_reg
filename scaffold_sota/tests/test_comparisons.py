"""End-to-end small replay fixtures retain missing-arm and failed-pose rows."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scaffold_sota.comparisons import frozen_comparison
from scaffold_sota.config import load_config
from scaffold_sota.io import study_root, append_jsonl
from scaffold_sota.evaluation import evaluate_prediction
from test_core import sample_fixture


class QuietGuard:
    def __init__(self, *args, **kwargs):
        pass
    def require_idle_device(self):
        pass
    def check(self, **kwargs):
        return {"status": "fixture"}


class ReplayContracts(unittest.TestCase):
    def test_full_a0_a7_denominator_when_priors_are_missing(self):
        sample = sample_fixture()
        other = {**sample, "source_id": "source-b", "pattern_id": "pattern-b"}
        samples = [sample, other]
        class Dataset:
            fingerprint = "manifest"
            def __len__(self):
                return len(samples)
            def __getitem__(self, index):
                return samples[index]
        with tempfile.TemporaryDirectory() as temporary:
            root = study_root(Path(temporary)/"study", create=True)
            manifest = root/"manifest.json"
            manifest.write_text("{}")
            predictions = root/"predictions.jsonl"
            for index, observation in enumerate(samples):
                prediction = {"rotations": observation["rotations_gt"], "translations": observation["translations_gt"]} if index == 0 else {"rotations": None, "translations": None, "status": "failed"}
                row = evaluate_prediction(observation, prediction)
                row.update(prediction=prediction, points_per_fragment=16)
                append_jsonl(predictions, row)
            with patch("scaffold_sota.comparisons.make_dataset", return_value=Dataset()), patch("scaffold_sota.comparisons.make_prior", return_value=None), patch("scaffold_sota.comparisons.ResourceGuard", QuietGuard):
                report = frozen_comparison(root=root, manifest=manifest, predictions=predictions, output=root/"replay",
                                           cfg=load_config(), steps=1)
            self.assertEqual(report["points_per_fragment"], 16)
            self.assertEqual(len(report["conditions"]), 8)
            for condition, value in report["conditions"].items():
                self.assertEqual(value["metrics"]["count"], 2)
                self.assertEqual(value["metrics"]["failure_rate"], .5)
                if condition not in ("A0", "A1"):
                    self.assertEqual(value["status"], "dependency_pending")
                    self.assertIsNone(value["paired_vs_A0"])
                self.assertEqual(len((root/"replay"/(condition+".jsonl")).read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
