from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sota_repro.adapters import export_predictions
from sota_repro.evaluation import CONVENTION, evaluate_jsonl, evaluate_records
from sota_repro.materialize import create_native_view, materialize_common
from sota_repro.materialize_breaking_bad import materialize_breaking_bad
from sota_repro.registry import load_registry
from sota_repro.subset import _hash_tree, build_manifest, select_object_disjoint, Candidate


def _piece(path: Path) -> None:
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")


class SuiteTests(unittest.TestCase):
    def test_source_backed_registry_has_full_pins_and_exclusions(self) -> None:
        registry = load_registry()
        self.assertEqual(len([item for item in registry.values() if item.source_status == "released"]), 8)
        self.assertEqual(len(registry["cmnet"].revision), 40)
        self.assertEqual(registry["fragmentdiff"].source_status, "unavailable")

    def test_object_selection_is_deterministic_disjoint_and_capped(self) -> None:
        candidates = []
        for category in ("a", "b"):
            for split in ("train", "val"):
                for object_id in ("one", "two"):
                    candidates.append(Candidate("bb", split, category, object_id, f"{category}/{object_id}", 10, "x", 2))
        selected = select_object_disjoint(candidates, 30, 42, {"train", "val"}, True)
        self.assertEqual(selected, select_object_disjoint(reversed(candidates), 30, 42, {"train", "val"}, True))
        self.assertLessEqual(sum(item.size_bytes for item in selected), 30)
        self.assertEqual(len({(item.split, item.category, item.object_id) for item in selected}), len(selected))

    def test_planning_and_materialization_preserves_split_and_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bb = root / "bb"
            (bb / "data_split").mkdir(parents=True)
            (bb / "data_split" / "everyday.train.txt").write_text("Bottle/a\n", encoding="utf-8")
            (bb / "data_split" / "everyday.val.txt").write_text("Bottle/b\n", encoding="utf-8")
            (bb / "data_split" / "artifact.train.txt").write_text("a\n", encoding="utf-8")
            (bb / "data_split" / "artifact.val.txt").write_text("b\n", encoding="utf-8")
            for relative in ("everyday/Bottle/a/fractured_0", "everyday/Bottle/b/fractured_0", "artifact/a/fractured_0", "artifact/b/fractured_0"):
                folder = bb / relative
                folder.mkdir(parents=True)
                _piece(folder / "piece_0.obj")
                _piece(folder / "piece_1.obj")
            partnet = root / "partnet" / "train" / "chair" / "sample"
            partnet.mkdir(parents=True)
            np.save(partnet / "parts.npy", np.zeros((2, 4, 3), dtype=np.float32))
            np.save(partnet / "target.npy", np.zeros((5, 3), dtype=np.float32))
            np.save(partnet / "poses.npy", np.zeros((2, 7), dtype=np.float32))
            manifest_path = root / "source_manifest.json"
            manifest = build_manifest(bb, root / "partnet", manifest_path)
            self.assertGreaterEqual(len(manifest["samples"]), 3)
            materialized = root / "retained"
            result = materialize_common(manifest_path, materialized)
            self.assertLessEqual(result["bytes"], manifest["retained_cap_bytes"])
            self.assertTrue((materialized / "common_v1_manifest.json").exists())
            view = create_native_view("jigsaw", materialized, root / "views")
            self.assertTrue((view / "breaking_bad" / "data_split" / "everyday.train.txt").exists())
            self.assertTrue((view / "data_lists" / "everyday_train.txt").exists())

    def test_ground_truth_and_symmetric_equivalence_score_perfectly(self) -> None:
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        parts = [{"rotation": identity, "translation": [0, 0, 0], "part_chamfer": 0.0}, {"rotation": identity, "translation": [1, 0, 0], "part_chamfer": 0.0}]
        record = {"sample_id": "s", "convention": CONVENTION, "track": "partnet_gpat", "predicted_parts": list(reversed(parts)), "ground_truth_parts": parts, "equivalence_classes": [4, 4]}
        result = evaluate_records([record])
        self.assertAlmostEqual(result["rotation_degrees"]["rmse"], 0.0)
        self.assertAlmostEqual(result["translation"]["rmse"], 0.0)
        self.assertEqual(result["part_accuracy"], 1.0)

    def test_breaking_bad_materializer_stages_only_selected_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compressed = root / "compressed"
            selected = compressed / "everyday_compressed" / "Bottle" / "selected" / "fractured_0"
            selected.mkdir(parents=True)
            _piece(selected / "piece_0.obj")
            _piece(selected / "piece_1.obj")
            skipped = compressed / "everyday_compressed" / "Bottle" / "skipped" / "fractured_0"
            skipped.mkdir(parents=True)
            _piece(skipped / "piece_0.obj")
            _piece(skipped / "piece_1.obj")
            (compressed / "data_split").mkdir()
            official = root / "official"
            official.mkdir()
            (official / "decompress.py").write_text(
                "import argparse, shutil\n"
                "p=argparse.ArgumentParser(); p.add_argument('--data_root'); p.add_argument('--subset'); p.add_argument('--category'); a=p.parse_args()\n"
                "shutil.copytree(f'{a.data_root}/{a.subset}_compressed', f'{a.data_root}/{a.subset}')\n",
                encoding="utf-8",
            )
            relative = "everyday/Bottle/selected/fractured_0"
            manifest = {
                "samples": [{"track": "breaking_bad_everyday", "object_id": "Bottle/selected", "relative_path": relative,
                             "source_sha256": _hash_tree(selected), "size_bytes": sum(p.stat().st_size for p in selected.iterdir())}],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "retained"
            scratch = root / "scratch"
            materialize_breaking_bad(manifest_path, compressed, official, output, scratch, subsets=("everyday",))
            self.assertTrue((output / "breaking_bad_everyday" / relative).is_dir())
            self.assertTrue((output / "common_v1_manifest.json").is_file())
            self.assertFalse((output / "breaking_bad_everyday" / "everyday" / "Bottle" / "skipped").exists())
            self.assertEqual(list(scratch.iterdir()), [])

    def test_adapter_and_jsonl_evaluator_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            native = root / "native.json"
            native.write_text(json.dumps([{"sample_id": "x", "predicted_parts": [{"rotation": identity, "translation": [0, 0, 0], "part_chamfer": 0.0}], "ground_truth_parts": [{"rotation": identity, "translation": [0, 0, 0]}]}]), encoding="utf-8")
            output = root / "predictions.jsonl"
            self.assertEqual(export_predictions("cmnet", "breaking_bad_everyday", native, output), 1)
            summary = evaluate_jsonl(output, root / "summary.json")
            self.assertEqual(summary["samples"], 1)


if __name__ == "__main__":
    unittest.main()
