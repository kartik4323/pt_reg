from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from data.gpat_partnet_dataset import GpatPartNetAssemblyDataset, contact_graph_and_boundaries
from models.assembly import build_assembly_model

_prep_spec = importlib.util.spec_from_file_location(
    "prepare_partnet_gpat", Path(__file__).parents[1] / "scripts" / "prepare_partnet_gpat.py"
)
assert _prep_spec and _prep_spec.loader
_prep_module = importlib.util.module_from_spec(_prep_spec)
_prep_spec.loader.exec_module(_prep_module)
select_pilot = _prep_module.select_pilot
write_sample = _prep_module.write_sample


class PartNetGpatProtocolTests(unittest.TestCase):
    def test_contact_labels_are_symmetric_and_mark_both_boundaries(self) -> None:
        parts = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]],
                [[0.005, 0.0, 0.0], [0.005, 0.1, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.1, 0.0]],
            ],
            dtype=np.float32,
        )
        adjacency, boundary = contact_graph_and_boundaries(parts, threshold=0.01)
        self.assertTrue(adjacency[0, 1])
        self.assertTrue(adjacency[1, 0])
        self.assertFalse(adjacency[0, 2])
        self.assertTrue(boundary[0].all())
        self.assertTrue(boundary[1].all())

    def test_pilot_manifest_selection_is_fixed_and_stratified(self) -> None:
        samples = []
        for category in ("chair", "lamp", "faucet"):
            samples.extend({"id": f"{category}/train/{index:03d}", "category": category, "source_split": "train"} for index in range(16))
            samples.extend({"id": f"{category}/val/{index:03d}", "category": category, "source_split": "val"} for index in range(4))
            samples.extend({"id": f"{category}/test/{index:03d}", "category": category, "source_split": "test"} for index in range(4))
        for category in ("table", "display"):
            samples.extend({"id": f"{category}/test/{index:03d}", "category": category, "source_split": "test"} for index in range(6))
        first = select_pilot(samples, {})
        second = select_pilot(list(reversed(samples)), {})
        self.assertEqual([(item["id"], item["split"]) for item in first], [(item["id"], item["split"]) for item in second])
        self.assertEqual(sum(item["split"] == "train" for item in first), 48)
        self.assertEqual(sum(item["split"] == "val" for item in first), 12)
        self.assertEqual(sum(item["split"] == "test" for item in first), 24)

    def test_gpat_conversion_keeps_exact_pose_and_contact_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            parts = np.zeros((2, 1000, 3), dtype=np.float32)
            parts[1, :, 0] = 0.005
            np.save(source / "parts.npy", parts)
            np.save(source / "target.npy", np.concatenate((parts[0], parts[1]), axis=0)[:1000].repeat(5, axis=0))
            np.save(source / "poses.npy", np.array([[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]], dtype=np.float32))
            np.save(source / "labels.npy", np.zeros(5000, dtype=np.int64))
            np.save(source / "eq_class.npy", np.array([0, 0], dtype=np.int64))
            entry = write_sample(
                {"id": "train/Chair/7", "source_dir": source, "category": "chair", "split": "train", "num_parts": 2},
                root / "processed", overwrite=False,
            )
            with np.load(root / "processed" / entry["path"], allow_pickle=False) as converted:
                self.assertEqual(converted["canonical_parts"].shape, (2, 1000, 3))
                self.assertTrue(converted["adjacency"][0, 1])
                self.assertEqual(converted["contact_boundaries"].shape, (2, 1000))
            self.assertIn("parts.npy", entry["source_files"])
            (root / "processed" / "manifest.json").write_text(
                json.dumps({"protocol": "ours_exact", "samples": [entry]}), encoding="utf-8"
            )
            cfg = {
                "data": {"manifest": "manifest.json", "categories": ["chair"], "num_points_per_fragment": 1000, "num_points_per_object": 5000, "contact_threshold": 0.01, "seed": 1},
                "fragment": {"min_fragments": 2, "max_fragments": 20},
                "augmentation": {"random_fragment_rotation": False, "fragment_pose_translation": 0.0},
            }
            dataset = GpatPartNetAssemblyDataset(str(root / "processed"), "train", cfg)
            loaded = dataset[0]
            self.assertEqual(tuple(loaded["gpat_poses"].shape), (20, 7))
            self.assertEqual(tuple(loaded["target_labels"].shape), (5000,))

    def test_compatibility_off_has_no_graph_signal(self) -> None:
        with open("configs/partnet_gpat_pilot.yaml", "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        cfg = copy.deepcopy(cfg)
        cfg["model"]["encoder"].update({"embedding_dim": 16, "hidden_dim": 32, "num_tokens": 4, "num_neighbors": 2})
        cfg["model"]["compatibility"].update({"hidden_dims": [32], "edge_dim": 16})
        cfg["model"]["assembly"].update({"gnn_hidden_dim": 32, "gnn_layers": 1, "transformer_layers": 1, "output_points": 8, "disable_compatibility": True})
        model = build_assembly_model(cfg).eval()
        fragments = torch.randn(1, 3, 8, 3)
        mask = torch.tensor([[True, True, False]])
        output = model(fragments, mask)
        self.assertEqual(float(output.compatibility_scores.abs().max()), 0.0)
        self.assertEqual(float(output.edge_features.abs().max()), 0.0)
        self.assertEqual(float(output.interaction_features.abs().max()), 0.0)
        self.assertTrue(torch.allclose(output.node_features, output.fragment_embeddings))


if __name__ == "__main__":
    unittest.main()
