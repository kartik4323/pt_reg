"""CPU contract tests; mocked native boundaries do not certify CUDA execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scaffold_sota.models.conditioning import ConditioningBlock
from scaffold_sota.adapters.jigsaw import JigsawBackend, critical_layout, supported_match_graph
from scaffold_sota.adapters.ccs import CCSBackend
from scaffold_sota.adapters.cmnet import CMNetBackend, invariant_radial_prior
from scaffold_sota.adapters.pmtr import PMTRBackend
from scaffold_sota.adapters.matching_pairwise import native_pair_to_standard, supervised_pair
from scaffold_sota.adapters.matching_common import anchor_relative, input_parts


def prior(tokens=8):
    return {"query_xyz": torch.randn(tokens, 3), "distance": torch.linspace(-.1, .1, tokens),
            "log_scale": torch.linspace(-4, 0, tokens), "valid": torch.ones(tokens, dtype=torch.bool)}


def sample(parts=3, size=4):
    return {"points": torch.randn(3, size, 3),
            "fragment_mask": torch.arange(3) < parts, "anchor_index": torch.tensor(1),
            "canonical_points": torch.randn(3, size, 3),
            "fracture_labels": torch.ones(3, size),
            "rotations_gt": torch.eye(3).repeat(3, 1, 1),
            "translations_gt": torch.zeros(3, 3)}


class InputOnly(dict):
    def __getitem__(self, key):
        if key not in {"points", "fragment_mask", "anchor_index"}:
            raise AssertionError(f"Inference attempted to access target key {key}")
        return super().__getitem__(key)


class ConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.x = torch.randn(1, 5, 12)
        self.prior = prior()

    def test_zero_gate_preserves_native_features_and_has_gradient(self):
        for condition in ("B1", "B2", "B3", "B4"):
            block = ConditioningBlock(12, condition, 16, 4, 8)
            output = block(self.x, self.prior)
            self.assertTrue(torch.equal(output, self.x))
            output.square().sum().backward()
            self.assertTrue(torch.isfinite(block.gate.grad))
            self.assertNotEqual(float(block.gate.grad), 0.0)

    def test_equal_parameter_counts(self):
        counts = [sum(p.numel() for p in ConditioningBlock(12, c, 16, 4, 8).parameters())
                  for c in ("B1", "B2", "B3", "B4")]
        self.assertEqual(len(set(counts)), 1)

    def test_null_ignores_prior_and_pooled_tokens_have_no_local_positions(self):
        null = ConditioningBlock(12, "B1", 16, 4, 8)
        null.gate.data.fill_(1)
        self.assertTrue(torch.equal(null(self.x, None), null(self.x, {"bad": object()})))
        pooled = ConditioningBlock(12, "B2", 16, 4, 8)
        context, mask = pooled.encode_prior(self.prior, self.x)
        self.assertIsNone(mask)
        self.assertTrue(torch.equal(context[:, :1].expand_as(context), context))
        order = torch.randperm(8)
        reordered = {k: v[order] for k, v in self.prior.items()}
        self.assertTrue(torch.allclose(context, pooled.encode_prior(reordered, self.x)[0], atol=1e-6))

    def test_uncertainty_is_only_available_to_B4(self):
        changed = dict(self.prior, log_scale=self.prior["log_scale"] + 5)
        b3 = ConditioningBlock(12, "B3", 16, 4, 8)
        b3.gate.data.fill_(1)
        self.assertTrue(torch.equal(b3(self.x, self.prior), b3(self.x, changed)))
        b4 = ConditioningBlock(12, "B4", 16, 4, 8)
        b4.load_state_dict(b3.state_dict())
        self.assertFalse(torch.allclose(b4(self.x, self.prior), b4(self.x, changed)))

    def test_prior_validation_and_masked_nonfinite_tokens(self):
        block = ConditioningBlock(12, "B4", 16, 4, 8)
        with self.assertRaises(ValueError):
            block(self.x, prior(7))
        with self.assertRaises(ValueError):
            block(self.x, dict(self.prior, valid=torch.zeros(8, dtype=torch.bool)))
        masked = deepcopy(self.prior)
        masked["valid"][0] = False
        masked["query_xyz"][0] = float("nan")
        self.assertTrue(torch.isfinite(block(self.x, masked)).all())

    def test_prior_is_detached(self):
        block = ConditioningBlock(12, "B4", 16, 4, 8)
        block.gate.data.fill_(1)
        self.prior["distance"].requires_grad_(True)
        block(self.x, self.prior).sum().backward()
        self.assertIsNone(self.prior["distance"].grad)
        self.assertIsNotNone(block.field_encoder[0].weight.grad)


class PoseContracts(unittest.TestCase):
    def test_anchor_change_preserves_pairwise_geometry(self):
        r = torch.tensor([[[0., -1, 0], [1, 0, 0], [0, 0, 1]],
                          [[1., 0, 0], [0, 0, -1], [0, 1, 0]]])
        t = torch.tensor([[1., 2, 3], [4, 5, 6]])
        rr, tt = anchor_relative(r, t, 0)
        self.assertTrue(torch.allclose(rr[0], torch.eye(3)))
        self.assertTrue(torch.equal(tt[0], torch.zeros(3)))
        x = torch.randn(2, 7, 3)
        original = x @ r.transpose(-1, -2) + t[:, None]
        relative = x @ rr.transpose(-1, -2) + tt[:, None]
        expected = (original - t[0]) @ r[0]
        self.assertTrue(torch.allclose(relative, expected, atol=1e-6))

    def test_critical_layout_and_disconnected_graph(self):
        labels = torch.tensor([[1, 0, 1, 1, 1, 1, 0, 1]])
        indices, counts = critical_layout(labels, torch.tensor([[4, 4]]))
        self.assertEqual(counts.tolist(), [[3, 3]])
        self.assertEqual(indices[0, :3].tolist(), [0, 2, 3])
        perm = np.zeros((6, 6)); perm[:3, 3:] = np.eye(3)
        self.assertTrue(supported_match_graph(perm, [3, 3]))
        self.assertFalse(supported_match_graph(np.zeros((6, 6)), [3, 3]))

    def test_valid_fragments_can_be_noncontiguous(self):
        s = sample(); s["fragment_mask"] = torch.tensor([True, False, True]); s["anchor_index"] = 2
        points, indices, anchor = input_parts(s, "cpu")
        self.assertEqual(indices.tolist(), [0, 2]); self.assertEqual(anchor, 1)
        self.assertTrue(torch.equal(points, s["points"][[0, 2]]))


class FakeJigsaw(nn.Module):
    """Native boundary spy only; never available through adapter.build()."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.pc_classifier = nn.Conv1d(4, 1, 1)
        self.pc_classifier.weight.data.zero_(); self.pc_classifier.bias.data.fill_(10)
        self.pc_feat_dim, self.pc_cls_method, self.tf_layers = 4, "binary", []
        self.w_mat_loss, self.w_rig_loss = 1., 0.
        self.cfg = SimpleNamespace(MODEL=SimpleNamespace(ENCODER="boundary-spy"))

    def _extract_part_feats(self, points, lengths):
        return self.encoder(points)

    def forward(self, data):
        if not self.training:
            assert not any(k in data for k in ("gt_pcs", "part_quat", "part_trans"))
        counts = data["n_critical_pcs"][0]
        total = int(counts.sum()); matrix = torch.zeros(1, total, total)
        offset = 0
        for index in range(len(counts) - 1):
            a, b = int(counts[index]), int(counts[index + 1])
            matrix[0, offset:offset + min(a, b), offset + a:offset + a + min(a, b)] = torch.eye(min(a, b))
            offset += a
        return {"perm_mat": matrix, "features": data["part_feats"]}

    def _loss_function(self, data, output):
        assert "gt_pcs" in data
        return {"loss": output["features"].square().mean()}


class FakeRotation:
    def __init__(self, tensor, rot_type="rmat"):
        self.tensor = tensor
    def to_rmat(self):
        return self.tensor
    def convert(self, kind):
        return self


class FakeCCS(nn.Module):
    """Native boundary spy only; tests dispatch, never model performance."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.corr_module = FakeCorrelator()
        self.pose_head = nn.Linear(4, 3)
        self.pc_feat_dim, self.max_num_part, self.rot_type = 4, 20, "quat"
        self.cfg = SimpleNamespace(model=SimpleNamespace(encoder="boundary-spy", topk=10),
                                   loss=SimpleNamespace(noise_dim=0, collision_loss_C=.1, trans_loss_w=1.))

    def _extract_part_feats(self, points, valid):
        return self.encoder(points.mean(2)) * valid.unsqueeze(-1)

    def forward(self, data):
        shape = data["pre_pose_feats"].shape[:2]
        return {"rot": FakeRotation(torch.eye(3).expand(*shape, 3, 3)),
                "trans": self.pose_head(data["pre_pose_feats"])}

    def _calc_loss(self, prediction, data, C):
        return {"trans_loss": (prediction["trans"] - data["part_trans"]).square().mean()}, {}


class FakeCorrelator(nn.Module):
    def forward(self, features, mask):
        assert features.shape[1] == 20
        assert torch.equal(features[:, 3:], torch.zeros_like(features[:, 3:]))
        return features


class NativeBoundaryTests(unittest.TestCase):
    def test_jigsaw_prediction_never_requests_gt_pivot(self):
        calls = []
        def estimate(perm, points, valid, n_pcs, counts, indices, quat, trans, align_pivot):
            self.assertIsNone(quat); self.assertIsNone(trans); self.assertFalse(align_pivot)
            calls.append(1)
            count = int(valid[0])
            return {"rot": np.tile(np.eye(3), (1, count, 1, 1)),
                    "trans": np.arange(count * 3).reshape(1, count, 3).astype(float)}
        backend = JigsawBackend(FakeJigsaw(), {}, estimate)
        mock_o3d = SimpleNamespace(utility=SimpleNamespace(random=SimpleNamespace(seed=lambda _: None)))
        with patch.dict(sys.modules, {"open3d": mock_o3d}):
            result = backend.predict(InputOnly(sample()))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["rotations"].shape, (3, 3, 3))
        self.assertTrue(np.array_equal(result["translations"][1], np.zeros(3)))
        self.assertEqual(len(calls), 1)

    def test_ccs_prediction_padding_training_and_no_gt(self):
        backend = CCSBackend(FakeCCS(), {"condition": "B4", "num_tokens": 8,
                                       "token_dim": 16, "heads": 4}, FakeRotation)
        backend.conditioning.gate.data.fill_(1)
        for count in (2, 3):
            s = sample(count)
            result = backend.predict(InputOnly(s), prior())
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["rotations"].shape[0], count)
            loss = backend.loss(s, prior())["loss"]
            loss.backward()
            self.assertIsNotNone(backend.native.pose_head.weight.grad)

    def test_branch_loading_only_adds_conditioning_state(self):
        native = JigsawBackend(FakeJigsaw(), {}, lambda *args: None)
        branch = JigsawBackend(FakeJigsaw(), {"condition": "B1", "num_tokens": 8}, lambda *args: None)
        mismatch = branch.load_state_dict(native.state_dict(), strict=False)
        self.assertFalse(mismatch.unexpected_keys)
        self.assertTrue(mismatch.missing_keys)
        self.assertTrue(all(k.startswith("conditioning.") for k in mismatch.missing_keys))


class PairwiseContractTests(unittest.TestCase):
    def test_native_pair_export_convention(self):
        r = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        t = np.array([1., 2, 3]); x = np.arange(12).reshape(4, 3)
        standard_r, standard_t = native_pair_to_standard(r, t)
        native_aligned = (x + t) @ np.linalg.inv(r).T
        self.assertTrue(np.allclose(x @ standard_r.T + standard_t, native_aligned))

    def test_cmnet_pooled_context_is_invariant_to_rigid_rotation(self):
        p = prior()
        rotation = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        rotated = dict(p, query_xyz=p["query_xyz"] @ rotation.T)
        block = ConditioningBlock(12, "B2", 16, 4, 8)
        features = torch.randn(1, 4, 12)
        a = block.encode_prior(invariant_radial_prior(p), features)[0]
        b = block.encode_prior(invariant_radial_prior(rotated), features)[0]
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_cmnet_rejects_spatial_condition_before_native_loading(self):
        from scaffold_sota.adapters.cmnet import build
        for condition in ("B3", "B4"):
            with self.assertRaisesRegex(ValueError, "unsupported"):
                build("nonexistent", {"condition": condition})

    def test_native_supervised_pair_picks_overlapping_target(self):
        s = sample()
        s["canonical_points"][1] = s["canonical_points"][0]
        s["canonical_points"][2] += 100
        points, indices, _ = input_parts(s, "cpu")
        i, j, _ = supervised_pair(s, indices, points, training=False)
        self.assertEqual((i, j), (0, 1))

    def test_pmtr_prediction_strips_gt_from_pair_records(self):
        class Native(nn.Module):
            def __init__(self):
                super().__init__(); self.weight = nn.Parameter(torch.ones(1)); self.npts_per_node = 3
        def subsampling(points, lengths, stages, voxel, radius, limits):
            neighbors = torch.zeros(len(points), 1, dtype=torch.long)
            return [points] * 3, [lengths] * 3, [neighbors] * 3, [neighbors] * 2, [neighbors] * 2
        native = Native()
        backend = PMTRBackend(native, {}, subsampling, (), None)
        seen = []
        def predict_pair(data):
            self.assertFalse(any(k.startswith("gt_") for k in data))
            self.assertNotIn("pcd", data)
            seen.append(data)
            return {"estimated_rotat": torch.eye(3), "estimated_trans": torch.zeros(3),
                    "node_corr_scores": torch.ones(2)}
        def graph(outputs, count, anchor, kind, helper):
            self.assertEqual(len(outputs), count * (count - 1)); self.assertEqual(kind, "pmtr")
            return {"status": "ok", "rotations": np.tile(np.eye(3), (count, 1, 1)),
                    "translations": np.zeros((count, 3))}
        with patch.object(backend, "_predict_pair", side_effect=predict_pair), \
             patch("scaffold_sota.adapters.pmtr.native_pose_graph", side_effect=graph):
            result = backend.predict(InputOnly(sample()))
        self.assertEqual(result["status"], "ok"); self.assertEqual(len(seen), 6)


if __name__ == "__main__":
    unittest.main()
