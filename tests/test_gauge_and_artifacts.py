from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from models.classical_registration import register_fragment
from models.pose import apply_transform, pose_supervised_loss
from training.two_stage_trainer import labels_in_reconstruction_frame
from utils.run_artifacts import HistoryWriter


def rotation_z(angle: float) -> torch.Tensor:
    return torch.tensor(
        [[torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle)), 0.0],
         [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle)), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


class GaugeAndArtifactTests(unittest.TestCase):
    def test_global_gauge_change_preserves_pose_labels(self) -> None:
        torch.manual_seed(7)
        source = torch.randn(1, 2, 12, 3)
        gt_rotation = torch.stack((rotation_z(0.2), rotation_z(-0.35)))[None]
        gt_translation = torch.tensor([[[0.2, -0.1, 0.3], [-0.15, 0.25, -0.2]]])
        canonical = torch.einsum("bfnd,bfcd->bfnc", source, gt_rotation) + gt_translation[:, :, None, :]
        global_rotation = rotation_z(0.7)[None]
        global_translation = torch.tensor([[0.4, -0.25, 0.1]])
        transformed, transformed_rotation, transformed_translation = labels_in_reconstruction_frame(
            canonical, gt_rotation, gt_translation, global_rotation, global_translation
        )
        from_labels = torch.einsum("bfnd,bfcd->bfnc", source, transformed_rotation) + transformed_translation[:, :, None, :]
        inverse = global_rotation.transpose(-1, -2)
        inverse_translation = -torch.einsum("bd,bdc->bc", global_translation, global_rotation)
        direct = apply_transform(canonical.reshape(1, -1, 3), inverse, inverse_translation).reshape_as(canonical)
        self.assertTrue(torch.allclose(transformed, direct, atol=1.0e-5))
        self.assertTrue(torch.allclose(from_labels, direct, atol=1.0e-5))

        # A non-zero pose prediction has exactly the same supervised loss after
        # both label and prediction are expressed in the reconstruction gauge.
        predicted_rotation = rotation_z(0.05)[None, None] @ gt_rotation
        predicted_translation = gt_translation + torch.tensor([[[0.03, -0.01, 0.02], [0.03, -0.01, 0.02]]])
        original_loss = pose_supervised_loss(
            predicted_rotation, predicted_translation, gt_rotation, gt_translation,
            torch.tensor([[True, True]]),
        )["loss"]
        inverse_rotation = global_rotation.transpose(-1, -2)
        predicted_in_recon_rotation = inverse_rotation[:, None] @ predicted_rotation
        predicted_in_recon_translation = torch.einsum(
            "bfd,bdc->bfc", predicted_translation - global_translation[:, None, :], global_rotation
        )
        gauge_loss = pose_supervised_loss(
            predicted_in_recon_rotation, predicted_in_recon_translation,
            transformed_rotation, transformed_translation, torch.tensor([[True, True]]),
        )["loss"]
        self.assertTrue(torch.allclose(original_loss, gauge_loss, atol=1.0e-5))

        # Forward composition follows the row-vector convention used by every
        # Stage-3 pose: x' = x @ R.T + t.
        composed_rotation = global_rotation[:, None] @ gt_rotation
        composed_translation = torch.einsum("bfd,bcd->bfc", gt_translation, global_rotation) + global_translation[:, None, :]
        sequential = apply_transform(
            apply_transform(source[:, 0], gt_rotation[:, 0], gt_translation[:, 0]),
            global_rotation, global_translation,
        )
        composed = apply_transform(source[:, 0], composed_rotation[:, 0], composed_translation[:, 0])
        self.assertTrue(torch.allclose(sequential, composed, atol=1.0e-5))

    def test_registration_recovers_nearby_known_transform(self) -> None:
        torch.manual_seed(3)
        source = torch.randn(256, 3)
        rotation = rotation_z(0.08)
        translation = torch.tensor([0.03, -0.02, 0.01])
        target = apply_transform(source[None], rotation[None], translation[None])[0]
        recovered_rotation, recovered_translation = register_fragment(
            source, target, method="multi_icp", icp_inits=1, icp_iters=30,
            icp_src_points=256, icp_tgt_points=256,
        )
        aligned = apply_transform(source[None], recovered_rotation[None], recovered_translation[None])[0]
        self.assertLess(float(torch.cdist(aligned[None], target[None]).min(dim=2).values.mean()), 0.01)

    def test_history_writer_appends_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            HistoryWriter(output, "stage2").append({"epoch": 1, "loss": 1.0})
            HistoryWriter(output, "stage2").append({"epoch": 2, "loss": 0.5})
            history = (output / "stage2_history.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()
