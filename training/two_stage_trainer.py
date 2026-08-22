"""Single-process trainers for the three-stage assembly pipeline."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.assembly_dataset import build_object_dataset, build_pair_dataset
from losses.assembly_losses import build_assembly_loss
from losses.occupancy_losses import build_occupancy_loss, sample_occupancy_queries
from losses.stage1_losses import build_stage1_loss
from models.assembly import FragmentAssemblyModel, build_assembly_model
from models.compatibility_three import Stage1CompatibilityModel, build_stage1_model
from models.pose import (
    apply_transform,
    build_pose_estimator,
    build_pose_loss,
    differentiable_icp_initialization,
    pose_supervised_loss,
    rotation_geodesic_error,
)
from models.classical_registration import register_fragment
from utils.point_cloud_utils import deterministic_fps_indices, gather_points
from utils.experiment_metrics import equivalent_part_pose_metrics, symmetric_chamfer
from utils.run_artifacts import (
    HistoryWriter,
    RunLogger,
    build_results_markdown,
    summarize_distribution,
    write_manifest,
)


def _current_lrs(optimizer) -> List[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def _move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _mean_ci95(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "ci95": float("nan"), "count": 0}
    tensor = torch.as_tensor(values, dtype=torch.float64)
    ci = 0.0 if len(tensor) < 2 else float(1.96 * tensor.std(unbiased=True) / math.sqrt(len(tensor)))
    return {"mean": float(tensor.mean()), "ci95": ci, "count": int(len(tensor))}


def _subsample_points(points: torch.Tensor, count: int) -> torch.Tensor:
    """Deterministic FPS used by the memory-bounded Stage-3 protocol."""
    if count <= 0 or points.shape[1] <= count:
        return points
    return gather_points(points, deterministic_fps_indices(points, count))


def _subsample_indices(points: torch.Tensor, count: int) -> Optional[torch.Tensor]:
    if count <= 0 or points.shape[1] <= count:
        return None
    return deterministic_fps_indices(points, count)


@torch.no_grad()
def estimate_reconstruction_to_gt_frame(
    reconstruction: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    method: str = "auto",
    voxel: float = 0.05,
    icp_inits: int = 8,
    icp_iterations: int = 12,
    source_points: int = 512,
    target_points: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Register a Stage-2 target to GT, returning ``G: reconstruction -> GT``.

    Registration is intentionally detached: it establishes a coordinate gauge
    for supervision, not a path through which Stage 3 can optimise Stage 2.
    The existing FPFH+RANSAC implementation is used when Open3D is available;
    otherwise its multi-start ICP fallback is used.
    """
    rotations, translations, residuals = [], [], []
    for source, target in zip(reconstruction, ground_truth):
        try:
            rotation, translation = register_fragment(
                source,
                target,
                method=method,
                voxel=voxel,
                icp_inits=icp_inits,
                icp_iters=icp_iterations,
                icp_src_points=source_points,
                icp_tgt_points=target_points,
            )
            rotation = rotation.to(reconstruction.device, reconstruction.dtype)
            translation = translation.to(reconstruction.device, reconstruction.dtype)
            aligned = apply_transform(source.unsqueeze(0), rotation.unsqueeze(0), translation.unsqueeze(0))[0]
            # Symmetric mean Euclidean Chamfer in the normalised object frame.
            dist = torch.cdist(aligned.unsqueeze(0), target.unsqueeze(0))[0]
            residual = 0.5 * (dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean())
        except Exception:
            rotation = torch.eye(3, device=reconstruction.device, dtype=reconstruction.dtype)
            translation = torch.zeros(3, device=reconstruction.device, dtype=reconstruction.dtype)
            residual = torch.tensor(float("inf"), device=reconstruction.device, dtype=reconstruction.dtype)
        rotations.append(rotation)
        translations.append(translation)
        residuals.append(residual)
    return torch.stack(rotations), torch.stack(translations), torch.stack(residuals)


def labels_in_reconstruction_frame(
    canonical_fragments: torch.Tensor,
    gt_rotations: torch.Tensor,
    gt_translations: torch.Tensor,
    reconstruction_to_gt_rotation: torch.Tensor,
    reconstruction_to_gt_translation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Express every GT-derived label in Stage 2's target coordinate frame.

    If ``x_gt = R_g x_recon + t_g``, then canonical points and poses are
    transformed by ``G^-1`` before they are compared to a reconstructed target.
    This fixes the gauge error where an otherwise correct global Stage-2 pose
    produced contradictory segmentation, correspondence, and RT supervision.
    """
    r_global = reconstruction_to_gt_rotation
    t_global = reconstruction_to_gt_translation
    r_inverse = r_global.transpose(-1, -2)
    t_inverse = -torch.einsum("bd,bdc->bc", t_global, r_global)
    canonical = torch.einsum("bfnd,bdc->bfnc", canonical_fragments, r_global)
    canonical = canonical + t_inverse[:, None, None, :]
    rotations = r_inverse[:, None] @ gt_rotations
    translations = torch.einsum("bfd,bdc->bfc", gt_translations - t_global[:, None, :], r_global)
    return canonical, rotations, translations


class Stage1Trainer:
    """Compatibility pretraining with binary BCE + InfoNCE objective."""

    def __init__(self, cfg: dict, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.out_dir = Path(cfg["output"]["dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.model = build_stage1_model(cfg).to(device)
        self.criterion = build_stage1_loss(cfg).to(device)
        stage_cfg = cfg.get("stage1", {})
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=stage_cfg.get("learning_rate", 1.0e-4),
            weight_decay=stage_cfg.get("weight_decay", 1.0e-4),
        )

        dataset = build_pair_dataset(cfg, "train", stage_cfg.get("epoch_size"))
        self.loader = DataLoader(
            dataset,
            batch_size=stage_cfg.get("batch_size", 8),
            shuffle=True,
            num_workers=stage_cfg.get("num_workers", 0),
            drop_last=True,
        )
        self.epochs = stage_cfg.get("epochs", 5)
        self.grad_clip = stage_cfg.get("gradient_clip", 1.0)

    def train(self) -> Path:
        logger = RunLogger(self.out_dir, "stage1")
        history = HistoryWriter(self.out_dir, "stage1")
        write_manifest(
            self.out_dir,
            "stage1",
            self.cfg,
            self.device,
            extra={"epochs": self.epochs, "num_batches_per_epoch": len(self.loader)},
        )
        metrics: Dict[str, float] = {}
        for epoch in range(self.epochs):
            start = time.perf_counter()
            metrics = self._train_epoch(epoch)
            elapsed = time.perf_counter() - start
            history.append(
                {"epoch": epoch + 1, "seconds": elapsed, "lr": _current_lrs(self.optimizer), **metrics}
            )
            logger.log(
                "Stage 1 "
                f"epoch={epoch + 1}/{self.epochs} "
                f"loss={metrics['loss']:.4f} "
                f"bce={metrics['loss_bce']:.4f} "
                f"contrast={metrics['loss_contrast']:.4f}"
            )

        ckpt_path = self.out_dir / "stage1_pretrained.pt"
        torch.save(
            {
                "encoder": self.model.encoder.state_dict(),
                "compatibility": self.model.scorer.state_dict(),
                "metrics": metrics,
                "cfg": self.cfg,
            },
            ckpt_path,
        )
        _save_json(self.out_dir / "stage1_metrics.json", metrics)
        build_results_markdown(self.out_dir)
        return ckpt_path

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {}
        steps = 0
        for batch in tqdm(self.loader, desc=f"Stage1 epoch {epoch + 1}", leave=False):
            batch = _move_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(
                batch["frag_a"],
                batch["frag_b"],
            )
            losses = self.criterion(
                output,
                labels=batch["label"],
                boundary_a=batch.get("boundary_a"),
                boundary_b=batch.get("boundary_b"),
            )
            losses["loss"].backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            steps += 1

        return {key: value / max(steps, 1) for key, value in totals.items()}


class Stage2Trainer:
    """Graph assembly training with frozen-then-finetuned pretrained modules."""

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        stage1_checkpoint: Optional[str] = None,
        freeze_pretrained: bool = True,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.out_dir = Path(cfg["output"]["dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.model = build_assembly_model(cfg).to(device)
        self.reference_model: Optional[FragmentAssemblyModel] = None
        if stage1_checkpoint is not None:
            self.load_stage1(stage1_checkpoint)
            if cfg.get("loss", {}).get("stage2", {}).get("lambda_compat", 0.0) > 0:
                self.reference_model = build_assembly_model(cfg).to(device)
                self.reference_model.encoder.load_state_dict(self.model.encoder.state_dict())
                self.reference_model.compatibility.load_state_dict(
                    self.model.compatibility.state_dict()
                )
                self.reference_model.eval()
                self.reference_model.freeze_pretrained()
        if freeze_pretrained and stage1_checkpoint is None:
            print(
                "WARNING [Stage 2]: freeze_pretrained requested but no Stage 1 "
                "checkpoint was loaded. Freezing a randomly-initialized encoder "
                "would leave it untrained forever; training the encoder from "
                "scratch instead. Provide --stage1-checkpoint to freeze."
            )
            freeze_pretrained = False
        if freeze_pretrained:
            self.model.freeze_pretrained()

        assembly_cfg = cfg.get("model", {}).get("assembly", {})
        self.occupancy_mode = assembly_cfg.get("decoder", "points") == "occupancy"
        self.occupancy_queries = int(assembly_cfg.get("occupancy_train_queries", 4096))
        self.criterion = (
            build_occupancy_loss(cfg) if self.occupancy_mode else build_assembly_loss(cfg)
        ).to(device)
        stage_cfg = cfg.get("stage2", {})
        self.stage_cfg = stage_cfg
        self.freeze_pretrained = freeze_pretrained
        self.optimizer = self._build_optimizer()
        dataset = build_object_dataset(cfg, "train", stage_cfg.get("epoch_size"))
        self.loader = DataLoader(
            dataset,
            batch_size=stage_cfg.get("batch_size", 4),
            shuffle=True,
            num_workers=stage_cfg.get("num_workers", 0),
            drop_last=True,
        )
        self.epochs = stage_cfg.get("epochs", 5)
        self.grad_clip = stage_cfg.get("gradient_clip", 1.0)
        self.use_subset = cfg.get("loss", {}).get("stage2", {}).get("lambda_consistency", 0.0) > 0
        self.use_dropout = cfg.get("loss", {}).get("stage2", {}).get("lambda_dropout", 0.0) > 0
        self.unfreeze_epoch = stage_cfg.get("unfreeze_pretrained_epoch")
        requested_validation = int(stage_cfg.get("validation_max_samples", 0))
        self.validation_dataset = build_object_dataset(cfg, "val", requested_validation or None)
        self.validation_interval = max(1, int(stage_cfg.get("validation_interval", 1)))

    def load_stage1(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.encoder.load_state_dict(checkpoint["encoder"], strict=True)
        self.model.compatibility.load_state_dict(checkpoint["compatibility"], strict=True)

    def train(self) -> Path:
        logger = RunLogger(self.out_dir, "stage2")
        history = HistoryWriter(self.out_dir, "stage2")
        write_manifest(
            self.out_dir,
            "stage2",
            self.cfg,
            self.device,
            extra={
                "epochs": self.epochs,
                "num_batches_per_epoch": len(self.loader),
                "freeze_pretrained": self.freeze_pretrained,
                "unfreeze_pretrained_epoch": self.unfreeze_epoch,
            },
        )
        metrics: Dict[str, float] = {}
        best_validation = float("inf")
        validation_history: List[dict] = []
        for epoch in range(self.epochs):
            if (
                self.freeze_pretrained
                and self.unfreeze_epoch is not None
                and epoch == int(self.unfreeze_epoch)
            ):
                self.model.unfreeze_pretrained()
                self.freeze_pretrained = False
                self.optimizer = self._build_optimizer()
                logger.log(f"Stage 2 unfroze pretrained modules at epoch {epoch + 1}")
            start = time.perf_counter()
            metrics = self._train_epoch(epoch)
            elapsed = time.perf_counter() - start
            if (epoch + 1) % self.validation_interval == 0 or epoch + 1 == self.epochs:
                from utils.experiment_metrics import evaluate_stage2

                validation = evaluate_stage2(
                    self.model, self.validation_dataset, self.device, batch_size=1
                )
                validation_score = validation["aligned_reconstruction_chamfer"]
                validation_history.append({"epoch": epoch + 1, **validation})
                metrics["val_aligned_reconstruction_chamfer"] = validation_score
                if validation_score < best_validation:
                    best_validation = validation_score
                    torch.save(
                        {
                            "model": self.model.state_dict(),
                            "metrics": metrics,
                            "validation": validation,
                            "cfg": self.cfg,
                        },
                        self.out_dir / "stage2_best_validation.pt",
                    )
            history.append(
                {"epoch": epoch + 1, "seconds": elapsed, "lr": _current_lrs(self.optimizer), **metrics}
            )
            if self.occupancy_mode:
                logger.log(
                    "Stage 2 "
                    f"epoch={epoch + 1}/{self.epochs} "
                    f"loss={metrics['loss']:.4f} "
                    f"bce={metrics.get('occupancy_bce', 0.0):.4f} "
                    f"IoU={metrics.get('occupancy_iou', 0.0):.4f} "
                    f"acc={metrics.get('occupancy_accuracy', 0.0):.4f} "
                    f"conf={metrics.get('loss_confidence', 0.0):.4f} "
                    f"occ_frac={metrics.get('occupancy_target_frac', 0.0):.3f}"
                )
            else:
                logger.log(
                    "Stage 2 "
                    f"epoch={epoch + 1}/{self.epochs} "
                    f"loss={metrics['loss']:.4f} "
                    f"cd={metrics['loss_cd']:.4f} "
                    f"cov={metrics['loss_coverage']:.4f}"
                )

        ckpt_path = self.out_dir / "stage2_assembly.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "metrics": metrics,
                "cfg": self.cfg,
            },
            ckpt_path,
        )
        _save_json(self.out_dir / "stage2_metrics.json", metrics)
        _save_json(
            self.out_dir / "stage2_validation.json",
            {
                "selection_metric": "aligned_reconstruction_chamfer",
                "best": best_validation,
                "history": validation_history,
            },
        )
        build_results_markdown(self.out_dir)
        return ckpt_path

    def _build_optimizer(self) -> AdamW:
        lr = self.stage_cfg.get("learning_rate", 1.0e-4)
        pretrained_lr = self.stage_cfg.get("pretrained_learning_rate", lr)
        weight_decay = self.stage_cfg.get("weight_decay", 1.0e-4)

        pretrained_params = []
        pretrained_ids = set()
        for module in (self.model.encoder, self.model.compatibility):
            for param in module.parameters():
                if param.requires_grad:
                    pretrained_params.append(param)
                    pretrained_ids.add(id(param))

        new_params = [
            param
            for param in self.model.parameters()
            if param.requires_grad and id(param) not in pretrained_ids
        ]

        groups = []
        if new_params:
            groups.append({"params": new_params, "lr": lr})
        if pretrained_params:
            groups.append({"params": pretrained_params, "lr": pretrained_lr})
        if not groups:
            raise RuntimeError("Stage 2 has no trainable parameters")
        return AdamW(groups, weight_decay=weight_decay)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {}
        steps = 0

        for batch in tqdm(self.loader, desc=f"Stage2 epoch {epoch + 1}", leave=False):
            batch = _move_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)

            if self.occupancy_mode:
                # Sample query points once and reuse them for the full / subset /
                # dropout passes so the three predictions are directly comparable.
                query_xyz, occ_labels = sample_occupancy_queries(
                    batch["occupancy"], num_queries=self.occupancy_queries
                )
                output = self.model(batch["fragments"], batch["fragment_mask"], query_xyz=query_xyz)
            else:
                output = self.model(batch["fragments"], batch["fragment_mask"])
            with torch.no_grad():
                if self.reference_model is not None:
                    reference_scores = self.reference_model.compatibility_scores_only(
                        batch["fragments"], batch["fragment_mask"]
                    )
                else:
                    reference_scores = output.compatibility_scores.detach().clone()

            subset_pred = None
            if self.use_subset:
                subset_mask = self._random_subset_mask(batch["fragment_mask"])
                if self.occupancy_mode:
                    subset_pred = self.model(
                        batch["fragments"], subset_mask, query_xyz=query_xyz
                    ).occupancy_logits
                else:
                    subset_pred = self.model(batch["fragments"], subset_mask).point_cloud

            dropout_pred = None
            if self.use_dropout:
                dropout_mask = self._drop_one_mask(batch["fragment_mask"])
                if self.occupancy_mode:
                    dropout_pred = self.model(
                        batch["fragments"], dropout_mask, query_xyz=query_xyz
                    ).occupancy_logits
                else:
                    dropout_pred = self.model(batch["fragments"], dropout_mask).point_cloud

            if self.occupancy_mode:
                losses = self.criterion(
                    occupancy_logits=output.occupancy_logits,
                    occupancy_labels=occ_labels,
                    occupancy_confidence=output.occupancy_confidence,
                    current_scores=output.compatibility_scores,
                    reference_scores=reference_scores,
                    fragment_mask=batch["fragment_mask"],
                    subset_logits=subset_pred,
                    dropout_logits=dropout_pred,
                )
            else:
                losses = self.criterion(
                    pred=output.point_cloud,
                    target=batch["target"],
                    target_boundary=batch["target_boundary"],
                    canonical_fragments=batch["canonical_fragments"],
                    fragment_mask=batch["fragment_mask"],
                    current_scores=output.compatibility_scores,
                    reference_scores=reference_scores,
                    subset_pred=subset_pred,
                    dropout_pred=dropout_pred,
                )
            losses["loss"].backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            steps += 1

        return {key: value / max(steps, 1) for key, value in totals.items()}

    @staticmethod
    def _random_subset_mask(mask: torch.Tensor) -> torch.Tensor:
        keep = (torch.rand_like(mask.float()) > 0.35) & mask
        empty = keep.sum(dim=1) == 0
        if empty.any():
            first = mask.float().argmax(dim=1)
            keep[empty, first[empty]] = True
        return keep

    @staticmethod
    def _drop_one_mask(mask: torch.Tensor) -> torch.Tensor:
        keep = mask.clone()
        for batch_idx in range(mask.shape[0]):
            valid = torch.where(mask[batch_idx])[0]
            if len(valid) > 1:
                drop_idx = valid[torch.randint(len(valid), (1,), device=mask.device)]
                keep[batch_idx, drop_idx] = False
        return keep


class Stage3PoseTrainer:
    """Train Part 3 fragment-to-object pose estimation with Stage 2 frozen."""

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        stage2_checkpoint: str,
        freeze_reconstruction: bool = True,
        target_source: str = "reconstruction",
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.out_dir = Path(cfg["output"]["dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stage_cfg = cfg.get("stage3", {})
        self.freeze_reconstruction = freeze_reconstruction
        if target_source not in {"reconstruction", "ground_truth"}:
            raise ValueError("target_source must be 'reconstruction' or 'ground_truth'")
        self.target_source = target_source
        if cfg.get("model", {}).get("pose", {}).get("architecture") == "classical":
            raise ValueError(
                "model.pose.architecture='classical' is training-free (no parameters). "
                "Skip Stage-3 training and evaluate/infer directly, e.g. "
                "--mode stage3-gt / --mode stage3-eval, or scripts/infer_assembly.py "
                "--pose-mode classical."
            )

        # Build and load the stage-2 checkpoint on CPU first to reduce peak GPU
        # memory usage, then move the model to the target device once.
        self.reconstruction_model = build_assembly_model(cfg)
        checkpoint = torch.load(stage2_checkpoint, map_location="cpu")
        self.reconstruction_model.load_state_dict(checkpoint["model"], strict=True)
        self.reconstruction_model.to(device)
        self.reconstruction_model.eval()
        if freeze_reconstruction:
            for param in self.reconstruction_model.parameters():
                param.requires_grad = False

        self.pose_model = build_pose_estimator(cfg).to(device)
        self._load_encoder_weights_from_stage2()
        self.criterion = build_pose_loss(cfg).to(device)
        self.optimizer = AdamW(
            [param for param in self.pose_model.parameters() if param.requires_grad],
            lr=self.stage_cfg.get("learning_rate", 1.0e-4),
            weight_decay=self.stage_cfg.get("weight_decay", 1.0e-4),
        )

        # AMP: enable only on CUDA devices
        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        dataset = build_object_dataset(cfg, "train", self.stage_cfg.get("epoch_size", 512))
        self.loader = DataLoader(
            dataset,
            batch_size=self.stage_cfg.get("batch_size", 2),
            shuffle=True,
            num_workers=self.stage_cfg.get("num_workers", 0),
            drop_last=True,
        )
        self.epochs = self.stage_cfg.get("epochs", 10)
        self.grad_clip = self.stage_cfg.get("gradient_clip", 1.0)
        # Effective batch = batch_size * grad_accum_steps. Lets a 24 GB card keep
        # batch_size=1 (memory-bound by the dense object attention) while still
        # taking optimizer steps over several fragments' worth of gradient.
        self.grad_accum_steps = max(1, int(self.stage_cfg.get("grad_accum_steps", 1)))
        self.target_feature_points = int(self.stage_cfg.get("target_feature_points", 0))
        self.geometry_loss_points = int(self.stage_cfg.get("geometry_loss_points", 0))
        self.gauge_aware = bool(self.stage_cfg.get("gauge_aware", False))
        self.registration_max_residual = float(self.stage_cfg.get("registration_max_residual", 0.02))
        self.registration_cfg = self.stage_cfg.get("global_registration", {})
        self.registration_records: List[dict] = []
        self.initialization_icp_iterations = self.stage_cfg.get(
            "initialization_icp_iterations",
            min(3, self.stage_cfg.get("icp_iterations", 3)),
        )
        if cfg.get("model", {}).get("pose", {}).get("architecture", "target_segmentation") == "target_segmentation":
            self.initialization_icp_iterations = 0
        self.pose_only_epochs = int(self.stage_cfg.get("pose_only_epochs", max(1, self.epochs // 3)))
        self.geometry_epochs = int(self.stage_cfg.get("geometry_epochs", max(1, self.epochs // 3)))

    def _load_encoder_weights_from_stage2(self) -> None:
        self.pose_model.fragment_encoder.load_state_dict(
            self.reconstruction_model.encoder.state_dict(),
            strict=True,
        )
        self.pose_model.object_encoder.load_state_dict(
            self.reconstruction_model.encoder.state_dict(),
            strict=True,
        )
        if self.pose_model.freeze_encoders:
            self.pose_model.freeze_pretrained()

    def train(self) -> Path:
        stage_tag = "stage3_gt_target" if self.target_source == "ground_truth" else "stage3"
        logger = RunLogger(self.out_dir, stage_tag)
        history = HistoryWriter(self.out_dir, stage_tag)
        write_manifest(
            self.out_dir,
            stage_tag,
            self.cfg,
            self.device,
            extra={
                "epochs": self.epochs,
                "num_batches_per_epoch": len(self.loader),
                "target_source": self.target_source,
                "freeze_reconstruction": self.freeze_reconstruction,
                "pose_architecture": self.cfg.get("model", {}).get("pose", {}).get("architecture"),
                "target_feature_points": self.target_feature_points,
                "geometry_loss_points": self.geometry_loss_points,
                "gauge_aware": self.gauge_aware,
            },
        )
        metrics: Dict[str, float] = {}
        for epoch in range(self.epochs):
            start = time.perf_counter()
            metrics = self._train_epoch(epoch)
            elapsed = time.perf_counter() - start
            history.append(
                {
                    "epoch": epoch + 1,
                    "seconds": elapsed,
                    "lr": _current_lrs(self.optimizer),
                    "weights": self._loss_weights_for_epoch(epoch),
                    **metrics,
                }
            )
            logger.log(
                "Stage 3 "
                f"epoch={epoch + 1}/{self.epochs} "
                f"loss={metrics['loss']:.4f} "
                f"seg={metrics['loss_matching']:.4f} "
                f"corr={metrics.get('loss_correspondence', 0.0):.4f} "
                f"corr_acc={metrics.get('correspondence_accuracy', 0.0):.3f} "
                f"align={metrics.get('loss_align', 0.0):.4f} "
                f"pose={metrics.get('loss_pose', 0.0):.4f} "
                f"seg_acc={metrics.get('matching_accuracy', 0.0):.3f} "
                f"recon_cd={metrics['loss_recon_cd']:.4f} "
                f"rot_deg={metrics.get('rotation_error_deg', 0.0):.2f} "
                f"trans={metrics.get('translation_error', 0.0):.4f} "
                f"overlap={metrics.get('loss_overlap', 0.0):.4f}"
            )
            if self.registration_records:
                diagnostics_path = self.out_dir / "stage3_registration_diagnostics.jsonl"
                with open(diagnostics_path, "a", encoding="utf-8") as handle:
                    for record in self.registration_records:
                        handle.write(json.dumps(record) + "\n")
                self.registration_records.clear()

        ckpt_name = "stage3_pose_gt_target.pt" if self.target_source == "ground_truth" else "stage3_pose.pt"
        ckpt_path = self.out_dir / ckpt_name
        torch.save(
            {
                "model": self.pose_model.state_dict(),
                "metrics": metrics,
                "cfg": self.cfg,
                "target_source": self.target_source,
            },
            ckpt_path,
        )
        metrics_name = (
            "stage3_pose_gt_target_metrics.json"
            if self.target_source == "ground_truth"
            else "stage3_pose_metrics.json"
        )
        _save_json(self.out_dir / metrics_name, metrics)
        build_results_markdown(self.out_dir)
        return ckpt_path

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.pose_model.train()
        self.reconstruction_model.eval()
        totals: Dict[str, float] = {}
        steps = 0

        accum = self.grad_accum_steps
        num_batches = len(self.loader)
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(
            tqdm(self.loader, desc=f"Stage3 epoch {epoch + 1}", leave=False)
        ):
            batch = _move_to_device(batch, self.device)
            is_step_boundary = ((batch_idx + 1) % accum == 0) or (batch_idx + 1 == num_batches)

            # Use AMP context for forward and loss computation to reduce memory
            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", enabled=True)
                if self.use_amp
                else nullcontext()
            )
            with autocast_ctx:
                if self.freeze_reconstruction:
                    with torch.no_grad():
                        reconstruction = self.reconstruction_model(
                            batch["fragments"], batch["fragment_mask"]
                        ).point_cloud
                else:
                    reconstruction = self.reconstruction_model(
                        batch["fragments"], batch["fragment_mask"]
                    ).point_cloud

                if self.target_source == "ground_truth":
                    target_object_full = batch["target"]
                else:
                    target_object_full = reconstruction
                target_indices = _subsample_indices(target_object_full, self.target_feature_points)
                target_object = target_object_full if target_indices is None else gather_points(target_object_full, target_indices)

                target_fragments = batch["canonical_fragments"]
                gt_rotations = batch["align_rotations"]
                gt_translations = batch["align_translations"]
                target_segmentation_labels = None
                if self.target_source == "ground_truth" and "target_labels" in batch:
                    target_segmentation_labels = batch["target_labels"]
                    if target_indices is not None:
                        target_segmentation_labels = torch.gather(target_segmentation_labels, 1, target_indices)
                ground_truth_for_loss = batch["target"]
                loss_weights = self._loss_weights_for_epoch(epoch)
                registration_residual = target_object.new_zeros(target_object.shape[0])
                registration_reliable = torch.ones_like(registration_residual, dtype=torch.bool)
                if self.target_source == "reconstruction" and self.gauge_aware:
                    global_rotation, global_translation, registration_residual = estimate_reconstruction_to_gt_frame(
                        reconstruction.detach(),
                        batch["target"].detach(),
                        method=self.registration_cfg.get("method", "auto"),
                        voxel=float(self.registration_cfg.get("voxel", 0.05)),
                        icp_inits=int(self.registration_cfg.get("icp_inits", 8)),
                        icp_iterations=int(self.registration_cfg.get("icp_iterations", 12)),
                        source_points=int(self.registration_cfg.get("source_points", 512)),
                        target_points=int(self.registration_cfg.get("target_points", 1024)),
                    )
                    target_fragments, gt_rotations, gt_translations = labels_in_reconstruction_frame(
                        batch["canonical_fragments"],
                        batch["align_rotations"],
                        batch["align_translations"],
                        global_rotation,
                        global_translation,
                    )
                    # Reconstructed target points have no index-preserving
                    # correspondence to GPAT's target cloud.  The loss derives
                    # their labels from the gauge-transformed canonical parts,
                    # which is the correctly transformed dense supervision.
                    target_segmentation_labels = None
                    inverse_rotation = global_rotation.transpose(-1, -2)
                    inverse_translation = -torch.einsum(
                        "bd,bdc->bc", global_translation, global_rotation
                    )
                    ground_truth_for_loss = apply_transform(
                        batch["target"], inverse_rotation, inverse_translation
                    )
                    registration_reliable = registration_residual <= self.registration_max_residual
                    object_ids = batch.get("object_id", [f"batch_{batch_idx}_{idx}" for idx in range(target_object.shape[0])])
                    for record_idx, object_id in enumerate(object_ids):
                        self.registration_records.append(
                            {
                                "epoch": epoch + 1,
                                "object_id": str(object_id),
                                "rotation": global_rotation[record_idx].detach().float().cpu().tolist(),
                                "translation": global_translation[record_idx].detach().float().cpu().tolist(),
                                "residual": float(registration_residual[record_idx].detach().cpu()),
                                "reliable": bool(registration_reliable[record_idx].detach().cpu()),
                                "threshold": self.registration_max_residual,
                            }
                        )
                    # With batch size 1 (the enforced 24-GB configuration), a
                    # failed registration must not inject invalid GT labels. The
                    # reconstruction-frame geometry terms continue training.
                    if not bool(registration_reliable.all()):
                        loss_weights = dict(loss_weights)
                        loss_weights.update(
                            {
                                "lambda_matching": 0.0,
                                "lambda_segmentation": 0.0,
                                "lambda_pose": 0.0,
                                "lambda_correspondence": 0.0,
                                "lambda_align": 0.0,
                                "lambda_gt": 0.0,
                            }
                        )

                if self.initialization_icp_iterations > 0:
                    init_rotations, init_translations, _ = differentiable_icp_initialization(
                        batch["fragments"],
                        target_object.detach(),
                        batch["fragment_mask"],
                        iterations=self.initialization_icp_iterations,
                    )
                else:
                    init_rotations = None
                    init_translations = None

                output = self.pose_model(
                    batch["fragments"],
                    target_object,
                    batch["fragment_mask"],
                    initial_rotations=init_rotations,
                    initial_translations=init_translations,
                )
                criterion_kwargs = dict(
                    output=output,
                    fragments=batch["fragments"],
                    fragment_mask=batch["fragment_mask"],
                    reconstructed_object=target_object.detach() if self.freeze_reconstruction else target_object,
                    ground_truth_object=_subsample_points(ground_truth_for_loss, self.geometry_loss_points),
                    target_fragments=target_fragments,
                    gt_rotations=gt_rotations,
                    gt_translations=gt_translations,
                    weights=loss_weights,
                )
                if target_segmentation_labels is not None and self.cfg.get("model", {}).get("pose", {}).get("architecture", "target_segmentation") == "target_segmentation":
                    criterion_kwargs["target_segmentation_labels"] = target_segmentation_labels
                losses = self.criterion(**criterion_kwargs)

            losses["registration_residual"] = registration_residual.mean().detach()
            losses["registration_reliable_frac"] = registration_reliable.to(target_object.dtype).mean().detach()

            # Scale the loss so accumulated gradients average over the micro-batches.
            loss = losses["loss"] / accum
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if is_step_boundary:
                    if self.grad_clip > 0:
                        # unscale before clipping
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.pose_model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if is_step_boundary:
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.pose_model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            # Report the unscaled per-batch losses (not the /accum training loss).
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            steps += 1

        return {key: value / max(steps, 1) for key, value in totals.items()}

    def _loss_weights_for_epoch(self, epoch: int) -> Dict[str, float]:
        base = self.cfg.get("loss", {}).get("stage3", {})
        final_recon = base.get("lambda_recon", 1.0)
        final_coverage = base.get("lambda_coverage", 1.0)
        final_gt = base.get("lambda_gt", 0.0)
        final_overlap = base.get("lambda_overlap", 0.1)
        final_matching = base.get("lambda_matching", 2.0)
        final_pose = base.get("lambda_pose", 0.0)
        final_correspondence = base.get("lambda_correspondence", 0.0)
        final_align = base.get("lambda_align", 0.0)
        weights = {
            "lambda_matching": final_matching,
            "lambda_recon": final_recon,
            "lambda_coverage": final_coverage,
            "lambda_gt": final_gt,
            "lambda_overlap": final_overlap,
            "lambda_pose": final_pose,
            "lambda_correspondence": final_correspondence,
            "lambda_align": final_align,
        }

        if epoch < self.pose_only_epochs:
            weights.update(
                {
                    "lambda_matching": base.get("lambda_matching_early", final_matching * 2.0),
                    "lambda_recon": 0.0,
                    "lambda_coverage": 0.0,
                    "lambda_gt": 0.0,
                    "lambda_overlap": 0.0,
                    "lambda_correspondence": base.get(
                        "lambda_correspondence_early", final_correspondence * 1.5
                    ),
                    "lambda_align": base.get("lambda_align_early", final_align * 1.5),
                }
            )
        elif epoch < self.pose_only_epochs + self.geometry_epochs:
            mid_progress = (
                (epoch - self.pose_only_epochs + 1)
                / max(self.geometry_epochs, 1)
            )
            weights.update(
                {
                    "lambda_matching": base.get("lambda_matching_mid", final_matching),
                    "lambda_recon": base.get("lambda_recon_mid", final_recon * 0.5 * mid_progress),
                    "lambda_coverage": base.get("lambda_coverage_mid", final_coverage * 0.5 * mid_progress),
                    "lambda_gt": base.get("lambda_gt_mid", final_gt * 0.5 * mid_progress),
                    "lambda_overlap": base.get("lambda_overlap_mid", final_overlap * 0.5 * mid_progress),
                    "lambda_pose": base.get("lambda_pose_mid", final_pose),
                    "lambda_correspondence": base.get("lambda_correspondence_mid", final_correspondence),
                    "lambda_align": base.get("lambda_align_mid", final_align),
                }
            )
        else:
            late_epochs = max(self.epochs - self.pose_only_epochs - self.geometry_epochs, 1)
            late_progress = min(
                1.0,
                (epoch - self.pose_only_epochs - self.geometry_epochs + 1) / late_epochs,
            )
            weights.update(
                {
                    "lambda_matching": base.get("lambda_matching_late", final_matching * 0.5),
                    "lambda_recon": final_recon * late_progress,
                    "lambda_coverage": final_coverage * late_progress,
                    "lambda_gt": final_gt * late_progress,
                    "lambda_overlap": final_overlap * late_progress,
                    "lambda_pose": base.get("lambda_pose_late", final_pose),
                    "lambda_correspondence": base.get("lambda_correspondence_late", final_correspondence),
                    "lambda_align": base.get("lambda_align_late", final_align),
                }
            )
        return weights


@torch.no_grad()
def run_pose_stage(
    cfg: dict,
    device: torch.device,
    stage2_checkpoint: str,
    stage3_checkpoint: Optional[str] = None,
    split: str = "test",
    icp_iterations: int = 5,
    target_source: str = "reconstruction",
    prediction_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Evaluate a Stage-3 checkpoint in GT, raw-reconstruction, or oracle frame.

    ``oracle_reconstruction`` only changes the *test target* by registering the
    Stage-2 cloud to the PartNet target.  It is never used for training and
    therefore separates target fidelity from global-frame ambiguity.
    """
    if target_source not in {"reconstruction", "ground_truth", "oracle_reconstruction"}:
        raise ValueError("target_source must be reconstruction, ground_truth, or oracle_reconstruction")
    model = build_assembly_model(cfg).to(device)
    checkpoint = torch.load(stage2_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    pose_model = None
    stage_cfg = cfg.get("stage3", {})
    architecture = cfg.get("model", {}).get("pose", {}).get("architecture", "target_segmentation")
    initialization_icp_iterations = stage_cfg.get(
        "initialization_icp_iterations",
        min(3, stage_cfg.get("icp_iterations", icp_iterations)),
    )
    if architecture in {"target_segmentation", "direct_regression", "geotransformer", "classical"}:
        initialization_icp_iterations = 0
    if architecture == "classical":
        # Training-free Stage 3: no checkpoint exists or is needed.
        pose_model = build_pose_estimator(cfg).to(device)
        pose_model.eval()
    elif stage3_checkpoint is not None:
        pose_model = build_pose_estimator(cfg).to(device)
        pose_checkpoint = torch.load(stage3_checkpoint, map_location=device)
        state_dict = pose_checkpoint["model"] if "model" in pose_checkpoint else pose_checkpoint
        pose_model.load_state_dict(state_dict, strict=True)
        pose_model.eval()

    requested = cfg.get("evaluation", {}).get("max_samples", 0)
    dataset = build_object_dataset(cfg, split, requested or None)
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("stage3", {}).get("batch_size", 2),
        shuffle=False,
        num_workers=0,
    )

    totals: Dict[str, float] = {}
    steps = 0
    rot_deg_all: List[float] = []
    trans_all: List[float] = []
    global_rot_deg_all: List[float] = []
    global_trans_all: List[float] = []
    equivalent_rot_deg_all: List[float] = []
    equivalent_trans_all: List[float] = []
    part_chamfers: List[float] = []
    successes: List[float] = []
    target_chamfers: List[float] = []
    gt_chamfers: List[float] = []
    registrations: List[float] = []
    saved_predictions: List[dict] = []
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="Stage3 pose", leave=False):
        batch = _move_to_device(batch, device)
        output = model(batch["fragments"], batch["fragment_mask"])
        batch_size = batch["fragments"].shape[0]
        global_rotation = torch.eye(3, device=device, dtype=output.point_cloud.dtype)[None].repeat(batch_size, 1, 1)
        global_translation = torch.zeros(batch_size, 3, device=device, dtype=output.point_cloud.dtype)
        prediction_to_gt_rotation = global_rotation
        prediction_to_gt_translation = global_translation
        registration_residual = torch.zeros(batch_size, device=device, dtype=output.point_cloud.dtype)
        if target_source == "ground_truth":
            target_object = batch["target"]
            expected_rotations = batch["align_rotations"]
            expected_translations = batch["align_translations"]
        else:
            if target_source == "oracle_reconstruction":
                global_rotation, global_translation, registration_residual = estimate_reconstruction_to_gt_frame(
                    output.point_cloud,
                    batch["target"],
                    method=stage_cfg.get("global_registration", {}).get("method", "auto"),
                    voxel=float(stage_cfg.get("global_registration", {}).get("voxel", 0.05)),
                    icp_inits=int(stage_cfg.get("global_registration", {}).get("icp_inits", 8)),
                    icp_iterations=int(stage_cfg.get("global_registration", {}).get("icp_iterations", 12)),
                    source_points=int(stage_cfg.get("global_registration", {}).get("source_points", 512)),
                    target_points=int(stage_cfg.get("global_registration", {}).get("target_points", 1024)),
                )
                target_object = apply_transform(output.point_cloud, global_rotation, global_translation)
                expected_rotations = batch["align_rotations"]
                expected_translations = batch["align_translations"]
            else:
                target_object = output.point_cloud
                global_rotation, global_translation, registration_residual = estimate_reconstruction_to_gt_frame(
                    output.point_cloud,
                    batch["target"],
                    method=stage_cfg.get("global_registration", {}).get("method", "auto"),
                    voxel=float(stage_cfg.get("global_registration", {}).get("voxel", 0.05)),
                    icp_inits=int(stage_cfg.get("global_registration", {}).get("icp_inits", 8)),
                    icp_iterations=int(stage_cfg.get("global_registration", {}).get("icp_iterations", 12)),
                    source_points=int(stage_cfg.get("global_registration", {}).get("source_points", 512)),
                    target_points=int(stage_cfg.get("global_registration", {}).get("target_points", 1024)),
                )
                _, expected_rotations, expected_translations = labels_in_reconstruction_frame(
                    batch["canonical_fragments"],
                    batch["align_rotations"],
                    batch["align_translations"],
                    global_rotation,
                    global_translation,
                )
                prediction_to_gt_rotation = global_rotation
                prediction_to_gt_translation = global_translation
        if pose_model is not None:
            if initialization_icp_iterations > 0:
                init_rotations, init_translations, _ = differentiable_icp_initialization(
                    batch["fragments"],
                    target_object,
                    batch["fragment_mask"],
                    iterations=initialization_icp_iterations,
                )
            else:
                init_rotations = None
                init_translations = None
            pose_output = pose_model(
                batch["fragments"],
                target_object,
                batch["fragment_mask"],
                initial_rotations=init_rotations,
                initial_translations=init_translations,
            )
            rotations = pose_output.rotations
            translations = pose_output.translations
        else:
            rotations, translations, _ = differentiable_icp_initialization(
                batch["fragments"],
                output.point_cloud,
                batch["fragment_mask"],
                iterations=icp_iterations,
            )
        # "Raw" preserves the historical target-frame comparison.  For a raw
        # Stage-2 target it exposes global-frame error; the global metric below
        # composes the oracle transform before comparing to PartNet GT.
        losses = pose_supervised_loss(
            rotations,
            translations,
            batch["align_rotations"],
            batch["align_translations"],
            batch["fragment_mask"],
        )
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        steps += 1

        global_rotations = prediction_to_gt_rotation[:, None] @ rotations
        global_translations = torch.einsum("bfd,bcd->bfc", translations, prediction_to_gt_rotation) + prediction_to_gt_translation[:, None, :]
        global_losses = pose_supervised_loss(
            global_rotations,
            global_translations,
            batch["align_rotations"],
            batch["align_translations"],
            batch["fragment_mask"],
        )
        for key, value in global_losses.items():
            totals[f"global_{key}"] = totals.get(f"global_{key}", 0.0) + float(value.detach().cpu())
        # Place each predicted fragment in the PartNet frame for an invariant
        # object-level comparison.  ``pose_output`` is available for learned
        # methods; ICP falls back to the explicitly transformed fragments.
        aligned_fragments = (
            pose_output.aligned_fragments
            if pose_model is not None
            else torch.einsum("bfnd,bfcd->bfnc", batch["fragments"], rotations) + translations[:, :, None, :]
        )
        aligned_union = aligned_fragments.reshape(batch_size, -1, 3)
        target_chamfers.extend(symmetric_chamfer(aligned_union, target_object).detach().cpu().tolist())
        aligned_union_global = apply_transform(
            aligned_union,
            prediction_to_gt_rotation,
            prediction_to_gt_translation,
        )
        gt_chamfers.extend(symmetric_chamfer(aligned_union_global, batch["target"]).detach().cpu().tolist())

        equivalent = equivalent_part_pose_metrics(
            batch["fragments"], rotations, translations,
            expected_rotations, expected_translations,
            batch.get("equivalence_classes", torch.arange(rotations.shape[1], device=device)[None].expand(batch_size, -1)),
            batch["fragment_mask"],
        )
        part_chamfers.extend(equivalent["part_chamfers"])
        equivalent_rot_deg_all.extend(equivalent["rotation_degrees"])
        equivalent_trans_all.extend(equivalent["translation_errors"])
        successes.append(float(equivalent["assembly_success_rate"]))
        registrations.extend(registration_residual.detach().cpu().tolist())

        # Per-fragment errors over the valid fragments, for distribution stats.
        mask = batch["fragment_mask"].reshape(-1).bool()
        rot_deg = (
            rotation_geodesic_error(rotations, batch["align_rotations"]).reshape(-1)
            * 180.0
            / math.pi
        )
        trans = torch.linalg.vector_norm(
            translations - batch["align_translations"], dim=-1
        ).reshape(-1)
        rot_deg_all.extend(rot_deg[mask].detach().cpu().tolist())
        trans_all.extend(trans[mask].detach().cpu().tolist())
        global_rot_deg = (
            rotation_geodesic_error(global_rotations, batch["align_rotations"]).reshape(-1)
            * 180.0 / math.pi
        )
        global_trans = torch.linalg.vector_norm(
            global_translations - batch["align_translations"], dim=-1
        ).reshape(-1)
        global_rot_deg_all.extend(global_rot_deg[mask].detach().cpu().tolist())
        global_trans_all.extend(global_trans[mask].detach().cpu().tolist())

        if prediction_dir is not None:
            object_ids = batch.get("object_id", [str(steps)] * batch_size)
            for item_idx, object_id in enumerate(object_ids):
                safe_id = str(object_id).replace("/", "_").replace("\\", "_")
                file_name = f"{safe_id}_{steps}_{item_idx}.npz"
                np.savez_compressed(
                    prediction_dir / file_name,
                    object_id=np.asarray(str(object_id)),
                    rotations=rotations[item_idx].detach().cpu().numpy(),
                    translations=translations[item_idx].detach().cpu().numpy(),
                    aligned_fragments=aligned_fragments[item_idx].detach().cpu().numpy(),
                    target=target_object[item_idx].detach().cpu().numpy(),
                    registration_rotation=global_rotation[item_idx].detach().cpu().numpy(),
                    registration_translation=global_translation[item_idx].detach().cpu().numpy(),
                    registration_residual=np.asarray(float(registration_residual[item_idx].cpu())),
                )
                saved_predictions.append({"object_id": str(object_id), "file": file_name, "registration_residual": float(registration_residual[item_idx].cpu())})

    metrics = {key: value / max(steps, 1) for key, value in totals.items()}
    out_dir = Path(cfg["output"]["dir"])

    # Rich, target-source-specific eval report (fixes the old unconditional
    # stage3_pose_metrics.json overwrite). Means hide the tail, so also report
    # the full error distribution.
    report = {
        "target_source": target_source,
        "split": split,
        "num_samples": len(dataset),
        "num_fragments_evaluated": len(rot_deg_all),
        "has_pose_checkpoint": pose_model is not None,
        "means": metrics,
        "rotation_error_deg": summarize_distribution(rot_deg_all),
        "translation_error": summarize_distribution(trans_all),
        "global_rotation_error_deg": summarize_distribution(global_rot_deg_all),
        "global_translation_error": summarize_distribution(global_trans_all),
        "equivalent_part_rotation_error_deg": summarize_distribution(equivalent_rot_deg_all),
        "equivalent_part_translation_error": summarize_distribution(equivalent_trans_all),
        "gpat_compatible_chamfer": summarize_distribution(target_chamfers),
        "gt_frame_chamfer": summarize_distribution(gt_chamfers),
        "part_accuracy_at_0.01": float(sum(value <= 0.01 for value in part_chamfers) / max(1, len(part_chamfers))),
        "assembly_success_rate": float(sum(successes) / max(1, len(successes))),
        "registration_residual": summarize_distribution(registrations),
        "registration_exclusion_rate": float(sum(value > float(stage_cfg.get("registration_max_residual", 0.02)) for value in registrations) / max(1, len(registrations))),
        "confidence_intervals": {
            "gpat_compatible_chamfer": _mean_ci95(target_chamfers),
            "gt_frame_chamfer": _mean_ci95(gt_chamfers),
            "part_accuracy_at_0.01": _mean_ci95([float(value <= 0.01) for value in part_chamfers]),
            "assembly_success_rate": _mean_ci95(successes),
            "global_rotation_error_deg": _mean_ci95(global_rot_deg_all),
            "global_translation_error": _mean_ci95(global_trans_all),
        },
        "predictions": saved_predictions,
    }
    _save_json(out_dir / f"stage3_eval_{target_source}.json", report)
    build_results_markdown(out_dir)
    return metrics
