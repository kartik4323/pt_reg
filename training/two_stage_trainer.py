"""Single-process trainers for the three-stage assembly pipeline."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.assembly_dataset import AssemblyObjectDataset, CompatibilityPairDataset
from losses.assembly_losses import build_assembly_loss
from losses.stage1_losses import build_stage1_loss
from models.assembly import FragmentAssemblyModel, build_assembly_model
from models.compatibility_three import Stage1CompatibilityModel, build_stage1_model
from models.pose import (
    build_pose_estimator,
    build_pose_loss,
    differentiable_icp_initialization,
    pose_supervised_loss,
)


def _move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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

        dataset = CompatibilityPairDataset(
            data_root=cfg["data"]["shapenet_root"],
            split="train",
            cfg=cfg,
            epoch_size=stage_cfg.get("epoch_size"),
        )
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
        metrics: Dict[str, float] = {}
        for epoch in range(self.epochs):
            metrics = self._train_epoch(epoch)
            print(
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
        if freeze_pretrained:
            self.model.freeze_pretrained()

        self.criterion = build_assembly_loss(cfg).to(device)
        stage_cfg = cfg.get("stage2", {})
        self.stage_cfg = stage_cfg
        self.freeze_pretrained = freeze_pretrained
        self.optimizer = self._build_optimizer()
        dataset = AssemblyObjectDataset(
            data_root=cfg["data"]["shapenet_root"],
            split="train",
            cfg=cfg,
            epoch_size=stage_cfg.get("epoch_size"),
        )
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

    def load_stage1(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.encoder.load_state_dict(checkpoint["encoder"], strict=True)
        self.model.compatibility.load_state_dict(checkpoint["compatibility"], strict=True)

    def train(self) -> Path:
        metrics: Dict[str, float] = {}
        for epoch in range(self.epochs):
            if (
                self.freeze_pretrained
                and self.unfreeze_epoch is not None
                and epoch == int(self.unfreeze_epoch)
            ):
                self.model.unfreeze_pretrained()
                self.freeze_pretrained = False
                self.optimizer = self._build_optimizer()
            metrics = self._train_epoch(epoch)
            print(
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
                subset_output = self.model(batch["fragments"], subset_mask)
                subset_pred = subset_output.point_cloud

            dropout_pred = None
            if self.use_dropout:
                dropout_mask = self._drop_one_mask(batch["fragment_mask"])
                dropout_output = self.model(batch["fragments"], dropout_mask)
                dropout_pred = dropout_output.point_cloud

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

        self.reconstruction_model = build_assembly_model(cfg).to(device)
        # Load stage-2 checkpoint on CPU first to reduce peak GPU memory usage
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

        dataset = AssemblyObjectDataset(
            data_root=cfg["data"]["shapenet_root"],
            split="train",
            cfg=cfg,
            epoch_size=self.stage_cfg.get("epoch_size", 512),
        )
        self.loader = DataLoader(
            dataset,
            batch_size=self.stage_cfg.get("batch_size", 2),
            shuffle=True,
            num_workers=self.stage_cfg.get("num_workers", 0),
            drop_last=True,
        )
        self.epochs = self.stage_cfg.get("epochs", 10)
        self.grad_clip = self.stage_cfg.get("gradient_clip", 1.0)
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
        metrics: Dict[str, float] = {}
        for epoch in range(self.epochs):
            metrics = self._train_epoch(epoch)
            print(
                "Stage 3 "
                f"epoch={epoch + 1}/{self.epochs} "
                f"loss={metrics['loss']:.4f} "
                f"match={metrics['loss_matching']:.4f} "
                f"pose={metrics.get('loss_pose', 0.0):.4f} "
                f"acc={metrics.get('matching_accuracy', 0.0):.3f} "
                f"recon_cd={metrics['loss_recon_cd']:.4f} "
                f"rot_deg={metrics.get('rotation_error_deg', 0.0):.2f} "
                f"trans={metrics.get('translation_error', 0.0):.4f} "
                f"overlap={metrics.get('loss_overlap', 0.0):.4f}"
            )

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
        return ckpt_path

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.pose_model.train()
        self.reconstruction_model.eval()
        totals: Dict[str, float] = {}
        steps = 0

        for batch in tqdm(self.loader, desc=f"Stage3 epoch {epoch + 1}", leave=False):
            batch = _move_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)

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
                    target_object = batch["target"]
                else:
                    target_object = reconstruction

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
                losses = self.criterion(
                    output=output,
                    fragments=batch["fragments"],
                    fragment_mask=batch["fragment_mask"],
                    reconstructed_object=target_object.detach() if self.freeze_reconstruction else target_object,
                    ground_truth_object=batch["target"],
                    target_fragments=batch["canonical_fragments"],
                    gt_rotations=batch["align_rotations"],
                    gt_translations=batch["align_translations"],
                    weights=self._loss_weights_for_epoch(epoch),
                )

            loss = losses["loss"]
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    # unscale before clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.pose_model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.pose_model.parameters(), self.grad_clip)
                self.optimizer.step()

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
        weights = {
            "lambda_matching": final_matching,
            "lambda_recon": final_recon,
            "lambda_coverage": final_coverage,
            "lambda_gt": final_gt,
            "lambda_overlap": final_overlap,
            "lambda_pose": final_pose,
        }

        if epoch < self.pose_only_epochs:
            weights.update(
                {
                    "lambda_matching": base.get("lambda_matching_early", final_matching * 2.0),
                    "lambda_recon": 0.0,
                    "lambda_coverage": 0.0,
                    "lambda_gt": 0.0,
                    "lambda_overlap": 0.0,
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
) -> Dict[str, float]:
    """Evaluate Stage 3 learned pose estimation, or ICP if no pose checkpoint is supplied."""
    if target_source not in {"reconstruction", "ground_truth"}:
        raise ValueError("target_source must be 'reconstruction' or 'ground_truth'")
    model = build_assembly_model(cfg).to(device)
    checkpoint = torch.load(stage2_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    pose_model = None
    stage_cfg = cfg.get("stage3", {})
    initialization_icp_iterations = stage_cfg.get(
        "initialization_icp_iterations",
        min(3, stage_cfg.get("icp_iterations", icp_iterations)),
    )
    if cfg.get("model", {}).get("pose", {}).get("architecture", "target_segmentation") == "target_segmentation":
        initialization_icp_iterations = 0
    if stage3_checkpoint is not None:
        pose_model = build_pose_estimator(cfg).to(device)
        pose_checkpoint = torch.load(stage3_checkpoint, map_location=device)
        state_dict = pose_checkpoint["model"] if "model" in pose_checkpoint else pose_checkpoint
        pose_model.load_state_dict(state_dict, strict=True)
        pose_model.eval()

    dataset = AssemblyObjectDataset(
        data_root=cfg["data"]["shapenet_root"],
        split=split,
        cfg=cfg,
        epoch_size=cfg.get("stage3", {}).get("epoch_size", 8),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("stage3", {}).get("batch_size", 2),
        shuffle=False,
        num_workers=0,
    )

    totals: Dict[str, float] = {}
    steps = 0
    for batch in tqdm(loader, desc="Stage3 pose", leave=False):
        batch = _move_to_device(batch, device)
        output = model(batch["fragments"], batch["fragment_mask"])
        target_object = batch["target"] if target_source == "ground_truth" else output.point_cloud
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

    metrics = {key: value / max(steps, 1) for key, value in totals.items()}
    out_dir = Path(cfg["output"]["dir"])
    _save_json(out_dir / "stage3_pose_metrics.json", metrics)
    return metrics
