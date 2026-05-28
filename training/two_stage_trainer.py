"""Single-process trainers for the two-stage assembly pipeline."""

from __future__ import annotations

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
from models.pose import differentiable_icp_initialization, pose_supervised_loss


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


@torch.no_grad()
def run_pose_stage(
    cfg: dict,
    device: torch.device,
    stage2_checkpoint: str,
    split: str = "test",
    icp_iterations: int = 5,
) -> Dict[str, float]:
    """Run Stage 3 post-hoc ICP pose estimation on one loader pass."""
    model = build_assembly_model(cfg).to(device)
    checkpoint = torch.load(stage2_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

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
