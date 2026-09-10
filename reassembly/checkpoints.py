"""Explicit fresh-run lineage; no legacy checkpoint discovery or migration."""
from __future__ import annotations

import os
import random
import copy
import json
from pathlib import Path

import numpy as np
import torch

from . import ARCHITECTURE, SCHEMA_VERSION
from .config import model_fingerprint


def save_checkpoint(path, model, optimizer, cfg, dataset_fingerprint, stage, step, *, purpose="pilot", metrics=None, scaler=None, guard=None, run_id=None):
    lineage = copy.deepcopy(cfg.get("run_lineage", {}))
    lineage[str(stage)] = {
        "updates": int(cfg["train"]["overfit_updates"] if purpose == "overfit" else cfg["train"]["max_updates"]),
        "checkpoint_step": int(step), "batch_size": int(cfg["train"]["batch_size"]),
        "grad_accum_steps": int(cfg["train"]["grad_accum_steps"]),
        "seed": int(cfg["train"]["seed"]), "condition": cfg["train"].get("condition", "predicted"),
        "loss": copy.deepcopy(cfg.get("loss", {})),
    }
    checkpoint_config = copy.deepcopy(cfg)
    checkpoint_config["run_lineage"] = lineage
    state = {
        "schema_version": SCHEMA_VERSION, "architecture": ARCHITECTURE,
        "model_fingerprint": model_fingerprint(cfg), "dataset_fingerprint": dataset_fingerprint,
        "cfg": checkpoint_config, "training_lineage": lineage,
        "stage": int(stage), "step": int(step), "purpose": purpose,
        "run_id": run_id,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics or {},
        "random_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }
    path = Path(path)
    if guard:
        parameter_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
        guard.check(additional_bytes=4 * parameter_bytes + 1024 ** 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, *, cfg=None, dataset_fingerprint=None, stage=None, purpose=None, run_id=None):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != SCHEMA_VERSION or checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Legacy or external checkpoint rejected. v2 trains from scratch and accepts only explicit v2 run checkpoints.")
    required = {"model", "optimizer", "cfg", "random_state", "step", "stage", "purpose", "dataset_fingerprint", "model_fingerprint"}
    if not required.issubset(checkpoint):
        raise ValueError("Incomplete v2 checkpoint: required training state is missing")
    if not isinstance(checkpoint["step"], int) or checkpoint["step"] < 0 or checkpoint["stage"] not in (1, 2, 3):
        raise ValueError("Invalid checkpoint stage or update number")
    if cfg is not None and checkpoint.get("model_fingerprint") != model_fingerprint(cfg):
        raise ValueError("Checkpoint architecture configuration differs from the resolved model configuration")
    if dataset_fingerprint is not None and checkpoint.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("Checkpoint dataset fingerprint differs from this prepared dataset")
    if stage is not None and checkpoint.get("stage") != int(stage):
        raise ValueError(f"Expected a stage-{stage} checkpoint, received stage {checkpoint.get('stage')}")
    if purpose is not None and checkpoint.get("purpose") != purpose:
        raise ValueError(f"Checkpoint purpose must be {purpose}; overfit weights cannot initialize a held-out pilot")
    if run_id is not None and checkpoint.get("run_id") != run_id:
        raise ValueError("Checkpoint belongs to a different run; resume only this run's explicit latest.pt or best.pt")
    return checkpoint


def require_completed_run(path, checkpoint):
    """A selected best checkpoint is usable only after its run really finished."""
    report_path = Path(path).resolve().parent / "training_report.json"
    if not report_path.is_file():
        raise ValueError("Stage handoff requires a completed previous run's training_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    budget_key = "overfit_updates" if checkpoint["purpose"] == "overfit" else "max_updates"
    budget = int(checkpoint["cfg"]["train"][budget_key])
    expected = {"status": "completed", "kind": "training", "stage": checkpoint["stage"],
                "purpose": checkpoint["purpose"], "dataset_fingerprint": checkpoint["dataset_fingerprint"],
                "run_id": checkpoint.get("run_id"), "updates": budget}
    if not checkpoint.get("run_id") or any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage handoff requires a completed prior run with matching stage, purpose, dataset, run identity, and full update budget")


def require_resume_config(checkpoint, cfg):
    """Resume preserves the experiment; only bounded budget increases are allowed."""
    previous = checkpoint["cfg"]
    for section in ("data", "model", "loss", "solver"):
        if previous.get(section) != cfg.get(section):
            raise ValueError(f"Cannot change {section} configuration when resuming")
    budgets = {"max_updates", "overfit_updates"}
    old_train = {key: value for key, value in previous["train"].items() if key not in budgets}
    new_train = {key: value for key, value in cfg["train"].items() if key not in budgets}
    if old_train != new_train:
        raise ValueError("Cannot change training or optimizer configuration when resuming; only update-budget increases are allowed")
    for key in budgets:
        old, new = int(previous["train"][key]), int(cfg["train"][key])
        if not old <= new <= 2000:
            raise ValueError("Resume update budgets may only increase and must remain at most 2000")


def finalize_checkpoint_budget(path, cfg, stage, purpose, run_id, guard=None):
    """Retain selected best weights while recording a completed extended budget.

    A resumed run can finish without improving its former best checkpoint. Its
    selected step/weights/RNG stay intact; allocated-budget metadata is updated.
    """
    path = Path(path)
    checkpoint = load_checkpoint(path, stage=stage, purpose=purpose, run_id=run_id)
    budget_key = "overfit_updates" if purpose == "overfit" else "max_updates"
    if checkpoint["cfg"]["train"][budget_key] == cfg["train"][budget_key]:
        return
    for key in ("max_updates", "overfit_updates"):
        checkpoint["cfg"]["train"][key] = int(cfg["train"][key])
    lineage = checkpoint["training_lineage"]
    lineage[str(stage)]["updates"] = int(cfg["train"][budget_key])
    checkpoint["cfg"]["run_lineage"] = copy.deepcopy(lineage)
    if guard:
        guard.check(additional_bytes=path.stat().st_size + 1024 ** 2)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def restore_random_state(checkpoint):
    state = checkpoint["random_state"]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])
