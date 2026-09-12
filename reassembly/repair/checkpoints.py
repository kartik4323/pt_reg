"""Explicit v3 checkpoint lineage; old weights cannot initialize a repair run."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import torch

from . import ARCHITECTURE, SCHEMA_VERSION
from .config import signature


def save_checkpoint(path, model, optimizer, scaler, cfg, fingerprint, *, stage, step,
                    purpose, run_id, metrics, lineage, query_cache_hash=None, guard=None):
    state = {"schema_version": SCHEMA_VERSION, "architecture": ARCHITECTURE,
        "cfg": copy.deepcopy(cfg), "model_signature": signature(cfg, model_only=True),
        "dataset_fingerprint": fingerprint, "stage": stage, "step": step,
        "purpose": purpose, "run_id": run_id, "metrics": metrics,
        "training_lineage": copy.deepcopy(lineage), "query_cache_hash": query_cache_hash,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "random_state": {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
    path = Path(path)
    if guard:
        guard.check(additional_bytes=4 * sum(t.numel()*t.element_size() for t in model.state_dict().values()) + 1024**2)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, *, cfg=None, fingerprint=None, stage=None, purpose=None):
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("architecture") != ARCHITECTURE or state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Legacy/external checkpoint rejected: repair runs require explicit v3 weights")
    required = {"cfg", "model", "optimizer", "scaler", "random_state", "stage", "step", "run_id",
                "purpose", "dataset_fingerprint", "model_signature", "training_lineage", "metrics", "query_cache_hash"}
    if not required.issubset(state) or state["stage"] not in (1, 2) or state["step"] < 1:
        raise ValueError("Incomplete or invalid v3 checkpoint")
    if signature(state["cfg"], model_only=True) != state["model_signature"]:
        raise ValueError("Checkpoint model configuration signature is invalid")
    if cfg is not None and signature(cfg, model_only=True) != state["model_signature"]:
        raise ValueError("Checkpoint architecture/experiment differs from resolved configuration")
    if fingerprint is not None and state["dataset_fingerprint"] != fingerprint:
        raise ValueError("Checkpoint dataset fingerprint mismatch")
    if stage is not None and state["stage"] != stage:
        raise ValueError(f"Expected stage-{stage} checkpoint")
    if purpose is not None and state["purpose"] != purpose:
        raise ValueError("Overfit and experiment checkpoint purposes cannot be interchanged")
    return state


def require_resume_config(state, cfg):
    old, new = copy.deepcopy(state["cfg"]), copy.deepcopy(cfg)
    for key in ("max_updates", "overfit_updates"):
        if new["train"][key] < old["train"][key]:
            raise ValueError("Resume budgets may only increase")
        old["train"].pop(key); new["train"].pop(key)
    old.pop("resources", None); new.pop("resources", None)
    if old != new:
        raise ValueError("Cannot change model, data, optimizer, losses or solver when resuming")
