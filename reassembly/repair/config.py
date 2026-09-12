"""Explicit experiment configuration, separate from the bounded v2 pilot."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from reassembly.config import DEFAULTS as V2_DEFAULTS, merge_config, validate_config as validate_v2

DEFAULTS = merge_config(V2_DEFAULTS, {
    "version": 3,
    "data": {"max_sources": 498, "source_pool_reassessment": True},
    "train": {"max_updates": 10000, "overfit_updates": 10000,
              "validation_interval": 500, "validation_samples": 48},
    "repair": {"geometry_variant": "revised", "view_supervision": "resampled_contrastive",
               "view_positive_radius": .05, "view_negative_radius": .1, "view_temperature": .1,
               "curriculum_updates": [600, 1300], "evaluation_seeds": [4101, 4102, 4103],
               "training_seeds": [42, 43, 44], "contact_train_success": .8,
               "overfit_robust_success": .9, "validation_success": .8,
               "qualitative_limit": 8},
})


def load_config(path=None):
    override = {} if path is None else yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = merge_config(DEFAULTS, override)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg.get("version") != 3:
        raise ValueError("Repair experiments require version: 3")
    # Reuse geometric/resource checks without changing the v2 pilot's limits.
    legacy = copy.deepcopy(cfg)
    legacy["version"] = 2
    legacy["train"].update(max_updates=2000, overfit_updates=2000)
    validate_v2(legacy)
    for key in ("max_updates", "overfit_updates"):
        if not isinstance(cfg["train"][key], int) or cfg["train"][key] <= 0:
            raise ValueError(f"train.{key} must be an explicit positive update budget")
    r = cfg["repair"]
    if r["geometry_variant"] not in ("existing", "revised"):
        raise ValueError("geometry_variant must be existing or revised")
    if r["view_supervision"] not in ("existing", "resampled_contrastive"):
        raise ValueError("view_supervision must be existing or resampled_contrastive")
    if not 0 < r["view_positive_radius"] < r["view_negative_radius"] or r["view_temperature"] <= 0:
        raise ValueError("Contrastive radii require 0 < positive < negative and positive temperature")
    if len(r["curriculum_updates"]) != 2 or not 0 < r["curriculum_updates"][0] < r["curriculum_updates"][1]:
        raise ValueError("Use two increasing absolute curriculum transitions")
    if cfg["data"]["sdf_queries"] % 4:
        raise ValueError("Four balanced field query groups require sdf_queries divisible by four")
    if cfg["solver"]["success_threshold"] != .01:
        raise ValueError("The geometric acceptance threshold remains 0.01")
    if cfg["solver"].get("min_correspondence_weight", .001) != .001 or cfg["solver"].get("min_pair_mass", .05) != .05:
        raise ValueError("Production confidence thresholds must remain 0.001 and 0.05")
    if (r["contact_train_success"], r["overfit_robust_success"], r["validation_success"]) != (.8, .9, .8):
        raise ValueError("Accepted assembly gate criteria cannot be weakened")
    if list(r["evaluation_seeds"]) != [4101, 4102, 4103] or list(r["training_seeds"]) != [42, 43, 44]:
        raise ValueError("Use the predeclared evaluation and training seed sets")


def signature(cfg, *, model_only=False):
    keys = ("version", "model", "repair") if model_only else ("version", "model", "repair", "data", "train", "loss", "solver")
    return hashlib.sha256(json.dumps({k: cfg[k] for k in keys}, sort_keys=True).encode()).hexdigest()


def variant_config(cfg, geometry, supervision, seed=42):
    result = copy.deepcopy(cfg)
    result["repair"].update(geometry_variant=geometry, view_supervision=supervision)
    result["train"]["seed"] = int(seed)
    validate_config(result)
    return result
