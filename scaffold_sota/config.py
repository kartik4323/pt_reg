"""Versioned, explicit settings for the independent bottle experiment."""
from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULTS = {
    "schema_version": 1,
    "experiment_id": "scaffold_sota",
    "data": {"points_per_fragment": 1024, "sdf_queries": 512,
             "seed": 42, "verify": True, "hard_noise_std": .001,
             "fixed_training": False, "train_limit": None},
    "prior": {"query_count": 512, "extent": 1.5},
    "model": {"token_dim": 128, "heads": 4},
    "train": {"updates": 10000, "grad_accum_steps": 8, "learning_rate": .0001,
              "weight_decay": .0001, "betas": [.9, .999], "eps": 1e-8, "validation_interval": 500,
              "checkpoint_interval": 500, "log_interval": 25,
              "gradient_clip": 1., "amp": False, "seed": 42},
    "evaluation": {"threshold": .01, "observation_seeds": [4101, 4102, 4103],
                   "bootstrap_samples": 2000},
    "resources": {"cap_gib": 75., "min_free_gib": 50.,
                  "max_reserved_gib": 20., "min_gpu_free_gib": 2.,
                  "max_external_gpu_mib": 512},
}


def merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path=None):
    result = copy.deepcopy(DEFAULTS)
    if path:
        path = Path(path)
        if path.suffix.lower() == ".json":
            overrides = json.loads(path.read_text(encoding="utf-8"))
        else:
            import yaml
            overrides = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError("Study configuration must be a mapping")
        unknown = set(overrides) - set(DEFAULTS)
        if unknown:
            raise ValueError("Unknown study sections: " + ", ".join(sorted(unknown)))
        result = merge(result, overrides)
    validate_config(result)
    return result


def validate_config(cfg):
    if cfg["experiment_id"] != "scaffold_sota" or cfg["schema_version"] != 1:
        raise ValueError("Only scaffold_sota schema 1 is accepted")
    for key in ("updates", "grad_accum_steps", "validation_interval", "checkpoint_interval", "log_interval"):
        if not isinstance(cfg["train"][key], int) or cfg["train"][key] < 1:
            raise ValueError("train.%s must be a positive integer" % key)
    for key in ("learning_rate", "gradient_clip"):
        if cfg["train"][key] <= 0:
            raise ValueError("train.%s must be positive" % key)
    if cfg["data"]["points_per_fragment"] < 16 or cfg["prior"]["query_count"] < 8:
        raise ValueError("Insufficient point/query budget")
    if cfg["data"].get("train_limit") is not None and cfg["data"]["train_limit"] < 1:
        raise ValueError("Training subset must be positive when explicitly requested")
    if cfg["model"]["token_dim"] % cfg["model"]["heads"]:
        raise ValueError("token_dim must be divisible by heads")
    if cfg["evaluation"]["threshold"] <= 0:
        raise ValueError("Evaluation threshold must be positive")
    if len(cfg["train"]["betas"]) != 2 or any(not 0 <= x < 1 for x in cfg["train"]["betas"]) or cfg["train"]["eps"] <= 0:
        raise ValueError("Invalid AdamW betas/eps")
    for key in ("cap_gib", "max_reserved_gib"):
        if cfg["resources"][key] <= 0:
            raise ValueError("resources.%s must be positive" % key)
    if cfg["resources"]["min_free_gib"] < 0:
        raise ValueError("Free space reserve cannot be negative")
    return cfg
