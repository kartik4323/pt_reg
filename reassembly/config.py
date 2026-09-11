"""Resolved defaults and validation for the supported v2 experiment."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

DEFAULTS = {
    "version": 2,
    "data": {
        "category": "02876657", "seed": 42, "points_per_fragment": 1024,
        "reservoir_points": 4096, "sdf_queries": 2048,
        "sdf_reservoir_queries": 8192, "target_points": 2048,
        "max_sources": 100, "min_sources": 30, "patterns_per_source": 8,
        "split_ratios": [0.8, 0.1, 0.1], "max_fragments": 3,
        "min_fragments": 2, "translation_range": 0.7,
    },
    "model": {
        "sample_counts": [256, 128, 64], "dim": 128, "neighbors": 16,
        "attention_layers": 2, "heads": 4, "truncation": 0.1, "contact_points": 256,
        "log_scale_min": -6.0, "log_scale_max": -2.0,
    },
    "loss": {
        "segmentation": 1.0, "matching": 1.0, "view_consistency": 0.1,
        "sdf": 1.0, "calibration": 0.01, "geometry_retention": 0.25, "matching_localization": 1.0,
    },
    "train": {
        "max_updates": 2000, "batch_size": 2, "grad_accum_steps": 4,
        "learning_rate": 0.001, "weight_decay": 0.0001,
        "gradient_clip": 1.0, "validation_interval": 100,
        "validation_samples": 32, "num_workers": 0,
        "overfit_patterns": 16, "overfit_updates": 2000,
        "seed": 42, "amp": True, "contact_radius": .05, "contact_sigma": .01,
    },
    "solver": {
        "resolution": 32, "field_chunk": 2048, "field_extent": 2.25,
        "max_pair_candidates": 16, "keep_pair_candidates": 4,
        "refine_candidates": 4, "refinement_iterations": 5,
        "prior_weight": 0.25, "huber_delta": 0.02,
        "success_threshold": 0.01,
    },
    "resources": {
        "managed_root": "~/reassembly_v2", "cap_gib": 40,
        "min_free_gib": 50, "max_vram_gib": 20,
    },
}


def merge_config(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None = None) -> dict:
    override = {}
    if path is not None:
        with open(path, encoding="utf-8") as handle:
            override = yaml.safe_load(handle) or {}
    cfg = merge_config(DEFAULTS, override)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if cfg.get("version") != 2:
        raise ValueError("Only configuration version 2 is supported; legacy weights/configs are not migrated")
    if str(cfg["data"]["category"]) != "02876657":
        raise ValueError("The first supported pilot is ShapeNet bottles (02876657)")
    if (cfg["data"]["min_fragments"], cfg["data"]["max_fragments"]) != (2, 3):
        raise ValueError("v2 requires complete sets of exactly 2 or 3 pieces")
    for section, names in {
        "data": ["points_per_fragment", "sdf_queries", "reservoir_points", "target_points"],
        "model": ["dim", "neighbors", "heads", "attention_layers"],
        "train": ["max_updates", "batch_size", "grad_accum_steps", "validation_interval"],
        "solver": ["resolution", "field_chunk"],
    }.items():
        for name in names:
            if int(cfg[section][name]) <= 0:
                raise ValueError(f"{section}.{name} must be positive")
    if cfg["data"]["reservoir_points"] < cfg["data"]["points_per_fragment"]:
        raise ValueError("Prepare distinct point reservoirs at least as large as training inputs")
    if cfg["model"]["dim"] % cfg["model"]["heads"]:
        raise ValueError("model.dim must be divisible by model.heads")
    if not 3 <= cfg["model"]["contact_points"] <= cfg["data"]["points_per_fragment"]:
        raise ValueError("model.contact_points must fit within the input point budget and be at least 3")
    if not 0 < cfg["train"]["contact_sigma"] <= cfg["train"]["contact_radius"]:
        raise ValueError("Use 0 < contact_sigma <= contact_radius")
    if cfg["loss"]["matching_localization"] <= 0:
        raise ValueError("Distance-aware contact supervision requires a positive matching_localization weight")
    if not 0 <= cfg["solver"]["prior_weight"] <= 0.25:
        raise ValueError("solver.prior_weight must be between 0 and 0.25")
    if cfg["train"]["max_updates"] > 2000:
        raise ValueError("This workflow is a bounded pilot: train.max_updates must not exceed 2000")
    if not 1 <= cfg["train"]["overfit_updates"] <= 2000 or cfg["train"]["overfit_patterns"] != 16:
        raise ValueError("The fixed pilot check requires exactly 16 patterns and 1–2000 updates")
    reassessment = cfg["data"].get("source_pool_reassessment", False)
    if not isinstance(reassessment, bool):
        raise ValueError("data.source_pool_reassessment must be a boolean")
    source_cap = 498 if reassessment else 100
    if not 30 <= cfg["data"]["min_sources"] <= 100 or not 1 <= cfg["data"]["max_sources"] <= source_cap:
        raise ValueError(f"Source preparation examines at most {source_cap} candidates and requires at least 30 sources; "
                         "use the explicit bottle source-pool reassessment configuration to expand the initial 100")
    if list(cfg["data"]["split_ratios"]) != [0.8, 0.1, 0.1]:
        raise ValueError("The supported source split is 80/10/10")
    if not 0 < cfg["resources"]["cap_gib"] <= 40 or cfg["resources"]["min_free_gib"] < 50:
        raise ValueError("Managed storage must be capped at 40 GiB or less with at least 50 GiB free")
    if not 0 < cfg["resources"]["max_vram_gib"] <= 20:
        raise ValueError("The A5000 pilot memory cap must be 20 GiB or less")
    if cfg["model"]["truncation"] <= 0 or cfg["solver"]["resolution"] < 2:
        raise ValueError("Use a positive SDF truncation and a field resolution of at least 2")


def model_fingerprint(cfg: dict) -> str:
    fields = {"version": 2, "model": cfg["model"]}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
