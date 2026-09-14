"""Import only encoder/field tensors; never construct the original pose solver."""
from __future__ import annotations

import copy
import hashlib
import json
import inspect
import os
from pathlib import Path

import torch
from torch import nn

from ..data import sanitize_input, observation_fingerprint
from ..geometry import fixed_query_bank


EXPORT_SCHEMA = "scaffold-sota-frozen-prior-v1"
ARCHITECTURES = {"coarse-scaffold-reassembly-v2.1-local-contacts": 2,
                 "fragment-assembly-repair-v3": 3}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_state(path):
    # Full upstream run checkpoints contain NumPy/Python RNG state. They are
    # explicit local artifacts; their original training workflows are not run.
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    state = torch.load(path, **options)
    if state.get("export_schema") == EXPORT_SCHEMA:
        if state.get("architecture") not in ARCHITECTURES or set(state.get("weights", {})) != {"encoder", "field"}:
            raise ValueError("Malformed isolated prior export")
        return state
    if state.get("architecture") not in ARCHITECTURES or state.get("schema_version") != ARCHITECTURES[state["architecture"]]:
        raise ValueError("Only explicitly identified v2/v3 Stage 2 checkpoints are supported")
    allowed_stages = (2, 3) if ARCHITECTURES[state["architecture"]] == 2 else (2,)
    if state.get("stage") not in allowed_stages or int(state.get("step", 0)) < 1:
        raise ValueError("Export requires a trained v3 Stage 2 or v2 Stage 2/3 snapshot")
    if not state.get("dataset_fingerprint") or "cfg" not in state or "model" not in state:
        raise ValueError("Checkpoint lacks model or dataset provenance")
    weights = {name: {key[len(name) + 1:]: value.detach().cpu().clone()
                      for key, value in state["model"].items() if key.startswith(name + ".")}
               for name in ("encoder", "field")}
    if any(not values for values in weights.values()):
        raise ValueError("Checkpoint must contain both encoder and field parameters")
    return {"export_schema": EXPORT_SCHEMA, "architecture": state["architecture"],
            "schema_version": state["schema_version"], "cfg": copy.deepcopy(state["cfg"]),
            "weights": weights, "provenance": {
                "source_checkpoint_sha256": sha256_file(path),
                "dataset_fingerprint": state["dataset_fingerprint"], "stage": int(state["stage"]),
                "source_checkpoint_stage": int(state["stage"]), "exported_components": ["encoder", "field"],
                "step": int(state["step"]), "purpose": state.get("purpose"),
                "run_id": state.get("run_id"),
                "training_lineage": copy.deepcopy(state.get("training_lineage", {}))}}


def export_prior(source_checkpoint, output, training_manifest=None):
    """Write a new standalone encoder/field artifact, without original run gates."""
    source_checkpoint, output = Path(source_checkpoint).resolve(), Path(output).resolve()
    if output == source_checkpoint or output.exists():
        raise FileExistsError("Prior export requires a fresh output file")
    state = _read_state(source_checkpoint)
    state["provenance"].setdefault("training_source_ids", None)
    state["provenance"].setdefault("source_identity_status", "unknown_without_training_manifest")
    if training_manifest is not None:
        from ..data import verify_manifest
        verification = verify_manifest(training_manifest)
        if verification["dataset_fingerprint"] != state["provenance"]["dataset_fingerprint"]:
            raise ValueError("Prior training manifest differs from checkpoint dataset fingerprint")
        document = json.loads(Path(training_manifest).read_text(encoding="utf-8"))
        training_sources = sorted({record["source_id"] for record in document["patterns"] if record["split"] == "train"})
        source_hashes = {record["source_id"]: record["sha256"] for record in document["sources"] if record["source_id"] in training_sources}
        state["provenance"].update(training_source_ids=training_sources, training_source_sha256=source_hashes,
                                   source_identity_status="verified_manifest_training_partition",
                                   training_manifest_sha256=sha256_file(training_manifest))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError("Prior export temporary file already exists")
    try:
        torch.save(state, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": str(output), "sha256": sha256_file(output), "architecture": state["architecture"],
            **state["provenance"]}


class FrozenPrior(nn.Module):
    def __init__(self, checkpoint, device="cpu", query_count=512, extent=1.5,
                 expected_fingerprint=None, expected_train_source_ids=None):
        super().__init__()
        checkpoint = Path(checkpoint).resolve()
        state = _read_state(checkpoint)
        cfg = state["cfg"].get("model", {})
        from reassembly.model import GeometryEncoder
        from .architectures import ShapeField, AdaptedShapeField
        if state["architecture"] == "fragment-assembly-repair-v3":
            from reassembly.repair.model import InvariantEncoder
            variant = state["cfg"].get("repair", {}).get("geometry_variant", "revised")
            if variant not in ("existing", "revised"):
                raise ValueError("Unknown v3 geometry variant")
            self.encoder = GeometryEncoder(cfg) if variant == "existing" else InvariantEncoder(cfg)
            self.field_module = AdaptedShapeField(cfg)
        else:
            self.encoder, self.field_module = GeometryEncoder(cfg), ShapeField(cfg)
        self.encoder.load_state_dict(state["weights"]["encoder"], strict=True)
        self.field_module.load_state_dict(state["weights"]["field"], strict=True)
        if any(not torch.isfinite(value).all() for values in state["weights"].values() for value in values.values() if value.is_floating_point()):
            raise ValueError("Prior contains nonfinite parameters")
        self.provenance = {**state["provenance"], "artifact_sha256": sha256_file(checkpoint),
                           "architecture": state["architecture"], "query_count": int(query_count),
                           "query_extent": float(extent), "support": "fixed-multiscale-sobol-v1",
                           "truncation": float(cfg.get("truncation", .1)),
                           "coordinate_frame": "normalized_anchor", "sdf_convention": "negative_inside",
                           "diagnostic_only": False}
        if expected_fingerprint is not None and self.provenance["dataset_fingerprint"] != expected_fingerprint:
            raise ValueError("Prior training dataset fingerprint mismatch")
        if expected_fingerprint is not None and (self.provenance.get("training_source_ids") is None or self.provenance.get("source_identity_status") != "verified_manifest_training_partition"):
            raise ValueError("Main-study prior use requires a standalone export with verified training source identities; re-export with training_manifest")
        if expected_train_source_ids is not None:
            known = self.provenance.get("training_source_ids")
            if known is None or set(known) != set(expected_train_source_ids):
                raise ValueError("Prior training source identities are unknown or differ")
        self.register_buffer("queries", fixed_query_bank(query_count, extent))
        self.requires_grad_(False)
        self.eval()
        self.to(device)

    def train(self, mode=True):
        # Recipient .train() recursion must never unfreeze dropout/normalization.
        return super().train(False)

    @property
    def device(self):
        return self.queries.device

    def encode(self, sample):
        if sample.get("split") in ("val", "test", "cut_holdout") and sample.get("source_id") in (self.provenance.get("training_source_ids") or []):
            raise ValueError("Held-out observation source was used to train this prior")
        observed = sanitize_input(sample, self.device)
        with torch.no_grad():
            encoded = self.encoder(observed["points"][None])
            encoded.update(fragment_mask=observed["fragment_mask"][None], anchor_index=observed["anchor_index"][None])
            context = tuple(value.detach() for value in self.field_module.context(encoded))
        return encoded, context

    @torch.no_grad()
    def exterior_probabilities(self, sample):
        """Input-only common exterior estimator, usable identically in all arms."""
        encoded, _ = self.encode(sample)
        return (1-encoded["fracture_logits"][0].sigmoid()).detach()

    def bind(self, sample):
        """Return a differentiable query function with cached frozen context."""
        encoded, context = self.encode(sample)
        def query(query_xyz):
            query_xyz = torch.as_tensor(query_xyz, device=self.device, dtype=self.queries.dtype)
            if query_xyz.ndim != 2 or query_xyz.shape[-1] != 3:
                raise ValueError("Field queries require [Q,3]")
            output = self.field_module(encoded, query_xyz[None], context=context)
            distance, log_scale = output["distance"][0], output["log_scale"][0]
            return {"distance": distance, "log_scale": log_scale,
                    "valid": torch.isfinite(distance) & torch.isfinite(log_scale)}
        return query

    def field(self, sample, query_xyz):
        return self.bind(sample)(query_xyz)

    @torch.no_grad()
    def sample_tokens(self, sample):
        output = self.field(sample, self.queries)
        return {"query_xyz": self.queries.detach().clone(), **{k: v.detach() for k, v in output.items()},
                "provenance": {**self.provenance, "observation_id": observation_fingerprint(sample)}}
