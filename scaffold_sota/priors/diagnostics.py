"""Explicit oracle diagnostics: field calibration and pose-attraction stability.

These helpers never train a recipient, select a test configuration or modify the
original pipeline. Oracle initialization and mesh evaluations are labelled in
every saved record so they cannot be confused with deployed assembly results.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..data import observation_fingerprint
from ..evaluation import aggregate, compare_conditions, evaluate_prediction, refine_prediction
from ..geometry import as_numpy
from ..io import append_jsonl, jsonable, write_json
from .cache import TokenSnapshotPrior, token_fingerprint
from .controls import GroundTruthPrior


def _output(path, diagnostic_only):
    if not diagnostic_only:
        raise ValueError("Oracle diagnostics require explicit diagnostic_only=True")
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("Diagnostic output must be a fresh directory")
    path.mkdir(parents=True)
    return path


def _translation_magnitude(degrees, magnitude=None):
    if magnitude is None:
        defaults = {0.: 0., 5.: .02, 15.: .05}
        if float(degrees) not in defaults:
            raise ValueError("Custom perturbation angles require an explicit translation magnitude")
        magnitude = defaults[float(degrees)]
    if not np.isfinite(magnitude) or magnitude < 0:
        raise ValueError("Translation perturbation magnitude must be finite and nonnegative")
    return float(magnitude)


def perturb_oracle_prediction(sample, degrees=0., seed=42, translation_magnitude=None):
    """Oracle starts with paired fixed-angle and fixed-length pose errors.

    Defaults in normalized anchor-frame units: exact, 5 degrees/.02, and
    15 degrees/.05. The same seed reuses rotation axes and translation directions
    across magnitudes, while the anchor and padding stay exactly unchanged.
    """
    if not np.isfinite(degrees) or degrees < 0:
        raise ValueError("Perturbation angle must be finite and nonnegative")
    translation_magnitude = _translation_magnitude(degrees, translation_magnitude)
    salt = int(hashlib.sha256((str(sample["source_id"])+str(sample["pattern_id"])).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed+salt)
    rotations, translations = as_numpy(sample["rotations_gt"]).copy(), as_numpy(sample["translations_gt"]).copy()
    mask = as_numpy(sample["fragment_mask"]).astype(bool)
    anchor = int(as_numpy(sample["anchor_index"]))
    for index in np.flatnonzero(mask):
        if index == anchor:
            continue
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        rotations[index] = Rotation.from_rotvec(axis*math.radians(degrees)).as_matrix() @ rotations[index]
        translations[index] += translation_magnitude*direction
    return {"rotations": rotations, "translations": translations, "status": "oracle_initialization",
            "diagnostic_only": True, "oracle_pose_initialization": True,
            "rotation_perturbation_deg": float(degrees), "translation_perturbation": translation_magnitude,
            "translation_units": "normalized_anchor_frame", "seed": seed}


def evaluate_pose_stability(provider, dataset, output, *, diagnostic_only=False,
                            degrees=(0, 5, 15), conditions=("A1", "A2", "A3", "A5"),
                            steps=25, seed=42, device="cpu", guard=None, limit=None,
                            translation_magnitudes=None):
    magnitudes = [None]*len(degrees) if translation_magnitudes is None else list(translation_magnitudes)
    if len(magnitudes) != len(degrees):
        raise ValueError("Supply one translation magnitude per rotation angle")
    translations = {float(angle): _translation_magnitude(angle, magnitude)
                    for angle, magnitude in zip(degrees, magnitudes)}
    output = _output(output, diagnostic_only)
    if any(condition not in ("A1", "A2", "A3", "A5") for condition in conditions):
        raise ValueError("Pose stability supports contact-only, predicted and explicit GT fields")
    if not hasattr(provider, "exterior_probabilities"):
        raise ValueError("Pose stability requires the declared predicted exterior estimator")
    oracle = GroundTruthPrior(diagnostic_only=True, query_count=len(provider.queries),
                              extent=provider.provenance.get("query_extent", 1.5), device=device)
    groups = {(float(angle), condition): [] for angle in degrees for condition in conditions}
    initial_rows = {float(angle): [] for angle in degrees}
    count = len(dataset) if limit is None else min(len(dataset), int(limit))
    for index in range(count):
        if guard:
            guard.check(additional_bytes=1024**2)
        sample = dataset[index]
        exterior = provider.exterior_probabilities(sample)
        predicted = TokenSnapshotPrior(provider.sample_tokens(sample), sample)
        truth = TokenSnapshotPrior(oracle.sample_tokens(sample), sample) if "A5" in conditions else None
        for angle in degrees:
            initial = perturb_oracle_prediction(sample, angle, seed, translations[float(angle)])
            baseline = evaluate_prediction(sample, initial)
            initial_rows[float(angle)].append(baseline)
            for condition in conditions:
                field = truth if condition == "A5" else None if condition == "A1" else predicted
                result = refine_prediction(sample, initial, field, condition=condition, steps=steps,
                                           exterior_probabilities=exterior, device=device)
                row = evaluate_prediction(sample, result)
                row.update(diagnostic_only=True, oracle_pose_initialization=True,
                           initial_rotation_error_deg=float(angle), initial_prediction=jsonable(initial),
                           initial_translation_error=translations[float(angle)],
                           translation_units="normalized_anchor_frame",
                           prediction=jsonable(result), input_sha256=observation_fingerprint(sample),
                           provider_provenance=field.provenance if field else None, condition=condition)
                groups[(float(angle), condition)].append(row)
                append_jsonl(output/("%gdeg_%s.jsonl" % (angle, condition)), row)
    report = {"diagnostic_only": True, "oracle_pose_initialization": True,
              "kind": "controlled_pose_stability", "count": count, "seed": seed,
              "dataset_fingerprint": dataset.fingerprint, "provider": provider.provenance,
              "translation_perturbations": {"%gdeg" % angle: magnitude for angle, magnitude in translations.items()},
              "translation_units": "normalized_anchor_frame", "groups": {}}
    for (angle, condition), rows in groups.items():
        report["groups"]["%gdeg_%s" % (angle, condition)] = {
            "rotation_perturbation_deg": angle, "translation_perturbation": translations[angle],
            "initial": aggregate(initial_rows[angle]), "refined": aggregate(rows),
            "paired_vs_initial": compare_conditions(initial_rows[angle], rows, seed=seed)}
    write_json(output/"pose_stability.json", report)
    return report


def evaluate_field_quality(provider, dataset, output, *, diagnostic_only=False, guard=None, limit=None):
    output = _output(output, diagnostic_only)
    oracle = GroundTruthPrior(diagnostic_only=True, query_count=len(provider.queries),
                              extent=provider.provenance.get("query_extent", 1.5), device=provider.queries.device)
    rows = []
    count = len(dataset) if limit is None else min(len(dataset), int(limit))
    for index in range(count):
        if guard:
            guard.check(additional_bytes=1024**2)
        sample = dataset[index]
        predicted, target = provider.sample_tokens(sample), oracle.sample_tokens(sample)
        valid = as_numpy(predicted["valid"] & target["valid"]).astype(bool)
        distance, truth = as_numpy(predicted["distance"]), as_numpy(target["distance"])
        sigma = np.exp(as_numpy(predicted["log_scale"]))
        valid &= np.isfinite(distance) & np.isfinite(truth) & np.isfinite(sigma) & (sigma > 0)
        residual = np.abs(distance[valid]-truth[valid])
        near = np.abs(truth[valid]) < .02
        row = {key: str(sample[key]) for key in ("source_id", "pattern_id", "observation_id")}
        row.update(diagnostic_only=True, query_count=len(valid), valid_queries=int(valid.sum()),
                   near_surface_queries=int(near.sum()), token_sha256=token_fingerprint(predicted),
                   field_mae=float(residual.mean()) if len(residual) else None,
                   near_surface_mae=float(residual[near].mean()) if near.any() else None,
                   sign_accuracy=float(np.mean((distance[valid] < 0)==(truth[valid] < 0))) if len(residual) else None,
                   laplace_1sigma_coverage=float(np.mean(residual <= sigma[valid])) if len(residual) else None,
                   laplace_95_coverage=float(np.mean(residual <= -math.log(.05)*sigma[valid])) if len(residual) else None,
                   provider_provenance=predicted.get("provenance", {}), oracle_provenance=target["provenance"])
        rows.append(row)
        append_jsonl(output/"field_quality.jsonl", row)
    summary = {}
    for metric in ("field_mae", "near_surface_mae", "sign_accuracy", "laplace_1sigma_coverage", "laplace_95_coverage"):
        sources = sorted({row["source_id"] for row in rows})
        means = []
        for source in sources:
            values = [row[metric] for row in rows if row["source_id"] == source and row[metric] is not None]
            if values:
                means.append(float(np.mean(values)))
        summary[metric] = {"source_macro_mean": float(np.mean(means)) if means else None, "source_count": len(means)}
    report = {"diagnostic_only": True, "kind": "field_quality", "count": len(rows),
              "dataset_fingerprint": dataset.fingerprint, "provider": provider.provenance, "metrics": summary,
              "uncertainty_distribution_assumption": "Laplace scale; expected one-sigma coverage=1-exp(-1), expected 95% coverage=.95",
              "representation": "same fixed input-only query support, truncated signed distances"}
    write_json(output/"field_quality.json", report)
    return report
