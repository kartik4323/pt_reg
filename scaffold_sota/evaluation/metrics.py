"""Paired source-level metrics; failed and missing solves remain denominators."""
from __future__ import annotations

import numpy as np

from ..geometry import as_numpy, chamfer, reference_gauge, transform_points, validate_poses


def _identity(sample):
    return {key: str(sample.get(key, "")) for key in ("source_id", "pattern_id", "observation_id")}


def _key(row):
    return tuple(row.get(key, "") for key in ("source_id", "pattern_id", "observation_id"))


def failure_record(sample, reason="missing_prediction", status="failed"):
    mask = as_numpy(sample["fragment_mask"]).astype(bool)
    return {**_identity(sample), "status": status, "failed": True, "failure_reason": str(reason),
            "success": False, "part_accuracy": 0., "moving_part_accuracy": 0.,
            "num_parts": int(mask.sum()), "whole_chamfer": None, "per_part_chamfer": None,
            "rotation_deg": None, "translation_error": None, "contact_gap": None,
            "runtime_seconds": None}


def evaluate_prediction(sample, prediction, threshold=.01):
    if threshold <= 0:
        raise ValueError("Assembly distance threshold must be positive")
    result = failure_record(sample)
    if prediction is None:
        return result
    result["runtime_seconds"] = prediction.get("runtime_seconds", prediction.get("runtime"))
    result["refinement_skipped"] = prediction.get("refinement", {}).get("status") == "skipped"
    result["refinement_skip_reason"] = prediction.get("refinement", {}).get("skip_reason")
    if prediction.get("rotations") is None or prediction.get("translations") is None:
        result.update(failure_reason=str(prediction.get("reason", prediction.get("error", "missing_poses"))),
                      status=prediction.get("status", "failed"))
        return result
    mask = as_numpy(sample["fragment_mask"]).astype(bool)
    ids = np.flatnonzero(mask)
    anchor = int(np.flatnonzero(ids == int(as_numpy(sample["anchor_index"])))[0])
    try:
        rotations, translations = as_numpy(prediction["rotations"]), as_numpy(prediction["translations"])
        if len(rotations) == len(mask):
            rotations, translations = rotations[mask], translations[mask]
        rotations, translations = validate_poses(rotations, translations, len(ids))
        rotations, translations = reference_gauge(rotations, translations, anchor)
    except (ValueError, TypeError, IndexError) as exc:
        result.update(failure_reason=str(exc), status="invalid_pose")
        return result
    points, target = as_numpy(sample["points"])[mask], as_numpy(sample["canonical_points"])[mask]
    aligned = transform_points(points, rotations, translations)
    part_cd = np.asarray([chamfer(a, b) for a, b in zip(aligned, target)])
    whole_cd = chamfer(aligned.reshape(-1, 3), target.reshape(-1, 3))
    ground_rotation, ground_translation = as_numpy(sample["rotations_gt"])[mask], as_numpy(sample["translations_gt"])[mask]
    cosine = np.clip((np.einsum("fij,fij->f", rotations, ground_rotation) - 1) / 2, -1, 1)
    moving = np.arange(len(ids)) != anchor
    result.update(status=prediction.get("status", "ok"), failed=False, failure_reason=None,
                  success=bool(np.all(part_cd <= threshold) and whole_cd <= threshold),
                  part_accuracy=float(np.mean(part_cd <= threshold)),
                  moving_part_accuracy=float(np.mean(part_cd[moving] <= threshold)),
                  whole_chamfer=whole_cd, per_part_chamfer=part_cd.tolist(),
                  rotation_deg=np.degrees(np.arccos(cosine))[moving].tolist(),
                  translation_error=np.linalg.norm(translations-ground_translation, axis=-1)[moving].tolist(),
                  threshold=float(threshold), gauge="predicted_anchor", chamfer_definition="mean_symmetric_unsquared")
    if "interface_ids" in sample:
        labels = as_numpy(sample["interface_ids"])[mask]
        gaps = []
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                for label in np.intersect1d(labels[i], labels[j]):
                    if label >= 0:
                        gaps.append(chamfer(aligned[i][labels[i] == label], aligned[j][labels[j] == label]))
        result["contact_gap"] = float(np.mean(gaps)) if gaps else None
    return result


def _unique_rows(rows):
    keyed = {}
    for row in rows:
        key = _key(row)
        if key in keyed:
            raise ValueError("Duplicate source/pattern/observation evaluation row")
        keyed[key] = row
    return keyed


def aggregate(rows, expected_ids=None):
    """Expected identities may be samples/records; absent rows count as failures.

    Distance summaries are explicitly valid-solve-only; success and part accuracy
    always use the full declared denominator. No arbitrary finite failure penalty.
    """
    rows = list(rows)
    keyed = _unique_rows(rows)
    expected = None if expected_ids is None else {_key(item) if isinstance(item, dict) else tuple(item) for item in expected_ids}
    if expected is not None and not set(keyed).issubset(expected):
        raise ValueError("Evaluation contains identities outside the expected manifest")
    denominator = len(rows) if expected is None else len(expected)
    missing = denominator - len(rows)
    result = {"count": denominator, "recorded_count": len(rows), "missing_count": missing,
              "success_rate": None, "failure_rate": None, "part_accuracy": None,
              "source_count": len({key[0] for key in (expected if expected is not None else keyed)}),
              "distance_denominator_note": "Distance/pose summaries exclude failed or missing solves; accuracy and success include all expected observations."}
    if not denominator:
        return result
    result.update(success_rate=sum(bool(row["success"]) for row in rows)/denominator,
                  failure_rate=(missing + sum(bool(row["failed"]) for row in rows))/denominator,
                  part_accuracy=sum(float(row.get("part_accuracy", 0)) for row in rows)/denominator,
                  moving_part_accuracy=sum(float(row.get("moving_part_accuracy", 0)) for row in rows)/denominator)
    all_keys = set(keyed) if expected is None else expected
    source_keys = {source: [key for key in all_keys if key[0] == source] for source in {key[0] for key in all_keys}}
    for metric, name in (("success", "success_rate"), ("part_accuracy", "part_accuracy"), ("moving_part_accuracy", "moving_part_accuracy")):
        means = [sum(float(keyed.get(key, {}).get(metric, 0)) for key in group)/len(group) for group in source_keys.values()]
        result["source_macro_" + name] = float(np.mean(means))
    result["refinement_skipped_count"] = sum(bool(row.get("refinement_skipped")) for row in rows)
    result["primary_aggregation"] = "source_macro"
    for metric in ("whole_chamfer", "per_part_chamfer", "rotation_deg", "translation_error", "contact_gap", "runtime_seconds"):
        values = [float(value) for row in rows if row.get(metric) is not None
                  for value in np.asarray(row[metric]).reshape(-1) if np.isfinite(value)]
        result[metric] = {"count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
                          "p90": float(np.percentile(values, 90))} if values else None
    return result


def compare_conditions(baseline_rows, treatment_rows, seed=42, bootstrap_samples=2000):
    """Paired bootstrap resamples source objects, keeping their fractures intact."""
    baseline, treatment = _unique_rows(baseline_rows), _unique_rows(treatment_rows)
    if set(baseline) != set(treatment):
        raise ValueError("Paired comparison requires identical full observation sets, including failure rows")
    keys = sorted(baseline)
    if not keys:
        return {"count": 0, "source_count": 0, "paired": True}
    sources = sorted({key[0] for key in keys})
    if any(not key[0] or not key[1] for key in keys):
        raise ValueError("Paired comparisons need source and pattern identities")
    groups = [np.asarray([i for i, key in enumerate(keys) if key[0] == source]) for source in sources]
    rng = np.random.default_rng(seed)
    resamples = [rng.integers(len(groups), size=len(groups)) for _ in range(bootstrap_samples)]
    report = {"count": len(keys), "source_count": len(sources), "paired": True,
              "bootstrap_unit": "source_object", "bootstrap_samples": bootstrap_samples,
              "seed": seed, "interval_note": "Exploratory when few independent source objects are available."}
    for metric in ("success", "part_accuracy", "moving_part_accuracy"):
        difference = np.asarray([float(treatment[key].get(metric, 0)) - float(baseline[key].get(metric, 0)) for key in keys])
        sums, counts = np.asarray([difference[group].sum() for group in groups]), np.asarray([len(group) for group in groups])
        boot = [float(sums[indices].sum()/counts[indices].sum()) for indices in resamples]
        report[metric + "_delta"] = {"mean": float(difference.mean()),
            "ci95": np.percentile(boot, [2.5, 97.5]).tolist() if boot else None,
            "aggregation": "observation_weighted"}
        source_means = sums/counts
        macro_boot = [float(source_means[indices].mean()) for indices in resamples]
        report[metric + "_source_macro_delta"] = {"mean": float(source_means.mean()),
            "ci95": np.percentile(macro_boot, [2.5, 97.5]).tolist() if macro_boot else None,
            "aggregation": "source_macro"}
    baseline_success = sum(bool(baseline[key]["success"]) for key in keys)
    harmed = sum(bool(baseline[key]["success"]) and not treatment[key]["success"] for key in keys)
    helped = sum(not baseline[key]["success"] and bool(treatment[key]["success"]) for key in keys)
    report.update(harmed_count=harmed, helped_count=helped, baseline_success_count=baseline_success,
                  harm_rate_all=harmed/len(keys), harm_rate_previously_successful=harmed/baseline_success if baseline_success else None,
                  primary_comparison="success_source_macro_delta")
    return report
