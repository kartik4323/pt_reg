"""Paired A0-A7 experiments replaying sealed observations and native outputs."""
from __future__ import annotations

import copy
import hashlib
import time

from .data import observation_fingerprint
from .evaluation import evaluate_prediction, aggregate, compare_conditions
from .evaluation.refinement import refine_prediction, rank_candidates
from .io import RunLock, append_jsonl, read_jsonl, study_root, owned_output, write_json, jsonable, sha256_file, digest_json
from .priors import GroundTruthPrior, WrongPrior, GenericPrior, ContentControl, TokenSnapshotPrior, select_control
from .training import make_dataset, make_prior
from .resources import ResourceGuard


def key(row):
    return tuple(str(row[name]) for name in ("source_id", "pattern_id", "observation_id"))


def _skip(original, condition, reason):
    chosen = copy.deepcopy(original)
    chosen["refinement"] = {"condition": condition, "status": "skipped", "skip_reason": reason, "steps": 0}
    return chosen


def frozen_comparison(*, root, manifest, predictions, output, cfg, prior_v3=None, prior_v2=None,
                      split="val", observation_seed=4101, mode="refine", steps=25,
                      device="cpu", conditions=None, v3_control="predicted", control_seed=None,
                      control_magnitude=.01):
    if mode not in ("refine", "rank"):
        raise ValueError("Frozen comparison mode must be refine or rank")
    root = study_root(root)
    output = owned_output(root, output)
    if output.exists():
        raise ValueError("Comparison output already exists")
    rows = read_jsonl(predictions)
    budgets = {int(row["points_per_fragment"]) for row in rows if "points_per_fragment" in row}
    if len(budgets) > 1 or (budgets and any("points_per_fragment" not in row for row in rows)):
        raise ValueError("Sealed observations must all declare the same input point budget")
    cfg = copy.deepcopy(cfg)
    if budgets:
        cfg["data"]["points_per_fragment"] = budgets.pop()
    keyed = {key(row): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("Duplicate baseline observation records")
    conditions = list(conditions or ["A%d" % i for i in range(8)])
    if len(conditions) != len(set(conditions)) or any(x not in ["A%d" % i for i in range(8)] for x in conditions):
        raise ValueError("Unknown or duplicate frozen condition")
    dataset = make_dataset(manifest, split, cfg, fixed=True, seed=observation_seed)
    if len(dataset) != len(rows):
        raise ValueError("Baseline export must contain every requested observation, including failures")
    report = {"experiment_id": "scaffold_sota", "mode": mode, "split": split,
              "observation_seed": observation_seed, "baseline_sha256": sha256_file(predictions),
              "manifest_sha256": sha256_file(manifest), "dataset_fingerprint": dataset.fingerprint,
              "points_per_fragment": cfg["data"]["points_per_fragment"],
              "v3_content_control": v3_control, "control_magnitude": control_magnitude,
              "conditions": {}, "expected_count": len(rows),
              "note": "Every condition retains every observation. Unavailable interventions preserve native poses and are explicitly skipped; dependency-pending arms are not evidence of a zero effect."}
    guard = ResourceGuard(root, cfg["resources"], device)
    with RunLock(root):
        guard.require_idle_device()
        guard.check()
        v3 = make_prior(prior_v3, cfg, dataset, device=device)
        v2 = make_prior(prior_v2, cfg, dataset, device=device)
        if v3 and v3.provenance["architecture"] != "fragment-assembly-repair-v3":
            raise ValueError("--prior-v3 must actually be v3")
        if v2 and v2.provenance["architecture"] != "coarse-scaffold-reassembly-v2.1-local-contacts":
            raise ValueError("--prior-v2 must actually be v2")
        exterior_provider = v3 or v2
        v3_intervention = select_control(v3, v3_control, magnitude=control_magnitude,
                                         seed=cfg["train"]["seed"] if control_seed is None else control_seed) if v3 else None
        report["exterior_estimator"] = exterior_provider.provenance if exterior_provider else None
        output.mkdir(parents=True)
        generic, train_set, donor_indices = None, None, []
        if v3 and ("A6" in conditions or "A7" in conditions):
            train_set = make_dataset(manifest, "train", cfg, fixed=True, seed=cfg["data"]["seed"])
            selected = {}
            for i, record in enumerate(train_set.records):
                selected.setdefault(record["source_id"], i)
            donor_indices = list(selected.values())
            if "A7" in conditions:
                generic = GenericPrior.from_dataset(v3, train_set, indices=donor_indices)
        oracle = GroundTruthPrior(diagnostic_only=True, query_count=cfg["prior"]["query_count"],
                                  extent=cfg["prior"]["extent"], device=device) if "A5" in conditions else None
        providers = {"A0": None, "A1": None, "A2": v3_intervention, "A3": v3_intervention, "A4": v2,
                     "A5": oracle, "A6": v3, "A7": generic}
        dependency_pending = {condition for condition in conditions if condition not in ("A0", "A1")
                              and (providers[condition] is None or exterior_provider is None)}
        results = {name: [] for name in conditions}
        native_rows = []
        expected = []
        for index in range(len(dataset)):
            sample = dataset[index]
            actual_hash = observation_fingerprint(sample)
            if actual_hash != sample["observation_id"]:
                raise ValueError("Dataset observation ID does not match its input points")
            identity = key(sample)
            expected.append(identity)
            if identity not in keyed:
                raise ValueError("Baseline and replay inputs differ: point budget, seed and exact point hash must match")
            baseline_record = keyed[identity]
            if baseline_record.get("input_sha256", actual_hash) != actual_hash:
                raise ValueError("Sealed baseline input hash differs from replayed points")
            original = baseline_record.get("prediction")
            if original is None:
                raise ValueError("Baseline rows need sealed prediction payloads, including failures")
            if "candidates" in baseline_record and "candidates" not in original:
                original = {**original, "candidates": baseline_record["candidates"]}
            native_rows.append(evaluate_prediction(sample, original, threshold=cfg["evaluation"]["threshold"]))
            started = time.monotonic()
            exterior_error, exterior = None, None
            try:
                exterior = exterior_provider.exterior_probabilities(sample) if exterior_provider else None
            except (ValueError, RuntimeError) as exc:
                exterior_error = type(exc).__name__+": "+str(exc)
            estimator_seconds = time.monotonic()-started
            snapshots, generation_seconds = {}, {}
            for condition in conditions:
                provider = providers[condition]
                started = time.monotonic()
                chosen, intervention_error, provenance = None, None, None
                if condition in dependency_pending:
                    chosen = _skip(original, condition, "dependency_pending")
                elif condition == "A0":
                    chosen = copy.deepcopy(original)  # Native output, even when its selected candidate is not index zero.
                elif exterior_error and condition != "A1":
                    chosen = _skip(original, condition, "exterior_estimator_failed")
                    intervention_error = exterior_error
                else:
                    try:
                        if condition == "A6":
                            eligible = [i for i in donor_indices if train_set.records[i]["source_id"] != sample["source_id"]]
                            if not eligible:
                                raise ValueError("No different-source training donor")
                            offset = int(hashlib.sha256(sample["source_id"].encode()).hexdigest()[:16], 16) % len(eligible)
                            provider = WrongPrior(v3, train_set[eligible[offset]])  # One lazy donor, never all meshes in memory.
                        if provider is not None:
                            cache_key = "v3" if condition in ("A2", "A3") else condition
                            if cache_key not in snapshots:
                                generation_started = time.monotonic()
                                snapshots[cache_key] = TokenSnapshotPrior(provider.sample_tokens(sample), sample)
                                generation_seconds[cache_key] = time.monotonic()-generation_started
                            provider = snapshots[cache_key]
                            if condition != "A3":
                                provider = ContentControl(provider, "constant_uncertainty")
                            provenance = provider.sample_tokens(sample)["provenance"]
                        if mode == "rank":
                            candidates = original.get("candidates")
                            unique_count = baseline_record.get("unique_valid_candidates")
                            if unique_count is None and candidates:
                                unique_count = len({digest_json({"rotations": item["rotations"], "translations": item["translations"]})
                                                    for item in candidates if item.get("rotations") is not None and item.get("translations") is not None})
                            if not candidates or len(candidates) < 2 or unique_count < 2:
                                chosen = _skip(original, condition, "insufficient_candidate_pool")
                            elif condition == "A1":
                                chosen = copy.deepcopy(original)
                                chosen["ranking"] = {"selector": "native_persisted", "candidate_count": len(candidates)}
                            else:
                                ranking = rank_candidates(sample, candidates, provider, exterior_probabilities=exterior,
                                                          uncertainty=condition == "A3", device=device)
                                selected_index = ranking["selected_index"]
                                chosen = copy.deepcopy(candidates[selected_index]) if selected_index is not None else {"status": "no_valid_candidate", "rotations": None, "translations": None}
                                chosen["ranking"] = ranking
                        else:
                            chosen = refine_prediction(sample, original, provider, condition=condition,
                                                       steps=steps, exterior_probabilities=exterior, device=device)
                    except (ValueError, RuntimeError, KeyError, TypeError, IndexError) as exc:
                        intervention_error = type(exc).__name__+": "+str(exc)
                        chosen = _skip(original, condition, "intervention_error")
                row = evaluate_prediction(sample, chosen, threshold=cfg["evaluation"]["threshold"])
                row.update(condition=condition, prediction=jsonable(chosen), input_sha256=actual_hash,
                           points_per_fragment=cfg["data"]["points_per_fragment"],
                           native_prediction_sha256=digest_json(original), provider_provenance=provenance,
                           intervention_error=intervention_error, common_estimator_seconds=estimator_seconds if condition != "A0" else 0.,
                           prior_generation_seconds=generation_seconds.get("v3" if condition in ("A2", "A3") else condition, 0.),
                           intervention_seconds=time.monotonic()-started)
                results[condition].append(row)
                append_jsonl(output/(condition+".jsonl"), row)
            if index % 8 == 0:
                guard.check()
        for condition, values in results.items():
            if len(values) != len(rows):
                raise ValueError("A condition lost expected observations")
            metrics = aggregate(values, expected_ids=expected)
            skipped = metrics["refinement_skipped_count"]
            status = "dependency_pending" if condition in dependency_pending else "all_interventions_skipped" if skipped == len(values) else "completed_with_skips" if skipped else "completed"
            report["conditions"][condition] = {"status": status, "metrics": metrics,
                "paired_vs_A0": None if status in ("dependency_pending", "all_interventions_skipped") else compare_conditions(native_rows, values, seed=cfg["train"]["seed"],
                                                     bootstrap_samples=cfg["evaluation"]["bootstrap_samples"])}
        report["resources"] = guard.check()
        write_json(output/"comparison.json", report)
    return report
