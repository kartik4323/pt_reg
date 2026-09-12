"""Paired candidate evaluation, robust contact gates, and XYZ-only inference."""
from __future__ import annotations

import copy
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from reassembly.data import FractureDataset, verify_manifest
from reassembly.evaluation import aggregate, assembly_metrics
from reassembly.geometry import normalize_fragments, export_transforms
from reassembly.losses import contact_geometry
from reassembly.prepare import _file_sha256
from reassembly.resources import write_json
from reassembly.training import _check_cuda_memory
from diagnostics.reassembly_v2.variants import variant, choose_subset

from . import ARCHITECTURE
from .checkpoints import load_checkpoint
from .config import signature
from .data import RepairDataset, collate_samples
from .provenance import append_jsonl, provenance


def load_model(path, device, fingerprint=None):
    from .model import RepairModel, configure_stage
    state = load_checkpoint(path, fingerprint=fingerprint)
    model = RepairModel(state["cfg"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    configure_stage(model, 3)
    return model, state


def probability_metrics(encoded, pairs, sample, cfg):
    batch = collate_samples([sample], encoded["point_xyz"].device)
    count = int(sample["fragment_mask"].sum())
    fragments, pair_rows = [], []
    for i in range(count):
        p = encoded["fracture_logits"][0, i].detach().float().sigmoid().cpu().numpy()
        y = sample["fracture_labels"][i].numpy() > .5
        yes = p >= .5
        tp, fp, fn, tn = [int(mask.sum()) for mask in (yes & y, yes & ~y, ~yes & y, ~yes & ~y)]
        fragments.append({"fragment": i, "precision": tp/max(tp+fp, 1), "recall": tp/max(tp+fn, 1),
            "iou": tp/max(tp+fp+fn, 1), "balanced_accuracy": .5*(tp/max(tp+fn, 1)+tn/max(tn+fp, 1)),
            "predicted_fraction": float(yes.mean()), "true_fraction": float(y.mean()),
            "probability_histogram": np.histogram(p, np.linspace(0, 1, 11))[0].tolist()})
    qa = sample.get("preparation_metadata", {}).get("qa", {})
    for pair in pairs:
        if not bool(pair["valid"][0]):
            continue
        positive, distance = contact_geometry(pair, batch, cfg["train"]["contact_radius"])
        y, d = positive[0].cpu().numpy(), distance[0].cpu().numpy()
        p = pair["source_prob"][0].detach().float().cpu().numpy()
        other = pair["target_prob"][0].detach().float().cpu().numpy()
        weights = pair["weights"][0].detach().float().cpu().numpy()
        rows = np.arange(len(p)); top = p[:, :-1].argmax(-1); valid = y.any(-1)
        i, j = pair["i"], pair["j"]
        interfaces = [set(sample["interface_ids"][k].tolist()) - {-1} for k in (i, j)]
        true_contact = bool(qa["adjacency"][i][j]) if "adjacency" in qa else bool(interfaces[0] & interfaces[1])
        record = {"i": i, "j": j, "true_contact": true_contact,
            "truth_source": "preparation_adjacency" if "adjacency" in qa else "sampled_interfaces",
            "positive_pairs": int(y.sum()), "matchable_rows": int(valid.sum()),
            "top1_recall": float(y[rows, top][valid].mean()) if valid.any() else None,
            "top1_chance": float(y[valid].mean()) if valid.any() else None,
            "top_match_distance": float(d[rows, top][valid].mean()) if valid.any() else None,
            "dustbin_mean": float(p[:, -1].mean()),
            "dustbin_matchable": float(p[valid, -1].mean()) if valid.any() else None,
            "dustbin_unmatched": float(p[~valid, -1].mean()) if (~valid).any() else None,
            "entropy": float(-(p*np.log(np.maximum(p, 1e-12))).sum(-1).mean()),
            "directional_mass": float(p[:, :-1].sum()), "bidirectional_mass": float((p[:, :-1]*other[:, :-1].T).sum()),
            "gated_mass": float(weights.sum()), "max_weight": float(weights.max()),
            "selected_fracture_points": {}, "selected_interfaces": {}}
        for part, indices in ((i, pair["source_indices"][0].cpu()), (j, pair["target_indices"][0].cpu())):
            ids = sample["interface_ids"][part, indices]
            record["selected_fracture_points"][str(part)] = int((ids >= 0).sum())
            record["selected_interfaces"][str(part)] = {str(int(k)): int((ids == k).sum()) for k in ids.unique() if k >= 0}
        pair_rows.append(record)
    return {"fragments": fragments, "pairs": pair_rows}


def summarize(rows):
    result = aggregate(rows)
    sources = sorted({row["source_id"] for row in rows})
    by_source = {s: aggregate([r for r in rows if r["source_id"] == s]) for s in sources}
    result.update(successes=sum(r["success"] for r in rows), source_count=len(sources),
        source_macro_success=float(np.mean([r["success_rate"] for r in by_source.values()])) if sources else None,
        by_source=by_source, by_band={b: aggregate([r for r in rows if r["band"] == b]) for b in ("easy", "intermediate", "hard")},
        by_pieces={str(n): aggregate([r for r in rows if r["pieces"] == n]) for n in (2, 3)},
        failure_reasons=dict(Counter(r.get("reason") or r["status"] for r in rows if not r["success"])))
    return result


def sample_results(model, sample, cfg, device, conditions=("contact_only",), *, measure=True, tensor_path=None, guard=None):
    from .assembly import build_candidates, solve_candidates, candidate_pose_metrics
    from .fields import ContinuousNeuralField, ContinuousGTField, ShiftedField
    batch = collate_samples([sample], device)
    cache = build_candidates(model, batch, cfg)
    coverage = candidate_pose_metrics(cache, sample, cfg["solver"]["success_threshold"])
    encoded = cache.encoded
    metrics = probability_metrics(encoded, cache.pairs, sample, cfg) if measure else {}
    if tensor_path is not None:
        from .losses import contact_metrics
        with torch.no_grad():
            metrics["representation"] = {k: float(v) for k, v in contact_metrics(encoded, cache.pairs, batch, cfg, include_rank=True).items()}
        arrays = {"points": sample["points"].numpy(), "canonical_points": sample["canonical_points"].numpy(),
            "fragment_mask": sample["fragment_mask"].numpy(), "interface_ids": sample["interface_ids"].numpy(),
            "descriptor": encoded["descriptor"].detach().cpu().numpy().astype(np.float16),
            "fracture_probability": encoded["fracture_logits"].detach().float().sigmoid().cpu().numpy()}
        for pair in cache.pairs:
            if not bool(pair["valid"][0]):
                continue
            prefix = f"pair_{pair['i']}_{pair['j']}_"
            for key in ("source_indices", "target_indices", "source_prob", "target_prob", "weights", "source_embedding", "target_embedding"):
                value = pair[key][0].detach().cpu().numpy()
                arrays[prefix+key] = value.astype(np.float16) if "embedding" in key else value
        if guard:
            guard.check(additional_bytes=sum(a.nbytes for a in arrays.values())+4096)
        Path(tensor_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tensor_path, **arrays)
    field = None
    outputs = {}
    for condition in conditions:
        if condition not in ("contact_only", "predicted", "gt", "perturbed"):
            raise ValueError(f"Unknown field condition {condition}")
        if condition in ("predicted", "perturbed") and field is None:
            field = ContinuousNeuralField(model, encoded, chunk_size=cfg["solver"]["field_chunk"])
        selected_field = {"contact_only": None, "predicted": field}.get(condition)
        if condition == "gt":
            selected_field = ContinuousGTField.from_sample(sample, truncation=cfg["model"]["truncation"])
        elif condition == "perturbed":
            selected_field = ShiftedField(field, np.array([.12, -.08, .06]))
        result = solve_candidates(cache, cfg, selected_field)
        row = assembly_metrics(sample, result, cfg["solver"]["success_threshold"])
        row.update(pattern_id=sample["pattern_id"], source_id=sample["source_id"], band=sample["band"],
            pieces=int(sample["fragment_mask"].sum()), cut_family=sample["cut_family"], condition=condition,
            reason=result.get("reason"), diagnostics=result["diagnostics"], candidate_coverage=coverage, **metrics)
        row["false_contact_candidate_pairs"] = sum(
            not p["true_contact"] and result["diagnostics"]["pairs"].get(f"{p['i']}-{p['j']}", {}).get("candidate_count", 0) > 0
            for p in metrics.get("pairs", []))
        outputs[condition] = (row, result)
    _check_cuda_memory(torch.device(device), cfg)
    return outputs


def evaluate(checkpoint, manifest, output, device, *, split="val", conditions=("contact_only", "predicted", "gt", "perturbed"),
             seeds=(4101, 4102, 4103), guard=None, query_cache=None, overfit=False, modes=("samples_poses",), resume=False):
    integrity = verify_manifest(manifest)
    model, state = load_model(checkpoint, device, integrity["dataset_fingerprint"])
    cfg = state["cfg"]
    if state["stage"] == 1 and any(c != "contact_only" for c in conditions):
        raise ValueError("A stage-1 checkpoint has no trained scaffold; evaluate contact_only")
    if (state["purpose"] == "overfit") != overfit:
        raise ValueError("Overfit checkpoints are evaluated only on their fixed training geometry")
    dataset = FractureDataset(manifest, "train" if overfit else split, cfg, fixed=True, limit=16 if overfit else None)
    balancer = RepairDataset(manifest, "train" if overfit else split, cfg, fixed=True, views=False,
                             balanced_fields=True, query_cache=query_cache)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Choose a fresh evaluation directory")
    output.mkdir(parents=True, exist_ok=True)
    before = _file_sha256(checkpoint)
    invocation = {"architecture": ARCHITECTURE, "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": before,
        "dataset_fingerprint": dataset.fingerprint, "config_signature": signature(cfg), "split": split,
        "purpose": state["purpose"], "seeds": list(seeds), "modes": list(modes), "conditions": list(conditions),
        "pattern_ids": [r["pattern_id"] for r in dataset.records], "provenance": provenance(device)}
    rows, started = [], time.perf_counter()
    invocation_path = output / "invocation.json"
    if resume and invocation_path.exists():
        old = json.loads(invocation_path.read_text(encoding="utf-8"))
        keys = ("architecture", "checkpoint", "checkpoint_sha256", "dataset_fingerprint", "config_signature", "split",
                "purpose", "seeds", "modes", "conditions", "pattern_ids")
        if any(old.get(k) != invocation[k] for k in keys) or old["provenance"]["code"] != invocation["provenance"]["code"]:
            raise ValueError("Evaluation resume inputs or implementation changed")
        rows_path = output / "examples.jsonl"
        if rows_path.exists():
            # A killed append can leave only its final line incomplete. Keep all
            # completed rows and record the recovery without accepting corruption.
            raw = rows_path.read_bytes()
            offset = 0
            for line in raw.splitlines(keepends=True):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if offset + len(line) != len(raw) or line.endswith(b"\n"):
                        raise ValueError("Corrupt completed evaluation record")
                    with rows_path.open("r+b") as handle:
                        handle.truncate(offset)
                    write_json(output / "resume_recovery.json", {"discarded_incomplete_tail_bytes": len(line)}, guard)
                offset += len(line)
        invocation = old
    else:
        write_json(invocation_path, invocation, guard)
    row_key = lambda r: (r["mode"], r["seed"], r["pattern_id"], r["condition"])
    completed = {row_key(r) for r in rows}
    tensor_subset = {dataset.records[i]["pattern_id"] for i in choose_subset(dataset.records, 12)}
    if len(completed) != len(rows):
        raise ValueError("Duplicate evaluation observations in resume evidence")
    expected = {(m, s, p["pattern_id"], c) for m in modes for s in ([0] if m == "unchanged" else seeds)
                for p in dataset.records for c in conditions}
    if not completed.issubset(expected):
        raise ValueError("Unexpected examples in resumed evaluation")
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for mode in modes:
        for seed in ([0] if mode == "unchanged" else seeds):
            for index, record in enumerate(dataset.records):
                if all((mode, int(seed), record["pattern_id"], c) in completed for c in conditions):
                    continue
                sample = variant(dataset, index, mode, seed)
                with np.load(dataset.root / record["path"], allow_pickle=False) as archive:
                    sample["preparation_metadata"] = json.loads(str(archive["metadata"].item()))
                tensor_path = (output / "tensors" / (record["pattern_id"] + ".npz")) if (
                    mode == "samples_poses" and seed == 4101 and record["pattern_id"] in tensor_subset) else None
                results = sample_results(model, sample, cfg, device, conditions, tensor_path=tensor_path, guard=guard)
                if state["stage"] == 2:
                    balancer.balance_queries(sample)
                    batch = collate_samples([sample], device)
                    from .losses import compute_losses
                    with torch.no_grad():
                        field_metrics = {k: float(v) for k, v in compute_losses(model, batch, 2, cfg).items()}
                else:
                    field_metrics = {}
                for condition, (row, result) in results.items():
                    row.update(seed=int(seed), mode=mode, field_metrics=field_metrics)
                    if row_key(row) in completed:
                        continue
                    rows.append(row)
                    append_jsonl(output / "examples.jsonl", row, guard)
                    completed.add(row_key(row))
                    # Qualitative outputs are strictly bounded, even for GT runs.
                    if len(rows) <= cfg["repair"]["qualitative_limit"]:
                        arrays = {"points": sample["points"].numpy(), "target": sample["canonical_points"].numpy(),
                                  "mask": sample["fragment_mask"].numpy()}
                        if result["rotations"] is not None:
                            arrays.update(rotations=result["rotations"], translations=result["translations"])
                        if guard:
                            guard.check(additional_bytes=sum(x.nbytes for x in arrays.values()) + 4096)
                        np.savez_compressed(output / f"example_{len(rows):02d}.npz", **arrays)
                write_json(output / "progress.json", {"completed_examples": len(rows), "mode": mode, "seed": seed,
                    "pattern_id": record["pattern_id"], "seconds": time.perf_counter()-started}, guard)
                print(f"evaluate {split} {mode} seed={seed} {index+1}/{len(dataset)} " +
                      " ".join(f"{c}={int(r[0]['success'])}" for c, r in results.items()), flush=True)
    if _file_sha256(checkpoint) != before:
        raise RuntimeError("Checkpoint changed during evaluation")
    summaries = {f"{mode}/{seed}/{condition}": summarize([r for r in rows if (r["mode"], r["seed"], r["condition"]) == (mode, seed, condition)])
        for mode in modes for seed in ([0] if mode == "unchanged" else seeds) for condition in conditions}
    report = dict(invocation, status="completed", kind="repair_evaluation", summaries=summaries,
        seconds=time.perf_counter()-started, checkpoint_unchanged=True,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if torch.device(device).type == "cuda" else None,
        metric_threshold=.01, results_sha256=_file_sha256(output / "examples.jsonl"),
        tensor_pattern_ids=sorted(tensor_subset), tensor_observation_seed=4101,
        source_note="Patterns from one source are correlated; cut_holdout reuses test sources previously inspected.")
    write_json(output / "evaluation.json", report, guard)
    return report


def contact_gate(checkpoint, overfit_checkpoint, manifest, output, device, guard=None, *, resume=False):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Choose a fresh contact gate directory")
    state = load_checkpoint(checkpoint, stage=1, purpose="experiment")
    overfit_state = load_checkpoint(overfit_checkpoint, cfg=state["cfg"], fingerprint=state["dataset_fingerprint"], stage=1, purpose="overfit")
    regression = evaluate(overfit_checkpoint, manifest, output / "overfit", device,
        split="train", conditions=("contact_only",), seeds=(4101, 4102, 4103), overfit=True,
        modes=("unchanged", "rotation", "samples", "samples_poses"), guard=guard, resume=resume)
    training = evaluate(checkpoint, manifest, output / "training", device, split="train",
        conditions=("contact_only",), seeds=(4101,), modes=("samples_poses",), guard=guard, resume=resume)
    fixed = regression["summaries"]["unchanged/0/contact_only"]
    robustness = {mode: sum(regression["summaries"][f"{mode}/{s}/contact_only"]["successes"] for s in (4101, 4102, 4103))/48
                  for mode in ("rotation", "samples", "samples_poses")}
    training_rate = training["summaries"]["samples_poses/4101/contact_only"]["success_rate"]
    checks = {"fixed_16": fixed["count"] == 16 and fixed["successes"] == 16 and fixed["low_confidence_rate"] == 0,
        **{f"robust_{mode}": rate >= .9 for mode, rate in robustness.items()}, "training_80_percent": training_rate >= .8}
    report = {"kind": "repair_contact_gate", "architecture": ARCHITECTURE, "passed": all(checks.values()),
        "checks": checks, "robustness": robustness, "training_success": training_rate,
        "dataset_fingerprint": state["dataset_fingerprint"], "model_signature": state["model_signature"],
        "checkpoint_sha256": _file_sha256(checkpoint), "overfit_checkpoint_sha256": _file_sha256(overfit_checkpoint),
        "evidence": {name: {"path": str(output / name / "evaluation.json"), "sha256": _file_sha256(output / name / "evaluation.json")}
                     for name in ("overfit", "training")}}
    write_json(output / "contact_gate.json", report, guard)
    return report


def require_contact_gate(path, checkpoint, fingerprint):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("kind") != "repair_contact_gate" or not report.get("passed") or not all(report.get("checks", {}).values()):
        raise ValueError("Contact assembly gate has not passed; field training is blocked")
    if report["checkpoint_sha256"] != _file_sha256(checkpoint) or report["dataset_fingerprint"] != fingerprint:
        raise ValueError("Contact gate does not certify this checkpoint and dataset")
    expected = {"fixed_16", "robust_rotation", "robust_samples", "robust_samples_poses", "training_80_percent"}
    if (set(report["checks"]) != expected or report["training_success"] < .8
            or set(report.get("robustness", {})) != {"rotation", "samples", "samples_poses"}
            or any(x < .9 for x in report["robustness"].values())
            or set(report.get("evidence", {})) != {"overfit", "training"}):
        raise ValueError("Incomplete or weakened contact gate")
    for name, record in report["evidence"].items():
        evidence = Path(record["path"])
        if _file_sha256(evidence) != record["sha256"]:
            raise ValueError("Contact gate evidence changed")
        document = json.loads(evidence.read_text(encoding="utf-8"))
        expected_checkpoint = report["overfit_checkpoint_sha256"] if name == "overfit" else report["checkpoint_sha256"]
        if (document.get("status") != "completed" or document.get("dataset_fingerprint") != fingerprint
                or document.get("checkpoint_sha256") != expected_checkpoint or document.get("metric_threshold") != .01):
            raise ValueError("Contact gate evaluation identity or criterion differs")
        if _file_sha256(evidence.parent / "examples.jsonl") != document["results_sha256"]:
            raise ValueError("Contact gate per-example evidence changed")
        if name == "overfit":
            fixed = document["summaries"]["unchanged/0/contact_only"]
            if fixed["count"] != 16 or fixed["successes"] != 16 or fixed["low_confidence_rate"] != 0:
                raise ValueError("Original sixteen-example contact regression did not pass")
            for mode in ("rotation", "samples", "samples_poses"):
                summaries = [document["summaries"][f"{mode}/{seed}/contact_only"] for seed in (4101, 4102, 4103)]
                if any(s["count"] != 16 for s in summaries) or sum(s["successes"] for s in summaries)/48 < .9:
                    raise ValueError("Incomplete or failed rotation/resampling contact controls")
        else:
            summary = document["summaries"]["samples_poses/4101/contact_only"]
            if summary["count"] != len(document["pattern_ids"]) or summary["count"] < 1 or summary["successes"]/summary["count"] < .8:
                raise ValueError("The deterministic fresh training-geometry gate did not pass")
    return report


def infer_fragments(fragments, checkpoint, device, *, condition="predicted", save_scaffold=False):
    from .assembly import build_candidates, solve_candidates
    from .fields import ContinuousNeuralField
    if condition not in ("contact_only", "predicted"):
        raise ValueError("XYZ inference does not accept oracle fields")
    model, state = load_model(checkpoint, device)
    if condition == "predicted" and state["stage"] != 2:
        raise ValueError("Predicted-scaffold inference requires a trained stage-2 checkpoint")
    cfg = state["cfg"]
    normalized = normalize_fragments(fragments, cfg["data"]["points_per_fragment"], cfg["data"]["seed"])
    batch = {"points": torch.as_tensor(normalized["points"], device=device)[None],
        "fragment_mask": torch.as_tensor(normalized["fragment_mask"], device=device)[None],
        "anchor_index": torch.tensor([normalized["anchor_index"]], device=device)}
    cache = build_candidates(model, batch, cfg)
    field = ContinuousNeuralField(model, cache.encoded, chunk_size=cfg["solver"]["field_chunk"]) if condition == "predicted" else None
    result = solve_candidates(cache, cfg, field)
    if result["rotations"] is not None:
        result.update(export_transforms(result["rotations"], result["translations"], normalized))
    else:
        result.update(transforms=None, aligned_fragments=None)
    result.update(architecture=ARCHITECTURE, schema_version=3, reference_index=normalized["anchor_index"],
        convention="x_aligned = x @ R.T + t", checkpoint_sha256=_file_sha256(checkpoint),
        dataset_fingerprint=state["dataset_fingerprint"])
    result["scaffold_frame"] = {"name": "normalized_reference", "scale": normalized["scale"],
        "reference_centroid": normalized["centroids"][normalized["anchor_index"]],
        "to_output_coordinates": "x_output = scale * x_field + reference_centroid"}
    if save_scaffold:
        if field is None:
            raise ValueError("Scaffold visualization requires predicted-field inference")
        from .fields import diagnostic_grid
        result["scaffold"] = diagnostic_grid(field, resolution=cfg["solver"]["resolution"],
            extent=cfg["solver"]["field_extent"], chunk_size=cfg["solver"]["field_chunk"]).as_dict()
        result["scaffold_note"] = "Coarse visualization only; poses use continuous field evaluation."
    return result
