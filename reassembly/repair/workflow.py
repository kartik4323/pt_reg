"""Predeclared factorial experiments, replication, gated fields, and final tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reassembly.prepare import _file_sha256
from reassembly.resources import write_json

from .config import signature, variant_config
from .data import supplement_queries
from .evaluation import evaluate, contact_gate
from .reporting import acceptance, read_rows, render_and_bundle
from .training import dataset_integrity, preflight, train_stage, verify_run_proofs
from .provenance import code_inventory


VARIANTS = [(g, v) for g in ("existing", "revised") for v in ("existing", "resampled_contrastive")]


def label(geometry, supervision):
    return f"{geometry}__{supervision}"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _reuse(path, resume):
    if Path(path).exists():
        if not resume:
            raise FileExistsError(f"Existing experiment artifact: {path}. Use --resume for this workflow.")
        return _read(path)
    return None


def _evaluation_evidence(path, checkpoint, fingerprint, split, conditions):
    report, rows = read_rows(path)
    if (report["checkpoint_sha256"] != _file_sha256(checkpoint) or report["dataset_fingerprint"] != fingerprint
            or report["split"] != split or report["modes"] != ["samples_poses"]
            or report["seeds"] != [4101, 4102, 4103] or set(report["conditions"]) != set(conditions)
            or report["provenance"]["code"]["sha256"] != code_inventory()["sha256"]):
        raise ValueError("Evaluation evidence differs from this locked experiment")
    return report, rows


def _train(cfg, manifest, root, stage, device, purpose, diagnostic, profile, query_cache,
           resume, guard, initialize=None, gate=None):
    report_path = root / "training_report.json"
    budget = cfg["train"]["overfit_updates" if purpose == "overfit" else "max_updates"]
    checkpoint = None
    old = _reuse(report_path, resume)
    if old is not None:
        metadata = verify_run_proofs(root)
        if metadata["field_diagnostic_sha256"] != _file_sha256(diagnostic):
            raise ValueError("Reused workflow diagnostic proof changed")
        previous = _read(root / "config.resolved.json")
        if signature(previous) != signature(cfg):
            raise ValueError("Workflow configuration changed; explicit train resume handles deliberate budget extensions")
        if old.get("status") == "completed" and old.get("updates") == budget:
            if _file_sha256(root / "best.pt") != old["best_checkpoint_sha256"]:
                raise ValueError("A completed workflow checkpoint changed")
            return old
        checkpoint = root / "latest.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"Interrupted run has no checkpoint yet: {root}. Use a fresh output directory.")
    return train_stage(cfg, manifest, root, stage, device, purpose=purpose,
        preflight_report=profile, field_diagnostic_report=diagnostic, query_cache=query_cache,
        initialize_from=initialize if checkpoint is None else None, resume=checkpoint,
        contact_gate_report=gate, guard=guard)


def _contact_experiment(cfg, manifest, root, device, diagnostic, query_cache, resume, guard):
    profile_path = root / "preflight" / "preflight.json"
    profile = _reuse(profile_path, resume)
    if profile is None:
        profile = preflight(cfg, manifest, profile_path.parent, device, query_cache=query_cache, guard=guard)
    resolved = profile["config"]
    if profile["status"] == "failed":
        raise RuntimeError(f"Preflight failed: {profile_path}")
    # Immutable comparison inputs, even if a prior preflight selected batch1.
    for section in ("repair", "model", "data", "loss", "solver"):
        if resolved[section] != cfg[section]:
            raise ValueError("Existing preflight belongs to a different comparison")
    for key in cfg["train"]:
        if key not in ("batch_size", "grad_accum_steps") and cfg["train"][key] != resolved["train"][key]:
            raise ValueError("Existing preflight uses a different training seed/budget/optimizer")
    original = _train(resolved, manifest, root / "overfit", 1, device, "overfit", diagnostic,
        profile_path, query_cache, resume, guard)
    contacts = _train(resolved, manifest, root / "contacts", 1, device, "experiment", diagnostic,
        profile_path, query_cache, resume, guard)
    gate_path = root / "gate" / "contact_gate.json"
    gate = _reuse(gate_path, resume)
    if gate is None:
        gate = contact_gate(contacts["best_checkpoint"], original["best_checkpoint"], manifest, gate_path.parent, device, guard, resume=resume)
    else:
        if (gate["checkpoint_sha256"] != _file_sha256(contacts["best_checkpoint"])
                or gate["overfit_checkpoint_sha256"] != _file_sha256(original["best_checkpoint"])):
            raise ValueError("Reused gate checkpoint changed")
        if gate["passed"]:
            from .evaluation import require_contact_gate
            require_contact_gate(gate_path, contacts["best_checkpoint"], contacts["dataset_fingerprint"])
    validation_path = root / "contact_validation" / "evaluation.json"
    validation = _reuse(validation_path, resume)
    if validation is None:
        validation = evaluate(contacts["best_checkpoint"], manifest, validation_path.parent, device,
            split="val", conditions=("contact_only",), query_cache=query_cache, guard=guard, resume=resume)
    document, rows = _evaluation_evidence(validation_path, contacts["best_checkpoint"], contacts["dataset_fingerprint"], "val", ("contact_only",))
    sources = sorted({r["source_id"] for r in rows})
    success = float(np.mean([r["success"] for r in rows]))
    macro = float(np.mean([np.mean([r["success"] for r in rows if r["source_id"] == source]) for source in sources]))
    recall = [p["top1_recall"] for r in rows for p in r["pairs"] if p["top1_recall"] is not None]
    entry = {"variant": label(cfg["repair"]["geometry_variant"], cfg["repair"]["view_supervision"]),
        "geometry_variant": cfg["repair"]["geometry_variant"], "view_supervision": cfg["repair"]["view_supervision"],
        "training_seed": cfg["train"]["seed"], "root": str(root), "gate_passed": gate["passed"],
        "contact_success": success, "source_macro_success": macro, "matching_top1_recall": float(np.mean(recall)) if recall else 0,
        "checkpoint": contacts["best_checkpoint"], "validation_report": str(validation_path),
        "validation_sha256": _file_sha256(validation_path), "checkpoint_sha256": _file_sha256(contacts["best_checkpoint"]),
        "gate_report": str(gate_path), "profile_report": str(profile_path), "config": resolved}
    write_json(root / "experiment.json", entry, guard)
    return entry


def run(cfg, manifest, output, device, *, phase, field_diagnostic_report, query_cache=None, resume=False, guard=None):
    output = Path(output).resolve()
    integrity = dataset_integrity(manifest, cfg)
    diagnostic = _read(field_diagnostic_report)
    if diagnostic.get("kind") != "focused_field_diagnostic" or diagnostic.get("status") != "complete" or diagnostic.get("dataset_fingerprint") != integrity["dataset_fingerprint"]:
        raise ValueError("Complete the focused VM field diagnostic before starting repair experiments")
    metadata_path = output / "workflow.json"
    old = _reuse(metadata_path, resume)
    identity = {"dataset_fingerprint": integrity["dataset_fingerprint"], "config_signature": signature(cfg),
        "field_diagnostic_sha256": _file_sha256(field_diagnostic_report)}
    if old and any(old.get(k) != v for k, v in identity.items()):
        raise ValueError("Workflow inputs changed; select a fresh experiment root")
    if not old:
        output.mkdir(parents=True, exist_ok=True)
        write_json(metadata_path, dict(identity, configuration=cfg), guard)
    if query_cache is None:
        query_cache = output / "queries" / "query_cache.json"
        if not query_cache.exists():
            supplement_queries(manifest, query_cache.parent, guard)
    query_cache = Path(query_cache)
    if phase == "comparisons":
        entries = []
        for geometry, supervision in VARIANTS:
            variant = variant_config(cfg, geometry, supervision, 42)
            entry = _contact_experiment(variant, manifest, output / "experiments" / label(geometry, supervision) / "seed42",
                device, field_diagnostic_report, query_cache, resume, guard)
            entries.append(entry)
            write_json(output / "comparison_progress.json", {"completed": entries}, guard)
        eligible = [e for e in entries if e["gate_passed"]]
        eligible.sort(key=lambda e: (-e["source_macro_success"], -e["contact_success"], -e["matching_top1_recall"], e["variant"]))
        selection = dict(identity, kind="repair_selection", status="selected" if eligible else "contact_gate_failed",
            selected=eligible[0]["variant"] if eligible else None, experiments=entries,
            selection_rule="Passed contacts gate, then source-macro assembly success, assembly success, retrieval; deterministic name tie-break")
        write_json(output / "selection.json", selection, guard)
        result = selection
    else:
        selection = _read(output / "selection.json")
        if selection.get("status") != "selected":
            raise ValueError("No contact configuration passed its gate; review comparison results before further stages")
        chosen = next(e for e in selection["experiments"] if e["variant"] == selection["selected"])
        if phase == "replicate":
            entries = []
            pairs = {(chosen["geometry_variant"], chosen["view_supervision"]), ("existing", "existing")}
            for geometry, supervision in sorted(pairs):
                for seed in (43, 44):
                    entry = _contact_experiment(variant_config(cfg, geometry, supervision, seed), manifest,
                        output / "experiments" / label(geometry, supervision) / f"seed{seed}", device,
                        field_diagnostic_report, query_cache, resume, guard)
                    entries.append(entry)
                    write_json(output / "replication_progress.json", {"completed": entries}, guard)
            result = {"kind": "repair_replication", "status": "completed", "experiments": entries}
            write_json(output / "replication.json", result, guard)
        elif phase == "scaffold":
            reports = []
            for seed in (42, 43, 44):
                root = output / "experiments" / chosen["variant"] / f"seed{seed}"
                entry = _read(root / "experiment.json")
                if not entry["gate_passed"]:
                    raise ValueError(f"Contact gate failed for seed{seed}; no scaffold training may proceed")
                trained = _train(entry["config"], manifest, root / "scaffold", 2, device, "experiment", field_diagnostic_report,
                    entry["profile_report"], query_cache, resume, guard, initialize=entry["checkpoint"], gate=entry["gate_report"])
                report_path = root / "paired_validation" / "evaluation.json"
                evaluation = _reuse(report_path, resume)
                if evaluation is None:
                    evaluate(trained["best_checkpoint"], manifest, report_path.parent, device,
                        split="val", query_cache=query_cache, guard=guard, resume=resume)
                _evaluation_evidence(report_path, trained["best_checkpoint"], integrity["dataset_fingerprint"], "val",
                                     ("contact_only", "predicted", "gt", "perturbed"))
                reports.append(report_path)
            result = acceptance(reports, output, guard)
        elif phase == "final-test":
            decision = _read(output / "acceptance.json")
            if not decision.get("passed"):
                raise ValueError("Validation acceptance failed; final-test configuration is not approved by the experiment gates")
            reports = []
            for record in decision["runs"]:
                if _file_sha256(record["report"]) != record["sha256"]:
                    raise ValueError("Validation evidence changed after configuration selection")
                validation, _ = read_rows(record["report"])
                checkpoint = validation["checkpoint"]
                if _file_sha256(checkpoint) != record["checkpoint_sha256"]:
                    raise ValueError("Selected field checkpoint changed after validation")
                for split in ("test", "cut_holdout"):
                    destination = output / "final_test" / f"seed{record['training_seed']}" / split
                    report = _reuse(destination / "evaluation.json", resume)
                    if report is None:
                        report = evaluate(checkpoint, manifest, destination, device, split=split, query_cache=query_cache, guard=guard, resume=resume)
                    _evaluation_evidence(destination / "evaluation.json", checkpoint, integrity["dataset_fingerprint"], split,
                                         ("contact_only", "predicted", "gt", "perturbed"))
                    reports.append(str(destination / "evaluation.json"))
            result = {"kind": "repair_final_test", "status": "completed", "reports": reports,
                "note": "Configuration locked on validation. These test sources were inspected in earlier diagnostics."}
            write_json(output / "final_test.json", result, guard)
        else:
            raise ValueError("phase must be comparisons, replicate, scaffold, or final-test")
    result["bundle"] = render_and_bundle(output, guard)
    return result
