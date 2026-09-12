"""Measured, fresh repair experiments with gates and explicit same-run resume."""
from __future__ import annotations

import copy
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from reassembly.checkpoints import restore_random_state
from reassembly.data import verify_manifest
from reassembly.precision import NumericalUpdateError, optimizer_update, NUMERICS_VERSION
from reassembly.prepare import _file_sha256
from reassembly.resources import seed_all, write_json, GIB
from reassembly.training import _autocast, _scaler, _optimizer, _loss_numbers, _check_cuda_memory

from . import ARCHITECTURE
from .checkpoints import load_checkpoint, save_checkpoint, require_resume_config
from .config import signature
from .data import RepairDataset, collate_samples
from .provenance import append_jsonl, provenance, code_inventory


def dataset_integrity(manifest, cfg):
    result = verify_manifest(manifest)
    expected = cfg["data"].get("expected_fingerprint")
    if expected and result["dataset_fingerprint"] != expected:
        raise ValueError("Prepared dataset differs from the configured VM fingerprint; no local substitution is allowed")
    return result


def _datasets(manifest, cfg, stage, purpose, query_cache):
    limit = 16 if purpose == "overfit" else None
    # A fixed SET of sixteen patterns, with fresh samples and poses each update.
    train = RepairDataset(manifest, "train", cfg, stage=stage, fixed=False, limit=limit,
                          fixed_geometry=purpose == "overfit", balanced_fields=stage == 2,
                          views=stage == 1, query_cache=query_cache)
    val = RepairDataset(manifest, "train" if purpose == "overfit" else "val", cfg,
                        stage=stage, fixed=True, limit=limit, balanced_fields=stage == 2,
                        views=stage == 1, query_cache=query_cache)
    if not len(train) or not len(val):
        raise ValueError("Training and validation geometry must be available")
    if purpose == "overfit" and len(train) != 16:
        raise ValueError("The overfit regression requires exactly 16 distinct patterns")
    return train, val


def preflight(cfg, manifest, output, device, *, query_cache=None, guard=None):
    from .model import RepairModel, configure_stage
    from .losses import compute_losses
    from .assembly import build_candidates, solve_candidates
    from .fields import ContinuousNeuralField
    from reassembly.validation import run_correctness_checks
    device = torch.device(device)
    integrity = dataset_integrity(manifest, cfg)
    correctness = run_correctness_checks()
    if not correctness["passed"]:
        raise RuntimeError("Geometry/solver correctness checks failed")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset = RepairDataset(manifest, "all", cfg, fixed=True, query_cache=query_cache)
    indices = [i for i, r in enumerate(dataset.records) if r["pieces"] == 3]
    if not indices:
        raise ValueError("Preflight must exercise an actual complete three-piece pattern")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh preflight directory")
    output.mkdir(parents=True, exist_ok=True)
    attempts, resolved = [], copy.deepcopy(cfg)
    effective_batch = cfg["train"]["batch_size"] * cfg["train"]["grad_accum_steps"]
    for size in dict.fromkeys([cfg["train"]["batch_size"], 1]):
        resolved["train"].update(batch_size=size, grad_accum_steps=effective_batch//size)
        attempt = {"batch_size": size, "stages": []}
        model = optimizer = scaler = batch = cache = field = None
        started = time.perf_counter()
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            batch = collate_samples([dataset[indices[i % len(indices)]] for i in range(size)], device)
            for stage in (1, 2):
                seed_all(cfg["train"]["seed"])
                model = RepairModel(resolved).to(device).train()
                configure_stage(model, stage)
                optimizer, scaler = _optimizer(model, resolved), _scaler(device.type == "cuda" and cfg["train"]["amp"])
                def backward():
                    totals = {}
                    for _ in range(resolved["train"]["grad_accum_steps"]):
                        with _autocast(device, scaler.is_enabled()):
                            losses = compute_losses(model, batch, stage, resolved)
                        if not torch.isfinite(losses["loss"]):
                            raise RuntimeError("Nonfinite preflight loss")
                        scaler.scale(losses["loss"] / resolved["train"]["grad_accum_steps"]).backward()
                        for k, v in _loss_numbers(losses).items():
                            totals[k] = totals.get(k, 0) + v/resolved["train"]["grad_accum_steps"]
                    return totals
                update = optimizer_update(model, optimizer, scaler, backward,
                    gradient_clip=cfg["train"]["gradient_clip"], context={"phase": "repair_preflight", "stage": stage, "update": 1})
                attempt["stages"].append({"stage": stage, **update})
                _check_cuda_memory(device, resolved)
            model.eval(); configure_stage(model, 3)
            one = collate_samples([dataset[indices[0]]], device)
            cache = build_candidates(model, one, resolved)
            field = ContinuousNeuralField(model, cache.encoded, chunk_size=resolved["solver"]["field_chunk"])
            values = field.sample(one["sdf_queries"][0].cpu().numpy())
            if not all(np.isfinite(value).all() for value in values):
                raise RuntimeError("Nonfinite direct field value or query gradient")
            solve_candidates(cache, resolved, field)
            _check_cuda_memory(device, resolved)
            attempt.update(passed=True, max_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None)
        except (RuntimeError, ValueError) as exc:
            attempt.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            is_memory = isinstance(exc, torch.cuda.OutOfMemoryError) or "memory cap" in str(exc).lower()
            if not is_memory:
                attempt["non_memory_failure"] = True
        finally:
            attempt["seconds"] = time.perf_counter()-started
            attempts.append(attempt)
            model = optimizer = scaler = batch = cache = field = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if attempt["passed"] or attempt.get("non_memory_failure"):
            break
    report = {"kind": "repair_preflight", "architecture": ARCHITECTURE,
        "status": ("passed" if device.type == "cuda" else "cpu_verified_cuda_unmeasured") if attempts[-1]["passed"] else "failed",
        "dataset_fingerprint": integrity["dataset_fingerprint"], "integrity": integrity, "correctness": correctness,
        "attempts": attempts, "effective_batch": effective_batch, "config_signature": signature(resolved),
        "query_cache_hash": dataset.query_cache_hash, "provenance": provenance(device), "device": str(device),
        "config": resolved, "fresh_allocation_models_only": True}
    write_json(output / "config.resolved.json", resolved, guard)
    write_json(output / "preflight.json", report, guard)
    return report


def require_preflight(path, cfg, fingerprint, device, cache_hash):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = "passed" if torch.device(device).type == "cuda" else "cpu_verified_cuda_unmeasured"
    if report.get("kind") != "repair_preflight" or report.get("status") != allowed:
        raise ValueError("Actual repair-model preflight has not passed on the requested device")
    if (report["dataset_fingerprint"], report["config_signature"], report["device"], report["query_cache_hash"]) != (fingerprint, signature(cfg), str(torch.device(device)), cache_hash):
        raise ValueError("Preflight does not cover this dataset/configuration/device/query cache")
    if report["provenance"]["code"]["sha256"] != code_inventory()["sha256"]:
        raise ValueError("Implementation changed after preflight; reprofile the actual implementation")


def verify_run_proofs(output):
    """A resumed/completed run remains bound to its original execution evidence."""
    metadata = json.loads((Path(output) / "run.json").read_text(encoding="utf-8"))
    if metadata["provenance"]["code"]["sha256"] != code_inventory()["sha256"]:
        raise ValueError("Implementation changed since this run; existing results cannot silently resume")
    for name in ("preflight", "field_diagnostic", "contact_gate"):
        path = metadata.get(name + "_path")
        digest = metadata.get(name + "_sha256")
        if digest is not None and (path is None or _file_sha256(path) != digest):
            raise ValueError(f"Original {name} proof changed or is missing")
    return metadata


def validate(model, dataset, stage, cfg, device):
    from .losses import compute_losses
    from .evaluation import sample_results, summarize
    totals, rows = {}, []
    model.eval()
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        for index in range(len(dataset)):
            torch.manual_seed(cfg["data"]["seed"] + index)
            sample = dataset[index]
            values = _loss_numbers(compute_losses(model, collate_samples([sample], device), stage, cfg))
            if not values or not all(math.isfinite(v) for v in values.values()):
                raise RuntimeError("Nonfinite validation metrics")
            for key, value in values.items():
                totals[key] = totals.get(key, 0) + value
            if stage == 1:
                row, _ = sample_results(model, sample, cfg, device, ("contact_only",), measure=False)["contact_only"]
                rows.append(row)
    result = {k: v / len(dataset) for k, v in totals.items()}
    result.update(pattern_count=len(dataset), source_count=len({r["source_id"] for r in dataset.records}))
    if rows:
        result["assembly_success"] = float(np.mean([r["success"] for r in rows]))
        result["assembly"] = summarize(rows)
    return result


def selection_score(metrics, stage):
    if stage == 1:
        return (-metrics["assembly_success"], -metrics["matching_top1_recall"], metrics["matching"])
    return (metrics["near_surface_sdf_l1"], metrics["sdf_l1"])


def train_stage(cfg, manifest, output, stage, device, *, preflight_report, field_diagnostic_report,
                purpose="experiment", initialize_from=None, resume=None, contact_gate_report=None,
                query_cache=None, guard=None):
    from .model import RepairModel, configure_stage
    from .losses import compute_losses
    from .evaluation import require_contact_gate
    if stage not in (1, 2):
        raise ValueError("Stage 3 is pose assembly and has no optimizer/training phase")
    if purpose not in ("experiment", "overfit"):
        raise ValueError("purpose must be experiment or overfit")
    if initialize_from and resume or stage == 1 and initialize_from:
        raise ValueError("Stage1 starts fresh; use explicit same-run resume only")
    if stage == 2 and not (initialize_from or resume):
        raise ValueError("Stage2 requires a gated stage1 checkpoint")
    if purpose == "overfit" and stage != 1:
        raise ValueError("The repaired overfit gate tests contact assembly, before field training")
    device, output = torch.device(device), Path(output).resolve()
    integrity = dataset_integrity(manifest, cfg)
    fingerprint = integrity["dataset_fingerprint"]
    if integrity["verified_sources"] < cfg["data"]["min_sources"]:
        raise ValueError("Prepared source yield is below the held-out experiment requirement")
    diagnostic = json.loads(Path(field_diagnostic_report).read_text(encoding="utf-8"))
    if (diagnostic.get("kind") != "focused_field_diagnostic" or diagnostic.get("status") != "complete"
            or diagnostic.get("dataset_fingerprint") != fingerprint or diagnostic.get("optimizer_updates") != 0
            or any(diagnostic.get(k, ["missing"]) for k in ("checkpoint_changes", "input_changes", "failed_jobs", "unrun_jobs"))
            or not diagnostic.get("checkpoint_verification")
            or not all(v.get("unchanged") and v.get("before") == v.get("after") for v in diagnostic["checkpoint_verification"].values())):
        raise ValueError("Complete the focused field diagnostic on this VM dataset before training")
    train, val = _datasets(manifest, cfg, stage, purpose, query_cache)
    require_preflight(preflight_report, cfg, fingerprint, device, train.query_cache_hash)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Choose a fresh repair run directory or explicitly resume this run")
    steps = cfg["train"]["overfit_updates" if purpose == "overfit" else "max_updates"]
    seed_all(cfg["train"]["seed"])
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    model = RepairModel(cfg).to(device)
    configure_stage(model, stage)
    optimizer, scaler = _optimizer(model, cfg), _scaler(device.type == "cuda" and cfg["train"]["amp"])
    run_id, start_step, best, lineage = str(uuid.uuid4()), 0, None, {}
    previous_metadata = None
    if initialize_from:
        parent = load_checkpoint(initialize_from, cfg=cfg, fingerprint=fingerprint, stage=1, purpose="experiment")
        require_resume_config(parent, cfg)
        completion = json.loads((Path(initialize_from).parent / "training_report.json").read_text(encoding="utf-8"))
        if completion.get("status") != "completed" or completion.get("run_id") != parent["run_id"]:
            raise ValueError("The parent contact run must have completed its declared comparison budget")
        if not contact_gate_report:
            raise ValueError("Stage2 requires --contact-gate-report")
        require_contact_gate(contact_gate_report, initialize_from, fingerprint)
        model.load_state_dict(parent["model"], strict=True)
        lineage = copy.deepcopy(parent["training_lineage"])
        lineage["parent_checkpoint"] = {"path": str(Path(initialize_from).resolve()), "sha256": _file_sha256(initialize_from)}
    if resume:
        if Path(resume).resolve().parent != output or not (output / "run.json").exists():
            raise ValueError("Resume must select a checkpoint in its existing run directory")
        state = load_checkpoint(resume, cfg=cfg, fingerprint=fingerprint, stage=stage, purpose=purpose)
        metadata = verify_run_proofs(output)
        previous_metadata = metadata
        if state["run_id"] != metadata["run_id"]:
            raise ValueError("Checkpoint belongs to a different run")
        require_resume_config(state, cfg)
        if state["query_cache_hash"] != train.query_cache_hash:
            raise ValueError("Supplemental field queries changed during resume")
        if metadata["field_diagnostic_sha256"] != _file_sha256(field_diagnostic_report):
            raise ValueError("Cannot replace the diagnostic proof while resuming")
        if metadata.get("contact_gate_path") and contact_gate_report is None:
            contact_gate_report = Path(metadata["contact_gate_path"])
        if metadata.get("contact_gate_sha256") != (_file_sha256(contact_gate_report) if contact_gate_report else None):
            raise ValueError("Cannot replace the contact acceptance proof while resuming")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        run_id, start_step, best = state["run_id"], state["step"], tuple(state["metrics"]["best_score"])
        lineage = state["training_lineage"]
        restore_random_state(state)
        if start_step >= steps:
            raise ValueError("Checkpoint already reached the requested update budget")
    output.mkdir(parents=True, exist_ok=True)
    lineage[str(stage)] = {"budget": steps, "purpose": purpose, "training_seed": cfg["train"]["seed"],
        "geometry_variant": cfg["repair"]["geometry_variant"], "view_supervision": cfg["repair"]["view_supervision"],
        "effective_batch": cfg["train"]["batch_size"] * cfg["train"]["grad_accum_steps"]}
    metadata = {"architecture": ARCHITECTURE, "run_id": run_id, "stage": stage, "purpose": purpose,
        "dataset_fingerprint": fingerprint, "query_cache_hash": train.query_cache_hash,
        "preflight_sha256": _file_sha256(preflight_report), "field_diagnostic_sha256": _file_sha256(field_diagnostic_report),
        "contact_gate_sha256": _file_sha256(contact_gate_report) if contact_gate_report else None,
        "preflight_path": str(Path(preflight_report).resolve()),
        "field_diagnostic_path": str(Path(field_diagnostic_report).resolve()),
        "contact_gate_path": str(Path(contact_gate_report).resolve()) if contact_gate_report else None,
        "provenance": provenance(device), "lineage": lineage,
        "pattern_ids": [r["pattern_id"] for r in train.records]}
    if previous_metadata:
        metadata["provenance"] = previous_metadata["provenance"]
        metadata["resume_history"] = previous_metadata.get("resume_history", []) + [{
            "from_step": start_step, "previous_preflight_sha256": previous_metadata["preflight_sha256"],
            "preflight_sha256": metadata["preflight_sha256"], "budget": steps}]
    write_json(output / "run.json", metadata, guard)
    write_json(output / "config.resolved.json", cfg, guard)
    write_json(output / "training_report.json", dict(metadata, kind="repair_training", status="running", updates=start_step), guard)
    started, completed, last_metrics = time.perf_counter(), start_step, {}
    try:
        for step in range(start_step, steps):
            train.set_step(step)
            model.train(); configure_stage(model, stage)
            def backward():
                totals, ids = {}, []
                for _ in range(cfg["train"]["grad_accum_steps"]):
                    selected = np.random.choice(len(train), cfg["train"]["batch_size"], replace=len(train) < cfg["train"]["batch_size"])
                    batch = collate_samples([train[int(i)] for i in selected], device)
                    ids.extend(batch["pattern_id"])
                    with _autocast(device, scaler.is_enabled()):
                        losses = compute_losses(model, batch, stage, cfg)
                    if not torch.isfinite(losses["loss"]):
                        raise NumericalUpdateError("nonfinite_loss", {"stage": stage, "update": step+1}, pattern_ids=ids)
                    scaler.scale(losses["loss"] / cfg["train"]["grad_accum_steps"]).backward()
                    for k, v in _loss_numbers(losses).items():
                        totals[k] = totals.get(k, 0) + v/cfg["train"]["grad_accum_steps"]
                return dict(totals, pattern_ids=ids)
            update = optimizer_update(model, optimizer, scaler, backward,
                gradient_clip=cfg["train"]["gradient_clip"], context={"stage": stage, "update": step+1},
                on_overflow=lambda event: append_jsonl(output / "numerics.jsonl", event, guard))
            completed = step + 1
            _check_cuda_memory(device, cfg)
            append_jsonl(output / "updates.jsonl", {"update": completed, **update}, guard)
            if completed % cfg["train"]["validation_interval"] == 0 or completed == steps:
                metrics = validate(model, val, stage, cfg, device)
                _check_cuda_memory(device, cfg)
                score = selection_score(metrics, stage)
                if not all(math.isfinite(v) for v in score):
                    raise RuntimeError("Nonfinite checkpoint selection score")
                improved = best is None or score < best
                best = score if improved else best
                last_metrics = {"train": update["metrics"], "validation": metrics, "best_score": list(best)}
                for name in (["best.pt", "latest.pt"] if improved else ["latest.pt"]):
                    save_checkpoint(output / name, model, optimizer, scaler, cfg, fingerprint, stage=stage,
                        step=completed, purpose=purpose, run_id=run_id, metrics=last_metrics,
                        lineage=lineage, query_cache_hash=train.query_cache_hash, guard=guard)
                elapsed = time.perf_counter()-started
                append_jsonl(output / "history.jsonl", {"update": completed, "seconds": elapsed,
                    "updates_per_second": (completed-start_step)/elapsed, **last_metrics}, guard)
                print(f"repair stage={stage} update={completed}/{steps} selection={score}", flush=True)
    except BaseException as error:
        failure = {"error": f"{type(error).__name__}: {error}", "updates": completed,
                   "diagnostics": getattr(error, "diagnostics", None)}
        write_json(output / "failure.json", failure, guard)
        write_json(output / "training_report.json", dict(metadata, kind="repair_training", status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", **failure), guard)
        raise
    elapsed = time.perf_counter()-started
    # A former best checkpoint can survive a budget extension. Update only its
    # declared budget/lineage; retain selected weights, optimizer, RNG and step.
    best_path = output / "best.pt"
    best_state = load_checkpoint(best_path)
    if best_state["cfg"]["train"] != cfg["train"]:
        import os
        best_state["cfg"] = copy.deepcopy(cfg)
        best_state["training_lineage"] = copy.deepcopy(lineage)
        if guard:
            guard.check(additional_bytes=best_path.stat().st_size + 4096)
        temporary = best_path.with_suffix(".pt.tmp")
        torch.save(best_state, temporary); os.replace(temporary, best_path)
    report = dict(metadata, kind="repair_training", status="completed", updates=completed, metrics=last_metrics,
        seconds=elapsed, updates_per_second=(completed-start_step)/elapsed,
        examples_per_second=(completed-start_step)*cfg["train"]["batch_size"]*cfg["train"]["grad_accum_steps"]/elapsed,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        best_checkpoint=str(output / "best.pt"), best_checkpoint_sha256=_file_sha256(output / "best.pt"),
        storage=guard.check() if guard else None, acceptance="not_implied_by_training_completion")
    write_json(output / "training_report.json", report, guard)
    return report
