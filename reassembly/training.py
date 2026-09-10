"""Bounded, reproducible stage training and real optimizer-step preflight."""
from __future__ import annotations

import copy
import json
import math
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .checkpoints import (finalize_checkpoint_budget, load_checkpoint, require_completed_run,
                          require_resume_config, restore_random_state, save_checkpoint)
from .resources import GIB, jsonable, seed_all, write_json


def collate_samples(samples: list[dict], device) -> dict:
    result = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values).to(device)
        elif isinstance(values[0], (str, dict, list)):
            result[key] = values
        elif isinstance(values[0], (int, float, np.number)):
            result[key] = torch.tensor(values, device=device)
        elif isinstance(values[0], np.ndarray):
            try:
                result[key] = torch.from_numpy(np.stack(values)).to(device)
            except ValueError:
                result[key] = values
    return result


def _autocast(device, enabled):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled and device.type == "cuda" else nullcontext()


def _scaler(enabled):
    # torch>=2.1 includes the CUDA namespace; unlike torch.amp.GradScaler,
    # this spelling also works on the oldest supported torch release.
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _optimizer(model, cfg):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=float(cfg["train"]["learning_rate"]),
                            weight_decay=float(cfg["train"]["weight_decay"]))


def _loss_numbers(losses):
    return {name: float(value.detach().float().cpu()) for name, value in losses.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1}


def _check_cuda_memory(device, cfg):
    if device.type == "cuda":
        peak = torch.cuda.max_memory_reserved(device)
        cap = float(cfg["resources"]["max_vram_gib"]) * GIB
        if peak >= cap:
            raise RuntimeError(f"Runtime CUDA memory cap exceeded: peak reserved {peak} bytes must be below {int(cap)}. Re-run preflight with batch size 1.")


def synthetic_batch(cfg, batch_size, device):
    """Allocation fixture, explicitly not a substitute for research training data."""
    n = cfg["data"]["points_per_fragment"]
    q = cfg["data"]["sdf_queries"]
    canonical = torch.rand(batch_size, 3, n, 3, device=device) * 0.4 - 0.2
    # Give all three pairs positive interface support so preflight exercises
    # positive matching as well as unmatched/background loss paths.
    canonical[:, 1] = canonical[:, 0] + 0.001
    canonical[:, 2] = canonical[:, 0] - 0.001
    labels = torch.zeros(batch_size, 3, n, device=device)
    labels[..., :n // 2] = 1
    interface_ids = torch.full((batch_size, 3, n), -1, device=device, dtype=torch.long)
    interface_ids[..., :n // 2] = 0
    queries = torch.rand(batch_size, q, 3, device=device) * 2 - 1
    return {
        "points": canonical - canonical.mean(-2, keepdim=True),
        "canonical_points": canonical,
        "fragment_mask": torch.ones(batch_size, 3, dtype=torch.bool, device=device),
        "anchor_index": torch.zeros(batch_size, dtype=torch.long, device=device),
        "fracture_labels": labels, "interface_ids": interface_ids,
        "sdf_queries": queries, "sdf_values": queries.norm(dim=-1) - 0.4,
        "sdf_near_mask": torch.arange(q, device=device)[None].expand(batch_size, -1) < q // 2,
        "rotations_gt": torch.eye(3, device=device)[None, None].repeat(batch_size, 3, 1, 1),
        "translations_gt": canonical.mean(-2), "target_points": canonical[:, 0],
    }


def preflight(cfg, device, *, sample_batch=None) -> tuple[dict, dict]:
    from .losses import compute_losses, configure_stage
    from .model import ReassemblyModel

    device = torch.device(device)
    resolved = copy.deepcopy(cfg)
    initial_batch = int(cfg["train"]["batch_size"])
    effective_batch = initial_batch * int(cfg["train"]["grad_accum_steps"])
    attempts = []
    for batch_size in dict.fromkeys([initial_batch, 1]):
        resolved["train"]["batch_size"] = batch_size
        resolved["train"]["grad_accum_steps"] = max(1, effective_batch // batch_size)
        model = optimizer = scaler = gradients = batch = losses = encoded = field = queries = lin = None
        start = time.perf_counter()
        attempt = {"batch_size": batch_size, "stages": [], "device": str(device)}
        try:
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA was requested but is unavailable; run on the A5000 host")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            for stage in (1, 2, 3):
                seed_all(cfg["train"]["seed"])
                model = ReassemblyModel(resolved).to(device)
                model.train()
                configure_stage(model, stage)
                optimizer = _optimizer(model, resolved)
                scaler = _scaler(device.type == "cuda" and cfg["train"]["amp"])
                batch = synthetic_batch(resolved, batch_size, device) if sample_batch is None else sample_batch(batch_size)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, scaler.is_enabled()):
                    losses = compute_losses(model, batch, stage, resolved)
                if not torch.isfinite(losses["loss"]):
                    raise RuntimeError(f"Non-finite stage-{stage} preflight loss")
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                    raise RuntimeError(f"Missing or non-finite stage-{stage} gradients")
                scaler.step(optimizer)
                scaler.update()
                # Include full inference field evaluation, chunked exactly as
                # the solver, with optimizer state still allocated.
                if stage == 3:
                    model.eval()
                    with torch.no_grad():
                        encoded = model.encode(batch["points"], batch["fragment_mask"], batch["anchor_index"])
                        res = resolved["solver"]["resolution"]
                        extent = resolved["solver"]["field_extent"]
                        lin = torch.linspace(-extent, extent, res, device=device)
                        queries = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
                        chunk = resolved["solver"]["field_chunk"]
                        for offset in range(0, len(queries), chunk):
                            field = model.scaffold(encoded, queries[offset:offset + chunk][None].expand(batch_size, -1, -1))
                attempt["stages"].append({"stage": stage, "losses": _loss_numbers(losses)})
                model = optimizer = scaler = gradients = batch = losses = encoded = field = queries = lin = None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                attempt["max_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                attempt["max_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                attempt["gpu_name"] = torch.cuda.get_device_name(device)
            else:
                attempt["max_reserved_bytes"] = None
            attempt["seconds"] = time.perf_counter() - start
            within_limit = device.type != "cuda" or attempt["max_reserved_bytes"] < cfg["resources"]["max_vram_gib"] * GIB
            attempt["passed"] = within_limit
            attempts.append(attempt)
            if within_limit:
                return {"status": "passed" if device.type == "cuda" else "cpu_verified_cuda_unmeasured",
                        "attempts": attempts, "fallback_applied": batch_size != initial_batch,
                        "effective_batch": effective_batch, "max_vram_gib": cfg["resources"]["max_vram_gib"]}, resolved
        except torch.cuda.OutOfMemoryError as exc:
            attempt.update({"passed": False, "error": str(exc), "seconds": time.perf_counter() - start})
            attempts.append(attempt)
        finally:
            model = optimizer = scaler = gradients = batch = losses = encoded = field = queries = lin = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {"status": "failed", "attempts": attempts, "reason": "Memory limit exceeded even at batch size 1"}, resolved


def _validate(model, dataset, stage, cfg, device):
    from .losses import compute_losses
    model.eval()
    totals = {}
    count = min(len(dataset), int(cfg["train"]["validation_samples"]))
    if count < 1:
        raise ValueError("Validation must evaluate at least one sample")
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        for idx in range(count):
            torch.manual_seed(int(cfg["data"]["seed"]) + idx)
            batch = collate_samples([dataset[idx]], device)
            losses = compute_losses(model, batch, stage, cfg)
            values = _loss_numbers(losses)
            if "loss" not in values or not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError(f"Non-finite or missing stage-{stage} validation metrics at sample {idx}")
            for key, value in values.items():
                totals[key] = totals.get(key, 0) + value
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_stage(cfg, manifest, run_dir, stage, device, *, initialize_from=None, resume=None,
                overfit=False, condition="predicted", guard=None):
    from .data import FractureDataset
    from .losses import compute_losses, configure_stage
    from .model import ReassemblyModel

    if stage not in (1, 2, 3):
        raise ValueError("stage must be 1, 2 or 3")
    if initialize_from and resume:
        raise ValueError("Use initialize_from for a new stage or resume for the same stage, not both")
    if stage > 1 and not (initialize_from or resume):
        raise ValueError("Specify the previous v2 stage checkpoint explicitly with --initialize-from")
    if stage == 1 and initialize_from:
        raise ValueError("Stage 1 must start from fresh random weights; only explicit same-run resume is allowed")
    if condition not in ("predicted", "contact_only"):
        raise ValueError("Training condition must be predicted or contact_only")
    device = torch.device(device)
    cfg = copy.deepcopy(cfg)
    cfg["train"]["condition"] = condition
    # Lineage comes only from an explicitly verified checkpoint, never from
    # caller-provided metadata or an old checkpoint search.
    cfg["run_lineage"] = {}
    purpose = "overfit" if overfit else "pilot"
    steps = int(cfg["train"]["overfit_updates"] if overfit else cfg["train"]["max_updates"])
    if not 1 <= steps <= 2000:
        raise ValueError("Pilot and overfit stages require 1..2000 updates; full training is not launched automatically")
    if overfit and int(cfg["train"]["overfit_patterns"]) != 16:
        raise ValueError("The fixed overfit acceptance check requires exactly 16 patterns")
    path = Path(manifest)
    document = json.loads(path.read_text(encoding="utf-8"))
    source_count = len({p["source_id"] for p in document["patterns"]})
    if not overfit and source_count < cfg["data"]["min_sources"]:
        raise ValueError(f"Learning pilot requires {cfg['data']['min_sources']} accepted source objects; found {source_count}. Inspect preparation yield first.")
    fingerprint = document["fingerprint"]
    run_dir = Path(run_dir).resolve()
    run_id = str(uuid.uuid4())
    previous_metadata = None
    if resume:
        resume_path = Path(resume).resolve()
        metadata_path = run_dir / "run.json"
        if (resume_path.parent != run_dir or resume_path.name not in ("latest.pt", "best.pt")
                or not metadata_path.is_file()):
            raise ValueError("Resume requires this existing run directory and its explicit latest.pt or best.pt")
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run_id = previous_metadata.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Run metadata has no v2 run identity; start a fresh run")
        expected = {"stage": stage, "purpose": purpose, "condition": condition, "dataset_fingerprint": fingerprint}
        if any(previous_metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Resume run metadata differs in stage, purpose, condition, or dataset")
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(f"Use a fresh run directory or explicitly resume: {run_dir}")
    if guard:
        guard.check()
    seed_all(cfg["train"]["seed"])
    limit = cfg["train"]["overfit_patterns"] if overfit else None
    dataset = FractureDataset(path, "train", cfg, stage=stage, fixed=overfit, limit=limit)
    validation = dataset if overfit else FractureDataset(path, "val", cfg, stage=stage, fixed=True)
    if overfit and len(dataset) != 16:
        raise ValueError(f"The fixed overfit check requires 16 distinct prepared patterns; found {len(dataset)}")
    if len(dataset) == 0 or len(validation) == 0:
        raise ValueError("Training and validation sets must both be nonempty")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; run on the A5000 host")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = ReassemblyModel(cfg).to(device)
    configure_stage(model, stage)
    optimizer = _optimizer(model, cfg)
    scaler = _scaler(device.type == "cuda" and cfg["train"]["amp"])
    initial_step, best = 0, float("inf")
    checkpoint = None
    if initialize_from:
        checkpoint = load_checkpoint(initialize_from, cfg=cfg, dataset_fingerprint=fingerprint, stage=stage - 1, purpose=purpose)
        require_completed_run(initialize_from, checkpoint)
        model.load_state_dict(checkpoint["model"], strict=True)
        cfg["run_lineage"] = copy.deepcopy(checkpoint.get("training_lineage", {}))
    if resume:
        checkpoint = load_checkpoint(resume, cfg=cfg, dataset_fingerprint=fingerprint, stage=stage, purpose=purpose, run_id=run_id)
        require_resume_config(checkpoint, cfg)
        cfg["run_lineage"] = copy.deepcopy(checkpoint.get("training_lineage", {}))
        if checkpoint["cfg"]["train"].get("condition", "predicted") != condition:
            raise ValueError("Cannot change training condition when resuming")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        initial_step = checkpoint["step"]
        if initial_step >= steps:
            raise ValueError(f"Checkpoint already reached update {initial_step}; resume budget must be larger and remain at most 2000")
        best = checkpoint.get("metrics", {}).get("best_validation", best)
        if not isinstance(best, (int, float)) or not math.isfinite(best):
            raise ValueError("Resume checkpoint lacks a finite best-validation score")
        restore_random_state(checkpoint)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.resolved.json", cfg, guard)
    write_json(run_dir / "run.json", {"schema_version": 2, "run_id": run_id, "stage": stage, "purpose": purpose,
        "condition": condition, "dataset_fingerprint": fingerprint, "manifest": str(path.resolve()),
        "device": str(device), "torch": torch.__version__, "source_count": source_count,
        "initialize_from": str(initialize_from) if initialize_from else (previous_metadata or {}).get("initialize_from"),
        "resume": str(resume) if resume else None}, guard)
    write_json(run_dir / "training_report.json", {"kind": "training", "schema_version": 2,
        "status": "running", "run_id": run_id, "stage": stage, "purpose": purpose,
        "dataset_fingerprint": fingerprint, "updates": initial_step, "planned_updates": steps}, guard)
    accum = int(cfg["train"]["grad_accum_steps"])
    bs = int(cfg["train"]["batch_size"])
    start = time.perf_counter()
    last_metrics = {}
    for step in range(initial_step, steps):
        dataset.set_step(step)
        model.train()
        configure_stage(model, stage)
        optimizer.zero_grad(set_to_none=True)
        train_values = {}
        for _ in range(accum):
            indices = np.random.choice(len(dataset), bs, replace=len(dataset) < bs)
            batch = collate_samples([dataset[int(idx)] for idx in indices], device)
            with _autocast(device, scaler.is_enabled()):
                losses = compute_losses(model, batch, stage, cfg)
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError(f"Non-finite stage {stage} loss at update {step + 1}")
            scaler.scale(losses["loss"] / accum).backward()
            for key, value in _loss_numbers(losses).items():
                train_values[key] = train_values.get(key, 0) + value / accum
        scaler.unscale_(optimizer)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError(f"Missing/non-finite gradients at stage {stage}, update {step + 1}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        gradients = None
        _check_cuda_memory(device, cfg)
        if (step + 1) % cfg["train"]["validation_interval"] == 0 or step + 1 == steps:
            metrics = _validate(model, validation, stage, cfg, device)
            _check_cuda_memory(device, cfg)
            score = metrics.get("sdf_l1", metrics["loss"]) if stage == 2 else metrics["loss"]
            if not math.isfinite(score):
                raise RuntimeError("Cannot checkpoint a non-finite validation score")
            improved = score < best
            best = min(best, score)
            last_metrics = {"train": train_values, "validation": metrics, "best_validation": best}
            elapsed = time.perf_counter() - start
            record = {"update": step + 1, "seconds": elapsed,
                      "updates_per_second": (step + 1 - initial_step) / max(elapsed, 1e-6), **last_metrics}
            if guard:
                guard.check()
            with open(run_dir / "history.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(jsonable(record), allow_nan=False) + "\n")
            for name in (["latest.pt", "best.pt"] if improved else ["latest.pt"]):
                save_checkpoint(run_dir / name, model, optimizer, cfg, fingerprint, stage, step + 1,
                                purpose=purpose, metrics=last_metrics, scaler=scaler, guard=guard, run_id=run_id)
            print(f"stage={stage} update={step+1}/{steps} train={train_values['loss']:.5f} val={score:.5f}", flush=True)
    finalize_checkpoint_budget(run_dir / "best.pt", cfg, stage, purpose, run_id, guard)
    report = {"status": "completed", "run_id": run_id, "stage": stage, "purpose": purpose, "condition": condition,
              "updates": steps, "seconds": time.perf_counter() - start, "metrics": last_metrics,
              "kind": "training", "schema_version": 2, "dataset_fingerprint": fingerprint,
              "updates_this_invocation": steps - initial_step,
              "best_checkpoint": str((run_dir / "best.pt").resolve()),
              "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
              "long_training_started": False}
    report["updates_per_second"] = (steps-initial_step) / max(report["seconds"], 1e-6)
    report["examples_per_second"] = report["updates_per_second"] * bs * accum
    write_json(run_dir / "training_report.json", report, guard)
    return report
