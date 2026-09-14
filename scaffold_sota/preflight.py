"""Actual native forward/backward/reload checks, separated from learning gates."""
from __future__ import annotations

import time
from pathlib import Path

import torch

from .data import sanitize_input
from .io import RunLock, owned_output, study_root, write_json, jsonable
from .registry import source_info, materialize_source, build_backend
from .resources import ResourceGuard, directory_bytes
from .training import (make_dataset, make_prior, prior_tokens, seed_all, to_device, _loss,
                       working_directory, runtime_info, make_optimizer, validate_prediction_contract)


def preflight(root, model, manifest, cfg, *, source=None, device="cuda:0", condition="native",
              stage="assembly", prior=None, feature_checkpoint=None, options=None):
    root = study_root(root)
    manifest = Path(manifest).expanduser().resolve()
    prior = str(Path(prior).expanduser().resolve()) if prior else None
    feature_checkpoint = str(Path(feature_checkpoint).expanduser().resolve()) if feature_checkpoint else None
    guard = ResourceGuard(root, cfg["resources"], device)
    label = "%s_%s_%s_%s" % (model, stage, condition, time.time_ns())
    output = owned_output(root, root / "preflight" / label)
    report = {"experiment_id": "scaffold_sota", "model": model, "condition": condition,
              "stage": stage, "device": device, "status": "starting", "runtime": runtime_info(),
              "learning_acceptance": False, "cuda_verified": False}
    with RunLock(root):
        guard.require_idle_device()
        info = source_info(model, source)
        guard.check(additional_bytes=directory_bytes(info["path"]))
        output.mkdir(parents=True)
        try:
            dataset = make_dataset(manifest, "train", cfg, fixed=True)
            # Real largest-fragment-count examples, not a shortened tensor stand-in.
            indices = sorted(range(len(dataset)), key=lambda i: dataset.records[i]["pieces"], reverse=True)[:2]
            if not indices:
                raise ValueError("No preflight examples")
            native_source, _ = materialize_source(model, root, output, source)
            model_options = dict(cfg["model"])
            model_options.update(options or {})
            model_options.update(condition=condition, stage=stage, num_tokens=cfg["prior"]["query_count"])
            if feature_checkpoint:
                model_options["feature_checkpoint"] = feature_checkpoint
            with working_directory(native_source):
                seed_all(cfg["train"]["seed"])
                backend = build_backend(model, native_source, model_options, device)
                if getattr(backend, "requires_amp", False) and str(device).startswith("cuda") and not cfg["train"]["amp"]:
                    raise ValueError("This backend requires train.amp=true for scaled FP16 training")
                provider = make_prior(prior, cfg, dataset)
                optimizer = make_optimizer([p for p in backend.parameters() if p.requires_grad], cfg)
                amp = bool(cfg["train"]["amp"] and str(device).startswith("cuda"))
                scaler = torch.cuda.amp.GradScaler(enabled=amp)
                if str(device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(device)
                rows = []
                for index in indices:
                    sample = dataset[index]
                    tokens = prior_tokens(provider, sample, device)
                    backend.train()
                    optimizer.zero_grad(set_to_none=True)
                    started = time.monotonic()
                    with torch.cuda.amp.autocast(enabled=amp):
                        loss, _ = _loss(backend, to_device(sample, device), tokens)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in backend.parameters()):
                        raise RuntimeError("Non-finite backward gradients")
                    scaler.step(optimizer)
                    scaler.update()
                    prediction = None
                    if stage == "assembly":
                        backend.eval()
                        with torch.no_grad():
                            prediction = backend.predict(sanitize_input(sample, device=device), tokens, seed=4101)
                        validate_prediction_contract(sample, prediction)
                    rows.append({"pattern_id": sample["pattern_id"], "pieces": int(sample["fragment_mask"].sum()),
                                 "points_per_fragment": cfg["data"]["points_per_fragment"],
                                 "loss": float(loss.detach().cpu()), "prediction": jsonable(prediction),
                                 "seconds": time.monotonic() - started, "resources": guard.check()})
                # State reload catches adapter naming/device contracts without
                # changing any user checkpoint or labelling random poses accurate.
                state = {k: v.detach().clone() for k, v in backend.state_dict().items()}
                backend.load_state_dict(state, strict=True)
                report.update(status="execution_checks_passed", cuda_verified=str(device).startswith("cuda"),
                              examples=rows, source=info, options=model_options, resources=guard.check())
        except BaseException as exc:
            report.update(status="failed", error=str(exc))
            raise
        finally:
            write_json(output / "preflight.json", report)
    report["report_path"] = str(output / "preflight.json")
    return report
