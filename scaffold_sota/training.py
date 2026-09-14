"""Study-owned native-model training, prediction and checkpoint lineage."""
from __future__ import annotations

import contextlib
import inspect
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import EXPERIMENT_ID, SCHEMA_VERSION
from .config import load_config
from .io import (PACKAGE, REPO, append_jsonl, digest_json, jsonable, owned_output,
                 read_json, sha256_file, study_root, write_json, RunLock)
from .registry import build_backend, materialize_source, source_info
from .resources import ResourceGuard, directory_bytes


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def code_hash():
    files = {str(path.relative_to(REPO)): sha256_file(path)
             for path in sorted(PACKAGE.rglob("*.py")) if "__pycache__" not in path.parts}
    for relative in ("reassembly/__init__.py", "reassembly/model.py", "reassembly/geometry.py",
                     "reassembly/prepare.py", "reassembly/repair/__init__.py", "reassembly/repair/model.py"):
        if (REPO / relative).is_file():
            files[relative] = sha256_file(REPO / relative)
    return digest_json(files)


def runtime_info():
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
            "host": platform.node()}


def torch_load(path):
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    return torch.load(Path(path), **options)


def load_study_checkpoint(path):
    state = torch_load(path)
    if not isinstance(state, dict) or state.get("experiment_id") != EXPERIMENT_ID or state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Resume/init accepts only independent scaffold_sota checkpoints")
    required = {"model", "optimizer", "step", "identity", "identity_hash", "rng", "options", "cfg"}
    if not required.issubset(state) or digest_json(state["identity"]) != state["identity_hash"]:
        raise ValueError("Incomplete or inconsistent study checkpoint")
    return state


def save_checkpoint(path, backend, optimizer, scaler, step, identity, cfg, options, score, guard):
    parameter_bytes = sum(x.numel() * x.element_size() for x in backend.state_dict().values())
    guard.check(additional_bytes=parameter_bytes * 5 + 1024 ** 2)
    state = {"experiment_id": EXPERIMENT_ID, "schema_version": SCHEMA_VERSION,
             "model_name": identity["model"], "stage": identity["stage"],
             "condition": identity["condition"], "step": int(step),
             "identity": identity, "identity_hash": digest_json(identity),
             "model": backend.state_dict(), "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict(), "cfg": cfg, "options": options, "best_score": score,
             "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                     "torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(str(temporary), str(path))


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def to_device(sample, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in sample.items()}


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(previous))


def make_dataset(manifest, split, cfg, fixed=None, seed=None, limit=None):
    from .data import StudyDataset
    return StudyDataset(manifest, split,
                        points_per_fragment=cfg["data"]["points_per_fragment"],
                        sdf_queries=cfg["data"]["sdf_queries"],
                        seed=cfg["data"]["seed"] if seed is None else seed,
                        fixed=fixed, limit=limit, verify=cfg["data"]["verify"],
                        hard_noise_std=cfg["data"]["hard_noise_std"])


def make_prior(path, cfg, dataset, device="cpu"):
    if path is None:
        return None
    from .priors import FrozenPrior, CachedPrior
    if Path(path).is_dir():
        return CachedPrior(path, device=device, expected_fingerprint=dataset.fingerprint)
    return FrozenPrior(path, device=device, query_count=cfg["prior"]["query_count"],
                       extent=cfg["prior"]["extent"], expected_fingerprint=dataset.fingerprint)


def prior_hash(path):
    if path is None:
        return None
    path = Path(path)
    return sha256_file(path / "cache.json" if path.is_dir() else path)


def prior_tokens(provider, sample, device):
    if provider is None:
        return None
    return to_device(provider.sample_tokens(sample), device)


def initialize_native(backend, checkpoint, identity):
    state = load_study_checkpoint(checkpoint)
    previous = state["identity"]
    for key in ("model", "stage", "manifest_hash", "dataset_fingerprint", "source_revision", "seed",
                "registry_hash", "code_hash", "feature_hash", "cfg_hash", "native_options_hash", "cpu_smoke"):
        if key not in previous or key not in identity or previous[key] != identity[key]:
            raise ValueError("Native initialization differs on " + key)
    if previous["condition"] != "native":
        raise ValueError("B0-B4 must branch from the native baseline, not another adapted condition")
    if hasattr(backend, "initialize_from_native"):
        backend.initialize_from_native(state["model"])
    else:
        incompatible = backend.load_state_dict(state["model"], strict=False)
        prefixes = getattr(backend, "conditioning_parameter_prefixes", ())
        bad = [key for key in incompatible.missing_keys if not any(key.startswith(prefix) for prefix in prefixes)]
        if incompatible.unexpected_keys or bad:
            raise ValueError("Native checkpoint does not match backend: " + str(incompatible))


def _loss(backend, sample, tokens):
    result = backend.loss(sample, tokens)
    loss = result["loss"] if isinstance(result, dict) else result
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not torch.isfinite(loss):
        raise RuntimeError("Native loss is missing, non-scalar or non-finite")
    return loss, result


def validate_prediction_contract(sample, prediction, allow_structured_failure=False):
    """Check operational pose validity without any ground-truth accuracy gate."""
    from .geometry import as_numpy, validate_poses
    if not isinstance(prediction, dict) or prediction.get("rotations") is None or prediction.get("translations") is None:
        if (allow_structured_failure and isinstance(prediction, dict)
                and prediction.get("rotations") is None and prediction.get("translations") is None
                and prediction.get("status") not in (None, "ok", "success")
                and (prediction.get("reason") or prediction.get("error"))):
            return False  # An untrained matcher may legitimately have no solve.
        raise RuntimeError("Native preflight returned no poses: " + str(prediction))
    mask = as_numpy(sample["fragment_mask"]).astype(bool)
    rotations, translations = as_numpy(prediction["rotations"]), as_numpy(prediction["translations"])
    if len(rotations) == len(mask):
        rotations, translations = rotations[mask], translations[mask]
    validate_poses(rotations, translations, int(mask.sum()))
    return True


def make_optimizer(parameters, cfg):
    return torch.optim.AdamW(parameters, lr=cfg["train"]["learning_rate"],
                             weight_decay=cfg["train"]["weight_decay"],
                             betas=tuple(cfg["train"].get("betas", [.9, .999])),
                             eps=cfg["train"].get("eps", 1e-8))


def validation(backend, dataset, provider, device, cfg, output=None, observation_seed=4101, stage="assembly", guard=None, candidates=1):
    from .data import sanitize_input
    from .evaluation import evaluate_prediction, aggregate
    if not isinstance(candidates, int) or not 1 <= candidates <= 64:
        raise ValueError("Candidate bank must contain 1 to 64 fixed draws")
    backend.eval()
    rows, losses = [], []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            tokens = prior_tokens(provider, sample, device)
            if stage == "pretrain":
                loss, _ = _loss(backend, to_device(sample, device), tokens)
                losses.append(float(loss.cpu()))
            else:
                bank, bank_rows, hashes = [], [], set()
                for draw in range(candidates):
                    draw_seed = observation_seed + index + draw * 1000003
                    seed_all(draw_seed)
                    try:
                        tick = time.monotonic()
                        prediction = backend.predict(sanitize_input(sample, device=device), tokens, seed=draw_seed)
                        prediction.setdefault("runtime_seconds", time.monotonic() - tick)
                    except Exception as exc:
                        if "out of memory" in str(exc).lower() or "device-side assert" in str(exc).lower():
                            raise
                        prediction = {"status": "native_prediction_error", "reason": str(exc),
                                      "rotations": None, "translations": None}
                    bank.append(prediction)
                    candidate_row = evaluate_prediction(sample, prediction, threshold=cfg["evaluation"]["threshold"])
                    bank_rows.append(candidate_row)
                    if not candidate_row["failed"]:
                        from .geometry import as_numpy, reference_gauge
                        mask = as_numpy(sample["fragment_mask"]).astype(bool)
                        r, t = as_numpy(prediction["rotations"]), as_numpy(prediction["translations"])
                        if len(r) == len(mask):
                            r, t = r[mask], t[mask]
                        anchor = list(np.flatnonzero(mask)).index(int(sample["anchor_index"]))
                        r, t = reference_gauge(r, t, anchor)
                        hashes.add(digest_json([r.round(5).tolist(), t.round(5).tolist()]))
                prediction = bank[0]  # Fixed native selection; GT never selects inference output.
                row = evaluate_prediction(sample, prediction, threshold=cfg["evaluation"]["threshold"])
                row["prediction"] = jsonable(prediction)
                row["points_per_fragment"] = int(sample["points"].shape[1])
                if candidates > 1:
                    row.update(candidates=jsonable(bank), candidate_count=candidates,
                               unique_valid_candidates=len(hashes), native_selection_index=0,
                               candidate_recall_at_k_diagnostic=any(x["success"] for x in bank_rows),
                               candidate_recall_note="GT scoring only; never used to select poses",
                               candidate_seeds=[observation_seed + index + k * 1000003 for k in range(candidates)])
                rows.append(row)
                if output:
                    append_jsonl(output, row)
            if guard and (index % 16 == 0):
                guard.check()
    if stage == "pretrain":
        if not losses:
            raise ValueError("No pretraining validation samples")
        return {"count": len(losses), "loss": float(np.mean(losses))}, -float(np.mean(losses))
    if not rows:
        raise ValueError("No validation observations")
    summary = aggregate(rows)
    successes = sum(bool(row.get("success")) for row in rows)
    # Source-macro selection, independent of number of cuts per source.
    source_rates = {}
    for row in rows:
        source_rates.setdefault(row["source_id"], []).append(bool(row.get("success")))
    score = float(np.mean([np.mean(x) for x in source_rates.values()]))
    summary.update(source_macro_success=score, successes=successes)
    if candidates > 1:
        recall_sources = {}
        for row in rows:
            recall_sources.setdefault(row["source_id"], []).append(row["candidate_recall_at_k_diagnostic"])
        summary.update(candidate_count=candidates, candidate_recall_at_k_diagnostic=float(np.mean([np.mean(x) for x in recall_sources.values()])),
                       degenerate_bank_count=sum(row["unique_valid_candidates"] < 2 for row in rows),
                       candidate_selection="fixed_first_draw_input_only")
    return summary, score


def train(*, root, model, manifest, cfg, condition="native", stage="assembly", source=None,
          prior=None, prior_device="cpu", feature_checkpoint=None, initialize=None,
          resume=False, device="cuda:0", run_id=None, model_options=None, cpu_smoke=False):
    root = study_root(root)
    manifest = str(Path(manifest).expanduser().resolve())
    prior = str(Path(prior).expanduser().resolve()) if prior else None
    initialize = str(Path(initialize).expanduser().resolve()) if initialize else None
    feature_checkpoint = str(Path(feature_checkpoint).expanduser().resolve()) if feature_checkpoint else None
    if device == "cpu" and not cpu_smoke:
        raise ValueError("Full native training requires CUDA; --cpu-smoke is explicitly diagnostic only")
    if cpu_smoke and cfg["train"]["updates"] > 3:
        raise ValueError("CPU smoke is limited to three optimizer updates")
    if stage == "pretrain" and condition != "native":
        raise ValueError("Prerequisite pretraining uses native condition only")
    if condition in ("B2", "B3", "B4") and not prior:
        raise ValueError("This learned condition requires an explicit frozen prior")
    if condition != "native" and not initialize:
        raise ValueError("Matched adaptation requires --initialize pointing to its native baseline")
    if condition == "native" and initialize:
        raise ValueError("Native baseline starts fresh; use explicit prerequisite feature checkpoint only")
    if prior and Path(prior).is_dir():
        raise ValueError("Static token caches are evaluation-only; training queries the frozen field on fresh observations")
    seed = int(cfg["train"]["seed"])
    cfg = dict(cfg)
    run_id = run_id or "%s_%s_%s_seed%d" % (model, stage, condition, seed)
    if not run_id or any(x not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for x in run_id):
        raise ValueError("Run ID must be a simple alphanumeric name")
    run = owned_output(root, root / "runs" / run_id)
    for input_path in (manifest, prior, feature_checkpoint, initialize):
        if input_path and not Path(input_path).expanduser().is_file():
            raise FileNotFoundError(str(input_path))
    guard = ResourceGuard(root, cfg["resources"], device)
    with RunLock(root):
        guard.check()
        guard.require_idle_device()
        if run.exists() and not resume:
            raise ValueError("Run already exists; use --resume for this exact run or a new run ID")
        if resume and not (run / "latest.pt").exists():
            raise ValueError("Resume requires this study run's latest.pt")
        info = source_info(model, source)
        guard.check(additional_bytes=directory_bytes(info["path"]))
        dataset = make_dataset(manifest, "train", cfg, fixed=cfg["data"].get("fixed_training", False),
                               limit=cfg["data"].get("train_limit"))
        val_seed = cfg["evaluation"]["observation_seeds"][0]
        val_dataset = make_dataset(manifest, "val", cfg, fixed=True, seed=val_seed)
        if not len(dataset) or not len(val_dataset):
            raise ValueError("Training and validation splits must both be nonempty")
        options = dict(cfg["model"])
        options.update(model_options or {})
        options.update(condition=condition, stage=stage, num_tokens=cfg["prior"]["query_count"])
        if feature_checkpoint:
            options["feature_checkpoint"] = str(Path(feature_checkpoint).expanduser().resolve())
        identity = {"experiment_id": EXPERIMENT_ID, "run_id": run_id, "model": model,
                    "stage": stage, "condition": condition, "seed": seed,
                    "manifest_hash": sha256_file(manifest), "dataset_fingerprint": dataset.fingerprint,
                    "source_revision": info["revision"], "registry_hash": info["registry_sha256"],
                    "code_hash": code_hash(), "cfg_hash": digest_json(cfg), "options_hash": digest_json(options),
                    "native_options_hash": digest_json({k: v for k, v in options.items() if k != "condition"}),
                    "prior_hash": sha256_file(prior) if prior else None,
                    "feature_hash": sha256_file(feature_checkpoint) if feature_checkpoint else None,
                    "native_parent_hash": sha256_file(initialize) if initialize else None,
                    "cpu_smoke": bool(cpu_smoke)}
        state = None
        if resume:
            state = load_study_checkpoint(run / "latest.pt")
            if state["identity"] != identity:
                raise ValueError("Resume identity mismatch: code/config/data/source/prior/parent must be unchanged")
        run.mkdir(parents=True, exist_ok=True)
        native_source, _ = materialize_source(model, root, run, source)
        write_json(run / "config.json", cfg)
        record = {"identity": identity, "manifest_path": str(Path(manifest).resolve()),
                  "prior_path": str(Path(prior).resolve()) if prior else None,
                  "runtime": runtime_info(), "options": options, "status": "starting"}
        write_json(run / "run.json", record)
        try:
            seed_all(seed)
            with working_directory(native_source):
                backend = build_backend(model, native_source, options, device)
                if getattr(backend, "requires_amp", False) and str(device).startswith("cuda") and not cfg["train"]["amp"]:
                    raise ValueError("This native backend uses FP16 internally; set train.amp=true for scaled gradients")
                provider = make_prior(prior, cfg, dataset, prior_device)
                if condition in ("B2", "B3", "B4") and provider.provenance.get("architecture") != "fragment-assembly-repair-v3":
                    raise ValueError("The planned learned-input arms require v3; evaluate v2 as an explicitly labelled frozen control")
                if initialize:
                    initialize_native(backend, initialize, identity)
                parameters = [p for p in backend.parameters() if p.requires_grad]
                if not parameters:
                    raise ValueError("No trainable native parameters")
                amp = bool(cfg["train"]["amp"] and str(device).startswith("cuda"))
                optimizer = make_optimizer(parameters, cfg)
                scaler = torch.cuda.amp.GradScaler(enabled=amp)
                if not resume:
                    from .data import sanitize_input
                    # Real largest-piece forward/backward/prediction gate, before
                    # committing optimizer updates. Restore BN/state and RNG so
                    # preflight does not change matched training initialization.
                    warm_state = {k: v.detach().cpu().clone() for k, v in backend.state_dict().items()}
                    warm_index = max(range(len(dataset)), key=lambda i: dataset.records[i]["pieces"])
                    warm_sample = dataset[warm_index]
                    warm_tokens = prior_tokens(provider, warm_sample, device)
                    backend.train()
                    with torch.cuda.amp.autocast(enabled=amp):
                        warm_loss, _ = _loss(backend, to_device(warm_sample, device), warm_tokens)
                    scaler.scale(warm_loss).backward()
                    scaler.unscale_(optimizer)
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                        raise RuntimeError("Preflight produced nonfinite gradients")
                    if stage == "assembly":
                        backend.eval()
                        with torch.no_grad():
                            warm_prediction = backend.predict(sanitize_input(warm_sample, device=device), warm_tokens, seed=4101)
                        validate_prediction_contract(warm_sample, warm_prediction, allow_structured_failure=True)
                    else:
                        warm_prediction = None
                    write_json(run / "preflight.json", {"status": "execution_checks_passed",
                        "identity_hash": digest_json(identity), "cuda_verified": str(device).startswith("cuda"),
                        "learning_acceptance": False, "pieces": int(warm_sample["fragment_mask"].sum()),
                        "points_per_fragment": cfg["data"]["points_per_fragment"],
                        "prediction_status": warm_prediction.get("status") if warm_prediction else None,
                        "resources": guard.check()})
                    backend.load_state_dict(warm_state, strict=True)
                    backend.zero_grad(set_to_none=True)
                    del warm_state, warm_loss, warm_sample, warm_tokens, warm_prediction
                # Reset scaler state after the diagnostic unscale; no diagnostic
                # optimizer update or momentum is carried into training.
                scaler = torch.cuda.amp.GradScaler(enabled=amp)
                start, best = 0, None
                if state:
                    backend.load_state_dict(state["model"], strict=True)
                    optimizer.load_state_dict(state["optimizer"])
                    scaler.load_state_dict(state["scaler"])
                    restore_rng(state["rng"])
                    start, best = state["step"], state["best_score"]
                    if (run / "best.pt").exists():
                        saved_best = load_study_checkpoint(run / "best.pt")
                        if saved_best["identity"] != identity:
                            raise ValueError("Best checkpoint belongs to a different run")
                        if saved_best["best_score"] is not None:
                            best = max(saved_best["best_score"], best) if best is not None else saved_best["best_score"]
                else:
                    seed_all(seed)  # condition initialization must not shift training RNG
                if str(device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(device)
                record.update(status="running", trainable_parameters=sum(p.numel() for p in parameters))
                record["native_configuration"] = getattr(backend, "native_config", {})
                write_json(run / "run.json", record)
                updates = cfg["train"]["updates"]
                for step in range(start, updates):
                    tick = time.monotonic()
                    backend.train()
                    optimizer.zero_grad(set_to_none=True)
                    accumulation = cfg["train"]["grad_accum_steps"]
                    total_loss = 0.
                    dataset.set_step(step)
                    for micro in range(accumulation):
                        sample = dataset[(step * accumulation + micro) % len(dataset)]
                        tokens = prior_tokens(provider, sample, device)
                        with torch.cuda.amp.autocast(enabled=amp):
                            loss, _ = _loss(backend, to_device(sample, device), tokens)
                        total_loss += float(loss.detach().cpu()) / accumulation
                        scaler.scale(loss / accumulation).backward()
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["train"]["gradient_clip"])
                    if not torch.isfinite(norm):
                        raise RuntimeError("Non-finite native gradients; no optimizer update applied")
                    scaler.step(optimizer)
                    scaler.update()
                    completed = step + 1
                    if completed == 1 or completed % cfg["train"]["log_interval"] == 0:
                        resources = guard.check()
                        append_jsonl(run / "history.jsonl", {"step": completed, "loss": total_loss,
                                     "seconds": time.monotonic() - tick, "resources": resources})
                        print("%s step %d/%d loss=%.6g" % (run_id, completed, updates, total_loss), flush=True)
                    if completed % cfg["train"]["validation_interval"] == 0 or completed == updates:
                        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                               "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
                        summary, score = validation(backend, val_dataset, provider, device, cfg, stage=stage, guard=guard)
                        restore_rng(rng)
                        append_jsonl(run / "validation.jsonl", {"step": completed, "metrics": summary, "selection_score": score})
                        if best is None or score > best:
                            best = score
                            save_checkpoint(run / "best.pt", backend, optimizer, scaler, completed, identity, cfg, options, best, guard)
                    if completed % cfg["train"]["checkpoint_interval"] == 0 or completed == updates:
                        save_checkpoint(run / "latest.pt", backend, optimizer, scaler, completed, identity, cfg, options, best, guard)
                    write_json(run / "progress.json", {"step": completed, "updates": updates, "status": "running"})
                record.update(status="completed", updates=updates, best_validation_score=best,
                              resources=guard.check(), completion_is_learning_acceptance=False,
                              checkpoint_hashes={name: sha256_file(run / name) for name in ("best.pt", "latest.pt")})
                write_json(run / "run.json", record)
                write_json(run / "progress.json", {"step": updates, "updates": updates, "status": "completed"})
        except BaseException as exc:
            record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            write_json(run / "run.json", record)
            raise
    return record


def evaluate(*, root, checkpoint, manifest, output, source=None, prior=None, prior_device="cpu",
             device="cuda:0", split="val", observation_seed=4101, limit=None, candidates=1,
             prior_control="predicted", control_seed=42, diagnostic_only=False):
    root = study_root(root)
    checkpoint = str(Path(checkpoint).expanduser().resolve())
    manifest = str(Path(manifest).expanduser().resolve())
    prior = str(Path(prior).expanduser().resolve()) if prior else None
    output = owned_output(root, output)
    if output.exists():
        raise ValueError("Evaluation output already exists; choose a fresh output path")
    state = load_study_checkpoint(checkpoint)
    cfg, identity = state["cfg"], state["identity"]
    if sha256_file(manifest) != identity["manifest_hash"]:
        raise ValueError("Evaluation manifest differs from the recorded native training corpus")
    if code_hash() != identity["code_hash"]:
        raise ValueError("Study code changed since training; use the recorded revision")
    if identity["stage"] != "assembly":
        raise ValueError("Pose evaluation requires an assembly checkpoint")
    if identity["condition"] in ("B2", "B3", "B4") and not prior:
        raise ValueError("Explicit prior required; prior substitutions are separately labelled")
    if prior_control != "predicted" and not prior:
        raise ValueError("Content controls require a prior")
    if prior_control == "wrong" and split == "train":
        raise ValueError("Wrong-source control uses training donors and requires a held-out recipient split")
    run = Path(checkpoint).resolve().parent
    if not (run / "source.json").exists():
        raise ValueError("Checkpoint's immutable source manifest is missing")
    guard = ResourceGuard(root, cfg["resources"], device)
    with RunLock(root):
        guard.require_idle_device()
        guard.check()
        dataset = make_dataset(manifest, split, cfg, fixed=True, seed=observation_seed, limit=limit)
        info = source_info(identity["model"], source)
        if info["revision"] != identity["source_revision"]:
            raise ValueError("Native source revision changed")
        # Evaluation uses a fresh owned copy; external/native default outputs cannot
        # mutate the training source copy or any original pipeline directory.
        work = output.parent / (output.stem + "_work")
        native_source, _ = materialize_source(identity["model"], root, work, source)
        with working_directory(native_source):
            backend = build_backend(identity["model"], native_source, state["options"], device)
            backend.load_state_dict(state["model"], strict=True)
            provider = make_prior(prior, cfg, dataset, prior_device)
            if prior_control != "predicted":
                from .priors import select_control, GenericPrior
                control_options = {}
                if prior_control in ("wrong", "generic"):
                    if Path(prior).is_dir():
                        raise ValueError("Donor controls require a live frozen prior to query training observations")
                    training = make_dataset(manifest, "train", cfg, fixed=True)
                    if not len(training):
                        raise ValueError("No training donors")
                    if prior_control == "wrong":
                        control_options["donor"] = training[control_seed % len(training)]
                    else:
                        control_options["generic"] = GenericPrior.from_dataset(provider, training)
                provider = select_control(provider, prior_control, seed=control_seed,
                                          diagnostic_only=diagnostic_only, **control_options)
            summary, _ = validation(backend, dataset, provider, device, cfg, output=output,
                                    observation_seed=observation_seed, guard=guard, candidates=candidates)
        result = {"experiment_id": EXPERIMENT_ID, "checkpoint_hash": sha256_file(checkpoint),
                  "condition": identity["condition"], "split": split, "observation_seed": observation_seed,
                  "data_config": cfg["data"], "candidate_count": candidates,
                  "prior_control": prior_control, "control_seed": control_seed,
                  "diagnostic_only": bool(diagnostic_only or prior_control == "ground_truth"),
                  "training_subset": cfg["data"].get("train_limit"),
                  "fixed_training_observations": cfg["data"].get("fixed_training", False),
                  "prior_provenance": provider.provenance if provider else None,
                  "prior_hash": prior_hash(prior),
                  "prior_substitution": prior_hash(prior) != identity["prior_hash"] or prior_control != "predicted",
                  "expected_count": len(dataset), "limited_evaluation": limit is not None,
                  "metrics": summary, "resources": guard.check()}
        write_json(output.with_suffix(".summary.json"), result)
        return result
