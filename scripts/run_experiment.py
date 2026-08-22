#!/usr/bin/env python3
"""Remote execution harness for the PartNet/GPAT research matrix.

Each invocation creates an immutable run directory.  It does not download data;
pass a VM-only processed PartNet root via ``--data-root``.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import io
import json
import math
import os
import random
import re
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from data.assembly_dataset import build_object_dataset
from training.two_stage_trainer import Stage1Trainer, Stage2Trainer, Stage3PoseTrainer, run_pose_stage
from utils.experiment_artifacts import ExperimentRun
from utils.experiment_metrics import evaluate_stage2
from utils.experiment_preflight import apply_stage3_memory_policy, profile_stage3_worst_case
from utils.experiment_plots import write_run_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible PartNet/GPAT experiment suites")
    parser.add_argument("--suite", required=True, choices=["pilot", "stage1_ablation", "target_fidelity", "target_noise", "gpat_reference"])
    parser.add_argument("--config", default="configs/partnet_gpat_pilot.yaml")
    parser.add_argument("--data-root", default=None, help="VM-only prepared PartNet root containing manifest.json")
    parser.add_argument("--run-root", default="./runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--condition", choices=["scratch", "pretrained_e2e", "pretrained_schedule", "no_compat_graph"], default=None)
    parser.add_argument(
        "--stage2-checkpoint", action="append", default=[],
        help="Stage-2 checkpoint, optionally NAME=/path/to/checkpoint; repeat for best and scratch.",
    )
    parser.add_argument(
        "--stage3-checkpoint", action="append", default=[],
        help="Stage-3 checkpoint, optionally NAME=/path/to/checkpoint; used by target_noise.",
    )
    parser.add_argument("--official-gpat-root", default=None)
    parser.add_argument("--official-command", default=None, help="Command executed unchanged inside --official-gpat-root")
    parser.add_argument("--official-checkpoint", default=None, help="Optional official GPAT checkpoint to hash and archive with the reference run")
    parser.add_argument("--resume-dir", default=None)
    return parser.parse_args()


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configured(base: dict, run: ExperimentRun, data_root: str | None, seed: int | None) -> dict:
    cfg = copy.deepcopy(base)
    if data_root:
        cfg["data"]["shapenet_root"] = str(Path(data_root).resolve())
    if seed is not None:
        cfg["data"]["seed"] = seed
    cfg.setdefault("output", {})["dir"] = str(run.artifact_dir)
    return cfg


class _Tee(io.TextIOBase):
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def capture_console(run: ExperimentRun):
    """Mirror stdout/stderr while retaining the raw VM console in a bundle."""
    with open(run.path / "stdout_stderr.log", "a", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
            yield


def make_run(args: argparse.Namespace, base: dict, suite: str, seed: int | None) -> tuple[ExperimentRun, dict]:
    run = ExperimentRun(
        args.run_root, suite, seed, base, sys.argv,
        resume_dir=args.resume_dir,
    )
    if args.resume_dir:
        resolved_path = run.path / "config.resolved.yaml"
        if not resolved_path.exists():
            raise RuntimeError("Cannot resume a run without its resolved configuration")
        with open(resolved_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    else:
        cfg = configured(base, run, args.data_root, seed)
    run.write_resolved_config(cfg)
    if args.data_root:
        run.write_dataset_manifest(str(Path(args.data_root).resolve() / cfg["data"].get("manifest", "manifest.json")))
        manifest_path = Path(args.data_root).resolve() / cfg["data"].get("manifest", "manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            examples = [entry["id"] for entry in manifest.get("samples", []) if entry.get("split") == "test"][:8]
            run.write_json("qualitative_examples.json", {"object_ids": examples, "selection": "first eight manifest test IDs"})
    return run, cfg


def checkpoint_specs(values: list[str], argument: str) -> list[tuple[str, str]]:
    parsed = []
    for index, value in enumerate(values):
        name, sep, path = value.partition("=")
        if not sep:
            path = name
            name = f"model{index + 1}"
        checkpoint = Path(path).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"{argument} does not exist: {checkpoint}")
        parsed.append((name, str(checkpoint)))
    return parsed


def compact_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "per_sample"}


def parse_console_metrics(text: str) -> dict:
    """Best-effort scalar extraction while retaining the authoritative raw log."""
    metrics = {}
    pattern = re.compile(r"(?:^|[,;\s])([A-Za-z][A-Za-z0-9_/@ .-]{1,48})\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    for key, value in pattern.findall(text):
        clean = "_".join(key.strip().lower().split())
        try:
            metrics.setdefault(clean, []).append(float(value))
        except ValueError:
            continue
    return {key: {"last": values[-1], "count": len(values)} for key, values in metrics.items()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_and_record(cfg: dict, checkpoint: str, device: torch.device, run: ExperimentRun) -> dict:
    from models.assembly import build_assembly_model

    model = build_assembly_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    metrics = {}
    for split in ("val", "test"):
        try:
            requested = cfg.get("evaluation", {}).get("max_samples", 0)
            dataset = build_object_dataset(cfg, split, requested or None)
            metrics[split] = evaluate_stage2(
                model,
                dataset,
                device,
                batch_size=1,
                prediction_dir=run.artifact_dir / "predictions" / "stage2" / split,
            )
        except RuntimeError as exc:
            metrics[split] = {"skipped": str(exc)}
    run.write_json("stage2_evaluation.json", metrics)
    return metrics


def run_stage1_then_stage2(cfg: dict, device: torch.device, condition: str) -> tuple[str | None, str]:
    stage1_checkpoint = None
    if condition != "scratch":
        existing_stage1 = Path(cfg["output"]["dir"]) / "stage1_pretrained.pt"
        stage1_checkpoint = str(existing_stage1 if existing_stage1.exists() else Stage1Trainer(cfg, device).train())
    stage_cfg = cfg.setdefault("stage2", {})
    if condition == "scratch":
        freeze = False
        stage_cfg["unfreeze_pretrained_epoch"] = None
    elif condition == "pretrained_e2e":
        freeze = False
        stage_cfg["unfreeze_pretrained_epoch"] = None
    else:
        freeze = True
        # Epochs [0, ceil(0.25*E)-1] stay frozen, then the loaded Stage-1
        # encoder/scorer are fine-tuned jointly.  The one-epoch pilot unfreezes
        # immediately because a fractional freeze interval is not representable.
        stage_cfg["unfreeze_pretrained_epoch"] = int(math.ceil(float(stage_cfg.get("epochs", 1)) * 0.25)) if int(stage_cfg.get("epochs", 1)) > 1 else 0
    existing_stage2 = Path(cfg["output"]["dir"]) / "stage2_assembly.pt"
    final_stage2_checkpoint = Path(
        existing_stage2 if existing_stage2.exists() else Stage2Trainer(
            cfg, device, stage1_checkpoint=stage1_checkpoint, freeze_pretrained=freeze
        ).train()
    )
    best_stage2_checkpoint = final_stage2_checkpoint.parent / "stage2_best_validation.pt"
    return stage1_checkpoint, str(best_stage2_checkpoint if best_stage2_checkpoint.exists() else final_stage2_checkpoint)


def run_pilot(args: argparse.Namespace, base: dict) -> None:
    run, cfg = make_run(args, base, "pilot", args.seeds[0])
    set_seed(args.seeds[0])
    try:
        with capture_console(run):
            preflight = profile_stage3_worst_case(cfg, torch.device(args.device))
            fallback = apply_stage3_memory_policy(cfg, preflight)
            preflight["fallback_applied"] = fallback
            run.write_json("stage3_preflight.json", preflight)
            run.write_resolved_config(cfg)
            _, stage2 = run_stage1_then_stage2(cfg, torch.device(args.device), "pretrained_schedule")
            evaluation = evaluate_and_record(cfg, stage2, torch.device(args.device), run)
            # The pilot validates the Stage-3 target contract but makes no accuracy claim.
            checkpoint = Stage3PoseTrainer(cfg, torch.device(args.device), stage2, target_source="ground_truth").train()
            pose_metrics = run_pose_stage(
                cfg, torch.device(args.device), stage2, str(checkpoint), split="test", target_source="ground_truth",
                prediction_dir=run.artifact_dir / "predictions" / "stage3" / "ground_truth",
            )
        run.record_peak_memory()
        run.set_status("complete", stage2_checkpoint=stage2, stage2_final_checkpoint=str(Path(stage2).parent / "stage2_assembly.pt"), stage3_checkpoint=str(checkpoint), evaluation={k: compact_metrics(v) for k, v in evaluation.items()}, pose_metrics=pose_metrics)
        write_run_plots(run.path)
        run.write_results()
    except Exception as exc:
        run.record_peak_memory()
        run.set_status("failed", error=repr(exc))
        run.write_results()
        raise


def run_ablation(args: argparse.Namespace, base: dict) -> None:
    conditions = [args.condition] if args.condition else ["scratch", "pretrained_e2e", "pretrained_schedule", "no_compat_graph"]
    for condition in conditions:
        for seed in args.seeds:
            run, cfg = make_run(args, base, f"stage1_{condition}", seed)
            if condition == "no_compat_graph":
                cfg.setdefault("model", {}).setdefault("assembly", {})["disable_compatibility"] = True
                cfg.setdefault("loss", {}).setdefault("stage2", {})["lambda_compat"] = 0.0
            set_seed(seed)
            try:
                with capture_console(run):
                    stage1, stage2 = run_stage1_then_stage2(cfg, torch.device(args.device), condition)
                    evaluation = evaluate_and_record(cfg, stage2, torch.device(args.device), run)
                run.record_peak_memory()
                run.set_status("complete", condition=condition, stage1_checkpoint=stage1, stage2_checkpoint=stage2, stage2_final_checkpoint=str(Path(stage2).parent / "stage2_assembly.pt"), evaluation={k: compact_metrics(v) for k, v in evaluation.items()})
                write_run_plots(run.path)
                run.write_results()
            except Exception as exc:
                run.record_peak_memory()
                run.set_status("failed", condition=condition, error=repr(exc))
                write_run_plots(run.path)
                run.write_results()
                raise


def run_target_fidelity(args: argparse.Namespace, base: dict) -> None:
    stage2_models = checkpoint_specs(args.stage2_checkpoint, "--stage2-checkpoint")
    if not stage2_models:
        raise ValueError("--stage2-checkpoint is required for target_fidelity")
    for model_name, stage2_checkpoint in stage2_models:
        for name, source, gauge in (
            ("ground_truth", "ground_truth", False),
            ("reconstruction_baseline", "reconstruction", False),
            ("reconstruction_gauge", "reconstruction", True),
        ):
            run, cfg = make_run(args, base, f"target_fidelity_{model_name}_{name}", args.seeds[0])
            cfg.setdefault("stage3", {})["gauge_aware"] = gauge
            set_seed(args.seeds[0])
            try:
                with capture_console(run):
                    preflight = profile_stage3_worst_case(cfg, torch.device(args.device))
                    fallback = apply_stage3_memory_policy(cfg, preflight)
                    preflight["fallback_applied"] = fallback
                    run.write_json("stage3_preflight.json", preflight)
                    run.write_resolved_config(cfg)
                    existing = Path(cfg["output"]["dir"]) / ("stage3_pose_gt_target.pt" if source == "ground_truth" else "stage3_pose.pt")
                    checkpoint = existing if existing.exists() else Stage3PoseTrainer(cfg, torch.device(args.device), stage2_checkpoint, target_source=source).train()
                    metrics = {
                        evaluation_target: run_pose_stage(
                            cfg,
                            torch.device(args.device),
                            stage2_checkpoint,
                            str(checkpoint),
                            target_source=evaluation_target,
                            prediction_dir=run.artifact_dir / "predictions" / "stage3" / evaluation_target,
                        )
                        for evaluation_target in ("ground_truth", "reconstruction", "oracle_reconstruction")
                    }
                run.record_peak_memory()
                run.set_status("complete", model=model_name, target_source=source, gauge_aware=gauge, stage3_checkpoint=str(checkpoint), metrics=metrics)
                write_run_plots(run.path)
                run.write_results()
            except Exception as exc:
                run.record_peak_memory()
                run.set_status("failed", error=repr(exc))
                run.write_results()
                raise


def run_noise(args: argparse.Namespace, base: dict) -> None:
    from utils.target_noise import NOISE_LEVELS, evaluate_target_noise

    stage2 = dict(checkpoint_specs(args.stage2_checkpoint, "--stage2-checkpoint"))
    stage3 = checkpoint_specs(args.stage3_checkpoint, "--stage3-checkpoint")
    if not stage2 or not stage3:
        raise ValueError("target_noise requires named --stage2-checkpoint and --stage3-checkpoint inputs")
    for model_name, stage3_checkpoint in stage3:
        stage2_checkpoint = stage2.get(model_name)
        if stage2_checkpoint is None and len(stage2) == 1:
            stage2_checkpoint = next(iter(stage2.values()))
        if stage2_checkpoint is None:
            raise ValueError(f"No matching Stage-2 checkpoint for Stage-3 model {model_name!r}")
        run, cfg = make_run(args, base, f"target_noise_{model_name}", args.seeds[0])
        set_seed(args.seeds[0])
        run.write_json("noise_protocol.json", {"target_source": "ground_truth", "repetitions": 5, "levels": NOISE_LEVELS})
        try:
            with capture_console(run):
                result = evaluate_target_noise(
                    cfg, stage3_checkpoint, torch.device(args.device), repetitions=5,
                    max_samples=int(cfg.get("evaluation", {}).get("max_samples", 0)),
                )
                run.write_json("target_noise.json", result)
            run.record_peak_memory()
            run.set_status("complete", stage2_checkpoint=stage2_checkpoint, stage3_checkpoint=stage3_checkpoint, curves=result["curves"])
            write_run_plots(run.path)
            run.write_results()
        except Exception as exc:
            run.record_peak_memory()
            run.set_status("failed", error=repr(exc))
            run.write_results()
            raise


def run_reference(args: argparse.Namespace, base: dict) -> None:
    if not args.official_gpat_root or not args.official_command:
        raise ValueError("gpat_reference requires --official-gpat-root and --official-command")
    run, _ = make_run(args, base, "gpat_reference", None)
    source = Path(args.official_gpat_root).resolve()
    try:
        with capture_console(run):
            process = subprocess.run(
                args.official_command,
                shell=True,
                cwd=source,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        (run.path / "official_gpat_stdout.log").write_text(process.stdout, encoding="utf-8")
        run.write_json("official_gpat_metrics.json", parse_console_metrics(process.stdout))
        checkpoint_info = None
        if args.official_checkpoint:
            official_checkpoint = Path(args.official_checkpoint).resolve()
            if not official_checkpoint.is_file():
                raise FileNotFoundError(official_checkpoint)
            archived_checkpoint = run.artifact_dir / "official_gpat_checkpoint" / official_checkpoint.name
            archived_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(official_checkpoint, archived_checkpoint)
            checkpoint_info = {"source": str(official_checkpoint), "sha256": sha256(official_checkpoint), "archived": str(archived_checkpoint.relative_to(run.path))}
        run.write_json(
            "official_gpat.json",
            {"root": str(source), "command": args.official_command, "returncode": process.returncode,
             "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
             "checkpoint": checkpoint_info},
        )
        run.set_status("complete" if process.returncode == 0 else "failed", returncode=process.returncode)
        write_run_plots(run.path)
        run.write_results()
        if process.returncode:
            raise SystemExit(process.returncode)
    except Exception as exc:
        run.set_status("failed", error=repr(exc))
        run.write_results()
        raise


def main() -> None:
    args = parse_args()
    if args.resume_dir:
        multi = (
            (args.suite == "stage1_ablation" and (not args.condition or len(args.seeds) != 1))
            or (args.suite == "target_fidelity" and len(args.stage2_checkpoint) != 1)
            or (args.suite == "target_noise" and len(args.stage3_checkpoint) != 1)
        )
        if multi:
            raise ValueError("--resume-dir may resume exactly one named run; select one condition/seed/checkpoint")
    base = load_cfg(args.config)
    if args.suite == "pilot":
        run_pilot(args, base)
    elif args.suite == "stage1_ablation":
        run_ablation(args, base)
    elif args.suite == "target_fidelity":
        run_target_fidelity(args, base)
    elif args.suite == "target_noise":
        run_noise(args, base)
    else:
        run_reference(args, base)


if __name__ == "__main__":
    main()
