from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from .config import load_config, validate_config
from .io import jsonable, read_json, read_jsonl, study_root, owned_output, write_json, unlock
from .registry import MODEL_NAMES, CONDITIONS


def common(child, data=False, model=False):
    child.add_argument("--run-root", required=True)
    child.add_argument("--config")
    if data:
        child.add_argument("--manifest", required=True)
    if model:
        child.add_argument("--model", choices=MODEL_NAMES, required=True)
        child.add_argument("--source", help="Read-only pinned upstream path; copied into this study before use")


def parser():
    result = argparse.ArgumentParser(description="Independent scaffold transfer study; never starts the original pipeline")
    sub = result.add_subparsers(dest="command", required=True)
    child = sub.add_parser("init", help="Claim a new independent external run directory")
    common(child)
    child = sub.add_parser("check-data", help="Verify shared immutable bottle data")
    common(child, data=True)
    child = sub.add_parser("find-data", help="Locate prepared bottle manifests without importing ML dependencies")
    child.add_argument("--search-root", nargs="+", dest="search_roots")
    child.add_argument("--max-depth", type=int, default=8)
    child.add_argument("--max-files", type=int, default=50000)
    child.add_argument("--path", action="store_true", help="Print a path only when discovery finds one unambiguous complete candidate")
    child = sub.add_parser("setup-tools", help="Plan or install an isolated CPU environment for data and prior tools")
    common(child)
    child.add_argument("--execute", action="store_true")
    child = sub.add_parser("setup", help="Plan or install an isolated native environment")
    common(child, model=True)
    child.add_argument("--execute", action="store_true")
    child = sub.add_parser("export-prior", help="Extract frozen encoder/field tensors without running Stage 3")
    common(child)
    child.add_argument("--checkpoint", required=True)
    child.add_argument("--training-manifest", required=True)
    child.add_argument("--output", required=True)
    child = sub.add_parser("matrix", help="Write explicit native/pretraining/B0-B4 job dependencies")
    common(child, data=True)
    child.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["jigsaw", "ccs", "garf"])
    child.add_argument("--seeds", nargs="+", type=int, default=[42])
    child.add_argument("--conditions", nargs="+", choices=CONDITIONS[1:], default=list(CONDITIONS[1:]))
    child.add_argument("--prior-v3")
    child.add_argument("--prior-v2")
    child.add_argument("--output")
    child = sub.add_parser("run-matrix", help="Execute pending study jobs sequentially in their own environments")
    child.add_argument("--matrix", required=True)
    child.add_argument("--resume", action="store_true")
    for command in ("train", "preflight"):
        child = sub.add_parser(command)
        common(child, data=True, model=True)
        child.add_argument("--condition", choices=CONDITIONS, default="native")
        child.add_argument("--stage", choices=["pretrain", "assembly"], default="assembly")
        child.add_argument("--prior")
        child.add_argument("--feature-checkpoint")
        child.add_argument("--device", default="cuda:0")
        child.add_argument("--options", help="JSON file of explicit native backend options")
        if command == "train":
            child.add_argument("--initialize")
            child.add_argument("--resume", action="store_true")
            child.add_argument("--run-id")
            child.add_argument("--seed", type=int)
            child.add_argument("--updates", type=int)
            child.add_argument("--cpu-smoke", action="store_true")
            child.add_argument("--prior-device", default="cpu")
    child = sub.add_parser("evaluate", help="Export and score all sealed native predictions")
    common(child, data=True)
    child.add_argument("--checkpoint", required=True)
    child.add_argument("--source")
    child.add_argument("--prior")
    child.add_argument("--prior-device", default="cpu")
    child.add_argument("--device", default="cuda:0")
    child.add_argument("--split", choices=["train", "val", "test", "cut_holdout"], default="val")
    child.add_argument("--observation-seed", type=int, default=4101)
    child.add_argument("--output", required=True)
    child.add_argument("--limit", type=int, help="Explicit exploratory subset; never a full evaluation")
    child.add_argument("--candidates", type=int, default=1, help="Fixed native seed bank; first draw remains operational output")
    child.add_argument("--prior-control", choices=["predicted", "constant_uncertainty", "shuffled_uncertainty",
                       "null", "global", "wrong", "generic", "ground_truth", "distance_noise", "distance_bias",
                       "shuffled_distance", "sign_flip"], default="predicted")
    child.add_argument("--control-seed", type=int, default=42)
    child.add_argument("--diagnostic-only", action="store_true", help="Required for ground-truth scaffold oracle")
    for command in ("cache-prior", "prior-diagnostics"):
        child = sub.add_parser(command)
        common(child, data=True)
        child.add_argument("--prior", required=True)
        child.add_argument("--output", required=True)
        child.add_argument("--split", choices=["train", "val", "test", "cut_holdout"], default="val")
        child.add_argument("--observation-seed", type=int, default=4101)
        child.add_argument("--device", default="cpu")
        child.add_argument("--limit", type=int)
        if command == "prior-diagnostics":
            child.add_argument("--kind", choices=["field", "pose"], required=True)
            child.add_argument("--diagnostic-only", action="store_true", required=True)
            child.add_argument("--steps", type=int, default=25)
    child = sub.add_parser("frozen-compare", help="Replay identical starts across A0-A7")
    common(child, data=True)
    child.add_argument("--predictions", required=True)
    child.add_argument("--output", required=True)
    child.add_argument("--prior-v3")
    child.add_argument("--prior-v2")
    child.add_argument("--split", choices=["train", "val", "test", "cut_holdout"], default="val")
    child.add_argument("--observation-seed", type=int, default=4101)
    child.add_argument("--mode", choices=["refine", "rank"], default="refine")
    child.add_argument("--steps", type=int, default=25)
    child.add_argument("--device", default="cpu")
    child.add_argument("--conditions", nargs="+", choices=["A%d" % i for i in range(8)])
    child.add_argument("--v3-control", choices=["predicted", "constant_uncertainty", "shuffled_uncertainty", "null",
                       "global", "distance_noise", "distance_bias", "shuffled_distance", "sign_flip"], default="predicted")
    child.add_argument("--control-seed", type=int, default=42)
    child = sub.add_parser("compare", help="Source-clustered paired comparison of complete evaluation rows")
    common(child)
    child.add_argument("--baseline", required=True)
    child.add_argument("--treatment", required=True)
    child.add_argument("--output", required=True)
    child = sub.add_parser("status", help="Inspect only this experiment's jobs")
    common(child)
    child = sub.add_parser("unlock", help="Remove a stale study lock only after its process exited")
    common(child)
    return result


def dispatch(args):
    cfg = load_config(getattr(args, "config", None))
    if args.command == "find-data":
        from .manifests import find_manifests
        return find_manifests(args.search_roots, max_depth=args.max_depth, max_files=args.max_files)
    if args.command == "init":
        root = study_root(args.run_root, create=True)
        from .resources import ResourceGuard
        return {"run_root": str(root), "resources": ResourceGuard(root, cfg["resources"]).check()}
    if args.command == "run-matrix":
        from .matrix import execute_matrix
        return execute_matrix(args.matrix, resume=args.resume)
    root = study_root(args.run_root)
    if args.command == "unlock":
        return unlock(root)
    if args.command == "status":
        return {"experiment_id": "scaffold_sota", "runs": [read_json(p) for p in sorted((root / "runs").glob("*/run.json"))]}
    if args.command == "check-data":
        from .manifests import check_data
        result = check_data(args.manifest)
        write_json(root / "data_verification.json", result)
        return result
    if args.command == "setup-tools":
        from .bootstrap import setup_tools
        return setup_tools(root, cfg, execute=args.execute)
    if args.command == "setup":
        from .setup import setup_model
        return setup_model(root, args.model, cfg, execute=args.execute, source=args.source)
    if args.command == "export-prior":
        from .priors import export_prior
        from .resources import ResourceGuard
        destination = owned_output(root, args.output)
        ResourceGuard(root, cfg["resources"]).check(additional_bytes=Path(args.checkpoint).stat().st_size)
        return export_prior(args.checkpoint, destination, training_manifest=args.training_manifest)
    if args.command == "matrix":
        from .matrix import compile_matrix
        return compile_matrix(root, args.manifest, cfg, models=args.models, seeds=args.seeds,
                              prior_v3=args.prior_v3, prior_v2=args.prior_v2,
                              conditions=args.conditions, output=args.output)
    if args.command in ("train", "preflight"):
        options = read_json(args.options) if args.options else None
        if args.command == "preflight":
            from .preflight import preflight
            return preflight(root, args.model, args.manifest, cfg, source=args.source,
                             device=args.device, condition=args.condition, stage=args.stage,
                             prior=args.prior, feature_checkpoint=args.feature_checkpoint, options=options)
        from .training import train
        if args.seed is not None:
            cfg["train"]["seed"] = cfg["data"]["seed"] = args.seed
        if args.updates is not None:
            cfg["train"]["updates"] = args.updates
        validate_config(cfg)
        return train(root=root, model=args.model, manifest=args.manifest, cfg=cfg,
                     condition=args.condition, stage=args.stage, source=args.source,
                     prior=args.prior, prior_device=args.prior_device, feature_checkpoint=args.feature_checkpoint,
                     initialize=args.initialize, resume=args.resume, device=args.device,
                     run_id=args.run_id, model_options=options, cpu_smoke=args.cpu_smoke)
    if args.command == "evaluate":
        from .training import evaluate
        return evaluate(root=root, checkpoint=args.checkpoint, manifest=args.manifest, output=args.output,
                        source=args.source, prior=args.prior, prior_device=args.prior_device,
                        device=args.device, split=args.split, observation_seed=args.observation_seed, limit=args.limit,
                        candidates=args.candidates, prior_control=args.prior_control,
                        control_seed=args.control_seed, diagnostic_only=args.diagnostic_only)
    if args.command in ("cache-prior", "prior-diagnostics"):
        from .training import make_dataset, make_prior
        from .resources import ResourceGuard
        from .io import RunLock
        destination = owned_output(root, args.output)
        if destination.exists():
            raise ValueError("Use a fresh output path")
        guard = ResourceGuard(root, cfg["resources"], args.device)
        with RunLock(root):
            guard.require_idle_device()
            guard.check()
            dataset = make_dataset(args.manifest, args.split, cfg, fixed=True,
                                   seed=args.observation_seed, limit=args.limit)
            provider = make_prior(args.prior, cfg, dataset, args.device)
            if args.command == "cache-prior":
                from .priors import export_token_cache
                return export_token_cache(provider, dataset, destination, guard=guard)
            from .evaluation.diagnostics import evaluate_field_quality, evaluate_pose_stability
            if args.kind == "field":
                return evaluate_field_quality(provider, dataset, destination, diagnostic_only=True, guard=guard)
            return evaluate_pose_stability(provider, dataset, destination, diagnostic_only=True,
                                           steps=args.steps, device=args.device, guard=guard)
    if args.command == "frozen-compare":
        from .comparisons import frozen_comparison
        return frozen_comparison(root=root, manifest=args.manifest, predictions=args.predictions,
                                 output=args.output, cfg=cfg, prior_v3=args.prior_v3, prior_v2=args.prior_v2,
                                 split=args.split, observation_seed=args.observation_seed,
                                 mode=args.mode, steps=args.steps, device=args.device, conditions=args.conditions,
                                 v3_control=args.v3_control, control_seed=args.control_seed)
    if args.command == "compare":
        from .evaluation import compare_conditions
        result = compare_conditions(read_jsonl(args.baseline), read_jsonl(args.treatment),
                                    seed=cfg["train"]["seed"], bootstrap_samples=cfg["evaluation"]["bootstrap_samples"])
        write_json(owned_output(root, args.output), result)
        return result
    raise ValueError("Unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
        if args.command == "find-data" and args.path:
            candidates = result["candidates"]
            if (result.get("truncated") or result.get("errors") or len(candidates) != 1
                    or not candidates[0].get("assets_present") or not candidates[0].get("fingerprint_matches")):
                print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
                raise ValueError("No unambiguous complete manifest selected. Inspect find-data output and narrow --search-root to the intended prepared dataset; no data is generated or substituted.")
            print(candidates[0]["path"])
            return 0
    except ModuleNotFoundError as exc:
        name = exc.name or "an optional dependency"
        model = getattr(args, "model", None)
        if args.command in ("train", "preflight", "evaluate"):
            hint = ("Use this recipient's study environment; install it with setup --model " + model
                    if model else "Use the checkpoint recipient's study environment under envs/<model>")
        else:
            hint = "Run setup-tools --run-root \"$STUDY_ROOT\" --execute, then use its envs/tools/bin/python interpreter"
        print("scaffold_sota: Missing dependency '%s' in %s. %s.\nThe active environment has not been modified." % (name, sys.executable, hint), file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        print("scaffold_sota: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(jsonable(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
