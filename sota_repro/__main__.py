from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import bootstrap as source_bootstrap
from .evaluation import evaluate_jsonl
from .materialize_breaking_bad import materialize_breaking_bad
from .materialize import clean_native_view, create_native_view, materialize_common
from .registry import load_registry
from .reporting import write_report
from .subset import build_manifest


ROOT = Path(__file__).resolve().parent


def _models(values: list[str] | None) -> list[str]:
    registry = load_registry()
    names = values or [name for name, spec in registry.items() if spec.source_status == "released"]
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    return names


def _invoke(action: str, args: argparse.Namespace) -> int:
    code = 0
    for model in _models(args.model):
        command = [sys.executable, "-m", "sota_repro.model_entry", "--model", model, "--action", action,
                   "--data-root", args.data_root, "--run-root", args.run_root, "--seed", str(args.seed), "--gpu", args.gpu]
        if args.config:
            command += ["--config", args.config]
        if args.checkpoint:
            command += ["--checkpoint", args.checkpoint]
        if args.verifier_checkpoint:
            command += ["--verifier-checkpoint", args.verifier_checkpoint]
        if args.resume:
            command += ["--resume", args.resume]
        if args.track:
            command += ["--track", args.track]
        if args.keep_native_view:
            command.append("--keep-native-view")
        if args.dry_run:
            command.append("--dry-run")
        code |= subprocess.run(command, cwd=ROOT.parent).returncode
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reduced-data SOTA point-cloud assembly suite")
    subparsers = parser.add_subparsers(dest="command", required=True)
    bootstrap = subparsers.add_parser("bootstrap", help="Clone and verify pinned official sources")
    bootstrap.add_argument("--model", action="append")
    data = subparsers.add_parser("data", help="Plan/materialize the common 20 GiB corpus")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    plan = data_sub.add_parser("plan")
    plan.add_argument("--breaking-bad-root", required=True)
    plan.add_argument("--partnet-gpat-root", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--config")
    materialize = data_sub.add_parser("materialize")
    materialize.add_argument("--manifest", required=True)
    materialize.add_argument("--output", required=True)
    materialize.add_argument("--link", action="store_true")
    bb_materialize = data_sub.add_parser("breaking-bad-materialize", help="Decompress only manifest-selected Breaking Bad objects")
    bb_materialize.add_argument("--manifest", required=True)
    bb_materialize.add_argument("--compressed-root", required=True)
    bb_materialize.add_argument("--official-repository", required=True)
    bb_materialize.add_argument("--output-root", required=True)
    bb_materialize.add_argument("--scratch-root", required=True)
    bb_materialize.add_argument("--python", dest="decompressor_python", default=sys.executable)
    bb_materialize.add_argument("--subset", choices=["everyday", "artifact", "both"], default="both")
    bb_materialize.add_argument("--keep-staging", action="store_true")
    bb_materialize.add_argument("--dry-run", action="store_true")
    view = data_sub.add_parser("view")
    view.add_argument("--model", required=True)
    view.add_argument("--data-root", required=True)
    view.add_argument("--scratch-root", required=True)
    clean_view = data_sub.add_parser("clean-view")
    clean_view.add_argument("--model", required=True)
    clean_view.add_argument("--scratch-root", required=True)
    evaluate = subparsers.add_parser("evaluate", help="Score adapter prediction JSONL")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--part-accuracy-threshold", type=float, default=0.01)
    report = subparsers.add_parser("report", help="Build a run index")
    report.add_argument("--run-root", required=True)
    report.add_argument("--output", required=True)
    for action in ("setup", "smoke", "train", "test"):
        child = subparsers.add_parser(action)
        child.add_argument("--model", action="append")
        child.add_argument("--data-root", default="./sota_data")
        child.add_argument("--run-root", default="./sota_runs")
        child.add_argument("--config")
        child.add_argument("--checkpoint")
        child.add_argument("--verifier-checkpoint")
        child.add_argument("--resume")
        child.add_argument("--track")
        child.add_argument("--seed", type=int, default=42)
        child.add_argument("--gpu", default="0")
        child.add_argument("--keep-native-view", action="store_true")
        child.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "bootstrap":
        for result in source_bootstrap.bootstrap(ROOT, args.model):
            print(f"{result['model']}: {result['revision']} ({result['clean']})")
        return 0
    if args.command == "data":
        if args.data_command == "plan":
            manifest = build_manifest(args.breaking_bad_root, args.partnet_gpat_root, args.output, args.config)
            print(f"Selected {len(manifest['samples'])} samples / {manifest['selected_bytes']} bytes")
        elif args.data_command == "materialize":
            print(materialize_common(args.manifest, args.output, args.link))
        elif args.data_command == "breaking-bad-materialize":
            subsets = ("everyday", "artifact") if args.subset == "both" else (args.subset,)
            print(materialize_breaking_bad(
                args.manifest, args.compressed_root, args.official_repository, args.output_root, args.scratch_root,
                args.decompressor_python, subsets, args.keep_staging, args.dry_run,
            ))
        elif args.data_command == "view":
            print(create_native_view(args.model, args.data_root, args.scratch_root))
        else:
            clean_native_view(args.model, args.scratch_root)
        return 0
    if args.command == "evaluate":
        print(evaluate_jsonl(args.predictions, args.output, args.part_accuracy_threshold))
        return 0
    if args.command == "report":
        print(write_report(args.run_root, args.output))
        return 0
    return _invoke(args.command, args)


if __name__ == "__main__":
    raise SystemExit(main())
