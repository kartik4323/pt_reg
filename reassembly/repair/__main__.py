"""Versioned repair workflow, using explicitly selected prepared data and runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from reassembly.resources import make_guard, write_json, jsonable

from .config import load_config


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("supplement-queries", "preflight", "train", "evaluate", "contact-gate", "run", "infer", "bundle", "acceptance"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--managed-root", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda:0")
        if name in ("supplement-queries", "preflight", "train", "evaluate", "contact-gate", "run"):
            command.add_argument("--manifest", type=Path, required=True)
        if name in ("preflight", "train", "evaluate", "run"):
            command.add_argument("--query-cache", type=Path)
        if name in ("train", "run"):
            command.add_argument("--field-diagnostic-report", type=Path, required=True)
        if name in ("evaluate", "contact-gate", "infer"):
            command.add_argument("--checkpoint", type=Path, required=True)
        if name == "train":
            command.add_argument("--stage", type=int, choices=(1, 2), required=True)
            command.add_argument("--purpose", choices=("overfit", "experiment"), default="experiment")
            command.add_argument("--preflight-report", type=Path, required=True)
            command.add_argument("--initialize-from", type=Path)
            command.add_argument("--resume", type=Path)
            command.add_argument("--contact-gate-report", type=Path)
        if name == "evaluate":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--split", choices=("train", "val", "test", "cut_holdout"), default="val")
            command.add_argument("--conditions", nargs="+", choices=("contact_only", "predicted", "gt", "perturbed"),
                                 default=["contact_only", "predicted", "gt", "perturbed"])
        if name == "contact-gate":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--overfit-checkpoint", type=Path, required=True)
        if name == "run":
            command.add_argument("--phase", choices=("comparisons", "replicate", "scaffold", "final-test"), required=True)
            command.add_argument("--resume", action="store_true")
        if name == "infer":
            command.add_argument("--save-scaffold", action="store_true")
            command.add_argument("--fragment", type=Path, action="append", required=True)
            command.add_argument("--condition", choices=("contact_only", "predicted"), default="predicted")
        if name == "acceptance":
            command.add_argument("--report", type=Path, action="append", required=True)
    return p


def dispatch(args):
    cfg = load_config(args.config)
    root, output = args.managed_root.expanduser().resolve(), args.output.expanduser().resolve()
    if root not in output.parents:
        raise ValueError("Choose a dedicated output directory beneath --managed-root")
    if hasattr(args, "manifest"):
        prepared = args.manifest.resolve().parent
        if output == prepared or prepared in output.parents or output in prepared.parents:
            raise ValueError("Repair outputs must be separate from immutable prepared data")
    cfg["resources"]["managed_root"] = str(root)
    guard = make_guard(cfg, output, args.manifest.resolve().parent if hasattr(args, "manifest") else root)
    guard.check()
    if args.command == "supplement-queries":
        from .data import supplement_queries
        return supplement_queries(args.manifest, output, guard), 0
    if args.command == "preflight":
        from .training import preflight
        report = preflight(cfg, args.manifest, output, args.device, query_cache=args.query_cache, guard=guard)
        return report, 2 if report["status"] == "failed" else 0
    if args.command == "train":
        from .training import train_stage
        return train_stage(cfg, args.manifest, output, args.stage, args.device, preflight_report=args.preflight_report,
            field_diagnostic_report=args.field_diagnostic_report, purpose=args.purpose,
            initialize_from=args.initialize_from, resume=args.resume, contact_gate_report=args.contact_gate_report,
            query_cache=args.query_cache, guard=guard), 0
    if args.command == "evaluate":
        from .evaluation import evaluate
        return evaluate(args.checkpoint, args.manifest, output, args.device, split=args.split,
            conditions=args.conditions, query_cache=args.query_cache, guard=guard, resume=args.resume), 0
    if args.command == "contact-gate":
        from .evaluation import contact_gate
        report = contact_gate(args.checkpoint, args.overfit_checkpoint, args.manifest, output, args.device, guard, resume=args.resume)
        return report, 0 if report["passed"] else 2
    if args.command == "run":
        from .workflow import run
        report = run(cfg, args.manifest, output, args.device, phase=args.phase,
            field_diagnostic_report=args.field_diagnostic_report, query_cache=args.query_cache, resume=args.resume, guard=guard)
        return report, 2 if report.get("status") == "contact_gate_failed" or report.get("passed") is False else 0
    if args.command == "infer":
        from .evaluation import infer_fragments
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Choose a fresh inference directory")
        report = infer_fragments([np.load(p, allow_pickle=False) for p in args.fragment], args.checkpoint, args.device,
                                 condition=args.condition, save_scaffold=args.save_scaffold)
        output.mkdir(parents=True, exist_ok=True)
        aligned = report.pop("aligned_fragments")
        scaffold = report.pop("scaffold", None)
        if aligned is not None:
            arrays = {f"fragment_{i}": value for i, value in enumerate(aligned)}
            arrays.update(rotations=report["rotations"], translations=report["translations"], transforms=report["transforms"])
            guard.check(additional_bytes=sum(value.nbytes for value in arrays.values()) + 4096)
            np.savez_compressed(output / "assembly.npz", **arrays)
            report["aligned_fragments_file"] = str(output / "assembly.npz")
        if scaffold is not None:
            from reassembly.visualization import plot_scaffold
            guard.check(additional_bytes=sum(v.nbytes for v in scaffold.values() if isinstance(v, np.ndarray)) + 1024**2)
            np.savez_compressed(output / "scaffold.npz", **scaffold)
            plot_scaffold(scaffold, output / "scaffold.png")
            report.update(scaffold_file=str(output / "scaffold.npz"), scaffold_visualization=str(output / "scaffold.png"))
        write_json(output / "result.json", report, guard)
        return report, 0 if report["status"] == "ok" else 2
    if args.command == "acceptance":
        from .reporting import acceptance
        report = acceptance(args.report, output, guard)
        return report, 0 if report["passed"] else 2
    if args.command == "bundle":
        from .reporting import render_and_bundle
        return render_and_bundle(output, guard), 0
    raise ValueError("Unsupported command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        report, status = dispatch(args)
        print(json.dumps(jsonable({k: v for k, v in report.items() if k not in ("config", "experiments", "summaries", "runs")}), indent=2))
        return status
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"repair: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
