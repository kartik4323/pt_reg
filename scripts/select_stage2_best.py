#!/usr/bin/env python3
"""Select the Stage-2 condition from immutable ablation runs by validation CD."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def mean_ci(values: list[float]) -> dict:
    mean = sum(values) / len(values)
    if len(values) < 2:
        ci = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        ci = 1.96 * math.sqrt(variance / len(values))
    return {"mean": mean, "ci95": ci, "n": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a Stage-2 ablation winner by validation aligned Chamfer")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    records = []
    for status_path in root.glob("*_stage1_*_seed*/STATUS.json"):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "complete" or "condition" not in status:
            continue
        validation = status.get("evaluation", {}).get("val", {})
        score = validation.get("aligned_reconstruction_chamfer")
        checkpoint = status.get("stage2_checkpoint")
        if score is None or not checkpoint:
            continue
        records.append({"run": str(status_path.parent), "condition": status["condition"], "score": float(score), "checkpoint": checkpoint})
    if not records:
        raise RuntimeError("No completed Stage-2 ablation STATUS.json files found")
    grouped = defaultdict(list)
    for record in records:
        grouped[record["condition"]].append(record)
    summary = {condition: mean_ci([record["score"] for record in values]) for condition, values in grouped.items()}
    winner_condition = min(summary, key=lambda condition: summary[condition]["mean"])
    best_checkpoint = min(grouped[winner_condition], key=lambda record: record["score"])
    scratch_checkpoint = min(grouped["scratch"], key=lambda record: record["score"]) if "scratch" in grouped else None
    result = {
        "selection_metric": "mean validation aligned reconstruction Chamfer",
        "conditions": summary,
        "winner_condition": winner_condition,
        "best_checkpoint": best_checkpoint,
        "scratch_control_checkpoint": scratch_checkpoint,
        "all_runs": records,
    }
    destination = Path(args.output).resolve() if args.output else root / "stage2_selection.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
