"""Stable exchange format between heterogeneous official implementations and the common scorer.

Each official test wrapper must write an intermediate JSON/JSONL file containing
``sample_id``, ``predicted_parts``, and ``ground_truth_parts``.  This module
normalizes that public contract and refuses ambiguous transform conventions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .evaluation import CONVENTION


def normalize_record(record: dict[str, Any], model: str, default_track: str) -> dict[str, Any]:
    required = {"sample_id", "predicted_parts", "ground_truth_parts"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"{model}: native adapter record is missing {sorted(missing)}")
    result = dict(record)
    result["model"] = model
    result.setdefault("track", default_track)
    result.setdefault("convention", CONVENTION)
    if result["convention"] != CONVENTION:
        raise ValueError(f"{model}: convert native transforms to {CONVENTION} before export")
    for key in ("predicted_parts", "ground_truth_parts"):
        if not isinstance(result[key], list) or not result[key]:
            raise ValueError(f"{model}: {key} must be a non-empty list")
        for part in result[key]:
            if set(("rotation", "translation")) - set(part):
                raise ValueError(f"{model}: every part requires rotation and translation")
    if len(result["predicted_parts"]) != len(result["ground_truth_parts"]):
        raise ValueError(f"{model}: predicted/ground-truth part count differs")
    return result


def load_native_records(path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("JSON adapter input must be an array")
        return value
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def export_predictions(model: str, track: str, source: str | Path, output: str | Path) -> int:
    records = [normalize_record(item, model, track) for item in load_native_records(source)]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return len(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize an official model's prediction export for common evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--track", required=True, choices=["breaking_bad_everyday", "breaking_bad_artifact", "partnet_gpat"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(f"Exported {export_predictions(args.model, args.track, args.input, args.output)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
