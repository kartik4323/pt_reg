from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_json


def write_report(run_root: str | Path, output: str | Path) -> Path:
    run_root, output = Path(run_root), Path(output)
    rows: list[dict[str, Any]] = []
    for manifest in sorted(run_root.rglob("run_manifest.json")):
        item = read_json(manifest)
        item["path"] = str(manifest.parent)
        rows.append(item)
    lines = ["# SOTA reduced-data reproduction report", "", "These results are comparative reduced-subset runs, not full-paper reproductions.", "", "| Model | Action | Status | Seed | Run |", "|---|---|---|---:|---|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['action']} | {row['status']} | {row['seed']} | `{row['path']}` |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
