"""Result-artifact helpers for the assembly pipeline.

Every training / eval entry point uses these to emit self-describing outputs to
the run's ``output.dir`` so a run can be diagnosed after the fact. This matters
most when training happens on a remote GPU box and only the output directory is
copied back for analysis:

- ``run_manifest_{stage}.json`` -- exact config + environment that produced the run.
- ``{stage}_history.jsonl``     -- one JSON line per epoch (all metrics + lr + time).
- ``{stage}_train.log``         -- plaintext mirror of the console summary lines.
- ``stage3_eval_{source}.json`` -- pose-error *distributions*, not just means.
- ``RESULTS.md``                -- human-readable roll-up of everything present.

The module has no side effects on import and only depends on torch + stdlib.
"""

from __future__ import annotations

import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


def utc_now() -> str:
    """UTC timestamp, e.g. ``2026-07-26T12:34:56Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value):
    """Best-effort conversion of tensors / numpy scalars to JSON-native types."""
    if isinstance(value, torch.Tensor):
        return value.item() if value.ndim == 0 else value.tolist()
    module = type(value).__module__
    if module == "numpy":
        item = getattr(value, "item", None)
        if callable(item):
            return value.item()
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            return value.tolist()
    return value


# ── Manifest ────────────────────────────────────────────────────────────────

def write_manifest(
    out_dir,
    stage: str,
    cfg: dict,
    device,
    extra: Optional[dict] = None,
) -> Path:
    """Write ``run_manifest_{stage}.json`` capturing config + environment."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = device if isinstance(device, torch.device) else torch.device(device)
    cuda_ok = torch.cuda.is_available()
    manifest = {
        "stage": stage,
        "timestamp_utc": utc_now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_ok,
        "device": str(dev),
        "device_name": (
            torch.cuda.get_device_name(dev)
            if cuda_ok and dev.type == "cuda"
            else None
        ),
        "config": cfg,
    }
    if extra:
        manifest.update(extra)
    path = out_dir / f"run_manifest_{stage}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


# ── Per-epoch history (JSON Lines) ────────────────────────────────────────────

class HistoryWriter:
    """Append-only per-epoch metrics history as JSON Lines.

    Truncates any existing file on construction so each fresh run starts clean;
    each ``append`` flushes immediately so a crashed remote run still leaves a
    usable partial history.
    """

    def __init__(self, out_dir, stage: str) -> None:
        self.path = Path(out_dir) / f"{stage}_history.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def append(self, row: dict) -> None:
        clean = {key: _jsonable(value) for key, value in row.items()}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(clean) + "\n")


# ── Console logging (tee to file) ─────────────────────────────────────────────

class RunLogger:
    """Mirror per-epoch summary lines to ``{stage}_train.log`` and the console.

    Opens/closes the file per line so the log is always flushed (crash-safe on a
    remote box). tqdm progress bars go to stderr and are intentionally not
    captured -- the log stays a clean record of the summary lines.
    """

    def __init__(self, out_dir, stage: str) -> None:
        self.path = Path(out_dir) / f"{stage}_train.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"\n=== {utc_now()} :: {stage} run start ===\n")

    def log(self, message: str) -> None:
        print(message)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(message + "\n")


# ── Distribution summaries ────────────────────────────────────────────────────

def summarize_distribution(values: Sequence[float]) -> dict:
    """rmse / mean / std / min / p50 / p90 / p95 / max over a 1-D sequence.

    ``rmse`` is included because the Breaking Bad benchmark reports RMSE(R)/RMSE(T)
    rather than means; having it here means every existing report gains it for free.
    """
    values = [float(v) for v in values]
    if not values:
        return {"count": 0}
    tensor = torch.tensor(values, dtype=torch.float64)
    quantiles = torch.quantile(
        tensor, torch.tensor([0.5, 0.9, 0.95], dtype=torch.float64)
    ).tolist()
    return {
        "count": int(tensor.numel()),
        "rmse": float(tensor.pow(2).mean().sqrt()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "p50": quantiles[0],
        "p90": quantiles[1],
        "p95": quantiles[2],
        "max": float(tensor.max()),
    }


# ── RESULTS.md roll-up ────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _md_table(data: dict) -> str:
    rows = ["| metric | value |", "| --- | --- |"]
    for key, value in data.items():
        if isinstance(value, float):
            rows.append(f"| {key} | {value:.6g} |")
        else:
            rows.append(f"| {key} | {value} |")
    return "\n".join(rows)


def build_results_markdown(out_dir) -> Optional[Path]:
    """Scan ``out_dir`` for known artifact JSONs and (re)write ``RESULTS.md``.

    Tolerant of missing files -- only sections whose source JSON exists are
    rendered, so calling this after any subset of stages is safe.
    """
    out_dir = Path(out_dir)
    sections: List[Tuple[str, str]] = []

    # Environment header from whichever manifest is most complete.
    header_lines: List[str] = []
    for stage in ("stage3", "stage2", "stage1"):
        manifest = _load_json(out_dir / f"run_manifest_{stage}.json")
        if manifest:
            for key in ("timestamp_utc", "host", "device", "device_name", "torch"):
                if manifest.get(key) is not None:
                    header_lines.append(f"- **{key}**: {manifest[key]}")
            break

    flat_metrics = [
        ("Stage 1 (final epoch)", "stage1_metrics.json"),
        ("Stage 2 (final epoch)", "stage2_metrics.json"),
        ("Stage 3 train — reconstruction target (final epoch)", "stage3_pose_metrics.json"),
        ("Stage 3 train — ground-truth target (final epoch)", "stage3_pose_gt_target_metrics.json"),
    ]
    for heading, name in flat_metrics:
        data = _load_json(out_dir / name)
        if isinstance(data, dict):
            sections.append((heading, _md_table(data)))

    for source in ("reconstruction", "ground_truth"):
        report = _load_json(out_dir / f"stage3_eval_{source}.json")
        if not isinstance(report, dict):
            continue
        body = []
        info = {k: report[k] for k in ("split", "num_samples", "has_pose_checkpoint") if k in report}
        if info:
            body.append(_md_table(info))
            body.append("")
        for dist_key in ("rotation_error_deg", "translation_error"):
            dist = report.get(dist_key)
            if isinstance(dist, dict):
                body.append(f"**{dist_key}**")
                body.append("")
                body.append(_md_table(dist))
                body.append("")
        sections.append((f"Stage 3 eval — {source} target", "\n".join(body)))

    oracle = _load_json(out_dir / "oracle_alignment_report.json")
    if isinstance(oracle, dict):
        flat = {k: v for k, v in oracle.items() if not isinstance(v, (dict, list))}
        sections.append(("Oracle alignment (pre-flight)", _md_table(flat)))

    if not sections:
        return None

    lines = ["# Pipeline run results", "", f"_Generated {utc_now()}_", ""]
    if header_lines:
        lines.extend(header_lines)
        lines.append("")
    for heading, body in sections:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body)
        lines.append("")

    path = out_dir / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
