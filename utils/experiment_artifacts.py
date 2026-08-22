"""Immutable, portable experiment-run metadata for remote GPU execution."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _command(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


class ExperimentRun:
    """One append-only run directory, safe to copy without its source dataset."""

    def __init__(
        self,
        run_root: str,
        suite: str,
        seed: Optional[int],
        cfg: dict,
        command: list[str],
        *,
        resume_dir: Optional[str] = None,
    ) -> None:
        if resume_dir:
            self.path = Path(resume_dir).resolve()
            if not self.path.exists():
                raise FileNotFoundError(f"Cannot resume missing run directory: {self.path}")
        else:
            suffix = f"seed{seed}" if seed is not None else "reference"
            self.path = Path(run_root).resolve() / f"{_utc_stamp()}_{suite}_{suffix}"
            self.path.mkdir(parents=True, exist_ok=False)
        self.artifact_dir = self.path / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._write_context(cfg, command, suite, seed)
        self.set_status("running")

    def _write_context(self, cfg: dict, command: list[str], suite: str, seed: Optional[int]) -> None:
        resolved = self.path / "config.resolved.yaml"
        if not resolved.exists():
            with open(resolved, "w", encoding="utf-8") as handle:
                yaml.safe_dump(cfg, handle, sort_keys=False)
        command_path = self.path / "command.txt"
        if not command_path.exists():
            command_path.write_text(" ".join(command) + "\n", encoding="utf-8")
        revision = self.path / "source_revision.json"
        if not revision.exists():
            revision.write_text(
                json.dumps(
                    {
                        "suite": suite,
                        "seed": seed,
                        "python": sys.version,
                        "platform": platform.platform(),
                        "cwd": os.getcwd(),
                        "git_head": _command(["git", "rev-parse", "HEAD"]),
                        "git_status": _command(["git", "status", "--short"]),
                        "git_diff": _command(["git", "diff", "--binary"]),
                        "pip_freeze": _command([sys.executable, "-m", "pip", "freeze"]),
                        "conda_explicit": _command(["conda", "list", "--explicit"]),
                        "nvidia_smi": _command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
                        "torch": torch.__version__,
                        "torch_cuda": torch.version.cuda,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    def write_resolved_config(self, cfg: dict) -> Path:
        """Persist the actual post-CLI configuration, not just its template."""
        path = self.path / "config.resolved.yaml"
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def set_status(self, state: str, **extra: Any) -> None:
        payload = {
            "state": state,
            "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **extra,
        }
        (self.path / "STATUS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_json(self, name: str, payload: Dict[str, Any]) -> Path:
        path = self.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def write_dataset_manifest(self, source: str) -> None:
        source_path = Path(source)
        if source_path.exists():
            shutil.copy2(source_path, self.path / "dataset_manifest.json")

    def record_peak_memory(self) -> None:
        if torch.cuda.is_available():
            self.write_json(
                "gpu_memory.json",
                {
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                },
            )

    def write_results(self) -> Path:
        """Portable report index with the headline metrics and all artifact paths."""
        status = {}
        status_path = self.path / "STATUS.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
        files = sorted(
            item.relative_to(self.path).as_posix()
            for item in self.path.rglob("*")
            if item.is_file() and item.name != "RESULTS.md"
        )
        lines = [
            "# PartNet/GPAT experiment result bundle",
            "",
            f"- Status: `{status.get('state', 'unknown')}`",
            f"- Updated (UTC): `{status.get('updated_utc', 'unknown')}`",
            "- Raw PartNet data is deliberately excluded.",
            "",
        ]
        stage2_path = self.path / "stage2_evaluation.json"
        if stage2_path.exists():
            stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
            lines.extend(["## Stage 2", ""])
            for split, values in stage2.items():
                if isinstance(values, dict) and "aligned_reconstruction_chamfer" in values:
                    lines.append(
                        f"- {split}: aligned Chamfer `{values['aligned_reconstruction_chamfer']:.6f}`, "
                        f"raw Chamfer `{values.get('reconstruction_chamfer', float('nan')):.6f}`, "
                        f"contact accuracy `{values.get('contact_accuracy', float('nan')):.4f}`"
                    )
            lines.append("")
        stage3_reports = sorted(self.path.rglob("stage3_eval_*.json"))
        if stage3_reports:
            lines.extend(["## Stage 3", ""])
            for report_path in stage3_reports:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                gpat = report.get("gpat_compatible_chamfer", {}).get("mean", float("nan"))
                accuracy = report.get("part_accuracy_at_0.01", float("nan"))
                success = report.get("assembly_success_rate", float("nan"))
                lines.append(
                    f"- {report.get('target_source', report_path.stem)}: GPAT-compatible Chamfer `{gpat:.6f}`, "
                    f"part accuracy@0.01 `{accuracy:.4f}`, assembly success `{success:.4f}`"
                )
            lines.append("")
        noise_path = self.path / "target_noise.json"
        if noise_path.exists():
            noise = json.loads(noise_path.read_text(encoding="utf-8"))
            lines.extend(["## Target robustness", "", f"- Curves: `{len(noise.get('curves', []))}` fixed perturbation levels.", ""])
        lines.extend([
            "## Included artifacts",
            "",
        ])
        lines.extend(f"- `{file}`" for file in files)
        return self.write_text("RESULTS.md", "\n".join(lines) + "\n")
