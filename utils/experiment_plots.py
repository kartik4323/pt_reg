"""Small dependency-light plots included in portable experiment bundles."""

from __future__ import annotations

import json
from pathlib import Path


def write_run_plots(run_dir: str | Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    root = Path(run_dir)
    output = root / "plots"
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for history in root.rglob("*_history.jsonl"):
        records = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records or "loss" not in records[0]:
            continue
        fig, axis = plt.subplots(figsize=(5, 3))
        axis.plot([record.get("epoch", idx + 1) for idx, record in enumerate(records)], [record["loss"] for record in records])
        axis.set(xlabel="epoch", ylabel="loss", title=history.stem.replace("_", " "))
        axis.grid(alpha=0.25)
        path = output / f"{history.stem}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path.relative_to(root).as_posix())
    noise = root / "target_noise.json"
    if noise.exists():
        result = json.loads(noise.read_text(encoding="utf-8"))
        curves = result.get("curves", [])
        fig, axis = plt.subplots(figsize=(5, 3))
        for mode in sorted({row["mode"] for row in curves}):
            rows = [row for row in curves if row["mode"] == mode]
            axis.plot([row["post_alignment_chamfer"]["mean"] for row in rows], [row["part_accuracy_at_0.01"]["mean"] for row in rows], marker="o", label=mode)
        axis.set(xlabel="measured post-alignment Chamfer", ylabel="part accuracy @ 0.01", title="GT target robustness")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25)
        path = output / "target_noise_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path.relative_to(root).as_posix())
    return written
