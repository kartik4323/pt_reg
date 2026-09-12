"""Source-aware comparisons and bounded review bundles; failures keep denominators."""
from __future__ import annotations

import json
import copy
import tarfile
from pathlib import Path

import numpy as np

from reassembly.prepare import _file_sha256
from reassembly.resources import write_json, GIB
from .checkpoints import load_checkpoint
from .config import signature


def read_rows(report_path):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows_path = report_path.parent / "examples.jsonl"
    if (report.get("kind") != "repair_evaluation" or report.get("status") != "completed"
            or report.get("metric_threshold") != .01 or report.get("checkpoint_unchanged") is not True
            or _file_sha256(rows_path) != report["results_sha256"]):
        raise ValueError("Incomplete or changed evaluation evidence")
    if _file_sha256(report["checkpoint"]) != report["checkpoint_sha256"]:
        raise ValueError("Evaluated checkpoint changed")
    state = load_checkpoint(report["checkpoint"], fingerprint=report["dataset_fingerprint"])
    if signature(state["cfg"]) != report["config_signature"] or state["purpose"] != report["purpose"]:
        raise ValueError("Evaluation checkpoint configuration or purpose changed")
    patterns = report["pattern_ids"]
    if not patterns or len(patterns) != len(set(patterns)):
        raise ValueError("Evaluation requires distinct declared pattern identities")
    if any(len(report[k]) != len(set(report[k])) for k in ("modes", "seeds", "conditions")):
        raise ValueError("Duplicate evaluation modes, seeds, or conditions")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    key = lambda r: (r["pattern_id"], r["mode"], r["seed"], r["condition"])
    actual = {key(r) for r in rows}
    expected = {(p, m, s, c) for p in patterns for m in report["modes"]
                for s in ([0] if m == "unchanged" else report["seeds"]) for c in report["conditions"]}
    if len(actual) != len(rows) or actual != expected:
        raise ValueError("Evaluation coverage is incomplete or contains duplicate observations")
    identities = {}
    for row in rows:
        pattern, source = row["pattern_id"], row["source_id"]
        if pattern in identities and identities[pattern] != source:
            raise ValueError("Pattern-to-source identity changed across evaluation observations")
        identities[pattern] = source
    return report, rows


def source_bootstrap(differences, seed=42, draws=2000):
    """Bootstrap source means, not correlated fracture/pose observations."""
    means = np.array([np.mean(v) for _, v in sorted(differences.items())], dtype=float)
    if not len(means):
        return {"mean": None, "interval95": None, "sources": 0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(means, (draws, len(means)), replace=True).mean(1)
    return {"mean": float(means.mean()), "interval95": np.quantile(samples, [.025, .975]).tolist(),
            "sources": len(means), "draws": draws, "unit": "source_object"}


def paired_benefit(rows):
    key = lambda r: (r["source_id"], r["pattern_id"], r["mode"], r["seed"])
    contact = {key(r): r for r in rows if r["condition"] == "contact_only"}
    predicted = {key(r): r for r in rows if r["condition"] == "predicted"}
    if len(contact) != sum(r["condition"] == "contact_only" for r in rows) or len(predicted) != sum(r["condition"] == "predicted" for r in rows):
        raise ValueError("Duplicate paired evaluation observations")
    if not contact or contact.keys() != predicted.keys():
        raise ValueError("Scaffold comparisons require identical complete input sets")
    success, chamfer, harmed = {}, {}, 0
    for identity, base in contact.items():
        prior = predicted[identity]
        candidate_hash = base["candidate_coverage"].get("candidate_fingerprint")
        if not candidate_hash or candidate_hash != prior["candidate_coverage"].get("candidate_fingerprint"):
            raise ValueError("Field ablation changed contact candidates")
        source = identity[0]
        success.setdefault(source, []).append(int(prior["success"])-int(base["success"]))
        harmed += int(base["success"] and not prior["success"])
        if base["whole_chamfer"] is not None and prior["whole_chamfer"] is not None:
            chamfer.setdefault(source, []).append(base["whole_chamfer"]-prior["whole_chamfer"])
    delta = float(np.mean([v for values in success.values() for v in values]))
    cd_delta = float(np.mean([v for values in chamfer.values() for v in values])) if chamfer else None
    return {"success_delta": delta, "paired_solved_chamfer_reduction": cd_delta, "harmed_successes": harmed,
        "benefit": bool(delta > 0 or (delta == 0 and cd_delta is not None and cd_delta > 0)),
        "success_source_bootstrap": source_bootstrap(success), "chamfer_source_bootstrap": source_bootstrap(chamfer),
        "paired_count": len(contact), "chamfer_paired_solved_count": sum(map(len, chamfer.values())),
        "note": "Chamfer differences require both solves; assembly success includes every input."}


def acceptance(report_paths, output, guard=None):
    runs, fingerprints, model_signatures, seen_seeds, validation_identities = [], set(), set(), set(), set()
    for path in report_paths:
        report, rows = read_rows(path)
        if report["split"] != "val" or report["purpose"] != "experiment":
            raise ValueError("Acceptance is determined on held-out validation only")
        checkpoint_config = load_checkpoint(report["checkpoint"])["cfg"]
        resolved = json.loads((Path(report["checkpoint"]).parent / "config.resolved.json").read_text(encoding="utf-8"))
        if signature(resolved) != signature(checkpoint_config):
            raise ValueError("Run configuration changed after evaluation")
        if (len(report["pattern_ids"]) != 48 or set(report["seeds"]) != {4101, 4102, 4103}
                or report["modes"] != ["samples_poses"] or set(report["conditions"]) != {"contact_only", "predicted", "gt", "perturbed"}):
            raise ValueError("Acceptance requires all48 distinct validation patterns and the predeclared seeds/conditions")
        seed = checkpoint_config["train"]["seed"]
        if seed in seen_seeds:
            raise ValueError("Duplicate training seed is not an independent repetition")
        seen_seeds.add(seed); fingerprints.add(report["dataset_fingerprint"])
        comparable = copy.deepcopy(checkpoint_config)
        comparable["train"]["seed"] = 42
        # Batch1 fallback preserves the same effective batch and geometry.
        comparable["train"]["grad_accum_steps"] *= comparable["train"]["batch_size"]
        comparable["train"]["batch_size"] = 1
        model_signatures.add(signature(comparable))
        validation_identities.add(tuple(sorted({(r["pattern_id"], r["source_id"]) for r in rows})))
        checks = {}
        for observation_seed in (4101, 4102, 4103):
            selected = [r for r in rows if r["condition"] == "predicted" and r["mode"] == "samples_poses" and r["seed"] == observation_seed]
            checks[str(observation_seed)] = {"count": len(selected), "successes": sum(r["success"] for r in selected),
                "passed": len(selected) == 48 and sum(r["success"] for r in selected) >= 39}
        runs.append({"training_seed": seed, "checks": checks, "benefit": paired_benefit(rows),
            "report": str(Path(path).resolve()), "sha256": _file_sha256(path), "checkpoint_sha256": report["checkpoint_sha256"]})
    if len(fingerprints) != 1 or len(model_signatures) != 1 or len(validation_identities) != 1 or seen_seeds != {42, 43, 44}:
        raise ValueError("Acceptance requires one configuration/dataset with independent training seeds42,43,44")
    target = all(c["passed"] for r in runs for c in r["checks"].values())
    benefit = sum(r["benefit"]["benefit"] for r in runs) >= 2
    result = {"kind": "repair_acceptance", "passed": target and benefit,
        "validation_target_passed": target, "scaffold_benefit_repeated": benefit, "runs": runs,
        "dataset_fingerprint": next(iter(fingerprints)), "geometric_threshold": .01,
        "criterion": "39/48 per pose/sample seed; prior benefit in at least2/3 fresh training runs",
        "limitations": "Validation contains only six source objects. Previously inspected test sources are not an untouched benchmark."}
    write_json(Path(output) / "acceptance.json", result, guard)
    return result


def render_and_bundle(root, guard=None):
    root = Path(root).resolve()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    histories = sorted(root.rglob("history.jsonl"))
    for path in histories:
        records = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not records:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        for key in ("loss", "matching", "near_surface_sdf_l1"):
            selected = [r for r in records if key in r["validation"]]
            if selected:
                axes[0].plot([r["update"] for r in selected], [r["validation"][key] for r in selected], label=key)
        axes[0].set_xlabel("Optimizer updates"); axes[0].set_ylabel("Validation objective"); axes[0].legend()
        selected = [r for r in records if "assembly_success" in r["validation"]]
        if selected:
            axes[1].plot([r["update"] for r in selected], [r["validation"]["assembly_success"] for r in selected])
        axes[1].set_xlabel("Optimizer updates"); axes[1].set_ylabel("Assembly success"); axes[1].set_ylim(0, 1)
        fig.tight_layout()
        if guard:
            guard.check(additional_bytes=1024**2)
        fig.savefig(path.parent / "learning.png", dpi=130); plt.close(fig)
    allowed = {".json", ".jsonl", ".png", ".npz", ".md"}
    assets = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in allowed and not p.is_symlink()]
    size = sum(p.stat().st_size for p in assets)
    if size > 2*GIB:
        raise RuntimeError("Review artifacts exceed2GiB; checkpoint files are already excluded")
    if guard:
        guard.check(additional_bytes=size+1024**2)
    bundle = root / "repair_results.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path in assets:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    return {"bundle": str(bundle), "sha256": _file_sha256(bundle), "files": len(assets), "bytes": bundle.stat().st_size,
            "checkpoint_weights_included": False}
