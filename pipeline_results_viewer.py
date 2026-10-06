"""Read-only E0--E7 notebook support, deliberately outside the hashed package.

No model imports, inference, evaluation writes, checkpoint deserialization or
Store construction. Saved evaluator metrics and references are display-only.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

STAGES = ["E0", "E1", "E1_ORACLE", "E2", "E3", "E4", "E5", "E6_TRAIN", "E6", "E7"]
POSE_STAGES = {"E3", "E4", "E5", "E6", "E7"}


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def contained(base, relative):
    base = Path(base).resolve()
    target = (base / relative).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise ValueError(f"Artifact escapes its root: {relative}")
    return target


def flatten(value, prefix=""):
    result = {}
    for key, item in (value or {}).items():
        name = f"{prefix}{key}"
        if isinstance(item, dict):
            result.update(flatten(item, name + "."))
        else:
            result[name] = item
    return result


def discover_runs(paths):
    """Accept explicit runs or arbitrary nested group directories; deduplicate."""
    roots = set()
    warnings = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        if path.name == "study.json":
            path = path.parent
        if not path.exists():
            warnings.append(f"Run path unavailable: {path}")
            continue
        if (path / "study.json").is_file():
            roots.add(path)
        else:
            found = [p.parent for p in path.rglob("study.json")]
            roots.update(found)
            if not found:
                warnings.append(f"No study.json found under: {path}")
    return sorted(roots), warnings


class Catalog:
    def __init__(self, paths, datasets=None):
        self.paths = list(paths)
        # Optional run-path -> local dataset.json map for downloaded artifacts.
        self.datasets = {str(Path(k).expanduser().resolve()): Path(v).expanduser().resolve()
                         for k, v in (datasets or {}).items()}
        self.refresh()

    def _read(self, path, default=None):
        try:
            return read_json(path)
        except (OSError, ValueError) as exc:
            self.warnings.append(f"Cannot read {path}: {exc}")
            return default

    def _lines(self, path):
        result = []
        try:
            with path.open(encoding="utf-8-sig") as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        result.append(json.loads(line))
                    except ValueError:
                        self.warnings.append(f"Incomplete/invalid JSONL row: {path}:{number}")
        except OSError as exc:
            self.warnings.append(f"Cannot read {path}: {exc}")
        return result

    def refresh(self):
        self.roots, self.warnings = discover_runs(self.paths)
        self.runs, self.jobs, self.metrics, self.summaries, self.labels = {}, {}, {}, [], []
        superseded=set()
        rows = []
        for root in self.roots:
            run = str(root)
            study = self._read(root / "study.json")
            if not study:
                continue
            cfg = study.get("identity", {}).get("config", {})
            self.runs[run] = {"root": root, "study": study, "config": cfg}
            for path in sorted((root / "evaluation").glob("*/metrics.jsonl")):
                for metric in self._lines(path):
                    if metric.get("job_id"):
                        self.metrics[(run, metric["job_id"])] = metric
            for path in sorted((root / "evaluation").glob("*/summary.json")):
                value = self._read(path)
                if value:
                    self.summaries.append({"run": run, "split": path.parent.name, "summary": value})
            for path in sorted((root / "evaluation").glob("*/superseded_outcomes.json")):
                for value in self._read(path, []):
                    superseded.add((run,value['job_id']))
            label_doc = self._read(root / "pseudo_labels" / "train.json", {}) if (root / "pseudo_labels" / "train.json").exists() else {}
            for label in label_doc.get("labels", []):
                self.labels.append({"run": run, **label})
            # Directory-level scan also exposes a lock before result.json exists.
            for directory in sorted((root / "jobs").glob("*")):
                if not directory.is_dir():
                    continue
                result_path = directory / "result.json"
                job = self._read(result_path) if result_path.exists() else None
                lock = directory / "running.lock"
                if not job:
                    if lock.exists():
                        lock_info = self._read(lock, {})
                        job = {"job_id": directory.name, "stage": "UNKNOWN", "case_id": "unknown",
                               "arm": "unknown", "status": "locked", "lock": lock_info}
                    else:
                        continue
                jid = job.get("job_id", directory.name)
                self.jobs[(run, jid)] = job
                output = job.get("output") or {}
                metric = self.metrics.get((run, jid), {})
                row = {"run": run, "run_label": root.name, "job_id": jid,
                       "stage": job.get("stage", "UNKNOWN"), "case_id": job.get("case_id"),
                       "source_id": job.get("source_id"), "split": job.get("split", "unknown"),
                       "arm": job.get("arm"), "status": job.get("status", "unknown"),
                       "evaluation_status": metric.get("status", "not_evaluated"),
                       "oracle": bool(job.get("oracle", False)), "smoke": bool(job.get("smoke", cfg.get("smoke", False))),
                       "seconds": job.get("seconds"), "error": job.get("error"),
                       "artifact_dir": str(directory), "locked": lock.exists(),
                       "base_case_id": output.get("base_case_id", job.get("extra", {}).get("base_case", metric.get("base_case_id", job.get("case_id")))),
                       "condition": cfg.get("experiment_profile", root.name),
                       "N": cfg.get("solver", {}).get("candidates"),
                       "template_weight": cfg.get("solver", {}).get("template_weight"),
                       "primary_model": cfg.get("primary_model"), "primary_input": cfg.get("primary_input"),
                       "policy": output.get("policy", cfg.get("primary_policy")),
                       "model": output.get("model"), "input_type": output.get("input_type"),
                       "seed": output.get("seed"), "template_job": output.get("template_job"),
                       "not_applicable": output.get("not_applicable")}
                if lock.exists():
                    # A lock can be a live worker or a stale lock. Never infer liveness.
                    row["status"] = "locked"
                    row["evaluation_status"] = "not_evaluated"
                row.update(flatten(job.get("extra", {}), "factor."))
                row.update(flatten(output.get("selection", {}), "selection."))
                row.update(flatten(output.get("alignment", {}), "alignment."))
                row.update(flatten(output.get("diagnostics", {}), "diagnostic."))
                row["template_rejected"] = output.get("template_rejected", False)
                if not lock.exists() and job.get("status") == "complete":
                    row.update(flatten(metric.get("metrics", {}), "metric."))
                rows.append(row)
        basic = ["run", "run_label", "stage", "split", "case_id", "base_case_id", "source_id", "arm", "status", "evaluation_status", "oracle", "smoke", "job_id", "seconds"]
        derived = {(run, row['case_id']): (row.get('output') or {}).get('base_case_id', row.get('extra', {}).get('base_case'))
                   for (run, _), row in self.jobs.items() if row.get('stage') == 'E7'}
        for row in rows:
            row['superseded']=(row['run'],row['job_id']) in superseded
            base = derived.get((row['run'], row['case_id']))
            if base and row['base_case_id'] == row['case_id']:
                row['base_case_id'] = base
        self.frame = pd.DataFrame(rows)
        for column in basic:
            if column not in self.frame:
                self.frame[column] = pd.Series(dtype="object")
        self.frame = self.frame.sort_values(["run", "stage", "case_id", "arm"], na_position="last").reset_index(drop=True)
        return self.frame

    def select(self, stages=None, run=None, split=None, case=None, include_oracles=False, include_smoke=False):
        frame = self.frame.copy()
        if 'superseded' in frame:
            frame=frame[~frame.superseded.astype(bool)]
        if stages is not None:
            frame = frame[frame.stage.isin([stages] if isinstance(stages, str) else stages)]
        if run is not None:
            frame = frame[frame.run == str(Path(run).resolve())]
        if split is not None:
            frame = frame[frame.split == split]
        if case is not None:
            frame = frame[(frame.case_id == case) | (frame.base_case_id == case)]
        if not include_oracles:
            frame = frame[~frame.oracle.astype(bool)]
        if not include_smoke:
            frame = frame[~frame.smoke.astype(bool)]
        return frame.copy()

    def row(self, run, job_id):
        matching = self.frame[(self.frame.run == str(Path(run).resolve())) & (self.frame.job_id == job_id)]
        if matching.empty:
            raise KeyError(f"Unknown job: {run} / {job_id}")
        return matching.iloc[0]

    def job_dir(self, run, job_id):
        return contained(Path(run) / "jobs", job_id)

    def preparation(self, row, base=False):
        case_id = row["base_case_id"] if base else row["case_id"]
        found = self.frame[(self.frame.run == row["run"]) & (self.frame.case_id == case_id)
                           & (self.frame.stage == "E0") & (self.frame.status == "complete")]
        if found.empty:
            raise FileNotFoundError(f"No completed E0 observed points for {case_id}")
        return found.iloc[0]

    def dataset(self, run):
        root = Path(run)
        if run in self.datasets:
            return self.datasets[run] if self.datasets[run].is_file() else None
        candidates = [root / "dataset" / "dataset.json", root.parent / "data" / "dataset.json"]
        if (root / "experiment.json").exists():
            doc = self._read(root / "experiment.json", {})
            if doc.get("dataset"):
                supplied = Path(doc["dataset"]).expanduser()
                candidates.insert(0, supplied if supplied.is_absolute() else root / supplied)
        return next((p.resolve() for p in candidates if p.is_file()), None)

    def integrity(self, run, job_id):
        """Explicit, potentially expensive hash check, for one selected job only."""
        job = self.jobs[(str(Path(run).resolve()), job_id)]
        checks = []
        for asset in job.get("artifacts", []):
            try:
                path = contained(run, asset["path"])
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                state = "ok" if digest.hexdigest() == asset["sha256"] else "HASH_MISMATCH"
            except (OSError, ValueError, KeyError) as exc:
                state = str(exc)
            checks.append({"path": asset.get("path"), "integrity": state})
        return pd.DataFrame(checks)


def outcome_table(frame, groups=None):
    """Keep returned failures in source-macro denominators; unscored stays unknown.

    Does not infer missing/unstarted jobs. These are observed-job outcomes, not a
    full prespecified request manifest. Existing evaluator gates remain separate.
    """
    groups = groups or ["run", "stage", "arm"]
    rows = []
    for key, part in frame.groupby(groups, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        valid = part.status.eq("complete")
        success = pd.to_numeric(part.get("metric.success", pd.Series(np.nan, index=part.index)), errors="coerce")
        success = success.where(valid)
        # Failure outcomes count only for pose stages; lack of references is
        # still visible in the returned/scored counts.
        pose_failure = part.stage.isin(POSE_STAGES) & (
            part.status.eq("failed") | part.get("not_applicable", pd.Series(None, index=part.index, dtype="object")).notna())
        success = success.mask(pose_failure, 0.0)
        source_means = success.groupby(part.source_id).mean().dropna()
        rows.append(dict(zip(groups, key), observed_jobs=len(part), returned=int(valid.sum()),
                         failed=int(part.status.eq("failed").sum()), locked=int(part.status.eq("locked").sum()),
                         scored_or_failed=int(success.notna().sum()), sources=part.source_id.nunique(),
                         source_macro_success=source_means.mean() if len(source_means) else np.nan,
                         total_seconds=pd.to_numeric(part.seconds, errors="coerce").sum(min_count=1)))
    return pd.DataFrame(rows)


def comparison_table(catalog, split=None):
    rows = []
    for entry in catalog.summaries:
        if split and split != entry["split"]:
            continue
        for name in ("paired_comparisons", "targeted_comparisons", "robustness_comparisons"):
            for item in entry["summary"].get(name, []):
                rows.append({"run": entry["run"], "split": entry["split"], "comparison_type": name, **item})
    return pd.DataFrame(rows)


def image_gallery(frame, max_images=24, width=230):
    """Embed image bytes, rather than host-local links, in notebook output."""
    from PIL import Image
    tiles = []
    for _, row in frame.iterrows():
        if row["status"] != "complete":
            continue
        directory = Path(row["artifact_dir"])
        images = sorted(directory.glob("*/*.png")) if row["stage"] == "E0" else sorted(directory.glob("*.png"))
        for path in images:
            if len(tiles) >= max_images:
                break
            try:
                with Image.open(path) as original:
                    picture = original.convert("RGB")
                    picture.thumbnail((width, width))
                    buffer = io.BytesIO()
                    picture.save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                label = html.escape(f"{row['run_label']} | {row['arm']} | {path.relative_to(directory)}")
                badge = " ORACLE" if row["oracle"] else ""
                badge += " SMOKE" if row["smoke"] else ""
                tiles.append(f'<div style="display:inline-block;vertical-align:top;margin:8px;width:{width+15}px"><p>{label}{badge}</p><img style="max-width:100%" src="data:image/png;base64,{encoded}"></div>')
            except (OSError, ValueError) as exc:
                tiles.append(f"<p>{html.escape(str(path))}: {html.escape(str(exc))}</p>")
        if len(tiles) >= max_images:
            break
    return "".join(tiles) or "<p>No completed image artifacts in this selection.</p>"


def sample_points(points, cap=2500):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected Nx3 points")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > cap:
        points = points[np.linspace(0, len(points)-1, cap).astype(int)]
    return points


def load_observed(catalog, row, base=False):
    prep = catalog.preparation(row, base)
    path = Path(prep.artifact_dir) / "observed.npz"
    with np.load(path, allow_pickle=False) as archive:
        keys = sorted((k for k in archive.files if k.startswith("points_")), key=lambda k: int(k.split("_")[1]))
        pieces = [archive[k].copy() for k in keys]
        metadata = {k: archive[k].copy() for k in ("anchor", "scale", "centers")}
    return prep, pieces, metadata


def reference_panel(catalog, row, retained=None):
    """Explicit evaluator-only view; does not call any pipeline evaluator."""
    dataset = catalog.dataset(row["run"])
    if dataset is None:
        raise FileNotFoundError("Set DATASETS[run_path] to a local dataset.json for reference visualization.")
    doc = read_json(dataset)
    base_id = row["base_case_id"]
    rec = next((c for c in doc["cases"] if c["id"] == base_id), None)
    refs = read_json(dataset.parent / "evaluator_only" / "index.json")
    if rec is None or base_id not in refs:
        raise FileNotFoundError("Evaluator reference unavailable for this case.")
    _, _, base_meta = load_observed(catalog, row, base=True)
    centers = np.asarray(base_meta["centers"])
    anchor, scale = int(base_meta["anchor"]), float(base_meta["scale"])
    with np.load(contained(dataset.parent, rec["path"]), allow_pickle=False) as archive:
        original = [archive[f"points_{i}"].copy() for i in range(rec["pieces"])]
    with np.load(contained(dataset.parent, refs[base_id]["path"]), allow_pickle=False) as archive:
        world = archive["world_transforms"].copy()
    target = np.linalg.inv(world[anchor])[None] @ world
    truth = []
    for i, cloud in enumerate(original):
        if retained is not None and i not in retained:
            continue
        points = (cloud @ target[i, :3, :3].T + target[i, :3, 3] - centers[anchor]) / scale
        truth.append((f"Fragment {i}", points))
    return "EVALUATOR ONLY: reference placement", truth


def point_panels(catalog, row, show_reference=False, candidate_index=0):
    """Each panel carries its coordinate-frame label; raw meshes stay separate."""
    directory = Path(row["artifact_dir"])
    job = catalog.jobs[(row["run"], row["job_id"])]
    output = job.get("output") or {}
    panels = []
    if row["stage"] == "E2":
        shape = directory / "shape.npz"
        if not shape.exists():
            raise FileNotFoundError("Reconstruction has no shape.npz; inspect mesh.obj and error log.")
        with np.load(shape, allow_pickle=False) as archive:
            if "raw" in archive:
                panels.append(("Raw bridge surface (bridge coordinates)", [("Raw template", archive["raw"].copy())]))
            if "aligned" in archive:
                aligned = archive["aligned"].copy()
                _, pieces, metadata = load_observed(catalog, row)
                anchor = int(metadata["anchor"])
                panels.append(("Aligned prior + anchor (normalized anchor frame)",
                               [("Aligned prior", aligned), (f"Anchor fragment {anchor}", pieces[anchor])]))
        if show_reference:
            panels.append(reference_panel(catalog, row))
        return panels
    # E7 rotation predictions were mapped back to base observations before save.
    # Applying them to augmented E0 points would create a misleading assembly.
    factor = output.get("factor", job.get("extra", {}).get("factor"))
    prep, pieces, _ = load_observed(catalog, row, base=(row["stage"] == "E7" and factor == "rotation"))
    with np.load(Path(prep.artifact_dir) / "candidates.npz", allow_pickle=False) as archive:
        count = len(archive["poses"])
        index = min(max(0, int(candidate_index)), count-1)
        initial = archive["poses"][index].copy()
    colors = output.get("retained_ids", list(range(len(pieces))))
    if len(colors) != len(pieces):
        raise ValueError("Observed fragment count does not match retained_ids")
    if len(initial) != len(pieces):
        raise ValueError("Initial candidate count does not match observed fragments")
    transform = lambda clouds, poses: [(f"Fragment {i}", p @ pose[:3, :3].T + pose[:3, 3])
                                      for i, p, pose in zip(colors, clouds, poses)]
    panels.append((f"E0 proposal {index}/{count-1} (normalized anchor frame)", transform(pieces, initial)))
    if (directory / "starts.npz").exists():
        with np.load(directory / "starts.npz", allow_pickle=False) as archive:
            start = archive["poses"][min(index, len(archive["poses"])-1)].copy()
        panels.append(("E3 diagnostic start (may be privileged)", transform(pieces, start)))
    if (directory / "poses.npz").exists():
        with np.load(directory / "poses.npz", allow_pickle=False) as archive:
            poses = archive["normalized"].copy()
        if len(poses) != len(pieces):
            raise ValueError("Pose count does not match observed fragments")
        clouds = transform(pieces, poses)
        template_id = output.get("template_job")
        if template_id and (row["run"], template_id) in catalog.jobs and factor != "rotation":
            path = catalog.job_dir(row["run"], template_id) / "shape.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as archive:
                    if "aligned" in archive:
                        clouds.append(("Generated prior (auxiliary)", archive["aligned"].copy()))
        title = f"{row['stage']} | {row['arm']} (normalized anchor frame)"
        if row["stage"] == "E7" and factor == "rotation":
            title += " | mapped back to base input; prior overlay omitted"
        panels.append((title, clouds))
    if show_reference:
        # Evaluator files are accessed only upon the explicit display toggle.
        panels.append(reference_panel(catalog, row, colors))
    return panels


def point_figure(panels, max_points=2500, interactive=True):
    """One global cubic range; fixed piece colors and deterministic subsampling."""
    sampled = [(title, [(name, sample_points(p, max_points)) for name, p in clouds]) for title, clouds in panels]
    arrays = [p for _, clouds in sampled for _, p in clouds if len(p)]
    if not arrays:
        raise ValueError("No nonempty point clouds")
    joined = np.concatenate(arrays)
    center = (joined.min(0) + joined.max(0)) / 2
    half = max(float(np.ptp(joined, axis=0).max()) * .55, 1e-3)
    ranges = [[float(c-half), float(c+half)] for c in center]
    palette = ["#2563eb", "#ef4444", "#16a34a", "#f59e0b", "#8b5cf6", "#06b6d4"]
    names = list(dict.fromkeys(name for _, clouds in sampled for name, _ in clouds))
    colors = {name: palette[i % len(palette)] for i, name in enumerate(names)}
    for name in names:
        if "prior" in name.lower() or "template" in name.lower():
            colors[name] = "#94a3b8"
        elif name.startswith("Fragment "):
            colors[name] = palette[int(name.split()[-1]) % len(palette)]
    import textwrap
    panel_titles = [textwrap.fill(title, width=48) for title, _ in sampled]
    if interactive:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=len(sampled), specs=[[{"type": "scene"} for _ in sampled]],
                            subplot_titles=[title.replace("\n", "<br>") for title in panel_titles])
        shown = set()
        for column, (_, clouds) in enumerate(sampled, 1):
            for name, points in clouds:
                fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                              name=name, legendgroup=name, showlegend=name not in shown,
                              marker={"size": 2, "color": colors[name], "opacity": .4 if "prior" in name.lower() else .8}), row=1, col=column)
                shown.add(name)
        for column in range(1, len(sampled)+1):
            scene = "scene" if column == 1 else f"scene{column}"
            fig.update_layout(**{scene: {"aspectmode": "cube", "xaxis": {"range": ranges[0]},
                                        "yaxis": {"range": ranges[1]}, "zaxis": {"range": ranges[2]}}})
        fig.update_annotations(font_size=12)
        fig.update_layout(height=600, margin={"l": 0, "r": 0, "b": 0, "t": 130})
        return fig
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6*len(sampled), 6))
    for column, (title, clouds) in enumerate(sampled, 1):
        axis = fig.add_subplot(1, len(sampled), column, projection="3d")
        for name, points in clouds:
            axis.scatter(*points.T, s=2, color=colors[name], label=name, alpha=.6)
        axis.set(xlim=ranges[0], ylim=ranges[1], zlim=ranges[2])
        axis.set_title(panel_titles[column-1], fontsize=10)
        axis.set_box_aspect((1, 1, 1))
        axis.legend(fontsize=7)
    title_lines = max(title.count("\n") + 1 for title in panel_titles)
    fig.tight_layout(rect=(0, 0, 1, max(.72, 1 - .04*title_lines)))
    return fig


def learning_curves(catalog):
    rows = []
    for _, row in catalog.frame[catalog.frame.stage.eq("E6_TRAIN")].iterrows():
        path = Path(row.artifact_dir) / "learning_curve.jsonl"
        if path.exists():
            for item in catalog._lines(path):
                rows.append({"run": row.run, "arm": row.arm, "job_id": row.job_id, **item})
    return pd.DataFrame(rows)


def browser(catalog, include_smoke=False, interactive=True, max_points=2500):
    """Explicit refresh; no polling, automatic inference or experiment writes."""
    import ipywidgets as widgets
    from IPython.display import HTML, display
    run = widgets.Dropdown(description="Run", layout=widgets.Layout(width="95%"))
    stage = widgets.Dropdown(description="Stage", options=STAGES + ["UNKNOWN"])
    case = widgets.Dropdown(description="Case", layout=widgets.Layout(width="95%"))
    job = widgets.Dropdown(description="Output", layout=widgets.Layout(width="95%"))
    oracle = widgets.Checkbox(value=False, description="Include oracle jobs")
    reference = widgets.Checkbox(value=False, description="Show evaluator reference")
    refresh = widgets.Button(description="Refresh files")
    integrity = widgets.Button(description="Check selected hashes")
    out = widgets.Output()
    busy = [False]

    def fill(*_):
        if busy[0]:
            return
        busy[0] = True
        frame = catalog.select(include_oracles=oracle.value, include_smoke=include_smoke)
        old_run, old_case, old_job = run.value, case.value, job.value
        run.options = [(str(Path(p)), p) for p in sorted(frame.run.unique())]
        if old_run in frame.run.unique():
            run.value = old_run
        subset = frame[(frame.run == run.value) & frame.stage.eq(stage.value)]
        ids = sorted(subset.case_id.dropna().unique())
        case.options = ids
        if old_case in ids:
            case.value = old_case
        subset = subset[subset.case_id == case.value]
        job.options = [(f"{r.arm} | {r.status} | {r.job_id[:8]}", r.job_id) for _, r in subset.iterrows()]
        if old_job in subset.job_id.values:
            job.value = old_job
        busy[0] = False
        render()

    def render(*_):
        if busy[0]:
            return
        with out:
            out.clear_output(wait=True)
            if not run.value or not job.value:
                print("No jobs for this selection. Missing stages may still be pending.")
                return
            row = catalog.row(run.value, job.value)
            record = catalog.jobs[(run.value, job.value)]
            print(f"{row.stage} | {row.arm} | {row.status} | evaluator: {row.evaluation_status}")
            print(f"Artifact directory: {row.artifact_dir}")
            if row.smoke:
                print("SMOKE: implementation validation only; not research evidence.")
            if row.oracle:
                print("ORACLE: privileged diagnostic, excluded from deployable results.")
            display(pd.DataFrame([flatten(record.get("output", {}))]).T.rename(columns={0: "saved output"}))
            if (run.value, job.value) in catalog.metrics:
                display(pd.DataFrame([flatten(catalog.metrics[(run.value, job.value)].get("metrics", {}))]).T.rename(columns={0: "evaluator metric"}))
            display(HTML(image_gallery(pd.DataFrame([row]), max_images=24)))
            if row.status == "complete" and row.stage in POSE_STAGES | {"E0", "E2"}:
                try:
                    fig = point_figure(point_panels(catalog, row, reference.value), max_points, interactive)
                    if interactive:
                        # Full JS embedded so exported output remains usable offline.
                        display(HTML(fig.to_html(full_html=False, include_plotlyjs=True)))
                    else:
                        import matplotlib.pyplot as plt
                        display(fig)
                        plt.close(fig)
                except (OSError, ValueError, KeyError, ImportError) as exc:
                    print(f"3D preview unavailable: {exc}")
            directory = Path(row.artifact_dir)
            for name in ("prediction.json", "alignment.json", "checks.json", "backend.json", "training.json"):
                if (directory / name).exists():
                    print(f"\n{name}")
                    print(json.dumps(catalog._read(directory / name, {}), indent=2)[:12000])
            for name in ("error.txt", "worker.log"):
                if (directory / name).exists():
                    print(f"\n{name} (last 6000 characters)")
                    print((directory / name).read_text(encoding="utf-8", errors="replace")[-6000:])
            parent_rows = []
            ids = list(record.get("parents", []))
            for key in ("template_job", "image_job", "checkpoint_job"):
                value = record.get("output", {}).get(key)
                if value and value not in ids:
                    ids.append(value)
            for parent in ids:
                value = catalog.jobs.get((run.value, parent))
                parent_rows.append({"job_id": parent, "stage": value.get("stage") if value else "not local",
                                    "arm": value.get("arm") if value else None,
                                    "artifact_dir": str(catalog.job_dir(run.value, parent))})
            if parent_rows:
                print("\nSaved dependencies / selected image / template / checkpoint")
                display(pd.DataFrame(parent_rows))
            if row.stage == "E6_TRAIN":
                print("Checkpoint is listed only; no torch.load or unpickling is performed.")

    def update(*_):
        catalog.refresh()
        fill()

    def verify(*_):
        with out:
            if run.value and job.value:
                display(catalog.integrity(run.value, job.value))

    run.observe(fill, names="value")
    stage.observe(fill, names="value")
    case.observe(fill, names="value")
    oracle.observe(fill, names="value")
    job.observe(render, names="value")
    reference.observe(render, names="value")
    refresh.on_click(update)
    integrity.on_click(verify)
    fill()
    return widgets.VBox([run, stage, case, job, widgets.HBox([oracle, reference]),
                         widgets.HBox([refresh, integrity]), out])
