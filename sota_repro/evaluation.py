from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .utils import write_json


CONVENTION = "row_vector_xprime_eq_x_at_R_transpose_plus_t"


def _array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"Expected shape {shape}, received {result.shape}")
    return result


def apply_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ rotation.T + translation


def rotation_error_degrees(predicted: np.ndarray, target: np.ndarray) -> float:
    delta = predicted @ target.T
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _hungarian(cost: np.ndarray) -> list[int]:
    """Minimum-cost square assignment, adapted from the O(n^3) Hungarian algorithm."""
    n = cost.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    current = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if current < minv[j]:
                        minv[j], way[j] = current, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [0] * n
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


def equivalent_assignment(predictions: list[dict[str, Any]], targets: list[dict[str, Any]], classes: list[int] | None) -> list[int]:
    if not classes:
        return list(range(len(targets)))
    if len(predictions) != len(targets) or len(classes) != len(targets):
        raise ValueError("Part transforms and equivalence classes must have equal length")
    target_for_prediction = list(range(len(targets)))
    groups: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(classes):
        groups[int(group)].append(index)
    for indices in groups.values():
        if len(indices) == 1:
            continue
        cost = np.zeros((len(indices), len(indices)))
        for row, predicted_index in enumerate(indices):
            pred = predictions[predicted_index]
            pred_r = _array(pred["rotation"], (3, 3))
            pred_t = _array(pred["translation"], (3,))
            for col, target_index in enumerate(indices):
                target = targets[target_index]
                cost[row, col] = rotation_error_degrees(pred_r, _array(target["rotation"], (3, 3))) / 180.0 + np.linalg.norm(pred_t - _array(target["translation"], (3,)))
        for row, target_index in enumerate(_hungarian(cost)):
            target_for_prediction[indices[row]] = indices[target_index]
    return target_for_prediction


def chamfer_distance(source: np.ndarray, target: np.ndarray, block: int = 2048) -> float:
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if not len(source) or not len(target):
        raise ValueError("Chamfer distance requires non-empty point clouds")
    def nearest_mean(first: np.ndarray, second: np.ndarray) -> float:
        values = []
        for start in range(0, len(first), block):
            points = first[start:start + block]
            squared = ((points[:, None, :] - second[None, :, :]) ** 2).sum(axis=-1)
            values.append(squared.min(axis=1))
        return float(np.concatenate(values).mean())
    return nearest_mean(source, target) + nearest_mean(target, source)


def _records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                payload = json.loads(line)
                if payload.get("convention", CONVENTION) != CONVENTION:
                    raise ValueError(f"{path}:{line_number}: unsupported transform convention")
                yield payload


def evaluate_records(records: Iterable[dict[str, Any]], part_accuracy_threshold: float = 0.01) -> dict[str, Any]:
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    chamfers: list[float] = []
    part_accuracies: list[float] = []
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    count = 0
    for record in records:
        predicted = list(record["predicted_parts"])
        target = list(record["ground_truth_parts"])
        assignment = equivalent_assignment(predicted, target, record.get("equivalence_classes"))
        part_chamfers: list[float] = []
        for index, target_index in enumerate(assignment):
            pred_r, pred_t = _array(predicted[index]["rotation"], (3, 3)), _array(predicted[index]["translation"], (3,))
            gt_r, gt_t = _array(target[target_index]["rotation"], (3, 3)), _array(target[target_index]["translation"], (3,))
            rotation_errors.append(rotation_error_degrees(pred_r, gt_r))
            translation_errors.append(float(np.linalg.norm(pred_t - gt_t)))
            if "source_points" in predicted[index] and "target_points" in target[target_index]:
                source = apply_transform(np.asarray(predicted[index]["source_points"]), pred_r, pred_t)
                destination = apply_transform(np.asarray(target[target_index]["target_points"]), gt_r, gt_t)
                part_chamfers.append(chamfer_distance(source, destination))
            elif "part_chamfer" in predicted[index]:
                part_chamfers.append(float(predicted[index]["part_chamfer"]))
        if part_chamfers:
            chamfers.append(float(np.mean(part_chamfers)))
            part_accuracies.append(float(np.mean(np.asarray(part_chamfers) <= part_accuracy_threshold)))
        by_track[record.get("track", "unknown")].append({"rotation": rotation_errors[-len(predicted):], "translation": translation_errors[-len(predicted):], "chamfer": part_chamfers})
        count += 1
    if not count:
        raise ValueError("No prediction records supplied")
    def summary(values: list[float]) -> dict[str, float | None]:
        return {"mean": float(np.mean(values)) if values else None, "mae": float(np.mean(np.abs(values))) if values else None, "rmse": float(np.sqrt(np.mean(np.square(values)))) if values else None}
    return {
        "convention": CONVENTION, "samples": count, "rotation_degrees": summary(rotation_errors),
        "translation": summary(translation_errors), "chamfer": summary(chamfers),
        "part_accuracy": float(np.mean(part_accuracies)) if part_accuracies else None,
        "part_accuracy_threshold": part_accuracy_threshold,
        "tracks": {track: len(items) for track, items in by_track.items()},
    }


def evaluate_jsonl(predictions: str | Path, output: str | Path, threshold: float = 0.01) -> dict[str, Any]:
    result = evaluate_records(_records(Path(predictions)), threshold)
    write_json(Path(output), result)
    return result
