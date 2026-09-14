"""Study-owned input/export boundary around native PMTR/CMNet pose graphs."""
from __future__ import annotations

import importlib.util
import itertools
import numpy as np
import torch

from .matching_common import failed, pose_result


def supervised_pair(sample, indices, points, radius=.018, training=True, all_correspondences=False):
    """Match native MPA training: random source, strongest-overlap target.

    Only this explicitly supervised function accesses canonical points. The
    complete object's prior stays attached to the selected pair in the backend.
    """
    canonical = torch.as_tensor(sample["canonical_points"], device=points.device,
                                dtype=points.dtype)[indices]
    count = points.shape[0]
    source = int(torch.randint(count, ()).item()) if training else 0
    others = [j for j in range(count) if j != source]
    distances = [torch.cdist(canonical[source], canonical[j]) for j in others]
    overlaps = [(d < radius).sum() if all_correspondences else (d.min(dim=1).values < radius).sum()
                for d in distances]
    target = others[int(torch.stack(overlaps).argmax())]
    if int(max(overlaps)) == 0:
        raise ValueError("No sampled training contact for native pairwise supervision")
    if training and bool(torch.rand(()) < .5):
        source, target = target, source
    return source, target, canonical


def load_translation_solver(path, module_name):
    """Import the pinned native helper, without executing its dataset/test loop."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.estimate_poses_given_rot


def native_pair_to_standard(rotation, translation):
    """Native graph export applies R^-1 (x+t); return standard R*x+t."""
    r = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64)
    return np.linalg.inv(r), np.linalg.solve(r, t)


def native_pose_graph(outputs, count, anchor, kind, translation_solver):
    """Pinned edge scores and Shonan/linear translation solver, input-only.

    Native direct-anchor fallback is retained and labelled. A deterministic
    predicted-anchor initialization replaces unseeded GTSAM random initialization
    for paired replay; this declared policy is shared by every study condition.
    """
    import gtsam
    pair_indices = list(itertools.permutations(range(count), 2))
    edges = []
    if kind == "pmtr":
        for i, j in pair_indices:
            scores = outputs[(i, j)]["node_corr_scores"]
            score = float(((scores.detach().double().cpu() * 1e5) ** 2).mean())
            edges.append((i, j, score))
    elif kind == "cmnet":
        for i in range(count):
            choices = [(float(outputs[(j, i)]["matching_scores_drop"].detach().exp().sum().cpu()), j)
                       for j in range(count) if j != i]
            score, j = max(choices, key=lambda item: item[0])
            edges.append((i, j, score))
    else:
        raise ValueError(f"Unknown native graph policy {kind}")
    if any(not np.isfinite(score) or score <= 0 for _, _, score in edges):
        return failed("invalid_native_pair_score")
    factors = gtsam.BetweenFactorPose3s()
    uncertainty = []
    for i, j, score in edges:
        result = outputs[(i, j)]
        r = result["estimated_rotat"].detach().cpu().numpy()
        t = result["estimated_trans"].detach().cpu().numpy()
        if not np.isfinite(r).all() or not np.isfinite(t).all():
            return failed("nonfinite_native_pair_pose")
        factors.append(gtsam.BetweenFactorPose3(
            i, j, gtsam.Pose3(gtsam.Rot3(r), t),
            gtsam.noiseModel.Diagonal.Information(np.eye(6) / score)))
        uncertainty.append(1.0 / score)
    params = gtsam.ShonanAveragingParameters3(gtsam.LevenbergMarquardtParams.CeresDefaults())
    try:
        solver = gtsam.ShonanAveraging3(factors, params)
        initial = gtsam.Values()
        for i in range(count):
            r = np.eye(3) if i == anchor else outputs[(anchor, i)]["estimated_rotat"].detach().cpu().numpy()
            initial.insert(i, gtsam.Rot3(r))
        absolute, _ = solver.run(initial, 3, 40)
        reference = absolute.atRot3(anchor)
        relative = gtsam.Values()
        for i in range(count):
            relative.insert(i, reference.inverse().compose(absolute.atRot3(i)))
        poses = translation_solver(factors, relative, np.asarray(uncertainty), anchor)
        # Algebraically the same export as native _multi_part_assemble after
        # its r.rotate(t_anchor+t_i) translation conversion.
        r = np.stack([relative.atRot3(i).inverse().matrix() for i in range(count)])
        t = np.stack([poses.atPose3(anchor).translation() + poses.atPose3(i).translation()
                      for i in range(count)])
        return pose_result(r, t, anchor, graph_policy=kind, graph_fallback=False,
                           graph_initialization="predicted_anchor_pairs", edges=len(edges))
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        rotations, translations = [], []
        for i in range(count):
            if i == anchor:
                r, t = np.eye(3), np.zeros(3)
            else:
                pair = outputs[(anchor, i)]
                try:
                    r, t = native_pair_to_standard(pair["estimated_rotat"].detach().cpu().numpy(),
                                                   pair["estimated_trans"].detach().cpu().numpy())
                except np.linalg.LinAlgError:
                    return failed("native_graph_and_anchor_fallback_failed", error=str(exc))
            rotations.append(r); translations.append(t)
        return pose_result(np.stack(rotations), np.stack(translations), anchor,
                           graph_policy=kind, graph_fallback=True, graph_error=str(exc))
