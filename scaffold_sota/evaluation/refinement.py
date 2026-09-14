"""Common frozen-output refinement; only input geometry and predicted masks.

The sparse field is interpolated from the same 512-token representation in all
conditions, including the explicit GT diagnostic. This is approximate field
guidance, not exact collision checking or a substitute for contact estimation.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from ..data import sanitize_input
from ..geometry import as_numpy, reference_gauge, transform_points, validate_poses


def _field_values(xyz, tokens, neighbors=8):
    valid = tokens["valid"].bool() & torch.isfinite(tokens["distance"]) & torch.isfinite(tokens["log_scale"])
    if int(valid.sum()) < 4:
        raise ValueError("Insufficient valid field tokens")
    query = tokens["query_xyz"][valid]
    distance = torch.cdist(xyz.reshape(-1, 3), query)
    nearest, indices = distance.topk(min(neighbors, len(query)), largest=False)
    weights = nearest.clamp_min(1e-4).reciprocal().square()
    weights = weights/weights.sum(-1, keepdim=True)
    return {key: (weights*tokens[key][valid][indices]).sum(-1).reshape(xyz.shape[:-1])
            for key in ("distance", "log_scale")}


def _rotation_delta(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    return torch.matrix_exp(skew)


def refine_prediction(sample, prediction, prior=None, *, condition="A3", steps=25,
                      learning_rate=.01, exterior_probabilities=None,
                      trust_translation=.05, trust_rotation_deg=10.,
                      contact_threshold=.03, contact_weight=1., field_weight=1.,
                      trust_weight=.1, constant_sigma=.02, points_per_fragment=128,
                      device="cpu"):
    """Refine rigid poses under a trust region, retaining a common baseline.

    A0 returns the native result. A1 uses the identical contact/trust objective
    without shape guidance. Other A arms receive the caller-selected provider;
    A3 uses spatial uncertainty, while A2/A4/A5/A6/A7 use constant uncertainty.
    No source mesh, true pose, fracture label or target is read by this function.
    The A5 provider itself is explicitly diagnostic and may read its oracle.
    """
    if condition not in {f"A{i}" for i in range(8)}:
        raise ValueError("Expected a declared A0–A7 refinement condition")
    output = copy.deepcopy(prediction)
    if condition == "A0":
        return output
    if prediction.get("rotations") is None:
        output["refinement"] = {"condition": condition, "status": "skipped", "skip_reason": "no_initial_pose", "steps": 0}
        return output
    if steps < 0 or learning_rate <= 0 or trust_translation <= 0 or trust_rotation_deg <= 0 or constant_sigma <= 0:
        raise ValueError("Invalid refinement budget/trust region")
    observed = sanitize_input(sample, device)
    mask = observed["fragment_mask"]
    ids = mask.nonzero().flatten()
    rotations, translations = as_numpy(prediction["rotations"]), as_numpy(prediction["translations"])
    if len(rotations) == len(mask):
        rotations, translations = rotations[as_numpy(mask)], translations[as_numpy(mask)]
    rotations, translations = validate_poses(rotations, translations, len(ids))
    anchor = int((ids == observed["anchor_index"]).nonzero()[0])
    rotations, translations = reference_gauge(rotations, translations, anchor)
    indices = torch.linspace(0, observed["points"].shape[1]-1, min(points_per_fragment, observed["points"].shape[1]), device=device).long()
    points = observed["points"][mask][:, indices]
    r0 = torch.as_tensor(rotations, dtype=points.dtype, device=device)
    t0 = torch.as_tensor(translations, dtype=points.dtype, device=device)
    initial = points @ r0.transpose(-1, -2) + t0[:, None]
    # A declared common contact estimator: nearest pairs in the INITIAL assembly.
    # Freeze its candidate correspondences across arms; contact loss seeks closure.
    contacts = []
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            distances = torch.cdist(initial[i], initial[j])
            nearest, target = distances.min(-1)
            source = (nearest < contact_threshold).nonzero().flatten()
            if len(source):
                contacts.append((i, j, source, target[source]))
    tokens, exterior = None, None
    if condition != "A1":
        if prior is None:
            raise ValueError("Scaffold condition requires a prior provider")
        if prior.provenance.get("diagnostic_only", False) != (condition == "A5"):
            raise ValueError("GT geometry is only legal in the explicit A5 diagnostic")
        exterior_probabilities = prediction.get("exterior_probabilities") if exterior_probabilities is None else exterior_probabilities
        if exterior_probabilities is None and prediction.get("fracture_logits") is not None:
            exterior_probabilities = 1-torch.as_tensor(prediction["fracture_logits"]).sigmoid()
        if exterior_probabilities is None:
            output["refinement"] = {"condition": condition, "status": "skipped", "skip_reason": "missing_predicted_exterior_probabilities", "steps": 0}
            return output
        exterior = torch.as_tensor(exterior_probabilities, dtype=points.dtype, device=device)
        if exterior.shape == observed["points"].shape[:2]:
            exterior = exterior[mask]
        if exterior.shape != (len(ids), observed["points"].shape[1]) or not torch.isfinite(exterior).all() or (exterior < 0).any() or (exterior > 1).any():
            raise ValueError("Predicted exterior probabilities must be finite [F,N] values in [0,1]")
        exterior = exterior[:, indices]
        tokens = {key: value.detach().to(device) for key, value in prior.sample_tokens(sample).items() if isinstance(value, torch.Tensor)}
        if int((tokens["valid"].bool() & torch.isfinite(tokens["distance"]) & torch.isfinite(tokens["log_scale"])).sum()) < 4:
            output["refinement"] = {"condition": condition, "status": "skipped", "skip_reason": "insufficient_valid_field_support", "steps": 0}
            return output
    moving = torch.ones((len(ids), 1), device=device)
    moving[anchor] = 0
    translation = torch.zeros_like(t0, requires_grad=True)
    rotation = torch.zeros_like(t0, requires_grad=True)
    optimizer = torch.optim.Adam([translation, rotation], lr=learning_rate)
    def objective():
        # Normalize tanh vectors to enforce a Euclidean (not per-axis) bound.
        dt = translation.tanh()
        dt = dt/dt.norm(dim=-1, keepdim=True).clamp_min(1)*trust_translation*moving
        dr = rotation.tanh()
        dr = dr/dr.norm(dim=-1, keepdim=True).clamp_min(1)*math.radians(trust_rotation_deg)*moving
        matrices = _rotation_delta(dr) @ r0
        shifts = t0+dt
        xyz = points @ matrices.transpose(-1, -2)+shifts[:, None]
        loss = trust_weight*((dt.square().sum(-1)/trust_translation**2 + dr.square().sum(-1)/math.radians(trust_rotation_deg)**2)*moving[:, 0]).mean()
        if contacts:
            loss = loss+contact_weight*torch.stack([(xyz[i, a]-xyz[j, b]).square().sum(-1).mean() for i,j,a,b in contacts]).mean()
        if tokens is not None:
            values = _field_values(xyz, tokens)
            weight = exterior.clone()
            if condition == "A3":
                confidence = (-values["log_scale"].clamp(-12., 3.)).exp()
                # Equalize total field mass, isolating relative confidence.
                confidence = confidence/(confidence*exterior).sum().clamp_min(1e-8)*exterior.sum().clamp_min(1e-8)
                weight = weight*confidence
            error = F.huber_loss(values["distance"], torch.zeros_like(values["distance"]), reduction="none", delta=constant_sigma)
            loss = loss+field_weight*(error*weight).sum()/weight.sum().clamp_min(1e-8)
        return loss, matrices, shifts
    with torch.no_grad():
        start, best_r, best_t = objective()
        best_loss = float(start)
        best_r, best_t = best_r.clone(), best_t.clone()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = objective()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite refinement objective")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            new_loss, r, t = objective()
            if float(new_loss) < best_loss:
                best_loss, best_r, best_t = float(new_loss), r.clone(), t.clone()
    output.update(rotations=as_numpy(best_r), translations=as_numpy(best_t),
                  refinement={"condition": condition, "status": "completed", "steps": steps, "initial_objective": float(start),
                              "final_objective": best_loss, "contact_estimator": "frozen_initial_nearest_pairs",
                              "contact_pairs": sum(len(entry[2]) for entry in contacts),
                              "trust_translation": trust_translation, "trust_rotation_deg": trust_rotation_deg,
                              "field_representation": "fixed_tokens_inverse_distance_interpolation"})
    return output


def rank_candidates(sample, candidates, prior, *, exterior_probabilities, uncertainty=True, device="cpu"):
    """Rank one fixed candidate pool by field compatibility without oracle poses."""
    observed = sanitize_input(sample, device)
    mask = observed["fragment_mask"]
    ids = mask.nonzero().flatten()
    exterior = torch.as_tensor(exterior_probabilities, device=device, dtype=torch.float32)
    if exterior.shape == observed["points"].shape[:2]:
        exterior = exterior[mask]
    if exterior.shape != observed["points"][mask].shape[:2]:
        raise ValueError("Exterior prediction shape differs from observations")
    if not torch.isfinite(exterior).all() or (exterior < 0).any() or (exterior > 1).any():
        raise ValueError("Exterior predictions must be finite probabilities")
    tokens = {key: value.detach().to(device) for key, value in prior.sample_tokens(sample).items() if isinstance(value, torch.Tensor)}
    anchor = int((ids == observed["anchor_index"]).nonzero()[0])
    scores = []
    for candidate in candidates:
        try:
            r, t = as_numpy(candidate["rotations"]), as_numpy(candidate["translations"])
            if len(r) == len(mask):
                r, t = r[as_numpy(mask)], t[as_numpy(mask)]
            r, t = validate_poses(r, t, len(ids))
            r, t = reference_gauge(r, t, anchor)
            xyz = torch.as_tensor(transform_points(as_numpy(observed["points"][mask]), r, t), device=device, dtype=torch.float32)
            values = _field_values(xyz, tokens)
            weights = exterior * ((-values["log_scale"]).exp() if uncertainty else 1.)
            score = float((values["distance"].abs()*weights).sum()/weights.sum().clamp_min(1e-8))
            scores.append(score if np.isfinite(score) else float("inf"))
        except (ValueError, KeyError, TypeError):
            scores.append(float("inf"))
    finite = np.isfinite(scores)
    return {"selected_index": int(np.argmin(scores)) if finite.any() else None,
            "scores": [value if np.isfinite(value) else None for value in scores], "candidate_count": len(candidates)}
