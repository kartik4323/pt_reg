"""Input-only baseline proposals, template alignment and contact-preserving refinement.

This deliberately simple geometric baseline is not a reproduction of a learned
fracture matcher. Surface penetration and exterior classification are heuristics.
"""
from __future__ import annotations
import itertools
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


def apply(points, T):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def normals_features(p, k=20):
    idx = cKDTree(p).query(p, k=min(k, len(p)))[1]
    groups = p[idx] - p[idx].mean(1, keepdims=True)
    vals, vec = np.linalg.eigh(np.einsum('nki,nkj->nij', groups, groups))
    n = vec[:, :, 0]
    n *= np.where(np.sum(n * (p-p.mean(0)), 1) < 0, -1, 1)[:, None]
    curve = vals[:, 0] / np.maximum(vals.sum(1), 1e-10)
    exterior = np.clip(curve * 30, 0.1, 0.9)
    return n, exterior


def frame(n):
    n = n / max(np.linalg.norm(n), 1e-10)
    v = np.eye(3)[np.argmin(abs(n))]
    u = np.cross(v, n); u /= np.linalg.norm(u)
    return np.column_stack((u, np.cross(n, u), n))


def kabsch(a, b):
    ac, bc = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a-ac).T @ (b-bc))
    d = np.eye(3); d[2, 2] = np.linalg.det(vt.T @ u.T)
    T = np.eye(4); T[:3, :3] = vt.T @ d @ u.T; T[:3, 3] = bc - T[:3, :3] @ ac
    return T


def export_poses(case, poses):
    out = np.asarray(poses).copy()
    out[:, :3, 3] = case['centers'][case['anchor']] + case['scale'] * poses[:, :3, 3] - np.einsum('fij,fj->fi', poses[:, :3, :3], case['centers'])
    return out


def check_poses(poses, pieces, anchor=None):
    a = np.asarray(poses)
    if a.shape != (pieces, 4, 4) or not np.isfinite(a).all():
        raise ValueError('Invalid pose shape or values')
    if not np.allclose(a[:, 3], [0, 0, 0, 1], atol=1e-5) or not np.allclose(a[:, :3, :3] @ a[:, :3, :3].transpose(0, 2, 1), np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(a[:, :3, :3]), 1, atol=1e-4):
        raise ValueError('Poses must be proper rigid transforms')
    if anchor is not None and not np.allclose(a[anchor], np.eye(4), atol=1e-5):
        raise ValueError('Reference fragment must stay fixed')


def contact_pairs(case, poses, fraction=0.08):
    """Freeze a sparse MST contact graph from input geometry at this candidate."""
    pts = [apply(p, T) for p, T in zip(case['points'], poses)]
    ns = [n @ T[:3, :3].T for n, T in zip(case['normals'], poses)]
    edges = []
    for i, j in itertools.combinations(range(len(pts)), 2):
        dist, ix = cKDTree(pts[j]).query(pts[i])
        normal_penalty = (1 + (ns[i] * ns[j][ix]).sum(1)) / 2
        rank = dist + 0.03 * normal_penalty
        count = min(len(dist), max(3, int(len(dist) * fraction)))
        take = np.argsort(rank)[:count]
        edges.append((float(rank[take].mean()), i, j, take, ix[take]))
    parent = list(range(len(pts)))
    def find(i):
        while parent[i] != i: i = parent[i]
        return i
    selected = []
    for score, i, j, a, b in sorted(edges, key=lambda x: x[0]):
        if find(i) != find(j):
            parent[find(i)] = find(j)
            selected.append((i, j, a, b))
    return selected


def contact_residual(case, poses, edges):
    out = []
    for i, j, a, b in edges:
        d = apply(case['points'][i][a], poses[i]) - apply(case['points'][j][b], poses[j])
        out.extend((d / np.sqrt(len(a))).ravel())
        ni, nj = case['normals'][i][a] @ poses[i, :3, :3].T, case['normals'][j][b] @ poses[j, :3, :3].T
        out.extend((0.01 * (ni + nj) / np.sqrt(len(a))).ravel())
    return np.asarray(out)


def contact_score(case, poses, cfg):
    pairs = contact_pairs(case, poses, cfg['contact_fraction'])
    r = contact_residual(case, poses, pairs)
    return float(np.sqrt(np.sum(r*r) / max(1, len(pairs))))


def penetration_proxy(case, poses):
    values = []
    for i, j in itertools.combinations(range(len(poses)), 2):
        a, b = apply(case['points'][i], poses[i]), apply(case['points'][j], poses[j])
        dist, ids = cKDTree(b).query(a)
        n = case['normals'][j] @ poses[j, :3, :3].T
        signed = ((a-b[ids]) * n[ids]).sum(1)
        values.append(float(np.maximum(-signed[dist < 0.08] - 0.005, 0).mean()) if np.any(dist < 0.08) else 0.)
    return float(np.mean(values)) if values else 0.


def candidates(case, cfg, seed, budget=None):
    rng = np.random.default_rng(seed)
    n, anchor = len(case['points']), case['anchor']
    bank, scores = [], []
    for _ in range(budget or cfg['candidates']):
        T = np.repeat(np.eye(4)[None], n, 0)
        placed = [anchor]
        remaining = list(rng.permutation([i for i in range(n) if i != anchor]))
        for i in remaining:
            j = int(rng.choice(placed))
            p, q = case['points'][i], case['points'][j]
            # Prefer locally planar patches, but retain diverse contacts.
            probs = 1-case['exterior'][i]; probs /= probs.sum()
            a = int(rng.choice(len(p), p=probs))
            options = np.argsort(abs(case['exterior'][j] - case['exterior'][i][a]))[:cfg['patches']]
            b = int(rng.choice(options))
            target_n = case['normals'][j][b] @ T[j, :3, :3].T
            basis = frame(-target_n)
            twist = Rotation.from_rotvec(np.array([0, 0, rng.uniform(-np.pi, np.pi)])).as_matrix()
            R = basis @ twist @ frame(case['normals'][i][a]).T
            T[i, :3, :3] = R
            T[i, :3, 3] = apply(q[b:b+1], T[j])[0] - R @ p[a]
            placed.append(i)
        score = contact_score(case, T, cfg) + penetration_proxy(case, T)
        bank.append(T); scores.append(score)
    order = np.argsort(scores)
    return np.asarray(bank)[order], np.asarray(scores)[order]


def fit_template(template, case, cfg, seed):
    """One global similarity fitted only to observed reference-fragment evidence."""
    template = np.asarray(template, float)
    if len(template) < 16 or not np.isfinite(template).all():
        raise ValueError('Invalid template')
    center = template.mean(0)
    radius = np.linalg.norm(template-center, axis=1).max()
    unit = (template-center) / max(2*radius, 1e-10)
    anchor = case['anchor']
    p, weights = case['points'][anchor], case['exterior'][anchor]
    use = np.argsort(weights)[len(weights)//2:]
    # Independent held-out anchor points are never used by alignment.
    fit, hold = p[use[::2]], p[use[1::2]]
    if len(hold) < 3: hold = p[::2]
    rng = np.random.default_rng(seed)
    rotations = [np.eye(3)] + list(Rotation.random(max(0, cfg['alignment_starts']-1), random_state=rng).as_matrix())
    best = None
    for s in cfg['scales']:
        for R in rotations:
            q = unit @ R.T * s
            T = np.eye(4); T[:3, 3] = fit.mean(0) - q.mean(0)
            for _ in range(cfg['alignment_iterations']):
                aligned = apply(q, T)
                _, ids = cKDTree(aligned).query(fit)
                delta = kabsch(aligned[ids], fit)
                T = delta @ T
            aligned = apply(q, T)
            error = float(cKDTree(aligned).query(fit)[0].mean())
            if best is None or error < best[0]:
                best = error, aligned, s, R, T
    error, q, s, R, T = best
    return q, dict(fit_error=error, heldout_error=float(cKDTree(q).query(hold)[0].mean()),
                    source_center=center.tolist(), source_radius=float(radius), scale=float(s),
                    rotation=(T[:3, :3] @ R).tolist(), translation=T[:3, 3].tolist(),
                    uses_ground_truth_alignment=False)


def exterior_score(case, poses, template):
    tree = cKDTree(template)
    return float(np.mean([np.average(tree.query(apply(p, T))[0], weights=e) for p, T, e in zip(case['points'], poses, case['exterior'])]))


def solve(case, bank, cfg, template=None, policy='gated', refine=False, alignment_error=None):
    scores = np.array([contact_score(case, T, cfg) + penetration_proxy(case, T) for T in bank])
    base = int(np.argmin(scores))
    weight = cfg['template_weight'] if template is not None else 0.
    if policy == 'weak': weight *= 0.1
    accepted = template is not None
    ext = np.zeros(len(bank))
    if template is not None:
        ext = np.array([exterior_score(case, T, template) for T in bank])
        chosen = int(np.argmin(scores + weight*ext))
        if policy == 'gated':
            accepted = bool(ext[chosen] <= cfg['gate_exterior'] and scores[chosen] <= cfg['gate_contact_ratio'] * max(scores[base], 1e-5)
                            and (alignment_error is None or alignment_error <= cfg['gate_exterior']))
            if not accepted: weight = 0.
    chosen = int(np.argmin(scores + weight*ext))
    poses = bank[chosen].copy()
    before = contact_score(case, poses, cfg)
    refined = False
    if refine:
        edges = contact_pairs(case, poses, cfg['contact_fraction'])
        moving = [i for i in range(len(poses)) if i != case['anchor']]
        initial = poses.copy()
        tree = cKDTree(template) if template is not None and weight else None
        def decode(v):
            out = initial.copy()
            for k, i in enumerate(moving):
                delta = v[k*6:k*6+6]
                out[i, :3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix() @ initial[i, :3, :3]
                out[i, :3, 3] += delta[3:]
            return out
        def residual(v):
            T = decode(v)
            pieces = [contact_residual(case, T, edges), 0.02*v]
            if tree:
                for i in moving:
                    dist = tree.query(apply(case['points'][i], T[i]))[0]
                    pieces.append(np.sqrt(weight*case['exterior'][i]/len(dist))*np.minimum(dist, 0.2))
            # Signed-surface heuristic, explicitly not a solid collision test.
            pieces.append(np.array([penetration_proxy(case, T)]))
            return np.concatenate(pieces)
        opt = least_squares(residual, np.zeros(6*len(moving)), bounds=(-0.25, 0.25),
                            max_nfev=cfg['refine_evaluations'], loss='soft_l1', f_scale=0.03)
        proposal = decode(opt.x)
        if policy != 'gated' or contact_score(case, proposal, cfg) <= max(before*cfg['gate_contact_ratio'], 1e-5):
            poses, refined = proposal, True
    check_poses(poses, len(poses), case['anchor'])
    return poses, dict(candidate_index=chosen, base_candidate_index=base, template_accepted=accepted,
                       refined=refined, contact_before=before, contact_after=contact_score(case, poses, cfg),
                       exterior_error=float(ext[chosen]) if template is not None else None,
                       penetration_proxy=penetration_proxy(case, poses), solid_collision_measured=False,
                       confidence_kind='heuristic_not_calibrated_probability')
