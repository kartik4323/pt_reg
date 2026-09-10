"""Small deterministic correctness gate, requiring neither data nor checkpoints."""
from __future__ import annotations

import numpy as np

from .geometry import export_transforms, normalize_fragments, so3_exp, weighted_kabsch
from .solver import ScaffoldGrid, solve_from_matches


def run_correctness_checks() -> dict:
    """Exercise the production solver and coordinate contracts for preflight.

    These are numerical implementation checks, never a claim about learned
    reconstruction or pose accuracy. Errors are returned as failed checks so a
    preflight report can explain a refused pilot without hiding the evidence.
    """
    checks = []

    def record(name, function):
        try:
            measurements = function()
            checks.append({"name": name, "passed": True, "measurements": measurements or {}})
        except Exception as error:
            checks.append({"name": name, "passed": False,
                           "error": f"{type(error).__name__}: {error}"})

    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    def oracle(count):
        rng = np.random.default_rng(410 + count)
        rotations = np.stack([so3_exp(rng.normal(size=3)) for _ in range(count)])
        translations = rng.normal(size=(count, 3)) * .2
        pairs = []
        for i in range(count - 1):
            world = rng.normal(size=(20, 3)) * [.15, .1, .05]
            pairs.append({"i": i, "j": i+1, "source_xyz": (world-translations[i]) @ rotations[i],
                          "target_xyz": (world-translations[i+1]) @ rotations[i+1],
                          "weights": np.eye(len(world))})
        if count == 3:
            # A non-contacting edge must not prevent a valid connected chain.
            pairs.append({"i": 0, "j": 2, "source_xyz": rng.normal(size=(9, 3)),
                          "target_xyz": rng.normal(size=(11, 3)), "weights": np.zeros((9, 11))})
        maximum_error = 0.
        for anchor in range(count):
            result = solve_from_matches(pairs, count, anchor)
            require(result["status"] == "ok", "oracle assembly was not accepted")
            expected_r = np.einsum("ij,fjk->fik", rotations[anchor].T, rotations)
            expected_t = (translations-translations[anchor]) @ rotations[anchor]
            error = max(float(np.abs(result["rotations"]-expected_r).max()),
                        float(np.abs(result["translations"]-expected_t).max()))
            maximum_error = max(maximum_error, error)
            require(error < 1e-5, "oracle transform error exceeds 1e-5")
            require(np.array_equal(result["rotations"][anchor], np.eye(3)), "reference rotation moved")
            require(np.array_equal(result["translations"][anchor], np.zeros(3)), "reference translation moved")
        return {"max_transform_absolute_error": maximum_error, "anchor_choices": count,
                "includes_noncontacting_pair": count == 3}

    def degeneracy():
        rng = np.random.default_rng(712)
        plane = rng.normal(size=(15, 3)) * .1
        plane[:, 2] = 0
        rotation = so3_exp(np.array([.2, .5, -.7]))
        target = plane @ rotation.T + [.1, -.2, .1]
        fit = weighted_kabsch(plane, target)
        require(fit["valid"], "planar non-collinear correspondences rejected")
        require(abs(np.linalg.det(fit["rotation"])-1) < 1e-10, "rotation is not proper")
        line = plane.copy()
        line[:, 1:] = 0
        failed = solve_from_matches([{"i": 0, "j": 1, "source_xyz": line,
                                      "target_xyz": line @ rotation.T, "weights": np.eye(len(line))}], 2)
        require(failed["status"] == "failed" and failed["rotations"] is None,
                "collinear support must fail without identity fallback")
        dustbin = solve_from_matches([{"i": 0, "j": 1, "source_xyz": plane,
                                       "target_xyz": target, "weights": np.eye(len(plane))*1e-8}], 2)
        require(dustbin["status"] == "failed", "tiny true correspondence mass was renormalized into a match")
        return {"planar_rms": fit["rms"], "collinear_status": failed["status"],
                "dustbin_status": dustbin["status"]}

    def exported_frame():
        rng = np.random.default_rng(817)
        world = rng.normal(size=(33, 3))
        rotation, translation = so3_exp(np.array([.2, -.5, .8])), np.array([5., -4., 2.])
        fragments = [world, (world-translation) @ rotation]
        normalized = normalize_fragments(fragments, 20)
        anchor = normalized["anchor_index"]
        local = [(p-c)/normalized["scale"] for p, c in zip(fragments, normalized["centroids"])]
        other = 1-anchor
        fit = weighted_kabsch(local[other], local[anchor])
        r = np.repeat(np.eye(3)[None], 2, axis=0)
        t = np.zeros((2, 3))
        r[other], t[other] = fit["rotation"], fit["translation"]
        exported = export_transforms(r, t, normalized)
        error = float(np.max(np.abs(exported["aligned_fragments"][other]-fragments[anchor])))
        require(error < 1e-5, "exported transforms do not operate on original coordinates")
        require(np.array_equal(exported["transforms"][anchor], np.eye(4)), "exported reference is not identity")
        before = np.linalg.norm(fragments[other][1:]-fragments[other][0], axis=1)
        moved = exported["aligned_fragments"][other]
        after = np.linalg.norm(moved[1:]-moved[0], axis=1)
        rigidity_error = float(np.max(np.abs(before-after)))
        require(rigidity_error < 1e-10, "pose export changed fragment distances")
        scaled = normalize_fragments([p*3 for p in fragments], 20)
        require(np.allclose(normalized["points"], scaled["points"], atol=1e-7), "common scaling changed normalized inputs")
        permuted = normalize_fragments([p[rng.permutation(len(p))] for p in fragments], 20)
        require(np.allclose(normalized["points"], permuted["points"], atol=1e-7), "input point order changed sampled geometry")
        return {"max_original_coordinate_error": error, "max_rigidity_error": rigidity_error,
                "shared_scale": normalized["scale"]}

    def field_contract():
        axis = np.linspace(-1, 1, 8)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        field = ScaffoldGrid(x+2*y-z, np.full_like(x, .01), np.array([[-1]*3, [1]*3]))
        points = np.array([[.2, -.3, .4], [1., .5, -.2], [-.5, 0., .1]])
        distance, confidence, gradient = field.sample(points)
        error = float(np.max(np.abs(distance-points @ np.array([1., 2., -1.]))))
        require(error < 1e-10, "signed trilinear values are incorrect")
        require(np.allclose(gradient, [1., 2., -1.], atol=1e-10), "field gradient is incorrect")
        require(distance.min() < 0 < distance.max(), "field sign was lost")
        outside, outside_confidence, outside_gradient = field.sample(np.array([[2., 0., 0.]]))
        require(outside[0] > 1 and outside_confidence[0] > 0, "out-of-field poses have zero cost")
        require(np.allclose(outside_gradient[0], [1., 0., 0.]), "field boundary gradient is incorrect")
        uncertain = ScaffoldGrid(field.distance, np.full_like(x, .12), field.bounds)
        require(np.all(uncertain.sample(points)[1] < confidence), "uncertainty does not reduce field confidence")
        return {"signed_interpolation_error": error, "outside_distance": float(outside[0])}

    record("two_fragment_oracle", lambda: oracle(2))
    record("three_fragment_oracle_with_noncontacting_edge", lambda: oracle(3))
    record("planar_support_degeneracy_and_dustbin", degeneracy)
    record("original_frame_export_rigidity_normalization", exported_frame)
    record("signed_field_gradient_boundary_uncertainty", field_contract)
    passed = all(check["passed"] for check in checks)
    return {"kind": "correctness", "schema_version": 2, "passed": passed,
            "status": "passed" if passed else "failed", "checks": checks,
            "note": "Deterministic implementation checks; not trained-model performance."}
