"""Content interventions preserve the same shape-independent token locations."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from ..data import observation_fingerprint
from ..geometry import as_numpy, fixed_query_bank


class GroundTruthPrior:
    """Explicit diagnostic oracle; never available through ordinary inference."""
    def __init__(self, *, diagnostic_only=False, query_count=512, extent=1.5, sigma=.02, device="cpu", truncation=.1):
        if not diagnostic_only:
            raise ValueError("GroundTruthPrior requires explicit diagnostic_only=True")
        if sigma <= 0:
            raise ValueError("Oracle sigma must be a positive independent constant")
        self.queries = fixed_query_bank(query_count, extent, device=device)
        self.sigma = float(sigma)
        self.truncation = float(truncation)
        if self.truncation <= 0:
            raise ValueError("Oracle truncation must be positive")
        self.provenance = {"kind": "ground_truth", "diagnostic_only": True,
                           "constant_sigma": sigma, "truncation": self.truncation,
                           "support": "fixed-multiscale-sobol-v1"}

    def field(self, sample, query_xyz):
        from .._observations import make_oracle_field
        values = make_oracle_field(sample)(as_numpy(query_xyz))
        distance = torch.as_tensor(values, dtype=self.queries.dtype, device=self.queries.device)
        return {"distance": distance.clamp(-self.truncation, self.truncation),
                "log_scale": torch.full_like(distance, math.log(self.sigma)),
                "valid": torch.isfinite(distance)}

    def sample_tokens(self, sample):
        from .frozen import sha256_file
        return {"query_xyz": self.queries.clone(), **self.field(sample, self.queries),
                "provenance": {**self.provenance, "source_mesh_sha256": sha256_file(sample["source_mesh_path"]),
                               "observation_id": observation_fingerprint(sample),
                               "anchor_source_rotation": as_numpy(sample["anchor_source_rotation"]).tolist(),
                               "anchor_source_centroid": as_numpy(sample["anchor_source_centroid"]).tolist(),
                               "shared_scale": float(as_numpy(sample["shared_scale"])),
                               "coordinate_frame": "normalized_anchor"}}


class WrongPrior:
    """Another source's predicted field queried on the unchanged shared bank."""
    def __init__(self, provider, donor):
        if "source_id" not in donor:
            raise ValueError("Wrong prior requires a known donor identity")
        self.provider, self.donor = provider, donor
        self.queries = provider.queries
        self.provenance = {**provider.provenance, "kind": "wrong_source", "donor_source_id": donor["source_id"],
                           "donor_observation_id": observation_fingerprint(donor)}

    def sample_tokens(self, sample):
        if sample.get("source_id") == self.donor["source_id"]:
            raise ValueError("Wrong prior donor must be a different source object")
        output = self.provider.sample_tokens(self.donor)
        return {**output, "query_xyz": self.queries.clone(),
                "provenance": {**self.provenance, "observation_id": observation_fingerprint(sample)}}


class GenericPrior:
    """A frozen field mean derived only from declared training observations."""
    def __init__(self, tokens, source_ids, training_fingerprint, sigma=.02):
        if not tokens or sigma <= 0 or not source_ids or not training_fingerprint:
            raise ValueError("Generic scaffold needs training-only tokens and provenance")
        query = tokens[0]["query_xyz"].detach().clone()
        if any(not torch.equal(query, item["query_xyz"]) for item in tokens):
            raise ValueError("Generic fields must share the exact spatial support")
        valid = torch.stack([item["valid"].bool() for item in tokens])
        values = torch.stack([item["distance"] for item in tokens])
        weights = valid.to(values.dtype)
        distance = (values.masked_fill(~valid, 0) * weights).sum(0) / weights.sum(0).clamp_min(1)
        self.queries = query
        self.tokens = {"distance": distance, "valid": valid.any(0),
                       "log_scale": torch.full_like(distance, math.log(sigma))}
        self.provenance = {"kind": "training_generic", "training_fingerprint": training_fingerprint,
                           "training_source_ids": sorted(set(source_ids)), "constant_sigma": sigma,
                           "content_sha256": hashlib.sha256(distance.cpu().numpy().tobytes()).hexdigest()}

    @classmethod
    def from_dataset(cls, provider, dataset, indices=None, sigma=.02):
        if dataset.split != "train":
            raise ValueError("Generic prior may only be constructed from training observations")
        indices = list(range(len(dataset))) if indices is None else list(indices)
        total, count, query, source_ids = None, None, None, []
        observations = []
        for index in indices:
            sample = dataset[index]
            if sample["split"] != "train":
                raise ValueError("Generic scaffold includes held-out data")
            if sample["source_id"] in source_ids:
                continue  # One deterministic donor per source gives equal source weight.
            item = provider.sample_tokens(sample)
            if query is None:
                query = item["query_xyz"].detach().clone()
                total, count = torch.zeros_like(item["distance"]), torch.zeros_like(item["distance"])
            if not torch.equal(query, item["query_xyz"]):
                raise ValueError("Generic fields must share exact spatial support")
            valid = item["valid"].bool() & torch.isfinite(item["distance"])
            total += item["distance"].masked_fill(~valid, 0)
            count += valid.to(count.dtype)
            source_ids.append(sample["source_id"])
            observations.append(observation_fingerprint(sample))
        if query is None:
            raise ValueError("No training donors available for generic scaffold")
        item = {"query_xyz": query, "distance": total/count.clamp_min(1), "valid": count > 0}
        result = cls([item], source_ids, dataset.fingerprint, sigma)
        result.provenance.update(provider=provider.provenance, donor_observation_ids=observations,
                                 aggregation="one_fixed_observation_per_training_source", streaming=True)
        return result

    def sample_tokens(self, sample):
        return {"query_xyz": self.queries.clone(), **{key: value.clone() for key, value in self.tokens.items()},
                "provenance": {**self.provenance, "observation_id": observation_fingerprint(sample)}}


class PerturbedPrior:
    """Stress only field content, preserving query support and uncertainty."""
    def __init__(self, provider, mode="distance_noise", magnitude=.01, seed=42):
        if mode not in ("distance_noise", "distance_bias", "shuffled_distance", "sign_flip"):
            raise ValueError("Unknown prior stress control")
        if magnitude < 0:
            raise ValueError("Perturbation magnitude must be nonnegative")
        self.provider, self.mode, self.magnitude, self.seed = provider, mode, float(magnitude), int(seed)
        self.queries = provider.queries
        self.provenance = {**provider.provenance, "stress_control": mode, "magnitude": magnitude, "seed": seed}

    def sample_tokens(self, sample):
        output = self.provider.sample_tokens(sample)
        result = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in output.items()}
        distance = result["distance"]
        valid = result["valid"].bool()
        salt = int(observation_fingerprint(sample)[:8], 16)
        generator = torch.Generator(device="cpu").manual_seed((self.seed+salt) % (2**63-1))
        if self.mode == "distance_noise":
            noise = torch.randn(distance.shape, generator=generator).to(distance.device)*self.magnitude
            distance[valid] += noise[valid]
        elif self.mode == "distance_bias":
            distance[valid] += self.magnitude
        elif self.mode == "shuffled_distance":
            ids = valid.nonzero().flatten()
            permutation = torch.randperm(len(ids), generator=generator).to(ids.device)
            distance[ids] = distance[ids[permutation]].clone()
        else:
            distance[valid] *= -1
        result["provenance"] = {**output.get("provenance", {}), **self.provenance,
                                "observation_id": observation_fingerprint(sample)}
        from .cache import token_fingerprint
        result["provenance"].update(parent_token_sha256=output.get("provenance", {}).get("token_sha256"),
                                    token_sha256=token_fingerprint(result))
        return result


def select_control(provider, name="predicted", *, donor=None, generic=None, sigma=.02,
                   magnitude=.01, seed=42, diagnostic_only=False):
    """Resolve explicit evaluation substitutions without changing recipient weights."""
    if name == "predicted":
        return provider
    if name in ("constant_uncertainty", "shuffled_uncertainty", "null", "global"):
        return ContentControl(provider, name, sigma=sigma, seed=seed)
    if name in ("distance_noise", "distance_bias", "shuffled_distance", "sign_flip"):
        return PerturbedPrior(provider, name, magnitude=magnitude, seed=seed)
    if name == "wrong":
        if donor is None:
            raise ValueError("Wrong-field substitution requires a declared different-source donor")
        return WrongPrior(provider, donor)
    if name == "generic":
        if generic is None:
            raise ValueError("Generic-field substitution requires a training-derived provider")
        return generic
    if name == "ground_truth":
        return GroundTruthPrior(diagnostic_only=diagnostic_only, query_count=len(provider.queries),
                                extent=provider.provenance.get("query_extent", 1.5), sigma=sigma,
                                device=provider.queries.device)
    raise ValueError("Unknown learned-evaluation prior substitution")


class ContentControl:
    """Independent content ablations, retaining geometry support and validity."""
    def __init__(self, provider, mode="constant_uncertainty", sigma=.02, seed=42):
        if mode not in ("constant_uncertainty", "shuffled_uncertainty", "null", "global"):
            raise ValueError("Unknown field content control")
        if sigma <= 0:
            raise ValueError("Constant sigma must be positive")
        self.provider, self.mode, self.sigma, self.seed = provider, mode, sigma, seed
        self.queries = provider.queries
        self.provenance = {**provider.provenance, "content_control": mode, "constant_sigma": sigma}

    def sample_tokens(self, sample):
        output = self.provider.sample_tokens(sample)
        result = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in output.items()}
        if self.mode == "constant_uncertainty":
            result["log_scale"].fill_(math.log(self.sigma))
        elif self.mode == "shuffled_uncertainty":
            # Permute valid positions only, exactly preserving effective mass.
            ids = result["valid"].nonzero().flatten()
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            permutation = torch.randperm(len(ids), generator=generator).to(ids.device)
            result["log_scale"][ids] = result["log_scale"][ids[permutation]].clone()
        elif self.mode == "null":
            result["distance"].zero_()
            result["log_scale"].fill_(math.log(self.sigma))
        elif self.mode == "global":
            for key in ("distance", "log_scale"):
                valid = result["valid"]
                if valid.any():
                    result[key].fill_(float(result[key][valid].mean()))
        result["provenance"] = {**output.get("provenance", {}), **self.provenance}
        from .cache import token_fingerprint
        result["provenance"].update(parent_token_sha256=output.get("provenance", {}).get("token_sha256"),
                                    token_sha256=token_fingerprint(result))
        return result
