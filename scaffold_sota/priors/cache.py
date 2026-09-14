"""Bounded, hash-verified field token caches with exact observation pairing."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from ..data import observation_fingerprint
from ..io import contained, digest_json, read_json, sha256_file, write_json


CACHE_SCHEMA = "scaffold-sota-field-token-cache-v1"


class TokenSnapshotPrior:
    """One immutable in-memory field replay, bound to its recipient observation."""
    def __init__(self, tokens, sample):
        self.tokens = {key: value.detach().clone() for key, value in tokens.items() if isinstance(value, torch.Tensor)}
        self.queries = self.tokens["query_xyz"]
        self.observation_id = observation_fingerprint(sample)
        self.provenance = {**tokens.get("provenance", {}), "observation_id": self.observation_id,
                           "token_sha256": token_fingerprint(self.tokens)}

    def sample_tokens(self, sample):
        if observation_fingerprint(sample) != self.observation_id:
            raise ValueError("Field snapshot belongs to different input points")
        return {**{key: value.clone() for key, value in self.tokens.items()}, "provenance": self.provenance}


def token_fingerprint(tokens):
    digest = hashlib.sha256()
    for key in ("query_xyz", "distance", "log_scale", "valid"):
        value = tokens[key].detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def cache_identity(sample):
    identity = {key: str(sample[key]) for key in ("source_id", "pattern_id")}
    identity["observation_id"] = observation_fingerprint(sample)
    if sample.get("observation_id", identity["observation_id"]) != identity["observation_id"]:
        raise ValueError("Declared observation hash does not match actual input points")
    return identity


def export_token_cache(provider, dataset, output, guard=None, include_exterior=True):
    """Stream fixed observations, storing no source meshes, targets or true poses."""
    if dataset.split == "train" and not dataset.fixed:
        raise ValueError("A static token cache requires fixed observations, not step-varying training views")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Token cache requires a fresh output directory")
    output.mkdir(parents=True)
    query = provider.queries.detach().cpu()
    np.savez_compressed(output/"queries.npz", query_xyz=query.numpy())
    document = {"schema": CACHE_SCHEMA, "dataset_fingerprint": dataset.fingerprint,
                "manifest_sha256": dataset.manifest_sha256, "split": dataset.split,
                "provider": provider.provenance, "queries_path": "queries.npz",
                "queries_sha256": sha256_file(output/"queries.npz"), "records": []}
    identities = set()
    for index in range(len(dataset)):
        if guard:
            guard.check(additional_bytes=1024**2)
        sample = dataset[index]
        identity = cache_identity(sample)
        key = digest_json(identity)
        if key in identities:
            raise ValueError("Duplicate cache observation identity")
        identities.add(key)
        tokens = provider.sample_tokens(sample)
        if not torch.equal(tokens["query_xyz"].detach().cpu(), query):
            raise ValueError("Token cache support changed between observations")
        arrays = {name: tokens[name].detach().cpu().numpy() for name in ("distance", "log_scale", "valid")}
        if include_exterior and hasattr(provider, "exterior_probabilities"):
            arrays["exterior_probabilities"] = provider.exterior_probabilities(sample).detach().cpu().numpy()
        path = output/(key+".npz")
        np.savez_compressed(path, **arrays)
        document["records"].append({**identity, "path": path.name, "sha256": sha256_file(path),
                                    "token_sha256": token_fingerprint(tokens), "provenance": tokens.get("provenance", {})})
    document["record_count"] = len(document["records"])
    document["index_fingerprint"] = digest_json(document)
    write_json(output/"cache.json", document)
    return {"path": str(output), "records": len(document["records"]),
            "sha256": sha256_file(output/"cache.json"), "provider": provider.provenance}


class CachedPrior:
    def __init__(self, cache, device="cpu", expected_fingerprint=None, expected_artifact_sha256=None):
        self.root = Path(cache).resolve()
        self.document = read_json(self.root/"cache.json")
        if self.document.get("schema") != CACHE_SCHEMA:
            raise ValueError("Unknown prior token cache schema")
        identity = {key: value for key, value in self.document.items() if key != "index_fingerprint"}
        if digest_json(identity) != self.document.get("index_fingerprint"):
            raise ValueError("Token cache index fingerprint mismatch")
        if expected_fingerprint is not None and self.document["dataset_fingerprint"] != expected_fingerprint:
            raise ValueError("Token cache dataset fingerprint mismatch")
        self.provenance = {**self.document["provider"], "cache_index_sha256": sha256_file(self.root/"cache.json")}
        if expected_fingerprint is not None and (self.provenance.get("training_source_ids") is None or
                self.provenance.get("source_identity_status") != "verified_manifest_training_partition"):
            raise ValueError("Main-study cache requires verified prior training source identities")
        if expected_artifact_sha256 is not None and self.provenance.get("artifact_sha256") != expected_artifact_sha256:
            raise ValueError("Token cache came from a different frozen prior")
        query_path = self._asset(self.document["queries_path"], self.document["queries_sha256"])
        with np.load(query_path, allow_pickle=False) as archive:
            self.queries = torch.as_tensor(archive["query_xyz"].copy(), device=device)
        self.records = {}
        for record in self.document["records"]:
            key = tuple(record[name] for name in ("source_id", "pattern_id", "observation_id"))
            if key in self.records:
                raise ValueError("Duplicate cached observation")
            self.records[key] = record
        if len(self.records) != self.document["record_count"]:
            raise ValueError("Token cache record denominator differs from index")

    def _asset(self, relative, expected_hash):
        path = self.root/relative
        if Path(relative).is_absolute() or not contained(path, self.root):
            raise ValueError("Token cache path escapes cache directory")
        if sha256_file(path) != expected_hash:
            raise ValueError("Token cache asset SHA-256 mismatch")
        return path

    def _load(self, sample):
        identity = cache_identity(sample)
        key = tuple(identity[name] for name in ("source_id", "pattern_id", "observation_id"))
        if key not in self.records:
            raise KeyError("No tokens for these exact input points; regenerate this observation's prior")
        record = self.records[key]
        path = self._asset(record["path"], record["sha256"])
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: torch.as_tensor(archive[key].copy(), device=self.queries.device) for key in archive.files}
        return record, arrays

    def sample_tokens(self, sample):
        record, arrays = self._load(sample)
        tokens = {"query_xyz": self.queries.clone(), **{key: arrays[key] for key in ("distance", "log_scale", "valid")}}
        if token_fingerprint(tokens) != record["token_sha256"]:
            raise ValueError("Cached token payload fingerprint mismatch")
        tokens["provenance"] = {**record["provenance"], "cache_index_sha256": self.provenance["cache_index_sha256"]}
        return tokens

    def exterior_probabilities(self, sample):
        _, arrays = self._load(sample)
        if "exterior_probabilities" not in arrays:
            raise ValueError("This cache does not contain the declared common exterior estimator")
        return arrays["exterior_probabilities"]
