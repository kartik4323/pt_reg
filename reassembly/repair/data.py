"""Independent views and balanced fields over immutable v2 prepared geometry."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from reassembly.data import FractureDataset, _contained
from reassembly.prepare import _seed, signed_distance, sample_surface, _file_sha256
from reassembly.resources import write_json


class _ExactRecordDataset(FractureDataset):
    def _record(self, index):
        return self.records[index]


def collate_samples(samples, device):
    result = {}
    for key, first in samples[0].items():
        values = [sample[key] for sample in samples]
        if isinstance(first, torch.Tensor):
            result[key] = torch.stack(values).to(device)
        elif isinstance(first, dict):
            result[key] = collate_samples(values, device)
        else:
            result[key] = values
    return result


def _common_canonical(view, reference):
    """Convert supervision only; view2 model coordinates stay independently posed."""
    canonical = view["canonical_points"].double()
    source = canonical * view["shared_scale"].double()
    source = source @ view["anchor_source_rotation"].double() + view["anchor_source_centroid"].double()
    common = (source - reference["anchor_source_centroid"].double()) @ reference["anchor_source_rotation"].double().T
    common /= reference["shared_scale"].double()
    common[~view["fragment_mask"]] = 0
    return common.float()


def supplement_queries(manifest, output, guard=None):
    """Write a separate provenance-bound cache only for missing signed groups.

    The original source/pattern assets and manifest are never rewritten.
    """
    import trimesh
    from reassembly.data import verify_manifest
    verify_manifest(manifest)
    document = json.loads(Path(manifest).read_text(encoding="utf-8"))
    root, output = Path(manifest).resolve().parent, Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh supplemental-query cache directory")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for record in document["sources"]:
        source_path = _contained(root, record["path"])
        with np.load(source_path, allow_pickle=False) as source:
            near, values = source["sdf_near_mask"], source["sdf_values"]
            missing = [sign for sign in (-1, 1) if not np.any(near & (values * sign > 0))]
            if not missing:
                continue
            mesh = trimesh.Trimesh(source["vertices"], source["faces"], process=False)
            rng = np.random.default_rng(_seed(42, record["source_id"], "repair-field-cache"))
            found = {-1: [], 1: []}
            for attempt in range(20):
                points, _ = sample_surface(mesh, 2048, rng)
                queries = points + rng.normal(0, .01 / (1 + attempt // 5), points.shape)
                distances = signed_distance(mesh, queries, chunk_size=512)
                for sign in missing:
                    selected = distances * sign > 0
                    found[sign].extend(zip(queries[selected], distances[selected]))
                if all(len(found[sign]) >= 512 for sign in missing):
                    break
            if any(not found[sign] for sign in missing):
                raise RuntimeError(f"No signed near-surface support for {record['source_id']}; inspect source geometry")
            chosen = [entry for sign in missing for entry in found[sign][:2048]]
            queries = np.asarray([x for x, _ in chosen], dtype=np.float32)
            distances = np.asarray([d for _, d in chosen], dtype=np.float32)
            path = output / (record["source_id"] + ".npz")
            if guard:
                guard.check(additional_bytes=queries.nbytes + distances.nbytes + 4096)
            np.savez_compressed(path, sdf_queries=queries, sdf_values=distances)
            records.append({"source_id": record["source_id"], "source_sha256": record["sha256"],
                            "path": path.name, "sha256": _file_sha256(path), "count": len(queries)})
    result = {"schema_version": 3, "dataset_fingerprint": document["fingerprint"], "sources": records}
    write_json(output / "query_cache.json", result, guard)
    return result


class RepairDataset(FractureDataset):
    def __init__(self, manifest, split, cfg, stage=1, fixed=False, limit=None,
                 *, balanced_fields=True, views=True, query_cache=None, fixed_geometry=False):
        super().__init__(manifest, split, cfg, stage, fixed, limit)
        self.balanced_fields, self.views = balanced_fields, views
        self.fixed_geometry = fixed_geometry
        self.first = _ExactRecordDataset(manifest, split, cfg, stage, fixed)
        self.second = _ExactRecordDataset(manifest, split, cfg, stage, fixed)
        self.first.records = self.records
        self.second.records = self.records
        self.second.seed = _seed(self.seed, "repair-independent-view")
        self.positions = {r["pattern_id"]: i for i, r in enumerate(self.records)}
        self.cache = {}
        self.query_cache_hash = None
        if query_cache:
            path = Path(query_cache).resolve()
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("schema_version") != 3 or not isinstance(metadata.get("sources"), list):
                raise ValueError("Supplemental query cache requires schema_version3 and source records")
            if metadata["dataset_fingerprint"] != self.fingerprint:
                raise ValueError("Supplemental field queries belong to a different dataset")
            source_records = {r["source_id"]: r for r in self.manifest["sources"]}
            for record in metadata["sources"]:
                if record["source_id"] not in source_records or record["source_id"] in self.cache:
                    raise ValueError("Unknown or duplicate supplemental query source")
                asset = _contained(path.parent, record["path"])
                if record["source_sha256"] != source_records[record["source_id"]]["sha256"] or _file_sha256(asset) != record["sha256"]:
                    raise ValueError("Supplemental query/source hash mismatch")
                with np.load(asset, allow_pickle=False) as extra:
                    q, d = extra["sdf_queries"], extra["sdf_values"]
                    if q.ndim != 2 or q.shape[1] != 3 or d.shape != (len(q),) or len(q) != record["count"] or not len(q):
                        raise ValueError("Invalid supplemental query shape/count")
                    if not np.isfinite(q).all() or not np.isfinite(d).all():
                        raise ValueError("Nonfinite supplemental field query")
                self.cache[record["source_id"]] = asset
            self.query_cache_hash = _file_sha256(path)

    def set_step(self, step):
        super().set_step(step)
        self.first.set_step(step)
        self.second.set_step(step)

    def _record(self, index):
        if self.split != "train" or self.fixed or self.fixed_geometry:
            return self.records[index]
        first, second = self.cfg["repair"]["curriculum_updates"]
        bands = {"easy"} if self.step < first else {"easy", "intermediate"} if self.step < second else {"easy", "intermediate", "hard"}
        pool = [r for r in self.records if r["band"] in bands]
        if not pool:
            raise ValueError("No geometry in the current absolute curriculum band")
        return pool[index % len(pool)]

    def __getitem__(self, index):
        record = self._record(index)
        index = self.positions[record["pattern_id"]]
        first = self.first[index]
        if self.views:
            other = self.second[index]
            other["canonical_points"] = _common_canonical(other, first)
            first["view2"] = {key: other[key] for key in ("points", "fragment_mask", "anchor_index",
                "canonical_points", "fracture_labels", "interface_ids")}
        if self.balanced_fields:
            self.balance_queries(first)
        with np.load(_contained(self.root, record["path"]), allow_pickle=False) as archive:
            if "metadata" in archive:
                first["preparation_metadata"] = json.loads(str(archive["metadata"].item()))
        return first

    def balance_queries(self, sample):
        q = int(self.cfg["data"]["sdf_queries"]) // 4
        stochastic = self.split == "train" and not self.fixed
        rng = np.random.default_rng(_seed(self.seed, sample["pattern_id"], self.step if stochastic else "fixed", "balanced-field"))
        with np.load(sample["source_mesh_path"], allow_pickle=False) as source:
            queries, distances, near = source["sdf_queries"], source["sdf_values"], source["sdf_near_mask"].astype(bool)
            target = source["target_points"]
            if sample["source_id"] in self.cache:
                with np.load(self.cache[sample["source_id"]], allow_pickle=False) as extra:
                    queries = np.concatenate((queries, extra["sdf_queries"]))
                    distances = np.concatenate((distances, extra["sdf_values"]))
                    near = np.concatenate((near, np.ones(len(extra["sdf_values"]), dtype=bool)))
            pieces = [target[rng.choice(len(target), q, replace=len(target) < q)]]
            values = [np.zeros(q)]
            for label, mask in (("near_inside", near & (distances < 0)),
                                ("near_outside", near & (distances > 0)), ("surrounding", ~near)):
                available = np.flatnonzero(mask)
                if not len(available):
                    raise ValueError(f"Source {sample['source_id']} lacks {label} queries; run supplement-queries")
                chosen = rng.choice(available, q, replace=len(available) < q)
                pieces.append(queries[chosen]); values.append(distances[chosen])
        rotation = sample["anchor_source_rotation"].numpy().astype(np.float64)
        center = sample["anchor_source_centroid"].numpy().astype(np.float64)
        scale = float(sample["shared_scale"])
        order = rng.permutation(4 * q)
        sample["sdf_queries"] = torch.from_numpy((((np.concatenate(pieces)-center) @ rotation.T / scale)[order]).astype(np.float32))
        sample["sdf_values"] = torch.from_numpy((np.concatenate(values)[order] / scale).astype(np.float32))
        sample["sdf_query_group"] = torch.from_numpy(np.repeat(np.arange(4), q)[order])
        sample["sdf_near_mask"] = sample["sdf_query_group"] != 3
