"""Versioned fracture datasets and the reference-frame training contract."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .prepare import _file_sha256, _seed, manifest_fingerprint, random_rotation, signed_distance


def _contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest contains a path outside the prepared dataset")
    return path


def verify_manifest(manifest: str | Path) -> dict:
    """Verify immutable dataset contents once before profiling/training.

    Hash files in bounded chunks, check paths after symlink resolution, and bind
    their hashes to the exact source/pattern assignments and preparation config.
    Ordinary minibatch reads deliberately do not repeat this complete scan.
    """
    manifest_path = Path(manifest).resolve()
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 2 or document.get("sdf_convention") != "negative_inside":
        raise ValueError("Manifest requires schema_version=2 and negative-inside SDF")
    root = manifest_path.parent
    seen_paths,source_ids,pattern_ids = set(),set(),set()
    total_bytes = 0
    try:
        for group in ("sources","patterns"):
            for record in document[group]:
                relative = Path(record["path"])
                if relative.is_absolute():
                    raise ValueError("Manifest paths must be relative to the prepared dataset")
                path = _contained(root,str(relative))
                if path in seen_paths:
                    raise ValueError(f"Manifest references duplicate asset path: {relative}")
                seen_paths.add(path)
                if not path.is_file():
                    raise ValueError(f"Missing prepared asset: {relative}")
                expected = record.get("sha256","")
                if len(expected) != 64 or _file_sha256(path) != expected:
                    raise ValueError(f"Prepared asset SHA-256 mismatch: {relative}")
                total_bytes += path.stat().st_size
                if group == "sources":
                    if record["source_id"] in source_ids:
                        raise ValueError("Duplicate source identity in manifest")
                    source_ids.add(record["source_id"])
                else:
                    if record["source_id"] not in source_ids:
                        raise ValueError("Pattern references unknown source identity")
                    if record["pattern_id"] in pattern_ids:
                        raise ValueError("Duplicate pattern identity in manifest")
                    pattern_ids.add(record["pattern_id"])
        fingerprint = manifest_fingerprint(document)
    except (KeyError,TypeError) as exc:
        raise ValueError(f"Malformed prepared manifest: {type(exc).__name__}") from exc
    if fingerprint != document.get("fingerprint"):
        raise ValueError("Dataset fingerprint mismatch: assignments or preparation metadata changed")
    return {"status":"verified","dataset_fingerprint":fingerprint,
            "verified_sources":len(source_ids),"verified_patterns":len(pattern_ids),"verified_bytes":total_bytes}


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def make_oracle_field(sample: dict):
    """Evaluation-only exact source-mesh field; never a model inference input.

    The returned callable receives queries in the normalized reference frame.
    The mesh retains modeled cavities. Querying is chunked to bound CPU memory.
    """
    import trimesh
    with np.load(sample["source_mesh_path"],allow_pickle=False) as source:
        mesh = trimesh.Trimesh(source["vertices"],source["faces"],process=False)
    rotation = _numpy(sample["anchor_source_rotation"]).reshape(3,3)
    center = _numpy(sample["anchor_source_centroid"]).reshape(3)
    scale = float(_numpy(sample["shared_scale"]))
    def evaluate(queries):
        source_queries = (np.asarray(queries)*scale) @ rotation + center
        return signed_distance(mesh,source_queries,chunk_size=512)/scale
    return evaluate


class FractureDataset(Dataset):
    """Complete 2–3 piece XYZ sets, independently posed on every training step.

    ``set_step`` is called by the trainer before each optimizer step. Evaluation
    and ``fixed=True`` are reproducible regardless of traversal or worker order.
    No original meshes, contact labels, or ground-truth fields are consumed by
    the model at inference; they exist here only for supervision/evaluation.
    """
    def __init__(self, manifest: str | Path, split: str, cfg: dict,
                 stage: int = 1, fixed: bool = False, limit: int | None = None):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 2:
            raise ValueError("FractureDataset requires schema_version=2; legacy data is not accepted")
        if self.manifest.get("sdf_convention") != "negative_inside":
            raise ValueError("Expected negative-inside signed-distance supervision")
        self.root = self.manifest_path.parent
        self.fingerprint = self.manifest["fingerprint"]
        self.cfg, self.split, self.stage, self.fixed = cfg, split, stage, fixed
        self.step = 0
        self.epoch = 0
        self.seed = int(cfg.get("data",{}).get("seed",42))
        self.records = [r for r in self.manifest["patterns"] if split == "all" or r["split"] == split]
        # Hash order mixes source IDs and difficulty without depending on disk order.
        self.records.sort(key=lambda r:_seed(self.seed,r["pattern_id"],"order"))
        if limit is not None:
            self.records = self.records[:limit]
        self.source_paths = {r["source_id"]:r.get("path",f"sources/{r['source_id']}.npz")
                             for r in self.manifest["sources"]}
        for record in self.records:
            if record.get("pieces") not in (2,3):
                raise ValueError("Manifest contains an incomplete/unsupported fragment set")
            _contained(self.root,record["path"])
        source_split = {}
        for record in self.manifest["patterns"]:
            effective = "test" if record["split"] == "cut_holdout" else record["split"]
            if record["source_id"] in source_split and source_split[record["source_id"]] != effective:
                raise ValueError("Source object leaks across dataset splits")
            source_split[record["source_id"]] = effective
            if record["cut_family"] == "heldout_radial" and effective != "test":
                raise ValueError("Held-out cut family leaked into training or validation")

    def __len__(self):
        return len(self.records)

    def set_step(self, step: int):
        if step < 0:
            raise ValueError("step must be nonnegative")
        self.step = int(step)
        self.epoch = int(step)

    def _record(self,index):
        if not self.records:
            raise IndexError(f"No prepared patterns in split {self.split}")
        if self.split == "train" and not self.fixed:
            progress = self.step / max(1,int(self.cfg.get("train",{}).get("max_updates",2000)))
            allowed = {"easy"} if progress < .3 else {"easy","intermediate"} if progress < .65 else {"easy","intermediate","hard"}
            pool = [record for record in self.records if record["band"] in allowed]
            if not pool:
                raise ValueError("Training split has no examples for the current curriculum band")
            return pool[index % len(pool)]
        return self.records[index]

    def __getitem__(self,index):
        record = self._record(index)
        stochastic = self.split == "train" and not self.fixed
        rng = np.random.default_rng(_seed(self.seed,record["pattern_id"],self.step if stochastic else "fixed"))
        data = self.cfg.get("data",{})
        n, q = int(data.get("points_per_fragment",1024)), int(data.get("sdf_queries",2048))
        with np.load(_contained(self.root,record["path"]),allow_pickle=False) as archive, \
             np.load(_contained(self.root,self.source_paths[record["source_id"]]),allow_pickle=False) as source_archive:
            reservoirs = archive["points"]
            count = len(reservoirs)
            if count not in (2,3) or reservoirs.shape[1] < n:
                raise ValueError("Prepared reservoirs must contain 2–3 full fragments and enough distinct points")
            selected = [rng.choice(reservoirs.shape[1],n,replace=False) for _ in range(count)]
            source_points = np.stack([reservoirs[i,indices] for i,indices in enumerate(selected)]).astype(np.float64)
            labels = np.stack([archive["fracture_labels"][i,indices] for i,indices in enumerate(selected)])
            interfaces = np.stack([archive["interface_ids"][i,indices] for i,indices in enumerate(selected)])
            rotation = np.stack([random_rotation(rng) for _ in range(count)])
            shift = rng.uniform(-float(data.get("translation_range",.7)),float(data.get("translation_range",.7)),size=(count,3))
            original = np.einsum("fni,fji->fnj",source_points,rotation)+shift[:,None,:]
            # Noise affects only observations; supervision remains the intact source.
            if record["band"] == "hard":
                original += rng.normal(scale=float(data.get("hard_noise_std",.001)),size=original.shape)
            centroids = original.mean(axis=1)
            centered = original-centroids[:,None,:]
            scale = float(np.linalg.norm(centered,axis=-1).max(axis=1).sum())
            if not np.isfinite(scale) or scale <= 1e-10:
                raise ValueError("Input fragments have degenerate shared scale")
            rms = np.sqrt(np.mean(np.sum(centered**2,axis=-1),axis=-1))
            anchor = int(np.argmax(rms))
            # Exact inverse observed anchor transform, including anchor centering.
            source_center = (centroids[anchor]-shift[anchor]) @ rotation[anchor]
            anchor_rotation = rotation[anchor]
            canonical = np.einsum("fni,ji->fnj",source_points-source_center,anchor_rotation)/scale
            gt_rotation = anchor_rotation[None,:,:] @ rotation.transpose(0,2,1)
            gt_translation = np.einsum("fi,fij->fj",centroids-shift,rotation)
            gt_translation = (gt_translation-source_center) @ anchor_rotation.T/scale
            points = centered/scale
            padded_points = np.zeros((3,n,3),dtype=np.float32)
            padded_points[:count] = points
            padded_labels = np.zeros((3,n),dtype=np.float32)
            padded_labels[:count] = labels
            padded_ids = np.full((3,n),-1,dtype=np.int64)
            padded_ids[:count] = interfaces
            padded_canonical = np.zeros((3,n,3),dtype=np.float32)
            padded_canonical[:count] = canonical
            rotations = np.tile(np.eye(3,dtype=np.float32),(3,1,1))
            rotations[:count] = gt_rotation
            translations = np.zeros((3,3),dtype=np.float32)
            translations[:count] = gt_translation
            rotations[anchor], translations[anchor] = np.eye(3),0
            available_near = np.flatnonzero(source_archive["sdf_near_mask"])
            available_uniform = np.flatnonzero(~source_archive["sdf_near_mask"])
            if not len(available_near) or not len(available_uniform):
                raise ValueError("SDF reservoir must include near-surface and surrounding-space samples")
            indices = np.concatenate([rng.choice(available_near,q//2,replace=len(available_near)<q//2),
                                      rng.choice(available_uniform,q-q//2,replace=len(available_uniform)<q-q//2)])
            rng.shuffle(indices)
            queries = (source_archive["sdf_queries"][indices]-source_center) @ anchor_rotation.T/scale
            sdf_values = source_archive["sdf_values"][indices]/scale
            near_mask = source_archive["sdf_near_mask"][indices]
            target = (source_archive["target_points"]-source_center) @ anchor_rotation.T/scale
        view_rotation = np.stack([random_rotation(rng) for _ in range(3)])
        second_view = np.einsum("fni,fji->fnj",padded_points,view_rotation).astype(np.float32)
        tensor = lambda value:torch.from_numpy(np.asarray(value))
        result = {"points":tensor(padded_points),"fragment_mask":tensor(np.arange(3)<count),
                  "fracture_labels":tensor(padded_labels),"interface_ids":tensor(padded_ids),
                  "canonical_points":tensor(padded_canonical),"anchor_index":torch.tensor(anchor,dtype=torch.long),
                  "sdf_queries":tensor(queries.astype(np.float32)),"sdf_values":tensor(sdf_values.astype(np.float32)),
                  "sdf_near_mask":tensor(near_mask.astype(bool)),"target_points":tensor(target.astype(np.float32)),
                  "rotations_gt":tensor(rotations),"translations_gt":tensor(translations),
                  "points_view2":tensor(second_view),"view2_rotations":tensor(view_rotation.astype(np.float32)),
                  "source_id":record["source_id"],"pattern_id":record["pattern_id"],"band":record["band"],
                  "cut_family":record["cut_family"],"source_mesh_path":str(_contained(self.root,self.source_paths[record["source_id"]])),
                  "anchor_source_rotation":tensor(anchor_rotation.astype(np.float32)),
                  "anchor_source_centroid":tensor(source_center.astype(np.float32)),"shared_scale":torch.tensor(scale,dtype=torch.float32)}
        # All metadata are padded so torch default_collate supports 2/3-piece batches.
        original_padded = np.zeros((3,n,3),dtype=np.float32)
        original_padded[:count] = original
        centers_padded = np.zeros((3,3),dtype=np.float32)
        centers_padded[:count] = centroids
        result.update(original_points=tensor(original_padded),centroids=tensor(centers_padded))
        return result

    def oracle_field(self,index:int,query_xyz_normalized_ref):
        if self.split == "train" and not self.fixed:
            raise ValueError("Oracle field is only supported for deterministic evaluation/fixed samples")
        return make_oracle_field(self[index])(query_xyz_normalized_ref)
