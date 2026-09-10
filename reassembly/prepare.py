"""Deterministic complementary procedural fractures with preserved face provenance.

Only conservative cleanup is allowed. Failed source geometry is reported rather
than filled, convexified, or reduced to its largest component.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import trimesh


SCHEMA_VERSION = 2
BANDS = ("easy", "intermediate", "hard")


def _seed(*values) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, values)).encode()).digest()[:8], "little")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_fingerprint(manifest: dict) -> str:
    """Hash identity-bearing metadata using the same canonicalization everywhere."""
    identity = {"schema_version":manifest["schema_version"],"records":manifest["patterns"],
                "sources":manifest["sources"],"seed":manifest["seed"],"config":manifest["config"]}
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def _guard(guard, amount=0):
    if guard is not None:
        guard.check(additional_bytes=int(amount))


def source_splits(source_ids: list[str], seed: int = 42) -> dict[str, str]:
    """Assign entire source objects before fractures, with exact 80/10/10 counts."""
    ids = sorted(set(source_ids), key=lambda value: _seed(seed, value))
    n_train, n_val = int(len(ids) * .8), int(len(ids) * .1)
    return {value: "train" if i < n_train else "val" if i < n_train + n_val else "test"
            for i, value in enumerate(ids)}


def select_complete_breaking_bad_sets(raw_root: Path) -> list[dict]:
    """Read-only transfer-data inventory; select whole 2/3-piece fracture sets.

    This is deliberately not a training adapter: released Breaking Bad geometry
    needs its own intact-surface/contact-supervision validation before converting
    to the v2 schema. Larger sets are skipped, never truncated.
    """
    raw_root = Path(raw_root).resolve()
    result = []
    for pattern in sorted(raw_root.rglob("*")):
        if not pattern.is_dir() or not pattern.name.startswith(("fractured","fracture","mode")):
            continue
        pieces = sorted(pattern.glob("piece_*.obj")) or sorted(pattern.glob("*.obj"))
        if len(pieces) not in (2,3):
            continue
        result.append({"source_id":str(pattern.parent.relative_to(raw_root)),
                       "pattern_id":str(pattern.relative_to(raw_root)),
                       "piece_paths":[str(piece) for piece in pieces],"pieces":len(pieces)})
    return result


def _mesh_members(names: list[str]) -> list[str]:
    names = [n for n in names if Path(n).suffix.lower() in (".obj", ".ply", ".stl", ".off", ".glb")]
    bottles = [n for n in names if "02876657" in Path(n).parts]
    if bottles:
        names = bottles
    # ShapeNet includes multiple representations; never count them as separate objects.
    normalized = [n for n in names if Path(n).name == "model_normalized.obj"]
    if normalized:
        return sorted(normalized)
    originals = [n for n in names if Path(n).name == "model.obj"]
    return sorted(originals or names)


def _source_id(member: str) -> str:
    path = Path(member)
    if path.name in ("model_normalized.obj", "model.obj"):
        return path.parent.parent.name if path.parent.name == "models" else path.parent.name
    return path.stem + "_" + hashlib.sha256(member.encode()).hexdigest()[:10]


def iter_source_meshes(source: Path, max_sources: int = 100):
    """Yield one mesh at a time without extracting archives or resolving textures."""
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            for member in _mesh_members(archive.namelist())[:max_sources]:
                try:
                    with archive.open(member) as stream:
                        mesh = trimesh.load(io.BytesIO(stream.read()), file_type=Path(member).suffix[1:],
                                            process=False, force="mesh", skip_materials=True)
                    yield _source_id(member), mesh, member, None
                except Exception as exc:
                    yield _source_id(member), None, member, f"load_{type(exc).__name__}"
    else:
        members = _mesh_members([str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()]) if source.is_dir() else [source.name]
        for member in members[:max_sources]:
            path = source / member if source.is_dir() else source
            try:
                mesh = trimesh.load(path, process=False, force="mesh", skip_materials=True)
                yield _source_id(member), mesh, member, None
            except Exception as exc:
                yield _source_id(member), None, member, f"load_{type(exc).__name__}"


def validate_source(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict]:
    """Conservatively clean topology and reject invalid/disconnected source solids."""
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) < 4:
        raise ValueError("not_a_triangle_mesh")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("nonfinite_vertices")
    mesh = mesh.copy()
    original_vertices, original_faces = len(mesh.vertices), len(mesh.faces)
    # OBJ material/normal seams can store coincident triangles with distinct
    # indices. Merge positions before finding duplicate or degenerate faces.
    # Doing this in the opposite order leaves double-sided ShapeNet faces and
    # incorrectly classifies their edges as non-manifold.
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        raise ValueError("not_watertight")
    if len(mesh.split(only_watertight=False)) != 1:
        raise ValueError("disconnected_source")
    # Choosing one copy of a double-sided triangle can leave arbitrary winding.
    # Orient the existing closed, connected surface without adding faces,
    # moving vertices, filling cavities, or retaining a selected component.
    before_orientation = mesh.faces.copy()
    mesh.fix_normals(multibody=False)
    flipped_faces = int(np.any(mesh.faces != before_orientation, axis=1).sum())
    if not mesh.is_winding_consistent:
        raise ValueError("inconsistent_winding")
    if not np.isfinite(mesh.volume) or mesh.volume <= 1e-12:
        raise ValueError("nonpositive_volume")
    center = mesh.bounds.mean(axis=0)
    scale = float(np.linalg.norm(mesh.vertices - center, axis=1).max())
    if scale <= 1e-12:
        raise ValueError("zero_extent")
    mesh.vertices = (mesh.vertices - center) / scale
    return mesh, {"source_center": center.tolist(), "source_scale": scale,
                  "vertices": len(mesh.vertices), "faces": len(mesh.faces), "normalized_volume": float(mesh.volume),
                  "cleanup": {"original_vertices": original_vertices, "original_faces": original_faces,
                              "removed_duplicate_or_degenerate_faces": original_faces-len(mesh.faces),
                              "reoriented_faces": flipped_faces}}


def _manifold(mesh, original_id: int):
    import manifold3d as m3d
    value = m3d.Manifold(m3d.Mesh(
        vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
        tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint32),
        run_index=np.array([0, len(mesh.faces) * 3], dtype=np.uint32),
        run_original_id=np.array([original_id], dtype=np.uint32),
        face_id=np.arange(len(mesh.faces), dtype=np.uint32)))
    if value.is_empty() or str(value.status()) != "Error.NoError":
        raise ValueError("manifold_rejected_mesh")
    return value


def _to_mesh(solid, interface_map: dict[int, int]):
    encoded = solid.to_mesh()
    mesh = trimesh.Trimesh(np.asarray(encoded.vert_properties)[:, :3],
                           np.asarray(encoded.tri_verts), process=False)
    labels = np.full(len(mesh.faces), -1, dtype=np.int32)
    runs = list(encoded.run_index)
    for i, original in enumerate(encoded.run_original_id):
        labels[runs[i] // 3:runs[i + 1] // 3] = interface_map.get(int(original), -1)
    # Manifold may duplicate vertices at property seams; merge positions while
    # retaining face order and provenance.
    mesh.merge_vertices()
    return mesh, labels


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform SO(3) from normalized Gaussian quaternions."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def heightfield_cutter(mesh: trimesh.Trimesh, rng: np.random.Generator,
                       band: str, family: str = "smooth", grid_size: int = 17):
    """Closed solid below a triangulated smooth, generally asymmetric height field."""
    rotation = random_rotation(rng)
    center = mesh.bounds.mean(axis=0)
    local = (mesh.vertices - center) @ rotation
    radius = float(np.linalg.norm(local, axis=1).max())
    extent = radius * 2.5
    # Harder cuts vary both balance and contact footprint, subject to QA below.
    if band == "easy":
        offset = float(rng.uniform(-.12,.12))*radius
    elif band == "intermediate":
        offset = float(rng.uniform(-.25,.25))*radius
    else:
        # Peripheral cuts reduce the mating footprint; rejected candidates are
        # resampled if this makes any piece smaller than 10% of the source.
        offset = float(rng.choice([-1,1])*rng.uniform(.25,.55))*radius
    amplitude = {"easy": .10, "intermediate": .20, "hard": .35}[band] * radius
    if family == "planar_control":
        amplitude = 0.
    axis = np.linspace(-extent, extent, grid_size)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    x, y = xx / radius, yy / radius
    phase = rng.uniform(-np.pi, np.pi, size=2)
    if family == "heldout_radial":
        height = np.sin(2.8 * np.sqrt((x-.2)**2 + (y+.3)**2)) + .25*x
    elif family == "symmetric_control":
        height = .6*(x*x+y*y)
    else:
        height = .6*np.sin(1.3*x+phase[0]) + .35*np.cos(1.7*y+phase[1]) + .15*x*y + .1*x
    zz = offset + amplitude * height
    top = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    bottom = top.copy()
    bottom[:, 2] = -4*radius
    vertices = np.concatenate([top, bottom])
    n = grid_size
    count = len(top)
    faces = []
    for row in range(n-1):
        for col in range(n-1):
            a = row*n+col
            b, c, d = a+1, a+n+1, a+n
            faces.extend([(a,b,c),(a,c,d),(a+count,c+count,b+count),(a+count,d+count,c+count)])
    boundary = list(range(n)) + [i*n+n-1 for i in range(1,n)] + list(range(n*n-2,n*(n-1)-1,-1)) + [i*n for i in range(n-2,0,-1)]
    for a,b in zip(boundary, boundary[1:]+boundary[:1]):
        faces.extend([(a,a+count,b+count),(a,b+count,b)])
    vertices = vertices @ rotation.T + center
    cutter = trimesh.Trimesh(vertices, np.asarray(faces), process=False)
    if not cutter.is_watertight or not cutter.is_winding_consistent or cutter.volume <= 0:
        raise ValueError("invalid_heightfield_cutter")
    params = {"family": family, "rotation": rotation.tolist(), "center": center.tolist(),
              "radius": radius, "offset": offset, "amplitude": amplitude,
              "phase": phase.tolist(), "grid_size": grid_size}
    return cutter, params


def fracture_mesh(mesh: trimesh.Trimesh, rng: np.random.Generator, band: str,
                  pieces: int = 2, family: str = "smooth", cfg: dict | None = None):
    """Return exact complementary manifold pieces, per-face interface IDs, and QA."""
    import manifold3d as m3d
    cfg = cfg or {}
    if pieces not in (2, 3):
        raise ValueError("Only complete two- or three-piece sets are supported")
    original_id = m3d.Manifold.reserve_ids(3)
    source = _manifold(mesh, original_id)
    solids = [source]
    interface_map = {}
    cutters = []
    for interface in range(pieces-1):
        chosen = max(range(len(solids)), key=lambda i: solids[i].volume())
        current_mesh, _ = _to_mesh(solids[chosen], interface_map)
        cutter, params = heightfield_cutter(current_mesh, rng, band, family, int(cfg.get("cutter_grid", 17)))
        cutter_id = original_id + interface + 1
        cutter_solid = _manifold(cutter, cutter_id)
        first, second = solids[chosen].split(cutter_solid)
        if first.is_empty() or second.is_empty():
            raise ValueError("empty_complement")
        solids[chosen:chosen+1] = [first, second]
        interface_map[cutter_id] = interface
        cutters.append(params)
    outputs = [_to_mesh(s, interface_map) for s in solids]
    volumes = np.array([s.volume() for s in solids])
    ratios = volumes / source.volume()
    minimum = {"easy": .25, "intermediate": .15, "hard": .10}[band]
    if np.any(ratios < minimum) or (band == "easy" and np.any(ratios > .75)):
        raise ValueError("piece_volume_fraction")
    relative_error = float(abs(volumes.sum()-source.volume()) / source.volume())
    tolerance = float(cfg.get("volume_tolerance", 2e-5))
    if relative_error > tolerance:
        raise ValueError("volume_not_preserved")
    maximum_overlap = 0.
    adjacency = np.zeros((pieces,pieces), dtype=bool)
    interface_areas = []
    for i, (piece, labels) in enumerate(outputs):
        if not piece.is_watertight or not piece.is_winding_consistent or len(piece.split(only_watertight=False)) != 1:
            raise ValueError("disconnected_or_invalid_piece")
        areas = {int(k): float(piece.area_faces[labels == k].sum()) for k in np.unique(labels) if k >= 0}
        interface_areas.append(areas)
        if not areas or not np.any(labels == -1):
            raise ValueError("missing_surface_provenance")
        for j in range(i):
            overlap = (solids[i] ^ solids[j]).volume()
            maximum_overlap = max(maximum_overlap, overlap / source.volume())
            # Same original cutter may be inherited by three descendants; use
            # actual shared boundary proximity to exclude disjoint patches.
            shared = set(areas) & set(interface_areas[j])
            for key in shared:
                a = piece.triangles_center[labels == key]
                b = outputs[j][0].triangles_center[outputs[j][1] == key]
                if not len(a) or not len(b):
                    continue
                _, distances, _ = trimesh.proximity.closest_point(outputs[j][0], a)
                if np.any(distances < 2e-5):
                    adjacency[i,j] = adjacency[j,i] = True
    if maximum_overlap > tolerance:
        raise ValueError("interior_overlap")
    reached = {0}
    for _ in range(pieces):
        reached |= {j for i in reached for j in range(pieces) if adjacency[i,j]}
    if len(reached) != pieces:
        raise ValueError("disconnected_contact_graph")
    # Make contact-size bands measurable; hard examples are sampled with smaller
    # cuts but never admitted if they violate completeness or piece-size limits.
    contact_fraction = sum(sum(a.values()) for a in interface_areas) / sum(x[0].area for x in outputs)
    if band == "easy" and contact_fraction < float(cfg.get("easy_min_contact_fraction", .08)):
        raise ValueError("easy_contact_too_small")
    return outputs, {"volume_ratios": ratios.tolist(), "volume_relative_error": relative_error,
                     "max_overlap_fraction": maximum_overlap, "adjacency": adjacency.tolist(),
                     "interface_areas": interface_areas, "contact_fraction": float(contact_fraction),
                     "cutters": cutters}


def sample_surface(mesh, count, rng):
    """Area-weighted independent sampling without sharing interface samples."""
    face = rng.choice(len(mesh.faces), size=count, p=mesh.area_faces/mesh.area)
    u = rng.random((count,2))
    flip = u.sum(axis=1) > 1
    u[flip] = 1-u[flip]
    triangle = mesh.triangles[face]
    points = triangle[:,0] + u[:,:1]*(triangle[:,1]-triangle[:,0]) + u[:,1:]*(triangle[:,2]-triangle[:,0])
    return points.astype(np.float32), face


def signed_distance(mesh, points, chunk_size=1024):
    """Trimesh uses positive-inside; this package always uses negative-inside."""
    return np.concatenate([-trimesh.proximity.signed_distance(mesh, points[i:i+chunk_size])
                           for i in range(0,len(points),chunk_size)]).astype(np.float32)


def prepare_dataset(source: Path, output: Path, cfg: dict, resource_guard=None) -> dict:
    """Prepare a bounded geometry pilot, writing a fingerprinted v2 manifest.

    This function never starts training. Insufficient source yield is represented
    by ``learning_ready=False`` and must be respected by training callers.
    """
    data = cfg.get("data", {})
    seed = int(data.get("seed", cfg.get("seed",42)))
    maximum = int(data.get("max_sources",100))
    minimum = int(data.get("min_sources",30))
    output = Path(output).resolve()
    _guard(resource_guard)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite an existing prepared manifest; select a new output directory")
    # Discover names first so source splits are locked before any fractures.
    if Path(source).suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            members = _mesh_members(archive.namelist())[:maximum]
    elif Path(source).is_dir():
        members = _mesh_members([str(p.relative_to(source)) for p in Path(source).rglob("*") if p.is_file()])[:maximum]
    else:
        members = [Path(source).name]
    splits = source_splits([_source_id(x) for x in members], seed)
    records, failures, sources = [], [], []
    accepted = 0
    reservoir = int(data.get("reservoir_points",4096))
    sdf_count = int(data.get("sdf_reservoir_queries",8192))
    target_count = int(data.get("target_points",2048))
    pattern_count = int(data.get("patterns_per_source",8))
    attempts = int(data.get("fracture_attempts",20))
    if min(reservoir,sdf_count,target_count,pattern_count) < 1:
        raise ValueError("Preparation sample and pattern counts must be positive")
    for source_number, (source_id, raw, member, load_error) in enumerate(iter_source_meshes(source, maximum), start=1):
        print(f"prepare source={source_number}/{len(members)} accepted={accepted} patterns={len(records)} id={source_id}", flush=True)
        _guard(resource_guard)
        if load_error:
            failures.append({"source_id":source_id,"reason":load_error})
            continue
        try:
            mesh, metadata = validate_source(raw)
            source_rng = np.random.default_rng(_seed(seed,source_id,"target"))
            target, _ = sample_surface(mesh,target_count,source_rng)
            near, _ = sample_surface(mesh,sdf_count//2,source_rng)
            near += source_rng.normal(scale=float(data.get("sdf_surface_noise",.04)),size=near.shape)
            uniform = source_rng.uniform(mesh.bounds[0]-.4,mesh.bounds[1]+.4,size=(sdf_count-len(near),3))
            queries = np.concatenate([near,uniform]).astype(np.float32)
            values = signed_distance(mesh,queries,int(data.get("sdf_chunk_size",512)))
            near_mask = np.arange(sdf_count) < len(near)
        except (ValueError, RuntimeError, ImportError) as exc:
            failures.append({"source_id":source_id,"reason":str(exc)})
            continue
        source_records = []
        for pattern_index in range(pattern_count):
            band = BANDS[pattern_index % 3]
            # The holdout family is a separate split on held-out source objects;
            # it cannot leak through another fracture of a training source.
            family = "smooth"
            if bool(data.get("include_controls",True)) and pattern_count >= 5:
                # Controls occupy easy/intermediate slots. In particular, the
                # default hard three-piece slot retains its curved cut family.
                if pattern_index == pattern_count-2:
                    family = "planar_control"
                elif pattern_index == pattern_count-1:
                    family = "symmetric_control"
            if splits[source_id] == "test" and pattern_index == pattern_count-1:
                family = "heldout_radial"
            pieces = 2 if band == "easy" else 2 + (pattern_index % 2)
            pattern_id = f"{source_id}_{pattern_index:03d}"
            reason = "no_attempt"
            for attempt in range(attempts):
                attempt_seed = _seed(seed,source_id,pattern_index,attempt)
                rng = np.random.default_rng(attempt_seed)
                try:
                    fractured, qa = fracture_mesh(mesh,rng,band,pieces,family,data)
                    point_sets, label_sets, interface_sets = [], [], []
                    for piece, face_ids in fractured:
                        points, face_index = sample_surface(piece,reservoir,rng)
                        ids = face_ids[face_index]
                        if not np.any(ids >= 0):
                            raise ValueError("reservoir_missed_contact")
                        point_sets.append(points)
                        label_sets.append((ids >= 0).astype(np.uint8))
                        interface_sets.append(ids)
                    pattern_meta = dict(metadata,source_id=source_id,pattern_id=pattern_id,band=band,
                                        cut_family=family,seed=attempt_seed,qa=qa,source_member=member)
                    filename = f"patterns/{pattern_id}.npz"
                    path = output / filename
                    path.parent.mkdir(parents=True,exist_ok=True)
                    payload = {"points":np.asarray(point_sets,dtype=np.float32),
                               "fracture_labels":np.asarray(label_sets,dtype=np.uint8),
                               "interface_ids":np.asarray(interface_sets,dtype=np.int16),
                               "metadata":np.asarray(json.dumps(pattern_meta))}
                    _guard(resource_guard,sum(x.nbytes for x in payload.values()))
                    with path.open("xb") as stream:
                        np.savez_compressed(stream,**payload)
                    digest = _file_sha256(path)
                    split = "cut_holdout" if family == "heldout_radial" else splits[source_id]
                    source_records.append({"path":filename,"source_id":source_id,"pattern_id":pattern_id,
                                           "split":split,"band":band,"cut_family":family,"pieces":pieces,"sha256":digest})
                    break
                except (ValueError, RuntimeError) as exc:
                    reason = str(exc)
            else:
                failures.append({"source_id":source_id,"pattern_id":pattern_id,"band":band,"reason":reason})
        if source_records:
            accepted += 1
            records.extend(source_records)
            source_path = output / "sources" / f"{source_id}.npz"
            source_path.parent.mkdir(parents=True,exist_ok=True)
            _guard(resource_guard,mesh.vertices.nbytes+mesh.faces.nbytes+queries.nbytes+values.nbytes+near_mask.nbytes+target.nbytes)
            with source_path.open("xb") as stream:
                np.savez_compressed(stream,vertices=np.asarray(mesh.vertices,dtype=np.float32),
                                    faces=np.asarray(mesh.faces,dtype=np.int32),sdf_queries=queries,
                                    sdf_values=values,sdf_near_mask=near_mask,target_points=target)
            sources.append(dict(source_id=source_id,split=splits[source_id],patterns=len(source_records),
                                path=f"sources/{source_id}.npz",sha256=_file_sha256(source_path),**metadata))
        else:
            failures.append({"source_id":source_id,"reason":"no_valid_patterns"})
    report = {"candidates":len(members),"accepted_sources":accepted,"accepted_patterns":len(records),
              "minimum_sources":minimum,"learning_ready":accepted >= minimum,
              "rejections":failures,"split_counts":{s:sum(r["split"] == s for r in records)
                                                     for s in ("train","val","test","cut_holdout")}}
    manifest = {"schema_version":SCHEMA_VERSION,"dataset":"shapenet_bottle_procedural",
                "source":str(Path(source).resolve()),"seed":seed,"patterns":records,"sources":sources,"report":report,
                "config":data,"sdf_convention":"negative_inside"}
    manifest["fingerprint"] = manifest_fingerprint(manifest)
    report.update(kind="preparation",dataset_fingerprint=manifest["fingerprint"])
    encoded = json.dumps(manifest,indent=2)
    _guard(resource_guard,len(encoded.encode()))
    (output / "manifest.json").write_text(encoded,encoding="utf-8")
    return dict(report,manifest=str(output / "manifest.json"),fingerprint=manifest["fingerprint"])
