"""Geometry and dataset contract checks; no downloads or pretrained weights."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
import torch
import trimesh

from reassembly.acquire import _CredentialSafeRedirect, acquire_bottles
from reassembly.data import FractureDataset, make_oracle_field, verify_manifest
from reassembly.prepare import (fracture_mesh, iter_source_meshes, prepare_dataset,
                                sample_surface, select_complete_breaking_bad_sets,
                                signed_distance, source_splits, validate_source)


def fixture_config():
    return {"data":{"seed":31,"points_per_fragment":48,"reservoir_points":128,
                    "sdf_queries":32,"sdf_reservoir_queries":64,"target_points":96,
                    "patterns_per_source":3,"max_sources":100,"min_sources":30,
                    "hard_noise_std":0.,"cutter_grid":11},
            "train":{"max_updates":100}}


class FractureGeometryTests(unittest.TestCase):
    def test_complementarity_provenance_and_independent_samples(self):
        mesh,_ = validate_source(trimesh.creation.icosphere(subdivisions=2))
        pieces,qa = fracture_mesh(mesh,np.random.default_rng(4),"easy",2)
        self.assertLess(qa["volume_relative_error"],2e-5)
        self.assertLess(qa["max_overlap_fraction"],2e-5)
        self.assertTrue(qa["adjacency"][0][1])
        for piece,face_ids in pieces:
            self.assertTrue(piece.is_watertight)
            self.assertEqual(set(np.unique(face_ids)),{-1,0})
            cut = piece.triangles_center[face_ids == 0]
            # New mating faces lie inside the source and do not get confused
            # with its original external surface.
            self.assertLess(float(np.median(signed_distance(mesh,cut))),-.03)
        a,_ = sample_surface(pieces[0][0],64,np.random.default_rng(1))
        b,_ = sample_surface(pieces[1][0],64,np.random.default_rng(2))
        self.assertFalse(np.any(np.all(a[:,None,:] == b[None,:,:],axis=-1)))

    def test_three_piece_contact_graph_and_provenance_survive_second_cut(self):
        mesh,_ = validate_source(trimesh.creation.icosphere(subdivisions=1))
        for seed in range(20):
            try:
                pieces,qa = fracture_mesh(mesh,np.random.default_rng(seed),"intermediate",3,cfg={"cutter_grid":11})
                break
            except ValueError:
                continue
        else:
            self.fail("No valid three-piece fixture in bounded attempts")
        self.assertEqual(len(pieces),3)
        self.assertEqual(set(np.concatenate([labels for _,labels in pieces])),{-1,0,1})
        self.assertTrue(all(sum(row)>=1 for row in qa["adjacency"]))
        self.assertGreaterEqual(sum(map(sum,qa["adjacency"])),4)
        self.assertLess(qa["volume_relative_error"],2e-5)

    def test_reject_open_and_disconnected_sources(self):
        cube = trimesh.creation.box()
        opened = cube.copy()
        opened.update_faces(np.arange(len(opened.faces)-1))
        with self.assertRaisesRegex(ValueError,"not_watertight"):
            validate_source(opened)
        second = cube.copy()
        second.apply_translation([3,0,0])
        with self.assertRaisesRegex(ValueError,"disconnected_source"):
            validate_source(trimesh.util.concatenate([cube,second]))

    def test_double_sided_obj_seams_clean_without_geometry_changes(self):
        cube = trimesh.creation.box()
        triangles = cube.triangles.copy()
        triangles[::2] = triangles[::2, ::-1]
        vertices = np.concatenate([triangles.reshape(-1,3), triangles[:,::-1].reshape(-1,3)])
        raw = trimesh.Trimesh(vertices, np.arange(len(vertices)).reshape(-1,3), process=False)
        cleaned,metadata = validate_source(raw)
        self.assertTrue(cleaned.is_watertight)
        self.assertTrue(cleaned.is_winding_consistent)
        self.assertEqual(len(cleaned.faces),len(cube.faces))
        self.assertEqual(metadata['cleanup']['removed_duplicate_or_degenerate_faces'],len(cube.faces))
        restored = cleaned.vertices*metadata['source_scale']+metadata['source_center']
        np.testing.assert_allclose(np.unique(restored,axis=0),np.unique(cube.vertices,axis=0))
        self.assertAlmostEqual(cleaned.volume*metadata['source_scale']**3,cube.volume)

    def test_closed_inverted_surface_is_reoriented_without_filling(self):
        cube = trimesh.creation.box()
        expected_volume = cube.volume
        cube.invert()
        cleaned,metadata = validate_source(cube)
        self.assertGreater(metadata['cleanup']['reoriented_faces'],0)
        self.assertAlmostEqual(cleaned.volume*metadata['source_scale']**3,expected_volume)

    def test_cavity_preserved_and_sdf_sign(self):
        import manifold3d as m3d
        outer = m3d.Manifold.cube([2,2,2],True)
        # Open cavity through the top, with connected inner/outer boundary.
        cavity = m3d.Manifold.cube([1,1,2],True).translate([0,0,.5])
        encoded = (outer-cavity).to_mesh()
        bowl = trimesh.Trimesh(np.asarray(encoded.vert_properties)[:,:3],encoded.tri_verts,process=False)
        cleaned,metadata = validate_source(bowl)
        self.assertAlmostEqual(cleaned.volume*metadata["source_scale"]**3,bowl.volume,places=5)
        original_queries = np.array([[0,0,.6],[.8,0,0],[4,0,0]])
        queries = (original_queries-metadata["source_center"])/metadata["source_scale"]
        sdf = signed_distance(cleaned,queries)
        self.assertGreater(sdf[0],0)
        self.assertLess(sdf[1],0)
        self.assertGreater(sdf[2],0)

    def test_source_split_counts_and_order_independence(self):
        ids = [str(i) for i in range(100)]
        first = source_splits(ids,19)
        self.assertEqual(first,source_splits(ids[::-1],19))
        self.assertEqual({s:list(first.values()).count(s) for s in ("train","val","test")},
                         {"train":80,"val":10,"test":10})

    def test_archive_stream_selects_only_bottles_and_one_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"objects.zip"
            obj = trimesh.creation.box().export(file_type="obj")
            with zipfile.ZipFile(path,"w") as archive:
                archive.writestr("02876657/bottle/models/model_normalized.obj",obj)
                archive.writestr("02876657/bottle/models/model.obj",obj)
                archive.writestr("03001627/chair/models/model_normalized.obj",obj)
            records = list(iter_source_meshes(path))
            self.assertEqual(len(records),1)
            self.assertEqual(records[0][0],"bottle")
            self.assertIsNone(records[0][3])

    def test_breaking_bad_inventory_never_truncates_large_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for count in (1,2,3,4,9):
                pattern = root/"bottle"/f"fractured_{count}"
                pattern.mkdir(parents=True)
                for i in range(count):
                    (pattern/f"piece_{i}.obj").write_text("# inventory fixture")
            selected = select_complete_breaking_bad_sets(root)
            self.assertEqual([item["pieces"] for item in selected],[2,3])
            for record in selected:
                self.assertEqual(len(record["piece_paths"]),record["pieces"])


class FractureDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)
        cls.cfg = fixture_config()
        source = cls.path/"inputs"
        source.mkdir()
        trimesh.creation.icosphere(subdivisions=1).export(source/"bottle.ply")
        cls.report = prepare_dataset(source,cls.path/"prepared",cls.cfg)
        cls.manifest = cls.path/"prepared"/"manifest.json"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_low_yield_is_reported_without_fake_learning_success(self):
        self.assertEqual(self.report["accepted_sources"],1)
        self.assertEqual(self.report["accepted_patterns"],3)
        self.assertFalse(self.report["learning_ready"])
        with self.assertRaises(FileExistsError):
            prepare_dataset(self.path/"inputs",self.path/"prepared",self.cfg)

    def test_intact_targets_are_stored_once_per_source(self):
        manifest = json.loads(self.manifest.read_text())
        self.assertEqual(len(manifest["sources"]),1)
        with np.load(self.manifest.parent/manifest["sources"][0]["path"],allow_pickle=False) as source:
            self.assertTrue({"vertices","faces","sdf_queries","sdf_values","target_points"}.issubset(source.files))
        for record in manifest["patterns"]:
            with np.load(self.manifest.parent/record["path"],allow_pickle=False) as pattern:
                self.assertNotIn("sdf_queries",pattern.files)
                self.assertNotIn("target_points",pattern.files)

    def test_content_verification_rejects_tampered_assets(self):
        verification = verify_manifest(self.manifest)
        self.assertEqual(verification["status"],"verified")
        self.assertEqual(verification["verified_sources"],1)
        self.assertEqual(verification["verified_patterns"],3)
        for group in ("sources","patterns"):
            with tempfile.TemporaryDirectory() as tmp:
                copied = Path(tmp)/"prepared"
                shutil.copytree(self.manifest.parent,copied)
                document = json.loads((copied/"manifest.json").read_text())
                with (copied/document[group][0]["path"]).open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError,"SHA-256 mismatch"):
                    verify_manifest(copied/"manifest.json")

    def test_content_verification_rejects_metadata_tampering_and_path_escape(self):
        document = json.loads(self.manifest.read_text())
        path = self.manifest.parent/"changed_manifest.json"
        modified = copy.deepcopy(document)
        modified["seed"] += 1
        path.write_text(json.dumps(modified))
        with self.assertRaisesRegex(ValueError,"fingerprint mismatch"):
            verify_manifest(path)
        modified = copy.deepcopy(document)
        modified["patterns"][0]["path"] = "../outside.npz"
        path.write_text(json.dumps(modified))
        with self.assertRaisesRegex(ValueError,"outside the prepared"):
            verify_manifest(path)

    def test_relative_transform_and_anchor_contract(self):
        dataset = FractureDataset(self.manifest,"all",self.cfg,fixed=True)
        for sample in dataset:
            transformed = sample["points"]@sample["rotations_gt"].transpose(-1,-2)+sample["translations_gt"][:,None,:]
            self.assertTrue(torch.allclose(transformed,sample["canonical_points"],atol=2e-6))
            anchor = int(sample["anchor_index"])
            self.assertTrue(torch.equal(sample["rotations_gt"][anchor],torch.eye(3)))
            self.assertTrue(torch.equal(sample["translations_gt"][anchor],torch.zeros(3)))
            self.assertTrue(torch.allclose(torch.linalg.det(sample["rotations_gt"]),torch.ones(3),atol=2e-6))
            active = sample["fragment_mask"]
            self.assertTrue(torch.allclose(sample["points"][active].mean(1),torch.zeros(int(active.sum()),3),atol=2e-6))
            radius_sum = torch.linalg.vector_norm(sample["points"][active],dim=-1).amax(1).sum()
            self.assertAlmostEqual(float(radius_sum),1.,places=5)

    def test_eval_and_fixed_repeatability_and_oracle_frame(self):
        dataset = FractureDataset(self.manifest,"all",self.cfg,fixed=True)
        first = dataset[0]
        dataset.set_step(85)
        second = dataset[0]
        self.assertTrue(torch.equal(first["points"],second["points"]))
        self.assertTrue(torch.equal(first["sdf_queries"],second["sdf_queries"]))
        oracle = make_oracle_field(first)
        evaluated = oracle(first["sdf_queries"].numpy())
        np.testing.assert_allclose(evaluated,first["sdf_values"].numpy(),atol=2e-6)
        # Optional GT target lies on the intact source surface.
        np.testing.assert_allclose(oracle(first["target_points"].numpy()),0,atol=3e-6)

    def test_schema_leakage_and_heldout_guards(self):
        manifest = json.loads(self.manifest.read_text())
        modified = copy.deepcopy(manifest)
        modified["patterns"][0]["split"] = "train"
        modified["patterns"][1]["split"] = "val"
        path = self.path/"leak.json"
        path.write_text(json.dumps(modified))
        with self.assertRaisesRegex(ValueError,"leaks"):
            FractureDataset(path,"all",self.cfg)
        modified["schema_version"] = 1
        path.write_text(json.dumps(modified))
        with self.assertRaisesRegex(ValueError,"schema_version"):
            FractureDataset(path,"all",self.cfg)

    def test_batch_contract_for_two_and_three_pieces(self):
        dataset = FractureDataset(self.manifest,"all",self.cfg,fixed=True)
        batch = next(iter(torch.utils.data.DataLoader(dataset,batch_size=3)))
        self.assertEqual(tuple(batch["points"].shape),(3,3,48,3))
        self.assertEqual(tuple(batch["canonical_points"].shape),(3,3,48,3))
        self.assertEqual(tuple(batch["sdf_queries"].shape),(3,32,3))
        self.assertEqual(set(batch["fragment_mask"].sum(1).tolist()),{2,3})

    def test_curriculum_adds_bands_without_dropping_fragments(self):
        manifest = json.loads(self.manifest.read_text())
        for record in manifest["patterns"]:
            record["split"] = "train"
            record["cut_family"] = "smooth"
        path = self.manifest.parent/"curriculum.json"
        path.write_text(json.dumps(manifest))
        dataset = FractureDataset(path,"train",self.cfg)
        self.assertEqual({dataset[i]["band"] for i in range(len(dataset))},{"easy"})
        first = dataset[0]["points"]
        dataset.set_step(40)
        self.assertEqual({dataset[i]["band"] for i in range(len(dataset))},{"easy","intermediate"})
        dataset.set_step(90)
        self.assertEqual({dataset[i]["band"] for i in range(len(dataset))},{"easy","intermediate","hard"})
        self.assertFalse(torch.equal(first,dataset[0]["points"]))
        dataset.set_step(0)
        self.assertTrue(torch.equal(first,dataset[0]["points"]))


class AcquisitionTests(unittest.TestCase):
    def test_dry_run_reads_only_metadata_and_checks_size(self):
        metadata = mock.Mock(size=1234,commit_hash="version",etag="hash")
        guard = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"not_created"
            with mock.patch("huggingface_hub.get_hf_file_metadata",return_value=metadata),mock.patch("huggingface_hub.get_token",return_value=None):
                result = acquire_bottles(path,{},guard,dry_run=True)
            self.assertEqual(result["status"],"available")
            self.assertEqual(result["category"],"02876657")
            guard.check.assert_called_once_with(additional_bytes=1234)
            self.assertFalse(path.exists())

    def test_redirect_does_not_forward_account_token_to_cdn(self):
        from urllib.request import Request
        request = Request("https://huggingface.co/datasets/example",headers={"Authorization":"Bearer private"})
        redirect = _CredentialSafeRedirect().redirect_request(request,None,302,"Found",{},"https://cdn.example/data?signature=opaque")
        self.assertIsNone(redirect.get_header("Authorization"))


if __name__ == "__main__":
    unittest.main()
