"""Immutable prepared NPZ fixtures for repaired views, curriculum, and fields."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import trimesh

from reassembly.data import verify_manifest
from reassembly.prepare import _file_sha256, manifest_fingerprint, sample_surface
from reassembly.repair.config import load_config, validate_config
from reassembly.repair.data import RepairDataset, collate_samples, supplement_queries
from reassembly.repair.losses import compute_losses
from reassembly.repair.model import RepairModel, configure_stage
from reassembly.repair.training import _datasets


def box_distance(points):
    offset = np.abs(points) - 1.
    return np.linalg.norm(np.maximum(offset, 0), axis=-1) + np.minimum(offset.max(-1), 0)


def tiny_config():
    cfg = load_config()
    cfg['data'].update(points_per_fragment=48, reservoir_points=192, sdf_queries=32,
                       target_points=96, hard_noise_std=0.)
    cfg['model'].update(dim=16, sample_counts=[20, 12, 8], neighbors=6,
                        contact_points=32, heads=4, attention_layers=2)
    return cfg


def write_prepared(directory):
    """Connected closed box fragments; surface samples differ across pieces."""
    directory = Path(directory)
    (directory / 'sources').mkdir(parents=True)
    (directory / 'patterns').mkdir()
    patterns, sources = [], []
    cube = trimesh.creation.box(extents=[2, 2, 2])
    for source_index, (source_id, split, pattern_count) in enumerate((('train_box', 'train', 18), ('val_box', 'val', 3))):
        rng = np.random.default_rng(218 + source_index)
        target, _ = sample_surface(cube, 192, rng)
        queries = np.concatenate((target[:64] * .98, target[64:128] * 1.02, rng.uniform(-1.5, 1.5, (64, 3))))
        source_path = directory / 'sources' / f'{source_id}.npz'
        np.savez_compressed(source_path, vertices=np.asarray(cube.vertices, np.float32),
            faces=np.asarray(cube.faces, np.int32), target_points=np.asarray(target, np.float32),
            sdf_queries=np.asarray(queries, np.float32), sdf_values=box_distance(queries).astype(np.float32),
            sdf_near_mask=np.arange(len(queries)) < 128)
        sources.append(dict(source_id=source_id, split=split, path=f'sources/{source_id}.npz',
                            sha256=_file_sha256(source_path), patterns=pattern_count))
        for index in range(pattern_count):
            band = ('easy', 'intermediate', 'hard')[index % 3]
            count = 2 if index % 2 == 0 else 3
            bounds = [-1, 0, 1] if count == 2 else [-1, -.25, .35, 1]
            points, labels = [], []
            for part in range(count):
                piece = trimesh.creation.box(extents=[bounds[part + 1] - bounds[part], 2, 2])
                piece.apply_translation([(bounds[part + 1] + bounds[part]) / 2, 0, 0])
                face_ids = np.full(len(piece.faces), -1, np.int16)
                for interface, cut in enumerate(bounds[1:-1]):
                    face_ids[np.isclose(piece.triangles_center[:, 0], cut)] = interface
                sampled, faces = sample_surface(piece, 192, np.random.default_rng(index * 10 + part + 991))
                points.append(sampled)
                labels.append(face_ids[faces])
            pattern_id = f'{source_id}_{index:03d}'
            path = directory / 'patterns' / f'{pattern_id}.npz'
            metadata = dict(source_id=source_id, pattern_id=pattern_id, band=band,
                qa=dict(volume_ratios=[(bounds[i + 1] - bounds[i]) / 2 for i in range(count)],
                        adjacency=[[abs(i - j) == 1 for j in range(count)] for i in range(count)]))
            ids = np.asarray(labels, np.int16)
            np.savez_compressed(path, points=np.asarray(points, np.float32), interface_ids=ids,
                fracture_labels=(ids >= 0).astype(np.uint8), metadata=np.asarray(json.dumps(metadata)))
            patterns.append(dict(source_id=source_id, pattern_id=pattern_id, split=split, band=band,
                pieces=count, cut_family='planar_control', path=f'patterns/{pattern_id}.npz', sha256=_file_sha256(path)))
    document = dict(schema_version=2, dataset='synthetic_unit_fixture', seed=42,
        config={'fixture': 'closed_complementary_boxes'}, patterns=patterns, sources=sources,
        sdf_convention='negative_inside')
    document['fingerprint'] = manifest_fingerprint(document)
    path = directory / 'manifest.json'
    path.write_text(json.dumps(document), encoding='utf-8')
    verify_manifest(path)
    return path


def rewrite_source(manifest, change):
    document = json.loads(Path(manifest).read_text(encoding='utf-8'))
    record = document['sources'][0]
    path = Path(manifest).parent / record['path']
    with np.load(path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    change(payload)
    np.savez_compressed(path, **payload)
    record['sha256'] = _file_sha256(path)
    document['fingerprint'] = manifest_fingerprint(document)
    Path(manifest).write_text(json.dumps(document), encoding='utf-8')
    verify_manifest(manifest)
    return document


def remove_near_inside_queries(payload):
    chosen = payload['sdf_near_mask'] & (payload['sdf_values'] < 0)
    payload['sdf_queries'][chosen] *= 1.05
    payload['sdf_values'] = box_distance(payload['sdf_queries']).astype(np.float32)


def cache_fixture(manifest, directory):
    document = json.loads(Path(manifest).read_text(encoding='utf-8'))
    record = document['sources'][0]
    directory = Path(directory)
    directory.mkdir()
    path = directory / f"{record['source_id']}.npz"
    points = np.array([[.99, 0, 0], [-.99, .1, .2], [0, -.99, 0]], dtype=np.float32)
    np.savez_compressed(path, sdf_queries=points, sdf_values=box_distance(points).astype(np.float32))
    cache = dict(schema_version=3, dataset_fingerprint=document['fingerprint'], sources=[
        dict(source_id=record['source_id'], source_sha256=record['sha256'], path=path.name,
             sha256=_file_sha256(path), count=len(points))])
    metadata = directory / 'query_cache.json'
    metadata.write_text(json.dumps(cache), encoding='utf-8')
    return metadata


class RepairDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)
        cls.manifest = write_prepared(cls.path / 'prepared')
        cls.cfg = tiny_config()
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        cls.temp.cleanup()

    def copy_prepared(self, output):
        shutil.copytree(self.manifest.parent, Path(output) / 'prepared')
        return Path(output) / 'prepared' / 'manifest.json'

    def test_independent_views_use_common_supervision_frame_without_reposing_inputs(self):
        dataset = RepairDataset(self.manifest, 'train', self.cfg, fixed=False, fixed_geometry=True, balanced_fields=False)
        dataset.set_step(1735)
        for index in (0, 1):
            sample = dataset[index]
            raw = dataset.second[index]
            reference = dataset.first[index]
            observed = sample['view2']['points']
            torch.testing.assert_close(observed, raw['points'], atol=0, rtol=0)
            restored_source = (raw['canonical_points'].double() * raw['shared_scale'].double()) @ raw['anchor_source_rotation'].double() + raw['anchor_source_centroid'].double()
            expected = (restored_source - reference['anchor_source_centroid'].double()) @ reference['anchor_source_rotation'].double().T / reference['shared_scale'].double()
            expected[~raw['fragment_mask']] = 0
            torch.testing.assert_close(sample['view2']['canonical_points'], expected.float(), atol=2e-6, rtol=1e-5)
            self.assertFalse(torch.equal(sample['points'], observed))
            self.assertFalse(torch.equal(sample['canonical_points'], sample['view2']['canonical_points']))
            self.assertFalse(torch.equal(reference['anchor_source_rotation'], raw['anchor_source_rotation']))
            # Every view2 canonical point unprojects onto an independently
            # selected reservoir point of the same fragment, at source scale.
            source_again = (sample['view2']['canonical_points'].double() * reference['shared_scale'].double()) @ reference['anchor_source_rotation'].double() + reference['anchor_source_centroid'].double()
            with np.load(dataset.root / dataset.records[index]['path'], allow_pickle=False) as archive:
                for part in range(int(raw['fragment_mask'].sum())):
                    distance = np.linalg.norm(source_again[part].numpy()[:, None] - archive['points'][part][None], axis=-1)
                    self.assertLess(float(distance.min(-1).max()), 2e-6)

    def test_absolute_curriculum_is_not_stretched_by_the_optimizer_budget(self):
        for budget in (2000, 10000, 30000):
            cfg = copy.deepcopy(self.cfg)
            cfg['train']['max_updates'] = budget
            dataset = RepairDataset(self.manifest, 'train', cfg, views=False, balanced_fields=False)
            for step, bands in ((0, {'easy'}), (599, {'easy'}), (600, {'easy', 'intermediate'}),
                                (1299, {'easy', 'intermediate'}), (1300, {'easy', 'intermediate', 'hard'}),
                                (2100, {'easy', 'intermediate', 'hard'})):
                dataset.set_step(step)
                self.assertEqual({dataset._record(i)['band'] for i in range(len(dataset))}, bands)

    def test_fixed_sixteen_geometries_refresh_points_and_poses(self):
        train, val = _datasets(self.manifest, self.cfg, 1, 'overfit', None)
        self.assertEqual(len(train), 16)
        self.assertEqual(len(val), 16)
        expected = [r['pattern_id'] for r in train.records]
        self.assertEqual(expected, [r['pattern_id'] for r in val.records])
        train.set_step(0)
        first = [train[i] for i in range(16)]
        train.set_step(1301)
        second = [train[i] for i in range(16)]
        self.assertEqual([s['pattern_id'] for s in first], expected)
        self.assertEqual([s['pattern_id'] for s in second], expected)
        self.assertTrue(all(not torch.equal(a['points'], b['points']) for a, b in zip(first, second)))
        self.assertTrue(all(not torch.equal(a['canonical_points'], b['canonical_points']) for a, b in zip(first, second)))
        original = val[0]
        val.set_step(9999)
        torch.testing.assert_close(original['points'], val[0]['points'], atol=0, rtol=0)

    def test_four_balanced_groups_have_correct_source_frame_zero_and_signs(self):
        dataset = RepairDataset(self.manifest, 'train', self.cfg, fixed=True, views=False)
        for index in (0, 1):
            sample = dataset[index]
            group, values = sample['sdf_query_group'], sample['sdf_values']
            self.assertEqual(torch.bincount(group).tolist(), [8, 8, 8, 8])
            self.assertTrue((values[group == 0] == 0).all())
            self.assertTrue((values[group == 1] < 0).all())
            self.assertTrue((values[group == 2] > 0).all())
            source = (sample['sdf_queries'].double() * sample['shared_scale'].double()) @ sample['anchor_source_rotation'].double() + sample['anchor_source_centroid'].double()
            exact = box_distance(source.numpy()) / float(sample['shared_scale'])
            np.testing.assert_allclose(values.numpy(), exact, atol=3e-7)
            torch.testing.assert_close(sample['sdf_near_mask'], group != 3)
            repeat = dataset[index]
            torch.testing.assert_close(sample['sdf_queries'], repeat['sdf_queries'], atol=0, rtol=0)

    def test_query_cache_repairs_missing_signed_group_and_preserves_base_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = self.copy_prepared(temp)
            document = rewrite_source(manifest, remove_near_inside_queries)
            before = {r['path']: _file_sha256(manifest.parent / r['path']) for r in document['sources'] + document['patterns']}
            missing = RepairDataset(manifest, 'train', self.cfg, fixed=True, views=False)
            with self.assertRaisesRegex(ValueError, 'near_inside'):
                missing[0]
            cache = cache_fixture(manifest, Path(temp) / 'cache')
            dataset = RepairDataset(manifest, 'train', self.cfg, fixed=True, views=False, query_cache=cache)
            sample = dataset[0]
            self.assertTrue((sample['sdf_values'][sample['sdf_query_group'] == 1] < 0).all())
            self.assertEqual(dataset.query_cache_hash, _file_sha256(cache))
            self.assertEqual(before, {r: _file_sha256(manifest.parent / r) for r in before})

    def test_cache_provenance_rejects_wrong_dataset_source_and_payload_hashes(self):
        for mutation, message in (('fingerprint', 'different dataset'), ('source', 'hash'), ('payload', 'hash')):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                cache_path = cache_fixture(self.manifest, Path(temp) / 'cache')
                cache = json.loads(cache_path.read_text())
                if mutation == 'fingerprint':
                    cache['dataset_fingerprint'] = '0' * 64
                elif mutation == 'source':
                    cache['sources'][0]['source_sha256'] = '0' * 64
                else:
                    asset = cache_path.parent / cache['sources'][0]['path']
                    asset.write_bytes(asset.read_bytes() + b'tampered')
                cache_path.write_text(json.dumps(cache))
                with self.assertRaisesRegex(ValueError, message):
                    RepairDataset(self.manifest, 'train', self.cfg, query_cache=cache_path)

    def test_cache_rejects_wrong_schema_duplicates_counts_shapes_and_nonfinite_coordinates(self):
        cases = (('schema', 'schema_version'), ('duplicate', 'duplicate'), ('unknown', 'Unknown'),
                 ('count', 'shape/count'), ('shape', 'shape/count'), ('nonfinite', 'Nonfinite'),
                 ('traversal', 'outside'))
        for mutation, message in cases:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                cache_path = cache_fixture(self.manifest, Path(temp) / 'cache')
                cache = json.loads(cache_path.read_text())
                record = cache['sources'][0]
                if mutation == 'schema':
                    cache['schema_version'] = 2
                elif mutation == 'duplicate':
                    cache['sources'].append(copy.deepcopy(record))
                elif mutation == 'unknown':
                    record['source_id'] = 'not_in_manifest'
                elif mutation == 'count':
                    record['count'] += 1
                elif mutation == 'traversal':
                    record['path'] = '../outside.npz'
                else:
                    asset = cache_path.parent / record['path']
                    with np.load(asset, allow_pickle=False) as stored:
                        points, values = stored['sdf_queries'], stored['sdf_values']
                    if mutation == 'shape':
                        points = points[:, :2]
                    else:
                        points[0, 0] = np.nan
                    np.savez_compressed(asset, sdf_queries=points, sdf_values=values)
                    record['sha256'] = _file_sha256(asset)
                cache_path.write_text(json.dumps(cache))
                with self.assertRaisesRegex(ValueError, message):
                    RepairDataset(self.manifest, 'train', self.cfg, query_cache=cache_path)

    def test_supplement_command_writes_only_bound_missing_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = self.copy_prepared(temp)
            document = rewrite_source(manifest, remove_near_inside_queries)
            before = _file_sha256(manifest)
            result = supplement_queries(manifest, Path(temp) / 'supplement')
            self.assertEqual(result['schema_version'], 3)
            self.assertEqual(result['dataset_fingerprint'], document['fingerprint'])
            self.assertEqual([r['source_id'] for r in result['sources']], ['train_box'])
            self.assertEqual(_file_sha256(manifest), before)
            dataset = RepairDataset(manifest, 'train', self.cfg, fixed=True, views=False,
                                    query_cache=Path(temp) / 'supplement' / 'query_cache.json')
            sample = dataset[0]
            self.assertEqual(torch.bincount(sample['sdf_query_group']).tolist(), [8, 8, 8, 8])

    def test_real_npz_views_collate_and_all_factorial_losses_backpropagate(self):
        for geometry in ('existing', 'revised'):
            for supervision in ('existing', 'resampled_contrastive'):
                with self.subTest(geometry=geometry, supervision=supervision):
                    cfg = copy.deepcopy(self.cfg)
                    cfg['repair'].update(geometry_variant=geometry, view_supervision=supervision)
                    dataset = RepairDataset(self.manifest, 'train', cfg, fixed_geometry=True, balanced_fields=False)
                    dataset.set_step(2100)
                    ids = [next(i for i, r in enumerate(dataset.records) if r['pieces'] == n) for n in (2, 3)]
                    batch = collate_samples([dataset[i] for i in ids], 'cpu')
                    self.assertEqual(batch['view2']['canonical_points'].shape, (2, 3, 48, 3))
                    model = RepairModel(cfg).train()
                    configure_stage(model, 1)
                    losses = compute_losses(model, batch, 1, cfg)
                    self.assertTrue(torch.isfinite(losses['loss']))
                    losses['loss'].backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    if supervision == 'resampled_contrastive':
                        self.assertGreater(losses['view_valid_rows'].item(), 0)

    def test_group_zero_has_no_sign_penalty_and_calibration_is_regional(self):
        dataset = RepairDataset(self.manifest, 'train', self.cfg, fixed=True, views=False)
        batch = collate_samples([dataset[0]], 'cpu')
        group = batch['sdf_query_group']
        prediction = batch['sdf_values'].clamp(-.1, .1).clone()
        prediction[group == 0] = -.05
        model = RepairModel(self.cfg)
        with mock.patch.object(model, 'scaffold', return_value=dict(distance=prediction, log_scale=torch.full_like(prediction, -4.))):
            metrics = compute_losses(model, batch, 2, self.cfg)
        self.assertEqual(metrics['sdf_sign_count'].item(), 24)
        self.assertEqual(metrics['sdf_sign_error'].item(), 0.)
        self.assertAlmostEqual(metrics['sdf_surface_l1'].item(), .05, places=6)
        self.assertAlmostEqual(metrics['sdf_inside_uncertainty_mae'].item(), np.exp(-4), places=6)
        self.assertAlmostEqual(metrics['sdf_surface_uncertainty_mae'].item(), .05 - np.exp(-4), places=6)
        self.assertGreater(metrics['signed_near_surface_zero_l1'].item(), 0.)

    def test_repair_config_allows_explicit_longer_budget_without_stretching_curriculum(self):
        cfg = load_config()
        cfg['train']['max_updates'] = 50000
        validate_config(cfg)
        self.assertEqual(cfg['repair']['curriculum_updates'], [600, 1300])
        cfg['data']['sdf_queries'] = 2049
        with self.assertRaisesRegex(ValueError, 'divisible by four'):
            validate_config(cfg)


if __name__ == '__main__':
    unittest.main()
