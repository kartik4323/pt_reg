"""Synthetic adapter tests; these are not results from the VM dataset."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from diagnostics.reassembly_field.__main__ import parse_args
from diagnostics.reassembly_field import probes
from diagnostics.reassembly_field.runner import (FINGERPRINT, build_jobs, conditional_256_needed,
                                                  finalize, selected_records, source_documents, validate_resume)
from diagnostics.reassembly_v2.runtime import read, sha256, write
from reassembly.geometry import so3_exp
from reassembly.model import ReassemblyModel
from reassembly.repair.fields import ContinuousNeuralField


class Sphere:
    truncation = .1

    def __init__(self):
        self.queried = 0

    def values(self, xyz, *, untruncated=False):
        self.queried += len(xyz)
        d = np.linalg.norm(xyz, axis=-1) - .3
        return d if untruncated else np.clip(d, -.1, .1), np.full(len(xyz), .01)

    def raw_distance(self, xyz):
        return self.values(xyz, untruncated=True)[0]

    def sample(self, xyz):
        norm = np.linalg.norm(xyz, axis=-1)
        d = norm - .3
        gradient = xyz / np.maximum(norm[:, None], 1e-12)
        gradient[np.abs(d) >= .1] = 0
        return np.clip(d, -.1, .1), np.full(len(xyz), .5), gradient


def sample_fixture():
    rng = np.random.default_rng(19)
    n = 96
    surface = rng.uniform(-.12, .12, (n, 3)).astype(np.float32)
    surface[:72, 2] = 0
    canonical = np.stack((surface, surface + rng.normal(0, .0002, surface.shape), np.zeros_like(surface))).astype(np.float32)
    r = np.stack((np.eye(3), so3_exp([.2, -.3, .1]), np.eye(3))).astype(np.float32)
    t = np.array([[0, 0, 0], [.07, -.02, .06], [0, 0, 0]], np.float32)
    points = np.einsum('fni,fij->fnj', canonical - t[:, None], r).astype(np.float32)
    labels = np.zeros((3, n), np.float32)
    labels[:2, :72] = 1
    interfaces = np.full((3, n), -1, np.int64)
    interfaces[:2, :72] = 0
    q = rng.uniform(-.4, .4, (48, 3)).astype(np.float32)
    return {'points': torch.from_numpy(points), 'canonical_points': torch.from_numpy(canonical),
            'rotations_gt': torch.from_numpy(r), 'translations_gt': torch.from_numpy(t),
            'fracture_labels': torch.from_numpy(labels), 'interface_ids': torch.from_numpy(interfaces),
            'fragment_mask': torch.tensor([True, True, False]), 'anchor_index': torch.tensor(0),
            'sdf_queries': torch.from_numpy(q), 'sdf_near_mask': torch.arange(48) < 24}


def fixture_config():
    return {'train': {'contact_radius': .05, 'contact_sigma': .01},
            'solver': {'success_threshold': .01, 'min_pair_mass': .05, 'min_correspondence_weight': .001,
                       'prior_weight': .25, 'refinement_iterations': 2}}


class FieldDiagnosticTests(unittest.TestCase):
    def test_default_has_no_deadline_and_rejects_dataset_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(['--managed-root', directory, '--device', 'cpu'])
            self.assertIsNone(args.hours)
            self.assertIn('field_diagnostics', args.output.parts)
            with self.assertRaises(SystemExit), patch('sys.stderr', new=io.StringIO()):
                parse_args(['--managed-root', directory, '--output', str(Path(directory) / 'bottles498/x')])
            with self.assertRaises(SystemExit), patch('sys.stderr', new=io.StringIO()):
                parse_args(['--managed-root', directory, '--hours', 'nan'])

    def test_same_recorded_pose_starts_proper_reference_fixed(self):
        sample = sample_fixture()
        starts = list(probes.pose_starts(sample))
        self.assertEqual([j['degrees'] for j, _ in starts], [0, 5, 15])
        for meta, (r, t) in starts:
            np.testing.assert_array_equal(r[0], np.eye(3))
            np.testing.assert_array_equal(t[0], np.zeros(3))
            np.testing.assert_allclose(np.linalg.det(r), 1., atol=1e-6)
        np.testing.assert_allclose(starts[0][1][0], sample['rotations_gt'][:2])
        repeated = list(probes.pose_starts(sample))
        np.testing.assert_array_equal(starts[-1][1][0], repeated[-1][1][0])

    def test_oracle_contacts_use_original_independent_points(self):
        sample = sample_fixture()
        contacts, info = probes.oracle_contacts(sample, fixture_config())
        self.assertTrue(info['connected'])
        self.assertEqual(len(contacts), 1)
        p = contacts[0]
        np.testing.assert_array_equal(p['source'], sample['points'][0, p['source_indices']])
        np.testing.assert_array_equal(p['target'], sample['points'][1, p['target_indices']])
        moved = p['source'] @ sample['rotations_gt'][0].numpy().T + sample['translations_gt'][0].numpy()
        target = p['target'] @ sample['rotations_gt'][1].numpy().T + sample['translations_gt'][1].numpy()
        self.assertGreater(float(np.linalg.norm(moved - target)), 0.)
        self.assertTrue(np.all(sample['interface_ids'][0, p['source_indices']].numpy() == 0))

    def test_noncontacting_and_collinear_support_do_not_become_oracle_contacts(self):
        sample = sample_fixture()
        sample['interface_ids'][1, :72] = 1
        contacts, info = probes.oracle_contacts(sample, fixture_config())
        self.assertEqual(contacts, [])
        self.assertFalse(info['connected'])
        sample = sample_fixture()
        sample['points'][:, :, 1:] = 0
        contacts, info = probes.oracle_contacts(sample, fixture_config())
        self.assertEqual(contacts, [])

    def test_grid_construction_resumes_fresh_nodes(self):
        sphere = Sphere()
        with tempfile.TemporaryDirectory() as directory:
            calls = [0]
            def stop():
                calls[0] += 1
                if calls[0] == 3:
                    raise TimeoutError('fixture interrupt')
            with self.assertRaises(TimeoutError):
                probes.fresh_grid_nodes(sphere, 4, directory, 'same-input', stop, lambda *_: None, chunk=8)
            self.assertEqual(read(Path(directory) / 'cursor.json')['completed_nodes'], 16)
            distance, sigma, bounds, profile = probes.fresh_grid_nodes(sphere, 4, directory, 'same-input', lambda: None, lambda *_: None, chunk=8)
            self.assertEqual(profile['resumed_nodes'], 16)
            self.assertEqual(sphere.queried, 64)
            xyz = np.stack(np.meshgrid(*([np.linspace(-2.25, 2.25, 4)] * 3), indexing='ij'), -1)
            np.testing.assert_allclose(distance, np.linalg.norm(xyz, axis=-1) - .3, atol=1e-6)
            del distance, sigma
            with self.assertRaises(RuntimeError):
                probes.fresh_grid_nodes(sphere, 4, directory, 'changed-input', lambda: None, lambda *_: None, chunk=8)

    def test_clamp_order_is_a_real_control(self):
        # A linear untruncated SDF stays exact under interpolation; preclipping
        # nodes changes the interpolant even within its central truncation band.
        axis = np.linspace(-1, 1, 4)
        raw = np.broadcast_to(axis[:, None, None], (4, 4, 4)).copy()
        variants = probes.grid_variants(raw, np.zeros_like(raw), np.array([[-1]*3, [1]*3]), .1)
        q = np.array([[.04, .01, .02]])
        before = variants['clamp_before'].sample(q)
        after = variants['clamp_after'].sample(q)
        self.assertGreater(abs(before[0][0] - .04), .02)
        np.testing.assert_allclose(after[0], .04, atol=1e-8)
        np.testing.assert_allclose(after[2], [[1, 0, 0]], atol=1e-8)

    def test_256_only_when_recorded_128_interpolation_error_exceeds_limit(self):
        previous = {'variants': {v: {'field_metrics': {'observed_original_surface': {'interpolation_mae': .001}}}
                                 for v in ('clamp_before', 'clamp_after')}}
        self.assertFalse(conditional_256_needed(previous)[0])
        previous['variants']['clamp_after']['field_metrics']['observed_original_surface']['interpolation_mae'] = .00101
        self.assertTrue(conditional_256_needed(previous)[0])

    def test_regional_metrics_separate_interpolation_from_reconstruction(self):
        field = Sphere()
        sample = sample_fixture()
        q, regions = probes.probe_queries(sample)
        metrics, arrays = probes.field_metrics(field, field, field, q, regions)
        self.assertEqual(metrics['near_surface']['interpolation_mae'], 0.)
        self.assertEqual(metrics['surrounding']['mae'], 0.)
        self.assertEqual(metrics['observed_original_surface']['count'], 48)
        self.assertEqual(len(arrays['query_xyz']), 96)

    def test_refinement_compares_fixed_starts_and_keeps_reference(self):
        sample = sample_fixture()
        contacts, status = probes.oracle_contacts(sample, fixture_config())
        encoded = {'fracture_logits': torch.zeros((1, 3, 96))}
        output = probes.refine_probes(Sphere(), sample, encoded, contacts, status, fixture_config())
        self.assertEqual(len(output), 24)
        for row in output:
            self.assertFalse(row['production_acceptance'])
            np.testing.assert_array_equal(row['before_rotations'][0], row['after_rotations'][0])
            np.testing.assert_array_equal(row['before_translations'][0], row['after_translations'][0])

    def test_conditioning_holds_sigma_and_reproduces_original_probabilities(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        cfg = {'model': {'dim': 16, 'heads': 4, 'sample_counts': [16, 8, 4], 'neighbors': 4, 'contact_points': 12}}
        model = ReassemblyModel(cfg).eval()
        points = torch.randn(1, 3, 32, 3) * .1
        with torch.no_grad():
            encoded = model.encode(points, torch.tensor([[1, 1, 0]], dtype=torch.bool), torch.tensor([0]))
        predicted = ContinuousNeuralField(model, encoded)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        report, arrays = probes.conditioning(model, encoded, predicted, Sphere())
        self.assertTrue(report['unchanged_path_reproduced'])
        self.assertEqual(set(report['controls']), {'predicted', 'gt', 'perturbed', 'disabled'})
        self.assertIn('fixed_sigma', arrays)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_exact_subset_from_original_bundle_without_extraction(self):
        records = [{'pattern_id': str(i), 'source_id': f's{i}', 'split': 'val', 'band': 'easy', 'cut_family': 'smooth'} for i in range(48)]
        subset = list(reversed(records[:12]))
        docs = {'inventory.json': {'dataset_integrity': {'dataset_fingerprint': FINGERPRINT}},
                'experiments.json': {'validation_subset': subset}, 'summary.json': {'status': 'complete'}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.tar.gz'
            with tarfile.open(path, 'w:gz') as archive:
                for name, value in docs.items():
                    data = json.dumps(value).encode()
                    entry = tarfile.TarInfo(name)
                    entry.size = len(data)
                    archive.addfile(entry, io.BytesIO(data))
            indices, selection, hashes = selected_records(types.SimpleNamespace(records=records), path)
            self.assertEqual(indices, list(reversed(range(12))))
            self.assertEqual(len(list(Path(directory).iterdir())), 1)
            self.assertEqual(hashes['source_bundle']['sha256'], sha256(path))
            jobs = build_jobs(types.SimpleNamespace(records=records), indices)
            self.assertEqual(len(jobs), 144)
            self.assertEqual(len({j['id'] for j in jobs}), 144)

    def test_resume_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint.pt'
            checkpoint.write_bytes(b'first')
            write(root / 'inventory.json', {'checkpoints': {'test': {'path': str(checkpoint), 'sha256': sha256(checkpoint)}}})
            validate_resume(root)
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'Resume input changed'):
                validate_resume(root)

    def test_partial_report_lists_unrun_and_packages_without_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = {'kind': 'continuous', 'id': 'fixture'}
            write(root / 'experiments.json', {'jobs': [job]})
            summary = finalize(root, 'fixture interruption')
            self.assertEqual(summary['kind'], 'focused_field_diagnostic')
            self.assertEqual(summary['status'], 'partial_or_blocked')
            self.assertEqual(summary['unrun_jobs'], [job])
            with tarfile.open(root / 'field_diagnostic_bundle.tar.gz') as archive:
                self.assertIn('REPORT.md', archive.getnames())
                self.assertFalse(any(name.endswith('.pt') for name in archive.getnames()))

    def test_worker_dispatches_incremental_job_without_loading_vm_data(self):
        # Exercise worker orchestration against explicitly mocked artifact/data
        # boundaries. This fixture never opens a real training checkpoint.
        from contextlib import ExitStack
        from diagnostics.reassembly_field import runner
        cfg = fixture_config()
        cfg['train']['seed'] = 42
        cfg['data'] = {'points_per_fragment': 1024}
        cfg['model'] = {'dim': 16, 'heads': 4, 'sample_counts': [16, 8, 4], 'neighbors': 4, 'contact_points': 12}
        model = ReassemblyModel(cfg).eval()
        sample = sample_fixture()
        sample.update(pattern_id='fixture', source_id='synthetic', band='easy', cut_family='fixture',
                      target_points=sample['canonical_points'][0])
        class Dataset:
            fingerprint = FINGERPRINT
            records = [{'pattern_id': 'fixture', 'source_id': 'synthetic', 'band': 'easy'}]
            def __len__(self):
                return 48
            def __getitem__(self, index):
                return sample
        job = {'id': 'fixture_condition', 'kind': 'conditioning', 'index': 0, 'pattern_id': 'fixture',
               'source_id': 'synthetic', 'band': 'easy', 'checkpoint': runner.CHECKPOINT}
        jobs = [job, {**job, 'id': 'fixture_continuous', 'kind': 'continuous', 'field': 'gt'},
                {**job, 'id': 'fixture_grid', 'kind': 'grid', 'field': 'gt', 'resolution': 4}]
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            output = root / 'out'
            output.mkdir()
            write(root / 'bottles498/eval/test/predicted/evaluation.json', {'config': cfg})
            inventory = {'dataset_integrity': {'dataset_fingerprint': FINGERPRINT},
                         'checkpoints': {runner.CHECKPOINT: {'sha256': 'fixture-only'}}, 'files': {}}
            stack.enter_context(patch.object(runner, 'Limits', return_value=types.SimpleNamespace(deadline=None, check=lambda: None)))
            stack.enter_context(patch.object(runner, 'original_inventory', return_value=inventory))
            stack.enter_context(patch.object(runner, 'FractureDataset', return_value=Dataset()))
            stack.enter_context(patch.object(runner, 'selected_records', return_value=([0], {'selection': 'fixture'}, {})))
            stack.enter_context(patch.object(runner, 'build_jobs', return_value=jobs))
            stack.enter_context(patch.object(runner, 'save_visual'))
            stack.enter_context(patch.object(runner, 'load_checkpoint', return_value={'cfg': cfg, 'model': model.state_dict()}))
            stack.enter_context(patch.object(runner.ContinuousGTField, 'from_sample', return_value=Sphere()))
            stack.enter_context(patch.object(runner, 'make_oracle_field', return_value=Sphere().raw_distance))
            for name in ('sample_contract', 'check_adapter', 'matching_contract'):
                stack.enter_context(patch.object(runner, name, return_value={'passed': True}))
            runner.run(root, output, torch.device('cpu'))
            result = read(output / 'results/fixture_condition.json')
            self.assertTrue(result['conditioning']['unchanged_path_reproduced'])
            self.assertFalse(result['production_acceptance'])
            self.assertTrue((output / 'tensors/fixture_condition.npz').is_file())
            grid = read(output / 'results/fixture_grid.json')
            self.assertEqual(set(grid['variants']), {'clamp_before', 'clamp_after'})
            self.assertEqual(len(grid['variants']['clamp_before']['refinement']), 24)
            self.assertFalse((output / 'scratch/fixture_grid').exists())


if __name__ == '__main__':
    unittest.main()
