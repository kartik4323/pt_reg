"""Tiny real-model lifecycle tests; synthetic geometry is never research evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import trimesh

from reassembly.prepare import _file_sha256
from reassembly.training import synthetic_batch
from reassembly.repair import training, evaluation
from reassembly.repair.checkpoints import load_checkpoint
from reassembly.repair.config import load_config
from reassembly.repair.model import RepairModel, configure_stage


FINGERPRINT = 'synthetic-repair-workflow-fixture'


def tiny_config():
    cfg = load_config()
    cfg['model'].update(dim=16, sample_counts=[8, 4, 2], neighbors=4, contact_points=8)
    cfg['data'].update(points_per_fragment=20, sdf_queries=16, min_sources=1)
    cfg['train'].update(batch_size=1, grad_accum_steps=1, max_updates=1,
                        overfit_updates=1, validation_interval=1, amp=False)
    cfg['solver'].update(field_chunk=8)
    return cfg


class FixtureDataset:
    def __init__(self, manifest, split, cfg, stage=1, fixed=False, limit=None, **kwargs):
        document = json.loads(Path(manifest).read_text())
        self.records = [r for r in document['patterns'] if split == 'all' or r['split'] == split]
        if limit is not None:
            self.records = self.records[:limit]
        self.query_cache_hash = None
        self.fingerprint = FINGERPRINT
        self.fixed, self.step, self.cfg = fixed, 0, cfg
        self.source_mesh_path = str(Path(manifest).parent/'source.npz')
        with torch.random.fork_rng():
            torch.manual_seed(199)
            self.batch = synthetic_batch(cfg, 1, torch.device('cpu'))
        # Re-express targets in the centered reference, preserving exact identity.
        reference_center = self.batch['translations_gt'][:, :1].clone()
        self.batch['canonical_points'] -= reference_center[:, :, None]
        self.batch['translations_gt'] -= reference_center

    def __len__(self):
        return len(self.records)

    def set_step(self, step):
        self.step = step

    def __getitem__(self, index):
        sample = {key: value[0].clone() for key, value in self.batch.items()}
        count, n, _ = sample['points'].shape
        rng = np.random.default_rng(902 + index + (0 if self.fixed else self.step*13))
        other = {key: sample[key].clone() for key in ('points', 'fragment_mask', 'anchor_index', 'canonical_points', 'fracture_labels', 'interface_ids')}
        for part in range(count):
            ids = rng.permutation(n)
            for key in ('points', 'canonical_points', 'fracture_labels', 'interface_ids'):
                other[key][part] = other[key][part, ids]
        sample['view2'] = other
        groups = torch.arange(4).repeat_interleave(self.cfg['data']['sdf_queries']//4)
        values = torch.tensor([0., -.02, .02, .2]).repeat_interleave(len(groups)//4)
        sample.update(sdf_query_group=groups, sdf_values=values,
                      sdf_queries=torch.stack((.4+values, torch.zeros_like(values), torch.zeros_like(values)), -1),
                      sdf_near_mask=groups != 3, pattern_id=self.records[index]['pattern_id'],
                      source_id=self.records[index]['source_id'], band='easy', cut_family='smooth',
                      source_mesh_path=self.source_mesh_path, anchor_source_rotation=torch.eye(3),
                      anchor_source_centroid=torch.zeros(3), shared_scale=torch.tensor(1.))
        return sample


class RepairWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = tiny_config()
        self.manifest = self.root/'manifest.json'
        self.manifest.write_text(json.dumps(dict(fingerprint=FINGERPRINT,
            patterns=[dict(source_id=split+'-source', pattern_id=f'{split}-{i}', split=split, pieces=3)
                      for split, count in [('train', 3), ('val', 2)] for i in range(count)])))
        mesh = trimesh.creation.box(extents=[.8]*3)
        np.savez(self.root/'source.npz', vertices=mesh.vertices, faces=mesh.faces)
        self.preflight = self.root/'preflight.json'; self.preflight.write_text('{}')
        self.diagnostic = self.root/'diagnostic.json'
        self.diagnostic.write_text(json.dumps(dict(kind='focused_field_diagnostic', status='complete',
            dataset_fingerprint=FINGERPRINT, optimizer_updates=0, checkpoint_changes=[], input_changes=[],
            failed_jobs=[], unrun_jobs=[], checkpoint_verification={'fixture': {'unchanged': True, 'before': 'fixture-hash', 'after': 'fixture-hash'}})))
        self.patches = [
            patch.object(training, 'RepairDataset', FixtureDataset),
            patch.object(training, 'dataset_integrity', return_value=dict(dataset_fingerprint=FINGERPRINT, verified_sources=2)),
            patch.object(training, 'require_preflight'),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def run_stage(self, name, stage=1, **kwargs):
        return training.train_stage(self.cfg, self.manifest, self.root/name, stage, 'cpu',
                                    preflight_report=self.preflight, field_diagnostic_report=self.diagnostic, **kwargs)

    def test_fresh_stage1_resume_matches_uninterrupted_updates(self):
        self.run_stage('resumed')
        checkpoint = self.root/'resumed/latest.pt'
        run_id = load_checkpoint(checkpoint)['run_id']
        with self.assertRaisesRegex(ValueError, 'already reached'):
            self.run_stage('resumed', resume=checkpoint)
        self.cfg['train']['max_updates'] = 2
        self.run_stage('resumed', resume=checkpoint)
        self.run_stage('whole')
        resumed, whole = load_checkpoint(checkpoint), load_checkpoint(self.root/'whole/latest.pt')
        self.assertEqual(resumed['step'], 2)
        self.assertEqual(resumed['run_id'], run_id)
        self.assertEqual(resumed['training_lineage']['1']['budget'], 2)
        for key in resumed['model']:
            torch.testing.assert_close(resumed['model'][key], whole['model'][key], atol=0, rtol=0)
        metrics = resumed['metrics']['validation']
        self.assertIn('matching_top1_recall', metrics)
        self.assertIn('assembly_success', metrics)

    def test_legacy_checkpoint_fresh_init_and_cross_run_resume_are_rejected(self):
        old = self.root/'old.pt'; torch.save(dict(architecture='coarse-scaffold-reassembly-v2.1-local-contacts'), old)
        with self.assertRaisesRegex(ValueError, 'Legacy/external'):
            load_checkpoint(old)
        with self.assertRaisesRegex(ValueError, 'starts fresh'):
            self.run_stage('bad', initialize_from=old)
        self.run_stage('first')
        self.cfg['train']['max_updates'] = 2
        with self.assertRaisesRegex(ValueError, 'existing run directory'):
            self.run_stage('different', resume=self.root/'first/latest.pt')

    def test_checkpoint_identity_configuration_and_purpose_are_bound(self):
        self.run_stage('contact')
        checkpoint = self.root/'contact/latest.pt'
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            load_checkpoint(checkpoint, fingerprint='another-dataset')
        with self.assertRaisesRegex(ValueError, 'purposes'):
            load_checkpoint(checkpoint, purpose='overfit')
        with self.assertRaisesRegex(ValueError, 'stage-2'):
            load_checkpoint(checkpoint, stage=2)
        different = copy.deepcopy(self.cfg)
        different['repair']['geometry_variant'] = 'existing'
        with self.assertRaisesRegex(ValueError, 'differs'):
            load_checkpoint(checkpoint, cfg=different)
        state = load_checkpoint(checkpoint)
        state['cfg'] = different
        changed = self.root/'changed.pt'; torch.save(state, changed)
        with self.assertRaisesRegex(ValueError, 'signature is invalid'):
            load_checkpoint(changed)

    def test_stage2_requires_contact_gate_and_does_not_change_contacts(self):
        self.run_stage('contact')
        parent_path = self.root/'contact/best.pt'
        parent = load_checkpoint(parent_path)
        with self.assertRaisesRegex(ValueError, 'contact-gate-report'):
            self.run_stage('field_missing', stage=2, initialize_from=parent_path)
        gate = self.root/'gate.json'; gate.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'gate has not passed'):
            self.run_stage('field_bad', stage=2, initialize_from=parent_path, contact_gate_report=gate)
        # Gate proof itself has a separate tampering test below; isolate actual
        # field optimizer steps here without claiming synthetic assembly passes.
        with patch.object(evaluation, 'require_contact_gate') as certify:
            report = self.run_stage('field', stage=2, initialize_from=parent_path, contact_gate_report=gate)
            certify.assert_called_once()
        after = load_checkpoint(report['best_checkpoint'], stage=2)
        changed_field = False
        for key, value in parent['model'].items():
            if key.startswith(('encoder.', 'matcher.')):
                torch.testing.assert_close(value, after['model'][key], atol=0, rtol=0)
            else:
                changed_field |= not torch.equal(value, after['model'][key])
        self.assertTrue(changed_field)
        self.assertIn('near_surface_sdf_l1', after['metrics']['validation'])
        self.assertIn('sdf_inside_count', after['metrics']['validation'])
        self.assertEqual(after['training_lineage']['parent_checkpoint']['sha256'], _file_sha256(parent_path))

    def test_stage2_resume_preserves_optimizer_state_and_frozen_contact_lineage(self):
        self.run_stage('contact')
        parent_path = self.root/'contact/best.pt'
        gate = self.root/'gate.json'; gate.write_text('{}')
        with patch.object(evaluation, 'require_contact_gate'):
            self.run_stage('field_resume', stage=2, initialize_from=parent_path, contact_gate_report=gate)
            self.cfg['train']['max_updates'] = 2
            self.run_stage('field_resume', stage=2, resume=self.root/'field_resume/latest.pt', contact_gate_report=gate)
            self.run_stage('field_whole', stage=2, initialize_from=parent_path, contact_gate_report=gate)
        resumed = load_checkpoint(self.root/'field_resume/latest.pt')
        whole = load_checkpoint(self.root/'field_whole/latest.pt')
        parent = load_checkpoint(parent_path)
        for key, value in resumed['model'].items():
            torch.testing.assert_close(value, whole['model'][key], atol=0, rtol=0)
            if key.startswith(('encoder.', 'matcher.')):
                torch.testing.assert_close(value, parent['model'][key], atol=0, rtol=0)
        self.assertEqual(resumed['training_lineage']['parent_checkpoint'], whole['training_lineage']['parent_checkpoint'])

    def test_preflight_executes_real_stage_updates_and_continuous_query_gradient(self):
        from reassembly.repair.fields import ContinuousNeuralField
        original = ContinuousNeuralField.sample
        sampled = []
        def observe(field, points):
            value = original(field, points)
            sampled.append(value)
            return value
        with patch('reassembly.validation.run_correctness_checks', return_value=dict(passed=True)), \
             patch.object(training, 'optimizer_update', wraps=training.optimizer_update) as updates, \
             patch.object(ContinuousNeuralField, 'sample', observe):
            report = training.preflight(self.cfg, self.manifest, self.root/'actual_preflight', 'cpu')
        self.assertEqual(report['status'], 'cpu_verified_cuda_unmeasured')
        self.assertEqual(updates.call_count, 2)
        self.assertEqual([value['stage'] for value in report['attempts'][0]['stages']], [1, 2])
        self.assertTrue(sampled)
        self.assertTrue(all(np.isfinite(value).all() for triple in sampled for value in triple))
        self.assertGreater(np.linalg.norm(sampled[0][2]), 0)

    def test_paired_fields_reuse_candidates_and_export_identical_coverage(self):
        model = RepairModel(self.cfg).eval()
        configure_stage(model, 3)
        sample = FixtureDataset(self.manifest, 'val', self.cfg, fixed=True)[0]
        with patch('reassembly.repair.evaluation.sample_results', wraps=evaluation.sample_results):
            rows = evaluation.sample_results(model, sample, self.cfg, 'cpu', ('contact_only', 'predicted', 'gt', 'perturbed'))
        fingerprints = {row['diagnostics']['candidate_fingerprint'] for row, result in rows.values()}
        self.assertEqual(len(fingerprints), 1)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all('candidate_oracle_success' in row['candidate_coverage'] for row, _ in rows.values()))
        self.assertTrue(all('source_id' in row and 'pairs' in row for row, _ in rows.values()))

    def test_incomplete_contact_gate_proof_is_rejected(self):
        checkpoint = self.root/'checkpoint.bin'; checkpoint.write_bytes(b'explicit fixture checkpoint')
        gate = self.root/'gate.json'
        gate.write_text(json.dumps(dict(kind='repair_contact_gate', passed=True,
            checkpoint_sha256=_file_sha256(checkpoint), dataset_fingerprint=FINGERPRINT,
            checks={name: True for name in ('fixed_16', 'robust_rotation', 'robust_samples', 'robust_samples_poses', 'training_80_percent')},
            training_success=.8, robustness={}, evidence={})))
        with self.assertRaises(ValueError):
            evaluation.require_contact_gate(gate, checkpoint, FINGERPRINT)


if __name__ == '__main__':
    unittest.main()
