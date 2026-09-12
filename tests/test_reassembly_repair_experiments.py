"""Experiment orchestration and evidence-contract tests; no training steps."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reassembly.prepare import _file_sha256
from reassembly.resources import write_json
from reassembly.repair import ARCHITECTURE, SCHEMA_VERSION
from reassembly.repair import reporting, workflow
from reassembly.repair.__main__ import dispatch, parser
from reassembly.repair.checkpoints import load_checkpoint, require_resume_config
from reassembly.repair.config import load_config, signature, variant_config


FINGERPRINT = 'synthetic-experiment-fixture'
CONDITIONS = ('contact_only', 'predicted', 'gt', 'perturbed')
SEEDS = (4101, 4102, 4103)


def evaluation_fixture(root, training_seed=42, *, configuration=None):
    """Real minimal v3 checkpoint metadata, wholly synthetic evaluation rows."""
    root = Path(root)
    cfg = copy.deepcopy(configuration or load_config())
    cfg['train']['seed'] = training_seed
    weights = root / f'seed{training_seed}' / 'scaffold'
    weights.mkdir(parents=True)
    write_json(weights / 'config.resolved.json', cfg)
    state = {'schema_version': SCHEMA_VERSION, 'architecture': ARCHITECTURE,
             'cfg': cfg, 'model': {}, 'optimizer': {}, 'scaler': {}, 'random_state': {},
             'stage': 2, 'step': cfg['train']['max_updates'], 'run_id': f'fixture-{training_seed}',
             'purpose': 'experiment', 'dataset_fingerprint': FINGERPRINT,
             'model_signature': signature(cfg, model_only=True), 'training_lineage': {},
             'metrics': {}, 'query_cache_hash': None}
    checkpoint = weights / 'best.pt'
    torch.save(state, checkpoint)
    output = root / f'seed{training_seed}' / 'evaluation'
    output.mkdir()
    patterns = [f'pattern{i:02d}' for i in range(48)]
    rows = []
    for seed in SEEDS:
        for i, pattern in enumerate(patterns):
            fingerprint = hashlib.sha256(f'{pattern}/{seed}'.encode()).hexdigest()
            for condition in CONDITIONS:
                success = i < {'contact_only': 39, 'predicted': 40, 'gt': 42, 'perturbed': 30}[condition]
                rows.append({'source_id': f'source{i // 8}', 'pattern_id': pattern,
                             'mode': 'samples_poses', 'seed': seed, 'condition': condition,
                             'candidate_coverage': {'candidate_fingerprint': fingerprint},
                             'success': success, 'status': 'ok' if success else 'failed',
                             'failed': not success, 'whole_chamfer': (.005 if condition == 'predicted' else .006) if success else None,
                             'band': 'easy', 'pieces': 2, 'pairs': [{'top1_recall': .8}]})
    report = {'kind': 'repair_evaluation', 'architecture': ARCHITECTURE, 'status': 'completed',
              'split': 'val', 'purpose': 'experiment', 'checkpoint': str(checkpoint),
              'checkpoint_sha256': _file_sha256(checkpoint), 'checkpoint_unchanged': True,
              'dataset_fingerprint': FINGERPRINT, 'config_signature': signature(cfg),
              'metric_threshold': .01, 'seeds': list(SEEDS), 'modes': ['samples_poses'],
              'conditions': list(CONDITIONS), 'pattern_ids': patterns}
    path = output / 'evaluation.json'
    save_evidence(path, report, rows)
    return path, report, rows


def save_evidence(path, report, rows):
    rows_path = Path(path).parent / 'examples.jsonl'
    rows_path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    report['results_sha256'] = _file_sha256(rows_path)
    write_json(path, report)


class RepairEvidenceTests(unittest.TestCase):
    def test_acceptance_requires_three_fresh_seeds_and_all_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [evaluation_fixture(directory, seed)[0] for seed in (42, 43, 44)]
            result = reporting.acceptance(paths, directory)
            self.assertTrue(result['passed'])
            self.assertEqual({r['training_seed'] for r in result['runs']}, {42, 43, 44})
            for run in result['runs']:
                self.assertEqual({c['count'] for c in run['checks'].values()}, {48})
                self.assertEqual(run['benefit']['paired_count'], 144)
                self.assertEqual(run['benefit']['success_source_bootstrap']['sources'], 6)

    def test_duplicate_observations_do_not_substitute_for48_distinct_patterns(self):
        with tempfile.TemporaryDirectory() as directory:
            path, report, rows = evaluation_fixture(directory)
            # Keep the row count unchanged, replacing pattern47 with another
            # observation of pattern00. A count-only gate would accept this.
            for row in rows:
                if row['pattern_id'] == 'pattern47':
                    row['pattern_id'], row['source_id'] = 'pattern00', 'source0'
                    row['success'] = True
            save_evidence(path, report, rows)
            with self.assertRaises(ValueError):
                reporting.read_rows(path)

    def test_missing_declared_condition_or_seed_is_incomplete_evidence(self):
        for condition in ('gt', 'perturbed'):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as directory:
                path, report, rows = evaluation_fixture(directory)
                rows = [r for r in rows if not (r['condition'] == condition and r['seed'] == 4103)]
                save_evidence(path, report, rows)
                with self.assertRaises(ValueError):
                    reporting.read_rows(path)

    def test_duplicate_training_seed_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = evaluation_fixture(directory)[0]
            with self.assertRaises(ValueError):
                reporting.acceptance([path, path, path], directory)

    def test_training_repetitions_use_identical_validation_pattern_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [evaluation_fixture(directory, seed)[0] for seed in (42, 43)]
            path, report, rows = evaluation_fixture(directory, 44)
            report['pattern_ids'] = ['other_' + p for p in report['pattern_ids']]
            for row in rows:
                row['pattern_id'] = 'other_' + row['pattern_id']
            save_evidence(path, report, rows)
            with self.assertRaises(ValueError):
                reporting.acceptance(paths + [path], directory)

    def test_pattern_source_identity_is_consistent_in_all_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            path, report, rows = evaluation_fixture(directory)
            rows[0]['source_id'] = 'another-source'
            save_evidence(path, report, rows)
            with self.assertRaises(ValueError):
                reporting.read_rows(path)

    def test_comparison_configuration_must_match_beyond_variant_names(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [evaluation_fixture(directory, seed)[0] for seed in (42, 43)]
            changed = load_config()
            changed['train']['learning_rate'] *= 2
            paths.append(evaluation_fixture(directory, 44, configuration=changed)[0])
            with self.assertRaises(ValueError):
                reporting.acceptance(paths, directory)

    def test_checkpoint_and_configuration_cannot_drift_after_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, report, rows = evaluation_fixture(directory)
            with Path(report['checkpoint']).open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaises(ValueError):
                reporting.read_rows(path)
        with tempfile.TemporaryDirectory() as directory:
            path, report, _ = evaluation_fixture(directory)
            config = Path(report['checkpoint']).parent / 'config.resolved.json'
            cfg = json.loads(config.read_text())
            cfg['train']['learning_rate'] *= 2
            write_json(config, cfg)
            with self.assertRaises(ValueError):
                reporting.acceptance([path], directory)

    def test_weakened_metric_threshold_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path, report, rows = evaluation_fixture(directory)
            report['metric_threshold'] = .1
            save_evidence(path, report, rows)
            with self.assertRaises(ValueError):
                reporting.read_rows(path)

    def test_paired_candidates_and_unique_keys_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, rows = evaluation_fixture(directory)
            pair = [r for r in rows if r['pattern_id'] == 'pattern00' and r['seed'] == 4101 and r['condition'] in ('contact_only', 'predicted')]
            self.assertEqual(reporting.paired_benefit(pair)['paired_count'], 1)
            with self.assertRaises(ValueError):
                reporting.paired_benefit(pair + [copy.deepcopy(pair[0])])
            missing = copy.deepcopy(pair)
            for r in missing:
                r['candidate_coverage'] = {}
            with self.assertRaises(ValueError):
                reporting.paired_benefit(missing)
            changed = copy.deepcopy(pair)
            changed[1]['candidate_coverage']['candidate_fingerprint'] = '0' * 64
            with self.assertRaises(ValueError):
                reporting.paired_benefit(changed)

    def test_failure_denominators_and_source_bootstrap(self):
        rows = []
        for i, (before, after) in enumerate(((False, False), (False, True), (True, True))):
            for condition, success in (('contact_only', before), ('predicted', after)):
                rows.append({'source_id': f's{i % 2}', 'pattern_id': str(i), 'mode': 'samples_poses', 'seed': 4101,
                             'condition': condition, 'success': success, 'whole_chamfer': .005 if success else None,
                             'candidate_coverage': {'candidate_fingerprint': 'a' * 64}})
        result = reporting.paired_benefit(rows)
        self.assertEqual(result['paired_count'], 3)
        self.assertEqual(result['chamfer_paired_solved_count'], 1)
        self.assertAlmostEqual(result['success_delta'], 1 / 3)
        # SourceA has many correlated fractures; it must not dominate sourceB.
        bootstrap = reporting.source_bootstrap({'A': [1.] * 100, 'B': [-1.]})
        self.assertEqual(bootstrap['mean'], 0.)
        self.assertEqual(bootstrap['sources'], 2)
        self.assertEqual(bootstrap, reporting.source_bootstrap({'A': [1.] * 100, 'B': [-1.]}))

    def test_legacy_initialization_and_changed_resume_seed_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pt'
            torch.save({'architecture': 'coarse-scaffold-reassembly-v2.1-local-contacts', 'schema_version': 2}, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)
        cfg = load_config()
        changed = copy.deepcopy(cfg)
        changed['train']['seed'] = 43
        with self.assertRaises(ValueError):
            require_resume_config({'cfg': cfg}, changed)


class RepairWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = load_config()
        self.manifest = self.root / 'prepared/manifest.json'
        write_json(self.manifest, {'fixture': True})
        self.diagnostic = self.root / 'field/summary.json'
        write_json(self.diagnostic, {'kind': 'focused_field_diagnostic', 'status': 'complete',
                                    'dataset_fingerprint': FINGERPRINT, 'optimizer_updates': 0,
                                    'checkpoint_changes': [], 'input_changes': [], 'failed_jobs': [], 'unrun_jobs': [],
                                    'checkpoint_verification': {'fixture': {'before': 'a', 'after': 'a', 'unchanged': True}}})
        self.cache = self.root / 'cache/query_cache.json'
        write_json(self.cache, {'fixture': True})

    def tearDown(self):
        self.temp.cleanup()

    def experiment(self, cfg, manifest, root, device, diagnostic, cache, resume, guard):
        geometry, views = cfg['repair']['geometry_variant'], cfg['repair']['view_supervision']
        score = .5 + .1 * (geometry == 'revised') + .1 * (views == 'resampled_contrastive')
        return {'variant': workflow.label(geometry, views), 'geometry_variant': geometry,
                'view_supervision': views, 'training_seed': cfg['train']['seed'], 'root': str(root),
                'gate_passed': True, 'source_macro_success': score, 'contact_success': score,
                'matching_top1_recall': score, 'checkpoint': str(root / 'contacts/best.pt'), 'config': cfg}

    def run_phase(self, phase, output=None, resume=True):
        return workflow.run(self.cfg, self.manifest, output or self.root / 'workflow', 'cpu', phase=phase,
                            field_diagnostic_report=self.diagnostic, query_cache=self.cache, resume=resume)

    def test_factorial_comparison_and_selected_control_replication(self):
        with patch.object(workflow, 'dataset_integrity', return_value={'dataset_fingerprint': FINGERPRINT}), \
             patch.object(workflow, 'render_and_bundle', return_value={}), \
             patch.object(workflow, '_contact_experiment', side_effect=self.experiment) as experiment:
            selection = self.run_phase('comparisons')
            calls = experiment.call_args_list
            self.assertEqual(len(calls), 4)
            self.assertEqual({(c.args[0]['repair']['geometry_variant'], c.args[0]['repair']['view_supervision']) for c in calls}, set(workflow.VARIANTS))
            self.assertEqual({c.args[0]['train']['seed'] for c in calls}, {42})
            self.assertEqual({c.args[0]['train']['max_updates'] for c in calls}, {self.cfg['train']['max_updates']})
            self.assertEqual(selection['selected'], 'revised__resampled_contrastive')
            experiment.reset_mock()
            self.run_phase('replicate')
            self.assertEqual(len(experiment.call_args_list), 4)
            self.assertEqual({(c.args[0]['repair']['geometry_variant'], c.args[0]['repair']['view_supervision'], c.args[0]['train']['seed']) for c in experiment.call_args_list},
                             {(g, v, seed) for g, v in (('existing', 'existing'), ('revised', 'resampled_contrastive')) for seed in (43, 44)})

    def test_selected_unchanged_control_is_not_replicated_twice(self):
        def tied(*args):
            return dict(self.experiment(*args), source_macro_success=.8, contact_success=.8, matching_top1_recall=.8)
        with patch.object(workflow, 'dataset_integrity', return_value={'dataset_fingerprint': FINGERPRINT}), \
             patch.object(workflow, 'render_and_bundle', return_value={}), \
             patch.object(workflow, '_contact_experiment', side_effect=tied) as experiment:
            selected = self.run_phase('comparisons')
            self.assertEqual(selected['selected'], 'existing__existing')
            experiment.reset_mock()
            self.run_phase('replicate')
            self.assertEqual(len(experiment.call_args_list), 2)
            self.assertEqual({c.args[0]['train']['seed'] for c in experiment.call_args_list}, {43, 44})

    def test_fresh_contact_training_never_inherits_checkpoint(self):
        with patch.object(workflow, 'train_stage', return_value={'status': 'fixture'}) as train:
            workflow._train(self.cfg, self.manifest, self.root / 'fresh', 1, 'cpu', 'experiment', self.diagnostic,
                            self.root / 'profile.json', self.cache, True, None)
            self.assertIsNone(train.call_args.kwargs['initialize_from'])
            self.assertIsNone(train.call_args.kwargs['resume'])

    def test_workflow_rejects_changed_diagnostic_proof_on_resume(self):
        with patch.object(workflow, 'dataset_integrity', return_value={'dataset_fingerprint': FINGERPRINT}), \
             patch.object(workflow, 'render_and_bundle', return_value={}), \
             patch.object(workflow, '_contact_experiment', side_effect=self.experiment):
            self.run_phase('comparisons')
            document = json.loads(self.diagnostic.read_text())
            document['changed'] = True
            write_json(self.diagnostic, document)
            with self.assertRaises(ValueError):
                self.run_phase('replicate')

    def test_failed_contact_selection_prevents_later_phases(self):
        def fail(*args):
            return dict(self.experiment(*args), gate_passed=False)
        with patch.object(workflow, 'dataset_integrity', return_value={'dataset_fingerprint': FINGERPRINT}), \
             patch.object(workflow, 'render_and_bundle', return_value={}), \
             patch.object(workflow, '_contact_experiment', side_effect=fail):
            result = self.run_phase('comparisons')
            self.assertEqual(result['status'], 'contact_gate_failed')
            with self.assertRaises(ValueError):
                self.run_phase('scaffold')

    def test_completed_training_fast_reuse_rejects_stale_proof(self):
        directory = self.root / 'completed'
        directory.mkdir()
        (directory / 'best.pt').write_bytes(b'fixture-weights')
        write_json(directory / 'config.resolved.json', self.cfg)
        write_json(directory / 'training_report.json', {'status': 'completed', 'updates': self.cfg['train']['max_updates'],
                    'best_checkpoint_sha256': _file_sha256(directory / 'best.pt'),
                    'field_diagnostic_sha256': 'stale-proof'})
        with patch.object(workflow, 'train_stage', side_effect=AssertionError('Training must not run')):
            with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
                workflow._train(self.cfg, self.manifest, directory, 1, 'cpu', 'experiment', self.diagnostic,
                                self.root / 'profile.json', self.cache, True, None)

    def test_run_proofs_bind_original_code_and_preflight_diagnostic_files(self):
        from reassembly.repair import training
        run = self.root / 'proof_run'
        run.mkdir()
        profile = self.root / 'proof_profile.json'
        write_json(profile, {'kind': 'synthetic-preflight'})
        metadata = {'provenance': {'code': {'sha256': 'fixture-code'}},
                    'preflight_path': str(profile), 'preflight_sha256': _file_sha256(profile),
                    'field_diagnostic_path': str(self.diagnostic), 'field_diagnostic_sha256': _file_sha256(self.diagnostic),
                    'contact_gate_path': None, 'contact_gate_sha256': None}
        write_json(run / 'run.json', metadata)
        with patch.object(training, 'code_inventory', return_value={'sha256': 'fixture-code'}):
            self.assertEqual(training.verify_run_proofs(run), metadata)
            write_json(profile, {'kind': 'changed-proof'})
            with self.assertRaises(ValueError):
                training.verify_run_proofs(run)
        with patch.object(training, 'code_inventory', return_value={'sha256': 'different-code'}):
            with self.assertRaises(ValueError):
                training.verify_run_proofs(run)

    def test_final_test_reuse_rejects_wrong_split_before_new_evaluation(self):
        with patch.object(workflow, 'dataset_integrity', return_value={'dataset_fingerprint': FINGERPRINT}), \
             patch.object(workflow, 'render_and_bundle', return_value={}), \
             patch.object(workflow, '_contact_experiment', side_effect=self.experiment):
            self.run_phase('comparisons')
            paths = [evaluation_fixture(self.root / 'evidence', seed)[0] for seed in (42, 43, 44)]
            reporting.acceptance(paths, self.root / 'workflow')
            stale = self.root / 'workflow/final_test/seed42/test'
            stale.mkdir(parents=True)
            # Internally valid validation evidence cannot be reused as test.
            (stale / 'evaluation.json').write_bytes(paths[0].read_bytes())
            (stale / 'examples.jsonl').write_bytes((paths[0].parent / 'examples.jsonl').read_bytes())
            with patch.object(workflow, 'evaluate', side_effect=AssertionError('Do not compute beyond stale evidence')) as evaluate:
                with self.assertRaises(ValueError):
                    self.run_phase('final-test')
                evaluate.assert_not_called()

    def test_scaffold_reuse_is_bound_to_just_trained_checkpoint(self):
        with patch.object(workflow, 'dataset_integrity', return_value={'dataset_fingerprint': FINGERPRINT}), \
             patch.object(workflow, 'render_and_bundle', return_value={}), \
             patch.object(workflow, '_contact_experiment', side_effect=self.experiment):
            selected = self.run_phase('comparisons')
            p42, r42, _ = evaluation_fixture(self.root / 'evidence', 42)
            p43, _, _ = evaluation_fixture(self.root / 'evidence', 43)
            root = self.root / 'workflow/experiments' / selected['selected'] / 'seed42'
            root.mkdir(parents=True)
            entry = copy.deepcopy(next(e for e in selected['experiments'] if e['variant'] == selected['selected']))
            entry.update(gate_report='synthetic-gate', profile_report='synthetic-profile')
            write_json(root / 'experiment.json', entry)
            stale = root / 'paired_validation'
            stale.mkdir()
            (stale / 'evaluation.json').write_bytes(p43.read_bytes())
            (stale / 'examples.jsonl').write_bytes((p43.parent / 'examples.jsonl').read_bytes())
            with patch.object(workflow, '_train', return_value={'best_checkpoint': r42['checkpoint']}), \
                 patch.object(workflow, 'evaluate', side_effect=AssertionError('No new evaluation expected')):
                with self.assertRaises(ValueError):
                    self.run_phase('scaffold')
    def test_cli_rejects_output_outside_managed_root_or_prepared_tree(self):
        for output in (self.root.parent / 'outside-fixture', self.manifest.parent / 'inside', self.root):
            with self.subTest(output=output):
                args = parser().parse_args(['preflight', '--managed-root', str(self.root), '--output', str(output),
                                            '--manifest', str(self.manifest), '--device', 'cpu'])
                with self.assertRaises(ValueError):
                    dispatch(args)
        with self.assertRaises(SystemExit), patch('sys.stderr', new=io.StringIO()):
            parser().parse_args(['run', '--managed-root', str(self.root), '--output', str(self.root / 'out'),
                                 '--manifest', str(self.manifest), '--field-diagnostic-report', str(self.diagnostic),
                                 '--phase', 'invalid'])


if __name__ == '__main__':
    unittest.main()
