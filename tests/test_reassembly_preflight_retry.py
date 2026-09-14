"""Cold CUDA startup recovery preserves failed evidence and never replaces run proofs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reassembly.resources import write_json
from reassembly.repair import workflow
from reassembly.repair.config import load_config, signature


class PreflightRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'experiment'
        self.path = self.root / 'preflight/preflight.json'
        self.cfg = load_config()
        self.report = {'kind': 'repair_preflight', 'status': 'failed', 'config': self.cfg,
            'config_signature': signature(self.cfg), 'device': 'cuda:0', 'dataset_fingerprint': 'fixture',
            'provenance': {'code': {'sha256': 'old-code'}},
            'attempts': [{'stages': [], 'error': 'RuntimeError: Invalid device argument ', 'passed': False}]}
        write_json(self.path, self.report)
        write_json(self.path.parent/'config.resolved.json', self.cfg)
        self.patches = [patch.object(workflow, 'code_inventory', return_value={'sha256': 'new-code'}),
                        patch.object(workflow, 'dataset_integrity', return_value={'dataset_fingerprint': 'fixture'})]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def retry(self):
        return workflow._retry_legacy_cuda_preflight(self.report, self.path, self.cfg, 'manifest', 'cuda:0')

    def test_preserves_old_preflight_bytes_and_allows_fresh_measurement(self):
        original = self.path.read_bytes()
        self.assertTrue(self.retry())
        archived = list(self.root.glob('preflight.failed.*/preflight.json'))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_bytes(), original)
        self.assertTrue((archived[0].parent/'config.resolved.json').is_file())
        self.assertFalse(self.path.parent.exists())

    def test_does_not_archive_passed_new_or_other_failures(self):
        for kind in ('passed', 'other_error', 'already_instrumented', 'stage_completed', 'same_code'):
            with self.subTest(kind=kind):
                original = copy.deepcopy(self.report)
                if kind == 'passed': self.report['status'] = 'passed'
                if kind == 'other_error': self.report['attempts'][0]['error'] = 'RuntimeError: nonfinite loss'
                if kind == 'already_instrumented': self.report['attempts'][0]['failed_operation'] = 'cuda_initialization'
                if kind == 'stage_completed': self.report['attempts'][0]['stages'] = [{'stage': 1}]
                if kind == 'same_code': self.report['provenance']['code']['sha256'] = 'new-code'
                self.assertFalse(self.retry())
                self.assertTrue(self.path.exists())
                self.report = original

    def test_changed_inputs_or_existing_training_cannot_replace_proof(self):
        for kind in ('configuration', 'dataset', 'device', 'training'):
            with self.subTest(kind=kind):
                original = copy.deepcopy(self.report)
                if kind == 'configuration': self.report['config_signature'] = 'different'
                if kind == 'dataset': self.report['dataset_fingerprint'] = 'different'
                if kind == 'device': self.report['device'] = 'cuda:1'
                if kind == 'training': (self.root/'contacts').mkdir()
                with self.assertRaises((RuntimeError, ValueError)):
                    self.retry()
                self.assertTrue(self.path.exists())
                self.report = original

    def test_comparisons_retries_once_and_prints_actual_failure(self):
        def failed_again(cfg, manifest, output, device, **kwargs):
            self.assertFalse(output.exists())
            report = copy.deepcopy(self.report)
            report['attempts'][0].update(error='RuntimeError: fixture next failure', failed_operation='batch_loading')
            write_json(output/'preflight.json', report)
            return report
        with patch.object(workflow, 'preflight', side_effect=failed_again) as preflight, \
                patch.object(workflow, '_train') as train:
            with self.assertRaisesRegex(RuntimeError, 'batch_loading: RuntimeError: fixture next failure'):
                workflow._contact_experiment(self.cfg, 'manifest', self.root, 'cuda:0', 'diagnostic', None, True, None)
            with self.assertRaisesRegex(RuntimeError, 'fixture next failure'):
                workflow._contact_experiment(self.cfg, 'manifest', self.root, 'cuda:0', 'diagnostic', None, True, None)
            preflight.assert_called_once()
            train.assert_not_called()
        self.assertEqual(len(list(self.root.glob('preflight.failed.*'))), 1)


if __name__ == '__main__':
    unittest.main()
