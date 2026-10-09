"""Artifact export safety and end-to-end checkpoint-only ablation tooling."""
import hashlib
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

def script(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class DiagnosticToolTests(unittest.TestCase):
    def test_export_hashes_excludes_assets_and_does_not_overwrite(self):
        export=script('export_reassembly_repair_diagnostics').export
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'experiment'
            for relative,content in [('contacts/history.jsonl',b'{}\n'),('contacts/best.pt',b'weights'),
                ('gate/training/examples.jsonl',b'{}\n'),('gate/training/example_01.npz',b'geometry'),
                ('queries/source.npz',b'prepared asset')]:
                path=source/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
            target=root/'review.tar.gz';export(source,target)
            before=target.read_bytes()
            with tarfile.open(target) as archive:
                names=archive.getnames()
                self.assertNotIn('contacts/best.pt',names);self.assertNotIn('queries/source.npz',names)
                manifest=json.load(archive.extractfile('bundle_manifest.json'))
                for name,digest in manifest['files'].items():
                    self.assertEqual(hashlib.sha256(archive.extractfile(name).read()).hexdigest(),digest)
            with self.assertRaises(FileExistsError):export(source,target)
            self.assertEqual(before,target.read_bytes())

    def test_ablation_completes_validation_after_failed_gate_without_training(self):
        from test_reassembly_repair_evaluation import RepairEvaluationTests
        module=script('run_reassembly_repair_ablation')
        fixture=RepairEvaluationTests();fixture.setUpClass();fixture.setUp()
        try:
            import torch
            from reassembly.repair.checkpoints import load_checkpoint
            source=fixture.root/'experiment'
            (source/'contacts').mkdir(parents=True);(source/'overfit').mkdir()
            original=fixture.checkpoint.read_bytes()
            (source/'contacts/best.pt').write_bytes(original)
            state=load_checkpoint(fixture.checkpoint);state['purpose']='overfit'
            torch.save(state,source/'overfit/best.pt')
            solver_config=fixture.root/'orientation.yaml'
            solver_config.write_text('solver:\n  contact_orientation:\n    enabled: true\n')
            output=fixture.root/'evaluation'
            argv=['ablation','--experiment',str(source),'--managed-root',str(fixture.root),
                '--manifest',str(fixture.manifest),'--output',str(output),
                '--solver-config',str(solver_config),'--device','cpu']
            with patch.object(sys,'argv',argv),patch.object(module,'make_guard',return_value=Mock()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(),2)
            gate=json.loads((output/'gate/contact_gate.json').read_text())
            validation=json.loads((output/'contact_validation/evaluation.json').read_text())
            self.assertTrue(gate['diagnostic_only']);self.assertFalse(gate['passed'])
            self.assertEqual(validation['status'],'completed')
            self.assertEqual((source/'contacts/best.pt').read_bytes(),original)
            self.assertFalse((source/'contacts/history.jsonl').exists())
        finally:
            fixture.tearDown();fixture.tearDownClass()


if __name__=='__main__':unittest.main()
