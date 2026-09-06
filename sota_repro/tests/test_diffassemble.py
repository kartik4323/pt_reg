from __future__ import annotations

import ast
import contextlib
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from sota_repro.materialize import create_native_view
from sota_repro.model_entry import _run_logged, _visible_gpu, main
from sota_repro.models.diffassemble.apply_native import apply_native, RELATIVE
from sota_repro.registry import load_registry
from sota_repro.utils import flatten_command


ROOT = Path(__file__).resolve().parents[1]


class DiffAssembleTests(unittest.TestCase):
    def test_native_patch_is_guarded_and_idempotent(self):
        lines = (ROOT / 'models/diffassemble/patches/3d-backbone-imports.patch').read_text().splitlines(keepends=True)
        before = ''.join(line[1:] for line in lines[3:] if line.startswith('-'))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'native'
            target = source / RELATIVE
            target.parent.mkdir(parents=True)
            target.write_text(before)
            with contextlib.redirect_stdout(io.StringIO()):
                apply_native(source, root / 'patch.json')
                self.assertEqual(target.read_text(), 'from .efficient_gat_3d import Eff_GAT_3d\n')
                apply_native(source, root / 'patch.json')
            self.assertTrue(json.loads((root / 'patch.json').read_text())['already_applied'])
            target.write_text('user changes\n')
            with self.assertRaisesRegex(RuntimeError, 'preimage mismatch'):
                apply_native(source, root / 'patch.json')
            self.assertEqual(target.read_text(), 'user changes\n')
            with self.assertRaisesRegex(ValueError, 'Refusing'):
                apply_native(root / 'upstream', root / 'patch.json')
            (source / '.git').mkdir()
            with self.assertRaisesRegex(ValueError, 'Refusing'):
                apply_native(source, root / 'patch.json')

    def test_setup_creates_then_updates_without_removing_environments(self):
        module_spec = importlib.util.spec_from_file_location('setup_repro', ROOT / 'models/diffassemble/setup_repro.py')
        setup = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(setup)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for exists, expected in [(False, 'create'), (True, 'update')]:
                def output(command, **kwargs):
                    if command[:3] == ['conda', 'env', 'list']:
                        return json.dumps({'envs': ['/envs/sota-diffassemble-repro-v1'] if exists else []})
                    return 'fixture-lock\n'
                with patch.object(setup.platform, 'system', return_value='Linux'), \
                        patch.object(setup.platform, 'machine', return_value='x86_64'), \
                        patch.object(setup.sys, 'argv', ['setup_repro.py', '--source', str(root), '--output', str(root / expected)]), \
                        patch.object(setup.subprocess, 'check_output', side_effect=output), \
                        patch.object(setup.subprocess, 'run') as run, \
                        contextlib.redirect_stdout(io.StringIO()):
                    setup.main()
                commands = [call.args[0] for call in run.call_args_list]
                self.assertEqual(commands[1][:3], ['conda', 'env', expected])
                self.assertFalse(any('remove' in command for command in commands))
                self.assertTrue((root / expected / 'recipe-hashes.json').is_file())

    def test_gpu_selection_respects_scheduler_and_disabled_devices(self):
        self.assertEqual(_visible_gpu('0', '3,5'), '3')
        self.assertEqual(_visible_gpu('1', '3,5'), '5')
        self.assertEqual(_visible_gpu('0', ''), '')
        self.assertEqual(_visible_gpu('0', '-1'), '-1')
        self.assertEqual(_visible_gpu('2', None), '2')
        with self.assertRaises(ValueError):
            _visible_gpu('2', '3,5')
        with self.assertRaises(ValueError):
            _visible_gpu('all', None)

    def test_pins_and_commands_are_isolated_and_resumable(self):
        spec = load_registry()['diffassemble'].data
        self.assertEqual(spec['environment']['name'], 'sota-diffassemble-repro-v1')
        profile = yaml.safe_load((ROOT / 'models/diffassemble/environment.repro.yaml').read_text())
        for pin in ('python=3.10.13', 'mkl=2024.0.0', 'pip=24.0', 'numpy=1.23.5'):
            self.assertIn(pin, profile['dependencies'])
        self.assertIn('pytorch3d::pytorch3d=0.7.2=py310_cu113_pyt1121', profile['dependencies'])
        self.assertNotIn('--help', spec['commands']['smoke'][0])
        self.assertIn('--offline', spec['commands']['train'][0])
        variables = dict(model_dir='/model', source='/source', run='/run', env='env')
        setup = flatten_command(spec['commands']['setup'][0], variables)
        self.assertEqual(setup[1], '/model/setup_repro.py')
        self.assertEqual(setup[-1], '/run')
        for name in ('setup_repro.py', 'doctor.py', 'native_entry.py', 'repair_runtime.py', 'apply_native.py'):
            # The launcher can still use the user's Python 3.8 environment.
            ast.parse((ROOT / 'models/diffassemble' / name).read_text(), feature_version=(3, 8))

    def test_runtime_repair_does_not_modify_conda_hardlinks(self):
        module_spec = importlib.util.spec_from_file_location('repair_runtime', ROOT / 'models/diffassemble/repair_runtime.py')
        repair = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(repair)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = root / 'env'
            purelib = prefix / 'lib/site-packages'
            library = purelib / 'torch/lib/libtorch_cpu.so'
            library.parent.mkdir(parents=True)
            shared = root / 'shared-package-cache.so'
            shared.write_bytes(b'original')
            os.link(shared, library)
            report = root / 'repair.json'
            failed = subprocess.CompletedProcess([], 1, stdout='libtorch_cpu.so: cannot enable executable stack as shared object requires')
            passed = subprocess.CompletedProcess([], 0, stdout='1.12.1')

            def fake_patchelf(command, **kwargs):
                Path(command[-1]).write_bytes(b'patched')

            with patch.object(repair, 'torch_import', side_effect=[failed, passed]), \
                    patch.object(repair, 'version', return_value='1.12.1'), \
                    patch.object(repair.sysconfig, 'get_path', return_value=str(purelib)), \
                    patch.object(repair.sys, 'prefix', str(prefix)), \
                    patch.object(repair.sys, 'argv', ['repair_runtime.py', '--report', str(report)]), \
                    patch.object(repair.subprocess, 'run', side_effect=fake_patchelf), \
                    contextlib.redirect_stdout(io.StringIO()):
                repair.main()
            self.assertEqual(shared.read_bytes(), b'original')
            self.assertEqual(library.read_bytes(), b'patched')
            self.assertEqual(library.with_name(library.name + '.before-noexecstack').read_bytes(), b'original')
            self.assertTrue(json.loads(report.read_text())['passed'])

    def test_selected_split_prefixes_and_cached_view_upgrade(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            retained = root / 'retained'
            rows = []
            for split, obj in [('train', 'a'), ('val', 'b')]:
                relative = f'everyday/Bottle/{obj}/fractured_0'
                (retained / 'breaking_bad_everyday' / relative).mkdir(parents=True)
                rows.append(dict(track='breaking_bad_everyday', object_id=f'Bottle/{obj}',
                                 split=split, relative_path=relative, num_parts=2))
            (retained / 'common_v1_manifest.json').write_text(json.dumps({'samples': rows}))
            view = create_native_view('diffassemble', retained, root / 'views')
            index = view / 'breaking_bad/data_split/everyday.train.txt'
            self.assertEqual(index.read_text(), 'everyday/Bottle/a\n')
            index.write_text('Bottle/a\n')  # A completed view from the old adapter.
            create_native_view('diffassemble', retained, root / 'views')
            self.assertEqual(index.read_text(), 'everyday/Bottle/a\n')
            jigsaw = create_native_view('jigsaw', retained, root / 'views')
            self.assertEqual((jigsaw / 'breaking_bad/data_split/everyday.train.txt').read_text(), 'Bottle/a\n')
            self.assertNotEqual(index.read_text(), (view / 'breaking_bad/data_split/everyday.val.txt').read_text())

    def test_native_failure_is_visible_and_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / 'native.stdout.log'
            terminal = io.StringIO()
            command = [sys.executable, '-c', "import sys; print('actual native failure', flush=True); sys.exit(7)"]
            with contextlib.redirect_stdout(terminal), contextlib.redirect_stderr(terminal):
                with self.assertRaises(subprocess.CalledProcessError) as error:
                    _run_logged([command], root, os.environ.copy(), log)
            self.assertEqual(error.exception.returncode, 7)
            self.assertIn('actual native failure', terminal.getvalue())
            self.assertIn('actual native failure', log.read_text())
            self.assertIn(str(log), terminal.getvalue())

    def test_missing_executable_is_logged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(OSError):
                    _run_logged([[str(root / 'nonexistent-executable')]], root, os.environ.copy(), root / 'log')
            self.assertIn('Command failed', (root / 'log').read_text())

    def test_unsupported_track_and_missing_checkpoint_fail_early(self):
        with self.assertRaisesRegex(ValueError, 'everyday only'):
            main(['--model', 'diffassemble', '--action', 'test', '--track', 'breaking_bad_artifact'])
        with self.assertRaisesRegex(ValueError, 'requires --checkpoint'):
            main(['--model', 'diffassemble', '--action', 'test'])


if __name__ == '__main__':
    unittest.main()
