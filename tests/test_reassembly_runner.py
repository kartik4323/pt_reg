"""Exercise Bash orchestration with simulated CLI results, without GPU training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

from reassembly.cli import parser


REPO = Path(__file__).resolve().parents[1]
GIT_BASH = Path('C:/Program Files/Git/bin/bash.exe')
BASH = str(GIT_BASH) if GIT_BASH.is_file() else shutil.which('bash')

# Only orchestration is simulated. Inline geometry/overfit gates execute their
# real code; source quality, CUDA and model performance have separate checks.
STUB = r'''
import json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
args = sys.argv[1:]
if args[0] == '-c':
    raise SystemExit(subprocess.call([sys.executable] + args))
if args[0] == '-':
    program = sys.stdin.read()
    if 'make_guard' in program or 'importlib.metadata' in program:
        print('Simulated resource/environment inspection')
    else:
        sys.argv = ['-'] + args[1:]
        exec(compile(program, '<runner-inline>', 'exec'))
    raise SystemExit(0)
assert args[:3] == ['-u', '-m', 'reassembly'], args
args = args[3:]
with open(os.environ['RUNNER_TEST_CALLS'], 'a') as stream:
    stream.write(json.dumps(args) + '\n')
command = args[0]
if command == os.environ.get('RUNNER_TEST_FAIL'):
    print('Simulated command failure')
    raise SystemExit(2)
def option(name):
    return args[args.index(name) + 1]
def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')
work = Path(option('--managed-root')) / 'bottles498'
if command == 'prepare':
    write(work / 'prepared/manifest.json', dict(fingerprint='fixture',
        patterns=[dict(source_id=str(i)) for i in range(30)],
        report=dict(learning_ready=True, minimum_sources=30)))
if command == 'preflight':
    from reassembly.config import load_config
    write(work / 'preflight/config.resolved.json', load_config(option('--config')))
if command == 'evaluate' and '--overfit' in args:
    cfg = json.loads((work / 'preflight/config.resolved.json').read_text())
    write(work / 'overfit/eval/evaluation.json', dict(kind='evaluation', purpose='overfit',
        fixed_fit_passed=os.environ.get('RUNNER_TEST_FAIL') != 'overfit_gate',
        condition='predicted', dataset_fingerprint='fixture', config=cfg,
        summary=dict(success_rate=1)))
print('Simulated CLI:', command)
'''


@unittest.skipUnless(BASH, 'Bash is required for the Linux runner checks')
class RunnerTests(unittest.TestCase):
    def run_fixture(self, failure=''):
        with tempfile.TemporaryDirectory(prefix='reassembly-runner-') as directory:
            root = Path(directory)
            stub = root / 'stub.py'
            stub.write_text(STUB, encoding='utf-8')
            launcher = root / 'python-stub.sh'
            launcher.write_text('#!/usr/bin/env bash\nexec ' + shlex.quote(Path(sys.executable).as_posix())
                                + ' ' + shlex.quote(stub.as_posix()) + ' "$@"\n', encoding='utf-8', newline='\n')
            launcher.chmod(0o755)
            calls = root / 'calls.jsonl'
            env = dict(os.environ, REASSEMBLY_ROOT=(root / 'managed').as_posix(),
                       REASSEMBLY_PYTHON=launcher.as_posix(), RUNNER_TEST_CALLS=str(calls),
                       RUNNER_TEST_FAIL=failure)
            result = subprocess.run([BASH, 'scripts/run_reassembly_v2_pilot.sh', 'all'],
                                    cwd=REPO, env=env, text=True, capture_output=True, timeout=120)
            recorded = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
            for args in recorded:
                parser().parse_args(args)  # Every emitted command uses the real CLI schema.
            return result, recorded

    def test_all_stages_are_ordered_and_controls_use_correct_checkpoints(self):
        result, calls = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls), 24)
        learned = [parser().parse_args(c) for c in calls if c[0] == 'train']
        self.assertEqual([a.stage for a in learned], [1, 2, 3] * 3)
        self.assertEqual([a.condition for a in learned], ['predicted'] * 6 + ['contact_only'] * 3)
        for args in learned:
            self.assertEqual(args.device, 'cuda')
            self.assertIsNone(args.resume)
            if args.stage == 1:
                self.assertIsNone(args.initialize_from)
            else:
                self.assertEqual(args.initialize_from, args.run_dir.parent / f's{args.stage - 1}' / 'best.pt')
        evaluated = [parser().parse_args(c) for c in calls if c[0] == 'evaluate' and '--overfit' not in c]
        self.assertEqual([(a.split, a.condition) for a in evaluated],
                         [(s, c) for s in ('test', 'cut_holdout') for c in ('contact_only', 'predicted', 'gt', 'perturbed')])
        for args in evaluated:
            model = 'contact_only' if args.condition == 'contact_only' else 'predicted'
            self.assertEqual(args.checkpoint.parent.parent.name, model)
        self.assertEqual([c[0] for c in calls[-2:]], ['report', 'report'])

    def test_prepare_failure_stops_before_cuda_or_training(self):
        result, calls = self.run_fixture('prepare')
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual([c[0] for c in calls], ['acquire', 'acquire', 'prepare'])
        self.assertIn('Stopped during prepare', result.stderr)

    def test_failed_overfit_gate_stops_before_held_out_training(self):
        result, calls = self.run_fixture('overfit_gate')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls[-1][0], 'evaluate')
        self.assertTrue(all('--overfit' in c for c in calls if c[0] == 'train'))
        self.assertIn('Stopped during overfit-check', result.stderr)

    def test_preflight_failure_stops_before_training(self):
        result, calls = self.run_fixture('preflight')
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual([c[0] for c in calls], ['acquire', 'acquire', 'prepare', 'preflight'])
        self.assertIn('Stopped during preflight', result.stderr)


if __name__ == '__main__':
    unittest.main()
