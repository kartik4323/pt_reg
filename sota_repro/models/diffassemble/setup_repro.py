"""Install the isolated DiffAssemble compatibility profile; safe to resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', default='sota-diffassemble-repro-v1')
    parser.add_argument('--source', type=Path, required=True, help='Disposable native_build copy, never upstream/')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'generated' / 'setup-repro')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        parser.error('This profile requires Linux x86_64 (CUDA 11.3 binaries).')
    model_dir = Path(__file__).resolve().parent
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', CONDA_ALWAYS_YES='true',
                       CONDA_CHANNEL_PRIORITY='flexible')
    # Prevent inherited package paths from another active Conda environment.
    environment.pop('PYTHONPATH', None)
    environment.pop('PYTHONHOME', None)

    def run(command):
        print('$ ' + ' '.join(map(str, command)), flush=True)
        subprocess.run(list(map(str, command)), cwd=model_dir, env=environment, check=True)

    run([sys.executable, model_dir / 'apply_native.py', '--source', args.source.resolve(),
         '--report', args.output / 'native-patches.json'])
    envs = json.loads(subprocess.check_output(['conda', 'env', 'list', '--json'], env=environment))['envs']
    exists = any(Path(path).name == args.env for path in envs)
    run(['conda', 'env', 'update' if exists else 'create', '-n', args.env,
         '-f', model_dir / 'environment.repro.yaml'])
    prefix = ['conda', 'run', '--no-capture-output', '-n', args.env]
    # Pip <24.1 is intentional: Lightning 1.7.7 has legacy torch>=1.9.* metadata.
    run(prefix + ['python', '-m', 'pip', 'install', '-r', model_dir / 'requirements.repro.txt',
                  '--report', args.output / 'pip-install-report.json'])
    run(prefix + ['python', '-m', 'pip', 'check'])
    run(prefix + ['python', model_dir / 'repair_runtime.py', '--report', args.output / 'runtime-repair.json'])
    for name, command in {
        'conda-explicit.txt': ['conda', 'list', '-n', args.env, '--explicit'],
        'pip-freeze.txt': prefix + ['python', '-m', 'pip', 'freeze'],
    }.items():
        (args.output / name).write_text(subprocess.check_output(command, text=True, env=environment), encoding='utf-8')
    recipe_hashes = {name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
                     for name in ('environment.repro.yaml', 'requirements.repro.txt', 'constraints.repro.txt', 'constraints.resolved.txt',
                                  'setup_repro.py', 'doctor.py', 'native_entry.py', 'repair_runtime.py',
                                  'apply_native.py', 'patches/3d-backbone-imports.patch')}
    (args.output / 'recipe-hashes.json').write_text(json.dumps(recipe_hashes, indent=2), encoding='utf-8')
    run(prefix + ['python', model_dir / 'doctor.py', '--source', args.source.resolve(),
                  '--imports-only', '--device', 'cpu', '--report', args.output / 'doctor.json'])
    print('Environment import checks passed. Run suite smoke on the CUDA host for native batch validation.', flush=True)


if __name__ == '__main__':
    main()
