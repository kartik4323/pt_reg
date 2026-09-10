"""Install or repair the isolated CCS compatibility environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def find_cuda_home(env_name, environment):
    """Find a CUDA 11 compiler; the cudatoolkit runtime alone has no nvcc."""
    candidates = []
    configured = environment.get('CUDA_HOME') or environment.get('CUDA_PATH')
    if configured:
        candidates.append(Path(configured))
    nvcc = shutil.which('nvcc', path=environment.get('PATH'))
    if nvcc:
        candidates.append(Path(nvcc).resolve().parent.parent)
    candidates.extend([Path('/usr/local/cuda-11.3'), Path('/usr/local/cuda')])
    candidates.extend(sorted(Path('/usr/local').glob('cuda-11*'), reverse=True))
    try:
        prefix = subprocess.check_output(
            ['conda', 'run', '-n', env_name, 'python', '-c', 'import sys; print(sys.prefix)'],
            env=environment, text=True).strip()
        candidates.append(Path(prefix))
    except (OSError, subprocess.CalledProcessError):
        pass
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        compiler = candidate / 'bin/nvcc'
        if compiler.is_file():
            output = subprocess.check_output([str(compiler), '--version'], env=environment, text=True)
            if 'release 11.' in output:
                return candidate, output.strip()
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', default='sota-ccs')
    parser.add_argument('--source', type=Path, required=True,
                        help='Disposable native_build copy, never upstream/')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        parser.error('This profile requires Linux x86_64 (CUDA 11.3 binaries).')

    model_dir = Path(__file__).resolve().parent
    source = args.source.resolve()
    if source.name == 'upstream' or (source / '.git').exists():
        parser.error('Refusing to build native extensions in the pinned upstream checkout.')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                       CONDA_ALWAYS_YES='true', CONDA_CHANNEL_PRIORITY='flexible')
    environment.pop('PYTHONPATH', None)
    environment.pop('PYTHONHOME', None)

    def run(command, cwd=model_dir):
        command = list(map(str, command))
        print('$ ' + ' '.join(command), flush=True)
        subprocess.run(command, cwd=cwd, env=environment, check=True)

    envs = json.loads(subprocess.check_output(
        ['conda', 'env', 'list', '--json'], env=environment, text=True))['envs']
    exists = any(Path(path).name == args.env for path in envs)
    if exists:
        # Remove the upstream recipe's pip wheel, which targets Torch 1.11.
        packages = json.loads(subprocess.check_output(
            ['conda', 'list', '-n', args.env, '--json'], env=environment, text=True))
        pip_pytorch3d = any(row.get('name') == 'pytorch3d' and row.get('channel') == 'pypi'
                             for row in packages)
        if pip_pytorch3d:
            run(['conda', 'run', '--no-capture-output', '-n', args.env,
                 'python', '-m', 'pip', 'uninstall', '-y', 'pytorch3d'])

    run(['conda', 'env', 'update' if exists else 'create', '-n', args.env,
         '-f', model_dir / 'environment.repro.yaml'])
    prefix = ['conda', 'run', '--no-capture-output', '-n', args.env]
    run(prefix + ['python', '-m', 'pip', 'install', '-r', model_dir / 'requirements.repro.txt',
                  '-c', model_dir / 'constraints.repro.txt',
                  '--report', args.output / 'pip-install-report.json'])

    cuda_home, nvcc_version = find_cuda_home(args.env, environment)
    if cuda_home is None:
        raise RuntimeError(
            'CCS must compile Chamfer and PointNet2, but no CUDA 11 nvcc was found. '
            'Load your cluster CUDA 11 module or install '
            '`conda install -n %s -c conda-forge cudatoolkit-dev=11.3.1`, then rerun setup.'
            % args.env)
    environment['CUDA_HOME'] = str(cuda_home)
    environment['CUDA_PATH'] = str(cuda_home)
    environment['PATH'] = str(cuda_home / 'bin') + os.pathsep + environment.get('PATH', '')
    environment['LD_LIBRARY_PATH'] = (str(cuda_home / 'lib64') + os.pathsep +
                                      environment.get('LD_LIBRARY_PATH', ''))
    print('Using CUDA_HOME=%s\n%s' % (cuda_home, nvcc_version), flush=True)

    # Rebuild the two CUDA extensions against the pinned Torch ABI. Remove only
    # generated extension binaries inside the disposable native_build tree.
    extension_roots = [
        source / 'multi_part_assembly/utils/chamfer',
        source / 'multi_part_assembly/models/modules/encoder/pointnet2/pointnet2_ops_lib',
    ]
    for extension_root in extension_roots:
        if source not in extension_root.parents:
            raise RuntimeError('Extension path escaped the disposable source tree')
        for binary in extension_root.rglob('*.so'):
            binary.unlink()
        build_dir = extension_root / 'build'
        if build_dir.exists():
            shutil.rmtree(str(build_dir))

    run(prefix + ['python', '-m', 'pip', 'install', '--no-deps', '-e', '.'], cwd=source)
    run(prefix + ['python', '-m', 'pip', 'install', '--no-deps', '-e', '.'], cwd=extension_roots[0])
    run(prefix + ['python', '-m', 'pip', 'install', '--no-deps', '-e', '.'], cwd=extension_roots[1])
    run(prefix + ['python', '-m', 'pip', 'check'])

    for name, command in {
        'conda-explicit.txt': ['conda', 'list', '-n', args.env, '--explicit'],
        'pip-freeze.txt': prefix + ['python', '-m', 'pip', 'freeze'],
    }.items():
        (args.output / name).write_text(
            subprocess.check_output(command, text=True, env=environment), encoding='utf-8')
    recipe_names = ('environment.repro.yaml', 'requirements.repro.txt',
                    'constraints.repro.txt', 'setup_repro.py', 'doctor.py')
    hashes = {name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
              for name in recipe_names}
    (args.output / 'recipe-hashes.json').write_text(
        json.dumps(hashes, indent=2), encoding='utf-8')
    run(prefix + ['python', model_dir / 'doctor.py', '--source', source,
                  '--imports-only', '--report', args.output / 'doctor.json'])
    print('CCS environment repair and import checks passed.', flush=True)


if __name__ == '__main__':
    main()
