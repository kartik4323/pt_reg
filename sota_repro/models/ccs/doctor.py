"""Check CCS versions, imports and compiled CUDA operators before training."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


IMPORTS = (
    'torch', 'torchvision', 'pytorch_lightning', 'torchmetrics',
    'pytorch3d._C', 'pytorch3d.transforms', 'numpy', 'scipy', 'pandas',
    'trimesh', 'pyntcloud', 'wandb', 'yacs', 'setproctitle', 'pympler',
    'chamfer_cuda', 'pointnet2_ops._ext', 'multi_part_assembly',
)
EXPECTED = {
    'torch': '1.10.2', 'torchvision': '0.11.3', 'pytorch3d': '0.7.2',
    'pytorch-lightning': '1.6.2', 'torchmetrics': '0.9.2',
    'numpy': '1.23.5', 'scipy': '1.10.1', 'pandas': '1.5.3',
    'setuptools': '59.5.0',
}


def operator_probe():
    import torch
    from chamfer import chamfer_distance
    from pointnet2_ops.pointnet2_utils import furthest_point_sample
    from pytorch3d.transforms import matrix_to_quaternion

    assert torch.cuda.is_available(), 'No CUDA GPU visible to CCS.'
    device = torch.device('cuda')
    points = torch.randn(1, 32, 3, device=device, requires_grad=True)
    dist1, dist2 = chamfer_distance(points, points.detach() + 0.01)
    (dist1.mean() + dist2.mean()).backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()
    indices = furthest_point_sample(points.detach().contiguous(), 8)
    assert indices.shape == (1, 8)
    quaternion = matrix_to_quaternion(torch.eye(3, device=device))
    assert torch.allclose(quaternion, torch.tensor([1., 0., 0., 0.], device=device))
    torch.cuda.synchronize()
    print('CCS Chamfer, PointNet2 and PyTorch3D CUDA operators passed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--imports-only', action='store_true')
    parser.add_argument('--probe', choices=('versions', 'operators'))
    args = parser.parse_args()
    source = args.source.resolve()
    report = args.report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                      WANDB_MODE='offline', WANDB_SILENT='true', MPLBACKEND='Agg')
    if args.probe == 'operators':
        operator_probe()
        return 0
    if args.probe == 'versions':
        mismatches = []
        for package, expected in EXPECTED.items():
            actual = version(package)
            print(package, actual, flush=True)
            if actual.split('+')[0] != expected:
                mismatches.append('%s: expected %s, got %s' % (package, expected, actual))
        assert not mismatches, '\n'.join(mismatches)
        return 0
    checks = []

    def check(name, command, timeout=180):
        print('\n[CHECK] ' + name, flush=True)
        try:
            result = subprocess.run(command, cwd=source, env=os.environ.copy(),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=timeout)
            output, code = result.stdout, result.returncode
        except (OSError, subprocess.TimeoutExpired):
            output, code = traceback.format_exc(), 1
        passed = code == 0
        checks.append({'name': name, 'passed': passed, 'returncode': code, 'output': output})
        print(('PASS' if passed else 'FAIL') + ': ' + name, flush=True)
        if output and (not passed or not name.startswith('import ')):
            print(output, flush=True)
        report.write_text(json.dumps({'checks': checks}, indent=2), encoding='utf-8')

    for module in IMPORTS:
        prelude = 'import torch; ' if module in ('pytorch3d._C', 'chamfer_cuda', 'pointnet2_ops._ext') else ''
        code = prelude + 'import importlib; importlib.import_module(%r)' % module
        check('import ' + module, [sys.executable, '-c', code])
    probe = [sys.executable, str(Path(__file__).resolve()), '--source', str(source),
             '--report', str(report)]
    check('pinned versions', probe + ['--probe', 'versions'])
    check('pip dependency consistency', [sys.executable, '-m', 'pip', 'check'])
    check('native train entry imports', [sys.executable, 'scripts/train.py', '--help'])
    if not args.imports_only and all(row['passed'] for row in checks):
        check('compiled CUDA operators', probe + ['--probe', 'operators'], timeout=300)
    elif not args.imports_only:
        checks.append({'name': 'compiled CUDA operators', 'passed': False,
                       'output': 'Skipped because a prerequisite failed.'})
    passed = all(row['passed'] for row in checks)
    report.write_text(json.dumps({'passed': passed, 'python': sys.version,
                                  'checks': checks}, indent=2), encoding='utf-8')
    print('\n%s: full report at %s' % ('PASS' if passed else 'FAIL', report), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
