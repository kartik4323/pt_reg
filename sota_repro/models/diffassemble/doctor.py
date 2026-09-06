"""Aggregate import, binary, native-entry and real-batch diagnostics."""
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


IMPORTS = ['pkg_resources', 'numpy', 'scipy', 'PIL.Image', 'networkx', 'torch',
           'torchvision', 'pytorch_lightning', 'torchmetrics', 'torch_geometric',
           'torch_scatter', 'torch_sparse', 'torch_cluster', 'torch_spline_conv',
           'pytorch3d._C', 'pytorch3d.transforms', 'pytorch3d.ops', 'einops', 'timm',
           'kornia', 'transformers.optimization', 'trimesh', 'matplotlib.pyplot',
           'wandb', 'yacs.config', 'torch_geometric.graphgym.register']

EXPECTED = {'torch': '1.12.1', 'torchvision': '0.13.1', 'pytorch3d': '0.7.2',
            'pytorch-lightning': '1.7.7', 'torchmetrics': '0.9.2', 'torch-geometric': '2.2.0',
            'numpy': '1.23.5', 'scipy': '1.9.3', 'Pillow': '9.2.0',
            'setuptools': '65.6.3', 'pip': '24.0'}


def model_probe():
    from model import spatial_diffusion_3d_test_double_diffusion as sd
    model = sd.GNN_Diffusion(steps=300, sampling='DDIM', inference_ratio=10,
                            model_mean_type=sd.ModelMeanType.START_X,
                            backbone='vn_dgcnn', freeze_backbone=False,
                            n_layers=4, max_num_part=20, max_epochs=500)
    model.initialize_torchmetrics({'smoke'})
    model.configure_optimizers()
    print('Native model and optimizer constructed:', sum(p.numel() for p in model.parameters()), 'parameters')


def loader_probe():
    """Exercise the native CPU preprocessing on a tiny synthetic mesh fixture."""
    import torch
    from torch_geometric.loader import DataLoader
    from dataset.breakingbad_dt import GeometryPartDataset
    from dataset.objects_dataset import Objects_Dataset
    with tempfile.TemporaryDirectory(prefix='diffassemble-loader-') as temporary:
        root = Path(temporary)
        pieces = root / 'everyday/Fixture/object/fractured_0'
        pieces.mkdir(parents=True)
        mesh = 'v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'
        for i in range(2):
            (pieces / f'piece_{i}.obj').write_text(mesh)
        (root / 'split.txt').write_text('everyday/Fixture/object\n')
        native = GeometryPartDataset(str(root), 'split.txt', ('part_ids',), num_points=1000,
                                     min_num_part=2, max_num_part=20)
        dataset = Objects_Dataset(native, lambda value: value)
        batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
        assert batch.pcds.shape == (2, 1000, 3)
        assert batch.x.shape == (2, 7) and torch.isfinite(batch.x).all()
        assert batch.category == ['Fixture']
        print('Native mesh sampling, SE(3) augmentation and PyG collation passed (synthetic fixture).')


def logger_probe():
    import numpy as np
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    with tempfile.TemporaryDirectory(prefix='diffassemble-logger-') as temporary:
        logger = WandbLogger(project='sota-diffassemble-diagnostic', offline=True, save_dir=temporary)
        try:
            logger.experiment.log({'fixture': wandb.Object3D(np.zeros((24, 6)))})
            print('Lightning/W&B offline point-cloud logging passed.')
        finally:
            wandb.finish()


def ops(device):
    import torch
    import numpy as np
    from torch_scatter import scatter
    from torch_sparse import SparseTensor
    from torch_cluster import knn_graph
    from torchvision.ops import nms
    from pytorch3d.ops import knn_points
    from pytorch3d.transforms import matrix_to_quaternion
    print('Torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda, flush=True)
    torch.set_num_threads(2)
    assert torch.__version__.split('+')[0] == '1.12.1'
    if device == 'cuda':
        assert torch.cuda.is_available(), 'No CUDA GPU visible. Check allocation/driver/CUDA_VISIBLE_DEVICES.'
        print('GPU:', torch.cuda.get_device_name(), 'capability:', torch.cuda.get_device_capability(), flush=True)
    torch.from_numpy(np.ones((2, 3), dtype=np.float32)).to(device).matmul(torch.ones(3, 2, device=device))
    x = torch.randn(2, 24, 3, device=device, requires_grad=True)
    d = knn_points(x, x.detach() + 0.1, K=1).dists.mean()
    d.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    index = torch.tensor([0, 0, 1, 1], device=device)
    assert scatter(torch.ones(4, device=device), index).tolist() == [2., 2.]
    edge = knn_graph(torch.randn(24, 3, device=device), k=3)
    sparse = SparseTensor(row=edge[0], col=edge[1], sparse_sizes=(24, 24))
    assert sparse.matmul(torch.ones(24, 2, device=device)).shape == (24, 2)
    nms(torch.tensor([[0., 0., 1., 1.]], device=device), torch.ones(1, device=device), 0.5)
    assert torch.allclose(matrix_to_quaternion(torch.eye(3, device=device)),
                          torch.tensor([1., 0., 0., 0.], device=device))
    if device == 'cuda':
        torch.cuda.synchronize()
    print('Torch/PyG/PyTorch3D forward and backward operators passed.', flush=True)


def native_batch(args):
    import torch
    import pytorch_lightning as pl
    import wandb
    from torch_geometric.loader import DataLoader
    from pytorch_lightning.loggers import WandbLogger
    from dataset import dataset_utils as du
    from model import spatial_diffusion_3d_test_double_diffusion as sd
    pl.seed_everything(int(os.environ.get('SOTA_SEED', '42')), workers=True)
    torch.set_num_threads(2)
    # The unmodified loader resolves datasets/breaking-bad in the native view.
    train, _, val = du.get_dataset_3d('breaking-bad', '', 20, 2, 0)
    assert len(train) and len(val), 'Selected everyday train/val data are empty; check split paths and manifest.'
    # Bound the smoke memory cost by choosing the smallest retained pattern.
    for dataset in (train, val):
        native = dataset.dataset
        smallest = min(native.data_list,
                       key=lambda path: (len(list((Path(native.data_dir) / path).glob('*.obj'))), path))
        native.data_list = [smallest]
        native.used_categories = {smallest.split('/')[1]}
        print('Smoke sample:', smallest, flush=True)
    model = sd.GNN_Diffusion(steps=20, sampling='DDIM', inference_ratio=10,
                            model_mean_type=sd.ModelMeanType.START_X,
                            backbone='vn_dgcnn', freeze_backbone=False,
                            n_layers=4, max_num_part=20, max_epochs=1)
    # Only validation updates metrics. Unobserved categories yield NaNs.
    model.initialize_torchmetrics(val.dataset.used_categories)
    logger = WandbLogger(project='sota-diffassemble-smoke', name='one-batch',
                         offline=True, save_dir=str(args.report.parent))
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=1, max_steps=1,
                         limit_train_batches=1, limit_val_batches=1, limit_test_batches=1,
                         check_val_every_n_epoch=1, num_sanity_val_steps=0,
                         logger=logger, enable_checkpointing=False, enable_progress_bar=False,
                         default_root_dir=str(args.report.parent))
    try:
        trainer.fit(model, DataLoader(train, batch_size=1, num_workers=0),
                    DataLoader(val, batch_size=1, num_workers=0))
        assert trainer.global_step == 1, 'No optimizer step completed'
        assert 'rmse_t_AVG' in trainer.callback_metrics, 'Native validation did not complete'
        for name, value in trainer.callback_metrics.items():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all(), f'Non-finite metric: {name}'
        checkpoint = args.report.parent / 'smoke-only.ckpt'
        trainer.save_checkpoint(str(checkpoint))
        restored = sd.GNN_Diffusion.load_from_checkpoint(str(checkpoint))
        restored.initialize_torchmetrics(val.dataset.used_categories)
        restored.test_dataset = val
        restored.save_eval_images = True
        results = trainer.test(restored, DataLoader(val, batch_size=1, num_workers=0))
        assert results and 'rmse_t_AVG' in results[0], 'Native test did not complete'
        import math
        assert all(math.isfinite(float(value)) for value in results[0].values()), 'Non-finite native test metric'
        print('Native train/validation, checkpoint reload and test batch passed (20 steps; diagnostic only).', flush=True)
    finally:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--imports-only', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--probe', choices=['ops', 'model', 'loader', 'logger', 'batch', 'versions'])
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.report = args.report.resolve()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', MPLBACKEND='Agg',
                      WANDB_MODE='offline', WANDB_SILENT='true',
                      XDG_CACHE_HOME=str(args.report.parent / 'cache'),
                      HF_HOME=str(args.report.parent / 'cache/huggingface'))
    sys.path.insert(0, str(args.source / 'puzzle_diff'))
    if args.probe:
        os.chdir(args.source)
        if args.probe == 'ops':
            ops(args.device)
        elif args.probe == 'model':
            model_probe()
        elif args.probe == 'loader':
            loader_probe()
        elif args.probe == 'logger':
            logger_probe()
        elif args.probe == 'versions':
            mismatches = []
            for package, expected in EXPECTED.items():
                actual = version(package)
                print(package, actual, flush=True)
                if actual.split('+')[0] != expected:
                    mismatches.append(f'{package}: expected {expected}, got {actual}')
            assert not mismatches, '\n'.join(mismatches)
        else:
            native_batch(args)
        return 0
    checks = []

    def check(name, command, timeout=120):
        print(f'\n[CHECK] {name}', flush=True)
        try:
            result = subprocess.run(command, cwd=args.source, env=os.environ.copy(),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    timeout=timeout)
            output, code = result.stdout, result.returncode
        except (OSError, subprocess.TimeoutExpired):
            output, code = traceback.format_exc(), 1
        passed = code == 0
        checks.append(dict(name=name, passed=passed, returncode=code, output=output))
        print(('PASS' if passed else 'FAIL') + ': ' + name, flush=True)
        if output and (not passed or not name.startswith('import ')):
            print(output, flush=True)
        args.report.write_text(json.dumps(dict(python=sys.version, executable=sys.executable,
                                               checks=checks), indent=2), encoding='utf-8')
        return passed

    for module in IMPORTS:
        # The raw PyTorch3D extension expects Torch's shared libraries loaded
        # first, just as the public pytorch3d.ops API does.
        prelude = 'import torch; ' if module == 'pytorch3d._C' else ''
        check('import ' + module, [sys.executable, '-c', prelude + f'import importlib; importlib.import_module({module!r})'])
    check('pip dependency consistency', [sys.executable, '-m', 'pip', 'check'])
    check('native train_3d --help', [sys.executable, 'puzzle_diff/train_3d.py', '--help'])
    probe = [sys.executable, str(Path(__file__).resolve()), '--source', str(args.source),
             '--report', str(args.report), '--device', args.device]
    check('pinned versions', probe + ['--probe', 'versions'])
    check('native model and optimizer construction', probe + ['--probe', 'model'])
    check('native loader synthetic fixture', probe + ['--probe', 'loader'])
    check('offline point-cloud logger', probe + ['--probe', 'logger'])
    check('compiled operators', probe + ['--probe', 'ops'], timeout=180)
    if not args.imports_only and all(row['passed'] for row in checks):
        check('native one-batch train/validation/checkpoint/test', probe + ['--probe', 'batch'], timeout=1200)
    elif not args.imports_only:
        checks.append(dict(name='native one-batch train/validation/checkpoint/test', passed=False,
                           output='Skipped because prerequisite checks failed.'))
    passed = all(row['passed'] for row in checks)
    args.report.write_text(json.dumps(dict(passed=passed, python=sys.version, executable=sys.executable,
                                          imports_only=args.imports_only, device=args.device,
                                          checks=checks), indent=2), encoding='utf-8')
    print(f'\n{"PASS" if passed else "FAIL"}: full report at {args.report}', flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
