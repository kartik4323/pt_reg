"""Diagnose field interpolation on the original VM artifacts, without training."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

from diagnostics.reassembly_v2.runtime import Limits, read, write


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--managed-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--source-diagnostics', type=Path,
                        help='Original diagnostic output directory or diagnostic_bundle.tar.gz; otherwise use the newest verified run or the original deterministic selection')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--hours', type=float, default=None,
                        help='Optional time limit; omitted means no time limit')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--deadline', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.hours is not None and (not math.isfinite(args.hours) or args.hours <= 0):
        parser.error('--hours must be finite and positive')
    args.managed_root = args.managed_root.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    args.output = (args.output or args.managed_root / 'field_diagnostics' / stamp).expanduser().resolve()
    work = args.managed_root / 'bottles498'
    if args.managed_root not in args.output.parents or args.output == work or work in args.output.parents:
        parser.error('--output must be a dedicated directory below managed root, outside bottles498')
    if args.source_diagnostics:
        args.source_diagnostics = args.source_diagnostics.expanduser().resolve()
    if args.resume and not (args.output / 'invocation.json').is_file():
        parser.error('--resume requires --output from an existing field diagnosis')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        import torch
        from .runner import run
        def forbidden(*a, **kw):
            raise RuntimeError('Optimizer updates are forbidden in field diagnostics')
        with ExitStack() as stack:
            for cls in vars(torch.optim).values():
                if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer):
                    stack.enter_context(patch.object(cls, 'step', forbidden))
            run(args.managed_root, args.output, torch.device(args.device),
                args.source_diagnostics, args.deadline, args.resume)
        return 0
    from .runner import finalize, validate_resume
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError('Choose a new output directory or explicitly resume the existing field diagnosis')
    if args.resume:
        previous = read(args.output / 'invocation.json')
        if previous['managed_root'] != str(args.managed_root) or previous['device'] != args.device:
            raise RuntimeError('Resume requires the original managed root and device')
        if args.source_diagnostics is None and previous.get('source_diagnostics'):
            args.source_diagnostics = Path(previous['source_diagnostics'])
        if previous.get('source_diagnostics') != (str(args.source_diagnostics) if args.source_diagnostics else None):
            raise RuntimeError('Resume source diagnostic input differs')
        validate_resume(args.output)
    deadline = None if args.hours is None else time.time() + args.hours * 3600
    limits = Limits(args.managed_root, args.output, deadline)
    limits.check()
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / 'invocation.json', {
        'managed_root': str(args.managed_root), 'device': args.device,
        'source_diagnostics': str(args.source_diagnostics) if args.source_diagnostics else None,
        'hours': args.hours, 'deadline_epoch': deadline, 'resume': args.resume,
        'started_utc': datetime.now(timezone.utc).isoformat()})
    command = [sys.executable, '-m', 'diagnostics.reassembly_field', '--worker',
               '--managed-root', str(args.managed_root), '--output', str(args.output), '--device', args.device]
    if args.source_diagnostics:
        command += ['--source-diagnostics', str(args.source_diagnostics)]
    if deadline is not None:
        command += ['--deadline', str(deadline)]
    if args.resume:
        command += ['--resume']
    reason, process = None, None
    print(f'Field diagnosis: {args.output}\nNo training. ' +
          ('No time limit.' if deadline is None else f'Limit: {args.hours:g} hours.'), flush=True)
    try:
        with (args.output / 'worker.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            notice = 0
            while process.poll() is None:
                limits.check()
                if time.monotonic() - notice > 30:
                    progress = args.output / 'progress.json'
                    if progress.exists():
                        current = read(progress)
                        print(f"{current.get('phase', 'running')}: {current.get('job_id', '')} "
                              f"{current.get('completed_nodes', '')}", flush=True)
                    notice = time.monotonic()
                time.sleep(1)
            if process.returncode:
                reason = f'Worker exited {process.returncode}; inspect worker.log'
    except (Exception, KeyboardInterrupt) as exc:
        reason = f'{type(exc).__name__}: {exc}'
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    finally:
        summary = finalize(args.output, reason)
        print(f"{summary['status']}: {summary['completed_jobs']}/{summary['planned_jobs']} jobs.\n"
              f"Return: {args.output / 'field_diagnostic_bundle.tar.gz'}", flush=True)
    return 0 if summary['status'] == 'complete' else 2


if __name__ == '__main__':
    sys.exit(main())
