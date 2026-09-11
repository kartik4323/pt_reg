"""Read-only diagnosis supervisor with an optional deadline. Run from the repository root."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

from .runtime import Limits, finalize, read, sha256, write


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--managed-root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--hours',type=float,default=None,help='Optional positive time limit in hours; default: run without a time limit')
    parser.add_argument('--resume',action='store_true',help='Continue completed diagnostic jobs only; never resume training')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--deadline',type=float,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.hours is not None and (not math.isfinite(args.hours) or args.hours<=0):
        parser.error('--hours must be a finite positive number; omit it for no time limit')
    args.managed_root=args.managed_root.expanduser().resolve()
    args.output=(args.output or args.managed_root/'diagnostics'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')).expanduser().resolve()
    if args.managed_root not in args.output.parents or args.output==args.managed_root/'bottles498' or args.managed_root/'bottles498' in args.output.parents:
        parser.error('--output must be a dedicated directory beneath --managed-root, outside bottles498')
    if args.resume and not (args.output/'invocation.json').is_file(): parser.error('--resume requires an existing diagnostic output')
    return args


def main(argv=None):
    args=parse_args(argv)
    if args.worker:
        import torch
        from contextlib import ExitStack
        from unittest.mock import patch
        from .worker import run
        def forbidden(*a,**kw): raise RuntimeError('Optimizer steps are forbidden in diagnostics')
        # Fail closed if an imported probe accidentally tries to update weights.
        with ExitStack() as stack:
            for cls in vars(torch.optim).values():
                if isinstance(cls,type) and issubclass(cls,torch.optim.Optimizer):
                    stack.enter_context(patch.object(cls,'step',forbidden))
            run(args.managed_root/'bottles498',args.output,torch.device(args.device),args.deadline)
        return 0
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError('Choose a fresh output directory or explicitly --resume diagnostics')
    deadline=None if args.hours is None else time.time()+args.hours*3600
    limits=Limits(args.managed_root,args.output,deadline)
    limits.check()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.resume:
        old=read(args.output/'invocation.json')
        if old['managed_root']!=str(args.managed_root) or old['device']!=args.device:
            raise RuntimeError('Resume requires the same input root and device')
        inv=read(args.output/'inventory.json')
        for section in ('files','production_files','diagnostic_files'):
            for name,entry in inv.get(section,{}).items():
                if sha256(entry['path'])!=entry['sha256']: raise RuntimeError(f'Resume input changed: {name}')
    write(args.output/'invocation.json',{'managed_root':str(args.managed_root),'device':args.device,'hours':args.hours,
        'started_utc':datetime.now(timezone.utc).isoformat(),'deadline_epoch':deadline,'resume':args.resume})
    command=[sys.executable,'-m','diagnostics.reassembly_v2','--worker','--managed-root',str(args.managed_root),
             '--output',str(args.output),'--device',args.device]
    if deadline is not None: command.extend(['--deadline',str(deadline)])
    reason=None; process=None
    duration='No time limit' if args.hours is None else f'Deadline: {args.hours:g} hours'
    print(f'Diagnostic output: {args.output}\n{duration}; no training or optimizer updates.',flush=True)
    try:
        with open(args.output/'worker.log','a',encoding='utf-8') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            last_notice=0
            while process.poll() is None:
                limits.check()
                if time.monotonic()-last_notice>30:
                    progress=args.output/'progress.json'
                    if progress.exists():
                        current=read(progress)['job']; print(f"Running {current['kind']}: {current['id']}",flush=True)
                    last_notice=time.monotonic()
                time.sleep(1 if deadline is None else min(1,max(.01,deadline-time.time())))
            if process.returncode: reason=f'Worker exited {process.returncode}; inspect worker.log and results errors'
    except (Exception,KeyboardInterrupt) as error:
        reason=f'{type(error).__name__}: {error}'
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
    finally:
        # The deadline stops compute. Hash verification/report packaging may finish afterward.
        summary=finalize(args.output,reason)
        print(f"{summary['status']}: {summary['completed_jobs']}/{summary['planned_jobs']} jobs.\nReturn: {args.output/'diagnostic_bundle.tar.gz'}",flush=True)
    return 0 if summary['status']=='complete' else 2


if __name__=='__main__':
    try: sys.exit(main())
    except Exception:
        traceback.print_exc(); sys.exit(2)
