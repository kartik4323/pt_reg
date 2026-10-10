"""Cooperative GPU admission and persistent bounded worker-wall-time accounting."""
import csv
import os
import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from .storage import read, write


class ResourceUnavailable(RuntimeError): pass
class BudgetExhausted(RuntimeError): pass


def gpu_snapshot(device='cuda'):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    cards = [dict(index=int(r[0]), uuid=r[1].strip(), name=r[2].strip(),
                  total_gib=float(r[3])/1024, free_gib=float(r[4])/1024)
             for r in csv.reader(output.splitlines())]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    logical = int(device.split(':')[1]) if ':' in device else 0
    if visible is not None:
        entries = visible.split(',')
        if logical >= len(entries): raise ResourceUnavailable('CUDA device is not visible')
        token = entries[logical].strip()
        chosen = next((c for c in cards if str(c['index']) == token or c['uuid'].startswith(token)), None)
    else:
        chosen = next((c for c in cards if c['index'] == logical), None)
    if chosen is None: raise ResourceUnavailable('Cannot map CUDA device to NVIDIA GPU; MIG needs an explicit supported binding')
    try:
        processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
            '--format=csv,noheader,nounits'], text=True)
    except subprocess.CalledProcessError: processes = 'unavailable'
    try:
        import psutil
        host_free = psutil.virtual_memory().available / 1024**3
    except ImportError: host_free = None
    return dict(**chosen, logical_device=device, cuda_visible_devices=visible,
                compute_processes=processes.splitlines(), host_free_gib=host_free)


@contextmanager
def atomic_lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    try: fd = os.open(str(path), os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: raise ResourceUnavailable(f'Cooperative lock occupied: {path}; inspect PID/host before removing')
    try:
        os.write(fd, f'{socket.gethostname()} {os.getpid()} {time.time()}'.encode()); os.close(fd)
        yield
    finally: path.unlink(missing_ok=True)


@contextmanager
def gpu_admission(req, directory):
    cfg = req.get('resources', {})
    if req.get('smoke') or not cfg.get('gpu_preflight') or req['kind'] == 'matte':
        yield None; return
    device = req['config'].get('device', 'cuda')
    if not device.startswith('cuda'): yield None; return
    minimum = cfg.get('reconstruction_free_gib',24) if req['kind']=='reconstruction' else cfg.get('image_free_gib',8)
    deadline = time.monotonic()+cfg.get('wait_seconds',1800)
    lock_base = Path(cfg.get('lock_dir') or Path(tempfile.gettempdir())/'generative-assembly-gpu-locks')
    while True:
        try:
            snapshot = gpu_snapshot(device)
            lock = atomic_lock(lock_base/f"{snapshot['uuid']}.lock")
            lock.__enter__()
        except (ResourceUnavailable, OSError, subprocess.SubprocessError) as exc:
            failure = str(exc)
        else:
            try:
                snapshot = gpu_snapshot(device)
                write(Path(directory)/'gpu_preflight.json', snapshot)
            except Exception:
                lock.__exit__(None,None,None)
                raise
            if snapshot['free_gib'] >= minimum:
                try: yield snapshot
                finally: lock.__exit__(None,None,None)
                return
            failure = f"resource_unavailable: {snapshot['free_gib']:.2f} GiB free, requires {minimum}"
            lock.__exit__(None,None,None)
        if time.monotonic() >= deadline: raise ResourceUnavailable(failure)
        time.sleep(min(cfg.get('poll_seconds',30), max(0,deadline-time.monotonic())))


@contextmanager
def worker_budget(req, directory, timeout):
    cfg = req.get('resources', {}); path = cfg.get('budget_path')
    if not path or req.get('smoke') or req['kind']=='matte': yield timeout; return
    path = Path(path); bucket = cfg.get('budget_bucket','screen')
    lock_path = path.with_suffix('.lock')
    token = f'{socket.gethostname()}:{os.getpid()}:{Path(directory).resolve()}'
    with atomic_lock(lock_path):
        ledger = read(path) if path.exists() else dict(limits={'screen':4*3600,'validation':20*3600},used={},reservations={},history=[])
        used = ledger['used'].get(bucket,0)
        reserved = sum(v['seconds'] for v in ledger['reservations'].values() if v['bucket']==bucket)
        remaining = ledger['limits'][bucket]-used-reserved
        model=req.get('model') or req.get('config',{}).get('backend')
        history = [v['seconds'] for v in ledger['history'] if v['kind']==req['kind'] and v.get('model')==model]
        estimate = max(1.,max(history[-5:])*1.5) if history else 120.
        if remaining < estimate: raise BudgetExhausted(f'budget_exhausted: {bucket}, {remaining:.1f}s remaining; estimated {estimate:.1f}s')
        allowed = min(float(timeout),remaining)
        ledger['reservations'][token] = dict(bucket=bucket,seconds=allowed,pid=os.getpid(),host=socket.gethostname())
        write(path,ledger)
    started = time.monotonic()
    try: yield allowed
    finally:
        elapsed = min(time.monotonic()-started,allowed)
        with atomic_lock(lock_path):
            ledger = read(path); ledger['reservations'].pop(token,None)
            ledger['used'][bucket] = ledger['used'].get(bucket,0)+elapsed
            ledger['history'].append(dict(bucket=bucket,kind=req['kind'],model=model,seconds=elapsed,job=str(directory)))
            write(path,ledger)
