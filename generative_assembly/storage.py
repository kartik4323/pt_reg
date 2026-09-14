"""Immutable experiment identity, atomic job records and checked artifact reuse."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def code_hash():
    base = Path(__file__).parent
    return fingerprint({str(p.relative_to(base)): digest(p) for p in sorted(base.rglob('*.py')) if '__pycache__' not in p.parts})


def checked(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path escapes artifact root: {relative}')
    return path


def verify_job(root, record):
    for asset in record.get('artifacts', []):
        path = checked(root, asset['path'])
        if not path.is_file() or digest(path) != asset['sha256']:
            raise ValueError(f'Artifact missing or changed: {path}')


class Store:
    def __init__(self, root, config, dataset, retry=False):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config, self.dataset, self.retry = config, dataset, retry
        self.identity = dict(schema_version=1, config=config, dataset_sha256=digest(dataset), code_sha256=code_hash())
        path = self.root / 'study.json'
        if path.exists():
            old = read(path)
            if old['identity'] != self.identity:
                raise ValueError('Run identity changed (code/config/dataset). Use a NEW --root; never mix old results.')
        else:
            write(path, dict(identity=self.identity, created=time.time(), python=sys.version, platform=platform.platform(), smoke=bool(config.get('smoke'))))

    def jobs(self, stage=None, split=None):
        records = [read(p) for p in sorted((self.root / 'jobs').glob('*/result.json'))]
        return [r for r in records if (stage is None or r['stage'] == stage) and (split is None or r['split'] == split)]

    def find(self, stage, case, arm=None):
        return [r for r in self.jobs(stage) if r['case_id'] == case and r['status'] == 'complete' and (arm is None or r['arm'] == arm)]

    def run(self, stage, case, arm, fn, *, oracle=False, extra=None, parents=()):
        if any(r.get('oracle') for r in parents) and not oracle:
            raise ValueError('Oracle ancestry cannot be relabelled as a public job')
        key = dict(stage=stage, case_id=case['id'], source_id=case['source_id'], split=case['split'], arm=arm,
                   oracle=bool(oracle), input_sha256=case.get('sha256'), extra=extra or {}, parents=[r['job_id'] for r in parents])
        jid = fingerprint(key)[:24]
        directory = self.root / 'jobs' / jid
        directory.mkdir(parents=True, exist_ok=True)
        result = directory / 'result.json'
        if result.exists():
            old = read(result)
            if old['status'] == 'complete':
                verify_job(self.root, old)
                return old
            if not self.retry:
                return old
        for parent in parents:
            verify_job(self.root, parent)
        lock = directory / 'running.lock'
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f'Job locked: {lock}. Check its PID/host before unlock.')
        os.write(fd, json.dumps(dict(pid=os.getpid(), host=platform.node(), time=time.time())).encode())
        os.close(fd)
        started = time.time()
        versions = {}
        for package in ('numpy', 'scipy', 'Pillow', 'trimesh', 'torch', 'diffusers'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        row = dict(key, job_id=jid, status='running', smoke=bool(self.config.get('smoke')), started=started, runtime_versions=versions)
        print(f'{stage} {case["id"]} {arm} {jid}', flush=True)
        try:
            value = fn(directory)
            assets = []
            for p in sorted(directory.rglob('*')):
                if p.is_file() and p.name not in ('running.lock', 'result.json') and not p.name.endswith('.tmp'):
                    assets.append(dict(path=str(p.relative_to(self.root)).replace('\\', '/'), sha256=digest(p), bytes=p.stat().st_size))
            row.update(status='complete', output=value, artifacts=assets)
        except Exception as exc:
            (directory / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}', artifacts=[])
            print(row['error'], flush=True)
        finally:
            row['seconds'] = time.time() - started
            write(result, row)
            lock.unlink(missing_ok=True)
        return row

    def artifact(self, row, filename):
        verify_job(self.root, row)
        path = self.root / 'jobs' / row['job_id'] / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def index(self):
        rows = self.jobs()
        text = ''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in rows)
        (self.root / 'results.jsonl').write_text(text, encoding='utf-8')
        write(self.root / 'status.json', {s: {k: sum(r['stage'] == s and r['status'] == k for r in rows)
              for k in ('complete', 'failed')} for s in sorted({r['stage'] for r in rows})})
        return rows
