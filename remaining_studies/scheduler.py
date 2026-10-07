"""Subprocess scheduler with isolated roots, explicit GPUs and exclusive timing."""
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from generative_assembly.storage import read, write, fingerprint, digest, code_hash, verify_job, checked
from . import report
from .plan import suite_hash


def lock(path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, __import__('json').dumps(dict(pid=os.getpid(), host=platform.node(), time=time.time(), **metadata)).encode())
    finally:
        os.close(fd)
    return path


def validate_graph(tasks):
    remaining = {t['id']: set(t['depends_on']) for t in tasks}
    if len(remaining) != len(tasks):
        raise ValueError('Duplicate task identifiers')
    if any(d not in remaining for deps in remaining.values() for d in deps):
        raise ValueError('Unknown task dependency')
    done = set()
    while remaining:
        ready = {tid for tid, deps in remaining.items() if deps <= done}
        if not ready:
            raise ValueError('Task dependency cycle')
        done |= ready
        for tid in ready:
            del remaining[tid]


def check_gpus(ids, allow_shared=False):
    try:
        rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
        processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('GPU launch needs nvidia-smi and explicitly free --gpus IDs') from exc
    known = dict(tuple(part.strip() for part in line.split(',', 1))
                 for line in rows.splitlines() if line.strip())
    busy = {line.split(',')[0].strip() for line in processes.splitlines() if line.strip()}
    selected = {}
    for identifier in ids:
        uuid = known.get(identifier, identifier if identifier in known.values() else None)
        if not uuid:
            available = ', '.join(f'{index} ({uuid})' for index, uuid in known.items()) or 'none'
            raise ValueError(f'Unknown GPU {identifier}. GPUs reported by nvidia-smi: {available}. '
                             'Set GA_GPUS to available, unoccupied IDs and remove them from GA_EXCLUDE_GPUS.')
        selected[identifier] = {'uuid': uuid, 'compute_processes_present': uuid in busy}
        if uuid in busy and not allow_shared:
            raise RuntimeError(f'GPU {identifier} has compute processes. Reserve a free GPU or use '
                               '--allow-shared-gpu to explicitly permit sharing. Existing jobs are never stopped.')
    return {'allow_shared_gpu': bool(allow_shared), 'selected_gpus': selected,
            'compute_processes_at_launch': processes.splitlines()}


class Leases:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)

    def acquire(self, task, gpu=None):
        mutex = self.root/'admission.lock'
        try:
            lock(mutex, {'task': task['id']})
        except FileExistsError:
            return None
        acquired = []
        try:
            exclusive = self.root/'exclusive.lock'
            if exclusive.exists() or (task['kind'] == 'exclusive' and list(self.root.glob('*.task.lock'))):
                return None
            if task['kind'] == 'exclusive':
                acquired.append(lock(exclusive, {'task': task['id']}))
            if gpu is not None:
                safe = gpu.replace('/','_').replace('\\','_')
                acquired.append(lock(self.root/f'gpu-{safe}.lock', {'task': task['id']}))
            acquired.append(lock(self.root/f'{os.getpid()}-{task["id"]}.task.lock', {'task': task['id']}))
            return acquired
        except FileExistsError:
            for path in acquired:
                path.unlink()
            return None
        finally:
            mutex.unlink()

    def release(self, paths):
        for path in paths:
            path.unlink(missing_ok=True)


def launch(root, plan, python=None, retry=False, lock_dir=None, threads=1):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    validate_graph(plan['tasks'])
    if not plan.get('model_locks_ready', False):
        raise ValueError('Lock model revisions before research launch: '+str(plan.get('missing_model_locks')))
    gpu_preflight = {'allow_shared_gpu': plan.get('allow_shared_gpu', False), 'selected_gpus': {}}
    if not plan['smoke'] and any(t['kind'] == 'gpu' for t in plan['tasks']):
        if not plan['gpu_ids']:
            raise ValueError('Real neural/training studies need explicit --gpus IDs')
        gpu_preflight = check_gpus(plan['gpu_ids'], plan.get('allow_shared_gpu', False))
        if plan.get('allow_shared_gpu', False):
            print('Shared GPU mode: existing processes are permitted; isolated B5 timing comparisons are disabled.', flush=True)
    if threads < 1:
        raise ValueError('Thread limit must be positive')
    identity_path = root/'plan.json'
    if identity_path.exists():
        saved_plan=read(identity_path)
        semantic=lambda value:{k:v for k,v in value.items() if k!='measured_runtime_estimates'}
        if semantic(saved_plan) != semantic(plan):
            raise ValueError('Suite plan changed; choose a new --root')
        plan=saved_plan
    else:
        write(identity_path, plan)
    if plan['dataset_sha256'] != digest(plan['dataset']) or plan['engine_code_sha256'] != code_hash() or plan['suite_code_sha256'] != suite_hash():
        raise ValueError('Dataset/code changed since planning')
    lease_root = lock_dir or os.environ.get('GA_RESOURCE_LOCK_DIR') or Path(tempfile.gettempdir())/'generative-assembly-resource-locks'
    leases = Leases(lease_root)
    active, states = {}, {}
    state_path = root/'suite_status.json'
    for task in plan['tasks']:
        result = root/'runs'/task['id']/'task_result.json'
        states[task['id']] = read(result)['status'] if result.exists() else 'pending'
        if result.exists() and states[task['id']] in ('complete','complete_with_failures'):
            saved = read(result)
            run = result.parent
            study = read(run/'study.json')
            if saved.get('study_identity') != fingerprint(study['identity']):
                raise ValueError(f'Task identity changed: {run}')
            for path in (run/'jobs').glob('*/result.json'):
                row = read(path)
                if row['status']=='complete':
                    verify_job(run,row)
            for asset in saved.get('evaluation_artifacts',[]):
                if digest(checked(run,asset['path'])) != asset['sha256']:
                    raise ValueError(f'Task evaluation artifact changed: {asset["path"]}')
        if retry and states[task['id']] in ('failed','complete_with_failures'):
            states[task['id']] = 'pending'
        write(root/'configs'/f'{task["id"]}.json',task['config'])
    suite_lock = lock(root/'suite.lock', {'root': str(root)})
    def snapshot():
        write(state_path, {'tasks': states, 'active': list(active), 'resource_lock_dir': str(leases.root),
                           'smoke': plan['smoke'], 'allow_shared_gpu': plan.get('allow_shared_gpu', False)})
    try:
        write(root/'gpu_preflight.json', dict(gpu_preflight, checked_at=time.time()))
        while any(s in ('pending','running') for s in states.values()):
            for tid, (process, log, handles, gpu) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close(); leases.release(handles); del active[tid]
                path = root/'runs'/tid/'task_result.json'
                states[tid] = read(path)['status'] if code == 0 and path.exists() else 'failed'
                print(f'{tid}: {states[tid]} (exit {code})', flush=True)
            completed = {tid for tid,s in states.items() if s in ('complete','complete_with_failures')}
            ready = [t for t in plan['tasks'] if states[t['id']] == 'pending' and set(t['depends_on']) <= completed]
            for task in plan['tasks']:
                if states[task['id']] == 'pending' and any(states[d] in ('failed','blocked_dependency','blocked_gate') for d in task['depends_on']):
                    states[task['id']] = 'blocked_dependency'
            # An exclusive timing task drains this suite before admitting work.
            timed = [t for t in ready if t['kind'] == 'exclusive']
            if timed:
                ready = timed[:1] if not active else []
            for task in ready:
                if states[task['id']] != 'pending':
                    continue
                if task['gate_required']:
                    decision = report.development_gate(root, plan)
                    if not plan['smoke'] and not (decision['E4_pass'] and decision['E5_pass']):
                        states[task['id']] = 'blocked_gate'
                        print(f'{task["id"]}: development gates not passed; training not started',flush=True)
                        continue
                if any(next(t for t in plan['tasks'] if t['id']==tid)['kind']=='exclusive' for tid in active):
                    break
                cpu_active = sum(next(t for t in plan['tasks'] if t['id']==tid)['kind']=='cpu' for tid in active)
                if task['kind']=='cpu' and cpu_active >= plan['cpu_workers']:
                    continue
                used = {entry[3] for entry in active.values()}
                free = [g for g in plan['gpu_ids'] if g not in used]
                if task['kind']=='gpu' and not free and not plan['smoke']:
                    continue
                # Smoke workers share bounded CPU slots and never request GPUs.
                if plan['smoke'] and len(active) >= plan['cpu_workers']:
                    continue
                candidates = free if task['kind']=='gpu' and not plan['smoke'] else [None]
                handles, gpu = None, None
                for candidate in candidates:
                    handles=leases.acquire(task,candidate)
                    if handles is not None:
                        gpu=candidate
                        break
                if handles is None:
                    continue
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = gpu if gpu is not None else ''
                env['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent)+os.pathsep+env.get('PYTHONPATH','')
                env['PYTHONUNBUFFERED']='1'
                for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
                    env[key]=str(threads)
                log_path = root/'logs'/f'{task["id"]}.log'
                log_path.parent.mkdir(exist_ok=True)
                log = log_path.open('a',encoding='utf-8')
                command = [python or sys.executable,'-m','remaining_studies.worker','--plan',str(identity_path),
                           '--task',task['id'],'--root',str(root)]
                if retry:
                    command += ['--retry-failed']
                try:
                    process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,
                                               start_new_session=os.name!='nt',
                                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                except BaseException:
                    log.close(); leases.release(handles)
                    raise
                active[task['id']] = (process,log,handles,gpu)
                states[task['id']] = 'running'
                print(f'Start {task["id"]}: {task["kind"]}, GPU={gpu or "none"}, sources={len(task["sources"])}',flush=True)
            snapshot()
            if active:
                time.sleep(.5)
            elif any(s=='pending' for s in states.values()):
                # Resource leases may belong to another live suite; retain a
                # pending state and wait, never remove their locks automatically.
                time.sleep(1)
        decision = report.aggregate(root,plan)
        snapshot()
        return {'tasks':states,'development_gate':decision}
    finally:
        # Release resources only after terminating our own workers. Never touch
        # another suite's processes or any user's imagination workers.
        for process,log,handles,gpu in active.values():
            if os.name=='nt':
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
            else:
                import signal
                try:
                    os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            if os.name!='nt':
                try:
                    os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
            log.close(); leases.release(handles)
        suite_lock.unlink(missing_ok=True)
