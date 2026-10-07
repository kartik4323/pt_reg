"""One process, one immutable task root, one experiment writer."""
import argparse
import json
from pathlib import Path
from generative_assembly import data, experiments as ex, learning
from generative_assembly.storage import read, write, digest, fingerprint
from . import stages, report
from .reuse import import_completed, import_students
from .store import SuiteStore


def execute(plan_path, task_id, root, retry=False):
    plan = read(plan_path)
    task = next(t for t in plan['tasks'] if t['id'] == task_id)
    run_root = Path(root)/'runs'/task_id
    store = SuiteStore(run_root, task['config'], plan['dataset'], retry=retry)
    # Source roots are refreshed only between immutable, completed imports.
    parents = [Path(root)/'runs'/identifier for identifier in task['depends_on']]
    if task['mode'].startswith('train_seed'):
        parents += [Path(root)/'runs'/t['id'] for t in plan['tasks']
                    if t['mode'] in ('assembly_gate','train_teacher')]
    import_stages = ['E0','E1','E1_ORACLE','E2','E4','E5']
    if task['mode'] == 'robust_students':
        import_stages += ['E7']
    import_completed(store, plan['source_runs']+parents, task['sources'], import_stages)
    if task['mode'] == 'robust_students':
        import_students(store, [p for p in parents if (p/'pseudo_labels'/'train.json').exists()])
    write(run_root/'experiment.json', {'dataset': plan['dataset'], 'profile': task['condition'], 'task': task_id,
                                      'allow_shared_gpu': plan.get('allow_shared_gpu', False)})
    records = [c for c in data.inventory(plan['dataset'])['cases']
               if c['source_id'] in task['sources'] and c['split'] == task['split']]
    # Training tasks consume full train data and score full development data.
    if task['mode'].startswith('train_seed'):
        records = [c for c in data.inventory(plan['dataset'])['cases'] if c['split'] in ('train','dev')]
    write(run_root/'requests.json', {'case_ids': [r['id'] for r in records],
          'mode': task['mode'], 'counts': task['counts'], 'requested_before_execution': True})
    errors = []
    if task['mode'].startswith('train_seed'):
        rows = stages.train(store, records)
        errors.extend(r.get('error') for r in rows if r['status'] == 'failed')
        for rec in records:
            if rec['split'] == 'dev':
                case = data.load_case(store.dataset, rec, store.config)
                ex.e0(store, case)
                learning.infer(store, case)
    else:
        for rec in records:
            case = data.load_case(store.dataset, rec, store.config)
            mode = task['mode']
            if mode.startswith('robust_'):
                stages.e7(store, case, {'robust_cpu':'cpu', 'robust_generated':'generated', 'robust_students':'students'}[mode])
                continue
            ex.e0(store, case)
            if mode in ('generate','train_teacher','real') or (mode == 'assembly' and task['kind'] == 'gpu'):
                ex.e1(store, case)
                try:
                    ex.e2(store, case)
                except RuntimeError as exc:
                    errors.append(str(exc))
            if mode in ('assembly','assembly_gate','train_teacher','real'):
                stages.e4(store, case)
            if mode in ('gate','assembly_gate','train_teacher','real'):
                stages.e5(store, case)
            if mode == 'compute':
                stages.e4(store, case, True)
    store.index()
    evaluated = []
    for split in sorted({r['split'] for r in records},key=lambda s:(s!='train',s)):
        report.evaluate(store, split)
        evaluated.append(split)
    failed = [r for r in store.jobs() if r['status'] == 'failed']
    result = {'status': 'complete_with_failures' if errors or failed else 'complete',
              'errors': errors, 'failed_jobs': len(failed), 'evaluated_splits': evaluated,
              'smoke': store.config['smoke'], 'study_identity': fingerprint(store.identity),
              'evaluation_artifacts': [{'path': str(p.relative_to(run_root)).replace('\\','/'), 'sha256': digest(p)}
                  for p in sorted((run_root/'evaluation').rglob('*')) if p.is_file()]}
    write(run_root/'task_result.json', result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',required=True); p.add_argument('--task',required=True); p.add_argument('--root',required=True)
    p.add_argument('--retry-failed',action='store_true')
    args=p.parse_args()
    execute(args.plan,args.task,args.root,args.retry_failed)


if __name__ == '__main__':
    main()
