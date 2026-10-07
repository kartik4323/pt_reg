import argparse
import json
from pathlib import Path
from generative_assembly import storage
from .plan import make_plan
from .scheduler import launch, validate_graph
from .reuse import import_completed
from .report import aggregate


def main():
    parser = argparse.ArgumentParser(description='Parallel E4--E7 architecture studies; development only')
    commands = parser.add_subparsers(dest='command',required=True)
    for name in ('plan','launch'):
        p=commands.add_parser(name)
        p.add_argument('--base',required=True); p.add_argument('--dataset',required=True)
        p.add_argument('--root',required=True); p.add_argument('--source-runs',nargs='*',default=[])
        p.add_argument('--studies',nargs='+',choices=['E4','E5','E6','E7'],default=['E4','E5','E6','E7'])
        p.add_argument('--gpus',nargs='*',default=[]); p.add_argument('--exclude-gpus',nargs='*',default=[])
        p.add_argument('--allow-shared-gpu',action='store_true',help='Allow existing compute processes; record shared timing and invalidate isolated B5 timing comparisons')
        p.add_argument('--cpu-workers',type=int,default=2); p.add_argument('--shards',type=int)
        if name=='launch':
            p.add_argument('--dry-run',action='store_true'); p.add_argument('--retry-failed',action='store_true')
            p.add_argument('--python'); p.add_argument('--resource-lock-dir'); p.add_argument('--threads',type=int,default=1)
    for name in ('status','aggregate','import'):
        p=commands.add_parser(name); p.add_argument('--root',required=True)
        if name=='import':
            p.add_argument('--task',required=True); p.add_argument('--source-runs',nargs='+',required=True)
    p=commands.add_parser('unlock')
    p.add_argument('--lock',required=True); p.add_argument('--confirmed-process-dead',action='store_true')
    args=parser.parse_args()
    if args.command=='unlock':
        path=Path(args.lock).resolve()
        if not args.confirmed_process_dead or path.suffix!='.lock':
            raise ValueError('Inspect lock host/PID and confirm the process is dead before unlock')
        print(path.read_text()); path.unlink(); return
    root=Path(args.root).expanduser().resolve()
    if args.command in ('plan','launch'):
        plan=make_plan(args.base,args.dataset,args.studies,args.gpus,args.cpu_workers,args.shards,args.exclude_gpus,args.source_runs,args.allow_shared_gpu)
        validate_graph(plan['tasks'])
        if args.command=='plan' or args.dry_run:
            summary={k:v for k,v in plan.items() if k not in ('tasks','base_config')}
            summary['condition_count']=len({t['condition'] for t in plan['tasks']})
            summary['task_count']=len(plan['tasks'])
            summary['totals_upper_bound']={key:sum(t['counts'][key] for t in plan['tasks'])
                for key in ('edited_images_upper_bound','reconstructions_upper_bound')}
            summary['tasks']=[{k:v for k,v in t.items() if k!='config'} for t in plan['tasks']]
            print(json.dumps(summary,indent=2)); return
        result=launch(root,plan,args.python,args.retry_failed,args.resource_lock_dir,args.threads)
        print(json.dumps(result,indent=2))
        if any(s in ('failed','blocked_dependency','complete_with_failures') for s in result['tasks'].values()):
            raise SystemExit(1)
        return
    plan=storage.read(root/'plan.json')
    if args.command=='status':
        value=storage.read(root/'suite_status.json') if (root/'suite_status.json').exists() else {'tasks':{t['id']:'not_started' for t in plan['tasks']}}
        value['scheduler_lock_present']=(root/'suite.lock').exists()
        value['lock_note']='Lock files are not proof of process liveness; inspect host/PID before unlock.'
        print(json.dumps(value,indent=2))
    elif args.command=='aggregate':
        print(json.dumps(aggregate(root,plan),indent=2))
    else:
        if (root/'suite.lock').exists():
            raise ValueError('Stop the suite before an explicit import; one writer per root')
        task=next(t for t in plan['tasks'] if t['id']==args.task)
        store=storage.Store(root/'runs'/task['id'],task['config'],plan['dataset'])
        print(json.dumps(import_completed(store,args.source_runs,task['sources']),indent=2))


if __name__=='__main__':
    main()
