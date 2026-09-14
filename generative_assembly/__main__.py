from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from . import config,data
from .storage import Store,read,write,verify_job


def main():
    p=argparse.ArgumentParser(description='Generative fragment assembly experiments E0–E7')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ['run','train','evaluate','freeze','export','verify','status','doctor']:
        q=sub.add_parser(name)
        q.add_argument('--config',required=True); q.add_argument('--dataset',required=True); q.add_argument('--root',required=True)
        if name=='run':
            q.add_argument('--stages',nargs='+',default=['E0','E1','E2','E3','E4','E5'],choices=[f'E{i}' for i in range(8)])
            q.add_argument('--split',default='dev',choices=['train','dev','test','real'])
            q.add_argument('--limit',type=int); q.add_argument('--retry-failed',action='store_true'); q.add_argument('--allow-failures',action='store_true')
        if name=='train': q.add_argument('--retry-failed',action='store_true')
        if name=='evaluate': q.add_argument('--split',default='dev',choices=['train','dev','test','real'])
        if name=='export': q.add_argument('--out',required=True); q.add_argument('--kind',choices=['pipeline','research'],default='pipeline')
    q=sub.add_parser('demo'); q.add_argument('--out',required=True); q.add_argument('--sources',type=int,default=8)
    for name in ['import-v2','import-spec']:
        q=sub.add_parser(name); q.add_argument('--input',required=True); q.add_argument('--out',required=True); q.add_argument('--seed',type=int,default=42)
    q=sub.add_parser('lock-models'); q.add_argument('--config',required=True); q.add_argument('--out',required=True)
    q=sub.add_parser('unlock'); q.add_argument('--lock',required=True); q.add_argument('--confirmed-process-dead',action='store_true')
    args=p.parse_args()
    if args.command=='demo': data.demo(args.out,args.sources); return
    if args.command.startswith('import-'):
        (data.import_v2 if args.command=='import-v2' else data.import_spec)(args.input,args.out,args.seed); return
    if args.command=='lock-models':
        from .backends import lock_models
        cfg=config.load(args.config)
        write(args.out,cfg if cfg['smoke'] else lock_models(cfg)); print(args.out); return
    if args.command=='unlock':
        lock=Path(args.lock).resolve()
        if lock.name!='running.lock' or not args.confirmed_process_dead: raise ValueError('Inspect lock PID/host and pass --confirmed-process-dead')
        print(lock.read_text()); lock.unlink(); return
    cfg=config.load(args.config); ds=data.inventory(args.dataset)
    store=Store(args.root,cfg,Path(args.dataset).resolve(),getattr(args,'retry_failed',False))
    from .export import freeze,require_frozen,bundle
    if args.command=='doctor':
        import importlib.util
        import torch
        info=dict(python=sys.executable,torch=torch.__version__,cuda=torch.cuda.is_available(),
                  free_disk_gib=shutil.disk_usage(store.root).free/1024**3,
                  modules={m:importlib.util.find_spec(m) is not None for m in ['diffusers','accelerate','trimesh','open3d','huggingface_hub']},
                  source_counts={s:len({c['source_id'] for c in ds['cases'] if c['split']==s}) for s in ['train','dev','test','real']},
                  smoke=cfg['smoke'],model_locks=cfg['images']['revisions'],reconstruction_commit=cfg['reconstruction'].get('commit'))
        if torch.cuda.is_available(): info['gpu']=torch.cuda.get_device_name(0); info['gpu_total_bytes']=torch.cuda.get_device_properties(0).total_memory
        write(store.root/'doctor.json',info); print(json.dumps(info,indent=2)); return
    if args.command=='freeze': print(json.dumps(freeze(store),indent=2)); return
    if args.command=='export': print(json.dumps(bundle(store,args.out,args.kind),indent=2)); return
    if args.command=='verify':
        for row in store.jobs():
            if row['status']=='complete': verify_job(store.root,row)
        print('All completed artifact hashes verified'); return
    if args.command=='status': store.index(); print(json.dumps(read(store.root/'status.json'),indent=2)); return
    if args.command=='evaluate':
        if args.split=='test': require_frozen(store)
        from .evaluate import evaluate
        result=evaluate(store,args.split); print(f'Report: {store.root / "evaluation" / args.split / "REPORT.md"}; gate: {result["E4_primary_gate"]}'); return
    if args.command=='train':
        if (store.root/'frozen_method.json').exists(): raise ValueError('Training forbidden after method freeze')
        from .learning import train
        rows=train(store,ds['cases']); store.index()
        if any(r['status']=='failed' for r in rows): raise RuntimeError('Some E6 training arms failed; see status/logs and retry explicitly')
        return
    if args.split=='test': require_frozen(store)
    records=[c for c in ds['cases'] if c['split']==args.split]
    if args.limit: records=records[:args.limit]
    if not records: raise ValueError(f'No cases in {args.split}')
    from .experiments import STAGES
    for stage in args.stages:
        for rec in records:
            case=data.load_case(args.dataset,rec,cfg)
            if stage=='E6':
                from .learning import infer
                rows=infer(store,case)
            else: rows=STAGES[stage](store,case)
            store.index()
            if any(r['status']=='failed' for r in rows) and not args.allow_failures:
                raise RuntimeError(f'{stage} failure recorded; inspect error.txt/worker.log. Use --retry-failed after fixing the environment, or --allow-failures to retain failures in a benchmark.')


if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
