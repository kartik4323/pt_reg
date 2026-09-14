"""Small variable-part PointNet assembler; fragment-only geometric/pseudo supervision."""
from __future__ import annotations
import random
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from . import data, geometry as g
from .storage import write, read, fingerprint, digest
from .experiments import prepared, save_prediction


def make_model(hidden):
    import torch
    from torch import nn
    class Assembler(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(nn.Linear(3,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
            self.head=nn.Sequential(nn.Linear(4*hidden,hidden),nn.ReLU(),nn.Linear(hidden,9))
            with torch.no_grad():
                self.head[-1].bias.copy_(torch.tensor([1.,0,0,0,1,0,0,0,0]))
        def forward(self,points,anchor):
            f=torch.stack([self.encoder(p).max(0).values for p in points])
            context=torch.cat([f.mean(0),f.max(0).values,f[anchor]])
            out=self.head(torch.cat([f,context.expand(len(f),-1)],1))
            a=torch.nn.functional.normalize(out[:,:3],dim=1)
            b=out[:,3:6]-(a*out[:,3:6]).sum(1,keepdim=True)*a
            b=torch.nn.functional.normalize(b,dim=1)
            r=torch.stack([a,b,torch.linalg.cross(a,b)],dim=2)
            t=out[:,6:9]
            mask=torch.arange(len(f),device=f.device)!=anchor
            r=torch.where(mask[:,None,None],r,torch.eye(3,device=f.device)[None])
            t=torch.where(mask[:,None],t,torch.zeros_like(t))
            return r,t
    return Assembler()


def geometry_loss(points,normals,R,t,anchor,cfg):
    import torch
    P=[p@r.T+v for p,r,v in zip(points,R,t)]
    N=[n@r.T for n,r in zip(normals,R)]
    # Build a sparse contact graph with detached indices, not an all-to-all attraction.
    fake=dict(points=[p.detach().cpu().numpy() for p in points],normals=[n.detach().cpu().numpy() for n in normals])
    poses=np.repeat(np.eye(4)[None],len(P),0); poses[:,:3,:3]=R.detach().cpu().numpy(); poses[:,:3,3]=t.detach().cpu().numpy()
    edges=g.contact_pairs(fake,poses,cfg['contact_fraction'])
    loss=torch.zeros((),device=R.device)
    for i,j,a,b in edges:
        loss=loss+((P[i][a]-P[j][b])**2).mean()+0.01*((N[i][a]+N[j][b])**2).mean()
        # Local signed-surface proxy; explicit approximation for open scans.
        dist=torch.cdist(P[i],P[j]); ids=dist.argmin(1)
        signed=((P[i]-P[j][ids])*N[j][ids]).sum(1)
        mask=dist.min(1).values<0.08
        if mask.any(): loss=loss+torch.relu(-signed[mask]-0.005).square().mean()
    return loss/max(1,len(edges)) + 0.0001*t.square().mean()


def pseudo_labels(store,records):
    K=len(store.config['image_seeds']); labels=[]
    for rec in records:
        # Same teacher outputs for all/filtered/random controls.
        rows=store.find('E5',rec['id'],f'always__K{K}')
        gates=store.find('E5',rec['id'],f'gated__K{K}')
        if not rows: continue
        row=rows[0]
        if row['oracle'] or any(r['oracle'] for r in gates): raise ValueError('Oracle pseudo-label rejected')
        with np.load(store.artifact(row,'poses.npz'),allow_pickle=False) as f: poses=f['normalized']
        accepted=bool(gates and gates[0]['output']['diagnostics']['template_accepted'] and gates[0]['output']['diagnostics'].get('budget_complete'))
        diagnostics=row['output']['diagnostics']
        accepted=accepted and diagnostics['contact_after']<=max(diagnostics['contact_before']*store.config['solver']['gate_contact_ratio'],1e-5)
        labels.append(dict(case_id=rec['id'],source_id=rec['source_id'],teacher_job=row['job_id'],oracle=False,
                           gate_job=gates[0]['job_id'] if gates else None,smoke=row['smoke'],accepted=bool(accepted),poses=poses.tolist()))
    write(store.root/'pseudo_labels'/'train.json',dict(schema_version=1,split='train',labels=labels,selection_uses_reference=False))
    return labels


def train(store,records):
    import torch
    cfg=store.config; train_cfg=cfg['training']
    records=[r for r in records if r['split']=='train']
    if not records: raise ValueError('No training sources')
    labels=pseudo_labels(store,records)
    lookup={r['case_id']:r for r in labels}
    cases={r['id']:data.load_case(store.dataset,r,cfg) for r in records}
    rows=[]
    for seed in train_cfg['seeds']:
        accepted=[r['case_id'] for r in labels if r['accepted']]
        rng=np.random.default_rng(seed)
        random_ids=list(rng.choice(list(lookup),len(accepted),replace=False)) if accepted else []
        pools={'geometry':list(cases),'all':list(lookup),'filtered':accepted,'random_count':random_ids}
        for arm,pool in pools.items():
            record=dict(id=f'training_{seed}',source_id='training_pool',split='train',sha256=fingerprint(labels))
            def fit(out,arm=arm,pool=pool,seed=seed):
                if not pool: raise RuntimeError(f'No eligible {arm} pseudo-labels; this is an experimental failure, not successful training')
                device=train_cfg['device']
                if device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA training requested but unavailable')
                torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                model=make_model(train_cfg['hidden']).to(device)
                opt=torch.optim.Adam(model.parameters(),lr=train_cfg['lr'])
                checkpoint=out/'last.pt'; start=0
                if checkpoint.exists():
                    state=torch.load(checkpoint,map_location=device,weights_only=False)
                    if state['config_hash']!=fingerprint(cfg) or state['pool']!=pool: raise ValueError('Checkpoint lineage mismatch')
                    model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); start=state['step']
                    torch.set_rng_state(state['torch_rng'].cpu())
                    if state.get('cuda_rng') is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda_rng'])
                curve=[]
                for step in range(start,train_cfg['updates']):
                    prng=np.random.default_rng(data.seed_for(seed,step))
                    cid=pool[int(prng.integers(len(pool)))]; c=cases[cid]
                    Q=Rotation.random(len(c['points']),random_state=prng).as_matrix(); a=c['anchor']
                    points=[torch.tensor(p@q.T,dtype=torch.float32,device=device) for p,q in zip(c['points'],Q)]
                    ns=[torch.tensor(n@q.T,dtype=torch.float32,device=device) for n,q in zip(c['normals'],Q)]
                    R,t=model(points,a)
                    loss=geometry_loss(points,ns,R,t,a,cfg['solver'])
                    if arm!='geometry':
                        T=np.asarray(lookup[cid]['poses']); Rt=Q[a][None]@T[:,:3,:3]@Q.transpose(0,2,1); tt=T[:,:3,3]@Q[a].T
                        target_R=torch.tensor(Rt,dtype=torch.float32,device=device); target_t=torch.tensor(tt,dtype=torch.float32,device=device)
                        pseudo=sum(((p@r.T+v)-(p@tr.T+tv)).square().mean() for p,r,v,tr,tv in zip(points,R,t,target_R,target_t))/len(points)
                        loss=pseudo+train_cfg['consistency_weight']*loss
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
                    if step%10==0: curve.append(dict(step=step+1,loss=float(loss.detach())))
                    if (step+1)%train_cfg['checkpoint_every']==0 or step+1==train_cfg['updates']:
                        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step+1,hidden=train_cfg['hidden'],
                                   config_hash=fingerprint(cfg),pool=pool,seed=seed,arm=arm,oracle=False,smoke=cfg['smoke'],
                                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
                        torch.save(state,out/'last.pt.tmp'); (out/'last.pt.tmp').replace(checkpoint)
                        with (out/'learning_curve.jsonl').open('a',encoding='utf-8') as f:
                            for item in curve: f.write(json.dumps(item)+'\n')
                        curve=[]
                        print(f'E6 {arm} seed={seed} step={step+1} loss={float(loss.detach()):.6f}',flush=True)
                write(out/'training.json',dict(arm=arm,seed=seed,pool=pool,updates=train_cfg['updates'],reference_labels_used=False,
                                               teacher_kind='frozen_geometry_and_generated_template',round=1))
                return dict(arm=arm,seed=seed,updates=train_cfg['updates'],unique_cases=len(pool),checkpoint='last.pt')
            import json
            rows.append(store.run('E6_TRAIN',record,f'{arm}__seed{seed}',fit))
    return rows


def infer(store,case):
    import torch
    rows=[]
    for trained in store.jobs('E6_TRAIN'):
        if trained['status']!='complete': continue
        if trained['oracle']: raise ValueError('Oracle-trained checkpoint rejected')
        def predict(out,trained=trained):
            state=torch.load(store.artifact(trained,'last.pt'),map_location='cpu',weights_only=False)
            if state.get('oracle'): raise ValueError('Oracle checkpoint rejected')
            model=make_model(state['hidden']); model.load_state_dict(state['model']); model.eval()
            with torch.inference_mode():
                R,t=model([torch.tensor(p,dtype=torch.float32) for p in case['points']],case['anchor'])
            T=np.repeat(np.eye(4)[None],len(R),0); T[:,:3,:3]=R.numpy(); T[:,:3,3]=t.numpy()
            return save_prediction(out,case,T,dict(contact_after=g.contact_score(case,T,store.config['solver']),
                                                   solid_collision_measured=False,template_accepted=False),checkpoint_job=trained['job_id'])
        rows.append(store.run('E6',case['record'],trained['arm'],predict,parents=[trained]))
    if not rows: raise RuntimeError('No E6 checkpoints; run train first')
    return rows
