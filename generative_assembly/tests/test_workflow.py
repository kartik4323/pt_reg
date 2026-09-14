import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from generative_assembly import config,data,geometry as g,experiments as ex
from generative_assembly.storage import Store,read,write,digest
from generative_assembly.evaluate import pose_metrics,bootstrap_delta,evaluate
from generative_assembly.export import freeze,require_frozen,bundle


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        data.demo(self.root/'data',6)
        self.dataset=self.root/'data'/'dataset.json'
        self.cfg=config.load(Path(config.__file__).parent/'configs'/'smoke.json')
        self.doc=data.inventory(self.dataset)
        self.case=data.load_case(self.dataset,self.doc['cases'][2],self.cfg)
        self.store=Store(self.root/'run',self.cfg,self.dataset)

    def test_reference_gauge_and_export(self):
        ref=data.reference(self.dataset,self.case)
        self.assertTrue(np.allclose(ref['poses'][self.case['anchor']],np.eye(4)))
        metrics=pose_metrics(self.case,ref,ref['poses'],self.cfg)
        self.assertTrue(metrics['success'])
        self.assertLess(max(metrics['part_chamfer']),1e-10)
        export=g.export_poses(self.case,ref['poses'])
        self.assertTrue(np.allclose(export,ref['original_poses'],atol=1e-7))

    def test_public_loader_does_not_open_reference(self):
        original=np.load
        def guard(path,*args,**kwargs):
            self.assertNotIn('evaluator_only',str(path))
            return original(path,*args,**kwargs)
        with patch('numpy.load',side_effect=guard):
            c=data.load_case(self.dataset,self.doc['cases'][2],self.cfg)
            rows=ex.e0(self.store,c)
        self.assertEqual(rows[0]['status'],'complete')

    def test_label_leakage_rejected(self):
        doc=read(self.dataset); rec=doc['cases'][0]; p=self.dataset.parent/rec['path']
        with np.load(p) as f: content={k:f[k] for k in f.files}
        content['ground_truth_poses']=np.zeros((2,4,4)); np.savez_compressed(p,**content)
        rec['sha256']=digest(p); write(self.dataset,doc)
        with self.assertRaisesRegex(ValueError,'Public NPZ'): data.inventory(self.dataset)

    def test_import_v2_reuses_geometry_and_keeps_references_separate(self):
        source=self.root/'prepared_v2'; source.mkdir()
        rng=np.random.default_rng(42)
        points=rng.normal(size=(2,32,3)).astype(np.float32)
        np.savez_compressed(source/'source.npz',target_points=np.concatenate(points))
        np.savez_compressed(source/'pattern.npz',points=points,fracture_labels=np.zeros((2,32),np.uint8))
        manifest=dict(schema_version=2,fingerprint='fixture',sources=[dict(source_id='bottle',path='source.npz',sha256=digest(source/'source.npz'))],
                      patterns=[dict(source_id='bottle',pattern_id='break0',path='pattern.npz',sha256=digest(source/'pattern.npz'),
                                     split='cut_holdout',cut_family='held_out',band='hard')])
        write(source/'manifest.json',manifest)
        data.import_v2(source/'manifest.json',self.root/'imported')
        imported=self.root/'imported'/'dataset.json'; doc=data.inventory(imported)
        rec=doc['cases'][0]
        self.assertEqual(rec['split'],'test'); self.assertEqual(rec['original_split'],'cut_holdout')
        self.assertEqual(rec['cut_family'],'held_out'); self.assertEqual(doc['provenance']['source_fingerprint'],'fixture')
        refs=read(imported.parent/'evaluator_only'/'index.json')
        with np.load(imported.parent/rec['path']) as public, np.load(imported.parent/refs['break0']['path']) as reference:
            self.assertEqual(set(public.files),{'points_0','points_1'})
            for i in range(2):
                self.assertTrue(np.allclose(g.apply(public[f'points_{i}'],reference['world_transforms'][i]),points[i],atol=1e-6))
        self.assertEqual(digest(source/'source.npz'),manifest['sources'][0]['sha256'])
        self.assertEqual(digest(source/'pattern.npz'),manifest['patterns'][0]['sha256'])

    def test_source_split_leakage_rejected(self):
        doc=read(self.dataset); doc['cases'][2]['source_hash']=doc['cases'][0]['source_hash'];write(self.dataset,doc)
        with self.assertRaisesRegex(ValueError,'crosses splits'): data.inventory(self.dataset)

    def test_resume_hash_and_config_guards(self):
        row=ex.e0(self.store,self.case)[0]
        replay=ex.e0(self.store,self.case)[0]
        self.assertEqual(row['job_id'],replay['job_id']); self.assertEqual(row['started'],replay['started'])
        bad=copy.deepcopy(self.cfg);bad['seed']+=1
        with self.assertRaisesRegex(ValueError,'identity changed'): Store(self.store.root,bad,self.dataset)
        self.store.artifact(row,'checks.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'changed'): ex.e0(self.store,self.case)

    def test_failed_job_is_not_cached_success(self):
        def fail(out): raise RuntimeError('deliberate')
        r=self.store.run('X',self.case['record'],'fail',fail)
        self.assertEqual(r['status'],'failed')
        r2=self.store.run('X',self.case['record'],'fail',lambda out:dict(ok=True))
        self.assertEqual(r2['status'],'failed')
        retry=Store(self.store.root,self.cfg,self.dataset,retry=True)
        self.assertEqual(retry.run('X',self.case['record'],'fail',lambda out:dict(ok=True))['status'],'complete')

    def test_oracle_ancestry_rejected(self):
        oracle=self.store.run('X',self.case['record'],'oracle',lambda out:dict(ok=True),oracle=True)
        with self.assertRaisesRegex(ValueError,'Oracle ancestry'):
            self.store.run('X',self.case['record'],'public',lambda out:dict(ok=True),parents=[oracle])

    def test_all_stages_smoke_and_bundle(self):
        import torch
        torch.set_num_threads(1)
        for rec in self.doc['cases']:
            if rec['split'] not in ('train','dev'): continue
            case=data.load_case(self.dataset,rec,self.cfg)
            for stage in ['E0','E1','E2','E4','E5']:
                rows=ex.STAGES[stage](self.store,case)
                self.assertTrue(all(r['status']=='complete' for r in rows),[r.get('error') for r in rows])
        self.assertTrue(all(r['status']=='complete' for r in ex.e3(self.store,self.case)))
        from generative_assembly.learning import train,infer
        trained=train(self.store,self.doc['cases'])
        self.assertTrue(all(r['status']=='complete' for r in trained),[r.get('error') for r in trained])
        self.assertEqual(len(infer(self.store,self.case)),4)
        changed=ex.e7(self.store,self.case)
        self.assertTrue(all(r['status']=='complete' for r in changed),[r.get('error') for r in changed])
        summary=evaluate(self.store,'dev')
        self.assertEqual(summary['E4_primary_gate'],'SMOKE_NOT_RESEARCH_EVIDENCE')
        self.assertTrue(summary['targeted_comparisons'])
        self.assertTrue(summary['robustness_comparisons'])
        review_path=self.store.root/'evaluation'/'dev'/'image_review.json'
        review=read(review_path); review[0]['notes']='review retained'; write(review_path,review)
        evaluate(self.store,'dev')
        self.assertEqual(read(review_path)[0]['notes'],'review retained')
        evaluate(self.store,'train')
        self.assertTrue((self.store.root/'evaluation'/'train'/'pseudo_label_quality.json').exists())
        freeze(self.store);require_frozen(self.store)
        with self.assertRaisesRegex(ValueError,'Smoke'): bundle(self.store,self.root/'pipeline')
        output=bundle(self.store,self.root/'evidence','research')
        self.assertGreater(output['files'],20)
        self.assertTrue((self.root/'evidence'/'SHA256SUMS.json').is_file())

    def test_evaluator_reference_identity_guard(self):
        evaluate(self.store,'dev')
        index=self.dataset.parent/'evaluator_only'/'index.json'
        doc=read(index); doc['new_reference']={}; write(index,doc)
        with self.assertRaisesRegex(ValueError,'reference identity changed'):
            evaluate(self.store,'dev')

    def test_source_paired_statistics(self):
        a={'a':[1,1,1],'b':[0]};b={'a':[0,0,0],'b':[0]}
        self.assertAlmostEqual(bootstrap_delta(a,b,count=20)['difference'],0.5)


if __name__=='__main__': unittest.main()
