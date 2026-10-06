import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from generative_assembly import config, data, experiments as ex, geometry as g
from generative_assembly.storage import Store, read, write, digest
from remaining_studies.plan import make_plan, suite_hash
from remaining_studies.reuse import import_completed
from remaining_studies.scheduler import Leases, validate_graph, check_gpus, launch
from remaining_studies import stages, report


class RemainingStudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        data.demo(self.root/'data',6)
        self.dataset=self.root/'data'/'dataset.json'
        self.base=Path(config.__file__).parent/'configs'/'smoke.json'
        self.cfg=config.load(self.base)
        self.cfg.update(suite_code_sha256=suite_hash(),suite_selector='exterior')
        self.doc=data.inventory(self.dataset)
        self.rec=next(c for c in self.doc['cases'] if c['split']=='dev')
        self.case=data.load_case(self.dataset,self.rec,self.cfg)

    def store(self,name,cfg=None):
        return Store(self.root/name,copy.deepcopy(cfg or self.cfg),self.dataset)

    def test_dry_plan_source_shards_no_writes_and_gpu_exclusion(self):
        before=set(self.root.rglob('*'))
        plan=make_plan(self.base,self.dataset,['E4','E5','E6','E7'],['1','2'],shards=2,excluded=['0'])
        validate_graph(plan['tasks'])
        generated=[t for t in plan['tasks'] if t['mode']=='generate']
        assigned=[source for task in generated for source in task['sources']]
        self.assertEqual(len(assigned),len(set(assigned)))
        self.assertEqual(set(self.root.rglob('*')),before)
        self.assertTrue(all(t['gate_required'] for t in plan['tasks'] if t['mode'].startswith('train')))
        with self.assertRaises(ValueError):
            make_plan(self.base,self.dataset,['E4'],['0'],excluded=['0'])
        with self.assertRaises(ValueError):
            make_plan(self.base,self.dataset,['E4'],['1','1'])

    def test_compatible_import_reuses_bank_but_alignment_change_rejects_e2(self):
        source=self.store('source'); ex.e0(source,self.case); ex.e1(source,self.case); ex.e2(source,self.case)
        cfg=copy.deepcopy(self.cfg); cfg['solver']['template_weight']=0
        target=self.store('target',cfg)
        result=import_completed(target,[source.root])
        self.assertTrue(result['imports'])
        self.assertTrue(target.find('E2',self.rec['id']))
        source_bank=source.artifact(source.find('E0',self.rec['id'])[0],'candidates.npz')
        target_bank=target.artifact(target.find('E0',self.rec['id'])[0],'candidates.npz')
        self.assertEqual(digest(source_bank),digest(target_bank))
        cfg['solver']['alignment_starts']=2
        changed=self.store('changed',cfg)
        skipped=import_completed(changed,[source.root])
        self.assertTrue(changed.find('E1',self.rec['id']))
        self.assertFalse(changed.find('E2',self.rec['id']))
        self.assertTrue(any(r['reason']=='stage_config_mismatch' for r in skipped['skipped']))

    def test_corrupt_import_and_changed_proposer_are_rejected(self):
        source=self.store('source'); rows=ex.e0(source,self.case)
        cfg=copy.deepcopy(self.cfg); cfg['solver']['candidates']=3
        changed=self.store('changed',cfg)
        import_completed(changed,[source.root])
        self.assertFalse(changed.jobs())
        path=source.artifact(rows[0],'camera.json'); path.write_text('{}')
        with self.assertRaises(ValueError):
            import_completed(self.store('target'),[source.root])

    def test_live_source_is_not_copied(self):
        source=self.store('source'); row=ex.e0(source,self.case)[0]
        write(source.root/'jobs'/row['job_id']/'running.lock',{'pid':1})
        target=self.store('target')
        result=import_completed(target,[source.root])
        self.assertFalse(target.jobs())
        self.assertEqual(result['skipped'][0]['reason'],'active_or_stale_lock')

    def test_global_gpu_and_exclusive_timing_leases(self):
        leases=Leases(self.root/'locks')
        one=leases.acquire({'id':'one','kind':'gpu'},'0')
        self.assertIsNotNone(one)
        self.assertIsNone(leases.acquire({'id':'two','kind':'gpu'},'0'))
        self.assertIsNone(leases.acquire({'id':'timing','kind':'exclusive'}))
        two=leases.acquire({'id':'two','kind':'gpu'},'1')
        self.assertIsNotNone(two)
        leases.release(two); leases.release(one)
        timed=leases.acquire({'id':'timing','kind':'exclusive'})
        self.assertIsNotNone(timed)
        self.assertIsNone(leases.acquire({'id':'cpu','kind':'cpu'}))
        leases.release(timed)

    def test_busy_gpu_and_dependency_cycles_rejected(self):
        with patch('subprocess.check_output',side_effect=['0, GPU-a\n1, GPU-b\n','GPU-a, 123\n']):
            with self.assertRaises(RuntimeError):
                check_gpus(['0'])
        with self.assertRaises(ValueError):
            validate_graph([{'id':'a','depends_on':['b']},{'id':'b','depends_on':['a']}])

    def test_incomplete_k_and_no_prior_have_explicit_baseline_fallback(self):
        cfg=copy.deepcopy(self.cfg); cfg['image_seeds']=[11,23,37,51]
        store=self.store('run',cfg); ex.e0(store,self.case)
        rows=stages.e5(store,self.case)
        outputs=[r for r in rows if not r['oracle']]
        self.assertEqual(len(outputs),9)
        for row in outputs:
            diag=row['output']['diagnostics']
            self.assertFalse(diag['budget_complete'])
            self.assertEqual(diag['fallback'],'B0')
            self.assertFalse(diag['template_accepted'])

    def test_pseudo_label_uses_exact_gated_teacher_and_rejects_oracle_ancestry(self):
        rec=next(c for c in self.doc['cases'] if c['split']=='train')
        case=data.load_case(self.dataset,rec,self.cfg)
        store=self.store('run'); parent=ex.e0(store,case)[0]
        poses=np.eye(4)[None].repeat(len(case['points']),0)
        diag={'template_accepted':True,'budget_complete':True,'contact_before':1.,'contact_after':.1}
        def prediction(directory):
            return ex.save_prediction(directory,case,poses,diag)
        teacher=store.run('E5',rec,'gated__K1',prediction,parents=[parent])
        labels=stages.pseudo_labels(store,[rec])
        self.assertEqual(labels[0]['teacher_job'],teacher['job_id'])
        np.testing.assert_allclose(labels[0]['poses'],poses)
        self.assertTrue(labels[0]['accepted'])
        path=store.root/'jobs'/parent['job_id']/'result.json'
        row=read(path); row['oracle']=True; write(path,row)
        with self.assertRaises(ValueError):
            stages.pseudo_labels(store,[rec])

    def test_robustness_regenerates_corrupt_inputs_and_marks_missing_inapplicable(self):
        cfg=copy.deepcopy(self.cfg)
        cfg.update(image_models=[cfg['primary_model']],input_types=[cfg['primary_input']],oracles=False)
        cfg['robustness'].update(noise=[.005],dropout=[],erosion_fraction=0,missing_piece=True,repeats=1)
        store=self.store('run',cfg)
        with patch.object(ex,'e1',wraps=ex.e1) as generation:
            rows=stages.e7(store,self.case,'generated')
        self.assertTrue(generation.call_count>=3)
        generated_ids=[call.args[1]['record']['id'] for call in generation.call_args_list]
        self.assertTrue(all(identifier!=self.rec['id'] for identifier in generated_ids))
        self.assertEqual(len(set(generated_ids)),len(generated_ids))
        self.assertTrue(any(r.get('output',{}).get('not_applicable') for r in rows))
        self.assertTrue(all(r['status']=='complete' for r in rows))

    def test_paired_failures_source_averaging_and_reject_all_gate(self):
        rows=[]
        for source,repeats in [('a',3),('b',1)]:
            for repeat in range(repeats):
                cid=f'{source}{repeat}'
                for arm,success in [('B0__refine',False),('B2__sd15_depth__A__refine',source=='a')]:
                    rows.append({'job_id':cid+arm,'case_id':cid,'source_id':source,'stage':'E4','arm':arm,
                                 'oracle':False,'status':'failed' if not success else 'complete','metrics':{'success':success}})
        paired=report.paired(rows,'E4','B2__sd15_depth__A__refine')
        self.assertEqual(paired['difference'],.5)
        self.assertEqual(paired['sources'],2)
        self.assertFalse(report.gates(rows,self.cfg)['E5_pass'])

    def test_aggregation_serializes_metrics_and_keeps_cached_rows_unique(self):
        plan=make_plan(self.base,self.dataset,['E4'],[])
        target=self.root/'aggregate'
        row={'job_id':'job','case_id':'case','source_id':'source','stage':'E4','arm':'B0__refine',
             'oracle':False,'smoke':True,'status':'complete','metrics':{'success':True},'factors':{}}
        tasks=[t for t in plan['tasks'] if t['condition']=='baseline']
        for task in tasks:
            path=target/'runs'/task['id']/'evaluation'/'dev'/'metrics.jsonl'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(row)+'\n')
        report.aggregate(target,plan)
        output=target/'analysis'/'baseline__dev'/'per_source.csv'
        self.assertTrue(output.exists())
        rows=(target/'analysis'/'baseline__dev'/'metrics.jsonl').read_text().splitlines()
        self.assertEqual(len(rows),1)
        self.assertIn('metrics',(target/'analysis'/'per_case.csv').read_text())

    def test_production_gate_blocks_training_without_launching_a_worker(self):
        plan=make_plan(self.base,self.dataset,['E6'],[])
        task=next(t for t in plan['tasks'] if t['gate_required'])
        task['depends_on']=[]; task['kind']='cpu'
        plan['tasks']=[task]; plan['smoke']=False; plan['base_config']['smoke']=False
        with patch('subprocess.Popen') as process:
            result=launch(self.root/'blocked',plan,lock_dir=self.root/'locks')
        process.assert_not_called()
        self.assertEqual(result['tasks'][task['id']],'blocked_gate')

    def test_real_scans_without_reference_do_not_get_invented_accuracy(self):
        document=read(self.dataset)
        for rec in document['cases']:
            rec['split']='real'
        write(self.dataset,document)
        (self.dataset.parent/'evaluator_only'/'index.json').unlink()
        cfg=copy.deepcopy(self.cfg); cfg['oracles']=False
        store=self.store('real',cfg)
        rec=document['cases'][0]
        def fail(directory):
            raise RuntimeError('scan processing failure')
        store.run('E4',rec,'B0__refine',fail)
        summary=report.evaluate(store,'real')
        self.assertIsNone(summary['groups'][0]['success_rate'])
        row=json.loads((store.root/'evaluation'/'real'/'metrics.jsonl').read_text().strip())
        self.assertIsNone(row['metrics'])

    def test_missing_piece_retains_anchor_and_original_piece_ids(self):
        case=copy.deepcopy(self.case)
        for key in ('points','normals','exterior','original','indices'):
            case[key].append(case[key][0].copy())
        case['centers']=np.concatenate([case['centers'],case['centers'][:1]])
        case['anchor']=1
        changed=ex.perturb_case(case,'missing',1,42)
        self.assertEqual(changed['retained_ids'],[1,2])
        self.assertEqual(changed['anchor'],0)
        np.testing.assert_array_equal(changed['points'][0],case['points'][1])

    def test_source_aliases_cannot_inflate_source_statistics(self):
        document=read(self.dataset)
        dev=[c for c in document['cases'] if c['split']=='dev']
        dev[0]['source_hash']='same_original'; dev[1]['source_hash']='same_original'
        write(self.dataset,document)
        with self.assertRaisesRegex(ValueError,'source IDs'):
            make_plan(self.base,self.dataset,['E4'],[])

    def test_capped_compute_control_is_unmatched_even_if_solver_runs_long(self):
        store=self.store('run'); ex.e0(store,self.case)
        records={stage:[{'stage':stage,'seconds':10,'output':{'model':self.cfg['primary_model'],
                    'input_type':self.cfg['primary_input'],'seed':11}}] for stage in ('E1','E2')}
        original=store.find
        def find(stage,case,arm=None):
            return records[stage] if stage in records else original(stage,case,arm)
        with patch.object(store,'find',side_effect=find),patch.object(stages.time,'monotonic',side_effect=[0,1,100]):
            result=stages.e4(store,self.case,True)[0]
        self.assertFalse(result['output']['diagnostics']['compute_matched'])
        self.assertTrue(result['output']['diagnostics']['target_exceeds_cap'])


if __name__=='__main__':
    unittest.main()
