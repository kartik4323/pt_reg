"""Diagnostic controls must preserve the production computation and stored weights."""
import copy
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from diagnostics.reassembly_v2 import probes
from diagnostics.reassembly_v2.__main__ import main, parse_args
from diagnostics.reassembly_v2.contracts import matching_contract
from diagnostics.reassembly_v2.runtime import Limits, compare, finalize, job_id, read, sha256, write
from diagnostics.reassembly_v2.variants import choose_subset, variant
from diagnostics.reassembly_v2.worker import Engine, build_jobs
from reassembly.config import load_config
from reassembly.evaluation import evaluate
from reassembly.geometry import so3_exp
from reassembly.model import ReassemblyModel
from reassembly.training import collate_samples, synthetic_batch


def fixture():
    cfg=load_config()
    cfg['model'].update(dim=16,sample_counts=[12,8,4],neighbors=4,contact_points=16)
    cfg['data'].update(points_per_fragment=20,sdf_queries=24)
    cfg['solver'].update(resolution=4,field_chunk=16)
    torch.manual_seed(73)
    model=ReassemblyModel(cfg).eval()
    batch=synthetic_batch(cfg,1,torch.device('cpu'))
    sample={k:v[0].clone() for k,v in batch.items()}
    sample.update(pattern_id='fixture',source_id='source',band='easy',cut_family='fixture')
    sample['points_view2']=sample['points']@torch.tensor(so3_exp(np.array([.2,.3,-.1])),dtype=torch.float32).T
    return cfg,model,sample


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads(); torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_comparison_preserves_counts_statuses_nulls_and_tolerances(self):
        self.assertEqual(compare({'loss':.01,'count':100000,'status':'failed','pose':None},{'loss':.010001,'count':100000,'status':'failed','pose':None}),[])
        self.assertTrue(compare(100000,100001))
        self.assertTrue(compare(None,0))
        self.assertTrue(compare('failed','ok'))
        self.assertTrue(compare(.01,.1))

    def test_selector_override_is_exact_and_gating_keeps_indices(self):
        cfg,model,sample=fixture()
        self.assertTrue(matching_contract(model,sample,cfg,torch.device('cpu'))['passed'])
        batch=collate_samples([sample],torch.device('cpu'))
        with torch.no_grad():
            enc,pairs=probes.predictions(model,batch)
            _,changed=probes.predictions(model,batch,gating='oracle',encoded=enc)
        for a,b in zip(pairs,changed):
            for key in ('source_indices','target_indices','source_xyz','target_xyz'):
                torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
        self.assertTrue(any(not torch.equal(a['weights'],b['weights']) for a,b in zip(pairs,changed)))

    def test_gradient_probes_do_not_change_original_state_or_training_mode(self):
        cfg,model,sample=fixture(); before=copy.deepcopy(model.state_dict())
        with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No optimizer steps')):
            for stage in (1,2,3):
                result=probes.gradient_probe(model,sample,cfg,torch.device('cpu'),stage)
                self.assertTrue(all(v['finite'] for v in result['objectives'].values()))
                self.assertEqual(result['optimizer_updates'],0)
        self.assertFalse(model.training)
        for name,value in model.state_dict().items(): torch.testing.assert_close(before[name],value,atol=0,rtol=0)

    def test_oracle_distributions_preserve_selected_points_and_unmatched_rows(self):
        cfg,model,sample=fixture(); sample['interface_ids'][2]=-1; sample['fracture_labels'][2]=0
        with torch.no_grad():
            batch=collate_samples([sample],torch.device('cpu')); encoded,pairs=probes.predictions(model,batch)
            altered=probes.oracle_matches(pairs,batch,cfg,encoded)
        for out,original in zip(altered,probes.numpy_matches(pairs)):
            np.testing.assert_array_equal(out['source_xyz'],original['source_xyz'])
            np.testing.assert_array_equal(out['target_xyz'],original['target_xyz'])
            if out['j']==2:
                self.assertEqual(float(out['weights'].sum()),0)
                self.assertEqual(float(out['source_matchability'].sum()),0)

    def test_contact_truth_uses_mesh_adjacency_even_when_samples_are_misleading(self):
        cfg,model,sample=fixture()
        sample['_diagnostic_metadata']={'qa':{'adjacency':[[False,True,False],[True,False,True],[False,True,False]],
            'volume_ratios':[.5,.3,.2],'interface_areas':[{'0':1},{'0':1,'1':1},{'1':1}]}}
        result,_,_=probes.measure(model,sample,cfg,torch.device('cpu'),deep=True)
        self.assertFalse(result['pairs']['0-2']['true_contact'])
        self.assertEqual(result['pairs']['0-2']['contact_truth_source'],'mesh_preparation_adjacency')
        self.assertEqual(result['fragments'][0]['source_volume_fraction'],.5)

    def test_trace_retains_planar_fits_and_rejects_collinear_and_empty(self):
        cfg=load_config(); p=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
        pair=dict(i=0,j=1,source_xyz=p,target_xyz=p@so3_exp(np.array([.2,.1,.3])).T+.3,weights=np.eye(4))
        self.assertIsNone(probes.trace_pair(pair,cfg)['first_failure'])
        pair['source_xyz']=np.array([[0.,0,0],[1,0,0],[2,0,0],[3,0,0]])
        self.assertEqual(probes.trace_pair(pair,cfg)['first_failure'],'no_geometrically_valid_candidate')
        pair['weights']*=0
        self.assertEqual(probes.trace_pair(pair,cfg)['first_failure'],'fewer_than_three_thresholded_entries')

    def test_subset_rotates_sources_and_groups_without_scores(self):
        records=[dict(source_id=str(s),pattern_id=f'{s}-{b}-{p}',band=b,pieces=p) for s in range(12) for b in ('easy','hard','intermediate') for p in (2,3)]
        indices=choose_subset(records)
        self.assertEqual(indices,choose_subset(records)); self.assertEqual(len(indices),12)
        self.assertEqual(len({records[i]['source_id'] for i in indices}),12)
        self.assertEqual(len({(records[i]['band'],records[i]['pieces']) for i in indices}),6)

    def test_permutation_preserves_pointwise_ground_truth(self):
        cfg,model,sample=fixture(); sample['original_points']=sample['points'].clone()
        class DS:
            def __getitem__(self,index): return copy.deepcopy(sample)
        changed=variant(DS(),0,'permutation',4101)
        for i in range(3):
            old=sample['points'][i].numpy(); new=changed['points'][i].numpy()
            mapping=np.linalg.norm(new[:,None]-old[None],axis=-1).argmin(-1)
            for key in ('canonical_points','interface_ids','fracture_labels','points_view2'):
                torch.testing.assert_close(changed[key][i],sample[key][i,mapping])

    def test_production_evaluation_replay_with_same_model_and_samples(self):
        cfg,model,sample=fixture()
        class DS:
            fingerprint='fixture'
            def __len__(self): return 1
            def __getitem__(self,index): return copy.deepcopy(sample)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            state=dict(purpose='pilot',step=100,cfg=cfg,training_lineage={})
            checkpoint=root/'pilot/predicted/s3/best.pt'; report_path=root/'eval/test/predicted'
            with patch('reassembly.evaluation.FractureDataset',return_value=DS()),patch('reassembly.evaluation._load_model',return_value=(model,state)):
                report=evaluate(checkpoint,root/'unused.json',report_path,cfg,torch.device('cpu'))
            ds=DS(); ds.cfg=cfg
            engine=Engine(root,root,None,torch.device('cpu'),{'val':ds,'test':ds})
            row,_=engine.evaluate(model,sample,cfg,'predicted')
            self.assertEqual(compare(report['samples'][0],row),[])
            engine.model=lambda name:(model,cfg)
            with patch('diagnostics.reassembly_v2.worker.load_checkpoint',return_value=state):
                result=engine.execute(dict(kind='replay',cohort='test',checkpoint='pilot/predicted/s3/best.pt',index=0,condition='predicted',id='fixture'))
            self.assertEqual(result['replay_differences'],[])

    def test_experiment_manifest_has_complete_coverage_and_unique_jobs(self):
        cfg=load_config()
        class DS:
            def __init__(self,manifest,split,cfg,fixed,limit):
                n=limit or {'train':418,'val':48,'test':69,'cut_holdout':11}[split]
                self.records=[dict(pattern_id=f'{split}-{i}',source_id=str(i%11),band=['easy','intermediate','hard'][i%3],pieces=2+i%2) for i in range(n)]
            def __len__(self): return len(self.records)
        with patch('diagnostics.reassembly_v2.worker.FractureDataset',DS):
            manifest,_=build_jobs(Path('unused'),cfg)
        jobs=manifest['jobs']
        self.assertEqual(len(jobs),2523)
        self.assertEqual(len({j['id'] for j in jobs}),len(jobs))
        self.assertEqual(sum(j['kind']=='replay' for j in jobs),336)
        self.assertEqual(sum(j['kind']=='baseline' for j in jobs),932)
        self.assertEqual(sum(j['kind']=='threshold' for j in jobs),360)

    def test_fixed_refinement_starts_are_identical_across_fields(self):
        cfg,model,sample=fixture(); cfg['solver'].update(min_correspondence_weight=0,min_pair_mass=0)
        with torch.no_grad():
            encoded,pairs=probes.predictions(model,collate_samples([sample],torch.device('cpu')))
            matches=probes.numpy_matches(pairs); field=probes.grid(model,encoded,cfg,sample,'predicted')
            for match in matches:
                match.update(source_xyz=sample['points'][match['i']].numpy(),target_xyz=sample['points'][match['j']].numpy(),
                    weights=np.eye(20),source_matchability=np.ones(20),target_matchability=np.ones(20))
            _,a=probes.refinement_control(matches,sample,cfg,encoded,None)
            _,b=probes.refinement_control(matches,sample,cfg,encoded,field)
        self.assertGreater(len(a),0); self.assertEqual(len(a),len(b))
        for x,y in zip(a,b):
            np.testing.assert_array_equal(x['initial_rotations'],y['initial_rotations'])
            np.testing.assert_array_equal(x['initial_translations'],y['initial_translations'])

    def test_finalizer_reports_unrun_jobs_and_detects_changed_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); output=root/'diagnosis'; output.mkdir()
            checkpoint=root/'weights.pt'; checkpoint.write_bytes(b'old')
            write(output/'inventory.json',{'checkpoints':{'fixture':{'path':str(checkpoint),'sha256':sha256(checkpoint)}}})
            job=dict(kind='measurement',checkpoint='fixture',cohort='val',index=0); job['id']=job_id(job)
            write(output/'experiments.json',{'jobs':[job]})
            checkpoint.write_bytes(b'new')
            with patch('diagnostics.reassembly_v2.runtime.render'):
                result=finalize(output,'deadline')
            self.assertEqual(result['status'],'partial_or_blocked'); self.assertEqual(result['unrun_jobs'],[job])
            self.assertEqual(result['checkpoint_changes'],['fixture'])
            self.assertTrue((output/'diagnostic_bundle.tar.gz').is_file())

    def test_cli_defaults_to_unlimited_and_accepts_optional_long_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(parse_args(['--managed-root',directory]).hours)
            self.assertEqual(parse_args(['--managed-root',directory,'--hours','24']).hours,24)
            for value in ('0','-1','nan','inf'):
                with self.assertRaises(SystemExit): parse_args(['--managed-root',directory,'--hours',value])
            with self.assertRaises(SystemExit): parse_args(['--managed-root',directory,'--output',str(Path(directory)/'bottles498/logs')])

    def test_unlimited_runtime_still_checks_storage_and_optional_deadline_expires(self):
        with tempfile.TemporaryDirectory() as directory:
            limit=Limits(Path(directory),Path(directory)/'output',None)
            with patch.object(limit.guard,'check') as guard,patch('diagnostics.reassembly_v2.runtime.time.time',return_value=10**12):
                limit.check()
                guard.assert_called_once()
                limit.deadline=1
                with self.assertRaises(TimeoutError): limit.check()
            with patch.object(limit.guard,'check',side_effect=RuntimeError('storage limit')):
                limit.deadline=None
                with self.assertRaisesRegex(RuntimeError,'storage limit'): limit.check()

    def test_supervisor_passes_deadline_only_when_requested(self):
        for optional in ([],['--hours','12']):
            with self.subTest(optional=optional),tempfile.TemporaryDirectory() as directory:
                output=Path(directory)/'diagnostics'/'fixture'
                with patch.object(Limits,'check'),patch('diagnostics.reassembly_v2.__main__.subprocess.Popen') as launch, \
                     patch('diagnostics.reassembly_v2.__main__.time.sleep'), \
                     patch('diagnostics.reassembly_v2.__main__.finalize',return_value={'status':'complete','completed_jobs':1,'planned_jobs':1}):
                    launch.return_value.poll.side_effect=[None,0]
                    launch.return_value.returncode=0
                    self.assertEqual(main(['--managed-root',directory,'--output',str(output),'--device','cpu']+optional),0)
                command=launch.call_args.args[0]
                invocation=read(output/'invocation.json')
                self.assertEqual('--deadline' in command,bool(optional))
                self.assertEqual(invocation['deadline_epoch'] is None,not optional)


if __name__=='__main__': unittest.main()
