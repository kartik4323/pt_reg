import copy
import json
from pathlib import Path
import tempfile
import unittest
import types
import os
import sys
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from generative_assembly import config,data,geometry as g,render,experiments as ex
from generative_assembly.quality import score_alpha,save_matte,normalize_foreground,shape_metrics
from generative_assembly.resources import gpu_snapshot,gpu_admission,worker_budget,ResourceUnavailable,BudgetExhausted,atomic_lock
from generative_assembly.storage import Store,write,read,digest
from generative_assembly.study_cycle import build_config,collect_zip,estimate_phase,stress_dataset,run_phase
from generative_assembly import backends
from generative_assembly.promotion import assess


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.cfg=config.load(Path(config.__file__).parent/'configs'/'smoke.json')
        self.valid=np.zeros((64,64),bool); self.valid[25:35,25:35]=True

    def score(self,alpha): return score_alpha(alpha,self.valid,self.valid,self.cfg['matte'])

    def test_matte_grey_background_not_growth(self):
        alpha=self.valid.astype(np.uint8)*255
        score=self.score(alpha)
        self.assertTrue(score['valid_foreground']); self.assertEqual(score['growth_relative_to_input'],0)
        self.assertFalse(score['camera_preservation_verified'])

    def test_matte_pale_object_pixels_survive(self):
        image=Image.new('RGB',(64,64),(245,245,245))
        save_matte(image,self.valid.astype(np.uint8)*255,self.root,dict(source='fixture'))
        with Image.open(self.root/'image_white.png') as saved:
            self.assertEqual(saved.getpixel((30,30)),(245,245,245))
            self.assertEqual(saved.getpixel((0,0)),(255,255,255))

    def test_matte_empty_and_all_foreground_rejected(self):
        self.assertFalse(self.score(np.zeros((64,64),np.uint8))['valid_foreground'])
        self.assertFalse(self.score(np.full((64,64),255,np.uint8))['valid_foreground'])

    def test_matte_cannot_claim_retention_without_protected_support(self):
        result=score_alpha(self.valid.astype(np.uint8)*255,self.valid,np.zeros_like(self.valid),self.cfg['matte'])
        self.assertFalse(result['valid_foreground']); self.assertIn('protected_support_unavailable',result['rejection_reasons'])

    def test_matte_disconnected_and_loss_rejected(self):
        a=self.valid.astype(np.uint8)*255; a[40:50,40:50]=255
        self.assertIn('multiple_or_disconnected_objects',self.score(a)['rejection_reasons'])
        a[:]=0; a[26:34,26:34]=255
        self.assertIn('protected_region_loss',self.score(a)['rejection_reasons'])

    def test_rgba_normalization_inclusive_bbox_and_affine(self):
        a=np.zeros((64,64,4),np.uint8); a[10:30,20:30]=[150,150,150,255]
        src=self.root/'source.png'; Image.fromarray(a).save(src)
        transform=normalize_foreground(src,self.root/'normalized.png')
        self.assertEqual(transform['crop_box'],[20,10,30,30])
        with Image.open(self.root/'normalized.png') as saved: self.assertEqual(saved.mode,'RGBA')
        affine=np.array(transform['affine_source_to_output'])
        self.assertTrue(np.isfinite(affine).all()); self.assertEqual(affine.shape,(3,3))

    def test_rgba_rejects_rgb_and_empty(self):
        src=self.root/'source.png'; Image.new('RGB',(64,64),'white').save(src)
        with self.assertRaisesRegex(ValueError,'RGBA'): normalize_foreground(src,self.root/'out.png')
        Image.new('RGBA',(64,64),(0,0,0,0)).save(src)
        with self.assertRaisesRegex(ValueError,'Empty'): normalize_foreground(src,self.root/'out.png')

    def test_surfel_determinism_hole_and_ids(self):
        xy=np.array([(x,y) for x in np.linspace(-.8,.8,20) for y in np.linspace(-.4,.4,12) if abs(x)>.25])
        p=np.column_stack((xy,np.zeros(len(xy)))); n=np.tile([0,0,1.],(len(p),1))
        cam=dict(pixels=64,extent=1,basis=np.eye(3).tolist())
        first=render.render_surfel(p,n,np.ones(len(p)),cam); second=render.render_surfel(p,n,np.ones(len(p)),cam)
        np.testing.assert_array_equal(first['rgb'],second['rgb'])
        self.assertFalse(first['valid'][32,32]); self.assertLess(first['ids'].max(),len(p))
        self.assertEqual(first['renderer_actual'],'surfel')

    def test_surface_no_silent_fallback(self):
        with patch.dict('sys.modules',{'open3d':None}):
            with self.assertRaisesRegex(RuntimeError,'Explicit Poisson'): render.render_surface(np.zeros((20,3)),None,None,dict())

    def test_camera_center_and_up_rotation(self):
        p=np.random.default_rng(0).normal(size=(80,3))*.1+[2,3,4]
        cam=render.frame_observed(render.camera_for(p- [2,3,4],64,1),p,.35)
        r=render.render(p,None,None,cam,0); ys,xs=np.nonzero(r['valid']); B=np.array(cam['basis'])
        recovered=np.column_stack(((xs/63-.5)*2*cam['extent'],(.5-ys/63)*2*cam['extent'],r['depth'][ys,xs]))@B.T+cam['center']
        self.assertLess(np.max(np.linalg.norm(recovered-p[r['ids'][ys,xs]],axis=1)),.02)
        cameras=render.camera_for(p-[2,3,4],64,1,n_views=4)
        for c in cameras: np.testing.assert_allclose(np.array(c['basis'])[:,1],np.array(cameras[0]['basis'])[:,1],atol=1e-12)

    def test_dense_render_sampling_public_only(self):
        case=dict(original=[np.random.default_rng(0).normal(size=(1000,3))],centers=np.zeros((1,3)),scale=1,record=dict(id='fixture'))
        cfg=dict(seed=42,render_points=200)
        a=data.rendering_evidence(case,cfg); b=data.rendering_evidence(case,cfg)
        np.testing.assert_array_equal(a['indices'][0],b['indices'][0]); self.assertEqual(len(a['points'][0]),200)

    def test_source_balancing_not_variants(self):
        records=[dict(id=f'{s}-{i}',source_id=s,split='dev') for s in ['a','b','c'] for i in range(5)]
        selected=data.balanced_cases(records,'dev',3)
        self.assertEqual({r['source_id'] for r in selected},{'a','b','c'})

    def test_case_dependency_failure_recorded_without_erasing_cohort(self):
        data.demo(self.root/'data',6); ds=self.root/'data'/'dataset.json'; rec=data.inventory(ds)['cases'][2]
        case=data.load_case(ds,rec,self.cfg); store=Store(self.root/'run',self.cfg,ds)
        rows=ex.run_stage(store,'E2',case,True)
        self.assertEqual(rows[0]['status'],'failed'); self.assertIn('E0 prepare',rows[0]['error'])
        self.assertEqual(rows[0]['case_id'],rec['id'])

    def test_frozen_cycle_source_cohorts_and_refreeze_guard(self):
        data.demo(self.root/'data',34); ds=self.root/'data'/'dataset.json'; doc=read(ds)
        for i,c in enumerate(doc['cases']): c['split']='dev' if i<10 else 'test' if i<30 else 'train'
        write(ds,doc); data.inventory(ds)
        base=self.root/'base.json'; write(base,self.cfg)
        matte=self.root/'matte.json'; write(matte,dict(model='isnet-general-use',weights='fixture',sha256='a'*64))
        a=types.SimpleNamespace(root=self.root/'cycle',dataset=ds,base=base,matte_lock=matte,phase='validate',
            images_python=sys.executable,mesh_python=sys.executable,matte_python=sys.executable,
            representation=None,finalists=['models_sd15'],stages=None,profiles=None,retry_failed=False)
        def mock_execute(args,name,cfg,split,records,stages):
            write(args.root/'configs'/f'{name}-locked.json',cfg)
            store=Store(args.root/name,cfg,ds)
            write(store.root/'experiment.json',dict(dataset=str(ds),profile=name,split=split,case_ids=[r['id'] for r in records]))
            return store.root
        with patch('generative_assembly.study_cycle.execute',side_effect=mock_execute): run_phase(a)
        self.assertEqual(len(read(a.root/'validation_sd15'/'experiment.json')['case_ids']),10)
        a.phase='freeze'; run_phase(a)
        a.phase='test'
        with patch.object(ex,'run_stage',return_value=[]): run_phase(a)
        self.assertEqual(len(read(a.root/'test_started.json')['case_ids']),20)
        a.phase='freeze'
        with self.assertRaisesRegex(ValueError,'Test cohort exposed'): run_phase(a)

    def test_gpu_visible_device_mapping(self):
        outputs=['0, GPU-a, A, 32768, 26000\n1, GPU-b, B, 32768, 28000','GPU-b, 123, test, 1000']
        with patch('subprocess.check_output',side_effect=outputs),patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'1,0'}):
            info=gpu_snapshot('cuda:0')
        self.assertEqual(info['uuid'],'GPU-b'); self.assertEqual(info['index'],1)

    def test_gpu_busy_rejected_and_lock_released(self):
        req=dict(kind='reconstruction',smoke=False,config={},resources=dict(gpu_preflight=True,wait_seconds=0,lock_dir=str(self.root/'locks')))
        with patch('generative_assembly.resources.gpu_snapshot',return_value=dict(uuid='GPU-test',free_gib=10)):
            with self.assertRaises(ResourceUnavailable):
                with gpu_admission(req,self.root): pass
        self.assertFalse((self.root/'locks'/'GPU-test.lock').exists())

    def test_gpu_admission_and_exception_cleanup(self):
        req=dict(kind='reconstruction',smoke=False,config={},resources=dict(gpu_preflight=True,wait_seconds=0,lock_dir=str(self.root/'locks')))
        with patch('generative_assembly.resources.gpu_snapshot',return_value=dict(uuid='GPU-test',free_gib=28)):
            with gpu_admission(req,self.root): self.assertTrue((self.root/'locks'/'GPU-test.lock').exists())
        self.assertFalse((self.root/'locks'/'GPU-test.lock').exists())

    def test_budget_persistent_and_reservation_cleanup(self):
        path=self.root/'budget.json'
        req=dict(kind='image',resources=dict(budget_path=str(path)))
        with worker_budget(req,self.root,50) as allowed: self.assertEqual(allowed,50)
        ledger=read(path); self.assertFalse(ledger['reservations']); self.assertGreaterEqual(ledger['used']['screen'],0)
        ledger['used']['screen']=ledger['limits']['screen']; write(path,ledger)
        with self.assertRaises(BudgetExhausted):
            with worker_budget(req,self.root,50): pass

    def test_cooperative_lock_does_not_delete_owner(self):
        path=self.root/'owned.lock'
        with atomic_lock(path):
            with self.assertRaises(ResourceUnavailable):
                with atomic_lock(path): pass
            self.assertTrue(path.exists())
        self.assertFalse(path.exists())

    def test_continuous_similarity_export_and_holdout(self):
        p=np.random.default_rng(42).normal(size=(600,3))*[.2,.3,1]
        center=p.mean(0); radius=np.linalg.norm(p-center,axis=1).max()
        target=(p-center)/(2*radius)*2.3+[.1,.2,.3]
        case=dict(points=[target],exterior=[np.ones(len(target))],anchor=0)
        cfg=dict(alignment_starts=1,alignment_iterations=30,scales=[2.],continuous_scale=True,scale_bounds=[.5,6.])
        aligned,fit=g.fit_template(p,case,cfg,42)
        np.testing.assert_allclose(g.apply(p,np.array(fit['similarity_transform'])),aligned,atol=1e-10)
        self.assertFalse(set(fit['fit_indices'])&set(fit['heldout_indices']))
        self.assertTrue(.5<=fit['scale']<=6); self.assertLess(fit['heldout_error'],.05)

    def test_missing_shape_raw_fragment_not_completion(self):
        complete=np.array([[0,0,0],[1,0,0],[2,0,0]],float); observed=complete[:1]
        raw=shape_metrics(observed,complete,observed); full=shape_metrics(complete,complete,observed)
        self.assertGreater(raw['missing_surface_distance'],full['missing_surface_distance'])
        self.assertEqual(raw['missing_surface_recall'],0); self.assertEqual(full['missing_surface_fscore'],1)

    def test_full_object_oracle_control_cannot_enter_public_prior_bank(self):
        data.demo(self.root/'data',6); ds=self.root/'data'/'dataset.json'; rec=data.inventory(ds)['cases'][2]
        cfg=copy.deepcopy(self.cfg); cfg.update(render_mode='surfel',oracle_full_frame_control=True)
        cfg['matte']['enabled']=True
        case=data.load_case(ds,rec,cfg); store=Store(self.root/'run',cfg,ds)
        ex.e0(store,case); ex.e1(store,case); rows=ex.e2(store,case)
        oracle=next(r for r in rows if r['arm']=='true_image_full_frame')
        self.assertTrue(oracle['oracle']); self.assertEqual(oracle['status'],'complete')
        chosen=ex.select_priors(store,case)
        self.assertFalse(any(oracle['job_id'] in v['job_ids'] for v in chosen.values()))

    def test_retry_archives_stale_outputs(self):
        data.demo(self.root/'data',6); ds=self.root/'data'/'dataset.json'; rec=data.inventory(ds)['cases'][0]
        store=Store(self.root/'run',self.cfg,ds)
        def fail(out):
            (out/'mesh.obj').write_text('stale'); raise RuntimeError('fixture failure')
        store.run('X',rec,'retry',fail)
        retried=Store(store.root,self.cfg,ds,retry=True)
        def succeed(out):
            self.assertFalse((out/'mesh.obj').exists()); return dict(ok=True)
        row=retried.run('X',rec,'retry',succeed)
        self.assertEqual(row['status'],'complete'); self.assertTrue(list((store.root/'jobs'/row['job_id']/'attempts').rglob('mesh.obj')))

    def test_v3_configs_native_resolution_and_scoped_settings(self):
        base=self.root/'base.json'; write(base,self.cfg)
        cfg=build_config(base,'models_sdxl',dict(model='isnet-general-use',weights='fixture',sha256='a'*64),'images','mesh','matte',self.root/'budget.json','screen')
        self.assertEqual(cfg['images']['resolution'],1024); self.assertEqual(cfg['framing_fill'],.35)
        self.assertEqual(cfg['reconstruction']['input_mode'],'rgba'); self.assertEqual(cfg['primary_input'],'F')
        self.assertTrue(cfg['matte']['enabled']); self.assertTrue(cfg['solver']['continuous_scale'])

    def test_v3_smoke_no_reference_at_inference_and_safe_abstention(self):
        data.demo(self.root/'data',6); ds=self.root/'data'/'dataset.json'; rec=data.inventory(ds)['cases'][2]
        cfg=copy.deepcopy(self.cfg); cfg.update(render_mode='surfel',framing_fill=.35,oracles=False)
        cfg['matte']['enabled']=True; cfg['solver']['continuous_scale']=True
        case=data.load_case(ds,rec,cfg); store=Store(self.root/'run',cfg,ds)
        with patch.object(data,'reference',side_effect=AssertionError('Reference access forbidden')):
            ex.e0(store,case); e1=ex.e1(store,case); ex.e2(store,case); rows=ex.e4(store,case)
        self.assertTrue(all(r['status']=='complete' for r in e1))
        deploy=next(r for r in rows if r['arm']=='B2__deploy')
        self.assertEqual(deploy['status'],'complete'); self.assertTrue(deploy['output']['diagnostics']['abstained'])
        self.assertFalse(deploy['output']['diagnostics']['uses_ground_truth'])
        with np.load(store.artifact(deploy,'poses.npz')) as f: np.testing.assert_array_equal(f['normalized'][case['anchor']],np.eye(4))

    def test_collector_does_not_include_weights_or_recursive_zip(self):
        root=self.root/'run'; root.mkdir(); (root/'study.json').write_text('{}'); (root/'weight.ckpt').write_text('fixture')
        collect_zip(root,self.root/'results.zip')
        import zipfile
        with zipfile.ZipFile(self.root/'results.zip') as z: self.assertEqual(z.namelist(),['study.json'])
        with self.assertRaises(ValueError): collect_zip(root,root/'nested.zip')

    def test_preprocessing_verification_refuses_changed_upstream(self):
        src=self.root/'run.py'
        src.write_text('if not args.no_rembg:\n    image = resize_foreground(image, 0.85)\n')
        self.assertTrue(backends.verify_preprocessing(src)['no_rembg_disables_upstream_resize'])
        src.write_text('image = resize_foreground(image, 0.85)\n')
        with self.assertRaisesRegex(ValueError,'Unsupported'): backends.verify_preprocessing(src)

    def test_native_image_adapters_and_instruction_interface(self):
        captured={}
        class Pipeline:
            @classmethod
            def from_pretrained(cls,repo,**kwargs): captured['load']=kwargs; return cls()
            def to(self,device): captured['device']=device
            def __call__(self,**kwargs):
                captured['call']=kwargs
                return types.SimpleNamespace(images=[Image.new('RGB',(kwargs['width'],kwargs['height']),'grey')])
        module=types.SimpleNamespace(**{k:Pipeline for k in ('ControlNetModel','StableDiffusionControlNetInpaintPipeline',
            'StableDiffusionInpaintPipeline','AutoPipelineForInpainting','QwenImageEditPlusPipeline')})
        image=self.root/'input.png'; Image.new('RGB',(64,64),'grey').save(image)
        mask=self.root/'mask.png'; Image.new('L',(64,64),255).save(mask)
        for model in ('sd15','sdxl','qwen'):
            cfg=copy.deepcopy(self.cfg['images']); cfg.update(device='cpu',cpu_offload=False,resolution=512 if model=='sd15' else 1024)
            cfg['revisions']={cfg[model]:'a'*40}
            with patch.dict('sys.modules',{'diffusers':module}),patch.object(backends.importlib.metadata,'version',return_value='mock'):
                result=backends.run_image(dict(config=cfg,model=model,smoke=False,image=str(image),mask=str(mask),
                    control=str(image),prompt='Complete bottle',seed=11),self.root)
            self.assertEqual(captured['call']['width'],cfg['resolution'])
            self.assertEqual(result['generation_resolution'],cfg['resolution'])
            if model=='qwen':
                self.assertNotIn('mask_image',captured['call']); self.assertIsInstance(captured['call']['image'],list)
                self.assertEqual(captured['call']['true_cfg_scale'],cfg['qwen_cfg'])
            else: self.assertEqual(captured['call']['mask_image'].size,(cfg['resolution'],cfg['resolution']))

    def test_model_locks_use_author_pipeline_not_community_mirror(self):
        calls=[]
        class Api:
            def model_info(self, repo, revision):
                if repo == 'diffusers/community-pipelines-mirror':
                    raise RuntimeError('Dataset is not a model repository')
                calls.append((repo,revision))
                return types.SimpleNamespace(sha='a'*40)
        cfg=copy.deepcopy(self.cfg)
        cfg['reconstruction']['backend']='smoke'
        cfg['reconstruction']['revisions']={'sudo-ai/zero123plus-pipeline':'b'*40}
        with patch.dict('sys.modules',{'huggingface_hub':types.SimpleNamespace(HfApi=Api)}):
            result=backends.lock_models(cfg)
        self.assertIn(('sudo-ai/zero123plus-pipeline','b'*40),calls)
        self.assertEqual(set(result['reconstruction']['revisions']),
                         {'sudo-ai/zero123plus-v1.2','TencentARC/InstantMesh','sudo-ai/zero123plus-pipeline'})

    def test_reconstruction_rgba_and_pinned_local_pipeline_mock(self):
        repo=self.root/'repo'; repo.mkdir(); script=repo/'run.py'
        script.write_text('if not args.no_rembg:\n    image = resize_foreground(image, 0.85)\n')
        rgba=self.root/'input.png'; arr=np.zeros((64,64,4),np.uint8); arr[10:30,20:30]=[150,150,150,255]; Image.fromarray(arr).save(rgba)
        cfg=copy.deepcopy(self.cfg['reconstruction']); cfg.update(repo=str(repo),commit='a'*40,input_mode='rgba',
            local_hashes={'run.py':digest(script)},revisions={'sudo-ai/zero123plus-v1.2':'b'*40,'sudo-ai/zero123plus-pipeline':'c'*40})
        captured={}
        class Pipeline:
            @staticmethod
            def from_pretrained(model,**kw): captured.update(kw)
        module=types.SimpleNamespace(DiffusionPipeline=Pipeline)
        downloads=[]
        def download(*args,**kwargs):
            downloads.append(kwargs)
            return str(script)
        hub=types.SimpleNamespace(hf_hub_download=download)
        def run(*args,**kwargs):
            with self.assertRaisesRegex(ValueError,'Unsupported InstantMesh custom pipeline'):
                module.DiffusionPipeline.from_pretrained('sudo-ai/zero123plus-v1.2',custom_pipeline='unknown')
            module.DiffusionPipeline.from_pretrained('sudo-ai/zero123plus-v1.2',custom_pipeline='zero123plus')
            self.assertIn('--no_rembg',sys.argv)
            with Image.open(sys.argv[2]) as image: self.assertEqual(image.mode,'RGBA'); self.assertEqual(image.size,(512,512))
            mesh=self.root/'upstream'/'mock'/'meshes'; mesh.mkdir(parents=True); (mesh/'image.obj').write_text('fixture')
        self.addCleanup(os.chdir,os.getcwd()); self.addCleanup(setattr,sys,'argv',sys.argv.copy()); self.addCleanup(setattr,sys,'path',sys.path.copy())
        with patch.dict('sys.modules',{'diffusers':module,'diffusers.utils':types.SimpleNamespace(HF_MODULES_CACHE=str(self.root/'cache')),
             'huggingface_hub':hub}),patch.object(backends.subprocess,'check_output',side_effect=['a'*40,'']),\
             patch.object(backends.runpy,'run_path',side_effect=run),patch.object(backends.importlib.metadata,'version',return_value='mock'):
            result=backends.run_reconstruction(dict(config=cfg,image=str(rgba),smoke=False),self.root)
        self.assertNotIn('custom_revision',captured)
        self.assertEqual(captured['custom_pipeline'],str(script))
        self.assertEqual(captured['revision'],'b'*40)
        self.assertEqual(downloads,[dict(repo_id='sudo-ai/zero123plus-pipeline',repo_type='model',
                                         filename='pipeline.py',revision='c'*40)])
        self.assertEqual(result['downloaded'][0]['sha256'],digest(script))
        self.assertEqual(result['downloaded'][0]['revision'],'c'*40)
        self.assertTrue(result['foreground_resize'])
        self.assertEqual(result['local_hashes']['run.py'],digest(script))

    def test_worker_failure_keeps_diagnostics_and_peak_fields(self):
        request=self.root/'request.json'; write(request,dict(kind='image',smoke=False))
        with patch.object(backends,'run_image',side_effect=RuntimeError('CUDA out of memory')):
            with self.assertRaisesRegex(RuntimeError,'CUDA'): backends.worker(request)
        metadata=read(self.root/'backend.json'); self.assertTrue(metadata['failed'])
        self.assertIn('peak_cuda_allocated_bytes',metadata)

    def test_phase_budget_refuses_batch_before_workers(self):
        write(self.root/'budget.json',dict(limits=dict(screen=100,validation=200),used={},reservations={},
            history=[dict(kind='image',model=self.cfg['primary_model'],seconds=90)]))
        with self.assertRaises(BudgetExhausted): estimate_phase(self.root,[('fixture',self.cfg,[dict(id='x')],['E1'])],'screen')
        self.assertFalse(read(self.root/'phase_admission.json')['admitted'])

    def test_stress_perturbs_dense_public_input_deterministically(self):
        data.demo(self.root/'data',6); ds=self.root/'data'/'dataset.json'; rec=data.inventory(ds)['cases'][2]
        for condition in ('noise','dropout'):
            target=stress_dataset(ds,[rec],self.root/condition,condition,self.cfg)
            changed=data.load_case(target,data.inventory(target)['cases'][0],self.cfg)
            original=data.load_case(ds,rec,self.cfg)
            if condition=='dropout': self.assertLess(len(changed['original'][0]),len(original['original'][0]))
            else: self.assertFalse(np.array_equal(changed['original'][0],original['original'][0]))
            self.assertIsNotNone(data.reference(target,changed))
            self.assertEqual(target,stress_dataset(ds,[rec],self.root/condition,condition,self.cfg))

    def test_promotion_counts_missing_cases_and_smoke(self):
        run=self.root/'run'; run.mkdir(); write(run/'study.json',dict(identity=dict(config=self.cfg)))
        ds=self.root/'ds.json'; cases=[dict(id=str(i),source_id='same',split='test') for i in range(20)]; write(ds,dict(cases=cases))
        result=assess(run,ds,'test',[c['id'] for c in cases])
        self.assertEqual(result['eligible_cases'],20); self.assertEqual(result['sources'],1)
        self.assertFalse(result['approved']); self.assertFalse(result['checks']['source_count']); self.assertFalse(result['checks']['non_smoke'])

    def test_selection_duplicates_shortfall_and_original_coordinates(self):
        import trimesh
        data.demo(self.root/'data',6); ds=self.root/'data'/'dataset.json'; rec=data.inventory(ds)['cases'][2]
        cfg=copy.deepcopy(self.cfg); cfg['matte']['enabled']=True
        case=data.load_case(ds,rec,cfg); store=Store(self.root/'run',cfg,ds)
        base,_=data.sample_mesh(trimesh.creation.box(),200,np.random.default_rng(0))
        for seed,offset in ((11,0),(23,0),(37,.15)):
            image=store.run('E1',rec,f'image_{seed}',lambda d: dict(model=cfg['primary_model'],input_type='F',
                selection=dict(valid_foreground=True,growth_relative_to_input=1)))
            def shape(out,offset=offset,seed=seed,image=image):
                np.savez_compressed(out/'shape.npz',aligned=base+offset)
                mesh=trimesh.creation.box(); mesh.apply_translation([offset]*3); mesh.export(out/'mesh.obj')
                return dict(model=cfg['primary_model'],input_type='F',seed=seed,image_job=image['job_id'],
                    alignment=dict(heldout_error=.01,similarity_transform=np.eye(4).tolist()))
            store.run('E2',rec,f'shape_{seed}',shape,parents=[image])
        selected=ex.select_priors(store,case)[f'{cfg["primary_model"]}__F']
        self.assertEqual(selected['available'],2); self.assertIsNotNone(selected['shortfall_reason'])
        self.assertIn('duplicate_geometry',[r['reason'] for r in selected['filtered_candidates']])
        with np.load(store.root/selected['point_cloud_bundle']) as bundle:
            np.testing.assert_allclose(bundle['original_anchor_frame_1'],bundle['normalized_1']*case['scale']+case['centers'][case['anchor']])

    def test_promotion_pass_requires_paired_sources_review_and_public_hypotheses(self):
        cfg=copy.deepcopy(self.cfg); cfg['smoke']=False
        run=self.root/'run'; run.mkdir(); write(run/'study.json',dict(identity=dict(config=cfg)))
        cases=[dict(id=str(i),source_id=f'source{i}',split='test') for i in range(20)]
        ds=self.root/'ds.json'; write(ds,dict(cases=cases)); rows=[]; reviews={}
        for i,c in enumerate(cases):
            ids=[f'{i}-{j}' for j in range(3)]
            write(run/'priors'/f'{i}.json',{f'{cfg["primary_model"]}__{cfg["primary_input"]}':dict(job_ids=ids)})
            for jid in ids:
                rows.append(dict(job_id=jid,case_id=str(i),stage='E2',arm='generated',oracle=False,smoke=False,
                    metrics=dict(missing_surface_available=True,missing_surface_distance=.07,missing_surface_fscore=.8)))
                reviews[jid]=dict(bottle_identity=True,full_outline=True,same_view=True,surface_quality=True)
            rows.append(dict(job_id=f'raw{i}',case_id=str(i),stage='E2',arm=f'raw__{cfg["primary_input"]}',
                metrics=dict(missing_surface_available=True,missing_surface_distance=.1,missing_surface_fscore=.8)))
            for arm,value in (('B0__refine',i>=6),('B2__deploy',True)):
                rows.append(dict(case_id=str(i),stage='E4',arm=arm,metrics=dict(success=value)))
        target=run/'evaluation'/'test'/'metrics.jsonl'; target.parent.mkdir(parents=True)
        target.write_text('\n'.join(json.dumps(r) for r in rows)); write(run/'review.json',reviews)
        self.assertTrue(assess(run,ds,'test',[c['id'] for c in cases])['approved'])
        rows[0]['oracle']=True; target.write_text('\n'.join(json.dumps(r) for r in rows))
        result=assess(run,ds,'test',[c['id'] for c in cases]); self.assertFalse(result['approved'])
        self.assertFalse(result['checks']['public_hypotheses_only'])


if __name__=='__main__': unittest.main()
