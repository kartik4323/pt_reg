import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from generative_assembly import config, data, experiments as ex, render, geometry as g
from generative_assembly.storage import Store, write
from generative_assembly.imagination_report import collect, gallery
from generative_assembly.imagination_study import prepare


class ImaginationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = config.load(Path(config.__file__).parent / 'configs' / 'smoke.json')
        self.cfg.update(render_mode='splat', canvas_fill_target=0., n_render_views=2)
        data.demo(self.root / 'data', 6)
        self.ds = self.root / 'data' / 'dataset.json'
        self.case = data.load_case(self.ds, data.inventory(self.ds)['cases'][2], self.cfg)

    def test_threshold_migration_and_nested_precedence(self):
        path = self.root / 'config.json'
        write(path, dict(bg_obj_threshold=0))
        self.assertEqual(config.load(path)['images']['bg_obj_threshold'], 0)
        write(path, dict(bg_obj_threshold=140, images=dict(bg_obj_threshold=0)))
        self.assertEqual(config.load(path)['images']['bg_obj_threshold'], 0)

    def test_mask_polarity_and_crop_preserves_depth_and_ids(self):
        camera = render.camera_for(self.case['points'][0], 32, 1.5)
        rendered = render.render(self.case['points'][0], self.case['normals'][0], self.case['exterior'][0], camera)
        mask = render.inverted_mask(rendered)
        self.assertTrue((mask[rendered['valid']] == 0).all())
        cropped = render.tight_crop(rendered)
        self.assertTrue(set(np.unique(cropped['ids'])).issubset(set(np.unique(rendered['ids']))))
        original_depth = np.unique(rendered['depth']).astype(np.float32)
        self.assertTrue(set(np.unique(cropped['depth'])).issubset(set(original_depth)))
        same = render.crop_render(rendered, cropped['crop_box'])
        np.testing.assert_array_equal(same['valid'], cropped['valid'])

    def test_unchanged_fragment_gets_no_completion_credit(self):
        valid = np.zeros((32, 32), bool); valid[10:20, 10:20] = True
        image = np.full((32, 32, 3), 255, np.uint8); image[valid] = 100
        path = self.root / 'image.png'; Image.fromarray(image).save(path)
        score = render.image_score(path, dict(valid=valid, protected=valid))
        self.assertEqual(score['growth_relative_to_input'], 0)
        self.assertGreater(score['selection_score'], 0)

    def test_generator_raw_output_survives_legacy_cleanup(self):
        import torch  # Load before the sys.modules patch so restoring it cannot unload torch.
        from generative_assembly.backends import run_image
        class Pipeline:
            @classmethod
            def from_pretrained(cls, *args, **kwargs): return cls()
            def to(self, device): return self
            def __call__(self, **kwargs):
                return types.SimpleNamespace(images=[Image.new('RGB', (32, 32), (180, 180, 180))])
        fake = types.SimpleNamespace(ControlNetModel=Pipeline,
            StableDiffusionControlNetInpaintPipeline=Pipeline,
            StableDiffusionInpaintPipeline=Pipeline, AutoPipelineForInpainting=Pipeline)
        image = self.root / 'input.png'; mask = self.root / 'mask.png'
        Image.new('RGB', (32, 32), 'white').save(image)
        Image.new('L', (32, 32), 255).save(mask)
        cfg = copy.deepcopy(self.cfg['images'])
        cfg.update(device='cpu', cpu_offload=False, dtype='float32', bg_obj_threshold=140)
        cfg['revisions'] = {cfg['sd15']: 'a' * 40}
        with patch.dict('sys.modules', {'diffusers': fake}), patch('importlib.metadata.version', return_value='fixture'):
            run_image(dict(config=cfg, model='sd15', smoke=False, image=str(image), mask=str(mask), seed=11, prompt='fixture'), self.root)
        self.assertEqual(Image.open(self.root / 'image_raw.png').getpixel((16, 16)), (180, 180, 180))
        self.assertEqual(Image.open(self.root / 'image.png').getpixel((16, 16)), (255, 255, 255))

    def test_notebook_cells_execute_with_smoke_report(self):
        import contextlib
        import io
        import json
        import os
        store = Store(self.root / 'run', self.cfg, self.ds)
        ex.e0(store, self.case); ex.e1(store, self.case); ex.e2(store, self.case)
        notebook = Path(__file__).resolve().parents[2] / 'imagination_state_analysis.ipynb'
        namespace = {}
        # The chosen PROFILE is unavailable in this fixture, so no plot/browser is opened.
        with patch.dict(os.environ, {'RUN_GROUP': str(store.root), 'DATASET': str(self.ds)}), contextlib.redirect_stdout(io.StringIO()):
            for cell in json.loads(notebook.read_text(encoding='utf-8'))['cells']:
                if cell['cell_type'] == 'code':
                    exec(compile(''.join(cell['source']), str(notebook), 'exec'), namespace)
        self.assertTrue((store.root / 'analysis' / 'metrics.csv').exists())
        self.assertGreater(len(namespace['df']), 0)

    def test_shape_check_allows_slender_objects_but_rejects_sheets(self):
        rng = np.random.default_rng(42)
        bottle = rng.normal(size=(2000, 3)) * [0.1, 0.1, 2.]
        sheet = rng.normal(size=(2000, 3)) * [1., 1., .001]
        self.assertTrue(g.check_template_shape(bottle, {})[0])
        self.assertFalse(g.check_template_shape(sheet, {})[0])

    def test_rejection_report_and_multiview_lookup(self):
        store = Store(self.root / 'run', self.cfg, self.ds)
        ex.e0(store, self.case); ex.e1(store, self.case)
        with patch.object(g, 'fit_template', return_value=(None, dict(rejected='fixture_sheet', uses_ground_truth_alignment=False))):
            results = ex.e2(store, self.case)
        self.assertTrue(all(r['output']['template_rejected'] for r in results))
        self.assertEqual(ex.templates(store, self.case, 'raw', 'F'), [])
        report = collect(store.root, self.ds)
        rejected = [r for r in report if r['stage'] == 'E2']
        self.assertTrue(rejected)
        self.assertTrue(all('chamfer' not in r for r in rejected))
        for row in results:
            with np.load(store.artifact(row, 'shape.npz')) as f:
                self.assertNotIn('aligned', f.files)
        self.assertIn('data:image/png;base64,', gallery(report, self.case['record']['id']))
        # A valid multiview row must be found by its base input kind.
        def valid(out):
            np.savez_compressed(out / 'shape.npz', raw=self.case['points'][0], aligned=self.case['points'][0])
            return dict(model='raw', input_type='F_v9', seed=99, alignment=dict(heldout_error=.01))
        store.run('E2', self.case['record'], 'fixture_valid', valid)
        self.assertEqual(len(ex.templates(store, self.case, 'raw', 'F')), 1)
        groups = ex.select_priors(store, self.case)
        self.assertEqual(groups['raw__F']['available'], 1)
        self.assertEqual(groups['raw__F']['requested'], 3)
        store.run('E4', self.case['record'], 'fixture_missing',
            lambda out: dict(not_applicable='no compatible hypothesis'))
        unavailable = [r for r in collect(store.root, self.ds) if r['stage'] == 'E4']
        self.assertEqual(unavailable[0]['status'], 'not_applicable')

    def test_invalid_e1_does_not_reach_reconstructor(self):
        self.cfg.update(oracles=False, n_render_views=1)
        store = Store(self.root / 'run', self.cfg, self.ds)
        ex.e0(store, self.case)
        def bad(out):
            Image.new('RGB', (32, 32), 'white').save(out / 'image.png')
            return dict(model='sd15_depth', input_type='F', seed=11, selection=dict(valid_foreground=False))
        store.run('E1', self.case['record'], 'sd15_depth__F__11', bad)
        with patch.object(ex, '_reconstruct') as reconstruction:
            with self.assertRaisesRegex(RuntimeError, 'No valid E1'):
                ex.e2(store, self.case)
            reconstruction.assert_not_called()

    def test_profiles_change_declared_factors(self):
        base = self.root / 'base.json'; write(base, self.cfg)
        prepare(base, self.root / 'configs', ['legacy_cleanup', 'clean', 'completion', 'exterior'], 'images', 'mesh')
        configs = {p.stem: config.load(p) for p in (self.root / 'configs').glob('*.json')}
        self.assertEqual(configs['legacy_cleanup']['images']['bg_obj_threshold'], 140)
        self.assertEqual(configs['clean']['images']['bg_obj_threshold'], 0)
        self.assertEqual(configs['completion']['mask_type'], 'inverted')
        self.assertEqual(configs['exterior']['mask_type'], 'exterior')
        self.assertTrue(all(c['canvas_fill_target'] == 0 for c in configs.values()))


if __name__ == '__main__':
    unittest.main()
