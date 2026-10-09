"""Geometric invariants, inference isolation, and diagnostic override contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reassembly.geometry import so3_exp
from reassembly.repair import orientation
from reassembly.repair.assembly import build_candidates_from_matches, solve_candidates, build_candidates


def fixture():
    grid = np.stack(np.meshgrid(np.linspace(-.15,.15,7),np.linspace(-.15,.15,7)), -1).reshape(-1,2)
    face = np.column_stack((grid, np.zeros(len(grid))))
    points = [np.concatenate((face, face+[0,0,-.4])), np.concatenate((face,face+[0,0,.4]))]
    pair = dict(i=0,j=1,source_xyz=face,target_xyz=face,weights=np.eye(len(face))*.5,
                source_matchability=np.ones(len(face)),target_matchability=np.ones(len(face)))
    cfg = dict(solver=dict(contact_orientation=dict(enabled=True)))
    return points, pair, cfg


class OrientationTests(unittest.TestCase):
    def test_normals_opposing_and_rigid_permutation_equivariant(self):
        points, pair, cfg = fixture(); config=orientation.options(cfg)
        a, valid = orientation.estimate_normals(points[0],pair['source_xyz'],config)
        b, other = orientation.estimate_normals(points[1],pair['target_xyz'],config)
        self.assertTrue(valid.all()); self.assertTrue(other.all())
        np.testing.assert_allclose(a,-b,atol=1e-10)
        rotation=so3_exp(np.array([.7,-.4,.3])); translation=np.array([4.,-2.,7.])
        permutation=np.random.default_rng(8).permutation(len(points[0]))
        rotated, reliable=orientation.estimate_normals(points[0][permutation]@rotation.T+translation,
            pair['source_xyz']@rotation.T+translation,config)
        np.testing.assert_allclose(rotated,a@rotation.T,atol=1e-10)
        np.testing.assert_array_equal(reliable,valid)

    def test_curved_surface_and_degenerate_estimates(self):
        rng=np.random.default_rng(42)
        sphere=rng.normal(size=(1200,3));sphere/=np.linalg.norm(sphere,axis=1,keepdims=True)
        normals, reliable=orientation.estimate_normals(sphere,sphere[:40],orientation.options())
        self.assertGreater(reliable.mean(),.9)
        self.assertGreater(np.mean((normals*sphere[:40]).sum(-1)),.95)
        line=np.column_stack((np.arange(40),np.zeros((40,2))))
        for points in (line,np.zeros((40,3))):
            _,reliable=orientation.estimate_normals(points,points[:4],orientation.options())
            self.assertFalse(reliable.any())

    def test_orientation_score_rejects_same_facing_and_requires_support(self):
        points,pair,cfg=fixture()
        cache=build_candidates_from_matches(orientation.attach_normals([pair],points,cfg),2,cfg=cfg)
        poses=(np.stack([np.eye(3)]*2),np.zeros((2,3)))
        value,detail=orientation.score(poses,cache.correspondences,orientation.options(cfg))
        self.assertEqual(value,0);self.assertEqual(detail['orientation_reliable_fraction'],1)
        bad=(np.stack([np.eye(3),np.diag([1.,-1.,-1.])]),np.zeros((2,3)))
        value,detail=orientation.score(bad,cache.correspondences,orientation.options(cfg))
        self.assertAlmostEqual(value,.0004);self.assertAlmostEqual(detail['orientation_score'],1)
        weak=copy.deepcopy(cache.correspondences)
        weak[0]['weights']*=1e-5
        self.assertEqual(orientation.score(bad,weak,orientation.options(cfg))[0],0)

    def test_offsets_preserve_mass_and_candidate_caps(self):
        points,pair,cfg=fixture()
        augmented=orientation.attach_normals([pair],points,cfg)
        cache=build_candidates_from_matches(augmented,2,cfg=cfg)
        from reassembly.geometry import weighted_kabsch
        seen=[]
        def fit(a,b,w,**kwargs):
            if len(w)%3==0 and len(w)>len(pair['source_xyz']): seen.append(float(w.sum()))
            return weighted_kabsch(a,b,w,**kwargs)
        with patch.object(orientation,'weighted_kabsch',side_effect=fit):
            orientation.pair_candidates(cache.correspondences[0],{},orientation.options(cfg))
        self.assertTrue(seen)
        self.assertAlmostEqual(seen[0],float(pair['weights'].sum()))
        self.assertLessEqual(len(cache.pair_candidates[(0,1)]),4)
        self.assertTrue(any(f.get('proposal_type') for f in cache.pair_candidates[(0,1)]))

    def test_disabled_is_exact_and_unreliable_falls_back(self):
        points,pair,cfg=fixture()
        baseline=build_candidates_from_matches([pair],2,cfg={})
        disabled={'solver':{'contact_orientation':{'enabled':False}}}
        other=build_candidates_from_matches([pair],2,cfg=disabled)
        self.assertEqual(baseline.fingerprint,other.fingerprint)
        for a,b in zip(baseline.hypotheses,other.hypotheses):
            for x,y in zip(a['poses'],b['poses']):np.testing.assert_array_equal(x,y)
        unreliable=orientation.attach_normals([pair],points,cfg)[0]
        unreliable['source_normal_reliability'][:]=False
        cache=build_candidates_from_matches([unreliable],2,cfg=cfg)
        for a,b in zip(baseline.hypotheses,cache.hypotheses):
            for x,y in zip(a['poses'],b['poses']):np.testing.assert_array_equal(x,y)

    def test_fingerprint_covers_normals_and_options(self):
        points,pair,cfg=fixture()
        pairs=orientation.attach_normals([pair],points,cfg)
        first=build_candidates_from_matches(pairs,2,cfg=cfg)
        changed=copy.deepcopy(pairs);changed[0]['source_normals'] *= -1
        second=build_candidates_from_matches(changed,2,cfg=cfg)
        self.assertNotEqual(first.fingerprint,second.fingerprint)
        incompatible=copy.deepcopy(cfg);incompatible['solver']['contact_orientation']['offset']=.03
        with self.assertRaises(ValueError):solve_candidates(first,incompatible)

    def test_refinement_reverts_on_worse_orientation_without_changing_confidence(self):
        points,pair,cfg=fixture()
        cache=build_candidates_from_matches(orientation.attach_normals([pair],points,cfg),2,cfg=cfg)
        rotations=np.stack([np.eye(3),np.diag([1.,-1.,-1.])])
        with patch('reassembly.solver._refine',return_value=((rotations,np.zeros((2,3))),5)):
            result=solve_candidates(cache,cfg)
        self.assertTrue(all(x['reverted'] for x in result['diagnostics']['refinement_decisions']))
        self.assertEqual(result['diagnostics']['refinement_accepted_steps'],0)
        self.assertEqual(result['status'],'ok')
        expected=result['diagnostics']['contact_confidence']*np.exp(-result['diagnostics']['contact_rms']/.02)
        self.assertAlmostEqual(result['confidence'],expected)

    def test_three_piece_composition(self):
        points,pair,cfg=fixture()
        third={**pair,'i':1,'j':2}
        matches=orientation.attach_normals([pair,third],points+[points[0]],cfg)
        cache=build_candidates_from_matches(matches,3,anchor_index=1,cfg=cfg)
        result=solve_candidates(cache,cfg)
        self.assertIsNotNone(result['rotations'])
        np.testing.assert_allclose(result['rotations'][1],np.eye(3))
        np.testing.assert_allclose(np.linalg.det(result['rotations']),1)
        self.assertEqual(cache.num_fragments,3)

    def test_nonfinite_refinement_falls_back_to_serializable_initial_result(self):
        points,pair,cfg=fixture()
        cache=build_candidates_from_matches(orientation.attach_normals([pair],points,cfg),2,cfg=cfg)
        with patch('reassembly.solver._refine',return_value=((np.full((2,3,3),np.nan),np.zeros((2,3))),1)):
            result=solve_candidates(cache,cfg)
        self.assertEqual(result['status'],'ok')
        self.assertTrue(all(r['refined_score'] is None for r in result['diagnostics']['refinement_decisions']))
        json.dumps(result['diagnostics'],allow_nan=False)

    def test_inference_uses_only_active_local_xyz(self):
        points,pair,cfg=fixture()
        padded=np.stack([points[0],points[1],np.full_like(points[0],999.)])
        class Model:
            training=True
            def modules(self):return [self]
            def eval(self):self.training=False
            def encode(self,*args):
                return {'fracture_logits':torch.zeros(1,3,len(points[0]))}
            def match(self,*args,**kwargs):
                return [{**{k:torch.tensor(v)[None] for k,v in pair.items() if k not in ('i','j')},'i':0,'j':1,'valid':torch.tensor([True])}]
        batch={'points':torch.tensor(padded)[None], 'fragment_mask':torch.tensor([[True,True,False]]),
               'anchor_index':torch.tensor([0]), 'canonical_points':object(), 'interface_ids':object()}
        model=Model();cache=build_candidates(model,batch,cfg)
        self.assertTrue(model.training)
        self.assertEqual(cache.num_fragments,2)
        self.assertTrue(cache.correspondences[0]['source_normal_reliability'].all())

    def test_override_whitelist(self):
        for bad in ({'success_threshold':.03}, {'contact_orientation':{'enabled':True,'unknown':1}},
                    {'contact_orientation':{'weight':float('nan')}}, {'contact_orientation':{'neighbors':True}}):
            with self.assertRaises(ValueError):orientation.validate_overrides(bad)


if __name__=='__main__':unittest.main()
