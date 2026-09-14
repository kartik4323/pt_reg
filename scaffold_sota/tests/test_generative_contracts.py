"""CPU boundary tests; these do not certify native CUDA model execution."""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from scaffold_sota.adapters import garf, gpat, puzzlefusion_pp
from scaffold_sota.adapters.generative_common import (
    GenerativeBackend, estimate_normals, load_stage_weights, observation,
    packed_parts, pose_result, prediction_rng,
)


def observation_fixture():
    rng = torch.Generator().manual_seed(91)
    points = torch.randn((3, 24, 3), generator=rng) * 0.1
    points -= points.mean(dim=1, keepdim=True)
    points[1] = 0
    return {'points': points, 'fragment_mask': torch.tensor([True, False, True]),
            'anchor_index': 2}


class InferenceOnly(dict):
    """Poison supervision access rather than merely supplying empty GT."""
    def __getitem__(self, key):
        if key not in {'points', 'fragment_mask', 'anchor_index', 'normals'}:
            raise AssertionError(f'Inference attempted to read supervision: {key}')
        return super().__getitem__(key)


def test_garf_input_adaptation_has_no_supervision_dependency():
    raw = observation_fixture()
    batch, context = garf.make_batch(InferenceOnly(raw), 'cpu')
    assert batch['pointclouds'].shape == (1, 48, 3)
    recovered = batch['pointclouds'].reshape(2, 24, 3) * batch['scale'][0, :, None]
    torch.testing.assert_close(recovered, raw['points'][[0, 2]])
    assert batch['ref_part'].tolist() == [[False, True]]
    assert batch['points_per_part'].tolist() == [[24, 24]]
    assert not batch['graph'].any()
    assert 'quaternions' not in batch and 'translations' not in batch
    assert 'fracture_surface_gt' not in batch
    torch.testing.assert_close(batch['pointclouds_normals'].norm(dim=-1), torch.ones((1, 48)))


def test_garf_fracture_pretraining_reads_only_aligned_binary_labels():
    raw = observation_fixture()
    raw['fracture_labels'] = torch.zeros((3, 24))
    raw['fracture_labels'][2, :5] = 1
    batch, _ = garf.make_batch(raw, 'cpu', supervised=True)
    assert batch['fracture_surface_gt'].sum() == 5
    assert 'quaternions' not in batch
    raw['fracture_labels'][0, 0] = 0.3
    with pytest.raises(ValueError, match='binary'):
        garf.make_batch(raw, 'cpu', supervised=True)


def test_observed_normals_are_used_without_changing_fragment_scale():
    raw = observation_fixture()
    raw['normals'] = torch.zeros_like(raw['points'])
    raw['normals'][:, :, 2] = 2
    batch, _ = garf.make_batch(InferenceOnly(raw), 'cpu')
    assert (batch['pointclouds_normals'][..., 2] == 1).all()
    raw['normals'][2, 0] = 0
    with pytest.raises(ValueError, match='zero'):
        garf.make_batch(raw, 'cpu')


def test_normals_are_deterministic_on_planar_input():
    grid = torch.cartesian_prod(torch.linspace(-1, 1, 5), torch.linspace(-1, 1, 5))
    points = torch.cat((grid, torch.zeros((25, 1))), dim=-1).unsqueeze(0)
    normals = estimate_normals(points)
    torch.testing.assert_close(normals, torch.tensor([0., 0., 1.]).expand_as(normals))
    torch.testing.assert_close(normals, estimate_normals(points))


def test_input_rejects_invalid_anchor_and_nonfinite_active_fragment():
    raw = observation_fixture()
    raw['anchor_index'] = 1
    with pytest.raises(ValueError, match='active input anchor'):
        observation(raw, 'cpu')
    raw['anchor_index'] = 2
    raw['points'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        observation(raw, 'cpu')


class QuaternionTransforms:
    """Exact test-only unit quaternion conversion, not an assembly estimator."""
    @staticmethod
    def quaternion_to_matrix(q):
        w, x, y, z = q.unbind(-1)
        return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                            2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                            2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), dim=-1).reshape(-1, 3, 3)


def test_prediction_uses_predicted_anchor_and_preserves_slot_mapping():
    context = observation(observation_fixture(), 'cpu')
    angle = math.pi / 2
    q = torch.tensor([[1., 0., 0., 0.], [math.cos(angle/2), 0., 0., math.sin(angle/2)]])
    t = torch.tensor([[2., 3., 1.], [1., 1., 0.]])
    result = pose_result(context, q, t, QuaternionTransforms)
    assert result['status'] == 'ok'
    torch.testing.assert_close(result['rotations'][2], torch.eye(3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result['translations'][2], torch.zeros(3))
    torch.testing.assert_close(result['translations'][0], torch.tensor([2., -1., 1.]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result['rotations'][1], torch.eye(3))  # inactive padding
    broken = pose_result(context, torch.zeros_like(q), t, QuaternionTransforms)
    assert broken['status'] == 'failed' and 'rotations' not in broken


class IdentityTokenLayer(nn.Module):
    """Hook-routing fixture, deliberately not used by any backend build()."""
    def forward(self, hidden_states):
        return hidden_states

    def forward_sdpa(self, hidden_states):
        return hidden_states


class HookFixture(GenerativeBackend):
    def __init__(self, condition):
        super().__init__()
        self.native = nn.ModuleList([IdentityTokenLayer()])
        self.configure_conditioning({'condition': condition, 'token_dim': 8,
            'heads': 2, 'num_tokens': 4}, 8, layers=self.native)


def test_native_checkpoint_branches_and_garf_direct_sdpa_is_conditioned():
    native, branch = HookFixture('native'), HookFixture('B1')
    branch.initialize_from_native(native.state_dict())
    x = torch.randn(1, 7, 8)
    with branch.prior_context(None):
        torch.testing.assert_close(branch.native[0](hidden_states=x), x)
        torch.testing.assert_close(branch.native[0].forward_sdpa(hidden_states=x), x)
        branch.conditioner.gate.data.fill_(1)
        normal = branch.native[0](hidden_states=x)
        sdpa = branch.native[0].forward_sdpa(hidden_states=x)
    assert not torch.allclose(sdpa, x)
    torch.testing.assert_close(normal, sdpa)
    assert all(key.startswith('conditioner.') for key in branch.state_dict())


def test_prior_context_cleared_after_failure_and_missing_prior_rejected():
    backend = HookFixture('B4')
    with pytest.raises(ValueError, match='requires'):
        with backend.prior_context(None):
            pass
    with pytest.raises(RuntimeError, match='intentional'):
        with backend.prior_context({'distance': torch.zeros(4)}):
            raise RuntimeError('intentional')
    assert backend._active_prior is None


def test_feature_checkpoint_requires_correct_stage_and_prefix(tmp_path):
    path = tmp_path / 'feature.pt'
    torch.save({'model_name': 'garf', 'stage': 'pretrain',
                'model': {'feature_extractor.weight': torch.ones(2)}}, path)
    result = load_stage_weights(path, 'garf', 'pretrain', 'feature_extractor.')
    assert set(result) == {'weight'}
    with pytest.raises(ValueError, match='Expected'):
        load_stage_weights(path, 'puzzlefusion_pp', 'pretrain', 'autoencoder.')
    with pytest.raises(ValueError, match='prefix'):
        load_stage_weights(path, 'garf', 'pretrain', 'wrong.')


def test_assembly_without_required_feature_weights_fails_before_native_import():
    for backend in (garf, puzzlefusion_pp):
        with pytest.raises(ValueError, match='feature_checkpoint'):
            backend.build('/nonexistent/upstream', {'stage': 'assembly'}, 'cpu')
        with pytest.raises(ValueError, match='prior-independent'):
            backend.build('/nonexistent/upstream', {'stage': 'pretrain', 'condition': 'B4'}, 'cpu')
    with pytest.raises(ValueError, match='surface'):
        gpat.build('/nonexistent/upstream', {}, 'cpu')


def test_gpat_requires_surface_export_and_semantic_track():
    raw = observation_fixture()
    prior = {'surface_points': torch.randn(50, 3)}
    with pytest.raises(ValueError, match='dataset_track'):
        gpat.surface_input(raw, prior, 'cpu', target_points=50, part_points=24)
    raw['dataset_track'] = 'partnet'
    with pytest.raises(ValueError, match='surface_points'):
        gpat.surface_input(raw, {'query_xyz': torch.randn(50, 3)}, 'cpu', target_points=50, part_points=24)
    with pytest.raises(ValueError, match='contiguous'):
        gpat.surface_input(raw, prior, 'cpu', target_points=50, part_points=24)
    raw['fragment_mask'] = torch.tensor([True, True, False])
    raw['points'][1] = raw['points'][2]
    raw['anchor_index'] = 0
    target, _ = gpat.surface_input(raw, prior, 'cpu', target_points=50, part_points=24)
    torch.testing.assert_close(target, prior['surface_points'])
    changed = target.clone()
    changed[0, 0] += .01
    assert gpat.surface_digest(target) != gpat.surface_digest(changed)


def test_predict_rng_replays_without_advancing_training_rng():
    torch.manual_seed(7)
    before = torch.get_rng_state().clone()
    with prediction_rng(18, 'cpu'):
        first = torch.randn(6)
    torch.testing.assert_close(before, torch.get_rng_state())
    with prediction_rng(18, 'cpu'):
        second = torch.randn(6)
    torch.testing.assert_close(first, second)
