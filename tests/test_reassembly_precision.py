"""Real fp16/GradScaler overflow recovery on CPU, without requiring a CUDA host."""
import copy
import random
import unittest

import numpy as np
import torch

from reassembly.losses import directional_contact_loss
from reassembly.model import PartialContactMatcher
from reassembly.precision import NumericalUpdateError, optimizer_update
from reassembly.resources import seed_all
from reassembly.training import _scaler


class PrecisionTests(unittest.TestCase):
    def test_real_fp16_overflow_replays_accumulation_and_commits_once(self):
        def run(enabled):
            model = torch.nn.Linear(1, 1, bias=False)
            model.weight.data.fill_(1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            scaler = torch.amp.GradScaler('cpu', enabled=enabled)
            draws, events = [], []
            seed_all(71)
            def backward():
                for index in range(2):
                    draw = (random.random(), float(np.random.rand()), float(torch.rand(())))
                    draws.append(draw)
                    x = torch.tensor([[index + 1 + draw[2] * .05]])
                    with torch.autocast('cpu', dtype=torch.float16):
                        loss = model(x).float().sum() / 2
                    scaler.scale(loss).backward()
                return {'loss': float(loss.detach())}
            def overflow(event):
                # An overflow must neither mutate parameters nor allocate/update Adam state.
                self.assertEqual(float(model.weight.detach()), 1)
                self.assertFalse(optimizer.state)
                events.append(event)
            result = optimizer_update(model, optimizer, scaler, backward, gradient_clip=10,
                                      context={'stage': 1, 'update': 16}, on_overflow=overflow)
            return model, optimizer, result, draws, events
        model, optimizer, result, draws, events = run(True)
        baseline, base_optimizer, base_result, base_draws, _ = run(False)
        self.assertEqual(result['amp_retries'], 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['scale_before'], 65536)
        self.assertEqual(events[0]['scale_after'], 32768)
        self.assertEqual(draws[:2], draws[2:])
        self.assertEqual(draws[-2:], base_draws)
        torch.testing.assert_close(model.weight, baseline.weight, rtol=0, atol=0)
        state = optimizer.state[model.weight]
        self.assertEqual(int(state['step']), 1)
        torch.testing.assert_close(state['exp_avg'], base_optimizer.state[baseline.weight]['exp_avg'])
        self.assertEqual(base_result['amp_retries'], 0)

    def test_persistent_nonfinite_gradients_fail_with_names_and_no_update(self):
        model = torch.nn.Linear(1, 1, bias=False)
        original = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler('cpu', init_scale=16)
        hook = model.weight.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
        events = []
        def backward():
            loss = model(torch.ones(1, 1)).sum()
            scaler.scale(loss).backward()
            return {'loss': float(loss.detach())}
        with self.assertRaises(NumericalUpdateError) as raised:
            optimizer_update(model, optimizer, scaler, backward, gradient_clip=1,
                             context={'stage': 1, 'update': 16}, max_amp_retries=2,
                             on_overflow=events.append)
        hook.remove()
        self.assertEqual(raised.exception.diagnostics['reason'], 'persistent_nonfinite_gradients')
        self.assertEqual(raised.exception.diagnostics['parameters'], ['weight'])
        self.assertEqual(raised.exception.diagnostics['attempts'], 3)
        self.assertEqual(len(events), 3)
        self.assertFalse(optimizer.state)
        torch.testing.assert_close(model.weight, original['weight'])

    def test_fp32_nonfinite_gradients_and_missing_gradients_are_fatal(self):
        for missing in (False, True):
            model = torch.nn.Linear(1, 1, bias=False)
            optimizer = torch.optim.AdamW(model.parameters())
            scaler = _scaler(False)
            def backward():
                if not missing:
                    (model(torch.ones(1, 1)).sum() * float('nan')).backward()
                return {'loss': 0.0}
            with self.assertRaises(NumericalUpdateError) as raised:
                optimizer_update(model, optimizer, scaler, backward, gradient_clip=1,
                                 context={'stage': 1, 'update': 1})
            self.assertEqual(raised.exception.diagnostics['reason'],
                             'missing_gradients' if missing else 'nonfinite_gradients_without_amp')
            self.assertFalse(optimizer.state)

    def test_nonfinite_norm_of_finite_gradients_does_not_step(self):
        model = torch.nn.Linear(4, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = _scaler(False)
        def backward():
            model(torch.full((1, 4), 1e30)).sum().backward()
            return {'loss': 0.0}
        with self.assertRaisesRegex(NumericalUpdateError, 'nonfinite_gradient_norm'):
            optimizer_update(model, optimizer, scaler, backward, gradient_clip=1,
                             context={'stage': 1, 'update': 1})
        self.assertFalse(optimizer.state)

    def test_weak_fracture_logits_have_finite_scaled_matching_gradients(self):
        torch.manual_seed(4)
        matcher = PartialContactMatcher({'dim': 8})
        logits = torch.full((1, 2, 4), -9., requires_grad=True)
        encoded = {'token_indices': torch.arange(4)[None, None].expand(1, 2, 4),
                   'point_xyz': torch.randn(1, 2, 4, 3),
                   'point_features': torch.randn(1, 2, 4, 8),
                   'descriptor': torch.randn(1, 2, 4, 8),
                   'token_features': torch.randn(1, 2, 4, 8),
                   'token_xyz': torch.randn(1, 2, 4, 3),
                   'fracture_logits': logits.to(torch.float16),
                   'fragment_mask': torch.ones(1, 2, dtype=torch.bool)}
        with torch.autocast('cpu', dtype=torch.float16):
            pair = matcher(encoded, None)[0]
            positives = torch.eye(4, dtype=torch.bool)[None]
            loss = directional_contact_loss(pair['source_prob'], positives, pair['valid'])
        self.assertEqual(pair['source_prob'].dtype, torch.float32)
        self.assertEqual(pair['weights'].dtype, torch.float32)
        (loss * 1024).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in matcher.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
