"""Finite optimizer updates with bounded, reproducible AMP overflow recovery."""
from __future__ import annotations

import math
import random

import numpy as np
import torch

from .checkpoints import restore_random_state


NUMERICS_VERSION = "fp32-contacts-amp-retry-v1"


class NumericalUpdateError(RuntimeError):
    def __init__(self, reason, context, **details):
        self.diagnostics = dict(context, reason=reason, **details)
        super().__init__(f"{reason} at stage {context.get('stage')}, update {context.get('update')}")


def optimizer_update(model, optimizer, scaler, backward, *, gradient_clip,
                     context, on_overflow=None, max_amp_retries=8):
    """Commit exactly one finite update; replay the effective batch on overflow.

    ``backward`` recomputes every accumulation microbatch and returns scalar
    losses/sample IDs for diagnostics. RNG replay preserves batch selection and
    random views without caching their tensors or increasing GPU resolution.
    This model has no mutable training buffers (e.g. BatchNorm running means).
    """
    rng = {"random_state": {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }}
    for attempt in range(max_amp_retries + 1):
        if attempt:
            restore_random_state(rng)
        optimizer.zero_grad(set_to_none=True)
        metrics = backward()
        named = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
        if not named:
            raise NumericalUpdateError("missing_gradients", context, metrics=metrics)
        scaler.unscale_(optimizer)
        bad = [name for name, p in named if not bool(torch.isfinite(p.grad).all())]
        if bad:
            before = float(scaler.get_scale())
            if not scaler.is_enabled():
                raise NumericalUpdateError("nonfinite_gradients_without_amp", context,
                                           parameters=bad, metrics=metrics, amp_scale=before)
            # unscale_ recorded these nonfinite gradients. GradScaler.step skips
            # the optimizer, then update applies its normal backoff and resets
            # scaler bookkeeping. Never clip or step corrupt gradients directly.
            scaler.step(optimizer)
            scaler.update()
            after = float(scaler.get_scale())
            event = dict(context, event="amp_overflow", attempt=attempt + 1,
                         scale_before=before, scale_after=after,
                         retry_scheduled=attempt < max_amp_retries and math.isfinite(after) and 0 < after < before,
                         parameters=bad, metrics=metrics)
            if on_overflow is not None:
                on_overflow(event)
            if not math.isfinite(after) or not 0 < after < before:
                raise NumericalUpdateError("amp_scale_did_not_decrease", context,
                                           parameters=bad, metrics=metrics, amp_scale=after)
            if attempt == max_amp_retries:
                raise NumericalUpdateError("persistent_nonfinite_gradients", context,
                                           parameters=bad, metrics=metrics, amp_scale=after,
                                           attempts=attempt + 1)
            continue
        try:
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], gradient_clip,
                                                 error_if_nonfinite=True)
        except RuntimeError as exc:
            raise NumericalUpdateError("nonfinite_gradient_norm", context, metrics=metrics) from exc
        scaler.step(optimizer)
        scaler.update()
        return {"metrics": metrics, "gradient_norm": float(norm),
                "amp_scale": float(scaler.get_scale()), "amp_retries": attempt}
    raise AssertionError("Unreachable AMP retry state")
