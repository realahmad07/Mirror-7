import unittest

from training.model import require_torch
from training.native_brain import (
    NativeBrainConfig,
    _batch,
    _encode,
    build_native_brain,
)


class TestNativeBrain(unittest.TestCase):
    def test_forward_and_state_shape(self):
        torch, _ = require_torch()
        model = build_native_brain(NativeBrainConfig())
        ids, mask = _batch(
            [_encode("hello mirror", 64), _encode("reason about this", 64)],
            64,
        )
        state, predicted = model(ids, mask)
        self.assertEqual(state.shape, (2, 128))
        self.assertEqual(predicted.shape, (2, 128))
        self.assertTrue(torch.isfinite(state).all())
        self.assertTrue(torch.isfinite(predicted).all())

    def test_native_brain_is_small(self):
        model = build_native_brain(NativeBrainConfig())
        self.assertLess(sum(p.numel() for p in model.parameters()), 1_500_000)
        self.assertEqual(model.embedding.num_embeddings, 257)
