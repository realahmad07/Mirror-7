import unittest
import torch
from training.model import require_torch
from training.native_brain import NativeBrainConfig, build_native_brain, _batch, _encode

class TestNativeBrain(unittest.TestCase):
    def test_forward_and_state_shape(self):
        model = build_native_brain(NativeBrainConfig())
        # Mock encoding and batching
        ids, mask = _batch([_encode("hello mirror", 64), _encode("reason about this", 64)], 64)
        state, predicted = model(ids, mask)
        self.assertEqual(state.shape, (2, 128))
        self.assertEqual(predicted.shape, (2, 128))
